"""Gym-style residual-control environment over a frozen Cartesian policy.

The environment owns the residual state machine and reward.  Simulator-specific
operations are supplied through ``ResidualEnvBackend`` so the same logic can be
unit-tested without Isaac Sim and connected to the task-board runner without
putting simulator objects in the RL algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import numpy as np

from policies.residual_injector import ResidualInjector
from residual_task import (
    BoundedSnapReward,
    BoundedSnapRewardConfig,
    ResidualTask,
    RewardContext,
    SnapInsertionTask,
    SnapInsertionTaskConfig,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Isaac's base environment does not install Gymnasium.
    gym = None

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.broadcast_to(low, shape).astype(dtype, copy=True)
            self.high = np.broadcast_to(high, shape).astype(dtype, copy=True)
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value):
            value = np.asarray(value)
            return (
                value.shape == self.shape
                and np.isfinite(value).all()
                and np.all(value >= self.low)
                and np.all(value <= self.high)
            )

    class _Spaces:
        Box = _Box

    spaces = _Spaces()


_GymBase = gym.Env if gym is not None else object


@dataclass(frozen=True)
class ResidualEnvConfig:
    """Environment runtime defaults.

    Gate/reward fields remain for CLI and checkpoint compatibility.  New code
    can instead pass a fully configured ``ResidualTask`` to ``ResidualEnv``.
    """

    part_name: str = "hdmi"
    physical_obs_dim: int = 44
    w_pos: float = 1.0
    w_ori: float = 1.0
    smooth_weight: float = 1.0
    terminal_reward: float = 100.0
    gate_exit_reward: float = -100.0
    stuck_reward: float = -100.0
    gate_enter_pos_m: float = 0.060
    gate_exit_pos_m: float = 0.070
    gate_enter_ori_rad: float = math.radians(45.0)
    gate_exit_ori_rad: float = math.radians(60.0)
    stuck_window_steps: int = 10
    stuck_position_change_m: float = 0.0005
    max_steps: int = 64
    max_prefix_steps: int = 64
    max_reset_attempts: int = 50

    def __post_init__(self):
        if not self.part_name:
            raise ValueError("part_name must not be empty")
        if self.physical_obs_dim <= 0:
            raise ValueError("physical_obs_dim must be positive")
        if self.max_steps <= 0 or self.max_prefix_steps <= 0:
            raise ValueError("episode step limits must be positive")
        if self.stuck_window_steps <= 0 or self.stuck_position_change_m <= 0.0:
            raise ValueError("stuck detector parameters must be positive")
        if self.max_reset_attempts <= 0:
            raise ValueError("max_reset_attempts must be positive")
        if self.smooth_weight < 0.0:
            raise ValueError("smooth_weight must be non-negative")
        if not 0.0 < self.gate_enter_pos_m < self.gate_exit_pos_m:
            raise ValueError("position gate must have positive hysteresis")
        if not 0.0 < self.gate_enter_ori_rad < self.gate_exit_ori_rad:
            raise ValueError("orientation gate must have positive hysteresis")


class ResidualEnvBackend(Protocol):
    """Simulator bridge required by :class:`ResidualEnv`."""

    exec_horizon: int
    """Frozen-policy execution horizon. ResidualEnv requires this to be 1."""

    def reset_episode(self, part_name: str) -> None:
        """Restore the scene and reset the frozen policy for one attempt."""

    def get_physical_obs(self) -> np.ndarray:
        """Return the current fixed-size physical observation vector."""

    def get_se3_error(self, part_name: str) -> np.ndarray:
        """Return target-minus-current world-frame position/rotvec error."""

    def get_normalized_error(self, part_name: str) -> np.ndarray:
        """Return the 4-D error normalized by the part's snap thresholds."""

    def get_part_pose(self, part_name: str) -> np.ndarray:
        """Return movable mesh world pose as xyz plus wxyz quaternion."""

    def predict_bc_action(self) -> np.ndarray:
        """Replan once from the current observation and return a physical 7-D action."""

    def predict_prefix_bc_action(self) -> np.ndarray:
        """Return one BC action for reset-only pick/transport replay."""

    def begin_residual_control(self) -> None:
        """Discard prefix state before the first horizon-1 RL transition."""

    def apply_cartesian_action(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, gripper: float
    ) -> None:
        """Apply one absolute Cartesian target through IK."""

    def advance(self) -> None:
        """Advance the simulator by one RL control interval."""

    def snap_succeeded(self, part_name: str) -> bool:
        """Whether the part's snap gate has fired."""

    def gripper_is_open(self) -> bool:
        """Whether the gripper has opened after residual control began."""

    def gripper_is_closed(self) -> bool:
        """Whether the measured gripper joint is currently closed."""

    def closed_gripper_command(self) -> float:
        """Return the configured per-part closed command."""

    def part_is_grasped(self, part_name: str) -> bool:
        """Whether the part is currently following the gripper."""

    def get_part_contact_force(self, part_name: str) -> np.ndarray | None:
        """Return current net contact force, or None if unavailable."""

    def episode_ended(self) -> bool:
        """Whether the simulator/backend ended the attempt externally."""

    def close(self) -> None:
        """Release simulator and frozen-policy resources."""


class ResidualEnv(_GymBase):
    """Policy- and task-agnostic Gymnasium residual environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: ResidualEnvBackend,
        config: ResidualEnvConfig | None = None,
        injector: ResidualInjector | None = None,
        task: ResidualTask | None = None,
    ) -> None:
        required_methods = (
            "reset_episode",
            "get_physical_obs",
            "get_se3_error",
            "get_normalized_error",
            "get_part_pose",
            "predict_bc_action",
            "predict_prefix_bc_action",
            "begin_residual_control",
            "apply_cartesian_action",
            "advance",
            "snap_succeeded",
            "gripper_is_open",
            "gripper_is_closed",
            "closed_gripper_command",
            "part_is_grasped",
            "get_part_contact_force",
            "episode_ended",
            "close",
        )
        missing = [
            name
            for name in required_methods
            if not callable(getattr(backend, name, None))
        ]
        if missing or not hasattr(backend, "exec_horizon"):
            raise TypeError(
                "backend does not implement ResidualEnvBackend; "
                f"missing={missing} exec_horizon={hasattr(backend, 'exec_horizon')}"
            )
        if int(backend.exec_horizon) != 1:
            raise ValueError(
                "ResidualEnv requires a replanning execution horizon of 1, got "
                f"{backend.exec_horizon!r}"
            )
        self.backend = backend
        self.config = config or ResidualEnvConfig()
        self.task = task or SnapInsertionTask(
            SnapInsertionTaskConfig(
                target_name=self.config.part_name,
                gate_enter_pos_m=self.config.gate_enter_pos_m,
                gate_exit_pos_m=self.config.gate_exit_pos_m,
                gate_enter_ori_rad=self.config.gate_enter_ori_rad,
                gate_exit_ori_rad=self.config.gate_exit_ori_rad,
            ),
            reward=BoundedSnapReward(
                BoundedSnapRewardConfig(
                    w_pos=self.config.w_pos,
                    w_ori=self.config.w_ori,
                    smooth_weight=self.config.smooth_weight,
                    terminal_reward=self.config.terminal_reward,
                    gate_exit_reward=self.config.gate_exit_reward,
                    stuck_reward=self.config.stuck_reward,
                )
            ),
        )
        if not getattr(self.task, "target_name", None):
            raise TypeError("ResidualTask must expose a non-empty target_name")
        self.injector = injector or ResidualInjector()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        observation_dim = self.config.physical_obs_dim + 6 + 6 + 7 + 6 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.gate_active = False
        self.residual_executed_prev = np.zeros(6, dtype=np.float64)
        self._bc_action_current = np.zeros(7, dtype=np.float64)
        self._rl_steps = 0
        self._needs_reset = True
        self._reset_attempts = 0
        self._rejected_initializations = 0
        self._rejection_reasons: dict[str, int] = {}
        self._pose_gate_blocked_ungrasped = 0
        self._stagnant_steps = 0
        self._previous_position_error_norm = None
        self._previous_se3_error = None
        self._se3_error_delta = np.zeros(6, dtype=np.float64)
        self._gripper_hold_latched = False
        self._gripper_hold_prefix_step = None
        self._last_prefix_hold_activated = False
        self._last_prefix_hold_step = None
        self._prefix_hold_activations = 0

    @property
    def rejected_initializations(self) -> int:
        return self._rejected_initializations

    @property
    def reset_attempts(self) -> int:
        return self._reset_attempts

    @property
    def initialization_rejection_rate(self) -> float:
        if self._reset_attempts == 0:
            return 0.0
        return self._rejected_initializations / self._reset_attempts

    @property
    def rejection_reasons(self) -> dict[str, int]:
        return dict(self._rejection_reasons)

    @property
    def pose_gate_blocked_ungrasped(self) -> int:
        return self._pose_gate_blocked_ungrasped

    @property
    def last_prefix_hold_activated(self) -> bool:
        return self._last_prefix_hold_activated

    @property
    def last_prefix_hold_step(self) -> int | None:
        return self._last_prefix_hold_step

    @property
    def prefix_hold_activations(self) -> int:
        return self._prefix_hold_activations

    @staticmethod
    def _finite_vector(name: str, value, size: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be one finite {size}-D vector")
        return vector

    def _clear_residual_state(self) -> None:
        self.gate_active = False
        self.residual_executed_prev.fill(0.0)
        self._stagnant_steps = 0
        self._previous_position_error_norm = None
        self._previous_se3_error = None
        self._se3_error_delta.fill(0.0)
        self._gripper_hold_latched = False
        self._gripper_hold_prefix_step = None

    def _errors(self) -> tuple[np.ndarray, np.ndarray]:
        raw_se3_error, raw_normalized = self.task.get_errors(self.backend)
        se3_error = self._finite_vector(
            "se3_error", raw_se3_error, 6
        )
        normalized = np.asarray(raw_normalized, dtype=np.float64).reshape(-1)
        if normalized.size == 0 or not np.isfinite(normalized).all():
            raise ValueError("normalized_error must be one non-empty finite vector")
        return se3_error, normalized

    def _physical_obs(self) -> np.ndarray:
        return self._finite_vector(
            "physical_obs",
            self.backend.get_physical_obs(),
            self.config.physical_obs_dim,
        )

    def _predict_bc_action(self) -> np.ndarray:
        action = self._finite_vector("bc_action", self.backend.predict_bc_action(), 7)
        self._bc_action_current = action
        return action

    def _observation(self, se3_error: np.ndarray) -> np.ndarray:
        observation = np.concatenate(
            [
                self._physical_obs(),
                se3_error,
                self._se3_error_delta,
                self._bc_action_current,
                self.residual_executed_prev,
                [float(self.gate_active)],
            ]
        ).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"assembled observation has shape {observation.shape}, "
                f"expected {self.observation_space.shape}"
            )
        return observation

    def _apply(
        self, residual_action: np.ndarray, *, force_gripper_closed: bool = False
    ) -> tuple[np.ndarray, float, float]:
        inject = (
            self.injector.inject_canonical
            if getattr(self.backend, "base_action_rotation_repr", "euler_xyz")
            == "rotvec"
            else self.injector.inject
        )
        position, quaternion, gripper, executed = inject(
            self._bc_action_current, residual_action, self.residual_executed_prev
        )
        bc_gripper = float(gripper)
        commanded_gripper = (
            float(self.backend.closed_gripper_command())
            if force_gripper_closed
            else bc_gripper
        )
        self.backend.apply_cartesian_action(
            position, quaternion, commanded_gripper
        )
        self.residual_executed_prev = executed
        self.backend.advance()
        return executed, bc_gripper, commanded_gripper

    def _gate_entry_state(
        self,
        se3_error: np.ndarray,
        *,
        gripper_closed: bool,
        part_grasped: bool,
    ) -> str:
        state = self.task.gate_entry_state(
            se3_error,
            gripper_closed=gripper_closed,
            part_grasped=part_grasped,
        )
        if state not in {"outside", "ungrasped", "reject", "enter"}:
            raise ValueError(f"ResidualTask returned invalid gate state {state!r}")
        if state == "ungrasped":
            self._pose_gate_blocked_ungrasped += 1
        return state

    def _gate_exited(self, se3_error: np.ndarray) -> bool:
        return bool(self.task.gate_exited(se3_error))

    def reset(self, *, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)
        del options

        reset_started = time.perf_counter()
        for _ in range(self.config.max_reset_attempts):
            self._reset_attempts += 1
            self.backend.reset_episode(self.task.target_name)
            self._clear_residual_state()
            self._last_prefix_hold_activated = False
            self._last_prefix_hold_step = None
            self._rl_steps = 0
            rejected_reason = "prefix_timeout"

            for prefix_step in range(self.config.max_prefix_steps):
                se3_error, normalized = self._errors()
                if self._previous_se3_error is None:
                    se3_error_delta = np.zeros(6, dtype=np.float64)
                else:
                    se3_error_delta = se3_error - self._previous_se3_error
                gripper_closed = (
                    bool(self.backend.gripper_is_closed())
                    if self.task.requires_closed_gripper
                    else True
                )
                part_grasped = (
                    bool(self.backend.part_is_grasped(self.task.target_name))
                    if self.task.requires_grasp
                    else True
                )
                if (
                    self.task.hold_gripper_closed
                    and part_grasped
                    and not self._gripper_hold_latched
                ):
                    self._gripper_hold_latched = True
                    self._gripper_hold_prefix_step = prefix_step
                    self._prefix_hold_activations += 1
                entry_state = self._gate_entry_state(
                    se3_error,
                    gripper_closed=gripper_closed,
                    part_grasped=part_grasped,
                )
                if entry_state == "reject":
                    rejected_reason = "unrecoverable_orientation"
                    break
                if entry_state == "enter":
                    self.gate_active = True
                    self._se3_error_delta = se3_error_delta
                    self._previous_se3_error = se3_error.copy()
                    # Prefix action chunks never enter the RL replay buffer.
                    # Clear them at this boundary so every observed transition
                    # below is governed by a newly replanned horizon-1 action.
                    self.backend.begin_residual_control()
                    self._predict_bc_action()
                    self._previous_position_error_norm = float(
                        np.linalg.norm(se3_error[:3])
                    )
                    self._needs_reset = False
                    info = {
                        "gate_active": True,
                        "prefix_steps": prefix_step,
                        "reset_attempt": self._reset_attempts,
                        "rejected_initializations": self._rejected_initializations,
                        "initialization_rejection_rate": self.initialization_rejection_rate,
                        "se3_error": se3_error.copy(),
                        "se3_error_delta": self._se3_error_delta.copy(),
                        "normalized_error": normalized.copy(),
                        "gate_entry_gripper_closed": gripper_closed,
                        "gate_entry_part_grasped": part_grasped,
                        "gripper_hold_prefix_step": self._gripper_hold_prefix_step,
                        "reset_elapsed_s": time.perf_counter() - reset_started,
                        "rejection_reasons": self.rejection_reasons,
                        "pose_gate_blocked_ungrasped": (
                            self.pose_gate_blocked_ungrasped
                        ),
                    }
                    return self._observation(se3_error), info

                self._previous_se3_error = se3_error.copy()
                self._bc_action_current = self._finite_vector(
                    "prefix_bc_action",
                    self.backend.predict_prefix_bc_action(),
                    7,
                )
                executed, _, _ = self._apply(
                    np.zeros(6, dtype=np.float64),
                    force_gripper_closed=self._gripper_hold_latched,
                )
                if np.linalg.norm(executed) > 1e-15:
                    raise RuntimeError("zero residual changed the BC prefix action")
                if self.task.succeeded(self.backend):
                    rejected_reason = f"prefix_{self.task.success_reason}"
                    break
                if self.backend.episode_ended():
                    rejected_reason = "prefix_ended_before_gate"
                    break

            self._rejected_initializations += 1
            self._rejection_reasons[rejected_reason] = (
                self._rejection_reasons.get(rejected_reason, 0) + 1
            )
            self._last_prefix_hold_activated = self._gripper_hold_latched
            self._last_prefix_hold_step = self._gripper_hold_prefix_step
            self._clear_residual_state()
            if rejected_reason == "prefix_timeout":
                continue

        self._needs_reset = True
        raise RuntimeError(
            "failed to obtain a recoverable insertion-window state after "
            f"{self.config.max_reset_attempts} attempts; "
            f"rejection_rate={self.initialization_rejection_rate:.3f}"
        )

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before step()")
        residual_action = self._finite_vector("action", action, 6)
        if np.any(residual_action < -1.0) or np.any(residual_action > 1.0):
            raise ValueError("action components must lie in [-1, 1]")

        gripper_override_applied = bool(self._gripper_hold_latched)
        if not self.gate_active:
            residual_action = np.zeros(6, dtype=np.float64)
        delta_executed, bc_gripper, commanded_gripper = self._apply(
            residual_action,
            force_gripper_closed=self._gripper_hold_latched,
        )
        self._rl_steps += 1

        se3_error, normalized = self._errors()
        if self._previous_se3_error is None:
            self._se3_error_delta.fill(0.0)
        else:
            self._se3_error_delta = se3_error - self._previous_se3_error
        self._previous_se3_error = se3_error.copy()
        success = bool(self.task.succeeded(self.backend))
        gripper_open = bool(self.backend.gripper_is_open())
        gate_exit = bool(
            not success and self.gate_active and self._gate_exited(se3_error)
        )
        backend_ended = bool(self.backend.episode_ended())
        max_steps = self._rl_steps >= self.config.max_steps

        position_error_norm = float(np.linalg.norm(se3_error[:3]))
        if self._previous_position_error_norm is None:
            position_error_change = math.inf
        else:
            position_error_change = abs(
                position_error_norm - self._previous_position_error_norm
            )
        self._previous_position_error_norm = position_error_norm
        if position_error_change < self.config.stuck_position_change_m:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = 0
        stuck = bool(
            not success
            and not gate_exit
            and self._stagnant_steps >= self.config.stuck_window_steps
        )
        reward_result = self.task.reward(
            RewardContext(
                se3_error=se3_error.copy(),
                normalized_error=normalized.copy(),
                delta_executed=delta_executed.copy(),
                success=success,
                gate_exit=gate_exit,
                stuck=stuck,
            )
        )
        reward = float(reward_result.total)
        reward_components = {
            str(name): float(value)
            for name, value in reward_result.components.items()
        }
        if not np.isfinite(reward) or not all(
            np.isfinite(value) for value in reward_components.values()
        ):
            raise ValueError("ResidualTask reward and components must be finite")

        terminated = bool(success or gate_exit or stuck)
        truncated = bool((max_steps or backend_ended) and not terminated)
        done_reason = None
        if success:
            done_reason = self.task.success_reason
        elif gate_exit:
            done_reason = "gate_exit"
        elif stuck:
            done_reason = "stuck"
        elif max_steps:
            done_reason = "max_steps"
        elif backend_ended:
            done_reason = "backend_ended"

        contact_force = None
        part_pose = None
        if stuck:
            contact_force = self.backend.get_part_contact_force(
                self.task.target_name
            )
            part_pose = self._finite_vector(
                "part_pose", self.backend.get_part_pose(self.task.target_name), 7
            )
        stagnant_steps = self._stagnant_steps

        if not (terminated or truncated):
            self._predict_bc_action()

        observation = self._observation(se3_error)
        transition_se3_error_delta = self._se3_error_delta.copy()
        if terminated or truncated:
            self._clear_residual_state()
            self._needs_reset = True
        info = {
            "gate_active": bool(self.gate_active),
            "se3_error": se3_error.copy(),
            "se3_error_delta": transition_se3_error_delta,
            "normalized_error": normalized.copy(),
            "delta_executed": delta_executed.copy(),
            "reward_position": float(reward_components.get("position", 0.0)),
            "reward_orientation": float(reward_components.get("orientation", 0.0)),
            "reward_smooth": float(reward_components.get("smooth", 0.0)),
            "reward_terminal": float(reward_components.get("terminal", 0.0)),
            "reward_gate_exit": float(reward_components.get("gate_exit", 0.0)),
            "reward_stuck": float(reward_components.get("stuck", 0.0)),
            "reward_components": dict(reward_components),
            "done_reason": done_reason,
            "bottleneck_axis": self.task.bottleneck_label(normalized),
            "rl_steps": self._rl_steps,
            "bc_gripper": bc_gripper,
            "commanded_gripper": commanded_gripper,
            "gripper_override_active": gripper_override_applied,
            "measured_gripper_open": gripper_open,
            "position_error_change_m": position_error_change,
            "stagnant_steps": stagnant_steps,
            "contact_force": (
                None if contact_force is None else contact_force.copy()
            ),
            "part_pose": None if part_pose is None else part_pose.copy(),
        }
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self._clear_residual_state()
        self._needs_reset = True
        self.backend.close()
