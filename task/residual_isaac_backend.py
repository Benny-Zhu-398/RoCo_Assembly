"""Isaac-runner callback bridge for the policy-agnostic residual environment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import os

import numpy as np

from residual_policy import ResidualPolicyAdapter


@dataclass(frozen=True)
class GraspDetectionConfig:
    """Heuristic grasp-detector thresholds, overridable per task/robot."""

    sample_window: int = 4
    close_command_margin: float = 0.030
    open_command_margin: float = 0.030
    close_relative_distance_m: float = 0.23
    lifted_relative_distance_m: float = 0.25
    minimum_lift_m: float = 0.0005
    minimum_coupled_motion_m: float = 0.002
    coupled_motion_error_m: float = 0.003
    relative_motion_error_m: float = 0.003
    held_relative_error_m: float = 0.010

    def __post_init__(self) -> None:
        if self.sample_window < 3:
            raise ValueError("sample_window must be at least 3")
        values = tuple(
            value
            for name, value in vars(self).items()
            if name != "sample_window"
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("grasp detector thresholds must be finite and non-negative")


class IsaacResidualBackend:
    """Combine a frozen-policy adapter with simulator callbacks."""

    base_action_rotation_repr = "rotvec"

    def __init__(
        self,
        policy_adapter: ResidualPolicyAdapter,
        reset_episode: Callable[[str], None],
        get_observation: Callable[[], object],
        get_attacher: Callable[[str], object],
        apply_cartesian_action: Callable[[np.ndarray, np.ndarray, float], None],
        advance: Callable[[], None],
        episode_ended: Callable[[], bool] | None = None,
        get_contact_force: Callable[[str], np.ndarray | None] | None = None,
        prefix_exec_horizon: int = 16,
        part_config_provider: Callable[[str], dict] | None = None,
        grasp_config: GraspDetectionConfig | None = None,
    ) -> None:
        required = ("observation_features", "predict_actions", "clear_cache", "close")
        missing = [name for name in required if not callable(getattr(policy_adapter, name, None))]
        if missing:
            raise TypeError(f"invalid ResidualPolicyAdapter; missing={missing}")
        self.policy_adapter = policy_adapter
        self.exec_horizon = int(policy_adapter.exec_horizon)
        self._reset_episode = reset_episode
        self._get_observation = get_observation
        self._get_attacher = get_attacher
        self._apply_cartesian_action = apply_cartesian_action
        self._advance = advance
        self._episode_ended = episode_ended or (lambda: False)
        self._get_contact_force = get_contact_force or (lambda _part_name: None)
        self._part_config_provider = part_config_provider
        self.grasp_config = grasp_config or GraspDetectionConfig()
        if prefix_exec_horizon <= 0:
            raise ValueError("prefix_exec_horizon must be positive")
        self._prefix_exec_horizon = int(prefix_exec_horizon)
        self._prefix_actions = deque()
        self._part_name = None
        self._gripper_open_threshold = None
        self._gripper_open_command = None
        self._gripper_close_command = None
        self._grasp_samples = deque(maxlen=self.grasp_config.sample_window)
        self._grasp_latched = False
        self._grasp_relative_reference = None
        self._gate_observation = None
        self._part_initial_position = None
        self._grasp_query_count = 0
        self._grasp_debug = os.environ.get("ROCO_GRASP_DEBUG", "").lower() in {
            "1", "true", "yes", "on"
        }

    def reset_episode(self, part_name: str) -> None:
        self._part_name = str(part_name)
        self._prefix_actions.clear()
        self._grasp_samples.clear()
        self._grasp_latched = False
        self._grasp_relative_reference = None
        self._gate_observation = None
        self._grasp_query_count = 0
        self._reset_episode(self._part_name)
        self._part_initial_position = self.get_part_pose(self._part_name)[:3].copy()
        # The runner's reset callback calls policy.reset with PartTarget.
        # Recover its per-part gripper thresholds from param_config through
        # that same target configuration without hard-coding HDMI values.
        if self._part_config_provider is None:
            import param_config as pc

            cfg = pc.get_part_config(self._part_name)
        else:
            cfg = self._part_config_provider(self._part_name)
        gripper_open = float(cfg.get("gripper_open", 0.6649704))
        gripper_close = float(cfg.get("gripper_close", 0.0))
        self._gripper_open_command = gripper_open
        self._gripper_open_threshold = 0.5 * (gripper_open + gripper_close)
        self._gripper_close_command = gripper_close

    def _attacher(self, part_name: str):
        attacher = self._get_attacher(part_name)
        if attacher is None:
            raise RuntimeError(f"no SnapAttacher is active for {part_name!r}")
        return attacher

    def get_physical_obs(self) -> np.ndarray:
        return np.asarray(
            self.policy_adapter.observation_features(self._get_observation()),
            dtype=np.float64,
        )

    def get_se3_error(self, part_name: str) -> np.ndarray:
        return self._attacher(part_name).get_se3_error(part_name)

    def get_normalized_error(self, part_name: str) -> np.ndarray:
        return self._attacher(part_name).get_normalized_error(part_name)

    def get_part_pose(self, part_name: str) -> np.ndarray:
        return self._attacher(part_name).get_current_pose(part_name)

    def predict_bc_action(self) -> np.ndarray:
        actions = self.policy_adapter.predict_actions(
            self._get_observation(), horizon=1, replan=True
        )
        if not actions:
            raise RuntimeError("base policy adapter returned no action")
        return np.asarray(actions[0], dtype=np.float64)

    def predict_prefix_bc_action(self) -> np.ndarray:
        if not self._prefix_actions:
            actions = self.policy_adapter.predict_actions(
                self._get_observation(),
                horizon=self._prefix_exec_horizon,
                replan=False,
            )
            self._prefix_actions.extend(
                np.asarray(action, dtype=np.float64).reshape(7).copy()
                for action in actions
            )
        if not self._prefix_actions:
            raise RuntimeError("base policy adapter returned an empty prefix chunk")
        return self._prefix_actions.popleft()

    def begin_residual_control(self) -> None:
        self._prefix_actions.clear()
        self.policy_adapter.clear_cache()

    def apply_cartesian_action(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, gripper: float
    ) -> None:
        self._apply_cartesian_action(position, quaternion_wxyz, gripper)

    def advance(self) -> None:
        self._advance()

    def snap_succeeded(self, part_name: str) -> bool:
        return bool(self._attacher(part_name).attached)

    def gripper_is_open(self) -> bool:
        if self._gripper_open_threshold is None:
            raise RuntimeError("backend must be reset before querying gripper")
        obs = self._get_observation()
        return bool(
            not self._grasp_latched
            and float(obs.L_gripper_position) >= self._gripper_open_threshold
        )

    def gripper_is_closed(self) -> bool:
        if self._gripper_open_threshold is None:
            raise RuntimeError("backend must be reset before querying gripper")
        obs = self._get_observation()
        self._gate_observation = obs
        return bool(
            self._grasp_latched
            or float(obs.L_gripper_position) < self._gripper_open_threshold
        )

    def closed_gripper_command(self) -> float:
        if self._gripper_close_command is None:
            raise RuntimeError("backend must be reset before querying gripper command")
        return self._gripper_close_command

    def part_is_grasped(self, part_name: str) -> bool:
        """Infer grasp from coupled EE/part motion and stable relative position.

        This scene has no explicit grasp attachment.  Requiring non-zero,
        matching motion prevents a stationary closed gripper from being
        mistaken for a grasp, while the latched relative-position check still
        detects a part that is dropped before gate entry.
        """
        observation = self._gate_observation
        self._gate_observation = None
        if observation is None:
            observation = self._get_observation()
        ee_position = np.asarray(observation.ee_pose_L[0], dtype=np.float64)
        part_position = self.get_part_pose(part_name)[:3]
        relative = part_position - ee_position
        self._grasp_samples.append((ee_position.copy(), part_position.copy(), relative))
        self._grasp_query_count += 1
        gripper_position = float(observation.L_gripper_position)
        lift_delta = (
            float(part_position[2] - self._part_initial_position[2])
            if self._part_initial_position is not None
            else float("nan")
        )

        # When a rigid part sits between the fingers, the measured joint may
        # never reach the commanded close value.  For bolt_8mm, for example,
        # the fingers stabilize around 0.06 while the configured midpoint is
        # 0.05.  Treat contact-width closure near the part, or a small lift
        # during closing, as grasp evidence instead of requiring that midpoint.
        close_contact_evidence = bool(
            self._gripper_close_command is not None
            and gripper_position
            <= self._gripper_close_command + self.grasp_config.close_command_margin
            and np.linalg.norm(relative)
            <= self.grasp_config.close_relative_distance_m
        )
        lifted_evidence = bool(
            self._gripper_open_command is not None
            and gripper_position
            <= self._gripper_open_command + self.grasp_config.open_command_margin
            and lift_delta >= self.grasp_config.minimum_lift_m
            and np.linalg.norm(relative)
            <= self.grasp_config.lifted_relative_distance_m
        )
        pickup_evidence = close_contact_evidence or lifted_evidence

        if self._grasp_latched:
            still_held = bool(
                gripper_position
                <= self._gripper_open_command + self.grasp_config.open_command_margin
                and np.linalg.norm(relative - self._grasp_relative_reference)
                <= self.grasp_config.held_relative_error_m
            )
            if still_held:
                return True
            self._grasp_latched = False
            self._grasp_relative_reference = None

        if gripper_position >= self._gripper_open_threshold and not pickup_evidence:
            if self._grasp_debug:
                print(
                    f"[residual.grasp] query={self._grasp_query_count} "
                    f"gripper={gripper_position:.5f} "
                    f"threshold={self._gripper_open_threshold:.5f} "
                    f"lift_m={lift_delta:+.5f} rel_m={np.linalg.norm(relative):.5f} "
                    "result=open",
                    flush=True,
                )
            self._grasp_samples.clear()
            self._grasp_latched = False
            self._grasp_relative_reference = None
            return False

        if len(self._grasp_samples) < 3:
            return False
        ee_first, part_first, relative_first = self._grasp_samples[0]
        ee_last, part_last, relative_last = self._grasp_samples[-1]
        ee_delta = ee_last - ee_first
        part_delta = part_last - part_first
        moved_together = (
            np.linalg.norm(ee_delta) >= self.grasp_config.minimum_coupled_motion_m
            and np.linalg.norm(part_delta)
            >= self.grasp_config.minimum_coupled_motion_m
            and np.linalg.norm(part_delta - ee_delta)
            <= self.grasp_config.coupled_motion_error_m
            and np.linalg.norm(relative_last - relative_first)
            <= self.grasp_config.relative_motion_error_m
        )
        # Small rigid parts can slip a few millimetres between the fingers,
        # making the strict frame-to-frame coupling test above miss a visually
        # obvious pickup.  Contact-width closure near the part, or a part
        # lifted from its reset height while the fingers are closing, is
        # independent evidence of grasp and latches the global hold.
        lifted_from_reset = pickup_evidence
        if moved_together or lifted_from_reset:
            self._grasp_latched = True
            self._grasp_relative_reference = relative.copy()
        if self._grasp_debug:
            print(
                f"[residual.grasp] query={self._grasp_query_count} "
                f"gripper={gripper_position:.5f} "
                f"threshold={self._gripper_open_threshold:.5f} "
                f"lift_m={lift_delta:+.5f} rel_m={np.linalg.norm(relative):.5f} "
                f"moved_together={int(moved_together)} "
                f"close_contact={int(close_contact_evidence)} "
                f"lifted={int(lifted_from_reset)} "
                f"latched={int(self._grasp_latched)}",
                flush=True,
            )
        return self._grasp_latched

    def get_part_contact_force(self, part_name: str) -> np.ndarray | None:
        force = self._get_contact_force(part_name)
        if force is None:
            return None
        vector = np.asarray(force, dtype=np.float64).reshape(-1)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            return None
        return vector

    def episode_ended(self) -> bool:
        return bool(self._episode_ended())

    def close(self) -> None:
        self.policy_adapter.close()


class Pi05IsaacResidualBackend(IsaacResidualBackend):
    """Backward-compatible name; new code should use ``IsaacResidualBackend``."""
