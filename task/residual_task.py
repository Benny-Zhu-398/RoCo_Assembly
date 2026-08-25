"""Task and reward interfaces for residual Cartesian control.

The RL algorithm should not know which task-board part is active, how success
is detected, or how dense reward is shaped.  Those decisions live behind the
small interfaces in this module.  ``SnapInsertionTask`` preserves the original
HDMI behaviour and can also be configured for the other snap-attached parts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    """Task-independent signals available to a residual reward function."""

    se3_error: np.ndarray
    normalized_error: np.ndarray
    delta_executed: np.ndarray
    success: bool
    gate_exit: bool
    stuck: bool


@dataclass(frozen=True)
class RewardResult:
    """Scalar reward plus named terms for logging and ablations."""

    total: float
    components: dict[str, float]


class ResidualReward(Protocol):
    """Pluggable reward used by :class:`ResidualTask`."""

    def compute(self, context: RewardContext) -> RewardResult:
        """Compute one reward from the current task signals."""


@dataclass(frozen=True)
class BoundedSnapRewardConfig:
    """Configuration for the bounded snap-insertion reward."""

    w_pos: float = 1.0
    w_ori: float = 1.0
    smooth_weight: float = 1.0
    terminal_reward: float = 100.0
    gate_exit_reward: float = -100.0
    stuck_reward: float = -100.0

    def __post_init__(self) -> None:
        if self.w_pos < 0.0 or self.w_ori < 0.0:
            raise ValueError("position/orientation reward weights must be non-negative")
        if self.smooth_weight < 0.0:
            raise ValueError("smooth_weight must be non-negative")


class BoundedSnapReward:
    """Original bounded reward, separated from the environment state machine."""

    def __init__(self, config: BoundedSnapRewardConfig | None = None) -> None:
        self.config = config or BoundedSnapRewardConfig()

    def compute(self, context: RewardContext) -> RewardResult:
        normalized = np.asarray(context.normalized_error, dtype=np.float64).reshape(-1)
        if normalized.size < 4:
            raise ValueError("BoundedSnapReward requires at least four normalized errors")
        delta = np.asarray(context.delta_executed, dtype=np.float64).reshape(-1)
        position = -math.tanh(float(np.max(np.abs(normalized[:3]))))
        orientation = -math.tanh(abs(float(normalized[3])))
        smooth = -self.config.smooth_weight * float(np.linalg.norm(delta))
        terminal = self.config.terminal_reward if context.success else 0.0
        gate_exit = self.config.gate_exit_reward if context.gate_exit else 0.0
        stuck = self.config.stuck_reward if context.stuck else 0.0
        components = {
            "position": self.config.w_pos * position,
            "orientation": self.config.w_ori * orientation,
            "smooth": smooth,
            "terminal": terminal,
            "gate_exit": gate_exit,
            "stuck": stuck,
        }
        total = (
            components["position"]
            + components["orientation"]
            + smooth
            + terminal
            + gate_exit
            + stuck
        )
        return RewardResult(total=float(total), components=components)


class ResidualTask(Protocol):
    """Task semantics consumed by the generic residual environment."""

    name: str
    target_name: str
    success_reason: str
    requires_closed_gripper: bool
    requires_grasp: bool
    hold_gripper_closed: bool

    def get_errors(self, backend) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame SE(3) error and task-normalized errors."""

    def gate_entry_state(
        self,
        se3_error: np.ndarray,
        *,
        gripper_closed: bool,
        part_grasped: bool,
    ) -> str:
        """Return one of ``outside``, ``ungrasped``, ``reject``, or ``enter``."""

    def gate_exited(self, se3_error: np.ndarray) -> bool:
        """Whether a previously active residual window has been exited."""

    def succeeded(self, backend) -> bool:
        """Whether the task has reached its terminal success condition."""

    def reward(self, context: RewardContext) -> RewardResult:
        """Compute task reward and named telemetry components."""

    def bottleneck_label(self, normalized_error: np.ndarray) -> str:
        """Return a short diagnostic label for the dominant error."""


@dataclass(frozen=True)
class SnapInsertionTaskConfig:
    """Gate configuration shared by HDMI/USB/bolt/pin/rod insertion tasks."""

    target_name: str
    name: str | None = None
    gate_enter_pos_m: float = 0.060
    gate_exit_pos_m: float = 0.070
    gate_enter_ori_rad: float = math.radians(45.0)
    gate_exit_ori_rad: float = math.radians(60.0)
    require_closed_gripper: bool = True
    require_grasp: bool = True
    hold_gripper_closed: bool = True

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name must not be empty")
        if not 0.0 < self.gate_enter_pos_m < self.gate_exit_pos_m:
            raise ValueError("position gate must have positive hysteresis")
        if not 0.0 < self.gate_enter_ori_rad < self.gate_exit_ori_rad:
            raise ValueError("orientation gate must have positive hysteresis")


class SnapInsertionTask:
    """Residual task backed by a task-board ``SnapAttacher``."""

    success_reason = "snap"

    def __init__(
        self,
        config: SnapInsertionTaskConfig,
        reward: ResidualReward | None = None,
    ) -> None:
        self.config = config
        self.name = config.name or f"snap_insertion:{config.target_name}"
        self.target_name = config.target_name
        self.requires_closed_gripper = config.require_closed_gripper
        self.requires_grasp = config.require_grasp
        self.hold_gripper_closed = config.hold_gripper_closed
        self._reward = reward or BoundedSnapReward()

    def get_errors(self, backend) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(backend.get_se3_error(self.target_name), dtype=np.float64),
            np.asarray(backend.get_normalized_error(self.target_name), dtype=np.float64),
        )

    def gate_entry_state(
        self,
        se3_error: np.ndarray,
        *,
        gripper_closed: bool,
        part_grasped: bool,
    ) -> str:
        position_error = float(np.linalg.norm(se3_error[:3]))
        orientation_error = float(np.linalg.norm(se3_error[3:]))
        if position_error >= self.config.gate_enter_pos_m:
            return "outside"
        if (
            (self.config.require_closed_gripper and not gripper_closed)
            or (self.config.require_grasp and not part_grasped)
        ):
            return "ungrasped"
        if orientation_error >= self.config.gate_enter_ori_rad:
            return "reject"
        return "enter"

    def gate_exited(self, se3_error: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(se3_error[:3]) > self.config.gate_exit_pos_m
            or np.linalg.norm(se3_error[3:]) > self.config.gate_exit_ori_rad
        )

    def succeeded(self, backend) -> bool:
        return bool(backend.snap_succeeded(self.target_name))

    def reward(self, context: RewardContext) -> RewardResult:
        return self._reward.compute(context)

    def bottleneck_label(self, normalized_error: np.ndarray) -> str:
        normalized = np.asarray(normalized_error, dtype=np.float64).reshape(-1)
        # Keep the historical position-axis diagnostic.  Orientation is
        # already logged separately and has a different unit/threshold.
        labels = ("x", "y", "z")
        count = min(len(labels), normalized.size)
        if count == 0:
            return "unknown"
        return labels[int(np.argmax(np.abs(normalized[:count])))]


TaskFactory = Callable[..., ResidualTask]
_TASK_FACTORIES: dict[str, TaskFactory] = {}
RewardFactory = Callable[..., ResidualReward]
_REWARD_FACTORIES: dict[str, RewardFactory] = {}


def register_residual_task(name: str, factory: TaskFactory, *, replace: bool = False) -> None:
    """Register an application task without changing ``ResidualEnv``."""

    if not name or not callable(factory):
        raise ValueError("task registration requires a name and callable factory")
    if name in _TASK_FACTORIES and not replace:
        raise KeyError(f"residual task {name!r} is already registered")
    _TASK_FACTORIES[name] = factory


def make_residual_task(name: str, **kwargs) -> ResidualTask:
    """Construct a registered residual task."""

    try:
        factory = _TASK_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown residual task {name!r}; available={sorted(_TASK_FACTORIES)}"
        ) from exc
    return factory(**kwargs)


def register_residual_reward(
    name: str, factory: RewardFactory, *, replace: bool = False
) -> None:
    """Register a reward implementation independently from task dynamics."""

    if not name or not callable(factory):
        raise ValueError("reward registration requires a name and callable factory")
    if name in _REWARD_FACTORIES and not replace:
        raise KeyError(f"residual reward {name!r} is already registered")
    _REWARD_FACTORIES[name] = factory


def make_residual_reward(name: str, **kwargs) -> ResidualReward:
    """Construct a registered residual reward."""

    try:
        factory = _REWARD_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown residual reward {name!r}; available={sorted(_REWARD_FACTORIES)}"
        ) from exc
    return factory(**kwargs)


register_residual_task(
    "snap_insertion",
    lambda *, config, reward=None: SnapInsertionTask(config, reward=reward),
)
register_residual_reward(
    "bounded_snap",
    lambda *, config=None: BoundedSnapReward(config),
)
