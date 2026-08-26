"""Frozen-policy adapters used by residual RL.

All adapters expose one canonical physical action representation:
``[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]``.  This keeps the
environment independent of π0.5's Euler output and DP's rotation-vector
output, and gives future policies one small integration surface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation


class ResidualPolicyAdapter(Protocol):
    """Canonical Cartesian interface for a frozen base policy."""

    name: str
    observation_dim: int
    exec_horizon: int

    def observation_features(self, observation) -> np.ndarray:
        """Return the base policy's physical/state observation features."""

    def predict_actions(
        self, observation, *, horizon: int, replan: bool
    ) -> list[np.ndarray]:
        """Return canonical 7-D physical actions."""

    def clear_cache(self) -> None:
        """Discard hidden action-chunk state at the RL boundary."""

    def close(self) -> None:
        """Release policy resources."""


def _finite_action(value) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("base policy action must be one finite 7-D vector")
    return action


def canonical_from_euler(action) -> np.ndarray:
    """Convert physical ``xyz + euler_xyz + gripper`` to canonical form."""

    action = _finite_action(action)
    rotvec = Rotation.from_euler("xyz", action[3:6]).as_rotvec()
    return np.concatenate([action[:3], rotvec, action[6:7]])


def canonical_from_rotvec(action, *, gripper_scale: float = 1.0) -> np.ndarray:
    """Normalize a physical/normalized DP action into canonical form."""

    action = _finite_action(action).copy()
    action[6] *= float(gripper_scale)
    return action


def _close_policy(policy) -> None:
    close = getattr(policy, "close", None)
    if callable(close):
        close()
        return
    process = getattr(policy, "_proc", None)
    if process is not None:
        try:
            process.terminate()
        except Exception:
            pass


class Pi05ResidualPolicyAdapter:
    """Adapter for ``Pi05LeRobotPolicy``."""

    name = "pi05"
    observation_dim = 44
    exec_horizon = 1

    def __init__(self, policy) -> None:
        if not callable(getattr(policy, "predict_raw", None)):
            raise TypeError("π0.5 adapter requires predict_raw")
        self.policy = policy

    def observation_features(self, observation) -> np.ndarray:
        return np.asarray(self.policy._build_state(observation), dtype=np.float64)

    def predict_actions(self, observation, *, horizon: int, replan: bool) -> list[np.ndarray]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if replan:
            cache = getattr(self.policy, "_action_cache", None)
            if cache is not None:
                cache.clear()
        prediction = self.policy.predict_raw(observation, exec_horizon=horizon)
        values = prediction.get("actions")
        if values is None:
            values = [prediction["action"]]
        actions = [canonical_from_euler(np.asarray(value).reshape(-1)[:7]) for value in values]
        if not actions:
            raise RuntimeError("π0.5 returned an empty action chunk")
        return actions[:horizon]

    def clear_cache(self) -> None:
        cache = getattr(self.policy, "_action_cache", None)
        if cache is not None:
            cache.clear()

    def close(self) -> None:
        _close_policy(self.policy)


class DiffusionLeRobotResidualPolicyAdapter:
    """Adapter for the vision LeRobot DiffusionPolicy subprocess wrapper."""

    name = "dp_lerobot"
    observation_dim = 44
    exec_horizon = 1

    def __init__(self, policy) -> None:
        required = ("_build_state", "_send", "_recv")
        if any(not callable(getattr(policy, name, None)) for name in required):
            raise TypeError("vision DP adapter requires _build_state/_send/_recv")
        self.policy = policy

    def observation_features(self, observation) -> np.ndarray:
        return np.asarray(self.policy._build_state(observation), dtype=np.float64)

    def _payload(self, observation) -> dict:
        from policies.diffusion_lerobot import _resize_rgb

        return {
            "state": self.policy._build_state(observation),
            "head": _resize_rgb(observation.rgb.get("head")),
            "left": _resize_rgb(observation.rgb.get("L_wrist")),
            "right": _resize_rgb(observation.rgb.get("R_wrist")),
        }

    def predict_actions(self, observation, *, horizon: int, replan: bool) -> list[np.ndarray]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if replan:
            self.clear_cache()
        scale = float(getattr(self.policy, "GRIPPER_OPEN_LIMIT", 0.6649704))
        # LeRobot owns its action queue inside the inference subprocess.  Ask
        # for one action per simulator observation and let subsequent prefix
        # calls consume that queue; requesting a whole chunk here would feed a
        # stale observation when the server replans after its queue empties.
        actions = []
        payload = self._payload(observation)
        for _ in range(1):
            self.policy._send(payload)
            raw = np.asarray(self.policy._recv()["action"], dtype=np.float64).reshape(-1)
            actions.append(canonical_from_rotvec(raw[:7], gripper_scale=scale))
        return actions

    def clear_cache(self) -> None:
        self.policy._send({"cmd": "reset"})
        self.policy._recv()

    def close(self) -> None:
        _close_policy(self.policy)


class DiffusionStateOnlyResidualPolicyAdapter:
    """Adapter for this repository's state-only DiffusionPolicy."""

    name = "dp_stateonly"
    observation_dim = 22
    exec_horizon = 1

    def __init__(self, policy) -> None:
        required = ("_build_state", "_send", "_recv")
        if any(not callable(getattr(policy, name, None)) for name in required):
            raise TypeError("state-only DP adapter requires _build_state/_send/_recv")
        self.policy = policy
        self._queue: deque[np.ndarray] = deque()

    def observation_features(self, observation) -> np.ndarray:
        return np.asarray(self.policy._build_state(observation), dtype=np.float64)

    def predict_actions(self, observation, *, horizon: int, replan: bool) -> list[np.ndarray]:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if replan:
            self.clear_cache()
        if len(self._queue) < horizon:
            self.policy._send({"state": self.policy._build_state(observation)})
            raw_horizon = np.asarray(
                self.policy._recv()["action_horizon"], dtype=np.float64
            )
            self._queue.extend(canonical_from_rotvec(value[:7]) for value in raw_horizon)
        return [self._queue.popleft() for _ in range(min(horizon, len(self._queue)))]

    def clear_cache(self) -> None:
        self._queue.clear()
        queue = getattr(self.policy, "_queue", None)
        if queue is not None:
            queue.clear()
        self.policy._send({"cmd": "reset"})
        self.policy._recv()

    def close(self) -> None:
        _close_policy(self.policy)


@dataclass
class CallableResidualPolicyAdapter:
    """Small extension point for future policies and tests."""

    name: str
    observation_dim: int
    feature_fn: Callable[[object], np.ndarray]
    predict_fn: Callable[[object, int, bool], list[np.ndarray]]
    clear_fn: Callable[[], None] = lambda: None
    close_fn: Callable[[], None] = lambda: None
    exec_horizon: int = 1

    def observation_features(self, observation) -> np.ndarray:
        return np.asarray(self.feature_fn(observation), dtype=np.float64)

    def predict_actions(self, observation, *, horizon: int, replan: bool) -> list[np.ndarray]:
        return [_finite_action(action) for action in self.predict_fn(observation, horizon, replan)]

    def clear_cache(self) -> None:
        self.clear_fn()

    def close(self) -> None:
        self.close_fn()


def make_residual_policy_adapter(policy) -> ResidualPolicyAdapter:
    """Create the built-in adapter for a loaded runner policy."""

    class_name = type(policy).__name__
    if class_name == "Pi05LeRobotPolicy":
        return Pi05ResidualPolicyAdapter(policy)
    if class_name == "DiffusionLeRobotPolicy":
        return DiffusionLeRobotResidualPolicyAdapter(policy)
    if class_name == "DiffusionStateOnlyPolicy":
        return DiffusionStateOnlyResidualPolicyAdapter(policy)
    raise TypeError(
        f"no residual policy adapter for {type(policy).__module__}.{class_name}; "
        "implement ResidualPolicyAdapter or extend make_residual_policy_adapter()"
    )
