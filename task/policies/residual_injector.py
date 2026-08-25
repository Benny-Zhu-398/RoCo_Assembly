"""Cartesian residual injection for frozen end-effector policies.

The residual is expressed in the world frame.  Its first three components are
a translation offset and its last three components are an SO(3) rotation
vector.  Both the total correction and the per-control-step change are bounded
by Euclidean/geodesic norms rather than independently clipping each axis.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class ResidualInjector:
    """Scale, slew-limit, and apply a 6-D Cartesian residual to a BC action."""

    def __init__(
        self,
        delta_max_pos: float = 0.003,
        delta_max_ori: float = np.deg2rad(30.0),
        max_pos_step: float = 0.0005,
        max_ori_step: float = np.deg2rad(3.0),
    ) -> None:
        limits = {
            "delta_max_pos": delta_max_pos,
            "delta_max_ori": delta_max_ori,
            "max_pos_step": max_pos_step,
            "max_ori_step": max_ori_step,
        }
        for name, value in limits.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

        self.delta_max_pos = float(delta_max_pos)
        self.delta_max_ori = float(delta_max_ori)
        self.max_pos_step = float(max_pos_step)
        self.max_ori_step = float(max_ori_step)

    @staticmethod
    def _finite_vector(name: str, value, size: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be one finite {size}-D vector")
        return vector

    @staticmethod
    def _limit_vector_step(
        previous: np.ndarray, target: np.ndarray, max_step: float
    ) -> np.ndarray:
        difference = target - previous
        distance = float(np.linalg.norm(difference))
        if distance <= max_step:
            return target.copy()
        return previous + difference * (max_step / distance)

    @staticmethod
    def _limit_rotation_step(
        previous_rotvec: np.ndarray,
        target_rotvec: np.ndarray,
        max_step: float,
    ) -> np.ndarray:
        previous = Rotation.from_rotvec(previous_rotvec)
        target = Rotation.from_rotvec(target_rotvec)

        # Left-multiplicative relative rotation: step * previous = target.
        relative_rotvec = (target * previous.inv()).as_rotvec()
        relative_angle = float(np.linalg.norm(relative_rotvec))
        if relative_angle > max_step:
            relative_rotvec *= max_step / relative_angle

        executed = Rotation.from_rotvec(relative_rotvec) * previous
        return executed.as_rotvec()

    def inject(self, bc_action, u, prev_executed):
        """Return ``(position, quat_wxyz, gripper, delta_executed)``.

        ``bc_action`` may be the left-only 7-D action or the legacy 14-D
        bimanual action; only its first seven (left-arm) values are consumed.
        ``u`` must be in [-1, 1].  ``prev_executed`` is the actual 6-D
        residual returned by the previous call, not the actor's prior target.
        """
        bc_action = np.asarray(bc_action, dtype=np.float64).reshape(-1)
        if bc_action.shape not in {(7,), (14,)} or not np.isfinite(bc_action).all():
            raise ValueError("bc_action must be one finite 7-D or 14-D action")
        u = self._finite_vector("u", u, 6)
        prev_executed = self._finite_vector("prev_executed", prev_executed, 6)
        if np.any(u < -1.0) or np.any(u > 1.0):
            raise ValueError("u components must lie in [-1, 1]")

        u_pos = u[:3]
        u_ori = u[3:]
        delta_pos_target = (
            self.delta_max_pos * u_pos / max(1.0, float(np.linalg.norm(u_pos)))
        )
        delta_rotvec_target = (
            self.delta_max_ori * u_ori / max(1.0, float(np.linalg.norm(u_ori)))
        )

        delta_pos_executed = self._limit_vector_step(
            prev_executed[:3], delta_pos_target, self.max_pos_step
        )
        delta_rotvec_executed = self._limit_rotation_step(
            prev_executed[3:], delta_rotvec_target, self.max_ori_step
        )
        delta_executed = np.concatenate([delta_pos_executed, delta_rotvec_executed])

        bc_rotation = Rotation.from_euler("xyz", bc_action[3:6])
        final_rotation = Rotation.from_rotvec(delta_rotvec_executed) * bc_rotation
        quat_xyzw = final_rotation.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        position = bc_action[:3] + delta_pos_executed
        gripper = float(bc_action[6])
        return position, quat_wxyz, gripper, delta_executed

    def inject_canonical(self, bc_action, u, prev_executed):
        """Inject into canonical ``xyz + rotvec + gripper`` base actions.

        Policy adapters use this representation so π0.5 (Euler) and DP
        (rotation vector) share exactly the same residual-control path.
        """

        bc_action = self._finite_vector("bc_action", bc_action, 7)
        u = self._finite_vector("u", u, 6)
        prev_executed = self._finite_vector("prev_executed", prev_executed, 6)
        if np.any(u < -1.0) or np.any(u > 1.0):
            raise ValueError("u components must lie in [-1, 1]")

        u_pos = u[:3]
        u_ori = u[3:]
        delta_pos_target = (
            self.delta_max_pos * u_pos / max(1.0, float(np.linalg.norm(u_pos)))
        )
        delta_rotvec_target = (
            self.delta_max_ori * u_ori / max(1.0, float(np.linalg.norm(u_ori)))
        )
        delta_pos_executed = self._limit_vector_step(
            prev_executed[:3], delta_pos_target, self.max_pos_step
        )
        delta_rotvec_executed = self._limit_rotation_step(
            prev_executed[3:], delta_rotvec_target, self.max_ori_step
        )
        delta_executed = np.concatenate([delta_pos_executed, delta_rotvec_executed])

        base_rotation = Rotation.from_rotvec(bc_action[3:6])
        final_rotation = Rotation.from_rotvec(delta_rotvec_executed) * base_rotation
        quat_xyzw = final_rotation.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return (
            bc_action[:3] + delta_pos_executed,
            quat_wxyz,
            float(bc_action[6]),
            delta_executed,
        )
