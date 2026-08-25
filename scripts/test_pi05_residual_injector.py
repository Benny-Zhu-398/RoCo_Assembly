"""Unit tests for the pi0.5 Cartesian residual injection path."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from policies.residual_injector import ResidualInjector  # noqa: E402


def _rotation_from_wxyz(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


class ResidualInjectorTest(unittest.TestCase):
    def setUp(self):
        self.injector = ResidualInjector()
        self.bc_action = np.array(
            [0.42, -0.13, 1.08, 0.4, -0.7, 1.2, 0.21], dtype=np.float64
        )

    def test_zero_residual_is_identical_to_original_decode(self):
        position, quaternion, gripper, executed = self.injector.inject(
            self.bc_action, np.zeros(6), np.zeros(6)
        )
        np.testing.assert_array_equal(position, self.bc_action[:3])
        np.testing.assert_array_equal(executed, np.zeros(6))
        self.assertEqual(gripper, self.bc_action[6])

        original = Rotation.from_euler("xyz", self.bc_action[3:6])
        original_xyzw = original.as_quat()
        np.testing.assert_allclose(
            quaternion, original_xyzw[[3, 0, 1, 2]], atol=3e-16, rtol=0.0
        )
        injected = _rotation_from_wxyz(quaternion)
        error = np.linalg.norm((injected * original.inv()).as_rotvec())
        self.assertLess(error, 1e-15)

    def test_action_scaling_uses_separate_l2_norm_limits(self):
        _, _, _, executed = ResidualInjector(
            max_pos_step=1.0, max_ori_step=np.pi
        ).inject(self.bc_action, np.ones(6), np.zeros(6))
        self.assertAlmostEqual(np.linalg.norm(executed[:3]), 0.003, places=14)
        self.assertAlmostEqual(
            np.linalg.norm(executed[3:]), np.deg2rad(30.0), places=14
        )

    def test_position_slew_is_l2_limited(self):
        previous = np.zeros(6)
        _, _, _, executed = self.injector.inject(
            self.bc_action, np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), previous
        )
        self.assertAlmostEqual(np.linalg.norm(executed[:3]), 0.0005, places=14)

    def test_orientation_ramps_to_thirty_degrees_in_so3(self):
        u = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        previous = np.zeros(6)
        observed_degrees = [0.0]
        step_degrees = []
        for _ in range(11):
            _, _, _, executed = self.injector.inject(self.bc_action, u, previous)
            previous_rotation = Rotation.from_rotvec(previous[3:])
            executed_rotation = Rotation.from_rotvec(executed[3:])
            step_degrees.append(
                np.degrees(
                    np.linalg.norm(
                        (executed_rotation * previous_rotation.inv()).as_rotvec()
                    )
                )
            )
            observed_degrees.append(np.degrees(np.linalg.norm(executed[3:])))
            previous = executed

        np.testing.assert_allclose(
            observed_degrees,
            [
                0.0,
                3.0,
                6.0,
                9.0,
                12.0,
                15.0,
                18.0,
                21.0,
                24.0,
                27.0,
                30.0,
                30.0,
            ],
            atol=1e-12,
        )
        self.assertLessEqual(max(step_degrees), 3.0 + 1e-12)

    def test_world_frame_rotation_is_left_multiplied(self):
        injector = ResidualInjector(max_ori_step=np.pi)
        u = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        _, quaternion, _, executed = injector.inject(self.bc_action, u, np.zeros(6))
        actual = _rotation_from_wxyz(quaternion)
        expected = Rotation.from_rotvec(executed[3:]) * Rotation.from_euler(
            "xyz", self.bc_action[3:6]
        )
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-14)

    def test_rejects_out_of_range_actor_action(self):
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            self.injector.inject(
                self.bc_action, np.array([1.01, 0, 0, 0, 0, 0]), np.zeros(6)
            )


if __name__ == "__main__":
    unittest.main()
