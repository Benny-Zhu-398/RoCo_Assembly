"""Unit tests for the pi0.5 left-arm Cartesian safety filter."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from policies.pi05_lerobot import (  # noqa: E402
    _euler_xyz_to_quat_wxyz,
    _filter_left_action,
    _guard_left_joint_action,
)


def _quat_wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class SafetyFilterTest(unittest.TestCase):
    def setUp(self):
        self.position = np.array([0.2, -0.1, 1.3], dtype=np.float64)
        self.rotation = Rotation.from_euler("xyz", [20.0, -10.0, 30.0], degrees=True)
        self.quaternion = _quat_wxyz(self.rotation)

    def action(self, position_delta, rotation_delta_deg, gripper=0.7):
        target_rotation = (
            Rotation.from_rotvec(
                np.deg2rad(rotation_delta_deg)
                * np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
            * self.rotation
        )
        action = np.zeros(14, dtype=np.float64)
        action[:3] = self.position + np.asarray(position_delta, dtype=np.float64)
        action[3:6] = target_rotation.as_euler("xyz")
        action[6] = gripper
        return action

    def test_small_action_passes_unchanged(self):
        action = self.action([0.001, 0.002, 0.0], 2.0)
        filtered, diagnostics = _filter_left_action(
            action, self.position, self.quaternion, 0.25
        )
        np.testing.assert_allclose(filtered, action, atol=1e-12)
        self.assertFalse(diagnostics["held"])

    def test_translation_is_limited_to_five_mm(self):
        action = self.action([0.03, 0.04, 0.0], 0.0)
        filtered, diagnostics = _filter_left_action(
            action, self.position, self.quaternion, 0.25
        )
        self.assertAlmostEqual(
            np.linalg.norm(filtered[:3] - self.position), 0.005, places=12
        )
        self.assertFalse(diagnostics["held"])

    def test_rotation_is_limited_geometrically_to_five_degrees(self):
        action = self.action([0.0, 0.0, 0.0], 30.0)
        filtered, diagnostics = _filter_left_action(
            action, self.position, self.quaternion, 0.25
        )
        filtered_rotation = Rotation.from_euler("xyz", filtered[3:6])
        angle = np.linalg.norm((filtered_rotation * self.rotation.inv()).as_rotvec())
        self.assertAlmostEqual(np.degrees(angle), 5.0, places=10)
        self.assertFalse(diagnostics["held"])

    def test_large_translation_holds_pose_and_gripper(self):
        action = self.action([0.11, 0.0, 0.0], 0.0, gripper=0.9)
        filtered, diagnostics = _filter_left_action(
            action, self.position, self.quaternion, 0.25
        )
        np.testing.assert_allclose(filtered[:3], self.position, atol=1e-12)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", filtered[3:6]).as_matrix(),
            self.rotation.as_matrix(),
            atol=1e-12,
        )
        self.assertEqual(filtered[6], 0.25)
        self.assertEqual(diagnostics["hold_reason"], "translation")

    def test_large_rotation_holds_pose_and_gripper(self):
        action = self.action([0.001, 0.0, 0.0], 46.0, gripper=0.9)
        filtered, diagnostics = _filter_left_action(
            action, self.position, self.quaternion, 0.25
        )
        np.testing.assert_allclose(filtered[:3], self.position, atol=1e-12)
        self.assertEqual(filtered[6], 0.25)
        self.assertEqual(diagnostics["hold_reason"], "rotation")

    def test_formal_dataset_rotation_is_euler_xyz_not_rotvec(self):
        action_euler_xyz = np.array(
            [2.920008421, 0.717658699, 0.570686519], dtype=np.float64
        )
        expected_wxyz = np.array(
            [-0.1975835413, -0.8820108771, -0.2992008626, 0.3057719469],
            dtype=np.float64,
        )

        decoded_wxyz = _euler_xyz_to_quat_wxyz(*action_euler_xyz)
        decoded = Rotation.from_quat(decoded_wxyz[[1, 2, 3, 0]])
        expected = Rotation.from_quat(expected_wxyz[[1, 2, 3, 0]])
        euler_error_deg = np.degrees(
            np.linalg.norm((decoded * expected.inv()).as_rotvec())
        )
        rotvec_error_deg = np.degrees(
            np.linalg.norm(
                (Rotation.from_rotvec(action_euler_xyz) * expected.inv()).as_rotvec()
            )
        )

        self.assertLess(euler_error_deg, 1e-5)
        self.assertGreater(rotvec_error_deg, 50.0)

    def test_small_ik_joint_step_passes(self):
        current = np.zeros(16, dtype=np.float64)
        targets = [None] * 16
        for index in range(7):
            targets[index] = np.deg2rad(2.0)
        targets[14] = 0.4
        guarded, diagnostics = _guard_left_joint_action(
            targets, current, range(7), 14
        )
        self.assertFalse(diagnostics["held"])
        np.testing.assert_allclose(guarded[:7], np.deg2rad(2.0))
        self.assertEqual(guarded[14], 0.4)

    def test_large_ik_joint_step_holds_arm_and_gripper(self):
        current = np.linspace(0.0, 0.15, 16)
        targets = current.astype(object).tolist()
        targets[3] = float(current[3] + np.deg2rad(30.0))
        targets[14] = 0.6
        guarded, diagnostics = _guard_left_joint_action(
            targets, current, range(7), 14
        )
        self.assertTrue(diagnostics["held"])
        self.assertEqual(diagnostics["reason"], "joint_jump")
        np.testing.assert_allclose(guarded[:7], current[:7])
        self.assertEqual(guarded[14], current[14])

    def test_invalid_ik_joint_target_holds(self):
        current = np.zeros(16, dtype=np.float64)
        targets = current.astype(object).tolist()
        targets[2] = None
        guarded, diagnostics = _guard_left_joint_action(
            targets, current, range(7), 14
        )
        self.assertTrue(diagnostics["held"])
        self.assertEqual(diagnostics["reason"], "invalid_ik")
        np.testing.assert_allclose(guarded[:7], current[:7])


if __name__ == "__main__":
    unittest.main()
