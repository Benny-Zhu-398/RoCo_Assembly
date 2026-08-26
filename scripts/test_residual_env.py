"""Deterministic tests for the Gym-style residual environment state machine."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from residual_env import ResidualEnv, ResidualEnvConfig  # noqa: E402


class FakeBackend:
    exec_horizon = 1

    def __init__(self):
        self.error = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.normalized = np.array([40.0, 0.0, 0.0, 0.0])
        self.applied = []
        self.advance_count = 0
        self.reset_count = 0
        self.snap = False
        self.open = False
        self.ended = False
        self.residual_control_started = False
        self.closed = True
        self.grasped = True

    def reset_episode(self, part_name):
        assert part_name == "hdmi"
        self.error = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.normalized = np.array([40.0, 0.0, 0.0, 0.0])
        self.applied.clear()
        self.advance_count = 0
        self.snap = self.open = self.ended = False
        self.residual_control_started = False
        self.reset_count += 1

    def get_physical_obs(self):
        return np.arange(44, dtype=np.float64)

    def get_se3_error(self, part_name):
        assert part_name == "hdmi"
        return self.error.copy()

    def get_normalized_error(self, part_name):
        assert part_name == "hdmi"
        return self.normalized.copy()

    def get_part_pose(self, part_name):
        assert part_name == "hdmi"
        return np.array([0.2, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0])

    def predict_bc_action(self):
        return np.array([0.2, 0.3, 1.0, 0.1, 0.2, 0.3, 0.09])

    def predict_prefix_bc_action(self):
        return self.predict_bc_action()

    def begin_residual_control(self):
        self.residual_control_started = True

    def apply_cartesian_action(self, position, quaternion_wxyz, gripper):
        self.applied.append((position.copy(), quaternion_wxyz.copy(), gripper))

    def advance(self):
        self.advance_count += 1
        if self.advance_count == 1:
            self.error = np.array([0.065, 0, 0, 0, 0, 0], dtype=np.float64)
            self.normalized = np.array([32.5, 0, 0, 0], dtype=np.float64)
        elif self.advance_count == 2:
            self.error = np.array([0.05, 0, 0, 0, 0, 0], dtype=np.float64)
            self.normalized = np.array([25.0, 0, 0, 0], dtype=np.float64)
        else:
            self.error = np.array([0.065, 0, 0, 0, 0, 0], dtype=np.float64)
            self.normalized = np.array([32.5, 0, 0, 0], dtype=np.float64)

    def snap_succeeded(self, part_name):
        return self.snap

    def gripper_is_open(self):
        return self.open

    def gripper_is_closed(self):
        return self.closed

    def closed_gripper_command(self):
        return 0.04

    def part_is_grasped(self, part_name):
        assert part_name == "hdmi"
        return self.grasped

    def get_part_contact_force(self, part_name):
        assert part_name == "hdmi"
        return np.array([1.0, 2.0, 3.0])

    def episode_ended(self):
        return self.ended

    def close(self):
        pass


class ResidualEnvTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.env = ResidualEnv(
            self.backend,
            ResidualEnvConfig(max_steps=5, max_prefix_steps=4),
        )

    def test_reset_replays_zero_residual_until_gate_entry(self):
        observation, info = self.env.reset(seed=1)
        self.assertEqual(observation.shape, (70,))
        self.assertTrue(info["gate_active"])
        self.assertEqual(info["prefix_steps"], 2)
        self.assertEqual(len(self.backend.applied), 2)
        self.assertTrue(self.backend.residual_control_started)
        self.assertTrue(
            all(np.array_equal(a[0], [0.2, 0.3, 1.0]) for a in self.backend.applied)
        )
        self.assertTrue(all(a[2] == 0.04 for a in self.backend.applied))
        self.assertEqual(info["gripper_hold_prefix_step"], 0)
        np.testing.assert_allclose(
            observation[50:56], [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            info["se3_error_delta"], [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(self.env.residual_executed_prev, np.zeros(6))

    def test_hysteresis_does_not_exit_between_sixty_and_seventy_mm(self):
        self.env.reset()
        observation, _, terminated, truncated, info = self.env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["gate_active"])
        np.testing.assert_allclose(
            observation[50:56], [0.015, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            info["se3_error_delta"], [0.015, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_snap_reward_and_terminal_state_clear(self):
        self.env.reset()
        self.backend.snap = True
        _, reward, terminated, truncated, info = self.env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_terminal"], 100.0)
        self.assertEqual(info["done_reason"], "snap")
        self.assertGreater(reward, -20.0)
        self.assertFalse(info["gate_active"])
        np.testing.assert_array_equal(self.env.residual_executed_prev, np.zeros(6))

    def test_gate_exits_above_seventy_mm(self):
        self.env.reset()
        self.backend.error = np.array([0.071, 0, 0, 0, 0, 0])
        self.backend.normalized = np.array([35.5, 0, 0, 0])
        self.backend.advance = lambda: None
        _, reward, terminated, _, info = self.env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertEqual(info["done_reason"], "gate_exit")
        self.assertEqual(info["reward_gate_exit"], -100.0)
        self.assertLess(reward, -100.0)

    def test_unrecoverable_orientation_is_resampled(self):
        original_advance = self.backend.advance

        def advance_with_one_rejection():
            original_advance()
            if self.backend.reset_count == 1 and self.backend.advance_count == 2:
                self.backend.error[3] = np.deg2rad(176.0)
            elif self.backend.reset_count == 2 and self.backend.advance_count == 2:
                self.backend.error[3:] = 0.0

        self.backend.advance = advance_with_one_rejection
        _, info = self.env.reset()
        self.assertEqual(self.backend.reset_count, 2)
        self.assertEqual(info["rejected_initializations"], 1)
        self.assertAlmostEqual(info["initialization_rejection_rate"], 0.5)

    def test_gate_requires_grasp_and_closed_gripper(self):
        self.backend.grasped = False
        env = ResidualEnv(
            self.backend,
            ResidualEnvConfig(max_steps=5, max_prefix_steps=3, max_reset_attempts=1),
        )
        with self.assertRaises(RuntimeError):
            env.reset()
        self.assertGreater(env.pose_gate_blocked_ungrasped, 0)

    def test_gate_forces_closed_command_and_ignores_bc_open(self):
        self.env.reset()
        self.backend.open = True
        _, _, terminated, truncated, info = self.env.step(np.zeros(6))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["bc_gripper"], 0.09)
        self.assertEqual(info["commanded_gripper"], 0.04)
        self.assertTrue(info["gripper_override_active"])
        self.assertEqual(self.backend.applied[-1][2], 0.04)

    def test_stuck_detector_terminates_and_logs_force_and_pose(self):
        env = ResidualEnv(
            self.backend,
            ResidualEnvConfig(max_steps=15, max_prefix_steps=4),
        )
        env.reset()
        self.backend.advance = lambda: None
        for step in range(1, 11):
            _, _, terminated, truncated, info = env.step(np.zeros(6))
            if step < 10:
                self.assertFalse(terminated)
                self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertEqual(info["done_reason"], "stuck")
        self.assertEqual(info["reward_stuck"], -100.0)
        self.assertEqual(info["stagnant_steps"], 10)
        np.testing.assert_array_equal(info["contact_force"], [1.0, 2.0, 3.0])
        self.assertEqual(info["part_pose"].shape, (7,))

    def test_v3_defaults(self):
        config = ResidualEnvConfig()
        self.assertEqual(config.max_steps, 64)
        self.assertAlmostEqual(config.gate_enter_ori_rad, np.deg2rad(45.0))
        self.assertEqual(config.gate_exit_reward, -100.0)
        self.assertEqual(config.stuck_reward, -100.0)
        self.assertAlmostEqual(config.smooth_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
