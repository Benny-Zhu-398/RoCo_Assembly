"""CPU-only tests for residual policy/task extension interfaces."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from policies.residual_injector import ResidualInjector  # noqa: E402
from residual_policy import (  # noqa: E402
    CallableResidualPolicyAdapter,
    DiffusionLeRobotResidualPolicyAdapter,
    DiffusionStateOnlyResidualPolicyAdapter,
    Pi05ResidualPolicyAdapter,
    canonical_from_euler,
    make_residual_policy_adapter,
)
from residual_task import (  # noqa: E402
    BoundedSnapRewardConfig,
    RewardContext,
    RewardResult,
    SnapInsertionTaskConfig,
    make_residual_reward,
    make_residual_task,
    register_residual_reward,
)


class _FakePi05:
    def __init__(self):
        self._action_cache = [1]

    def _build_state(self, observation):
        del observation
        return np.arange(44, dtype=np.float32)

    def predict_raw(self, observation, exec_horizon=1):
        del observation
        action = np.array([0.2, 0.3, 1.0, 0.1, -0.2, 0.3, 0.04])
        return {"action": action, "actions": [action] * exec_horizon}


class _FakeStateOnlyDP:
    def __init__(self):
        self._queue = []
        self.sent = []

    def _build_state(self, observation):
        del observation
        return np.arange(22, dtype=np.float32)

    def _send(self, message):
        self.sent.append(message)

    def _recv(self):
        if self.sent[-1].get("cmd") == "reset":
            return {"ok": True}
        return {
            "action_horizon": [
                [0.2, 0.3, 1.0, 0.1, -0.2, 0.3, 0.04],
                [0.21, 0.3, 1.0, 0.1, -0.2, 0.3, 0.04],
            ]
        }


class _FakeVisionDP:
    def __init__(self):
        self.sent = []

    def _build_state(self, observation):
        del observation
        return np.arange(44, dtype=np.float32)

    def _send(self, message):
        self.sent.append(message)

    def _recv(self):
        if self.sent[-1].get("cmd") == "reset":
            return {"ok": True}
        return {"action": [0.2, 0.3, 1.0, 0.1, -0.2, 0.3, 0.5]}


class _FakeVisionObservation:
    rgb = {"head": None, "L_wrist": None, "R_wrist": None}


class _ErrorBackend:
    def __init__(self):
        self.success = False

    def get_se3_error(self, target_name):
        assert target_name == "part"
        return np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.1])

    def get_normalized_error(self, target_name):
        assert target_name == "part"
        return np.array([5.0, 0.0, 0.0, 0.5])

    def snap_succeeded(self, target_name):
        assert target_name == "part"
        return self.success


class ResidualInterfaceTest(unittest.TestCase):
    def test_pi05_euler_is_canonicalized_before_injection(self):
        raw = np.array([0.2, 0.3, 1.0, 0.1, -0.2, 0.3, 0.04])
        canonical = canonical_from_euler(raw)
        _, quat, _, executed = ResidualInjector().inject_canonical(
            canonical, np.zeros(6), np.zeros(6)
        )
        expected_xyzw = Rotation.from_euler("xyz", raw[3:6]).as_quat()
        np.testing.assert_allclose(quat, expected_xyzw[[3, 0, 1, 2]], atol=1e-12)
        np.testing.assert_array_equal(executed, np.zeros(6))

    def test_pi05_adapter_replans_and_exposes_44d_features(self):
        policy = _FakePi05()
        adapter = Pi05ResidualPolicyAdapter(policy)
        self.assertEqual(adapter.observation_features(None).shape, (44,))
        actions = adapter.predict_actions(None, horizon=1, replan=True)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].shape, (7,))
        self.assertEqual(policy._action_cache, [])

    def test_state_only_dp_adapter_exposes_canonical_chunk(self):
        policy = _FakeStateOnlyDP()
        adapter = DiffusionStateOnlyResidualPolicyAdapter(policy)
        self.assertEqual(adapter.observation_features(None).shape, (22,))
        actions = adapter.predict_actions(None, horizon=2, replan=True)
        self.assertEqual(len(actions), 2)
        np.testing.assert_allclose(actions[0][3:6], [0.1, -0.2, 0.3])
        self.assertEqual(policy.sent[0], {"cmd": "reset"})

    def test_vision_dp_adapter_scales_gripper_and_replans(self):
        policy = _FakeVisionDP()
        adapter = DiffusionLeRobotResidualPolicyAdapter(policy)
        actions = adapter.predict_actions(
            _FakeVisionObservation(), horizon=1, replan=True
        )
        self.assertEqual(policy.sent[0], {"cmd": "reset"})
        self.assertEqual(actions[0].shape, (7,))
        np.testing.assert_allclose(actions[0][3:6], [0.1, -0.2, 0.3])
        self.assertAlmostEqual(actions[0][6], 0.5 * 0.6649704)

    def test_callable_policy_adapter_is_a_public_extension_point(self):
        adapter = CallableResidualPolicyAdapter(
            name="custom",
            observation_dim=3,
            feature_fn=lambda observation: observation,
            predict_fn=lambda observation, horizon, replan: [
                np.zeros(7) for _ in range(horizon)
            ],
        )
        np.testing.assert_array_equal(adapter.observation_features([1, 2, 3]), [1, 2, 3])
        self.assertEqual(len(adapter.predict_actions(None, horizon=3, replan=False)), 3)

    def test_task_and_reward_are_independently_replaceable(self):
        class ConstantReward:
            def compute(self, context):
                return RewardResult(7.0 if context.success else -2.0, {"custom": 1.0})

        register_residual_reward(
            "test_constant", lambda *, config=None: ConstantReward(), replace=True
        )
        reward = make_residual_reward("test_constant")
        task = make_residual_task(
            "snap_insertion",
            config=SnapInsertionTaskConfig(target_name="part"),
            reward=reward,
        )
        backend = _ErrorBackend()
        se3, normalized = task.get_errors(backend)
        self.assertEqual(task.gate_entry_state(
            se3, gripper_closed=True, part_grasped=True
        ), "enter")
        result = task.reward(
            RewardContext(se3, normalized, np.zeros(6), False, False, False)
        )
        self.assertEqual(result.total, -2.0)
        backend.success = True
        self.assertTrue(task.succeeded(backend))

    def test_bounded_reward_factory_remains_configurable(self):
        reward = make_residual_reward(
            "bounded_snap",
            config=BoundedSnapRewardConfig(w_pos=2.0, w_ori=0.0),
        )
        result = reward.compute(
            RewardContext(
                np.zeros(6), np.array([1.0, 0.0, 0.0, 10.0]),
                np.zeros(6), False, False, False,
            )
        )
        self.assertAlmostEqual(result.components["orientation"], 0.0)
        self.assertLess(result.total, 0.0)


if __name__ == "__main__":
    unittest.main()
