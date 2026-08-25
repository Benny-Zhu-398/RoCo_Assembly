"""CPU-only tests for the residual TD3 implementation."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "task"))

from residual_td3 import Actor, ReplayBuffer, TD3  # noqa: E402


class ResidualTD3Test(unittest.TestCase):
    def test_actor_architecture_and_action_bounds(self):
        actor = Actor(64)
        layer_norm_count = sum(
            isinstance(module, torch.nn.LayerNorm) for module in actor.modules()
        )
        self.assertEqual(layer_norm_count, 3)
        action = actor(torch.zeros(5, 64))
        self.assertEqual(tuple(action.shape), (5, 6))
        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))

    def test_numpy_replay_buffer_wraps_and_samples(self):
        buffer = ReplayBuffer(4, 2, capacity=5, seed=3)
        for index in range(8):
            buffer.add(
                np.full(4, index),
                np.full(2, index),
                np.full(4, index + 1),
                index,
                index % 2,
            )
        self.assertEqual(len(buffer), 5)
        batch = buffer.sample(4, torch.device("cpu"))
        self.assertEqual(tuple(batch[0].shape), (4, 4))
        self.assertEqual(tuple(batch[1].shape), (4, 2))
        self.assertTrue(all(tensor.dtype == torch.float32 for tensor in batch))

    def test_td3_updates_critic_and_delays_actor(self):
        buffer = ReplayBuffer(8, 6, capacity=300, seed=1)
        rng = np.random.default_rng(2)
        for _ in range(300):
            buffer.add(
                rng.normal(size=8),
                rng.uniform(-1, 1, size=6),
                rng.normal(size=8),
                rng.normal(),
                rng.random() < 0.1,
            )
        td3 = TD3(8, seed=4)
        first = td3.train(buffer, batch_size=64)
        second = td3.train(buffer, batch_size=64)
        self.assertFalse(first["actor_updated"])
        self.assertTrue(second["actor_updated"])
        self.assertTrue(np.isfinite(first["critic_loss"]))
        self.assertTrue(np.isfinite(second["critic_loss"]))
        self.assertTrue(np.isfinite(second["actor_loss"]))


if __name__ == "__main__":
    unittest.main()
