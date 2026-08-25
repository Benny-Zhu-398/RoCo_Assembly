"""Minimal TD3 implementation for the task-board residual policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    policy_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    hidden_dim: int = 256
    hidden_layers: int = 3

    def __post_init__(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.policy_freq <= 0:
            raise ValueError("policy_freq must be positive")
        if self.policy_noise < 0.0 or self.noise_clip < 0.0:
            raise ValueError("target-noise parameters must be non-negative")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("network dimensions must be positive")


class ReplayBuffer:
    """Fixed-capacity NumPy replay buffer with uniform random sampling."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int = 100_000,
        seed: int = 0,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0 or capacity <= 0:
            raise ValueError("buffer dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.observations = np.empty(
            (self.capacity, observation_dim), dtype=np.float32
        )
        self.actions = np.empty((self.capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, observation, action, next_observation, reward, done) -> None:
        index = self._position
        self.observations[index] = np.asarray(observation, dtype=np.float32)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.next_observations[index] = np.asarray(
            next_observation, dtype=np.float32
        )
        self.rewards[index, 0] = float(reward)
        self.dones[index, 0] = float(bool(done))
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if batch_size <= 0 or self._size < batch_size:
            raise ValueError(
                f"cannot sample batch_size={batch_size} from size={self._size}"
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        arrays = (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.rewards[indices],
            self.dones[indices],
        )
        return tuple(torch.as_tensor(array, device=device) for array in arrays)


def _hidden_stack(input_dim: int, hidden_dim: int, hidden_layers: int):
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        layers.extend(
            [nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        )
        width = hidden_dim
    return layers, width


class Actor(nn.Module):
    """Three-hidden-layer LayerNorm MLP with a six-dimensional tanh head."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        layers, width = _hidden_stack(
            observation_dim, hidden_dim, hidden_layers
        )
        output = nn.Linear(width, action_dim)
        nn.init.uniform_(output.weight, -3e-3, 3e-3)
        nn.init.zeros_(output.bias)
        layers.extend([output, nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers, width = _hidden_stack(
            observation_dim + action_dim, hidden_dim, hidden_layers
        )
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=-1))


class TwinCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 6,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(
            observation_dim, action_dim, hidden_dim, hidden_layers
        )
        self.q2 = QNetwork(
            observation_dim, action_dim, hidden_dim, hidden_layers
        )

    def forward(self, observation, action):
        return self.q1(observation, action), self.q2(observation, action)


class TD3:
    """Twin Delayed DDPG with clipped target-policy smoothing."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 6,
        config: TD3Config | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config or TD3Config()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        actor_args = (
            observation_dim,
            action_dim,
            self.config.hidden_dim,
            self.config.hidden_layers,
        )
        self.actor = Actor(*actor_args).to(self.device)
        self.actor_target = Actor(*actor_args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TwinCritic(*actor_args).to(self.device)
        self.critic_target = TwinCritic(*actor_args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        self.total_updates = 0

    def select_action(self, observation) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(
                torch.as_tensor(vector, device=self.device)
            ).cpu().numpy()[0]
        return action.astype(np.float64)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    source_parameter, alpha=tau
                )

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        observation, action, next_observation, reward, done = replay_buffer.sample(
            batch_size, self.device
        )
        self.total_updates += 1
        with torch.no_grad():
            noise = torch.randn_like(action) * self.config.policy_noise
            noise.clamp_(-self.config.noise_clip, self.config.noise_clip)
            next_action = (self.actor_target(next_observation) + noise).clamp(
                -1.0, 1.0
            )
            target_q1, target_q2 = self.critic_target(
                next_observation, next_action
            )
            target_q = reward + (1.0 - done) * self.config.gamma * torch.minimum(
                target_q1, target_q2
            )

        current_q1, current_q2 = self.critic(observation, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = math.nan
        actor_updated = self.total_updates % self.config.policy_freq == 0
        if actor_updated:
            actor_loss = -self.critic.q1(
                observation, self.actor(observation)
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            self._soft_update(
                self.actor, self.actor_target, self.config.tau
            )
            self._soft_update(
                self.critic, self.critic_target, self.config.tau
            )

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "actor_updated": actor_updated,
            "total_updates": self.total_updates,
        }

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_updates": self.total_updates,
                "config": asdict(self.config),
                "metadata": metadata or {},
            },
            output,
        )
