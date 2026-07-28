"""Shared normalization stats: load/save norm_stats.json, apply/unapply.

Statistics are computed ONCE across all 9 parts' training splits by
compute_norm_stats.py and reused by every single-part run, so a later
single-part-vs-multi-part ablation never has the confound of different
normalizers. See compute_norm_stats.py for how the file is produced.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import ACTION_DIM, ACTION_NAMES, STATE_DIM, STATE_NAMES  # noqa: E402

_STD_FLOOR = 1e-6


@dataclass
class NormStats:
    state_mean: np.ndarray  # (STATE_DIM,)
    state_std: np.ndarray   # (STATE_DIM,)
    action_min: np.ndarray  # (ACTION_DIM,) from q0.01 over pooled train actions
    action_max: np.ndarray  # (ACTION_DIM,) from q0.99 over pooled train actions
    meta: dict

    def __post_init__(self) -> None:
        assert self.state_mean.shape == (STATE_DIM,)
        assert self.state_std.shape == (STATE_DIM,)
        # action dim is ACTION_DIM (7) for rotation_repr="rotvec" but 10 for
        # the reserved "rot6d" branch (xyz 3 + rot6d 6 + gripper 1) — only
        # check the two are mutually consistent, not a fixed constant.
        assert self.action_min.shape == self.action_max.shape
        assert self.action_min.ndim == 1

    def to_dict(self) -> dict:
        n_action = self.action_min.shape[0]
        action_names = list(ACTION_NAMES) if n_action == len(ACTION_NAMES) else [
            f"action_{i}" for i in range(n_action)
        ]
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "state_names": list(STATE_NAMES),
            "action_names": action_names,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            state_mean=np.asarray(d["state_mean"], dtype=np.float32),
            state_std=np.asarray(d["state_std"], dtype=np.float32),
            action_min=np.asarray(d["action_min"], dtype=np.float32),
            action_max=np.asarray(d["action_max"], dtype=np.float32),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def compute(
        cls,
        state_frames: np.ndarray,
        action_frames: np.ndarray,
        quantile_lo: float = 0.01,
        quantile_hi: float = 0.99,
        meta: Optional[dict] = None,
    ) -> "NormStats":
        state_frames = np.asarray(state_frames, dtype=np.float64)
        action_frames = np.asarray(action_frames, dtype=np.float64)
        state_mean = state_frames.mean(axis=0)
        state_std = state_frames.std(axis=0)
        state_std = np.maximum(state_std, _STD_FLOOR)
        # Quantile clip (not raw min/max) so the 0.9% of rotvec samples with
        # magnitude >= 2*pi don't blow the action range out for everyone else.
        action_min = np.quantile(action_frames, quantile_lo, axis=0)
        action_max = np.quantile(action_frames, quantile_hi, axis=0)
        # Degenerate dims (e.g. would-be-constant channels) get a minimum
        # span so normalize() never divides by ~0.
        span = action_max - action_min
        degenerate = span < _STD_FLOOR
        action_min = np.where(degenerate, action_min - 0.5, action_min)
        action_max = np.where(degenerate, action_max + 0.5, action_max)
        return cls(
            state_mean=state_mean.astype(np.float32),
            state_std=state_std.astype(np.float32),
            action_min=action_min.astype(np.float32),
            action_max=action_max.astype(np.float32),
            meta=meta or {},
        )


def normalize_state(state: np.ndarray, stats: NormStats) -> np.ndarray:
    return (state - stats.state_mean) / stats.state_std


def unnormalize_state(state_n: np.ndarray, stats: NormStats) -> np.ndarray:
    return state_n * stats.state_std + stats.state_mean


def normalize_action(action: np.ndarray, stats: NormStats) -> np.ndarray:
    span = stats.action_max - stats.action_min
    x = 2.0 * (action - stats.action_min) / span - 1.0
    return np.clip(x, -1.0, 1.0)


def unnormalize_action(action_n: np.ndarray, stats: NormStats) -> np.ndarray:
    span = stats.action_max - stats.action_min
    return (action_n + 1.0) / 2.0 * span + stats.action_min
