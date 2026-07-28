"""Config dataclasses for the state-only Diffusion Policy trainer.

Stage-1 curriculum = single part. `DataConfig.part` is a plain config field
(set via CLI/JSON), never hardcoded in code — the same `train.py` runs all
9 parts by varying this one field.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Tuple

from constants import ACTION_DIM, DATASET_DEFAULT_ROOT, NUM_PARTS, PART_ORDER, STATE_DIM

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    dataset_root: str = DATASET_DEFAULT_ROOT
    # Stage-1: a single part. Stage-2/3 curriculum reuses the exact same
    # dataset/model code with `parts` set to a group instead of one name —
    # nothing here is part-count-specific.
    part: str = "gear_20teeth"
    horizon: int = 16          # predicted action chunk length (T_pred)
    n_obs_steps: int = 1       # state-only single-timestep conditioning
    val_fraction: float = 0.1
    split_seed: int = 0        # must match the seed compute_norm_stats.py used

    def __post_init__(self) -> None:
        if self.part not in PART_ORDER:
            raise ValueError(f"unknown part {self.part!r}, expected one of {PART_ORDER}")
        if self.n_obs_steps != 1:
            raise NotImplementedError(
                "StateEncoder currently consumes a single timestep; "
                "n_obs_steps > 1 needs a history-stacking change in dataset.py/model.py first."
            )


@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    num_parts: int = NUM_PARTS
    task_emb_dim: int = 32
    state_hidden_dim: int = 128
    state_feature_dim: int = 128
    # "rotvec" (default, matches the raw action encoding) or "rot6d"
    # (Zhou et al. 2019 continuous repr; reserved for later, see
    # rotation_utils.py — dataset.py and this config both branch on it but
    # only "rotvec" has been exercised end-to-end).
    rotation_repr: str = "rotvec"
    diffusion_step_embed_dim: int = 128
    #need to change back to the previous version later
    #down_dims: Tuple[int, ...] = (256, 512, 1024)
    down_dims: Tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8

    @property
    def action_dim(self) -> int:
        if self.rotation_repr == "rotvec":
            return ACTION_DIM  # xyz(3) + rotvec(3) + gripper(1) = 7
        if self.rotation_repr == "rot6d":
            return ACTION_DIM + 3  # xyz(3) + rot6d(6) + gripper(1) = 10
        raise ValueError(f"unknown rotation_repr {self.rotation_repr!r}")

    def __post_init__(self) -> None:
        if self.rotation_repr not in ("rotvec", "rot6d"):
            raise ValueError(f"unknown rotation_repr {self.rotation_repr!r}")


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    num_inference_steps: int = 16


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_every: int = 1
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 0
    log_every_steps: int = 50
    val_every_epochs: int = 5
    ckpt_every_epochs: int = 20
    ckpt_dir: str = "training/diffusion_policy/outputs"
    norm_stats_path: str = "training/diffusion_policy/norm_stats.json"
    right_arm_constants_path: str = "training/diffusion_policy/right_arm_constants.json"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**d.get("data", {})),
            #model=ModelConfig(**{**d.get("model", {}), "down_dims": tuple(d.get("model", {}).get("down_dims", (256, 512, 1024)))}),
            model=ModelConfig(**{**d.get("model", {}), "down_dims": tuple(d.get("model", {}).get("down_dims", (128, 256, 512)))}),
            diffusion=DiffusionConfig(**d.get("diffusion", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def resolve_repo_path(p: str | Path) -> Path:
    """Resolve a path that may be given relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)
