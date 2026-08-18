"""Config dataclasses for the state-only Diffusion Policy trainer.

Stage-1 curriculum = single part. `DataConfig.part` is a plain config field
(set via CLI/JSON), never hardcoded in code — the same `train.py` runs all
9 parts by varying this one field.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

from constants import (
    ACTION_DIM, CAMERA_KEYS, DATASET_DEFAULT_ROOT, GROUP_ORDER, NUM_PARTS,
    PART_ORDER, PART_TO_GROUP, STATE_DIM,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    dataset_root: str = DATASET_DEFAULT_ROOT
    # Stage-1: a single part (used when group is None, the default).
    part: str = "gear_20teeth"
    # Stage-2/3: set to one of constants.GROUP_ORDER (train only that
    # group's parts -- e.g. for isolated debugging of one skill head) or the
    # literal string "all" (every part across every group, the real joint
    # run grouped_model.GroupedDiffusionPolicyNet is meant for -- one shared
    # trunk amortized over all 4 groups, each sample routed to its own
    # skill head). None (default) = ignore this field, use `part` alone via
    # the original single-part model.py::DiffusionPolicyNet path; train.py
    # branches its model class on group is None vs not, not on a separate
    # flag, so there is exactly one source of truth for which path runs.
    group: Optional[str] = None
    horizon: int = 16          # predicted action chunk length (T_pred)
    n_obs_steps: int = 1       # state-only single-timestep conditioning
    val_fraction: float = 0.1
    split_seed: int = 0        # must match the seed compute_norm_stats.py used
    # Decode-time resize for images (H, W), only used when
    # model.use_vision=True (see ModelConfig). None = keep native 240x320
    # (several GB of RAM per part at that resolution -- see dataset.py's
    # module docstring); shrink this if that's too much for the box.
    image_resize_hw: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.group is not None and self.group != "all" and self.group not in GROUP_ORDER:
            raise ValueError(f"unknown group {self.group!r}, expected 'all' or one of {GROUP_ORDER}")
        if self.group is None and self.part not in PART_ORDER:
            raise ValueError(f"unknown part {self.part!r}, expected one of {PART_ORDER}")
        if self.n_obs_steps != 1:
            raise NotImplementedError(
                "StateEncoder currently consumes a single timestep; "
                "n_obs_steps > 1 needs a history-stacking change in dataset.py/model.py first."
            )

    def resolved_parts(self) -> Tuple[str, ...]:
        """The actual list of parts to load, honoring group over part when
        group is set. Single source of truth train.py/dataset.py should call
        instead of branching on `group is None` themselves."""
        if self.group is None:
            return (self.part,)
        if self.group == "all":
            return PART_ORDER
        return tuple(p for p in PART_ORDER if PART_TO_GROUP[p] == self.group)


@dataclass
class ModelConfig:
    state_dim: int = STATE_DIM
    num_parts: int = NUM_PARTS
    task_emb_dim: int = 32
    state_hidden_dim: int = 128
    state_feature_dim: int = 128
    # "rotvec" (default) or "rot6d" (Zhou et al. 2019 continuous repr;
    # reserved for later, see rotation_utils.py -- dataset.py and this
    # config both branch on it but only "rotvec" has been exercised
    # end-to-end). NOTE: "rotvec" here is a misnomer for
    # tools/roco2026_by_part -- its action rotation dims are actually
    # Euler XYZ extrinsic (see constants.py's ACTION_ROT_SLICE comment and
    # rotation_convention_audit.py), not rotvec. This default path is
    # harmless regardless (dataset.py passes the 3 raw dims through
    # verbatim, no geometric conversion happens), but the "rot6d" branch
    # DOES geometrically convert via an explicit rotvec assumption
    # (rotation_utils.rotvec_to_rot6d) and is currently wrong for this
    # dataset if ever trained -- fix that conversion first.
    rotation_repr: str = "rotvec"
    diffusion_step_embed_dim: int = 128
    #need to change back to the previous version later
    #down_dims: Tuple[int, ...] = (256, 512, 1024)
    down_dims: Tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8

    # --- vision (optional; default False = original state-only behavior,
    # existing checkpoints/configs unaffected). See vision.py for the
    # encoder these fields configure and CAMERA_KEYS for which two cameras.
    # When True, dataset.py must be built with load_images=True (train.py
    # wires this automatically off this same flag -- see build_dataloaders).
    use_vision: bool = False
    camera_keys: Tuple[str, ...] = CAMERA_KEYS
    vision_backbone: str = "resnet18"
    vision_pretrained: bool = True
    vision_use_group_norm: bool = False  # must be False when vision_pretrained=True, see vision.py::RgbEncoder
    vision_num_keypoints: int = 32
    vision_crop_hw: Optional[Tuple[int, int]] = None
    vision_crop_is_random: bool = True

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
        data_d = dict(d.get("data", {}))
        if data_d.get("image_resize_hw") is not None:
            data_d["image_resize_hw"] = tuple(data_d["image_resize_hw"])

        model_d = dict(d.get("model", {}))
        model_d["down_dims"] = tuple(model_d.get("down_dims", (128, 256, 512)))
        model_d["camera_keys"] = tuple(model_d.get("camera_keys", CAMERA_KEYS))
        if model_d.get("vision_crop_hw") is not None:
            model_d["vision_crop_hw"] = tuple(model_d["vision_crop_hw"])

        return cls(
            data=DataConfig(**data_d),
            model=ModelConfig(**model_d),
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
