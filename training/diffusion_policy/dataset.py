"""PyTorch Dataset over tools/roco2026_by_part, left-arm-only.

Built to take a *list* of parts (defaulting to a single part for the
stage-1 curriculum) so stage-2/3 ("one group" / "three groups") reuse this
class unchanged -- only the `parts` argument grows.

State/action are always loaded into memory up front (~35 MB of parquet
total). Images are OPTIONAL (`load_images`, default True) and, when
enabled, are ALSO fully decoded into memory per used episode at __init__
time -- same "everything preloaded, __getitem__ is pure indexing"
philosophy, just a much bigger memory bill: at native 240x320 resolution, a
single part's ~11-14.5k frames x 2 cameras x 3 bytes is on the order of
several GB of RAM (measured, see vision.py's CAMERA_KEYS for which two
cameras). Pass `image_resize_hw` to decode smaller and cut that down, or
`load_images=False` to keep the original state-only behavior with zero
video I/O (dataset.py never touched images before this -- that path still
works unchanged for anyone still training state-only).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import (  # noqa: E402
    ACTION_ROT_SLICE,
    CAMERA_KEYS,
    LEFT_ACTION_IDX,
    LEFT_STATE_IDX,
    PART_TO_IDX,
)
from data_io import (  # noqa: E402
    episode_ids_for_part,
    load_data_table,
    load_episodes_table,
    read_episode_frames,
    resolve_video_path,
    split_episodes,
)
from normalization import NormStats, normalize_action, normalize_state  # noqa: E402
from rotation_utils import find_rotvec_jumps, rotvec_to_rot6d  # noqa: E402


class PartSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        parts: Union[str, Sequence[str]],
        split: str,
        horizon: int,
        norm_stats: NormStats,
        val_fraction: float = 0.1,
        split_seed: int = 0,
        rotation_repr: str = "rotvec",
        load_images: bool = True,
        camera_keys: Sequence[str] = CAMERA_KEYS,
        image_resize_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if isinstance(parts, str):
            parts = [parts]
        self.dataset_root = Path(dataset_root)
        self.parts = list(parts)
        self.split = split
        self.horizon = horizon
        self.norm_stats = norm_stats
        self.rotation_repr = rotation_repr
        self.load_images = load_images
        self.camera_keys = list(camera_keys)
        self.image_resize_hw = image_resize_hw

        episodes = load_episodes_table(self.dataset_root, camera_keys=self.camera_keys if load_images else ())
        data = load_data_table(self.dataset_root, columns=["observation.state", "action"])
        ep2task = dict(zip(episodes["episode_index"], episodes["task"]))
        data["task"] = data["episode_index"].map(ep2task)

        self._episode_state: Dict[Tuple[str, int], np.ndarray] = {}
        self._episode_action: Dict[Tuple[str, int], np.ndarray] = {}
        # {(part, episode_index): {camera_key: (L,H,W,3) uint8}} -- only
        # populated when load_images=True.
        self._episode_images: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self.samples: List[Tuple[str, int, int]] = []  # (part, episode_index, start_frame)
        self.rotvec_jump_warnings: List[dict] = []
        self.episode_ids_used: Dict[str, List[int]] = {}

        for part in self.parts:
            all_ids = episode_ids_for_part(episodes, part)
            train_ids, val_ids = split_episodes(all_ids, val_fraction, split_seed)
            use_ids = train_ids if split == "train" else val_ids
            self.episode_ids_used[part] = use_ids

            for ep in use_ids:
                sub = data.loc[data["episode_index"] == ep].sort_values("frame_index")
                state_full = np.stack(sub["observation.state"].to_numpy())
                action_full = np.stack(sub["action"].to_numpy())
                state = state_full[:, LEFT_STATE_IDX].astype(np.float32)
                action = action_full[:, LEFT_ACTION_IDX].astype(np.float32)

                # NOTE: this dataset's action rotation dims (ACTION_ROT_SLICE) are
                # Euler XYZ extrinsic, NOT rotvec -- see constants.py's
                # ACTION_ROT_SLICE comment and rotation_convention_audit.py.
                # find_rotvec_jumps is still a valid generic ">pi raw-component-diff"
                # discontinuity check regardless of that (it doesn't require rotvec
                # semantics), but read every "rotvec jump" in this warning as "jump in
                # the 3 raw rotation numbers", not literally rotvec.
                jump_frames = find_rotvec_jumps(action[:, ACTION_ROT_SLICE])
                for t in jump_frames:
                    record = {"part": part, "episode_index": int(ep), "frame": int(t)}
                    self.rotvec_jump_warnings.append(record)
                    warnings.warn(
                        f"[PartSequenceDataset] rotvec jump > pi within episode: "
                        f"part={part} episode={ep} frame={t} "
                        f"(||rotvec[t+1]-rotvec[t]||>pi) -- kept as-is, see rotation_utils.find_rotvec_jumps",
                        RuntimeWarning,
                    )

                if rotation_repr == "rot6d":
                    # WRONG for this dataset as written: rotvec_to_rot6d decodes
                    # ACTION_ROT_SLICE as rotvec via Rotation.from_rotvec, but it's
                    # actually Euler XYZ extrinsic (see rotation_utils.py's module
                    # docstring and rotation_convention_audit.py). Never exercised
                    # end-to-end (config.py: only rotation_repr="rotvec" has been
                    # trained) -- fix the conversion to build the rotation matrix via
                    # Euler-XYZ-extrinsic before ever training with rotation_repr="rot6d".
                    rot6d = rotvec_to_rot6d(action[:, ACTION_ROT_SLICE])
                    action = np.concatenate(
                        [action[:, :3], rot6d, action[:, 6:7]], axis=-1
                    ).astype(np.float32)

                key = (part, int(ep))
                self._episode_state[key] = state
                self._episode_action[key] = action

                if self.load_images:
                    row = episodes.loc[episodes["episode_index"] == ep].iloc[0]
                    self._episode_images[key] = {}
                    for cam in self.camera_keys:
                        prefix = f"videos/observation.images.{cam}"
                        vpath = resolve_video_path(
                            self.dataset_root, cam,
                            int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]),
                        )
                        frames = read_episode_frames(
                            vpath,
                            from_timestamp=float(row[f"{prefix}/from_timestamp"]),
                            to_timestamp=float(row[f"{prefix}/to_timestamp"]),
                            n_frames=len(state),
                            resize_hw=self.image_resize_hw,
                        )
                        self._episode_images[key][cam] = frames

                for t in range(len(state)):
                    self.samples.append((part, int(ep), t))

        if self.rotvec_jump_warnings:
            warnings.warn(
                f"[PartSequenceDataset] {len(self.rotvec_jump_warnings)} rotvec jump(s) "
                f"found across {split} split of parts={self.parts}; see .rotvec_jump_warnings",
                RuntimeWarning,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        part, ep, t = self.samples[idx]
        key = (part, ep)
        state = self._episode_state[key][t]           # (22,)
        action_seq = self._episode_action[key]         # (L, action_dim)
        L = len(action_seq)

        end = min(t + self.horizon, L)
        chunk = action_seq[t:end]
        n_valid = len(chunk)
        if n_valid < self.horizon:
            pad = np.repeat(chunk[-1:], self.horizon - n_valid, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        action_is_pad = np.zeros(self.horizon, dtype=bool)
        action_is_pad[n_valid:] = True

        state_n = normalize_state(state, self.norm_stats)
        action_n = normalize_action(chunk, self.norm_stats, part)

        sample = {
            "state": torch.from_numpy(state_n.astype(np.float32)),
            "action": torch.from_numpy(action_n.astype(np.float32)),
            "action_is_pad": torch.from_numpy(action_is_pad),
            "task_idx": torch.tensor(PART_TO_IDX[part], dtype=torch.long),
        }
        if self.load_images:
            # Single current-observation frame per camera (n_obs_steps=1,
            # same as state -- see config.py::DataConfig.n_obs_steps), HWC
            # uint8 -> CHW float32 in [0, 1].
            for cam in self.camera_keys:
                img = self._episode_images[key][cam][t]  # (H,W,3) uint8
                sample[f"image_{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return sample
