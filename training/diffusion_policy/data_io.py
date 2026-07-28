"""Low-level access to the tools/roco2026_by_part LeRobot v3.0 dataset.

Loads the parquet files directly (no `lerobot` dependency needed for
training-data access — only the sidecar eval side of this repo needs the
full lerobot package). Mirrors the chunked-glob pattern already used by
tools/segment_by_part.py::load_action_table so both tools agree on how to
read this dataset layout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import PART_ORDER  # noqa: E402


def load_episodes_table(dataset_root: Path) -> pd.DataFrame:
    """Return one row per episode: episode_index, task (part name), length."""
    files = sorted(Path(dataset_root).glob("meta/episodes/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes parquet under {dataset_root}/meta/episodes")
    dfs = [pq.read_table(f, columns=["episode_index", "tasks", "length"]).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["task"] = df["tasks"].apply(lambda x: x[0])
    return df.drop(columns=["tasks"]).sort_values("episode_index").reset_index(drop=True)


def load_data_table(
    dataset_root: Path,
    columns: Sequence[str] = ("observation.state", "action", "task_index"),
) -> pd.DataFrame:
    """Return the full per-frame table, sorted by (episode_index, frame_index).

    `episode_index` / `frame_index` are always included (needed to sort and
    to group into episodes) even if not listed in `columns`.
    """
    required = ["episode_index", "frame_index"]
    full_columns = required + [c for c in columns if c not in required]
    files = sorted(Path(dataset_root).glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {dataset_root}/data")
    dfs = [pq.read_table(f, columns=full_columns).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def episode_ids_for_part(episodes_df: pd.DataFrame, part: str) -> List[int]:
    ids = episodes_df.loc[episodes_df["task"] == part, "episode_index"].tolist()
    if not ids:
        raise ValueError(
            f"part {part!r} has 0 episodes in this dataset; known parts are {PART_ORDER}"
        )
    return sorted(int(i) for i in ids)


def split_episodes(
    episode_ids: Iterable[int], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Deterministic train/val split over episode ids (not frames — avoids
    leaking frames from the same trajectory across the split).

    Pure function of (episode_ids, val_fraction, seed) so
    compute_norm_stats.py and dataset.py always agree on which episodes are
    "train" without importing each other.
    """
    ids = sorted(int(i) for i in episode_ids)
    if val_fraction <= 0:
        return ids, []
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def episode_frames(
    data_df: pd.DataFrame, episode_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(state[L,44] float32, action[L,14] float32) for one episode, ordered by frame_index."""
    sub = data_df.loc[data_df["episode_index"] == episode_index]
    state = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(sub["action"].to_numpy()).astype(np.float32)
    return state, action
