"""Low-level access to the tools/roco2026_by_part LeRobot v3.0 dataset.

Loads the parquet files directly (no `lerobot` dependency needed for
training-data access — only the sidecar eval side of this repo needs the
full lerobot package). Mirrors the chunked-glob pattern already used by
tools/segment_by_part.py::load_action_table so both tools agree on how to
read this dataset layout.

Video frame access (`read_episode_frames` etc.) uses PyAV directly, not
lerobot's own video-loading code -- same "read the metadata format, write
our own small reader" choice as the rest of this package (see vision.py's
module docstring for why). Confirmed from meta/episodes/*.parquet
(2026-08-13) that this dataset's video<->frame mapping is simple: video fps
(10, info.json) equals dataset fps exactly (no resampling needed), and
every episode's frames live entirely within ONE video file per camera
(chunk_index/file_index never split mid-episode) -- so decoding one episode
is "seek near from_timestamp, decode forward, collect length frames", never
a multi-file stitch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import PART_ORDER  # noqa: E402

DATASET_FPS = 10.0
VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def load_episodes_table(dataset_root: Path, camera_keys: Sequence[str] = ()) -> pd.DataFrame:
    """Return one row per episode: episode_index, task (part name), length,
    plus (if `camera_keys` given) each camera's video chunk_index/file_index/
    from_timestamp/to_timestamp -- everything `read_episode_frames` needs to
    locate and decode that episode's footage."""
    files = sorted(Path(dataset_root).glob("meta/episodes/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes parquet under {dataset_root}/meta/episodes")
    base_cols = ["episode_index", "tasks", "length"]
    video_cols = []
    for cam in camera_keys:
        prefix = f"videos/observation.images.{cam}"
        video_cols += [f"{prefix}/chunk_index", f"{prefix}/file_index",
                       f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"]
    columns = base_cols + video_cols
    dfs = [pq.read_table(f, columns=columns).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["task"] = df["tasks"].apply(lambda x: x[0])
    return df.drop(columns=["tasks"]).sort_values("episode_index").reset_index(drop=True)


def resolve_video_path(dataset_root: Path, camera_key: str, chunk_index: int, file_index: int) -> Path:
    video_key = f"observation.images.{camera_key}"
    rel = VIDEO_PATH_TEMPLATE.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
    return Path(dataset_root) / rel


def read_episode_frames(
    video_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    n_frames: int,
    fps: float = DATASET_FPS,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode exactly `n_frames` consecutive frames (this episode's full
    length) starting at `from_timestamp` seconds into `video_path`.

    Seeks near `from_timestamp` (lands on the nearest preceding keyframe,
    not necessarily exact) then decodes forward, matching each decoded
    frame's presentation timestamp against the expected per-frame time grid
    (`from_timestamp + i/fps`) with a half-frame-interval tolerance to
    absorb float/PTS rounding -- so this is robust to the seek not landing
    exactly on frame 0, but still assumes video fps == `fps` (true for this
    dataset, see module docstring) so there's no frame to skip/repeat
    between grid points.
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        half_dt = 0.5 / fps
        target_times = [from_timestamp + i / fps for i in range(n_frames)]

        seek_pts = int(max(from_timestamp - half_dt, 0.0) / stream.time_base)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        frames: List[np.ndarray] = []
        ti = 0
        for frame in container.decode(stream):
            if ti >= n_frames:
                break
            t = float(frame.pts * stream.time_base)
            if t < target_times[ti] - half_dt:
                continue
            img = frame.to_ndarray(format="rgb24")
            if resize_hw is not None and (img.shape[0], img.shape[1]) != tuple(resize_hw):
                img = _resize_rgb(img, resize_hw)
            frames.append(img)
            ti += 1
            # advance ti past any further target times this same decoded
            # frame also satisfies (shouldn't normally happen at fps==fps,
            # but keeps this robust rather than silently misaligning).
            while ti < n_frames and t >= target_times[ti] - half_dt:
                frames.append(img)
                ti += 1
    finally:
        container.close()

    if len(frames) != n_frames:
        raise RuntimeError(
            f"decoded {len(frames)} frames, expected {n_frames} from {video_path} "
            f"[{from_timestamp}, {to_timestamp})"
        )
    return np.stack(frames)  # (n_frames, H, W, 3) uint8


def _resize_rgb(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    import torch
    import torchvision.transforms.functional as TF

    t = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)
    t = TF.resize(t, list(hw), antialias=True)
    return t.permute(1, 2, 0).numpy()


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
