"""
Segment the RoCo Industrial Assembly LeRobot v3.0 dataset by part.
Verified against lerobot 0.5.1 API (reader/writer architecture).

Each source episode is one full 9-part assembly. Part boundaries are detected
from the LEFT gripper command channel action[6] (open ratio in [0,1]:
open ~0.226, close 0.0 or ~0.060 for rod_16mm). Each part segment becomes its
own episode in a new dataset, with `task` set to the part name.

Usage (run in the `roco2` conda env — Isaac Sim NOT needed):
  python segment_by_part.py --inspect   # fast, parquet only, prints tables
  python segment_by_part.py --write     # slow, decodes/re-encodes video
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SRC_REPO_ID = "rocochallenge2025/rocochallenge2026_Industrial_Assembly"
DST_REPO_ID = "Benny-Zhu-398/roco2026_industrial_by_part"

# All outputs are anchored to this script's own directory (tools/),
# regardless of where you run the command from.
SCRIPT_DIR = Path(__file__).resolve().parent
DST_ROOT = SCRIPT_DIR / "roco2026_by_part"
INDEX_PATH = SCRIPT_DIR / "segment_index.json"


def resolve_src_root() -> Path:
    """Find the local copy of the source dataset in the HF/LeRobot cache."""
    candidates = []
    if os.environ.get("HF_LEROBOT_HOME"):
        candidates.append(Path(os.environ["HF_LEROBOT_HOME"]) / SRC_REPO_ID)
    home = Path.home()
    candidates += [
        home / ".cache" / "huggingface" / "lerobot" / SRC_REPO_ID,
        home / ".cache" / "huggingface" / "lerobot" / SRC_REPO_ID.replace("/", os.sep),
    ]
    for c in candidates:
        if (c / "meta" / "info.json").exists():
            return c
    raise FileNotFoundError(
        "Could not find the source dataset locally. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nEither download it first (LeRobotDataset(SRC_REPO_ID)) or set "
        "HF_LEROBOT_HOME to your cache directory."
    )


PART_ORDER = (
    "gear_20teeth",
    "gear_60teeth",
    "rod_16mm",
    "bolt_8mm",
    "usb_a",
    "hdmi",
    "pin",
    "battery_size1",
    "battery_size5",
)

# action[6] = left gripper command, normalized open ratio [0,1].
# open = 0.15 rad / 0.66497 ~= 0.226; close = 0 (or 0.04 rad ~= 0.060 for rod).
GRIPPER_ACTION_INDEX = 6
GRIPPER_THRESHOLD = 0.12   # midpoint with margin between 0.060 and 0.226
MIN_SEGMENT_FRAMES = 10

BOOKKEEPING_KEYS = ("index", "episode_index", "frame_index",
                    "timestamp", "task_index", "task")


# ----------------------------------------------------------------------------
# Step 1: boundary detection from parquet (fast, no video decode)
# ----------------------------------------------------------------------------
def load_action_table(src_root: Path):
    import pandas as pd

    parquet_files = sorted(src_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {src_root}/data")

    dfs = [pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
           for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def detect_segments(gripper: np.ndarray):
    """Split at close->open transitions (release events)."""
    closed = gripper < GRIPPER_THRESHOLD
    releases = np.where(closed[:-1] & ~closed[1:])[0]
    segments, start = [], 0
    for r in releases:
        end = int(r) + 1
        segments.append((start, end))
        start = end + 1
    if segments and start <= len(gripper) - 1:
        s0, _ = segments[-1]
        segments[-1] = (s0, len(gripper) - 1)  # tail goes to the last part
    return segments


def inspect(src_root: Path):
    df = load_action_table(src_root)
    index, bad_episodes = {}, []

    # quick sanity print of the gripper channel value levels
    a0 = np.stack(df[df["episode_index"] == df["episode_index"].iloc[0]]["action"].to_numpy())
    g0 = a0[:, GRIPPER_ACTION_INDEX]
    print(f"Gripper channel sanity (episode 0): min={g0.min():.3f} "
          f"max={g0.max():.3f} unique_levels~{np.unique(g0.round(2))[:6]}")

    for ep_idx, g in df.groupby("episode_index"):
        actions = np.stack(g["action"].to_numpy())
        gripper = actions[:, GRIPPER_ACTION_INDEX]
        segments = detect_segments(gripper)

        ok = len(segments) == len(PART_ORDER) and all(
            (e - s + 1) >= MIN_SEGMENT_FRAMES for s, e in segments)
        if not ok:
            bad_episodes.append(int(ep_idx))
            print(f"[!!] episode {ep_idx}: {len(segments)} segments "
                  f"(expected {len(PART_ORDER)}) — will be skipped")
            continue

        index[int(ep_idx)] = [
            {"part": PART_ORDER[k], "start": int(s), "end": int(e),
             "n_frames": int(e - s + 1)}
            for k, (s, e) in enumerate(segments)
        ]

    print(f"\n{len(index)} episodes segmented cleanly, {len(bad_episodes)} flagged.")
    if index:
        ep0 = sorted(index)[0]
        print(f"\nExample — episode {ep0}:")
        for seg in index[ep0]:
            print(f"  {seg['part']:<16} frames {seg['start']:>4}-{seg['end']:>4} "
                  f"({seg['n_frames']} frames)")
        print("\nPer-part frame count (mean ± std across episodes):")
        for k, part in enumerate(PART_ORDER):
            counts = [index[e][k]["n_frames"] for e in index]
            print(f"  {part:<16} {np.mean(counts):6.1f} ± {np.std(counts):4.1f}")

    INDEX_PATH.write_text(json.dumps(
        {"index": index, "bad_episodes": bad_episodes}, indent=2))
    print(f"\nSaved boundaries to {INDEX_PATH}")


# ----------------------------------------------------------------------------
# Step 2: rewrite into a new dataset, one episode per part segment
# ----------------------------------------------------------------------------
def rewrite(src_root: Path, index_path: Path = INDEX_PATH):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    payload = json.loads(index_path.read_text())
    index = {int(k): v for k, v in payload["index"].items()}

    src = LeRobotDataset(SRC_REPO_ID, root=src_root)

    features = {k: v for k, v in src.meta.features.items()
                if k not in BOOKKEEPING_KEYS}

    dst = LeRobotDataset.create(
        repo_id=DST_REPO_ID,
        fps=src.fps,
        root=DST_ROOT,
        features=features,
        use_videos=True,
        vcodec="h264",           # faster than default AV1, matches source
    )

    try:
        for ep_idx in sorted(index):
            # lerobot 0.5.1: per-episode global frame offset lives in meta.episodes
            base = int(src.meta.episodes[ep_idx]["dataset_from_index"])
            for seg in index[ep_idx]:
                for local_i in range(seg["start"], seg["end"] + 1):
                    item = src[base + local_i]  # decodes video frames
                    frame = {k: item[k] for k in features}
                    frame["task"] = seg["part"]
                    dst.add_frame(frame)   # tensors/CHW-float handled by writer
                dst.save_episode()
                print(f"[write] src ep {ep_idx} / {seg['part']} "
                      f"-> dst ep {dst.num_episodes - 1} ({seg['n_frames']} frames)")
    finally:
        dst.finalize()  # REQUIRED in 0.5.1 — without this the dataset is invalid

    print(f"\nDone. {dst.num_episodes} episodes, {dst.num_frames} frames "
          f"at {DST_ROOT.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--src-root", type=str, default=None,
                    help="Override auto-detected source dataset path")
    args = ap.parse_args()

    src_root = Path(args.src_root) if args.src_root else resolve_src_root()
    print(f"Source dataset: {src_root}")

    if args.inspect:
        inspect(src_root)
    elif args.write:
        rewrite(src_root)
    else:
        ap.print_help()
