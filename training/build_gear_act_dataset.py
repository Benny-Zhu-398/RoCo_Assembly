"""Merge the raw per-episode gear_20teeth captures into one training-ready
LeRobotDataset for a single-part, single-arm ACT policy.

Why this exists
----------------
`dataset/lerobot_v4_gear_20teeth/` holds 50 *independent* one-episode
LeRobot v3 dataset folders (30 `gear_20teeth_clean_*` + 20
`gear_20teeth_disturb_*`), each with its own meta/data/videos tree. Their
columns are also split per-arm (`observations.left_arm_ee_pose`,
`actions.right_arm_action`, ...) rather than the flat `observation.state`
/ `action` keys lerobot's ACT policy expects.

The previous training run (`training/train.json`, `outputs/train/act_stage1`)
trained on the full 9-part HF dataset with a 44-dim two-arm state and
14-dim two-arm action, even though the harness only ever executes the L
arm (R is held fixed at its init pose and isn't even observable at
runtime -- see `policy_api.Observation`, which has no `ee_pose_R`). That
mismatch forced `policies/act_eval.py` to fabricate a right-arm EE pose
via a hardcoded identity placeholder (`R_EE_HOME_POSE`), biasing every
prediction. Training one policy per part, on left-arm-only features that
exactly match what `Observation` exposes at runtime, removes that whole
class of bug.

What this script does
----------------------
For every source episode folder it:
  1. reads `data/chunk-*/file-*.parquet` for the left-arm state columns
     and the `actions.left_arm_action` label,
  2. decodes the selected camera(s)' `videos/<key>/chunk-*/file-*.mp4`,
  3. repacks each frame as `{observation.state (22,), observation.images.*,
     action (7,)}` and appends it to a single merged LeRobotDataset via
     the same `LeRobotDataset.create()` / `add_frame()` / `save_episode()`
     API `task/collect_lerobot_v3.py` uses to write datasets in the first
     place.

`clean` and `disturb` episodes are merged together on purpose: per
`taskboard_lerobot_summary.json`, disturb episodes perturb the *executed*
command but keep the *stored* `actions.left_arm_action` label as the
undisturbed expert target ("optional EE disturbances perturb executed
commands but not stored expert labels"). So a disturb episode's frames
teach the policy "given a state that's been knocked off-track, here is
the corrective action back toward the goal" -- free recovery/robustness
data, not just noisier demonstrations. Leaving them out throws that away.

Usage
-----
Run in a plain Python env with `lerobot==0.4.4`, `torch`, `opencv-python`,
`pandas`, and `pyarrow` installed (no Isaac Sim needed -- this only reads
already-recorded parquet/mp4 files):

    python training/build_gear_act_dataset.py \
        --source-root dataset/lerobot_v4_gear_20teeth \
        --output-root training/datasets/gear_20teeth_left_arm \
        --repo-id taskboard/gear_20teeth_left_arm

Add `--no-include-disturb` to train on the clean demos only, or
`--max-episodes N` to smoke-test the pipeline on a handful of episodes
before committing to the full merge.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK_DESCRIPTION = (
    "Pick up the 20-tooth gear and place it onto the rack post; recover "
    "toward the target when the end effector has been disturbed."
)

# short camera name -> (source video subdir key, destination feature key)
CAMERA_MAP = {
    "head": ("observations.rgb_head", "observation.images.head"),
    "left_hand": ("observations.rgb_left_hand", "observation.images.left_hand"),
    "right_hand": ("observations.rgb_right_hand", "observation.images.right_hand"),
}

# (source parquet column, dim) concatenated in this order to build
# `observation.state`. All are left-arm-only so they line up 1:1 with
# `policy_api.Observation`'s `ee_pose_L` / `joint_positions` / `joint_velocities`
# / `L_gripper_position` fields at eval time.
STATE_COLUMNS = [
    ("observations.left_arm_ee_pose", 7),
    ("observations.left_arm_joint_position", 7),
    ("observations.left_arm_joint_velocity", 7),
    ("observations.left_gripper_position", 1),
]
STATE_NAMES = (
    ["left_ee_x", "left_ee_y", "left_ee_z", "left_ee_qw", "left_ee_qx", "left_ee_qy", "left_ee_qz"]
    + [f"left_joint_pos_{i}" for i in range(7)]
    + [f"left_joint_vel_{i}" for i in range(7)]
    + ["left_gripper"]
)
STATE_DIM = sum(dim for _, dim in STATE_COLUMNS)  # 22

ACTION_COLUMN = "actions.left_arm_action"
ACTION_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
ACTION_DIM = 7


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=Path("dataset/lerobot_v4_gear_20teeth"))
    parser.add_argument("--output-root", type=Path, default=Path("training/datasets/gear_20teeth_left_arm"))
    parser.add_argument("--repo-id", type=str, default="taskboard/gear_20teeth_left_arm")
    parser.add_argument("--part-name", type=str, default="gear_20teeth")
    parser.add_argument(
        "--include-disturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the *_disturb_* episodes alongside *_clean_*. On by default -- see the "
        "module docstring for why disturb episodes are valuable recovery data, not noise to avoid.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=sorted(CAMERA_MAP),
        default=["head", "left_hand"],
        help="Which camera views to include as observation.images.<name>.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only process the first N discovered episodes (smoke-testing).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _discover_episode_dirs(source_root: Path, part_name: str, include_disturb: bool) -> List[Path]:
    # `source_root` also holds loose companion files (e.g.
    # `gear_20teeth_clean_000_label.json`, `..._results.json`) whose names
    # start with the same prefix as the episode directories -- filter to
    # directories only so those aren't picked up as episodes.
    clean = sorted(p for p in source_root.glob(f"{part_name}_clean_*") if p.is_dir())
    disturb = (
        sorted(p for p in source_root.glob(f"{part_name}_disturb_*") if p.is_dir())
        if include_disturb
        else []
    )
    episodes = clean + disturb
    if not episodes:
        raise FileNotFoundError(
            f"No episode directories matching '{part_name}_clean_*' or "
            f"'{part_name}_disturb_*' under {source_root}"
        )
    return episodes


def _read_state_table(episode_dir: Path) -> pd.DataFrame:
    files = sorted(episode_dir.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {episode_dir}/data/")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _read_video_frames(episode_dir: Path, source_key: str) -> List[np.ndarray]:
    files = sorted(episode_dir.glob(f"videos/{source_key}/chunk-*/file-*.mp4"))
    if not files:
        raise FileNotFoundError(f"No video files under {episode_dir}/videos/{source_key}/")
    out: List[np.ndarray] = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    return out


def _row_vec(row: pd.Series, column: str, dim: int) -> np.ndarray:
    arr = np.asarray(row[column], dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"column {column!r} expected dim {dim}, got shape {arr.shape}")
    return arr


def _episode_fps(episode_dir: Path) -> int:
    info = json.loads((episode_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    return int(info["fps"])


def _build_features(cameras: List[str]) -> Dict[str, dict]:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
    }
    for cam in cameras:
        _, dest_key = CAMERA_MAP[cam]
        features[dest_key] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        }
    return features


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists (pass --overwrite to replace it)")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    episode_dirs = _discover_episode_dirs(source_root, args.part_name, args.include_disturb)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    fps = _episode_fps(episode_dirs[0])
    print(f"[build] {len(episode_dirs)} source episodes, fps={fps}, cameras={args.cameras}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=fps,
        robot_type="vega_1u_taskboard_left_arm",
        features=_build_features(args.cameras),
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        vcodec="h264",
    )

    total_frames = 0
    for ep_idx, ep_dir in enumerate(episode_dirs):
        table = _read_state_table(ep_dir)
        cam_frames = {cam: _read_video_frames(ep_dir, CAMERA_MAP[cam][0]) for cam in args.cameras}

        n_rows = len(table)
        for cam, frames in cam_frames.items():
            if len(frames) != n_rows:
                raise ValueError(
                    f"{ep_dir.name}: {cam} has {len(frames)} decoded frames but the state "
                    f"table has {n_rows} rows"
                )

        for i in range(n_rows):
            row = table.iloc[i]
            state = np.concatenate(
                [_row_vec(row, col, dim) for col, dim in STATE_COLUMNS]
            ).astype(np.float32)
            action = _row_vec(row, ACTION_COLUMN, ACTION_DIM)

            frame = {
                "observation.state": state,
                "action": action,
                "task": TASK_DESCRIPTION,
            }
            for cam in args.cameras:
                _, dest_key = CAMERA_MAP[cam]
                frame[dest_key] = cam_frames[cam][i]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += n_rows
        print(f"[build] ({ep_idx + 1}/{len(episode_dirs)}) {ep_dir.name}: {n_rows} frames")

    dataset.finalize()
    print(f"[build] done: {len(episode_dirs)} episodes, {total_frames} frames -> {output_root}")


if __name__ == "__main__":
    main()
