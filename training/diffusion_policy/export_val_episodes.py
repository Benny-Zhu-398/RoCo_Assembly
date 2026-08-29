"""Export a checkpoint's actual val-split episodes' recorded left-arm GT
actions to a lightweight .npz, for GT-action-replay closed-loop rollouts
(task/policies/gt_replay.py).

Why not read the parquet directly from Isaac's process: data_io.py needs
pandas + pyarrow, which aren't part of Isaac's numpy<2-pinned env (see
training/diffusion_policy/README.md's "not part of the repo's Isaac Sim uv
project" note) and installing them there risks the same interpreter clash
diffusion_lerobot.py's docstring warns about for torch. This script runs
once, offline, in the training venv, and writes a plain-numpy .npz that
task/policies/gt_replay.py loads with zero extra deps.

Val split is recomputed identically to what the checkpoint was actually
evaluated against -- same (dataset_root, part, val_fraction, split_seed) read
back out of the checkpoint's own config, exactly like evaluate.py does, NOT
re-guessed -- so "replay val split episodes" in the debugging plan means the
SAME episodes evaluate.py / error_locality_analysis.py / speed_error_analysis.py
already scored, not an arbitrary resample.

Only the recorded action's left-arm slice (ACTION_XYZ_SLICE + ACTION_ROT_SLICE
+ ACTION_GRIPPER_IDX = 7-D, raw/native units, NOT normalized) is exported.
The right-arm action slice is dropped: run_pick_place.py's harness drives R
from its own held init pose every step and ignores whatever a Policy returns
for R dofs (see policy_api.Policy.act's docstring), so recorded right-arm
actions are never actually executed and have no bearing on a replay's outcome.

Usage:
    python export_val_episodes.py --part battery_size1 --ckpt outputs/battery_size1/final.pt \
        --out val_episodes_battery_size1.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import resolve_repo_path  # noqa: E402
from constants import LEFT_ACTION_IDX, PART_TO_IDX  # noqa: E402
from data_io import episode_frames, episode_ids_for_part, load_data_table, load_episodes_table, split_episodes  # noqa: E402
from inference_utils import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint whose config.data (dataset_root, "
                     "val_fraction, split_seed) determines which episodes are 'val'")
    ap.add_argument("--out", type=str, default=None, help="output .npz path (default: val_episodes_<part>.npz "
                     "next to this script)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.ckpt)
    if ckpt["part"] != args.part and args.part not in (ckpt.get("parts") or []):
        raise ValueError(f"checkpoint {args.ckpt} was trained for part={ckpt['part']!r} / "
                          f"parts={ckpt.get('parts')!r}, not --part {args.part!r}")
    cfg_dict = ckpt["config"]["data"]
    dataset_root = resolve_repo_path(cfg_dict["dataset_root"])
    val_fraction = cfg_dict["val_fraction"]
    split_seed = cfg_dict["split_seed"]

    episodes_df = load_episodes_table(dataset_root)
    all_ids = episode_ids_for_part(episodes_df, args.part)
    _train_ids, val_ids = split_episodes(all_ids, val_fraction, split_seed)
    if not val_ids:
        raise RuntimeError(f"val split for part={args.part!r} is empty (val_fraction={val_fraction})")

    data_df = load_data_table(dataset_root, columns=["observation.state", "action", "task_index"])

    out = {"part": args.part, "episode_ids": np.asarray(val_ids, dtype=np.int64),
           "dataset_root": str(dataset_root), "val_fraction": val_fraction, "split_seed": split_seed}
    for ep in val_ids:
        _state, action = episode_frames(data_df, ep)
        left_action = action[:, LEFT_ACTION_IDX].astype(np.float32)  # (L, 7) raw units
        out[f"ep_{ep}"] = left_action

    out_path = Path(args.out) if args.out else (_THIS_DIR / f"val_episodes_{args.part}.npz")
    np.savez(out_path, **out)
    lengths = [len(out[f"ep_{ep}"]) for ep in val_ids]
    print(f"[export_val_episodes] part={args.part}  n_val_episodes={len(val_ids)}  "
          f"episode_ids={val_ids}")
    print(f"[export_val_episodes] episode lengths: min={min(lengths)} max={max(lengths)} "
          f"mean={np.mean(lengths):.1f}")
    print(f"[export_val_episodes] wrote {out_path}")


if __name__ == "__main__":
    main()
