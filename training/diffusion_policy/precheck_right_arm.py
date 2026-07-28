"""Pre-flight check: is action[7:14] (right-arm target) actually constant
per part?

We only ever train on the left 7-D action slice; the right 7-D slice is
supposed to be filled back in at execution time from a stored per-part
constant. This script verifies that assumption against real data instead
of asserting it blind, and reports (does not silently "fix") any part
where it doesn't hold.

Usage:
    python precheck_right_arm.py [--threshold 1e-3]

Writes right_arm_constants.json next to this file:
    {part: {"constant": [7 floats], "std": [7 floats], "max_std": float,
            "all_below_threshold": bool}}
train.py reads this file and copies the entry for its configured part into
the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import REPO_ROOT, resolve_repo_path  # noqa: E402
from constants import PART_ORDER, RIGHT_ACTION_IDX, RIGHT_ACTION_NAMES  # noqa: E402
from data_io import load_data_table, load_episodes_table  # noqa: E402


def run(dataset_root: Path, threshold: float) -> dict:
    episodes = load_episodes_table(dataset_root)
    data = load_data_table(dataset_root, columns=["episode_index", "action"])
    ep2task = dict(zip(episodes["episode_index"], episodes["task"]))
    data["task"] = data["episode_index"].map(ep2task)

    action = np.stack(data["action"].to_numpy())  # (N, 14)
    right = action[:, RIGHT_ACTION_IDX]            # (N, 7)

    report: dict = {}
    print(f"{'part':<16} " + " ".join(f"{n:>12}" for n in RIGHT_ACTION_NAMES) + "   max_std   flag")
    for part in PART_ORDER:
        mask = (data["task"] == part).to_numpy()
        r = right[mask]
        std = r.std(axis=0)
        const = r.mean(axis=0)
        max_std = float(std.max())
        ok = bool(max_std < threshold)
        flag = "" if ok else "  <-- NOT CONSTANT (see design note below)"
        print(f"{part:<16} " + " ".join(f"{v:12.6f}" for v in std) + f"   {max_std:.6f}   {flag}")
        report[part] = {
            "constant": const.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "max_std": max_std,
            "n_frames": int(mask.sum()),
            "all_below_threshold": ok,
        }

    n_bad = sum(1 for v in report.values() if not v["all_below_threshold"])
    if n_bad:
        print(
            f"\n[precheck] {n_bad} part(s) have right-arm action std >= {threshold}. "
            "Reporting only, per instructions -- NOT changing the model/dataset design. "
            "The mean is still stored as the fallback constant; review before deploying "
            "a checkpoint for the flagged part(s)."
        )
    else:
        print(f"\n[precheck] all {len(PART_ORDER)} parts: right-arm action std < {threshold} on all 7 dims. "
              "Safe to treat action[7:14] as a per-part constant.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument(
        "--out", type=str,
        default=str(_THIS_DIR / "right_arm_constants.json"),
    )
    args = ap.parse_args()

    dataset_root = resolve_repo_path(args.dataset_root) if args.dataset_root else (
        REPO_ROOT / "tools" / "roco2026_by_part"
    )
    report = run(dataset_root, args.threshold)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
