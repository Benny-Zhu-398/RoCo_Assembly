"""Scan outputs/<part>/eval_*.json (written by evaluate.py) and print a
side-by-side table across all evaluated parts, so it's obvious at a glance
which parts are hard.

Usage:
    python summarize.py                       # ema weights, all parts found
    python summarize.py --weights both         # ema and raw rows side by side
    python summarize.py --csv summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import PART_ORDER  # noqa: E402

_WEIGHTS_TO_FILENAMES = {
    "ema": ["eval_ema.json"],
    "raw": ["eval_raw.json"],
    "both": ["eval_ema.json", "eval_raw.json"],
}


def find_eval_files(outputs_dir: Path, weights: str) -> list[Path]:
    names = _WEIGHTS_TO_FILENAMES[weights]
    files = []
    for part_dir in sorted(outputs_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        for name in names:
            f = part_dir / name
            if f.exists():
                files.append(f)
    return files


def load_row(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "part": d["part"],
        "weights": "ema" if d["use_ema"] else "raw",
        "n_val_episodes": d["n_val_episodes"],
        "n_valid_steps": d["n_valid_steps"],
        "pos_mm_mean": d["position_error_mm"]["mean"],
        "pos_mm_median": d["position_error_mm"]["median"],
        "pos_mm_p95": d["position_error_mm"]["p95"],
        "ang_deg_mean": d["angle_error_deg"]["mean"],
        "ang_deg_median": d["angle_error_deg"]["median"],
        "ang_deg_p95": d["angle_error_deg"]["p95"],
        "gripper_err_mean": d["gripper"]["continuous_error"]["mean"],
        "gripper_snap_acc_pct": d["gripper"]["snap_accuracy"] * 100.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", type=str, default=str(_THIS_DIR / "outputs"))
    ap.add_argument("--weights", choices=["ema", "raw", "both"], default="ema")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    files = find_eval_files(outputs_dir, args.weights)
    if not files:
        print(f"no eval_*.json found under {outputs_dir} -- run evaluate.py for at least one part first")
        return

    rows = [load_row(f) for f in files]
    df = pd.DataFrame(rows)
    part_rank = {p: i for i, p in enumerate(PART_ORDER)}
    df["_rank"] = df["part"].map(part_rank)
    df = df.sort_values(["_rank", "weights"]).drop(columns="_rank").reset_index(drop=True)

    missing = [p for p in PART_ORDER if p not in set(df["part"])]
    if missing:
        print(f"(no eval json yet for: {', '.join(missing)})\n")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220, "display.max_columns", None):
        print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
