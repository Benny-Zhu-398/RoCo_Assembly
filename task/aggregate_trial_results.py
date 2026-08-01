"""Aggregate N per-trial --results-json files from run_pick_place.py into a
success rate for one part.

run_pick_place.py has no in-process multi-episode loop (see
policies/gt_replay.py's docstring) -- getting a success rate over 10-20
trials means launching the harness 10-20 times, each with a distinct
--results-json path, then pointing this script at that directory.

Usage:
    python aggregate_trial_results.py --dir results/battery1_gt_replay --part battery_size1
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of --results-json trial files (*.json)")
    ap.add_argument("--part", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no *.json files in {args.dir}")

    outcomes = []
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        row = next((r for r in payload.get("per_part", []) if r.get("name") == args.part), None)
        if row is None:
            print(f"  {os.path.basename(p):<40} part {args.part!r} not found in per_part -- SKIPPED")
            continue
        outcomes.append((p, bool(row.get("pass"))))
        detail = f"pos_err_mm={row['pos_err_m']*1000:.2f}" if "pos_err_m" in row else f"snap_fired={row.get('snap_fired')}"
        print(f"  {os.path.basename(p):<40} pass={row['pass']!s:<5} ({detail})")

    n = len(outcomes)
    n_pass = sum(1 for _, ok in outcomes if ok)
    rate = n_pass / n if n else float("nan")
    print(f"\n[aggregate] part={args.part}  n_trials={n}  n_pass={n_pass}  success_rate={rate*100:.1f}%")


if __name__ == "__main__":
    main()
