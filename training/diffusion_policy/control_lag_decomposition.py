"""Decompose control lag ||action_xyz[t] - state_xyz[t+1]|| into two causes:

  1. "action-step-induced lag": the recorded action itself just jumped a lot
     from frame t-1 to t (a waypoint hop -- this scripted policy is
     waypoint-based, not a continuously-interpolated trajectory), so of
     course state[t+1] (moving smoothly) hasn't caught up to action[t] one
     frame later. Not a data-quality problem.
  2. "true tracking error": the action is smooth (small step from t-1 to
     t) but the state still lags behind it -- an actual IK/PD tracking
     failure baked into the demonstration.

Classifier: action_jump[t] = ||action_xyz[t] - action_xyz[t-1]|| (mm), the
size of the hop that landed on the current commanded target. Correlated
against lag[t] = ||action_xyz[t] - state_xyz[t+1]|| (mm) (same definition
used in speed_error_analysis.py's Q2, and in the earlier battery_size1
ep61 diagnostic plot where the 340mm lag spike at frame ~12 coincided with
an action step of x:250->30, y:-80->170mm).

Usage:
    python control_lag_decomposition.py --part battery_size1
    python control_lag_decomposition.py --all-parts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import ACTION_XYZ_SLICE, LEFT_ACTION_IDX, LEFT_STATE_IDX, PART_ORDER, STATE_XYZ_SLICE  # noqa: E402
from data_io import episode_frames, episode_ids_for_part, load_data_table, load_episodes_table  # noqa: E402

DEFAULT_ROOT = _THIS_DIR.parents[1] / "tools" / "roco2026_by_part"
JUMP_LOW_MM = 10.0   # "smooth action" ceiling
JUMP_HIGH_MM = 50.0  # "big waypoint hop" floor


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--part", type=str, default=None, choices=list(PART_ORDER))
    ap.add_argument("--all-parts", action="store_true")
    ap.add_argument("--csv", type=str, default=None)
    return ap.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(np.float64), np.argsort(np.argsort(y)).astype(np.float64))


def collect_samples(data_df, episodes_df, part: str):
    ep_ids = episode_ids_for_part(episodes_df, part)
    jumps, lags, eps, frames = [], [], [], []
    for ep in ep_ids:
        state_full, action_full = episode_frames(data_df, ep)
        state = state_full[:, LEFT_STATE_IDX]
        action = action_full[:, LEFT_ACTION_IDX]
        axyz = action[:, ACTION_XYZ_SLICE] * 1000.0
        sxyz = state[:, STATE_XYZ_SLICE] * 1000.0
        L = len(axyz)
        for t in range(1, L - 1):  # need t-1 for the jump and t+1 for the lag
            jump = float(np.linalg.norm(axyz[t] - axyz[t - 1]))
            lag = float(np.linalg.norm(axyz[t] - sxyz[t + 1]))
            jumps.append(jump)
            lags.append(lag)
            eps.append(ep)
            frames.append(t)
    return np.array(jumps), np.array(lags), np.array(eps), np.array(frames)


def report(part: str, jumps: np.ndarray, lags: np.ndarray):
    n = len(lags)
    r = pearson(jumps, lags)
    rho = spearman(jumps, lags)
    print(f"\n=== part={part}  n_samples={n} ===")
    print(f"  corr(action_jump_mm, control_lag_mm): pearson r={r:+.3f}  spearman rho={rho:+.3f}")

    low = jumps < JUMP_LOW_MM
    high = jumps >= JUMP_HIGH_MM
    mid = ~low & ~high
    for name, mask in [("smooth action (jump<%gmm) -- TRUE TRACKING ERROR pool" % JUMP_LOW_MM, low),
                        ("mid (%g<=jump<%gmm)" % (JUMP_LOW_MM, JUMP_HIGH_MM), mid),
                        ("big waypoint hop (jump>=%gmm) -- ACTION-STEP-INDUCED pool" % JUMP_HIGH_MM, high)]:
        cnt = int(mask.sum())
        if cnt == 0:
            print(f"  {name}: n=0")
            continue
        lm = lags[mask]
        print(f"  {name}: n={cnt} ({cnt/n:.1%})  lag mean={lm.mean():.2f}  median={np.median(lm):.2f}  "
              f"p95={np.percentile(lm,95):.2f}  max={lm.max():.2f} mm")

    # top-5% highest-lag samples: what fraction sit in the big-jump pool?
    thresh = np.percentile(lags, 95)
    top = lags >= thresh
    top_high_frac = float(high[top].mean())
    top_low_frac = float(low[top].mean())
    print(f"  top-5% lag samples (lag>={thresh:.1f}mm, n={int(top.sum())}): "
          f"{top_high_frac:.1%} are big-waypoint-hop frames, {top_low_frac:.1%} are smooth-action frames")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    parts = list(PART_ORDER) if args.all_parts else [args.part or "battery_size1"]

    data_df = load_data_table(dataset_root, columns=["observation.state", "action", "task_index"])
    episodes_df = load_episodes_table(dataset_root)

    all_rows = []
    for part in parts:
        jumps, lags, eps, frames = collect_samples(data_df, episodes_df, part)
        report(part, jumps, lags)
        if args.csv:
            for j, l, e, f in zip(jumps, lags, eps, frames):
                all_rows.append(dict(part=part, episode=int(e), frame=int(f),
                                      action_jump_mm=j, control_lag_mm=l))

    if args.csv and all_rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {args.csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
