"""Audit whether the recorded gripper release (open) action is truncated at
the episode boundary -- i.e. the scripted collector commands "open" but the
episode ends before the physical gripper state ever moves toward it.

Motivated by a diagnostic plot of battery_size1 val episode 61: the GT action
gripper value jumps from closed (0.105) to open (0.30) on the LAST recorded
frame, while the recorded state gripper value never moves off closed at all.
Root cause candidate: task/param_config.py SETTLE_OPEN=1 (vs SETTLE_CLOSE=10)
-- the path follower only dwells 1 step at the "open" waypoint (lock_pose, so
the pos/orn gate is trivially satisfied) before advancing to "lift_place" and
finishing the episode, giving the physical gripper ~1 control step to react
before recording stops.

Per-part cfg["gripper_open"/"gripper_close"] (task/param_config.py) turned
out to be EEPoseController *command* units that do NOT match the raw units
actually recorded in the dataset's action/state gripper channel (confirmed
by manual dump: battery_size1 cfg says open=0.2/close=0.07, but the data
shows ~0.301/~0.105). Multi-part sessions also mean an episode's very first
frames can carry an inherited gripper value from the PREVIOUS part in the
sequence, so "value at frame 0" isn't a safe "open" reference either.

So this script sidesteps both problems and works directly off each episode's
own trailing frames, semantics-free:
  - pre_ref: median state_grip over frames [-15:-5] -- a stable baseline from
    just before any end-of-episode gripper change.
  - opened_commanded: does the FINAL commanded action move away from pre_ref
    by more than a noise floor (0.02, well above the ~1e-4 jitter observed
    on flat plateaus and well below the ~0.03-0.15 open/close deltas seen
    across parts)?
  - frac_state_followed = (state[-1] - pre_ref) / (action[-1] - pre_ref):
    0 = state never moved, 1 = state fully caught up to the final command.
  - truncated: opened_commanded and frac_state_followed < 0.5.
  - action_changed_in_last_n / state_changed_in_last_n: whether the action
    (resp. state) trace itself moves within the literal last `--last-n`
    frames (matches the user's original "check the last 5 frames" ask).

Usage:
    python release_truncation_audit.py --part battery_size1
    python release_truncation_audit.py --all-parts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "task"))

from constants import ACTION_GRIPPER_IDX, LEFT_ACTION_IDX, LEFT_STATE_IDX, PART_ORDER, STATE_GRIPPER_IDX  # noqa: E402
from data_io import episode_frames, episode_ids_for_part, load_data_table, load_episodes_table  # noqa: E402

import param_config as pc  # noqa: E402

DEFAULT_ROOT = _REPO_ROOT / "tools" / "roco2026_by_part"
NOISE_FLOOR = 0.02
FOLLOWED_FRAC_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--part", type=str, default=None, choices=list(PART_ORDER))
    ap.add_argument("--all-parts", action="store_true")
    ap.add_argument("--last-n", type=int, default=5, help="window (frames) checked for the report's headline stat")
    ap.add_argument("--csv", type=str, default=None, help="optional per-episode CSV dump")
    return ap.parse_args()


def analyze_part(data_df, episodes_df, part: str, last_n: int):
    cfg = pc.get_part_config(part)
    release_mode = cfg.get("release_mode", "open")

    ep_ids = episode_ids_for_part(episodes_df, part)
    rows = []
    for ep in ep_ids:
        state_full, action_full = episode_frames(data_df, ep)
        action_grip = action_full[:, LEFT_ACTION_IDX][:, ACTION_GRIPPER_IDX]
        state_grip = state_full[:, LEFT_STATE_IDX][:, STATE_GRIPPER_IDX]
        L = len(action_grip)

        lo = max(0, L - 15)
        hi = max(lo + 1, L - 5)
        pre_ref = float(np.median(state_grip[lo:hi]))

        action_final = float(action_grip[-1])
        state_final = float(state_grip[-1])
        commanded_delta = action_final - pre_ref
        opened_commanded = abs(commanded_delta) > NOISE_FLOOR
        frac_followed = ((state_final - pre_ref) / commanded_delta) if opened_commanded else None
        truncated = bool(opened_commanded and (frac_followed is None or frac_followed < FOLLOWED_FRAC_THRESHOLD))

        n = min(last_n, L - 1)
        action_changed_last_n = bool(abs(action_grip[-1] - action_grip[-1 - n]) > NOISE_FLOOR) if n > 0 else False
        state_changed_last_n = bool(abs(state_grip[-1] - state_grip[-1 - n]) > NOISE_FLOOR) if n > 0 else False

        rows.append(dict(
            part=part, episode=ep, length=L, release_mode=release_mode,
            pre_ref=pre_ref, action_final=action_final, state_final=state_final,
            commanded_delta=commanded_delta, opened_commanded=opened_commanded,
            frac_state_followed=(None if frac_followed is None else round(frac_followed, 3)),
            truncated=truncated,
            action_changed_last_n=action_changed_last_n, state_changed_last_n=state_changed_last_n,
        ))
    return rows


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    parts = list(PART_ORDER) if args.all_parts else [args.part or "battery_size1"]

    data_df = load_data_table(dataset_root, columns=["observation.state", "action", "task_index"])
    episodes_df = load_episodes_table(dataset_root)

    all_rows = []
    for part in parts:
        rows = analyze_part(data_df, episodes_df, part, args.last_n)
        all_rows.extend(rows)

        n = len(rows)
        n_commanded = sum(r["opened_commanded"] for r in rows)
        n_truncated = sum(r["truncated"] for r in rows)
        n_action_last_n = sum(r["action_changed_last_n"] for r in rows)
        n_state_last_n = sum(r["state_changed_last_n"] for r in rows)
        fracs = [r["frac_state_followed"] for r in rows if r["frac_state_followed"] is not None]

        print(f"\n=== part={part}  release_mode={rows[0]['release_mode'] if rows else '?'}  n_episodes={n} ===")
        print(f"  final-frame gripper command differs from the pre-window baseline (opened_commanded): "
              f"{n_commanded}/{n}")
        print(f"  action changes within the last {args.last_n} frames: {n_action_last_n}/{n}")
        print(f"  state  changes within the last {args.last_n} frames: {n_state_last_n}/{n}")
        if n_commanded:
            print(f"  TRUNCATED (final command issued, state tracked < {FOLLOWED_FRAC_THRESHOLD:.0%} of the way): "
                  f"{n_truncated}/{n_commanded}")
            print(f"  frac_state_followed over commanded episodes: mean={np.mean(fracs):.3f} "
                  f"median={np.median(fracs):.3f} p90={np.percentile(fracs, 90):.3f}")

    if args.csv:
        import csv
        keys = list(all_rows[0].keys()) if all_rows else []
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {args.csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
