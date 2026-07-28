"""Compute the ONE shared norm_stats.json used by every part's training run.

Stats are pooled across all 9 parts' TRAIN splits (never val, never a
single part) so single-part vs. multi-part curriculum stages are never
confounded by different normalizers -- see conversation / config.py.

Usage:
    python compute_norm_stats.py [--val-fraction 0.1] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import REPO_ROOT, resolve_repo_path  # noqa: E402
from constants import LEFT_ACTION_IDX, LEFT_STATE_IDX, PART_ORDER  # noqa: E402
from data_io import episode_ids_for_part, load_data_table, load_episodes_table, split_episodes  # noqa: E402
from normalization import NormStats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantile-lo", type=float, default=0.01)
    ap.add_argument("--quantile-hi", type=float, default=0.99)
    ap.add_argument(
        "--out", type=str, default=str(_THIS_DIR / "norm_stats.json"),
    )
    args = ap.parse_args()

    dataset_root = resolve_repo_path(args.dataset_root) if args.dataset_root else (
        REPO_ROOT / "tools" / "roco2026_by_part"
    )

    episodes = load_episodes_table(dataset_root)
    data = load_data_table(dataset_root, columns=["observation.state", "action"])
    ep2task = dict(zip(episodes["episode_index"], episodes["task"]))
    data["task"] = data["episode_index"].map(ep2task)

    train_states, train_actions = [], []
    per_part_n_train_frames = {}
    for part in PART_ORDER:
        ep_ids = episode_ids_for_part(episodes, part)
        train_ids, val_ids = split_episodes(ep_ids, args.val_fraction, args.seed)
        mask = data["episode_index"].isin(train_ids).to_numpy() & (data["task"] == part).to_numpy()
        n = int(mask.sum())
        per_part_n_train_frames[part] = n
        print(f"{part:<16} train_episodes={len(train_ids):3d}  val_episodes={len(val_ids):3d}  train_frames={n}")

        state_full = np.stack(data.loc[mask, "observation.state"].to_numpy())
        action_full = np.stack(data.loc[mask, "action"].to_numpy())
        train_states.append(state_full[:, LEFT_STATE_IDX])
        train_actions.append(action_full[:, LEFT_ACTION_IDX])

    all_states = np.concatenate(train_states, axis=0)
    all_actions = np.concatenate(train_actions, axis=0)
    print(f"\npooled across all {len(PART_ORDER)} parts: {all_states.shape[0]} train frames")

    stats = NormStats.compute(
        all_states, all_actions,
        quantile_lo=args.quantile_lo, quantile_hi=args.quantile_hi,
        meta={
            "dataset_root": str(dataset_root),
            "parts": list(PART_ORDER),
            "val_fraction": args.val_fraction,
            "split_seed": args.seed,
            "quantile_lo": args.quantile_lo,
            "quantile_hi": args.quantile_hi,
            "n_train_frames_total": int(all_states.shape[0]),
            "n_train_frames_per_part": per_part_n_train_frames,
        },
    )
    stats.save(args.out)
    print(f"\nstate_mean: {stats.state_mean}")
    print(f"state_std:  {stats.state_std}")
    print(f"action_min: {stats.action_min}")
    print(f"action_max: {stats.action_max}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
