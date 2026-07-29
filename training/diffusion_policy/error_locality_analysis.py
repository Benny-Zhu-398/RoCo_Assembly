"""Where do the worst position-error predictions land in the episode, and do
they coincide with a gripper detent change?

Motivated by evaluate.py's per-timestep breakdown on battery_size1: mean
position error is *not* monotonic in prediction horizon (peaks at t=4, decays
by t=15), and p95 is flat ~340mm for t=1..7 then drops under 155mm from t=8
on, while the median stays ~2mm everywhere. That gap (mean >> median) means a
small subset of chunks carries huge error, not a uniform degradation with
horizon. This script isolates that subset and asks:

  1. Where does the *chunk* (not the erroring timestep) start within its
     episode -- concentrated in one phase of the episode, or spread out?
     Reported as start_frame / episode_length, i.e. "chunk 起始帧 / episode_length".
  2. Does the ground-truth gripper action change detent (of the 11
     evenly-spaced levels evaluate.py already snaps to) somewhere inside
     that chunk? Compared against the same rate over all valid chunks, to
     see if high error is enriched near a gripper transition rather than
     just uniformly more likely whenever a chunk happens to contain one.

Same run_eval() as evaluate.py (same checkpoint, same val split, same DDIM
sampling path) -- this only adds bookkeeping of which (episode, start_frame,
episode_length) each dataset row came from, which evaluate.py's aggregate-only
summarize()/per_timestep_summary() throw away.

Usage:
    python error_locality_analysis.py --part battery_size1 --ckpt outputs/battery_size1/final.pt
    python error_locality_analysis.py --part gear_60teeth --ckpt outputs/gear_60teeth/final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import resolve_repo_path  # noqa: E402
from constants import ACTION_GRIPPER_IDX, ACTION_XYZ_SLICE, PART_TO_IDX  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from inference_utils import (  # noqa: E402
    build_model_from_checkpoint,
    ddim_sample,
    load_checkpoint,
    load_norm_stats_from_checkpoint,
    part_to_idx_from_checkpoint,
)
from normalization import unnormalize_action  # noqa: E402

N_GRIPPER_LEVELS = 11
TOP_FRACTION = 0.05
PHASE_BINS = np.linspace(0.0, 1.0, 11)  # 10 deciles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    ap.add_argument("--dump-csv", type=str, default=None, help="optional path to dump the top-error rows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_ema = not args.no_ema

    ckpt = load_checkpoint(args.ckpt)
    if ckpt["part"] != args.part:
        raise ValueError(f"checkpoint {args.ckpt} was trained for part={ckpt['part']!r}, not --part {args.part!r}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[error_locality] CUDA not available, falling back to CPU (requested {args.device!r})")

    model, cfg = build_model_from_checkpoint(ckpt, device, use_ema=use_ema)
    if cfg.model.rotation_repr != "rotvec":
        raise NotImplementedError(f"rotation_repr={cfg.model.rotation_repr!r} not supported here")

    norm_stats = load_norm_stats_from_checkpoint(ckpt)
    part_to_idx = part_to_idx_from_checkpoint(ckpt)

    dataset_root = resolve_repo_path(cfg.data.dataset_root)
    val_ds = PartSequenceDataset(
        dataset_root=dataset_root,
        parts=cfg.data.part,
        split="val",
        horizon=cfg.data.horizon,
        norm_stats=norm_stats,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        rotation_repr=cfg.model.rotation_repr,
    )
    if len(val_ds) == 0:
        raise RuntimeError(f"val split for part={args.part!r} is empty")

    # samples[i] = (part, episode_index, start_frame); DataLoader below uses
    # shuffle=False so row i of the concatenated output tensors corresponds
    # 1:1 to val_ds.samples[i] -- same assumption evaluate.py relies on for
    # its own aggregate-only summary.
    episode_len = {ep: len(arr) for (p, ep), arr in val_ds._episode_action.items()}
    sample_ep = np.array([ep for (_, ep, _t) in val_ds.samples])
    sample_start = np.array([t for (_, _ep, t) in val_ds.samples])
    sample_eplen = np.array([episode_len[ep] for ep in sample_ep])

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    num_inference_steps = args.num_inference_steps or cfg.diffusion.num_inference_steps
    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim
    gen = torch.Generator(device=device).manual_seed(args.seed)

    pred_xyz_all, gt_xyz_all, pred_grip_all, gt_grip_all, is_pad_all = [], [], [], [], []
    for batch in loader:
        state = batch["state"].to(device)
        action_n_gt = batch["action"].to(device)
        is_pad = batch["action_is_pad"].numpy()
        task_idx = torch.full((state.shape[0],), part_to_idx[args.part], dtype=torch.long, device=device)

        action_n_pred = ddim_sample(
            model, state, task_idx, horizon, action_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=gen,
        )
        pred = unnormalize_action(action_n_pred.cpu().numpy(), norm_stats)
        gt = unnormalize_action(action_n_gt.cpu().numpy(), norm_stats)

        pred_xyz_all.append(pred[..., ACTION_XYZ_SLICE])
        gt_xyz_all.append(gt[..., ACTION_XYZ_SLICE])
        pred_grip_all.append(pred[..., ACTION_GRIPPER_IDX])
        gt_grip_all.append(gt[..., ACTION_GRIPPER_IDX])
        is_pad_all.append(is_pad)

    pred_xyz = np.concatenate(pred_xyz_all, axis=0)
    gt_xyz = np.concatenate(gt_xyz_all, axis=0)
    pred_grip = np.concatenate(pred_grip_all, axis=0)   # (N, horizon)
    gt_grip = np.concatenate(gt_grip_all, axis=0)
    is_pad = np.concatenate(is_pad_all, axis=0)
    valid = ~is_pad

    pos_err_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0  # (N, horizon)

    gripper_levels = np.linspace(
        norm_stats.action_min[ACTION_GRIPPER_IDX], norm_stats.action_max[ACTION_GRIPPER_IDX], N_GRIPPER_LEVELS,
    )
    gt_bin = np.argmin(np.abs(gt_grip[..., None] - gripper_levels[None, None, :]), axis=-1)  # (N, horizon)

    N = pos_err_mm.shape[0]
    assert N == len(val_ds.samples)

    # Flatten every valid (sample, t) prediction into one long table.
    sample_idx_flat, t_flat, err_flat = [], [], []
    for i in range(N):
        for t in range(horizon):
            if valid[i, t]:
                sample_idx_flat.append(i)
                t_flat.append(t)
                err_flat.append(pos_err_mm[i, t])
    sample_idx_flat = np.array(sample_idx_flat)
    t_flat = np.array(t_flat)
    err_flat = np.array(err_flat)
    n_flat = len(err_flat)

    threshold = np.percentile(err_flat, 100.0 * (1.0 - args.top_fraction))
    top_mask = err_flat >= threshold
    n_top = int(top_mask.sum())

    top_sample = sample_idx_flat[top_mask]
    top_t = t_flat[top_mask]
    top_err = err_flat[top_mask]
    top_start = sample_start[top_sample]
    top_eplen = sample_eplen[top_sample]
    top_ep = sample_ep[top_sample]

    chunk_phase = top_start / top_eplen                       # chunk 起始帧 / episode_length
    frame_phase = (top_start + top_t) / top_eplen             # (起始帧+t) / episode_length, supplementary

    print(f"\n=== error_locality_analysis: part={args.part} weights={'ema' if use_ema else 'raw'} "
          f"top_fraction={args.top_fraction} ===")
    print(f"n_valid_predictions={n_flat}  threshold(pos_err_mm)={threshold:.2f}  n_top={n_top}")

    # --- Q1: where in the episode do the top-error chunks start ---
    hist, edges = np.histogram(chunk_phase, bins=PHASE_BINS)
    print("\n[Q1] chunk start_frame / episode_length distribution (top-error predictions):")
    for i in range(len(hist)):
        frac = hist[i] / n_top * 100.0
        bar = "#" * int(round(frac / 2))
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.1f})  n={hist[i]:5d} ({frac:5.1f}%)  {bar}")

    hist2, edges2 = np.histogram(frame_phase, bins=PHASE_BINS)
    print("\n[Q1b] (start_frame + t) / episode_length distribution (supplementary, "
          "actual erroring frame instead of chunk start):")
    for i in range(len(hist2)):
        frac = hist2[i] / n_top * 100.0
        bar = "#" * int(round(frac / 2))
        print(f"  [{edges2[i]:.1f}, {edges2[i+1]:.1f})  n={hist2[i]:5d} ({frac:5.1f}%)  {bar}")

    # --- Q2: gripper detent changes ---
    def chunk_has_jump(i: int) -> bool:
        n_valid_i = int(valid[i].sum())
        bins = gt_bin[i, :n_valid_i]
        return bool(len(np.unique(bins)) > 1)

    def local_jump(i: int, t: int) -> bool:
        n_valid_i = int(valid[i].sum())
        lo, hi = max(0, t - 1), min(n_valid_i - 1, t + 1)
        return bool(len(np.unique(gt_bin[i, lo:hi + 1])) > 1)

    top_unique_samples = np.unique(top_sample)
    chunk_jump_top = np.array([chunk_has_jump(i) for i in top_sample])
    local_jump_top = np.array([local_jump(i, t) for i, t in zip(top_sample, top_t)])

    all_unique_samples = np.arange(N)
    chunk_jump_all = np.array([chunk_has_jump(i) for i in all_unique_samples])

    print(f"\n[Q2] gripper detent-change rate ({N_GRIPPER_LEVELS} levels):")
    print(f"  baseline: fraction of ALL {N} chunks whose GT gripper crosses >=1 detent "
          f"boundary somewhere in the chunk: {chunk_jump_all.mean()*100:.1f}%")
    print(f"  top-error predictions ({n_top}): fraction whose chunk contains a detent change: "
          f"{chunk_jump_top.mean()*100:.1f}%")
    print(f"  top-error predictions ({n_top}): fraction with a detent change LOCAL to t-1..t+1 "
          f"around the erroring timestep itself: {local_jump_top.mean()*100:.1f}%")

    if args.dump_csv:
        import csv
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "episode", "start_frame", "episode_length", "t",
                        "pos_err_mm", "chunk_phase", "frame_phase", "chunk_has_jump", "local_jump"])
            for k in range(n_top):
                i = top_sample[k]
                w.writerow([i, top_ep[k], top_start[k], top_eplen[k], top_t[k],
                            f"{top_err[k]:.3f}", f"{chunk_phase[k]:.4f}", f"{frame_phase[k]:.4f}",
                            chunk_jump_top[k], local_jump_top[k]])
        print(f"\nWrote {args.dump_csv} ({n_top} rows)")


if __name__ == "__main__":
    main()
