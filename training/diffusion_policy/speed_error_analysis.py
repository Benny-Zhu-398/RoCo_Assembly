"""Does position error track end-effector speed (accel/decel phases) rather
than the gripper detent-change events ruled out by error_locality_analysis.py?

Follow-up to that script's finding: the top-5% position-error predictions
cluster in the reach-to-grasp and lift-away-from-grasp windows that bracket
the gripper transition, but not on the transition frame itself, and are NOT
enriched for a gripper detent change near the erroring timestep. Hypothesis
here: those bracketing windows are exactly where GT end-effector speed is
highest (the grasp frame itself is close to stationary => easy target;
approach/retreat are the fast segments => harder target), and where the
recorded action is already loosely coupled to next-step state (IK/PD control
lag), i.e. part of the "error" is baked into the demonstration data, not
purely a model failure.

Reuses the same checkpoint / val split / DDIM sampling path as evaluate.py
and error_locality_analysis.py -- this only adds, per (chunk, t):
  - GT end-effector speed at that instant: ||action_xyz[af+1] - action_xyz[af]||
    * FPS, from the raw (unnormalized) recorded action trajectory (af = the
    absolute episode frame that prediction step t corresponds to).
  - GT joint-velocity norm at that instant, from observation.state's jvel
    block (STATE_JVEL_SLICE) -- an independent, non-finite-difference speed
    proxy, as a cross-check on the action-diff speed above.
  - control lag ||action_xyz[af] - state_xyz[af+1]|| (mm): how far the
    recorded action command is from where the arm actually ended up one
    frame later -- large values indicate the demonstration's IK/PD didn't
    track the command within a single frame.

Usage:
    python speed_error_analysis.py --part battery_size1 --ckpt outputs/battery_size1/final.pt
    python speed_error_analysis.py --part gear_60teeth --ckpt outputs/gear_60teeth/final.pt
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
from constants import (  # noqa: E402
    ACTION_XYZ_SLICE,
    DATASET_FPS,
    PART_TO_IDX,
    STATE_JVEL_SLICE,
    STATE_XYZ_SLICE,
)
from dataset import PartSequenceDataset  # noqa: E402
from inference_utils import (  # noqa: E402
    build_model_from_checkpoint,
    ddim_sample,
    load_checkpoint,
    load_norm_stats_from_checkpoint,
    part_to_idx_from_checkpoint,
)
from normalization import unnormalize_action  # noqa: E402

SPEED_BIN_EDGES_MM_S = [0, 25, 50, 100, 150, 200, 300, 500, 1e9]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bad-threshold-mm", type=float, default=200.0,
                     help="abs position-error cutoff (mm) used to define the 'bad' set for the "
                          "t=1..7 plateau overlap check (Q3)")
    ap.add_argument("--dump-csv", type=str, default=None)
    return ap.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(np.float64), np.argsort(np.argsort(y)).astype(np.float64))


def main() -> None:
    args = parse_args()
    use_ema = not args.no_ema

    ckpt = load_checkpoint(args.ckpt)
    if ckpt["part"] != args.part:
        raise ValueError(f"checkpoint {args.ckpt} was trained for part={ckpt['part']!r}, not --part {args.part!r}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[speed_error] CUDA not available, falling back to CPU (requested {args.device!r})")

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

    episode_len = {ep: len(arr) for (p, ep), arr in val_ds._episode_action.items()}
    sample_ep = np.array([ep for (_, ep, _t) in val_ds.samples])
    sample_start = np.array([t for (_, _ep, t) in val_ds.samples])
    sample_eplen = np.array([episode_len[ep] for ep in sample_ep])

    # Raw (unnormalized, meters) per-episode xyz trajectories + jvel, keyed
    # by episode_index -- val_ds._episode_state/_episode_action are already
    # left-arm-sliced but NOT normalized (normalization happens per-sample
    # in __getitem__), so these are exactly the recorded values.
    ep_action_xyz = {ep: arr[:, ACTION_XYZ_SLICE] for (p, ep), arr in val_ds._episode_action.items()}
    ep_state_xyz = {ep: arr[:, STATE_XYZ_SLICE] for (p, ep), arr in val_ds._episode_state.items()}
    ep_state_jvel = {ep: arr[:, STATE_JVEL_SLICE] for (p, ep), arr in val_ds._episode_state.items()}

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    num_inference_steps = args.num_inference_steps or cfg.diffusion.num_inference_steps
    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim
    gen = torch.Generator(device=device).manual_seed(args.seed)

    pred_xyz_all, gt_xyz_all, is_pad_all = [], [], []
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
        is_pad_all.append(is_pad)

    pred_xyz = np.concatenate(pred_xyz_all, axis=0)
    gt_xyz = np.concatenate(gt_xyz_all, axis=0)
    is_pad = np.concatenate(is_pad_all, axis=0)
    valid = ~is_pad
    pos_err_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0  # (N, horizon)

    N, H = pos_err_mm.shape
    assert N == len(val_ds.samples)

    # Per (sample, t): absolute frame af = start_frame + t. speed/lag need
    # af+1 to exist in the same episode; drop the (rare) boundary rows where
    # it doesn't rather than fabricating a value.
    idx_i, idx_t, err_v, speed_v, jvel_v, lag_v, af_v = [], [], [], [], [], [], []
    for i in range(N):
        ep = int(sample_ep[i])
        L = episode_len[ep]
        start = int(sample_start[i])
        axyz = ep_action_xyz[ep]
        sxyz = ep_state_xyz[ep]
        jvel = ep_state_jvel[ep]
        for t in range(H):
            if not valid[i, t]:
                continue
            af = start + t
            if af + 1 >= L:
                continue
            speed = float(np.linalg.norm(axyz[af + 1] - axyz[af]) * 1000.0 * DATASET_FPS)
            lag = float(np.linalg.norm(axyz[af] - sxyz[af + 1]) * 1000.0)
            jv = float(np.linalg.norm(jvel[af]))
            idx_i.append(i)
            idx_t.append(t)
            af_v.append(af)
            err_v.append(pos_err_mm[i, t])
            speed_v.append(speed)
            jvel_v.append(jv)
            lag_v.append(lag)

    idx_i = np.array(idx_i)
    idx_t = np.array(idx_t)
    af_v = np.array(af_v)
    err_v = np.array(err_v)
    speed_v = np.array(speed_v)
    jvel_v = np.array(jvel_v)
    lag_v = np.array(lag_v)
    n = len(err_v)

    print(f"\n=== speed_error_analysis: part={args.part} weights={'ema' if use_ema else 'raw'} ===")
    print(f"n_valid_predictions_with_speed={n} (dropped episode-boundary rows where af+1 didn't exist)")

    # --- Q1: correlation between position error and GT speed ---
    r_speed = pearson(err_v, speed_v)
    rho_speed = spearman(err_v, speed_v)
    r_jvel = pearson(err_v, jvel_v)
    rho_jvel = spearman(err_v, jvel_v)
    print(f"\n[Q1] corr(position_error_mm, ee_speed_mm_s):      pearson r={r_speed:+.3f}  spearman rho={rho_speed:+.3f}")
    print(f"[Q1] corr(position_error_mm, joint_vel_norm):      pearson r={r_jvel:+.3f}  spearman rho={rho_jvel:+.3f}")

    print("\n[Q1] mean position error by GT ee-speed bin:")
    print(f"{'speed range (mm/s)':>22} {'n':>8} {'err_mean':>10} {'err_median':>10} {'err_p95':>10}")
    edges = SPEED_BIN_EDGES_MM_S
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (speed_v >= lo) & (speed_v < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        label = f"[{lo:.0f}, {hi:.0f})" if hi < 1e9 else f"[{lo:.0f}, inf)"
        print(f"{label:>22} {cnt:8d} {err_v[m].mean():10.2f} {np.median(err_v[m]):10.2f} {np.percentile(err_v[m],95):10.2f}")

    # --- Q2: control lag ||action_xyz[t] - state_xyz[t+1]|| ---
    threshold = np.percentile(err_v, 95.0)
    top_mask = err_v >= threshold
    print(f"\n[Q2] control lag ||action_xyz[af]-state_xyz[af+1]|| (mm):")
    print(f"  baseline (all {n} valid predictions):        mean={lag_v.mean():.3f}  median={np.median(lag_v):.3f}  p95={np.percentile(lag_v,95):.3f}")
    print(f"  top-5% position-error predictions ({int(top_mask.sum())}): mean={lag_v[top_mask].mean():.3f}  "
          f"median={np.median(lag_v[top_mask]):.3f}  p95={np.percentile(lag_v[top_mask],95):.3f}")
    r_lag = pearson(err_v, lag_v)
    print(f"  corr(position_error_mm, control_lag_mm): pearson r={r_lag:+.3f}")

    # --- Q3: is the t=1..7 p95 plateau driven by the same chunks repeating ---
    print(f"\n[Q3] t=1..7 p95 plateau -- overlap of 'bad' (err > {args.bad_threshold_mm:.0f}mm) sample_idx across t:")
    bad_sets = {}
    for t in range(1, 8):
        m = (idx_t == t) & (err_v > args.bad_threshold_mm)
        bad_sets[t] = set(idx_i[m].tolist())
        print(f"  t={t}: n_bad={len(bad_sets[t])}")

    union_bad = set().union(*bad_sets.values()) if bad_sets else set()
    inter_bad = set.intersection(*bad_sets.values()) if all(bad_sets.values()) else set()
    print(f"  union across t=1..7:        {len(union_bad)} unique chunks")
    print(f"  intersection across t=1..7: {len(inter_bad)} chunks bad at EVERY one of t=1..7")

    if union_bad:
        union_arr = np.array(sorted(union_bad))
        starts = sample_start[union_arr]
        eplens = sample_eplen[union_arr]
        eps = sample_ep[union_arr]
        # how many distinct t's (of 1..7) each bad chunk shows up in
        counts_per_chunk = np.array([sum(i in bad_sets[t] for t in range(1, 8)) for i in union_arr])
        print(f"\n  {len(union_arr)} chunks behind the plateau -- how many of the 7 timesteps (t=1..7) "
              f"each one is 'bad' at:")
        for k in range(1, 8):
            cnt = int((counts_per_chunk == k).sum())
            if cnt:
                print(f"    bad at exactly {k}/7 timesteps: {cnt} chunks")
        print(f"\n  their start_frame/episode_length: mean={np.mean(starts/eplens):.3f} "
              f"min={np.min(starts/eplens):.3f} max={np.max(starts/eplens):.3f}")
        print(f"  distinct episodes represented: {len(set(eps.tolist()))} (of {len(set(sample_ep.tolist()))} val episodes)")
        # mean GT speed of these chunks over the t=1..7 window, vs population mean
        chunk_speed_mask = np.isin(idx_i, union_arr) & (idx_t >= 1) & (idx_t <= 7)
        pop_speed_mask = (idx_t >= 1) & (idx_t <= 7)
        print(f"  mean ee-speed over t=1..7: plateau chunks={speed_v[chunk_speed_mask].mean():.1f} mm/s  "
              f"vs all chunks={speed_v[pop_speed_mask].mean():.1f} mm/s")

    if args.dump_csv:
        import csv
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "t", "abs_frame", "pos_err_mm", "ee_speed_mm_s", "joint_vel_norm", "control_lag_mm"])
            for k in range(n):
                w.writerow([idx_i[k], idx_t[k], af_v[k], f"{err_v[k]:.3f}", f"{speed_v[k]:.3f}",
                            f"{jvel_v[k]:.4f}", f"{lag_v[k]:.3f}"])
        print(f"\nWrote {args.dump_csv} ({n} rows)")


if __name__ == "__main__":
    main()
