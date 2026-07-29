"""Open-loop evaluation of a trained per-part Diffusion Policy checkpoint
against its own val split. No Isaac Sim / simulation dependency -- this
only replays recorded (state, action) pairs and compares DDIM samples
against ground truth, all in the same venv as train.py.

Usage:
    python evaluate.py --part gear_60teeth --ckpt outputs/gear_60teeth/final.pt
    python evaluate.py --part gear_60teeth --ckpt outputs/gear_60teeth/final.pt --no-ema

Everything needed to reproduce the exact training-time setup (model
architecture, norm stats, val split, diffusion schedule) is read back out
of the checkpoint -- nothing is recomputed. Run once with EMA weights
(default, standard DP inference recipe) and once with --no-ema to compare;
both write separate JSON files (eval_ema.json / eval_raw.json) next to the
checkpoint so summarize.py can pick them both up.

All error metrics are reported in unnormalized physical units:
  - position error: mm (L2 over predicted vs GT xyz)
  - angle error: degrees, true SO(3) geodesic distance between predicted
    and GT rotation matrices (NOT rotvec MSE, which diverges from this
    near the pi-magnitude rotvec jump documented in rotation_utils.py)
  - gripper: continuous error in the action's native (unnormalized) units,
    plus accuracy after snapping both prediction and GT to the nearest of
    11 evenly-spaced levels spanning the pooled-training gripper range
    (norm_stats action_min/max) -- there is no discrete gripper-detent
    definition elsewhere in this repo, so this bucketing is evaluate.py's
    own convention, not derived from hardware.
Metrics (a)/(b) are also broken out per action-chunk timestep, to see how
error grows with prediction horizon (informs n_action_steps for the
execution-time policy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import resolve_repo_path  # noqa: E402
from constants import ACTION_GRIPPER_IDX, ACTION_ROT_SLICE, ACTION_XYZ_SLICE, PART_TO_IDX  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from inference_utils import (  # noqa: E402
    build_model_from_checkpoint,
    ddim_sample,
    geodesic_angle_deg,
    load_checkpoint,
    load_norm_stats_from_checkpoint,
    part_to_idx_from_checkpoint,
)
from normalization import unnormalize_action  # noqa: E402

N_GRIPPER_LEVELS = 11


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def per_timestep_summary(values: np.ndarray, valid: np.ndarray) -> dict:
    """values/valid: (N, horizon). Returns per-column mean/median/p95/n_valid,
    skipping timesteps where a mask column ends up empty."""
    horizon = values.shape[1]
    mean, median, p95, n_valid = [], [], [], []
    for t in range(horizon):
        v = values[valid[:, t], t]
        if len(v) == 0:
            mean.append(None)
            median.append(None)
            p95.append(None)
            n_valid.append(0)
        else:
            mean.append(float(np.mean(v)))
            median.append(float(np.median(v)))
            p95.append(float(np.percentile(v, 95)))
            n_valid.append(int(len(v)))
    return {"mean": mean, "median": median, "p95": p95, "n_valid": n_valid}


def run_eval(
    part: str,
    ckpt_path: str,
    use_ema: bool,
    device_str: str,
    batch_size: int,
    num_inference_steps_override: Optional[int],
    seed: int,
) -> dict:
    ckpt = load_checkpoint(ckpt_path)
    if ckpt["part"] != part:
        raise ValueError(f"checkpoint {ckpt_path} was trained for part={ckpt['part']!r}, not --part {part!r}")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device.type == "cpu" and device_str != "cpu":
        print(f"[evaluate] CUDA not available, falling back to CPU (requested {device_str!r})")

    model, cfg = build_model_from_checkpoint(ckpt, device, use_ema=use_ema)
    if cfg.model.rotation_repr != "rotvec":
        raise NotImplementedError(
            f"rotation_repr={cfg.model.rotation_repr!r} is not exercised end-to-end in this "
            "repo yet (see model.py) -- evaluate.py only supports 'rotvec'."
        )

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
        raise RuntimeError(f"val split for part={part!r} is empty (val_fraction={cfg.data.val_fraction})")
    n_val_episodes = len(val_ds.episode_ids_used[part])

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    num_inference_steps = num_inference_steps_override or cfg.diffusion.num_inference_steps
    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim

    gen = torch.Generator(device=device).manual_seed(seed)

    pred_xyz_all, gt_xyz_all = [], []
    pred_rotvec_all, gt_rotvec_all = [], []
    pred_gripper_all, gt_gripper_all = [], []
    is_pad_all = []

    for batch in loader:
        state = batch["state"].to(device)
        action_n_gt = batch["action"].to(device)
        is_pad = batch["action_is_pad"].numpy()
        task_idx = torch.full((state.shape[0],), part_to_idx[part], dtype=torch.long, device=device)

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
        pred_rotvec_all.append(pred[..., ACTION_ROT_SLICE])
        gt_rotvec_all.append(gt[..., ACTION_ROT_SLICE])
        pred_gripper_all.append(pred[..., ACTION_GRIPPER_IDX])
        gt_gripper_all.append(gt[..., ACTION_GRIPPER_IDX])
        is_pad_all.append(is_pad)

    pred_xyz = np.concatenate(pred_xyz_all, axis=0)          # (N, horizon, 3)
    gt_xyz = np.concatenate(gt_xyz_all, axis=0)
    pred_rotvec = np.concatenate(pred_rotvec_all, axis=0)
    gt_rotvec = np.concatenate(gt_rotvec_all, axis=0)
    pred_gripper = np.concatenate(pred_gripper_all, axis=0)  # (N, horizon)
    gt_gripper = np.concatenate(gt_gripper_all, axis=0)
    is_pad = np.concatenate(is_pad_all, axis=0)              # (N, horizon) bool
    valid = ~is_pad

    pos_err_mm = np.linalg.norm(pred_xyz - gt_xyz, axis=-1) * 1000.0
    ang_err_deg = geodesic_angle_deg(pred_rotvec, gt_rotvec)
    gripper_err = np.abs(pred_gripper - gt_gripper)

    gripper_levels = np.linspace(
        norm_stats.action_min[ACTION_GRIPPER_IDX], norm_stats.action_max[ACTION_GRIPPER_IDX], N_GRIPPER_LEVELS,
    )
    pred_bin = np.argmin(np.abs(pred_gripper[..., None] - gripper_levels[None, None, :]), axis=-1)
    gt_bin = np.argmin(np.abs(gt_gripper[..., None] - gripper_levels[None, None, :]), axis=-1)
    snap_correct = (pred_bin == gt_bin) & valid

    n_valid = int(valid.sum())

    return {
        "part": part,
        "ckpt": str(Path(ckpt_path).resolve()),
        "use_ema": bool(use_ema),
        "num_inference_steps": int(num_inference_steps),
        "horizon": int(horizon),
        "seed": int(seed),
        "n_val_episodes": int(n_val_episodes),
        "n_val_samples": int(len(val_ds)),
        "n_valid_steps": n_valid,
        "position_error_mm": summarize(pos_err_mm[valid]),
        "angle_error_deg": summarize(ang_err_deg[valid]),
        "gripper": {
            "continuous_error": summarize(gripper_err[valid]),
            "snap_num_levels": N_GRIPPER_LEVELS,
            "snap_levels": gripper_levels.tolist(),
            "snap_accuracy": float(snap_correct.sum() / max(n_valid, 1)),
        },
        "per_timestep": {
            "position_error_mm": per_timestep_summary(pos_err_mm, valid),
            "angle_error_deg": per_timestep_summary(ang_err_deg, valid),
        },
    }


def print_report(result: dict) -> None:
    print(f"\n=== evaluate.py: part={result['part']}  weights={'ema' if result['use_ema'] else 'raw'}  "
          f"inference_steps={result['num_inference_steps']} ===")
    print(f"val: {result['n_val_episodes']} episodes, {result['n_val_samples']} chunk samples, "
          f"{result['n_valid_steps']} non-pad steps")

    p = result["position_error_mm"]
    a = result["angle_error_deg"]
    g = result["gripper"]
    print(f"\nposition error (mm):   mean={p['mean']:.3f}  median={p['median']:.3f}  p95={p['p95']:.3f}")
    print(f"angle error (deg):     mean={a['mean']:.3f}  median={a['median']:.3f}  p95={a['p95']:.3f}")
    print(f"gripper error (raw):   mean={g['continuous_error']['mean']:.5f}  "
          f"median={g['continuous_error']['median']:.5f}  p95={g['continuous_error']['p95']:.5f}")
    print(f"gripper snap-accuracy ({g['snap_num_levels']} levels): {g['snap_accuracy']*100:.2f}%")

    pt_pos = result["per_timestep"]["position_error_mm"]
    pt_ang = result["per_timestep"]["angle_error_deg"]
    print(f"\nper-timestep breakdown (horizon={result['horizon']}):")
    print(f"{'t':>3} {'n':>8} {'pos_mean_mm':>12} {'pos_p95_mm':>12} {'ang_mean_deg':>13} {'ang_p95_deg':>12}")
    for t in range(result["horizon"]):
        n = pt_pos["n_valid"][t]
        if n == 0:
            print(f"{t:3d} {n:8d} {'--':>12} {'--':>12} {'--':>13} {'--':>12}")
            continue
        print(f"{t:3d} {n:8d} {pt_pos['mean'][t]:12.3f} {pt_pos['p95'][t]:12.3f} "
              f"{pt_ang['mean'][t]:13.3f} {pt_ang['p95'][t]:12.3f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--no-ema", action="store_true", help="use raw (non-EMA) weights instead of the default EMA shadow")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-inference-steps", type=int, default=None, help="override the checkpoint's config value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="output json path (default: <ckpt_dir>/eval_ema.json or eval_raw.json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_ema = not args.no_ema
    result = run_eval(
        part=args.part,
        ckpt_path=args.ckpt,
        use_ema=use_ema,
        device_str=args.device,
        batch_size=args.batch_size,
        num_inference_steps_override=args.num_inference_steps,
        seed=args.seed,
    )
    print_report(result)

    out_path = Path(args.out) if args.out else (
        Path(args.ckpt).resolve().parent / f"eval_{'ema' if use_ema else 'raw'}.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
