"""Open-loop evaluation of a trained GroupedDiffusionPolicyNet checkpoint
(training/diffusion_policy/{grouped_model.py,train.py}'s --group path)
against its own val split. Sibling of evaluate.py -- same metric
definitions (imported from there, see evaluate.py::compute_metrics), same
no-Isaac-Sim/replay-recorded-data approach; the only thing that differs is
which model class gets built and which DDIM sampler is called, since
GroupedDiffusionPolicyNet has no single `part`/`unet` (see
grouped_model.py's "SCOPE OF THIS PASS" note -- this file is the
inference-side adapter that note said was still missing).

A grouped checkpoint answers for every part in ckpt["parts"], not just one,
so this script evaluates each part in the group SEPARATELY (its own val
split, its own per-part gripper norm-stat range -- see
normalization.py::NormStats docstring for why gripper stats can't be
pooled across parts) through the one shared model, and reports metrics
per-part (no pooled group-level number -- see evaluate.py::compute_metrics,
which unnormalizes with one part's gripper range at a time).

Usage:
    python evaluate_grouped.py --group connectors --ckpt outputs_grouped/connectors/final.pt
    python evaluate_grouped.py --group gears --ckpt outputs_grouped/gears/final.pt --no-ema
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
from constants import GROUP_ORDER  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from evaluate import compute_metrics  # noqa: E402
from inference_utils import (  # noqa: E402
    build_grouped_model_from_checkpoint,
    ddim_sample_grouped,
    load_checkpoint,
    load_norm_stats_from_checkpoint,
    part_to_idx_from_checkpoint,
)


def run_eval_one_part(
    model, cfg, part: str, part_to_idx: dict, norm_stats, device: torch.device,
    batch_size: int, num_inference_steps: int, seed: int,
) -> dict:
    dataset_root = resolve_repo_path(cfg.data.dataset_root)
    val_ds = PartSequenceDataset(
        dataset_root=dataset_root,
        parts=part,
        split="val",
        horizon=cfg.data.horizon,
        norm_stats=norm_stats,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        rotation_repr=cfg.model.rotation_repr,
        load_images=cfg.model.use_vision,
        camera_keys=cfg.model.camera_keys,
        image_resize_hw=cfg.data.image_resize_hw,
    )
    if len(val_ds) == 0:
        raise RuntimeError(f"val split for part={part!r} is empty (val_fraction={cfg.data.val_fraction})")
    n_val_episodes = len(val_ds.episode_ids_used[part])

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    horizon = cfg.data.horizon
    action_dim = cfg.model.action_dim
    gen = torch.Generator(device=device).manual_seed(seed)

    pred_n_all, gt_n_all, is_pad_all = [], [], []
    for batch in loader:
        state = batch["state"].to(device)
        action_n_gt = batch["action"].to(device)
        is_pad = batch["action_is_pad"].numpy()
        task_idx = torch.full((state.shape[0],), part_to_idx[part], dtype=torch.long, device=device)
        images = {cam: batch[f"image_{cam}"].to(device) for cam in cfg.model.camera_keys}

        action_n_pred = ddim_sample_grouped(
            model, state, task_idx, horizon, action_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=gen,
            images=images,
        )

        pred_n_all.append(action_n_pred.cpu().numpy())
        gt_n_all.append(action_n_gt.cpu().numpy())
        is_pad_all.append(is_pad)

    pred_n = np.concatenate(pred_n_all, axis=0)
    gt_n = np.concatenate(gt_n_all, axis=0)
    is_pad = np.concatenate(is_pad_all, axis=0)

    metrics = compute_metrics(pred_n, gt_n, is_pad, norm_stats, part)
    return {
        "part": part,
        "n_val_episodes": int(n_val_episodes),
        "n_val_samples": int(len(val_ds)),
        **metrics,
    }, pred_n, gt_n, is_pad


def run_eval_group(
    group: str, ckpt_path: str, use_ema: bool, device_str: str, batch_size: int,
    num_inference_steps_override: Optional[int], seed: int,
) -> dict:
    ckpt = load_checkpoint(ckpt_path)
    if ckpt.get("group") != group:
        raise ValueError(f"checkpoint {ckpt_path} was trained for group={ckpt.get('group')!r}, not --group {group!r}")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device.type == "cpu" and device_str != "cpu":
        print(f"[evaluate_grouped] CUDA not available, falling back to CPU (requested {device_str!r})")

    model, cfg = build_grouped_model_from_checkpoint(ckpt, device, use_ema=use_ema)
    if cfg.model.rotation_repr != "rotvec":
        raise NotImplementedError(
            f"rotation_repr={cfg.model.rotation_repr!r} is not exercised end-to-end in this "
            "repo yet (see model.py) -- evaluate_grouped.py only supports 'rotvec'."
        )

    norm_stats = load_norm_stats_from_checkpoint(ckpt)
    part_to_idx = part_to_idx_from_checkpoint(ckpt)
    num_inference_steps = num_inference_steps_override or cfg.diffusion.num_inference_steps

    parts = list(ckpt["parts"])
    per_part = {}
    for part in parts:
        result, _pred_n, _gt_n, _is_pad = run_eval_one_part(
            model, cfg, part, part_to_idx, norm_stats, device,
            batch_size, num_inference_steps, seed,
        )
        per_part[part] = result

    return {
        "group": group,
        "parts": parts,
        "ckpt": str(Path(ckpt_path).resolve()),
        "use_ema": bool(use_ema),
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "per_part": per_part,
    }


def print_report(result: dict) -> None:
    print(f"\n=== evaluate_grouped.py: group={result['group']}  parts={result['parts']}  "
          f"weights={'ema' if result['use_ema'] else 'raw'}  "
          f"inference_steps={result['num_inference_steps']} ===")
    for part, r in result["per_part"].items():
        p, a, g = r["position_error_mm"], r["angle_error_deg"], r["gripper"]
        print(f"\n-- part={part}  ({r['n_val_episodes']} episodes, {r['n_val_samples']} chunk samples, "
              f"{r['n_valid_steps']} non-pad steps) --")
        print(f"position error (mm):   mean={p['mean']:.3f}  median={p['median']:.3f}  p95={p['p95']:.3f}")
        print(f"angle error (deg):     mean={a['mean']:.3f}  median={a['median']:.3f}  p95={a['p95']:.3f}")
        print(f"gripper snap-accuracy ({g['snap_num_levels']} levels): {g['snap_accuracy']*100:.2f}%")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", type=str, required=True, choices=list(GROUP_ORDER))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--no-ema", action="store_true", help="use raw (non-EMA) weights instead of the default EMA shadow")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-inference-steps", type=int, default=None, help="override the checkpoint's config value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="output json path (default: <ckpt_dir>/eval_grouped_ema.json or eval_grouped_raw.json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    use_ema = not args.no_ema
    result = run_eval_group(
        group=args.group,
        ckpt_path=args.ckpt,
        use_ema=use_ema,
        device_str=args.device,
        batch_size=args.batch_size,
        num_inference_steps_override=args.num_inference_steps,
        seed=args.seed,
    )
    print_report(result)

    out_path = Path(args.out) if args.out else (
        Path(args.ckpt).resolve().parent / f"eval_grouped_{'ema' if use_ema else 'raw'}.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
