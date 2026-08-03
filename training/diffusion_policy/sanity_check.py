"""Overfit sanity check: train a fresh model on --n-samples memorized
samples from one part and confirm training loss reaches near zero. This is
the only reliable way to catch a broken noise/denoise/unnormalize direction
or a dead conditioning path -- an end-to-end eval on real held-out data
can't isolate that class of bug as cleanly.

Usage:
    python sanity_check.py --part gear_60teeth
    python sanity_check.py --part gear_60teeth --n-samples 1 --num-steps 5000

Default is --n-samples 1: with exactly one memorized (state, action-chunk)
pair, eps = (x_t - sqrt(alpha_bar_t)*x0) / sqrt(1-alpha_bar_t) is a
DETERMINISTIC function of (x_t, t) for a fixed x0 -- there is no
theoretical loss floor above 0, no aliasing, no multimodality excuse.
Gate is strict: final loss must be < LOSS_PASS_THRESHOLD or this fails.

Internals: a single training STEP still uses >= --min-batch-size rows
(the --n-samples unique samples repeated to fill the batch, each row given
its own independently-resampled noise/timestep) even when --n-samples is
tiny. This is a gradient-variance fix, not a leniency knob: batch_size=1
with a freshly resampled (noise, timestep) every step is such a
high-variance gradient estimate that even a healthy pipeline looks like it
isn't learning; replicating rows (still zero aliasing -- same underlying
x0) averages that noise down per step without changing what's being
memorized. Confirmed empirically: raw batch_size=1 training on this repo's
model/dataset/train.py code as of this writing oscillates around
loss~0.9-1.3 for 2000 steps with no visible trend, while the same single
sample replicated to batch_size=64 shows a clean, mostly-monotonic
decrease -- but plateaus around loss~0.40-0.53 even after 5000 steps,
which for N=1 is itself a bug signature (see the training/diffusion_policy
conditioning-ablation investigation: real vs. zeroed global_cond converge
to statistically indistinguishable loss at 3000 steps of full-dataset
training, i.e. state conditioning was not reducing loss at all).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ExperimentConfig, resolve_repo_path  # noqa: E402
from constants import ACTION_GRIPPER_IDX, ACTION_NAMES, ACTION_ROT_SLICE, ACTION_XYZ_SLICE, PART_TO_IDX  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from inference_utils import ddim_sample, geodesic_angle_deg_euler_xyz  # noqa: E402
from model import DiffusionPolicyNet  # noqa: E402
from normalization import NormStats, unnormalize_action  # noqa: E402
from train import masked_mse  # noqa: E402

LOSS_PASS_THRESHOLD = 0.05   # final loss must be below this or the run FAILS -- not negotiable, see docstring


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--n-samples", type=int, default=1, help="unique (state, action-chunk) pairs to memorize")
    ap.add_argument("--min-batch-size", type=int, default=64,
                     help="replicate the n_samples rows (independently-resampled noise/timestep per row) to at "
                          "least this many rows per training step, to control gradient variance -- see docstring")
    ap.add_argument("--num-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--print-samples", type=int, default=3,
                     help="how many overfit samples to print the full per-timestep pred-vs-GT table for")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[sanity_check] CUDA not available, falling back to CPU (requested {args.device!r})")

    cfg = ExperimentConfig()
    cfg.data.part = args.part
    if args.horizon is not None:
        cfg.data.horizon = args.horizon
    cfg.data.__post_init__()
    cfg.model.__post_init__()

    norm_stats = NormStats.load(resolve_repo_path(cfg.train.norm_stats_path))

    dataset_root = resolve_repo_path(cfg.data.dataset_root)
    full_train_ds = PartSequenceDataset(
        dataset_root=dataset_root,
        parts=cfg.data.part,
        split="train",
        horizon=cfg.data.horizon,
        norm_stats=norm_stats,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        rotation_repr=cfg.model.rotation_repr,
    )
    if len(full_train_ds) < args.n_samples:
        raise ValueError(f"train split for part={args.part!r} has only {len(full_train_ds)} samples, "
                          f"fewer than --n-samples={args.n_samples}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(full_train_ds), size=args.n_samples, replace=False).tolist()
    overfit_ds = Subset(full_train_ds, indices)

    loader = DataLoader(overfit_ds, batch_size=len(overfit_ds), shuffle=False)
    batch = next(iter(loader))
    # Kept at the true --n-samples size for final DDIM sampling / printing --
    # only the training loop below trains on the replicated version.
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    is_pad = batch["action_is_pad"].to(device)
    task_idx = batch["task_idx"].to(device)

    replicas = max(1, -(-args.min_batch_size // len(overfit_ds)))  # ceil(min_batch_size / n_samples)
    state_r = state.repeat(replicas, 1)
    action_r = action.repeat(replicas, 1, 1)
    is_pad_r = is_pad.repeat(replicas, 1)
    task_idx_r = task_idx.repeat(replicas)

    print(f"[sanity_check] part={args.part}  overfitting on {len(overfit_ds)} unique sample(s)  "
          f"(replicated x{replicas} -> {state_r.shape[0]} rows/step for gradient-variance control)  "
          f"horizon={cfg.data.horizon}  action_dim={cfg.model.action_dim}  device={device}")

    model = DiffusionPolicyNet(cfg.model).to(device)

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Same cosine decay train.py uses.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.num_steps, 1))

    losses = []
    for step in range(args.num_steps):
        noise = torch.randn_like(action_r)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (action_r.shape[0],), device=device,
        ).long()
        noisy_action = noise_scheduler.add_noise(action_r, noise, timesteps)
        eps_pred = model(noisy_action, timesteps, state_r, task_idx_r)
        loss = masked_mse(eps_pred, noise, is_pad_r)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.num_steps - 1:
            print(f"[sanity_check] step={step:4d}  loss={loss.item():.6f}")

    avg_window = min(50, args.num_steps)
    initial_loss = float(np.mean(losses[:avg_window]))
    final_loss = float(np.mean(losses[-avg_window:]))
    print(f"\n[sanity_check] initial_loss(avg first {avg_window})={initial_loss:.6f}  "
          f"final_loss(avg last {avg_window})={final_loss:.6f}  "
          f"(pass threshold: <{LOSS_PASS_THRESHOLD})")

    # The real check: DDIM-sample from the overfit model and compare to GT
    # in unnormalized physical units. Loss alone can't catch e.g. a
    # flipped unnormalize -- a low loss with garbage sampled actions means
    # something downstream of the loss (sampling, unnormalize) is broken.
    model.eval()
    num_inference_steps = args.num_inference_steps or cfg.diffusion.num_inference_steps
    gen = torch.Generator(device=device).manual_seed(args.seed)
    action_n_pred = ddim_sample(
        model, state, task_idx, cfg.data.horizon, cfg.model.action_dim,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
        num_inference_steps=num_inference_steps,
        device=device,
        generator=gen,
    )

    pred = unnormalize_action(action_n_pred.cpu().numpy(), norm_stats)
    gt = unnormalize_action(action.cpu().numpy(), norm_stats)
    valid = ~is_pad.cpu().numpy()

    pos_err_mm = np.linalg.norm(pred[..., ACTION_XYZ_SLICE] - gt[..., ACTION_XYZ_SLICE], axis=-1) * 1000.0
    # Euler XYZ extrinsic, not rotvec -- see rotation_convention_audit.py
    # and inference_utils.geodesic_angle_deg's docstring.
    ang_err_deg = geodesic_angle_deg_euler_xyz(pred[..., ACTION_ROT_SLICE], gt[..., ACTION_ROT_SLICE])
    gripper_err = np.abs(pred[..., ACTION_GRIPPER_IDX] - gt[..., ACTION_GRIPPER_IDX])

    print(f"\n[sanity_check] DDIM-sampled vs GT on the {len(overfit_ds)} overfit samples "
          f"({num_inference_steps} inference steps), unnormalized units, valid steps only:")
    print(f"  position error (mm):  mean={pos_err_mm[valid].mean():.3f}  "
          f"median={np.median(pos_err_mm[valid]):.3f}  max={pos_err_mm[valid].max():.3f}")
    print(f"  angle error (deg):    mean={ang_err_deg[valid].mean():.3f}  "
          f"median={np.median(ang_err_deg[valid]):.3f}  max={ang_err_deg[valid].max():.3f}")
    print(f"  gripper error (raw):  mean={gripper_err[valid].mean():.5f}  "
          f"median={np.median(gripper_err[valid]):.5f}  max={gripper_err[valid].max():.5f}")

    n_print = min(args.print_samples, len(overfit_ds))
    print(f"\n[sanity_check] per-dimension pred vs GT for the first {n_print} sample(s), "
          f"non-padding steps only (columns = {', '.join(ACTION_NAMES)}):")
    for i in range(n_print):
        part, ep, t0 = full_train_ds.samples[indices[i]]
        print(f"\n  --- sample {i}: part={part} episode={ep} start_frame={t0} ---")
        header = "  t  " + "".join(f"{n:>16}" for n in ACTION_NAMES) + "   (pred/gt)"
        print(header)
        for t in range(cfg.data.horizon):
            if not valid[i, t]:
                continue
            cells = "".join(f"{p:8.3f}/{g:<7.3f}" for p, g in zip(pred[i, t], gt[i, t]))
            print(f"  {t:2d}  {cells}")

    print()
    any_nan = bool(np.isnan(pred).any() or np.isnan(gt).any() or np.isinf(pred).any())
    loss_ok = final_loss < LOSS_PASS_THRESHOLD

    if any_nan or not loss_ok:
        print(f"[sanity_check] FAIL: final_loss={final_loss:.6f} (threshold <{LOSS_PASS_THRESHOLD})  "
              f"nan_or_inf={any_nan} -- for --n-samples={args.n_samples} this is a real pipeline/model bug, "
              "not slow convergence (a memorized sample has no theoretical loss floor above 0). "
              "Inspect the per-dimension table above.")
        sys.exit(1)

    print(f"[sanity_check] PASS: final_loss={final_loss:.6f} < {LOSS_PASS_THRESHOLD}, no NaN/Inf. "
          f"pos_err_mean={pos_err_mm[valid].mean():.2f}mm  ang_err_mean={ang_err_deg[valid].mean():.2f}deg -- "
          "still eyeball the per-dimension table above.")


if __name__ == "__main__":
    main()
