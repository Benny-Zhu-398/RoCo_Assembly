"""Train a state-only Diffusion Policy for ONE part (stage-1 curriculum).

Part is a config field, not a hardcoded value -- run this 9 times (once
per PART_ORDER entry) to cover the whole stage-1 curriculum:

    python train.py --part gear_20teeth
    python train.py --part bolt_8mm
    ...

Prerequisites (run once, outputs are committed/shared, not per-part):
    python precheck_right_arm.py     # -> right_arm_constants.json
    python compute_norm_stats.py     # -> norm_stats.json

This module intentionally does NOT depend on `lerobot` or Isaac Sim -- only
torch, diffusers, numpy, pandas, pyarrow (see requirements.txt). Run it in
its own venv, same pattern as task/dp_server.py's sidecar process.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ExperimentConfig, REPO_ROOT, resolve_repo_path  # noqa: E402
from constants import PART_TO_IDX  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from model import DiffusionPolicyNet  # noqa: E402
from normalization import NormStats  # noqa: E402


class EMA:
    """Minimal exponential moving average of model weights (standard part
    of the Diffusion Policy training recipe -- stabilizes the sampling-time
    model relative to the raw SGD iterate)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(buf)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """pred/target: (B, horizon, action_dim). is_pad: (B, horizon) bool,
    True where the action-chunk step is padding (past the episode end) and
    must not contribute to the loss."""
    err = F.mse_loss(pred, target, reduction="none")   # (B, horizon, action_dim)
    valid = (~is_pad).unsqueeze(-1).to(err.dtype)        # (B, horizon, 1)
    return (err * valid).sum() / valid.sum().clamp_min(1.0) / err.shape[-1]


def build_dataloaders(cfg: ExperimentConfig, norm_stats: NormStats):
    dataset_root = resolve_repo_path(cfg.data.dataset_root)
    common = dict(
        dataset_root=dataset_root,
        parts=cfg.data.part,
        horizon=cfg.data.horizon,
        norm_stats=norm_stats,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        rotation_repr=cfg.model.rotation_repr,
    )
    train_ds = PartSequenceDataset(split="train", **common)
    val_ds = PartSequenceDataset(split="val", **common)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    return train_ds, val_ds, train_loader, val_loader


def train(cfg: ExperimentConfig) -> Path:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    if device.type == "cpu" and cfg.train.device != "cpu":
        print(f"[train] CUDA not available, falling back to CPU (requested {cfg.train.device!r})")

    norm_stats_path = resolve_repo_path(cfg.train.norm_stats_path)
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"{norm_stats_path} not found. Run compute_norm_stats.py first -- "
            "norm stats must be pooled across all 9 parts, not computed per-part."
        )
    norm_stats = NormStats.load(norm_stats_path)

    right_arm_path = resolve_repo_path(cfg.train.right_arm_constants_path)
    right_arm_entry = None
    if right_arm_path.exists():
        all_right_arm = json.loads(right_arm_path.read_text())
        right_arm_entry = all_right_arm.get(cfg.data.part)
    if right_arm_entry is None:
        print(
            f"[train] WARNING: no right-arm constant found for part={cfg.data.part!r} "
            f"in {right_arm_path}. Run precheck_right_arm.py first. Execution-time code "
            "will not be able to reconstruct the full 14-D action from this checkpoint."
        )
    elif not right_arm_entry["all_below_threshold"]:
        print(
            f"[train] WARNING: right-arm action for part={cfg.data.part!r} is NOT constant "
            f"(max_std={right_arm_entry['max_std']:.6f}); storing the mean as a fallback "
            "anyway, per precheck_right_arm.py's report-only policy."
        )

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg, norm_stats)
    print(f"[train] part={cfg.data.part}  train_samples={len(train_ds)}  val_samples={len(val_ds)}")
    if train_ds.rotvec_jump_warnings:
        print(f"[train] {len(train_ds.rotvec_jump_warnings)} rotvec-jump warning(s) in train split: "
              f"{train_ds.rotvec_jump_warnings}")

    model = DiffusionPolicyNet(cfg.model).to(device)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.use_ema else None

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    n_steps = cfg.train.num_epochs * max(len(train_loader), 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_steps, 1))

    ckpt_dir = resolve_repo_path(cfg.train.ckpt_dir) / cfg.data.part
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            is_pad = batch["action_is_pad"].to(device)
            task_idx = batch["task_idx"].to(device)

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device,
            ).long()
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            eps_pred = model(noisy_action, timesteps, state, task_idx)
            loss = masked_mse(eps_pred, noise, is_pad)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            lr_scheduler.step()
            if ema is not None and global_step % cfg.train.ema_update_every == 0:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % cfg.train.log_every_steps == 0:
                print(f"[train] part={cfg.data.part} epoch={epoch} step={global_step} "
                      f"loss={loss.item():.5f} lr={lr_scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.1f}s")

        mean_loss = epoch_loss / max(n_batches, 1)
        print(f"[train] === epoch {epoch} done: mean_train_loss={mean_loss:.5f} ===")

        if (epoch + 1) % cfg.train.val_every_epochs == 0 or epoch == cfg.train.num_epochs - 1:
            val_loss = evaluate(model, val_loader, noise_scheduler, device, seed=cfg.train.seed)
            print(f"[train] === epoch {epoch} val_loss={val_loss:.5f} ===")

        if (epoch + 1) % cfg.train.ckpt_every_epochs == 0 or epoch == cfg.train.num_epochs - 1:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1:04d}.pt", model, ema, cfg, norm_stats, right_arm_entry, epoch + 1)

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, model, ema, cfg, norm_stats, right_arm_entry, cfg.train.num_epochs)
    print(f"[train] wrote {final_path}")
    return final_path


@torch.no_grad()
def evaluate(model, val_loader, noise_scheduler, device, seed: int) -> float:
    if len(val_loader.dataset) == 0:
        return float("nan")
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    total, n = 0.0, 0
    for batch in val_loader:
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        is_pad = batch["action_is_pad"].to(device)
        task_idx = batch["task_idx"].to(device)

        noise = torch.randn(action.shape, generator=g).to(device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), generator=g,
        ).to(device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
        eps_pred = model(noisy_action, timesteps, state, task_idx)
        loss = masked_mse(eps_pred, noise, is_pad)
        total += loss.item() * action.shape[0]
        n += action.shape[0]
    model.train()
    return total / max(n, 1)


def save_checkpoint(path: Path, model, ema: Optional[EMA], cfg: ExperimentConfig,
                     norm_stats: NormStats, right_arm_entry: Optional[dict], epoch: int) -> None:
    payload = {
        "epoch": epoch,
        "part": cfg.data.part,
        "part_to_idx": PART_TO_IDX,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow.state_dict() if ema is not None else None,
        "config": cfg.to_dict(),
        "norm_stats": norm_stats.to_dict(),
        "right_arm_constant": right_arm_entry,
    }
    torch.save(payload, path)


def parse_args() -> ExperimentConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=str, required=True, choices=list(PART_TO_IDX.keys()))
    ap.add_argument("--config", type=str, default=None, help="path to a saved ExperimentConfig json")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rotation-repr", type=str, default=None, choices=["rotvec", "rot6d"])
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    cfg.data.part = args.part
    if args.horizon is not None:
        cfg.data.horizon = args.horizon
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_epochs is not None:
        cfg.train.num_epochs = args.num_epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.device is not None:
        cfg.train.device = args.device
    if args.rotation_repr is not None:
        cfg.model.rotation_repr = args.rotation_repr
    if args.no_ema:
        cfg.train.use_ema = False
    cfg.data.__post_init__()
    cfg.model.__post_init__()
    return cfg


if __name__ == "__main__":
    train(parse_args())
