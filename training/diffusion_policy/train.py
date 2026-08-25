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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ExperimentConfig, REPO_ROOT, resolve_repo_path  # noqa: E402
from constants import GROUP_ORDER, PART_TO_IDX  # noqa: E402
from dataset import PartSequenceDataset  # noqa: E402
from grouped_model import GroupedDiffusionPolicyNet  # noqa: E402
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


class MetricsLogger:
    """Appends one JSON record per line to ckpt_dir/metrics.jsonl -- a
    structured, durable counterpart to the [train] print statements (which
    are only ever captured if stdout happens to be redirected to a file).
    See plot_loss.py for the reader. Record kinds, distinguished by which
    keys are present: {"step", "epoch", "train_loss", "lr"} on every
    log_every_steps print, {"epoch", "val_loss"} on every val_every_epochs
    evaluation."""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "a", buffering=1)  # line-buffered: readable mid-run, not just after close

    def log(self, **fields) -> None:
        self._f.write(json.dumps(fields) + "\n")

    def close(self) -> None:
        self._f.close()


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
        parts=list(cfg.data.resolved_parts()),
        horizon=cfg.data.horizon,
        norm_stats=norm_stats,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        rotation_repr=cfg.model.rotation_repr,
        load_images=cfg.model.use_vision,
        camera_keys=cfg.model.camera_keys,
        image_resize_hw=cfg.data.image_resize_hw,
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


def _extract_images(batch: dict, camera_keys, device) -> Optional[dict]:
    """batch -> {camera_key: (B,3,H,W) tensor on device}, or None if the
    batch has no image_* keys (use_vision=False -- dataset.py only adds
    them when built with load_images=True, see PartSequenceDataset)."""
    if not camera_keys or f"image_{camera_keys[0]}" not in batch:
        return None
    return {cam: batch[f"image_{cam}"].to(device) for cam in camera_keys}


def train(cfg: ExperimentConfig, resume_from: Optional[str] = None) -> Path:
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

    # Grouped runs store one entry per resolved part (the right-arm value is
    # per-part, unlike everything else this loop treats as a single fixed
    # value) -- the training loop itself never reads this either way, it's
    # only threaded through to save_checkpoint for execution-time 14-D
    # action reconstruction (see check_deploy_consistency.py's own
    # STRUCTURAL WARNING on whether the adapter actually uses it today).
    right_arm_path = resolve_repo_path(cfg.train.right_arm_constants_path)
    all_right_arm = json.loads(right_arm_path.read_text()) if right_arm_path.exists() else {}
    right_arm_entries = {}
    for _part in cfg.data.resolved_parts():
        entry = all_right_arm.get(_part)
        if entry is None:
            print(
                f"[train] WARNING: no right-arm constant found for part={_part!r} "
                f"in {right_arm_path}. Run precheck_right_arm.py first. Execution-time code "
                "will not be able to reconstruct the full 14-D action from this checkpoint."
            )
        elif not entry["all_below_threshold"]:
            print(
                f"[train] WARNING: right-arm action for part={_part!r} is NOT constant "
                f"(max_std={entry['max_std']:.6f}); storing the mean as a fallback "
                "anyway, per precheck_right_arm.py's report-only policy."
            )
        right_arm_entries[_part] = entry
    # Single-part runs also keep the old singular field so nothing reading
    # ckpt["right_arm_constant"] (evaluate.py, dp_server_stateonly.py,
    # check_deploy_consistency.py) has to branch on grouped vs not.
    right_arm_entry = right_arm_entries[cfg.data.part] if cfg.data.group is None else None

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(cfg, norm_stats)
    run_label = cfg.data.group if cfg.data.group is not None else cfg.data.part
    print(f"[train] run={run_label} parts={cfg.data.resolved_parts()} "
          f"train_samples={len(train_ds)}  val_samples={len(val_ds)}")
    if train_ds.rotvec_jump_warnings:
        print(f"[train] {len(train_ds.rotvec_jump_warnings)} rotvec-jump warning(s) in train split: "
              f"{train_ds.rotvec_jump_warnings}")

    vision_image_hw = cfg.data.image_resize_hw or (240, 320)
    if cfg.data.group is not None:
        model = GroupedDiffusionPolicyNet(cfg.model, vision_image_hw=vision_image_hw).to(device)
    else:
        model = DiffusionPolicyNet(cfg.model, vision_image_hw=vision_image_hw).to(device)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.use_ema else None

    start_epoch = 0
    if resume_from is not None:
        resume_path = resolve_repo_path(resume_from)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_ckpt.get("group") != cfg.data.group or (
            cfg.data.group is None and resume_ckpt["part"] != cfg.data.part
        ):
            raise ValueError(
                f"--resume-from {resume_path} was trained for "
                f"group={resume_ckpt.get('group')!r} part={resume_ckpt.get('part')!r}, "
                f"not group={cfg.data.group!r} part={cfg.data.part!r}"
            )
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if ema is not None and resume_ckpt.get("ema_state_dict") is not None:
            ema.shadow.load_state_dict(resume_ckpt["ema_state_dict"])
        start_epoch = resume_ckpt["epoch"]
        print(f"[train] resumed from {resume_path} at epoch={start_epoch} "
              f"(model + EMA weights restored; optimizer state is NOT -- AdamW momentum restarts "
              "from zero, and the cosine LR schedule is fast-forwarded by step count, not restored "
              "from a saved scheduler state)")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    n_steps = cfg.train.num_epochs * max(len(train_loader), 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_steps, 1))
    start_step = start_epoch * max(len(train_loader), 1)
    for _ in range(start_step):  # fast-forward to where the LR schedule left off
        lr_scheduler.step()

    ckpt_dir = resolve_repo_path(cfg.train.ckpt_dir) / run_label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(ckpt_dir / "config.json")
    metrics = MetricsLogger(ckpt_dir / "metrics.jsonl")

    global_step = start_step
    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            is_pad = batch["action_is_pad"].to(device)
            task_idx = batch["task_idx"].to(device)
            images = _extract_images(batch, cfg.model.camera_keys, device)

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device,
            ).long()
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            eps_pred = model(noisy_action, timesteps, state, task_idx, images)
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
                print(f"[train] run={run_label} epoch={epoch} step={global_step} "
                      f"loss={loss.item():.5f} lr={lr_scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.1f}s")
                metrics.log(step=global_step, epoch=epoch, train_loss=loss.item(),
                            lr=lr_scheduler.get_last_lr()[0])

        mean_loss = epoch_loss / max(n_batches, 1)
        print(f"[train] === epoch {epoch} done: mean_train_loss={mean_loss:.5f} ===")
        metrics.log(epoch=epoch, mean_train_loss=mean_loss)

        if (epoch + 1) % cfg.train.val_every_epochs == 0 or epoch == cfg.train.num_epochs - 1:
            val_loss = evaluate(model, val_loader, noise_scheduler, device, seed=cfg.train.seed,
                                 camera_keys=cfg.model.camera_keys)
            print(f"[train] === epoch {epoch} val_loss={val_loss:.5f} ===")
            metrics.log(epoch=epoch, val_loss=val_loss)

        if (epoch + 1) % cfg.train.ckpt_every_epochs == 0 or epoch == cfg.train.num_epochs - 1:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1:04d}.pt", model, ema, cfg, norm_stats,
                             right_arm_entry, right_arm_entries, epoch + 1)

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, model, ema, cfg, norm_stats, right_arm_entry, right_arm_entries,
                     cfg.train.num_epochs)
    print(f"[train] wrote {final_path}")
    metrics.close()
    return final_path


@torch.no_grad()
def evaluate(model, val_loader, noise_scheduler, device, seed: int, camera_keys=()) -> float:
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
        images = _extract_images(batch, camera_keys, device)

        noise = torch.randn(action.shape, generator=g).to(device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), generator=g,
        ).to(device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
        eps_pred = model(noisy_action, timesteps, state, task_idx, images)
        loss = masked_mse(eps_pred, noise, is_pad)
        total += loss.item() * action.shape[0]
        n += action.shape[0]
    model.train()
    return total / max(n, 1)


def save_checkpoint(path: Path, model, ema: Optional[EMA], cfg: ExperimentConfig,
                     norm_stats: NormStats, right_arm_entry: Optional[dict],
                     right_arm_entries: dict, epoch: int) -> None:
    payload = {
        "epoch": epoch,
        # "part"/"right_arm_constant" (singular) stay populated exactly as
        # before for single-part runs (group=None) so every existing reader
        # (evaluate.py, dp_server_stateonly.py, check_deploy_consistency.py)
        # needs no change. Grouped runs leave both None and rely on the new
        # plural fields instead -- readers that don't yet understand
        # "group"/"parts" should fail loudly on a missing/None "part", not
        # silently misinterpret a multi-part checkpoint as single-part.
        "part": cfg.data.part if cfg.data.group is None else None,
        "group": cfg.data.group,
        "parts": list(cfg.data.resolved_parts()),
        "part_to_idx": PART_TO_IDX,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow.state_dict() if ema is not None else None,
        "config": cfg.to_dict(),
        "norm_stats": norm_stats.to_dict(),
        "right_arm_constant": right_arm_entry,
        "right_arm_constants": right_arm_entries,
    }
    torch.save(payload, path)


def parse_args() -> Tuple[ExperimentConfig, Optional[str]]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=str, default=None, choices=list(PART_TO_IDX.keys()),
                     help="single-part stage-1 run (mutually exclusive with --group)")
    ap.add_argument("--group", type=str, default=None, choices=list(GROUP_ORDER) + ["all"],
                     help="grouped multi-part run via grouped_model.GroupedDiffusionPolicyNet: "
                          "one of constants.GROUP_ORDER (that group's parts only) or 'all' (every "
                          "part, all 4 groups jointly -- the real multi-head run). Mutually "
                          "exclusive with --part.")
    ap.add_argument("--config", type=str, default=None, help="path to a saved ExperimentConfig json")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rotation-repr", type=str, default=None, choices=["rotvec", "rot6d"])
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--use-vision", action="store_true", help="add head+left_hand camera conditioning (see vision.py)")
    ap.add_argument("--image-resize", type=str, default=None,
                     help="decode-time image resize as 'H,W' (default: native 240x320 -- several GB RAM/part, see dataset.py)")
    ap.add_argument("--num-workers", type=int, default=None,
                     help="DataLoader worker processes (default: 4, or 0 when --use-vision is set "
                          "and --num-workers isn't -- Windows' spawn-based multiprocessing has to "
                          "pickle the WHOLE dataset (incl. the multi-GB in-memory image cache) to "
                          "every worker, which is slow at best and crashes at worst; images are "
                          "already fully preloaded so __getitem__ is cheap indexing anyway, workers "
                          "buy little here)")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                     help="override output dir (default: training/diffusion_policy/outputs, or "
                          ".../outputs_vision when --use-vision is set and --ckpt-dir isn't -- "
                          "kept separate so a vision run never overwrites a state-only checkpoint "
                          "for the same part, or vice versa)")
    ap.add_argument("--resume-from", type=str, default=None,
                     help="path to a .pt checkpoint (e.g. outputs_vision/hdmi/epoch_0100.pt) to "
                          "resume model+EMA weights from; training continues at that checkpoint's "
                          "epoch count through --num-epochs. Optimizer state is NOT restored (see "
                          "train()'s resume_from branch) -- Adam momentum restarts, LR schedule "
                          "position is recomputed to match, not reloaded.")
    args = ap.parse_args()
    if args.part is not None and args.group is not None:
        ap.error("--part and --group are mutually exclusive")
    if args.part is None and args.group is None:
        ap.error("one of --part or --group is required")

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    if args.group is not None:
        cfg.data.group = args.group
    else:
        cfg.data.part = args.part
        cfg.data.group = None
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
    if args.use_vision:
        cfg.model.use_vision = True
    if cfg.data.group is not None and not cfg.model.use_vision:
        ap.error("--group requires --use-vision -- GroupedDiffusionPolicyNet has no state-only "
                 "mode (see its __init__ docstring: it exists specifically to give the shared "
                 "trunk object-position grounding through vision)")
    if args.image_resize is not None:
        h, w = (int(x) for x in args.image_resize.split(","))
        cfg.data.image_resize_hw = (h, w)
    if args.ckpt_dir is not None:
        cfg.train.ckpt_dir = args.ckpt_dir
    elif cfg.data.group is not None:
        # Separate from outputs_vision: a grouped checkpoint's payload shape
        # (part=None, group=..., parts=[...], multi-head state_dict) is not
        # interchangeable with a single-part outputs_vision/<part>/final.pt,
        # so keeping them in different directories makes that obvious rather
        # than inviting evaluate.py/dp_server_vision.py to load one as if it
        # were the other.
        cfg.train.ckpt_dir = "training/diffusion_policy/outputs_grouped"
    elif cfg.model.use_vision:
        cfg.train.ckpt_dir = "training/diffusion_policy/outputs_vision"
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    elif cfg.model.use_vision:
        cfg.train.num_workers = 0
    cfg.data.__post_init__()
    cfg.model.__post_init__()
    return cfg, args.resume_from


if __name__ == "__main__":
    _cfg, _resume_from = parse_args()
    train(_cfg, resume_from=_resume_from)
