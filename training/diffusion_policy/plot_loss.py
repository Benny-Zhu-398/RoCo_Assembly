"""Plot a training run's loss curve from train.py's metrics.jsonl.

train.py (see its MetricsLogger) appends one JSON record per line to
<ckpt_dir>/metrics.jsonl while it runs -- this only READS that file, it
never touches the training process, so it's safe to run against a
still-running run (the file is opened line-buffered on the writer side
specifically so this works) as well as a finished one.

Usage:
    python plot_loss.py outputs_grouped/gears
    python plot_loss.py outputs/gear_20teeth --out loss.png
    python plot_loss.py outputs_grouped/all --smooth 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_metrics(ckpt_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """metrics.jsonl -> (step_records, epoch_mean_records, val_records),
    split by which keys each line has (see train.py::MetricsLogger's
    docstring for the three record kinds)."""
    path = ckpt_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- either this run predates the metrics.jsonl logger "
            "(train.py was updated 2026-08-18 to write this; older checkpoints only "
            "have the printed [train] stdout, which this script doesn't parse), or "
            "ckpt_dir is wrong."
        )
    step_records, epoch_records, val_records = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "train_loss" in rec:
                step_records.append(rec)
            elif "val_loss" in rec:
                val_records.append(rec)
            elif "mean_train_loss" in rec:
                epoch_records.append(rec)
    return step_records, epoch_records, val_records


def _moving_average(xs: List[float], window: int) -> List[float]:
    if window <= 1 or len(xs) < window:
        return xs
    out = []
    acc = 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot(ckpt_dir: Path, out_path: Path, smooth: int, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # headless-safe: no display required to write a PNG
    import matplotlib.pyplot as plt

    step_records, epoch_records, val_records = load_metrics(ckpt_dir)
    if not step_records and not epoch_records:
        raise RuntimeError(f"{ckpt_dir/'metrics.jsonl'} has no train-loss records to plot")

    fig, (ax_step, ax_epoch) = plt.subplots(2, 1, figsize=(9, 7))

    if step_records:
        steps = [r["step"] for r in step_records]
        losses = [r["train_loss"] for r in step_records]
        ax_step.plot(steps, losses, color="tab:blue", alpha=0.3, linewidth=0.8, label="train loss (raw)")
        if smooth > 1:
            ax_step.plot(steps, _moving_average(losses, smooth), color="tab:blue",
                         linewidth=1.6, label=f"train loss ({smooth}-step moving avg)")
        ax_step.set_xlabel("step")
        ax_step.set_ylabel("loss")
        ax_step.set_title("per-step train loss (masked MSE on predicted noise)")
        ax_step.legend()
        ax_step.grid(alpha=0.3)
    else:
        ax_step.text(0.5, 0.5, "no per-step records (log_every_steps never fired)",
                     ha="center", va="center", transform=ax_step.transAxes)

    if epoch_records:
        epochs = [r["epoch"] for r in epoch_records]
        mean_losses = [r["mean_train_loss"] for r in epoch_records]
        ax_epoch.plot(epochs, mean_losses, color="tab:blue", marker="o", label="mean train loss / epoch")
    if val_records:
        val_epochs = [r["epoch"] for r in val_records]
        val_losses = [r["val_loss"] for r in val_records]
        ax_epoch.plot(val_epochs, val_losses, color="tab:orange", marker="s", label="val loss")
    ax_epoch.set_xlabel("epoch")
    ax_epoch.set_ylabel("loss")
    ax_epoch.set_title("per-epoch train vs. val loss")
    ax_epoch.legend()
    ax_epoch.grid(alpha=0.3)

    fig.suptitle(f"{ckpt_dir}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot_loss] wrote {out_path}")
    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", type=str,
                     help="checkpoint directory containing metrics.jsonl, e.g. outputs_grouped/gears "
                          "or outputs/gear_20teeth (relative to this script's directory, or absolute)")
    ap.add_argument("--out", type=str, default=None,
                     help="output PNG path (default: <ckpt_dir>/loss_curve.png)")
    ap.add_argument("--smooth", type=int, default=20,
                     help="moving-average window (in steps) overlaid on the raw per-step curve, "
                          "which is noisy step-to-step because every step samples a fresh random "
                          "diffusion timestep -- 0 or 1 disables it (default: 20)")
    ap.add_argument("--show", action="store_true",
                     help="also open an interactive window (needs a display; default is save-only, "
                          "safe on a headless training box)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out) if args.out else ckpt_dir / "loss_curve.png"
    plot(ckpt_dir, out_path, smooth=args.smooth, show=args.show)
