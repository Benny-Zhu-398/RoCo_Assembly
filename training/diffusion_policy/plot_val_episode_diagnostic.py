"""Diagnostic plot for one battery_size1 val episode: GT action (commanded)
vs. recorded state (actually reached) left-arm ee_xyz, control lag
||action_xyz[t] - state_xyz[t+1]|| (same definition as speed_error_analysis.py
Q2), and gripper open/close, with the grasp frame marked.

Usage:
    python plot_val_episode_diagnostic.py --part battery_size1 --episode 61 \
        --out val_episode_battery_size1_ep61_diagnostic.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from constants import (  # noqa: E402
    ACTION_GRIPPER_IDX,
    ACTION_XYZ_SLICE,
    DATASET_FPS,
    LEFT_ACTION_IDX,
    LEFT_STATE_IDX,
    STATE_GRIPPER_IDX,
    STATE_XYZ_SLICE,
)
from data_io import episode_frames, load_data_table  # noqa: E402

# dataviz palette (references/palette.md): slot 1 blue, slot 2 orange, slot 3 aqua
COLOR_X = "#2a78d6"
COLOR_Y = "#eb6834"
COLOR_Z = "#1baf7a"
COLOR_LAG = "#2a78d6"
COLOR_GRASP = "#e34948"  # slot 8 red, status/marker use
AXIS_MUTED = "#898781"
GRID = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=str, default="battery_size1")
    ap.add_argument("--dataset-root", type=str,
                     default=str(Path(__file__).resolve().parents[2] / "tools" / "roco2026_by_part"))
    ap.add_argument("--episode", type=int, default=61, help="episode_index (must be in the val split)")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def find_grasp_frame(action_gripper: np.ndarray) -> int:
    """First frame where the commanded gripper value moves > 10% of its own
    episode range away from its opening value -- i.e. the first close command."""
    g0 = action_gripper[0]
    rng = action_gripper.max() - action_gripper.min()
    if rng < 1e-6:
        return 0
    thresh = 0.1 * rng
    moved = np.where(np.abs(action_gripper - g0) > thresh)[0]
    return int(moved[0]) if len(moved) else 0


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_path = Path(args.out) if args.out else (
        _THIS_DIR / f"val_episode_{args.part}_ep{args.episode}_diagnostic.png"
    )

    data_df = load_data_table(dataset_root, columns=["observation.state", "action", "task_index"])
    state_full, action_full = episode_frames(data_df, args.episode)  # (L,44), (L,14)
    state = state_full[:, LEFT_STATE_IDX]    # (L,22)
    action = action_full[:, LEFT_ACTION_IDX]  # (L,7)

    L = len(state)
    frames = np.arange(L)

    action_xyz = action[:, ACTION_XYZ_SLICE] * 1000.0  # m -> mm
    state_xyz = state[:, STATE_XYZ_SLICE] * 1000.0
    action_grip = action[:, ACTION_GRIPPER_IDX]
    state_grip = state[:, STATE_GRIPPER_IDX]

    # control lag ||action_xyz[t] - state_xyz[t+1]||, defined for t = 0..L-2
    lag_frames = frames[:-1]
    lag_mm = np.linalg.norm(action_xyz[:-1] - state_xyz[1:], axis=-1)

    grasp_frame = find_grasp_frame(action_grip)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1, 0.8]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS_MUTED)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # --- top: ee_xyz, action (dashed) vs state (solid) ---
    ax0 = axes[0]
    labels_xyz = ["x", "y", "z"]
    colors_xyz = [COLOR_X, COLOR_Y, COLOR_Z]
    for i, (lab, col) in enumerate(zip(labels_xyz, colors_xyz)):
        ax0.plot(frames, action_xyz[:, i], color=col, linestyle="--", linewidth=1.6,
                  label=f"action {lab}" if i == 0 else None, alpha=0.9)
        ax0.plot(frames, state_xyz[:, i], color=col, linestyle="-", linewidth=1.8,
                  label=f"state {lab}" if i == 0 else None)
    # per-component legend proxies (color = axis, linestyle = action/state)
    from matplotlib.lines import Line2D
    comp_handles = [Line2D([0], [0], color=c, lw=2, label=f"{l}") for l, c in zip(labels_xyz, colors_xyz)]
    style_handles = [
        Line2D([0], [0], color=TEXT_PRIMARY, lw=1.8, linestyle="-", label="state (actual)"),
        Line2D([0], [0], color=TEXT_PRIMARY, lw=1.6, linestyle="--", label="GT action (command)"),
    ]
    leg1 = ax0.legend(handles=comp_handles, loc="upper left", frameon=False, fontsize=8, title="component")
    ax0.add_artist(leg1)
    ax0.legend(handles=style_handles, loc="upper right", frameon=False, fontsize=8)
    ax0.set_ylabel("left ee_xyz (mm)", color=TEXT_PRIMARY, fontsize=10)
    ax0.set_title(f"battery_size1 val episode {args.episode} -- diagnostic", color=TEXT_PRIMARY,
                   fontsize=12, loc="left", pad=10)

    # --- middle: control lag ---
    ax1 = axes[1]
    ax1.plot(lag_frames, lag_mm, color=COLOR_LAG, linewidth=1.8)
    ax1.set_ylabel("control lag (mm)\n||action[t]-state[t+1]||", color=TEXT_PRIMARY, fontsize=10)

    # --- bottom: gripper ---
    ax2 = axes[2]
    ax2.plot(frames, action_grip, color=TEXT_PRIMARY, linestyle="--", linewidth=1.6, label="GT action (command)")
    ax2.plot(frames, state_grip, color=TEXT_PRIMARY, linestyle="-", linewidth=1.8, label="state (actual)")
    ax2.set_ylabel("left gripper", color=TEXT_PRIMARY, fontsize=10)
    ax2.set_xlabel(f"frame (@ {DATASET_FPS:.0f} fps)", color=TEXT_PRIMARY, fontsize=10)
    ax2.legend(loc="best", frameon=False, fontsize=8)

    for ax in axes:
        ax.axvline(grasp_frame, color=COLOR_GRASP, linewidth=1.4, linestyle=":", alpha=0.9)
    axes[0].text(grasp_frame, axes[0].get_ylim()[1], f" grasp @ frame {grasp_frame}",
                  color=COLOR_GRASP, fontsize=9, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot_val_episode_diagnostic] episode={args.episode} L={L} grasp_frame={grasp_frame}")
    print(f"[plot_val_episode_diagnostic] wrote {out_path}")


if __name__ == "__main__":
    main()
