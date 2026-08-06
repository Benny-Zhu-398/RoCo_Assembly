"""Shared checkpoint-loading / DDIM-sampling / geometry helpers for the
inference-side scripts (evaluate.py, sanity_check.py).

Kept separate from train.py so "load a checkpoint and sample from it"
has one implementation instead of being copy-pasted across scripts. No
Isaac Sim / lerobot dependency -- same constraint as the rest of this
package (see requirements.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import ExperimentConfig  # noqa: E402
from model import DiffusionPolicyNet  # noqa: E402
from normalization import NormStats  # noqa: E402


def load_checkpoint(ckpt_path: str | Path) -> dict:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(
    ckpt: dict, device: torch.device, use_ema: bool = True
) -> Tuple[DiffusionPolicyNet, ExperimentConfig]:
    """Rebuild the exact model architecture the checkpoint was trained with
    and load either the EMA shadow weights (default, standard DP inference
    recipe) or the raw optimizer-iterate weights."""
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    model = DiffusionPolicyNet(cfg.model).to(device)

    state_dict = None
    if use_ema:
        state_dict = ckpt.get("ema_state_dict")
        if state_dict is None:
            print("[inference_utils] WARNING: checkpoint has no EMA weights "
                  "(use_ema was False at train time); falling back to raw weights.")
    if state_dict is None:
        state_dict = ckpt["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def load_norm_stats_from_checkpoint(ckpt: dict) -> NormStats:
    """Never recompute norm stats for inference -- always the ones the
    checkpoint was actually trained/normalized with."""
    return NormStats.from_dict(ckpt["norm_stats"])


def part_to_idx_from_checkpoint(ckpt: dict) -> dict:
    part_order = ckpt["part_to_idx"]
    return {name: i for i, name in enumerate(part_order)}


@torch.no_grad()
def ddim_sample(
    model: DiffusionPolicyNet,
    state: torch.Tensor,
    task_idx: torch.Tensor,
    horizon: int,
    action_dim: int,
    num_train_timesteps: int,
    beta_schedule: str,
    prediction_type: str,
    clip_sample: bool,
    num_inference_steps: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Deterministic (eta=0) DDIM sampling. Returns normalized actions,
    shape (B, horizon, action_dim) -- caller unnormalizes.

    Ported from real-stanford/diffusion_policy's
    DiffusionUnetLowdimPolicy.conditional_sample (fetched 2026-07-29),
    specialized to our obs_as_global_cond=True case: the reference's
    conditional_sample also supports local_cond and impainting-style
    conditioning via a (condition_data, condition_mask) pair, but for
    global_cond mode that mask is always all-False (a no-op), so that
    machinery is dropped here rather than kept as permanently-dead code.
    global_cond is computed ONCE up front (state/task_idx don't change
    across denoising steps), then the bare unet is called directly each
    step -- calling the DiffusionPolicyNet wrapper per-step would
    needlessly recompute state_encoder/task_encoder every iteration.

    timestep_spacing="trailing" (not diffusers' default "leading"): with
    "leading" and num_inference_steps that doesn't evenly divide
    num_train_timesteps (16 into 100), the highest timestep actually used
    is floor(100/16)*15=90, not 99 -- but alpha_bar_90 ~= 0.02
    (sqrt(alpha_bar_90) ~= 0.14), so a real x_90 still carries ~14% signal
    magnitude. The initial `sample` here is pure torch.randn (0% signal,
    representing t~99), so labeling it "t=90" is a genuine train/inference
    mismatch at the single most-influential step of the reverse process.
    "trailing" spacing makes the sequence actually start at t=99, matching
    what `sample` is initialized to represent. (This is our own addition,
    not present in the reference, which leaves scheduler spacing to
    whatever the injected noise_scheduler config specifies.)
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        clip_sample=clip_sample,
        timestep_spacing="trailing",
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    B = state.shape[0]
    global_cond = model.global_cond(state, task_idx)
    trajectory = torch.randn((B, horizon, action_dim), generator=generator, device=device)

    for t in scheduler.timesteps:
        model_output = model.unet(trajectory, t, global_cond=global_cond)
        trajectory = scheduler.step(model_output, t, trajectory, eta=0.0, generator=generator).prev_sample

    return trajectory


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix via Rodrigues'
    formula, implemented directly in numpy rather than via
    scipy.spatial.transform.Rotation: this venv's installed scipy build
    requires numpy>=2 while requirements.txt targets plain `numpy` (and
    the sibling Isaac Sim project pins numpy<2), so scipy.spatial.transform
    is not reliably importable here (rotation_utils.py's scipy-based
    helpers hit the same wall but are only used by the unexercised
    rotation_repr="rot6d" path, so this was previously latent). Valid for
    any rotation magnitude, not just [0, pi] -- this dataset's rotvec
    action encoding is not confined to that range (see
    rotation_utils.find_rotvec_jumps).
    """
    v = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(v, axis=-1)  # (N,)
    theta_safe = np.where(theta > 1e-12, theta, 1.0)
    axis = v / theta_safe[:, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    K = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(-1, 3, 3)
    K2 = np.einsum("nij,njk->nik", K, K)
    eye = np.eye(3)[None, :, :]
    sin_t = np.sin(theta)[:, None, None]
    cos_t = np.cos(theta)[:, None, None]
    R = eye + sin_t * K + (1.0 - cos_t) * K2  # sin/cos -> 0 as theta -> 0, so this is safe there too
    return R.reshape(rotvec.shape[:-1] + (3, 3))


def geodesic_angle_deg(rotvec_pred: np.ndarray, rotvec_gt: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the relative rotation R_pred^T @ R_gt, i.e. the
    true geodesic distance on SO(3) -- NOT the same as elementwise rotvec
    MSE, which diverges from this near the pi-magnitude wraparound that is
    known to occur in this dataset's action encoding (see
    rotation_utils.find_rotvec_jumps).

    Inputs are interpreted as ROTATION VECTORS (axis-angle). Only valid for
    action data actually encoded that way -- confirmed true for the
    self-collected collect_lerobot_v3.py/v4.py datasets (metadata:
    "absolute_cartesian_target_xyz_rotvec_gripper"), confirmed FALSE for
    tools/roco2026_by_part (see rotation_convention_audit.py: its action
    rotation dims are Euler XYZ extrinsic, not rotvec). Use
    `geodesic_angle_deg_euler_xyz` below for anything derived from
    PartSequenceDataset (evaluate.py, sanity_check.py) -- this function is
    for the true-rotvec datasets only.

    rotvec_pred / rotvec_gt: (..., 3). Returns (...,) in degrees.
    """
    shape = rotvec_pred.shape[:-1]
    Rp = _rotvec_to_matrix(rotvec_pred).reshape(-1, 3, 3)
    Rg = _rotvec_to_matrix(rotvec_gt).reshape(-1, 3, 3)
    R_rel = np.einsum("nij,njk->nik", Rp.transpose(0, 2, 1), Rg)
    trace = np.einsum("nii->n", R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad).reshape(shape)


def _euler_xyz_extrinsic_to_matrix(euler: np.ndarray) -> np.ndarray:
    """(..., 3) [rx, ry, rz] Euler-XYZ EXTRINSIC angles -> (..., 3, 3),
    i.e. R = Rz(rz) @ Ry(ry) @ Rx(rx) (scipy's lowercase 'xyz' convention).

    This is the convention tools/roco2026_by_part's action rotation dims
    actually use -- empirically confirmed in rotation_convention_audit.py
    by comparing against same-frame state quaternions: pooled across all 9
    parts, decoding this way gives median/mean/p90 geodesic error of
    ~0.6/4.0/3.4 deg vs. ~1.0/16.1/60.9 deg for the axis-angle (rotvec)
    decode this metric used before, and ~1.0/22.3/103.5 deg for the
    intrinsic 'XYZ' variant. Do not decode roco2026_by_part's action
    rotation dims any other way.
    """
    e = np.asarray(euler, dtype=np.float64).reshape(-1, 3)
    zeros, ones = np.zeros(e.shape[0]), np.ones(e.shape[0])

    def axis_mat(axis: str, a: np.ndarray) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        if axis == "x":
            m = np.stack([ones, zeros, zeros, zeros, c, -s, zeros, s, c], axis=-1)
        elif axis == "y":
            m = np.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], axis=-1)
        else:
            m = np.stack([c, -s, zeros, s, c, zeros, zeros, zeros, ones], axis=-1)
        return m.reshape(-1, 3, 3)

    Rx, Ry, Rz = axis_mat("x", e[:, 0]), axis_mat("y", e[:, 1]), axis_mat("z", e[:, 2])
    R = np.einsum("nij,njk->nik", Rz, Ry)
    R = np.einsum("nij,njk->nik", R, Rx)
    return R.reshape(euler.shape[:-1] + (3, 3))


def geodesic_angle_deg_euler_xyz(euler_pred: np.ndarray, euler_gt: np.ndarray) -> np.ndarray:
    """Same geodesic-distance metric as `geodesic_angle_deg`, but interprets
    its inputs as Euler-XYZ EXTRINSIC angles instead of rotvec -- the
    correct convention for tools/roco2026_by_part's action rotation dims
    (see `_euler_xyz_extrinsic_to_matrix` and rotation_convention_audit.py).
    Use this, not `geodesic_angle_deg`, for any action-rotation error metric
    computed from a PartSequenceDataset-derived action.

    euler_pred / euler_gt: (..., 3). Returns (...,) in degrees.
    """
    shape = euler_pred.shape[:-1]
    Rp = _euler_xyz_extrinsic_to_matrix(euler_pred).reshape(-1, 3, 3)
    Rg = _euler_xyz_extrinsic_to_matrix(euler_gt).reshape(-1, 3, 3)
    R_rel = np.einsum("nij,njk->nik", Rp.transpose(0, 2, 1), Rg)
    trace = np.einsum("nii->n", R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad).reshape(shape)
