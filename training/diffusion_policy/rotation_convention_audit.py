"""Audit: is tools/roco2026_by_part's action rotation rotvec (axis-angle) or
Euler XYZ?

Motivation: task/policies/diffusion_stateonly.py's _rotvec_to_quat_wxyz
decodes the action's 3 rotation dims as axis-angle. If the source dataset
(HF `rocochallenge2025/rocochallenge2026_Industrial_Assembly`, sliced into
tools/roco2026_by_part by tools/segment_by_part.py) actually encodes them as
Euler XYZ angles, that decode is wrong -- most frames have small per-step
rotation deltas where axis-angle and Euler agree to first order (hence a
deceptively small median error), but larger reorientations diverge sharply
(reported: median ~1 deg, p90 ~61 deg), which is enough to make Lula IK
reject the target pose, freeze the arm, and time out the part.

This script settles it empirically rather than by inspecting metadata
strings: for every frame in every part, decode the LEFT action's rotation
dims both ways, convert to a rotation matrix, and compare (geodesic
distance in degrees) against the SAME FRAME's left state quaternion --
state and the action commanded at the same timestep are only ~1 control
step apart (100 ms at this dataset's 10 fps), so the correct decode should
track state closely almost everywhere; the wrong decode should not.

All math is plain numpy (no scipy): this training venv's installed scipy
build is unimportable against its numpy (`AttributeError: module 'numpy'
has no attribute 'long'`, hit while building check_deploy_consistency.py --
see inference_utils.py's own docstring on this same issue), so every
rotation conversion here is written out directly, mirroring
inference_utils._rotvec_to_matrix / geodesic_angle_deg's existing pattern
rather than depending on scipy.spatial.transform.

Usage:
    python rotation_convention_audit.py
    python rotation_convention_audit.py --parts battery_size1 gear_60teeth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from config import resolve_repo_path  # noqa: E402
from constants import ACTION_ROT_SLICE, LEFT_ACTION_IDX, LEFT_STATE_IDX, PART_ORDER  # noqa: E402
from data_io import episode_ids_for_part, episode_frames, load_data_table, load_episodes_table  # noqa: E402

# Raw 44-D state layout indices (constants.STATE_NAMES_FULL).
LEFT_STATE_QUAT = slice(3, 7)     # left_ee_qw,qx,qy,qz
RIGHT_STATE_QUAT = slice(10, 14)  # right_ee_qw,qx,qy,qz
# Raw 14-D action layout indices (constants.ACTION_NAMES_FULL).
LEFT_ACTION_ROT = slice(3, 6)     # left_ee_rx,ry,rz
RIGHT_ACTION_ROT = slice(10, 13)  # right_ee_rx,ry,rz


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3). Rodrigues' formula, valid for any
    magnitude (this dataset's rotvec-candidate values exceed pi -- see
    rotation_utils.py). Same formula as inference_utils._rotvec_to_matrix,
    duplicated here so this audit has no import-order dependency on it."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(v, axis=-1)
    theta_safe = np.where(theta > 1e-12, theta, 1.0)
    axis = v / theta_safe[:, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    K = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(-1, 3, 3)
    K2 = np.einsum("nij,njk->nik", K, K)
    eye = np.eye(3)[None, :, :]
    sin_t, cos_t = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    R = eye + sin_t * K + (1.0 - cos_t) * K2
    return R.reshape(rotvec.shape[:-1] + (3, 3))


def _axis_matrix(axis: str, angle: np.ndarray) -> np.ndarray:
    c, s, o, z = np.cos(angle), np.sin(angle), np.ones_like(angle), np.zeros_like(angle)
    if axis == "x":
        m = np.stack([o, z, z, z, c, -s, z, s, c], axis=-1)
    elif axis == "y":
        m = np.stack([c, z, s, z, o, z, -s, z, c], axis=-1)
    else:
        m = np.stack([c, -s, z, s, c, z, z, z, o], axis=-1)
    return m.reshape(-1, 3, 3)


def euler_xyz_to_matrix(euler: np.ndarray, intrinsic: bool) -> np.ndarray:
    """(..., 3) [rx, ry, rz] Euler-XYZ angles -> (..., 3, 3).

    intrinsic=True  -> scipy convention 'XYZ': R = Rx(rx) @ Ry(ry) @ Rz(rz)
    intrinsic=False -> scipy convention 'xyz': R = Rz(rz) @ Ry(ry) @ Rx(rx)
    Both variants are checked independently below since "Euler XYZ" alone
    doesn't disambiguate intrinsic vs extrinsic.
    """
    e = np.asarray(euler, dtype=np.float64).reshape(-1, 3)
    Rx, Ry, Rz = _axis_matrix("x", e[:, 0]), _axis_matrix("y", e[:, 1]), _axis_matrix("z", e[:, 2])
    if intrinsic:
        R = np.einsum("nij,njk->nik", Rx, Ry)
        R = np.einsum("nij,njk->nik", R, Rz)
    else:
        R = np.einsum("nij,njk->nik", Rz, Ry)
        R = np.einsum("nij,njk->nik", R, Rx)
    return R.reshape(euler.shape[:-1] + (3, 3))


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) unit quaternion wxyz -> (..., 3, 3)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)
    return R.reshape(quat_wxyz.shape[:-1] + (3, 3))


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """SO(3) geodesic distance (degrees) between matching rotation matrices."""
    Ra = Ra.reshape(-1, 3, 3)
    Rb = Rb.reshape(-1, 3, 3)
    R_rel = np.einsum("nij,njk->nik", Ra.transpose(0, 2, 1), Rb)
    trace = np.einsum("nii->n", R_rel)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def summarize(vals: np.ndarray) -> str:
    return (f"median={np.median(vals):7.3f}  mean={np.mean(vals):7.3f}  "
            f"p90={np.percentile(vals, 90):7.3f}  p99={np.percentile(vals, 99):7.3f}  max={np.max(vals):7.3f}")


def audit_part(data_df, episodes_df, part: str):
    ep_ids = episode_ids_for_part(episodes_df, part)
    all_state, all_action = [], []
    for ep in ep_ids:
        state, action = episode_frames(data_df, ep)
        all_state.append(state)
        all_action.append(action)

    results = {}
    for arm, state_quat_slice, action_rot_slice in (
        ("left", LEFT_STATE_QUAT, LEFT_ACTION_ROT),
        ("right", RIGHT_STATE_QUAT, RIGHT_ACTION_ROT),
    ):
        rotvec_err_same, rotvec_err_next = [], []
        euler_intr_err_same, euler_intr_err_next = [], []
        euler_extr_err_same, euler_extr_err_next = [], []

        for state, action in zip(all_state, all_action):
            L = len(state)
            if L < 2:
                continue
            state_quat = state[:, state_quat_slice].astype(np.float64)     # (L,4)
            action_rot = action[:, action_rot_slice].astype(np.float64)   # (L,3)

            R_state = quat_wxyz_to_matrix(state_quat)          # (L,3,3)
            R_rotvec = rotvec_to_matrix(action_rot)             # (L,3,3)
            R_euler_intr = euler_xyz_to_matrix(action_rot, intrinsic=True)
            R_euler_extr = euler_xyz_to_matrix(action_rot, intrinsic=False)

            rotvec_err_same.append(geodesic_deg(R_rotvec, R_state))
            euler_intr_err_same.append(geodesic_deg(R_euler_intr, R_state))
            euler_extr_err_same.append(geodesic_deg(R_euler_extr, R_state))

            rotvec_err_next.append(geodesic_deg(R_rotvec[:-1], R_state[1:]))
            euler_intr_err_next.append(geodesic_deg(R_euler_intr[:-1], R_state[1:]))
            euler_extr_err_next.append(geodesic_deg(R_euler_extr[:-1], R_state[1:]))

        results[arm] = {
            "n_frames": sum(len(s) for s in all_state),
            "rotvec_same": np.concatenate(rotvec_err_same),
            "euler_intrinsic_same": np.concatenate(euler_intr_err_same),
            "euler_extrinsic_same": np.concatenate(euler_extr_err_same),
            "rotvec_next": np.concatenate(rotvec_err_next),
            "euler_intrinsic_next": np.concatenate(euler_intr_err_next),
            "euler_extrinsic_next": np.concatenate(euler_extr_err_next),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--parts", nargs="+", default=list(PART_ORDER), choices=list(PART_ORDER))
    args = ap.parse_args()

    dataset_root = resolve_repo_path(args.dataset_root) if args.dataset_root else (
        _THIS_DIR.parents[1] / "tools" / "roco2026_by_part"
    )
    episodes_df = load_episodes_table(dataset_root)
    data_df = load_data_table(dataset_root, columns=["observation.state", "action"])

    agg = {}
    for part in args.parts:
        res = audit_part(data_df, episodes_df, part)
        agg[part] = res
        la = res["left"]
        print(f"\n=== part={part}  left arm, n_frames={la['n_frames']} "
              f"(action[t] vs state[t], same frame) ===")
        print(f"  rotvec (axis-angle)        deg-err: {summarize(la['rotvec_same'])}")
        print(f"  Euler XYZ intrinsic        deg-err: {summarize(la['euler_intrinsic_same'])}")
        print(f"  Euler XYZ extrinsic        deg-err: {summarize(la['euler_extrinsic_same'])}")
        print(f"  -- secondary (action[t] vs state[t+1]) --")
        print(f"  rotvec (axis-angle)        deg-err: {summarize(la['rotvec_next'])}")
        print(f"  Euler XYZ intrinsic        deg-err: {summarize(la['euler_intrinsic_next'])}")
        print(f"  Euler XYZ extrinsic        deg-err: {summarize(la['euler_extrinsic_next'])}")

    print("\n=== pooled across all requested parts, left arm, action[t] vs state[t] ===")
    for key, label in (
        ("rotvec_same", "rotvec (axis-angle)"),
        ("euler_intrinsic_same", "Euler XYZ intrinsic"),
        ("euler_extrinsic_same", "Euler XYZ extrinsic"),
    ):
        pooled = np.concatenate([agg[p]["left"][key] for p in args.parts])
        print(f"  {label:22s} deg-err: {summarize(pooled)}")

    print("\n=== pooled, right arm (near-static -- weaker discriminator, sanity only) ===")
    for key, label in (
        ("rotvec_same", "rotvec (axis-angle)"),
        ("euler_intrinsic_same", "Euler XYZ intrinsic"),
        ("euler_extrinsic_same", "Euler XYZ extrinsic"),
    ):
        pooled = np.concatenate([agg[p]["right"][key] for p in args.parts])
        print(f"  {label:22s} deg-err: {summarize(pooled)}")


if __name__ == "__main__":
    main()
