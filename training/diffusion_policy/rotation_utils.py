"""Rotation-convention helpers -- ROTVEC ONLY. See the WARNING below before
using anything in this file against tools/roco2026_by_part data.

state ee-orientation is a unit quaternion (wxyz); action ee-orientation is
assumed a rotation vector (axis-angle) below that is NOT confined to the
canonical magnitude range [0, pi] (see exploration report: 4.96% of samples
exceed pi, max 2.54*pi). The two conventions must never share normalization
statistics or be compared numerically without going through an explicit
conversion here.

*** WARNING -- this rotvec assumption is FALSE for tools/roco2026_by_part ***
rotation_convention_audit.py empirically confirmed that dataset's action
rotation dims are actually Euler XYZ EXTRINSIC, not rotvec (comparing both
decodes' geodesic distance to same-frame state quaternions, pooled across
all 9 parts: rotvec gives median/mean/p90 ~1.0/16.1/60.9 deg, Euler-XYZ-
extrinsic gives ~0.6/4.0/3.4 deg). Every function below
(`rotvec_to_quat_wxyz`, `quat_wxyz_to_rotvec`, `rotvec_to_rot6d`,
`rot6d_to_rotvec`) is only correct for data confirmed to actually be rotvec
-- true for the self-collected collect_lerobot_v3.py/v4.py datasets
(metadata: "absolute_cartesian_target_xyz_rotvec_gripper"; see
policies/act_eval_usb.py / act_eval_gear.py, which decode those correctly),
FALSE for tools/roco2026_by_part. `dataset.py`'s "rot6d" branch
(ModelConfig.rotation_repr == "rot6d") calls `rotvec_to_rot6d` on
roco2026_by_part action data and is therefore currently WRONG if ever
exercised -- it has never been trained end-to-end (see config.py), so this
is a live landmine, not an active bug; fix the conversion (build the
rotation matrix via Euler-XYZ-extrinsic, e.g.
inference_utils._euler_xyz_extrinsic_to_matrix, instead of
Rotation.from_rotvec) before ever setting rotation_repr="rot6d" for this
dataset. `find_rotvec_jumps` below is comparatively safe either way: it's a
generic ">pi raw-component-diff" discontinuity heuristic that doesn't
strictly require rotvec semantics to be useful, but its docstring/naming
still assumes rotvec -- read it as "large jump in the 3 raw rotation
numbers", not literally "rotvec jump", when applied to roco2026_by_part.

Requires scipy (not part of the Isaac Sim env — install it in the
training-side venv, see requirements.txt). NOTE: as of this writing this
training venv's installed scipy build is unimportable against its numpy
(AttributeError: module 'numpy' has no attribute 'long' -- a numpy/scipy
ABI mismatch, scipy wants numpy>=2.0 but 1.26.4 is installed here); every
function in this file will raise until that's fixed. check_deploy_consistency.py
and rotation_convention_audit.py route around this with pure-numpy
equivalents rather than depending on this module.
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz unit quaternion -> (..., 3) rotation vector."""
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat_wxyz, dtype=np.float64)
    xyzw = q[..., [1, 2, 3, 0]]
    rv = Rotation.from_quat(xyzw.reshape(-1, 4)).as_rotvec()
    return rv.reshape(q.shape[:-1] + (3,)).astype(np.float32)


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 4) wxyz unit quaternion."""
    from scipy.spatial.transform import Rotation

    r = np.asarray(rotvec, dtype=np.float64)
    xyzw = Rotation.from_rotvec(r.reshape(-1, 3)).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return wxyz.reshape(r.shape[:-1] + (4,)).astype(np.float32)


def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 6) continuous 6D repr (Zhou et al. 2019).

    The 6D repr is just the first two columns of the rotation matrix,
    flattened. Reserved for ModelConfig.rotation_repr == "rot6d"; not the
    default path.
    """
    from scipy.spatial.transform import Rotation

    r = np.asarray(rotvec, dtype=np.float64)
    rot_mat = Rotation.from_rotvec(r.reshape(-1, 3)).as_matrix()  # (N, 3, 3)
    d6 = rot_mat[:, :, :2].reshape(-1, 6)
    return d6.reshape(r.shape[:-1] + (6,)).astype(np.float32)


def rot6d_to_rotvec(d6: np.ndarray) -> np.ndarray:
    """(..., 6) continuous 6D repr -> (..., 3) rotation vector via Gram-Schmidt."""
    from scipy.spatial.transform import Rotation

    d = np.asarray(d6, dtype=np.float64).reshape(-1, 6)
    a1, a2 = d[:, 0:3], d[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mat = np.stack([b1, b2, b3], axis=-1)  # columns = b1,b2,b3
    rv = Rotation.from_matrix(rot_mat).as_rotvec()
    return rv.reshape(np.asarray(d6).shape[:-1] + (3,)).astype(np.float32)


def find_rotvec_jumps(rotvec_seq: np.ndarray, threshold: float = np.pi) -> np.ndarray:
    """Frame indices t (0-indexed into the diff array) where
    ||rotvec[t+1] - rotvec[t]|| > threshold. Empty array if none.

    Exploration finding: across all 1800 episodes this fires exactly once
    (episode 384 / pin / frame 3). Kept as a per-episode runtime check
    rather than a one-off finding so future data drops are re-validated
    automatically.
    """
    seq = np.asarray(rotvec_seq, dtype=np.float64)
    if len(seq) < 2:
        return np.empty((0,), dtype=np.int64)
    diffs = np.diff(seq, axis=0)
    mags = np.linalg.norm(diffs, axis=-1)
    return np.where(mags > threshold)[0]
