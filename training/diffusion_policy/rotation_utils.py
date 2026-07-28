"""Rotation-convention helpers.

state ee-orientation is a unit quaternion (wxyz); action ee-orientation is a
rotation vector (axis-angle) that is NOT confined to the canonical
magnitude range [0, pi] (see exploration report: 4.96% of samples exceed
pi, max 2.54*pi). The two conventions must never share normalization
statistics or be compared numerically without going through an explicit
conversion here.

Requires scipy (not part of the Isaac Sim env — install it in the
training-side venv, see requirements.txt).
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
