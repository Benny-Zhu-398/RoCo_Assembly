"""Offline deploy-layer consistency check -- no Isaac Sim needed.

Catches silent numeric divergence between the *training-time* state/action
processing (dataset.py, normalization.py, evaluate.py) and the
*execution-time* processing the Isaac-side adapter
(task/policies/diffusion_stateonly.py) and its sidecar server
(task/dp_server_stateonly.py) actually perform. Same bug class as the
gripper-unit mistake documented in diffusion_stateonly.py's own module
docstring: an earlier version of that file rescaled the gripper dim by
GRIPPER_OPEN_LIMIT by analogy with policies/diffusion_lerobot.py, which was
wrong for this checkpoint's raw-radian convention. That bug was caught by
checking one dimension by hand -- this script checks every state dim (22)
and every action dim (14, including the right-arm constant fill-back) on
every run, against real dataset frames.

Design choice: every check below calls the *actual* production code
(dataset.py's module-bound normalize_state/normalize_action,
dp_server_stateonly.py's module-bound unnormalize_action/normalize_state,
diffusion_stateonly.py's real _build_state/act) instead of re-deriving the
formulas locally. A hand-copied "reference" implementation only catches
bugs the copy happens to reproduce differently -- exactly how a
single-dimension unit bug slips through review.

Two independent checks:

  STATE:  raw 44-D observation.state frame
            -> dataset.py's path (LEFT_STATE_IDX slice + normalize_state)
            -> adapter's path (DiffusionStateOnlyPolicy._build_state fed a
               mocked Observation/L/R decomposed from the SAME frame, then
               dp_server_stateonly.normalize_state)
          compared dimension-by-dimension.

  ACTION: 7-D normalized left action
            -> evaluate.py's path (normalization.unnormalize_action) +
               right_arm_constants.json fill-back = 14-D command
            -> adapter's path (dp_server_stateonly.unnormalize_action, then
               diffusion_stateonly.py's real act() pos/quat/gripper
               handling) + the SAME right-arm constant fill-back = 14-D
               command
          compared dimension-by-dimension. Also flags, as a separate
          structural finding, whether the adapter ever actually reads back
          the checkpoint's stored right_arm_constant (as of writing, it
          does not -- see the WARNING this script prints).

Usage:
    python check_deploy_consistency.py
    python check_deploy_consistency.py --parts battery_size1 gear_60teeth
    python check_deploy_consistency.py --ckpt outputs/battery_size1/final.pt
    python check_deploy_consistency.py --tol 1e-5 --n-frames 10
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_TASK_DIR = _REPO_ROOT / "task"
for _p in (_THIS_DIR, _TASK_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import constants  # noqa: E402
import dataset  # noqa: E402  -- module-bound normalize_state/normalize_action/LEFT_STATE_IDX: dataset.py's real path
import dp_server_stateonly  # noqa: E402  -- module-bound unnormalize_action/normalize_state: the real server path
import evaluate as evaluate_mod  # noqa: E402  -- module-bound unnormalize_action: evaluate.py's real path
from config import resolve_repo_path  # noqa: E402
from data_io import episode_frames, episode_ids_for_part, load_data_table, load_episodes_table  # noqa: E402
from inference_utils import load_checkpoint, load_norm_stats_from_checkpoint  # noqa: E402
from normalization import NormStats  # noqa: E402

import policies.diffusion_stateonly as diffusion_stateonly  # noqa: E402  -- the real Isaac-side adapter

PART_ORDER = constants.PART_ORDER
STATE_NAMES = constants.STATE_NAMES
ACTION_NAMES_FULL = constants.ACTION_NAMES_FULL
LEFT_ACTION_IDX = constants.LEFT_ACTION_IDX

TOL_DEFAULT = 1e-6


def euler_xyz_extrinsic_to_matrix(euler: np.ndarray) -> np.ndarray:
    """(..., 3) [rx, ry, rz] Euler-XYZ EXTRINSIC -> (..., 3, 3), i.e.
    R = Rz(rz) @ Ry(ry) @ Rx(rx) -- the convention
    task/policies/diffusion_stateonly.py's action rotation dims actually
    use (see rotation_convention_audit.py). Pure numpy, no scipy: this
    venv's scipy is unimportable (see the fallback note in
    adapter_raw7_and_command below)."""
    e = np.asarray(euler, dtype=np.float64).reshape(-1, 3)
    z, o = np.zeros(e.shape[0]), np.ones(e.shape[0])

    def axis_mat(axis, a):
        c, s = np.cos(a), np.sin(a)
        if axis == "x":
            m = np.stack([o, z, z, z, c, -s, z, s, c], axis=-1)
        elif axis == "y":
            m = np.stack([c, z, s, z, o, z, -s, z, c], axis=-1)
        else:
            m = np.stack([c, -s, z, s, c, z, z, z, o], axis=-1)
        return m.reshape(-1, 3, 3)

    Rx, Ry, Rz = axis_mat("x", e[:, 0]), axis_mat("y", e[:, 1]), axis_mat("z", e[:, 2])
    R = np.einsum("nij,njk->nik", Rz, Ry)
    R = np.einsum("nij,njk->nik", R, Rx)
    return R.reshape(euler.shape[:-1] + (3, 3))


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """(3,3) proper rotation matrix -> (4,) wxyz unit quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) unit quaternion wxyz -> (3,3) rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def geodesic_deg_matrices(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R_rel = Ra.T @ Rb
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class Mismatch:
    def __init__(self, part, kind, sample, dim_name, a_label, b_label, a, b):
        self.part = part
        self.kind = kind
        self.sample = sample
        self.dim_name = dim_name
        self.a_label = a_label
        self.b_label = b_label
        self.a = float(a)
        self.b = float(b)
        self.diff = abs(self.a - self.b)

    def __str__(self) -> str:
        return (f"[{self.kind:6s}] part={self.part:15s} sample={self.sample:14s} dim={self.dim_name:14s}  "
                f"{self.a_label}={self.a:14.9g}  {self.b_label}={self.b:14.9g}  |diff|={self.diff:.3e}")


class _FakePoseSource:
    """Stand-in for env_info.L_controller / R_controller.

    diffusion_stateonly.py's _build_state only ever calls
    `.end_effector.get_world_pose()`, and its act() only ever calls
    `.forward(pos, quat, grip)` -- this fakes exactly those two entry
    points instead of a full Isaac controller.
    """

    def __init__(self, pos=None, quat=None):
        self._pose = (pos, quat)
        self.forward_calls: List[Tuple[np.ndarray, np.ndarray, float]] = []

    @property
    def end_effector(self):
        return self

    def get_world_pose(self):
        return self._pose

    def forward(self, pos, quat, grip):
        self.forward_calls.append((np.asarray(pos, dtype=np.float64),
                                    np.asarray(quat, dtype=np.float64),
                                    float(grip)))
        return ("FAKE_ARTICULATION_ACTION", pos, quat, grip)


def _decompose_full_state(frame44: np.ndarray):
    """Split a raw 44-D observation.state row into the named components
    diffusion_stateonly.py._build_state reads off obs/self.L/self.R at
    execution time -- same STATE_NAMES_FULL layout dataset.py reads the
    column out of (constants.py), just un-concatenated."""
    f = np.asarray(frame44, dtype=np.float64)
    assert f.shape == (44,), f"expected 44-D state, got {f.shape}"
    Lp, Lq = f[0:3], f[3:7]
    Rp, Rq = f[7:10], f[10:14]
    q_left, q_right = f[14:21], f[21:28]
    qd_left, qd_right = f[28:35], f[35:42]
    Lg, Rg = f[42], f[43]
    return Lp, Lq, Rp, Rq, q_left, q_right, qd_left, qd_right, Lg, Rg


def dataset_state_n(frame44: np.ndarray, norm_stats: NormStats) -> np.ndarray:
    """dataset.py's own path: PartSequenceDataset.__getitem__'s lines
    (state = state_full[:, LEFT_STATE_IDX]; normalize_state(state, stats)),
    called via dataset.py's own bound names -- without instantiating the
    full (memory-heavy) Dataset class."""
    left = np.asarray(frame44, dtype=np.float32)[dataset.LEFT_STATE_IDX]
    return dataset.normalize_state(left, norm_stats)


def adapter_state_n(frame44: np.ndarray, norm_stats: NormStats) -> np.ndarray:
    """Run a raw 44-D frame through the REAL
    DiffusionStateOnlyPolicy._build_state (execution-time state
    reconstruction from obs/L/R) followed by dp_server_stateonly.py's own
    normalize_state -- i.e. exactly what happens between Isaac's `obs` and
    the model receiving a normalized state, with no hand-copied formula in
    between."""
    Lp, Lq, Rp, Rq, q_left, q_right, qd_left, qd_right, Lg, Rg = _decompose_full_state(frame44)

    pol = diffusion_stateonly.DiffusionStateOnlyPolicy.__new__(diffusion_stateonly.DiffusionStateOnlyPolicy)
    pol._Li = list(range(7))
    pol._Ri = list(range(7, 14))
    pol._Lg = 14
    pol._Rg = 15
    pol.L = _FakePoseSource(Lp, Lq)
    pol.R = _FakePoseSource(Rp, Rq)

    q_full = np.zeros(16, dtype=np.float64)
    q_full[pol._Li] = q_left
    q_full[pol._Ri] = q_right
    q_full[pol._Lg] = Lg
    q_full[pol._Rg] = Rg
    qd_full = np.zeros(16, dtype=np.float64)
    qd_full[pol._Li] = qd_left
    qd_full[pol._Ri] = qd_right

    obs = types.SimpleNamespace(joint_positions=q_full, joint_velocities=qd_full)
    state_22_raw = diffusion_stateonly.DiffusionStateOnlyPolicy._build_state(pol, obs)
    return dp_server_stateonly.normalize_state(
        state_22_raw.reshape(1, -1).astype(np.float32), norm_stats
    )[0]


def eval_raw7(action_n: np.ndarray, norm_stats: NormStats) -> np.ndarray:
    """evaluate.py's own bound unnormalize_action."""
    return evaluate_mod.unnormalize_action(
        np.asarray(action_n, dtype=np.float32).reshape(1, -1), norm_stats
    )[0]


def _euler_xyz_to_quat_wxyz_numpy_fallback(rx, ry, rz):
    """Pure-numpy stand-in for diffusion_stateonly._euler_xyz_to_quat_wxyz
    (Euler XYZ EXTRINSIC -> quat; see that module's ACTION ROTATION
    CONVENTION docstring section), used ONLY when this venv's scipy is
    unimportable (see inference_utils.py's own docstring: this training
    venv's scipy build can hit a numpy ABI break -- confirmed here:
    `from scipy.spatial.transform import Rotation` raises `AttributeError:
    module 'numpy' has no attribute 'long'`). Exists solely so act()'s real
    pos/gripper pass-through control flow can still run end-to-end offline;
    every call site that uses this reports it explicitly (see
    `used_fallback` / the printed NOTE) rather than silently swapping in
    different math. Converts via rotation_matrix_to_quat_wxyz(
    euler_xyz_extrinsic_to_matrix(...)) rather than a hand-derived
    Euler->quat formula, so this fallback and the round-trip check below
    share the exact same matrix-construction code."""
    R = euler_xyz_extrinsic_to_matrix(np.array([rx, ry, rz], dtype=np.float64))
    return matrix_to_quat_wxyz(R)


def adapter_raw7_and_command(action_n: np.ndarray, norm_stats: NormStats):
    """Run a normalized 7-D action through:
      1. dp_server_stateonly.py's bound unnormalize_action -- the real
         de-normalization the sidecar model server performs before
         handing raw units back to the Isaac-side adapter, then
      2. diffusion_stateonly.py's real act() pos/quat/gripper handling --
         the actual execution-time code, not a re-derivation of it.
    Returns (raw7_from_server, pos, quat_wxyz, grip, used_scipy_fallback) so
    callers can compare both the pre-quat 7-D command and the rotation the
    adapter would hand the IK controller.
    """
    raw7 = dp_server_stateonly.unnormalize_action(
        np.asarray(action_n, dtype=np.float32).reshape(1, -1), norm_stats
    )[0]

    pol = diffusion_stateonly.DiffusionStateOnlyPolicy.__new__(diffusion_stateonly.DiffusionStateOnlyPolicy)
    pol._skip = False
    pol.L = _FakePoseSource()

    used_fallback = False
    pol._queue = [raw7.astype(np.float64)]
    try:
        diffusion_stateonly.DiffusionStateOnlyPolicy.act(pol, obs=None)
    except Exception:
        used_fallback = True
        orig = diffusion_stateonly._euler_xyz_to_quat_wxyz
        diffusion_stateonly._euler_xyz_to_quat_wxyz = _euler_xyz_to_quat_wxyz_numpy_fallback
        try:
            pol._queue = [raw7.astype(np.float64)]  # act() already popped it on the failed attempt
            diffusion_stateonly.DiffusionStateOnlyPolicy.act(pol, obs=None)
        finally:
            diffusion_stateonly._euler_xyz_to_quat_wxyz = orig

    pos, quat, grip = pol.L.forward_calls[-1]
    return raw7, pos, quat, grip, used_fallback


def quat_vs_euler_xyz_geodesic_deg(quat_wxyz: np.ndarray, euler_xyz: np.ndarray) -> float:
    """Geodesic distance (degrees) between a quat and the rotation encoded
    by Euler-XYZ-extrinsic angles -- used to check that
    diffusion_stateonly._euler_xyz_to_quat_wxyz's own scipy call is
    self-consistent with this script's independent pure-numpy matrix
    construction (a bug in the scipy Euler convention/argument order would
    show up here, independent of whether Euler is even the right dataset
    convention -- that question is rotation_convention_audit.py's job)."""
    Rq = quat_wxyz_to_matrix(quat_wxyz)
    Re = euler_xyz_extrinsic_to_matrix(np.asarray(euler_xyz, dtype=np.float64))
    return geodesic_deg_matrices(Rq, Re)


def load_right_arm_constant(path: Path, part: str) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    entry = data.get(part)
    if entry is None:
        raise ValueError(f"no right-arm constant for part={part!r} in {path}")
    return np.asarray(entry["constant"], dtype=np.float64)


def check_adapter_uses_right_arm_constant() -> Optional[str]:
    """Structural (not numeric) finding: train.py stores a per-part
    right_arm_constant in every checkpoint specifically so 'execution-time
    code' can reconstruct the full 14-D action (see train.py's own WARNING
    message when the constant is missing). Verify the adapter file
    actually reads it back, rather than assuming from the docstrings."""
    src = Path(diffusion_stateonly.__file__).read_text()
    if "right_arm_constant" in src:
        return None
    return (
        "task/policies/diffusion_stateonly.py never references "
        "'right_arm_constant', even though every checkpoint stores one "
        "(train.py) specifically for execution-time 14-D reconstruction. "
        "The 14-D command compared below is assembled by THIS SCRIPT for "
        "comparison purposes only -- the deployed adapter never builds "
        "one; the right arm is instead held in joint-space by "
        "run_pick_place.py's R_arm_hold_q, a separate mechanism that this "
        "script cannot exercise offline."
    )


def sample_frames(dataset_root: Path, part: str, n_frames: int):
    episodes = load_episodes_table(dataset_root)
    data = load_data_table(dataset_root, columns=["observation.state", "action"])
    ep = episode_ids_for_part(episodes, part)[0]
    state, action = episode_frames(data, ep)
    idx = np.linspace(0, len(state) - 1, num=min(n_frames, len(state))).astype(int)
    idx = sorted(set(idx.tolist()))
    return [(ep, t, state[t], action[t]) for t in idx]


def run_state_check(part: str, frames, norm_stats: NormStats, tol: float, mismatches: List[Mismatch]) -> int:
    n_checked = 0
    for ep, t, state44, _action14 in frames:
        a = dataset_state_n(state44, norm_stats)
        b = adapter_state_n(state44, norm_stats)
        n_checked += 1
        for d in range(len(STATE_NAMES)):
            if abs(a[d] - b[d]) > tol:
                mismatches.append(Mismatch(
                    part, "state", f"ep{ep}_t{t}", STATE_NAMES[d], "dataset.py", "adapter", a[d], b[d]
                ))
    return n_checked


def run_action_check(part: str, frames, norm_stats: NormStats, right_const: np.ndarray,
                      tol: float, mismatches: List[Mismatch], notes: List[str]) -> int:
    test_vectors: List[Tuple[str, np.ndarray]] = []
    for ep, t, _state44, action14 in frames:
        raw_left = np.asarray(action14, dtype=np.float32)[LEFT_ACTION_IDX]
        action_n = dataset.normalize_action(raw_left, norm_stats)
        test_vectors.append((f"ep{ep}_t{t}", action_n))
    # Synthetic boundary vectors on top of real frames: real val-set frames
    # may never touch the gripper-open extreme (see [[roco_release_action_truncated]]
    # -- gripper-open never physically observed in most parts' data), which
    # is exactly the kind of region a purely data-driven test would miss.
    for tag, val in (("boundary_open", 1.0), ("boundary_close", -1.0), ("boundary_mid", 0.0)):
        test_vectors.append((tag, np.full(7, val, dtype=np.float32)))

    n_checked = 0
    fallback_reported = False
    for label, action_n in test_vectors:
        raw7_eval = eval_raw7(action_n, norm_stats)
        raw7_adapter, pos, quat, grip, used_fallback = adapter_raw7_and_command(action_n, norm_stats)
        n_checked += 1
        if used_fallback and not fallback_reported:
            fallback_reported = True
            notes.append(
                "scipy.spatial.transform is unimportable in this venv "
                "(AttributeError: numpy has no attribute 'long' -- a real numpy/scipy ABI "
                "mismatch, see inference_utils.py's own docstring on this). act()'s real "
                "pos/gripper control flow was still exercised; only the Euler-XYZ->quat call "
                "was patched with a numpy-only equivalent so the run could complete. The "
                "quat round-trip geodesic check below is therefore NOT exercising "
                "diffusion_stateonly._euler_xyz_to_quat_wxyz itself -- fix scipy in "
                "training/diffusion_policy's venv to get real coverage of that function."
            )

        full_eval = np.concatenate([raw7_eval, right_const])
        full_adapter = np.concatenate([
            np.concatenate([pos, raw7_adapter[3:6], [grip]]), right_const,
        ])
        for d in range(len(ACTION_NAMES_FULL)):
            if abs(full_eval[d] - full_adapter[d]) > tol:
                mismatches.append(Mismatch(
                    part, "action", label, ACTION_NAMES_FULL[d],
                    "evaluate.py+const", "adapter+const", full_eval[d], full_adapter[d],
                ))

        # Informational: is the quat the adapter would actually hand the IK
        # controller the same rotation as evaluate.py's raw Euler-XYZ
        # values, independent of quat sign ambiguity? Skipped entirely when
        # used_fallback: the patched-in math would just be compared
        # against itself (see _euler_xyz_to_quat_wxyz_numpy_fallback).
        if not used_fallback:
            geo_deg = quat_vs_euler_xyz_geodesic_deg(quat, raw7_eval[3:6])
            if geo_deg > 1e-2:
                notes.append(
                    f"part={part} sample={label}: adapter's Euler-XYZ->quat conversion "
                    f"differs from this script's independent matrix construction by "
                    f"{geo_deg:.4f} deg (geodesic) -- investigate "
                    "_euler_xyz_to_quat_wxyz's scipy call (axis order / intrinsic vs "
                    "extrinsic argument)."
                )
    return n_checked


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str, default=None)
    ap.add_argument("--norm-stats", type=str, default=str(_THIS_DIR / "norm_stats.json"))
    ap.add_argument("--right-arm-constants", type=str, default=str(_THIS_DIR / "right_arm_constants.json"))
    ap.add_argument("--ckpt", type=str, default=None,
                     help="optional checkpoint .pt; overrides --norm-stats/--right-arm-constants "
                          "with the values actually stored in it")
    ap.add_argument("--parts", nargs="+", default=list(PART_ORDER), choices=list(PART_ORDER))
    ap.add_argument("--n-frames", type=int, default=5, help="frames sampled per part (state + action checks)")
    ap.add_argument("--tol", type=float, default=TOL_DEFAULT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_repo_path(args.dataset_root) if args.dataset_root else (
        _REPO_ROOT / "tools" / "roco2026_by_part"
    )

    right_arm_by_part = {}
    if args.ckpt:
        ckpt = load_checkpoint(args.ckpt)
        norm_stats = load_norm_stats_from_checkpoint(ckpt)
        entry = ckpt.get("right_arm_constant")
        if entry is None:
            print(f"[check] --ckpt {args.ckpt!r} has no right_arm_constant; "
                  f"falling back to {args.right_arm_constants} for all parts")
        else:
            right_arm_by_part[ckpt["part"]] = np.asarray(entry["constant"], dtype=np.float64)
    else:
        norm_stats = NormStats.load(args.norm_stats)

    mismatches: List[Mismatch] = []
    notes: List[str] = []
    n_state_checked = 0
    n_action_checked = 0

    for part in args.parts:
        frames = sample_frames(dataset_root, part, args.n_frames)
        n_state_checked += run_state_check(part, frames, norm_stats, args.tol, mismatches)

        right_const = right_arm_by_part.get(part)
        if right_const is None:
            right_const = load_right_arm_constant(Path(args.right_arm_constants), part)
        n_action_checked += run_action_check(part, frames, norm_stats, right_const, args.tol, mismatches, notes)

    fillback_warning = check_adapter_uses_right_arm_constant()

    print(f"[check] checked {n_state_checked} state sample(s) (22 dims each) and "
          f"{n_action_checked} action sample(s) (14 dims each, right-arm constant filled back) "
          f"across parts={args.parts}, tol={args.tol:g}\n")

    if fillback_warning:
        print(f"[check] STRUCTURAL WARNING: {fillback_warning}\n")

    seen = set()
    unique_notes = [n for n in notes if not (n in seen or seen.add(n))]
    for note in unique_notes:
        print(f"[check] NOTE: {note}")
    if unique_notes:
        print()

    if mismatches:
        print(f"[check] FAIL: {len(mismatches)} dimension mismatch(es) > {args.tol:g}:\n")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)

    print(f"[check] PASS: dataset.py/evaluate.py path and the diffusion_stateonly.py adapter path "
          f"agree on every checked dimension to within {args.tol:g}.")


if __name__ == "__main__":
    main()
