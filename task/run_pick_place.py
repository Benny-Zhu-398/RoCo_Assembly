# ruff: noqa: E402
# Eval harness for the IROS 2026 vega_1u assembly challenge.
#
# Iterates pc.part_order. For each part: optionally spawns it, creates a
# snap-attacher (success detector) if the release_mode is "snap", asks the
# loaded Policy to drive the L arm via act(obs) each physics step, and
# advances on policy.is_done() / snap fire / per-part timeout. Scores at
# the end via _grade_task (pass/fail per part, optionally written to JSON).
#
# Select the policy with --policy module.path.ClassName. Default:
# policies.baseline_scripted.BaselinePolicy (the reference scripted solver).
# Participants subclass policy_api.Policy and pass --policy <their module>.
#
# R arm holds its init joint pose every step.

import os
import subprocess
import sys
import time

_LAUNCH_CWD = os.getcwd()

from isaacsim import SimulationApp

def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name):
    raw = os.getenv(name)
    return None if raw in (None, "") else int(raw)


def _env_float(name):
    raw = os.getenv(name)
    return None if raw in (None, "") else float(raw)


_HEADLESS = _env_flag("ISAACSIM_HEADLESS")
_SIM_CONFIG = {
    "headless": _HEADLESS,
    "multi_gpu": _env_flag("ISAACSIM_MULTI_GPU", default=False),
}
_ACTIVE_GPU = _env_int("ISAACSIM_ACTIVE_GPU")
_PHYSICS_GPU = _env_int("ISAACSIM_PHYSICS_GPU")
if _ACTIVE_GPU is not None:
    _SIM_CONFIG["active_gpu"] = _ACTIVE_GPU
if _PHYSICS_GPU is not None:
    _SIM_CONFIG["physics_gpu"] = _PHYSICS_GPU

simulation_app = SimulationApp(_SIM_CONFIG)

import argparse
import importlib
import json
import math
import numpy as np

import param_config as pc
from controllers.vega_1u_setup import (
    restore_scene_part_xforms,
    setup_pick_place_sim,
    sync_fixed_camera_to_source,
)
from controllers.part_from_usd import DynamicPart
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
try:
    from deferred_video import DeferredFrameVideoRecorder
except ImportError:
    # Optional: only needed for --record-video-deferred. Absent in trimmed
    # checkouts; live ffmpeg recording and every non-recording run still work.
    DeferredFrameVideoRecorder = None
from policy_api import EnvInfo, Observation, PartTarget

# Physics material prim authored in the scene USD; bound to every spawned
# DynamicPart so newly imported parts share the same friction/restitution
# profile as rod_16mm / bolt_8mm (which are already in scene_base.usd).
_PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterial"

# snap_attach.py lives one directory up from Task_test, alongside the
# USD scene. Add that directory to sys.path so the runner can import it.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import omni.physx
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from snap_attach import SnapAttacher, _quat


# Collision approximation applied to each spawned part's Mesh descendants.
# convexDecomposition handles concave shapes well at the cost of cooking
# time. Override per-part via PART_CONFIG[name]["collision_approximation"]
# if a particular part wants e.g. "convexHull" (faster) or "sdf" (tighter).
_DEFAULT_COLLISION_APPROXIMATION = "convexDecomposition"

# Stuck detector: print one diagnostic block when the follower stalls on a
# single waypoint for this many physics steps without advancing.
# ~100 steps ≈ 0.5 s at 200 Hz physics. Re-armed on the next wp_idx change.
STUCK_LOG_STEPS = 100

# URDF<->USD frame offset on the EE link. Mirror of _STAGE_OFFSET_INV in
# controllers/lula_ik_controller.py — used here to convert Lula's FK output
# (URDF frame) into the stage-frame orientation we can compare directly
# against wp.orn. R_offset = (0, 0, 0, -1) (180° about Z). Without this
# composition the stuck-diagnostic comparison shows a fake ~180° error.
_R_OFFSET_FK_TO_STAGE = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)


class FfmpegVideoRecorder:
    deferred = False

    def __init__(self, path, fps=30, camera="head"):
        self.path = path
        self.fps = int(fps)
        self.camera = camera
        self._proc = None
        self._shape = None
        self.frames = 0

    @property
    def enabled(self):
        return bool(self.path)

    def write(self, frame):
        if not self.enabled or frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        if self._proc is None:
            self._start(w, h)
        if self._shape != (h, w):
            raise ValueError(
                f"video frame size changed from {self._shape} to {(h, w)}"
            )
        self._proc.stdin.write(arr.tobytes())
        self.frames += 1

    def _start(self, width, height):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._shape = (height, width)

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            ret = self._proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg exited with code {ret}")
            print(f"[video] wrote {self.frames} frames -> {self.path}")
        finally:
            self._proc = None


def _quat_mul(q1, q2):
    """Hamilton product (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ===========================================================================
# Joint-ownership masks for the L+R action merge.
# ===========================================================================
R_ARM_JOINT_NAMES = [f"R_arm_j{i}" for i in range(1, 8)]
L_OWNED_JOINTS = {
    "Lift", "torso_flip",
    "L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
    "L_arm_j5", "L_arm_j6", "L_arm_j7",
    "L_gripper_joint", "L_gripper_joint_01",
}
R_OWNED_JOINTS = {
    "R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
    "R_arm_j5", "R_arm_j6", "R_arm_j7",
    "R_gripper_joint", "R_gripper_joint_01",
}


def _value_or_none(seq, i):
    try:
        v = seq[i]
    except (TypeError, IndexError):
        return None
    return v


def merge_bimanual_actions(L_action, R_action, dof_names):
    n = len(dof_names)

    def _vec(action):
        v = getattr(action, "joint_positions", None)
        if v is None:
            return [None] * n
        return list(v)

    Lp = _vec(L_action)
    Rp = _vec(R_action)
    merged = [None] * n
    for i, jname in enumerate(dof_names):
        lv = _value_or_none(Lp, i)
        rv = _value_or_none(Rp, i)
        if jname in L_OWNED_JOINTS:
            merged[i] = lv if lv is not None else rv
        elif jname in R_OWNED_JOINTS:
            merged[i] = rv if rv is not None else lv
        else:
            merged[i] = lv if lv is not None else rv
    return ArticulationAction(joint_positions=merged)


def build_snap_attacher(stage, part_name, snap_cfg):
    """Construct a SnapAttacher from a param_config snap dict.

    Converts target_pos / target_rot tuples to Gf types and picks a joint
    path unique to the part. ``joint_path`` defaults to
    ``/World/_snap_joint_<part_name>`` so concurrent or sequential snaps
    don't collide on the same prim path.
    """
    target_pos = Gf.Vec3d(*snap_cfg["target_pos"])
    w, x, y, z = snap_cfg["target_rot"]
    target_rot = _quat(w, x, y, z)
    connect_rot_tup = snap_cfg.get("connect_rot")
    connect_rot = _quat(*connect_rot_tup) if connect_rot_tup is not None else None
    connect_offset_rot_tup = snap_cfg.get("connect_offset_rot")
    connect_offset_rot = (_quat(*connect_offset_rot_tup)
                          if connect_offset_rot_tup is not None else None)
    debug_every_override = os.environ.get("ROCO_SNAP_DEBUG_EVERY")
    return SnapAttacher(
        stage,
        movable_path=snap_cfg["movable_path"],
        parent_body_path=snap_cfg["parent_body_path"],
        target_pos=target_pos,
        target_rot=target_rot,
        pos_tol=snap_cfg.get("pos_tol", 0.005),
        pos_tol_axes=snap_cfg.get("pos_tol_axes"),
        rot_tol_deg=snap_cfg.get("rot_tol_deg", 5.0),
        joint_path=snap_cfg.get("joint_path",
                                f"/World/_snap_joint_{part_name}"),
        debug=(snap_cfg.get("debug", False)
               or debug_every_override is not None),
        debug_every=(snap_cfg.get("debug_every", 30)
                     if debug_every_override is None
                     else int(debug_every_override)),
        set_kinematic_on_snap=snap_cfg.get("set_kinematic", False),
        mesh_path=snap_cfg.get("mesh_path"),
        author_joint_on_snap=snap_cfg.get("author_joint", True),
        connect_pos=snap_cfg.get("connect_pos"),
        connect_rot=connect_rot,
        connect_offset_pos=snap_cfg.get("connect_offset_pos"),
        connect_offset_rot=connect_offset_rot,
        part_name=part_name,
    )


# Directory holding the per-part USD files (../parts relative to this script).
_PARTS_USD_DIR = os.path.join(_PARENT_DIR, "parts")


def _apply_mesh_colliders(stage, prim_path, approximation):
    """Walk every Mesh descendant of `prim_path` and apply
    UsdPhysics.CollisionAPI + MeshCollisionAPI(approximation=...).

    Most exported part USDs ship visual meshes only — no PhysicsCollisionAPI.
    Without this pass, DynamicPart's set_collision_approximation() call on the
    root xform has nothing to act on and the spawned part has zero physical
    collider, so the gripper passes through it. Idempotent: a mesh that
    already has CollisionAPI gets its approximation refreshed and nothing
    else. Returns the count of meshes touched.
    """
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return 0
    n = 0
    for p in Usd.PrimRange(root):
        if p.GetTypeName() != "Mesh":
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(p)
        mesh_api.CreateApproximationAttr().Set(approximation)
        n += 1
    return n


def import_missing_parts():
    """Populate the stage with every part recorded in part_init_poses.json
    that isn't already present, then handle any remaining pc.part_order
    parts not covered by the JSON.

    Spawn pose precedence per part:
      1. pc.PART_INIT_POSES[name]['pos' / 'orn'] (primary — bulk spawn).
      2. cfg['pick_pos'] for position and cfg['spawn_orn'] (or identity)
         for orientation (fallback for pc.part_order parts not in
         part_init_poses.json).

    Parts already present in the loaded scene USD are left alone.
    """
    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()

    # Wrap the scene-authored physics material so we can bind it to each
    # spawned part. If the prim is missing (older scene), fall through to
    # DynamicPart's auto-created default.
    shared_phys_mat = None
    if is_prim_path_valid(_PHYSICS_MATERIAL_PATH):
        shared_phys_mat = PhysicsMaterial(prim_path=_PHYSICS_MATERIAL_PATH)
    else:
        print(f"[setup] WARNING: {_PHYSICS_MATERIAL_PATH} not in scene — "
              f"spawned parts will get DynamicPart's default material.")

    def _spawn(name, pos, orn, source):
        prim_path = f"/World/parts/{name}"
        usd_path = os.path.join(_PARTS_USD_DIR, f"{name}.usdc")
        if not os.path.isfile(usd_path):
            raise FileNotFoundError(
                f"part {name!r} missing from stage and no USD at {usd_path}"
            )
        # Add the reference FIRST so the DynamicPart constructor sees an
        # existing prim. The "prim already valid" branch in
        # VisualPart.__init__ skips the default gray PreviewSurface override
        # that would otherwise mask the part USD's authored materials.
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        # Apply CollisionAPI + approximation to the Mesh descendants before
        # DynamicPart wraps the prim. Without this, the part has no physical
        # collider and the gripper passes through it.
        cfg = pc.get_part_config(name) if name in pc.PART_CONFIG else {}
        approximation = cfg.get(
            "collision_approximation", _DEFAULT_COLLISION_APPROXIMATION
        )
        _apply_mesh_colliders(stage, prim_path, approximation)
        DynamicPart(
            prim_path=prim_path,
            name=name,
            position=np.asarray(pos, dtype=np.float64),
            orientation=np.asarray(orn, dtype=np.float64),
            physics_material=shared_phys_mat,
        )

    # Pass 1: every part recorded in part_init_poses.json. Parts already
    # present in the scene USD (e.g. gears authored in scene_base.usd
    # so PhysX bakes their SDF collider at stage-load time) are left
    # alone here. JSON XY overrides for scene-resident parts already
    # happened pre-World in vega_1u_setup._override_scene_part_xy_inplace
    # (must run before task wrappers snapshot the "default" pose,
    # otherwise World.reset() reverts the override).
    for name, entry in pc.PART_INIT_POSES.items():
        if "pos" not in entry or "orn" not in entry:
            continue
        prim_path = f"/World/parts/{name}"
        if is_prim_path_valid(prim_path):
            continue
        _spawn(name, entry["pos"], entry["orn"], "part_init_poses.json")

    # Pass 2: anything still missing that pc.part_order asks for — falls back
    # to PART_CONFIG values (covers parts not in part_init_poses.json).
    for name in pc.part_order:
        if is_prim_path_valid(f"/World/parts/{name}"):
            continue
        cfg = pc.get_part_config(name)
        pos = cfg.get("pick_pos")
        if pos is None:
            raise ValueError(
                f"part {name!r} not in part_init_poses.json and has no "
                f"pick_pos in PART_CONFIG — cannot spawn."
            )
        spawn_orn = cfg.get("spawn_orn")
        if spawn_orn is None:
            spawn_orn = identity_q
        _spawn(name, pos, spawn_orn, "PART_CONFIG")


def _save_stage_snapshot(out_path):
    """Save the current Isaac Sim stage to a flattened USD file.

    Output is a single self-contained file that can be re-opened in Isaac
    Sim (or any USD viewer) and played to test post-placement physics
    (e.g. whether a part stays put or falls). Path is resolved relative
    to this script's directory if not absolute. Parent dirs are created
    if missing.
    """
    if not out_path:
        return
    abs_path = (out_path if os.path.isabs(out_path)
                else os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), out_path)))
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print(f"[setup] WARN: no stage to save to {abs_path}")
        return
    # stage.Export() flattens all sublayers + references into one file.
    try:
        stage.Export(abs_path)
        print(f"[setup] saved final stage snapshot -> {abs_path}")
    except Exception as e:
        print(f"[setup] WARN: failed to save stage to {abs_path}: {e}")


def _override_scene_part_xy(stage, prim_path, target_xy, name):
    """Surgically update the XY of a scene-resident part. Z preserved,
    scale and rotation preserved. Handles both authoring conventions:
      1. xformOp:translate / orient / scale (Isaac Sim default for
         spawned parts) — update the translate op's value directly.
      2. xformOp:transform (single matrix, Composer's default when you
         drag in a reference) — pull the translation out of the matrix,
         swap XY, write the matrix back. Rotation/scale rows are
         untouched.
    """
    target_x, target_y = (float(target_xy[0]), float(target_xy[1]))
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        print(f"[setup] {name}: prim {prim_path} not in stage — no XY override.")
        return

    # Pick the rigid-body prim if there is one; otherwise the root.
    body_prim = root
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            body_prim = p
            break

    xform = UsdGeom.Xformable(body_prim)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    transform_op = None
    for op in ops:
        t = op.GetOpType()
        if t == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif t == UsdGeom.XformOp.TypeTransform and transform_op is None:
            transform_op = op

    if translate_op is not None:
        cur = translate_op.Get()
        cur_z = float(cur[2]) if cur is not None else 0.0
        translate_op.Set(Gf.Vec3d(target_x, target_y, cur_z))
        print(f"[setup] {name}: overrode xformOp:translate XY to "
              f"({target_x:+.5f}, {target_y:+.5f}) on {body_prim.GetPath()} "
              f"(Z preserved at {cur_z:+.5f}).")
        return

    if transform_op is not None:
        mat = transform_op.Get()
        if mat is None:
            print(f"[setup] {name}: xformOp:transform has no authored value.")
            return
        old_t = mat.ExtractTranslation()
        cur_z = float(old_t[2])
        # SetTranslateOnly preserves the rotation/scale rows of the matrix.
        new_mat = Gf.Matrix4d(mat)
        new_mat.SetTranslateOnly(Gf.Vec3d(target_x, target_y, cur_z))
        transform_op.Set(new_mat)
        print(f"[setup] {name}: overrode xformOp:transform translation XY to "
              f"({target_x:+.5f}, {target_y:+.5f}) on {body_prim.GetPath()} "
              f"(Z preserved at {cur_z:+.5f}).")
        return

    op_names = [op.GetName() for op in ops]
    print(f"[setup] {name}: no xformOp:translate OR xformOp:transform on "
          f"{body_prim.GetPath()} (found: {op_names}) — can't override XY.")


def _grade_task(stage, snap_fired_parts, results_json_path=None, metadata=None):
    """End-of-iteration summary: pass/fail for every name in pc.part_order.

    Grading rule per part:
      release_mode == "snap"  -> pass iff name in snap_fired_parts.
      release_mode == "open"  -> pass iff the part's MESH world position
                                 is within GRADE_POS_TOL_M of place_pos.
                                 (No orientation check — batteries / gears
                                 are axis-symmetric.)

    If ``results_json_path`` is non-None (or pc.RESULTS_JSON_PATH is set),
    the per-part outcome is also written to that path as JSON for offline
    aggregation.
    """
    GRADE_POS_TOL_M = float(getattr(pc, "GRADE_POS_TOL_M", 0.01))
    print("=" * 72)
    print(f"[grade] task summary (pos tol = {GRADE_POS_TOL_M * 1000:.1f} mm):")
    n_pass = 0
    n_fail = 0
    n_missing = 0
    per_part_results = []
    for part in pc.part_order:
        if isinstance(part, str) and part.startswith("<"):
            continue
        cfg = pc.get_part_config(part)
        release_mode = cfg.get("release_mode", "open")
        if release_mode == "snap":
            fired = part in snap_fired_parts
            status = "pass" if fired else "FAIL"
            print(f"  {part:<16}  snap={'fired' if fired else 'NOT fired':<10}  "
                  f"-> {status}")
            per_part_results.append({
                "name": part,
                "release_mode": "snap",
                "snap_fired": bool(fired),
                "pass": bool(fired),
            })
            if fired:
                n_pass += 1
            else:
                n_fail += 1
            continue

        # Position-only grade. Prefer cfg["grade_pos"] (final settled
        # pose, post-release) over place_pos (gripper release pose) —
        # they often differ when the part sinks / rolls / settles after
        # the gripper opens.
        place_pos = cfg.get("grade_pos")
        if place_pos is None:
            place_pos = cfg.get("place_pos")
        if place_pos is None:
            print(f"  {part:<16}  no grade_pos / place_pos -> SKIP")
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "no grade_pos / place_pos",
            })
            continue
        prim = stage.GetPrimAtPath(f"/World/parts/{part}")
        if not prim or not prim.IsValid():
            print(f"  {part:<16}  prim missing from stage -> MISSING")
            n_missing += 1
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "prim missing",
            })
            continue
        deepest_mesh = None
        deepest_d = -1
        for p in Usd.PrimRange(prim):
            if p.GetTypeName() != "Mesh":
                continue
            d = p.GetPath().pathString.count("/")
            if d > deepest_d:
                deepest_d = d
                deepest_mesh = p
        if deepest_mesh is None:
            print(f"  {part:<16}  no Mesh descendant -> MISSING")
            n_missing += 1
            per_part_results.append({
                "name": part, "release_mode": "open",
                "pass": False, "reason": "no mesh descendant",
            })
            continue
        # Choose between mesh-translation and AABB-midpoint as the
        # "actual position" reading. For axis-symmetric parts like
        # batteries, the mesh local origin may sit anywhere — and a
        # rotation about the symmetry axis moves mesh_world_t even
        # though the part is geometrically in the same place. AABB
        # midpoint (world-axis-aligned bounds, midpoint of min/max per
        # axis) is invariant under that rotation and gives a fair
        # position-only comparison. Opt in per-part via cfg["grade_use_aabb"]
        # = True; grade_pos must also be expressed as the AABB midpoint
        # (not the mesh local origin) for the comparison to be apples-to-
        # apples.
        if cfg.get("grade_use_aabb", False):
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_],
            )
            bbox = bbox_cache.ComputeWorldBound(deepest_mesh)
            aabb = bbox.ComputeAlignedRange()
            mid = aabb.GetMidpoint()
            cur = np.array([float(mid[0]), float(mid[1]), float(mid[2])],
                           dtype=np.float64)
            measure = "AABBmid"
        else:
            m = UsdGeom.XformCache().GetLocalToWorldTransform(deepest_mesh)
            t = m.ExtractTranslation()
            cur = np.array([float(t[0]), float(t[1]), float(t[2])],
                           dtype=np.float64)
            measure = "meshT"
        target = np.asarray(place_pos, dtype=np.float64)
        delta = cur - target  # 3D per-axis error
        err = float(np.linalg.norm(delta))
        ok = err < GRADE_POS_TOL_M
        status = "pass" if ok else "FAIL"
        print(f"  {part:<16}  pos_err={err * 1000:7.2f} mm "
              f"d=({delta[0]*1000:+6.2f}, {delta[1]*1000:+6.2f}, "
              f"{delta[2]*1000:+6.2f}) mm ({measure})  -> {status}")
        # On FAIL, dump full actual/target so you can re-baseline grade_pos.
        if not ok:
            print(f"    actual=({float(cur[0]):.6f}, {float(cur[1]):.6f}, "
                  f"{float(cur[2]):.6f})")
            print(f"    target=({float(target[0]):.6f}, "
                  f"{float(target[1]):.6f}, {float(target[2]):.6f})")
        n_pass += 1 if ok else 0
        n_fail += 0 if ok else 1
        per_part_results.append({
            "name": part,
            "release_mode": "open",
            "measure": measure,
            "pos_err_m": err,
            "tolerance_m": GRADE_POS_TOL_M,
            "actual": [float(cur[0]), float(cur[1]), float(cur[2])],
            "target": [float(target[0]), float(target[1]), float(target[2])],
            "pass": bool(ok),
        })

    print(f"[grade] summary: pass={n_pass}  fail={n_fail}  missing={n_missing}")
    print("=" * 72)

    # Optional JSON dump for offline aggregation.
    out_path = (results_json_path
                if results_json_path is not None
                else getattr(pc, "RESULTS_JSON_PATH", None))
    if out_path:
        abs_path = (out_path if os.path.isabs(out_path)
                    else os.path.abspath(os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), out_path)))
        try:
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            payload = {
                "metadata": dict(metadata or {}),
                "pos_tol_m": GRADE_POS_TOL_M,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_missing": n_missing,
                "per_part": per_part_results,
            }
            with open(abs_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[grade] wrote results JSON -> {abs_path}")
        except Exception as e:
            print(f"[grade] WARN: failed to write results JSON to {abs_path}: {e}")


def _parse_args():
    """CLI: --policy and --results-json overrides."""
    parser = argparse.ArgumentParser(
        description="vega_1u assembly challenge eval harness."
    )
    parser.add_argument(
        "--policy",
        default="policies.baseline_scripted.BaselinePolicy",
        help="Dotted import path to a Policy subclass "
             "(default: policies.baseline_scripted.BaselinePolicy).",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Override pc.RESULTS_JSON_PATH. If set, _grade_task writes the "
             "per-part pass/fail summary to this file at the end of the run.",
    )
    parser.add_argument(
        "--record-video",
        default=None,
        help="Write an MP4 rollout video from one of the task cameras.",
    )
    parser.add_argument(
        "--record-video-camera",
        default="head",
        choices=("head", "L_wrist", "R_wrist"),
        help="Camera stream to record when --record-video is set.",
    )
    parser.add_argument(
        "--record-video-fps",
        type=int,
        default=30,
        help="Output video frame rate. Frames are sampled from sim time.",
    )
    parser.add_argument(
        "--record-video-deferred",
        action="store_true",
        default=_env_flag("ROCO_RECORD_VIDEO_DEFERRED"),
        help="Spool PNG frames and encode after Isaac exits. This avoids a "
             "live ffmpeg pipe during Windows simulation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("ROCO_EVAL_MAX_STEPS"),
        help="Stop after this many task-control steps and still write "
             "results JSON/video. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=_env_float("ROCO_EVAL_MAX_SIM_SECONDS"),
        help="Stop after this much simulated task time and still write "
             "results JSON/video. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=_env_int("ROCO_EVAL_MAX_PARTS"),
        help="Stop after this many parts have ended by policy done, snap, "
             "or timeout.",
    )
    parser.add_argument(
        "--pi05-one-step-dry-run",
        action="store_true",
        help="Capture one real observation and request one raw prediction "
             "without applying any robot action.",
    )
    parser.add_argument(
        "--pi05-dry-run-log",
        default="artifacts/pi05_one_step_dry_run.log",
        help="Output log for --pi05-one-step-dry-run.",
    )
    parser.add_argument(
        "--pi05-ik-dry-run",
        action="store_true",
        help="Request one prediction and solve IK without applying the returned action.",
    )
    parser.add_argument(
        "--pi05-ik-dry-run-log",
        default="artifacts/pi05_ik_dry_run.log",
        help="Output log for --pi05-ik-dry-run.",
    )
    parser.add_argument(
        "--pi05-five-request-diagnostic",
        action="store_true",
        help="Send 5 raw requests for one real observation without applying actions.",
    )
    parser.add_argument(
        "--pi05-five-request-log",
        default="artifacts/pi05_five_request_timing.log",
        help="Output log for --pi05-five-request-diagnostic.",
    )
    parser.add_argument(
        "--pi05-continuous-episode",
        action="store_true",
        default=_env_flag("PI05_CONTINUOUS_EPISODE"),
        help="Run one continuous pi0.5 episode with one reset and all snap "
             "detectors active.",
    )
    parser.add_argument(
        "--fix-task-board",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Make the task-board rigid body kinematic. Defaults on for "
             "Pi05LeRobotPolicy and off for other policies.",
    )
    parser.add_argument(
        "--policy-control-hz",
        type=float,
        default=_env_float("PI05_CONTROL_HZ"),
        help="Update the policy at this simulated-time frequency and hold the "
             "last articulation target between updates.",
    )
    parser.add_argument(
        "--residual-random-eval",
        action="store_true",
        help="Run the Gym-style ResidualEnv on the single part selected by "
             "ROCO_PART_ORDER.",
    )
    parser.add_argument(
        "--residual-td3-train",
        action="store_true",
        help="Train and periodically evaluate TD3 on the selected ResidualEnv task.",
    )
    parser.add_argument(
        "--residual-task",
        default="snap_insertion",
        help="Registered ResidualTask factory (default: snap_insertion).",
    )
    parser.add_argument(
        "--residual-reward",
        default="bounded_snap",
        help="Registered residual reward factory (default: bounded_snap).",
    )
    parser.add_argument(
        "--residual-episodes",
        type=int,
        default=10,
        help="Number of random ResidualEnv episodes (default: 10).",
    )
    parser.add_argument(
        "--residual-env-max-steps",
        type=int,
        default=64,
        help="Maximum residual actions per episode (default: 64).",
    )
    parser.add_argument(
        "--residual-prefix-horizon",
        type=int,
        default=16,
        help="BC chunk horizon used only during reset prefix replay (default: 16).",
    )
    parser.add_argument(
        "--residual-max-reset-attempts",
        type=int,
        default=50,
        help="Maximum BC prefix attempts within one residual episode. Set to "
             "1 when every reset must count as an evaluation trial.",
    )
    parser.add_argument(
        "--residual-seed",
        type=int,
        default=0,
        help="Random-action seed for ResidualEnv validation.",
    )
    parser.add_argument(
        "--residual-action-mode",
        choices=("random", "zero", "fixed-x"),
        default="random",
        help="Residual validation action source (default: random).",
    )
    parser.add_argument(
        "--residual-fixed-x",
        type=float,
        default=-1.0,
        help="X action used by --residual-action-mode=fixed-x (default: -1).",
    )
    parser.add_argument(
        "--residual-delta-max-pos",
        type=float,
        default=0.005,
        help="Residual position L2 limit in metres (default: 0.005).",
    )
    parser.add_argument(
        "--residual-delta-max-ori-deg",
        type=float,
        default=30.0,
        help="Residual orientation geodesic limit in degrees (default: 30).",
    )
    parser.add_argument(
        "--residual-gate-enter-ori-deg",
        type=float,
        default=45.0,
        help="Residual gate orientation admission threshold in degrees "
             "(default: 45).",
    )
    parser.add_argument(
        "--residual-smooth-weight",
        type=float,
        default=1.0,
        help="Penalty weight for the executed residual norm (default: 1.0).",
    )
    parser.add_argument(
        "--residual-log",
        default="artifacts/residual_env_random_eval.jsonl",
        help="JSONL path for random ResidualEnv validation telemetry.",
    )
    parser.add_argument("--td3-train-episodes", type=int, default=200)
    parser.add_argument("--td3-warmup-steps", type=int, default=1000)
    parser.add_argument("--td3-batch-size", type=int, default=256)
    parser.add_argument("--td3-buffer-capacity", type=int, default=100_000)
    parser.add_argument("--td3-eval-interval", type=int, default=50)
    parser.add_argument("--td3-eval-episodes", type=int, default=30)
    parser.add_argument("--td3-exploration-noise", type=float, default=0.03)
    parser.add_argument("--td3-device", default="cpu")
    parser.add_argument(
        "--td3-log",
        default="artifacts/residual_td3_training.jsonl",
    )
    parser.add_argument(
        "--td3-checkpoint-dir",
        default="artifacts/residual_td3_checkpoints",
    )
    # --- Grouped-vision DP residual (TD3 through policy.set_residual()) ---
    parser.add_argument(
        "--residual-grip-train",
        action="store_true",
        help="Train a TD3 xy residual on top of the frozen grouped-vision "
             "Diffusion Policy via its set_residual() hook. Requires "
             "--policy policies.diffusion_vision_grouped_grip."
             "DiffusionVisionGroupedGripPolicy and DP_CL_MODE=residual. "
             "Reuses the --td3-* knobs.",
    )
    parser.add_argument(
        "--residual-grip-log",
        default="artifacts/residual_grip_td3_training.jsonl",
        help="JSONL telemetry path for --residual-grip-train.",
    )
    parser.add_argument(
        "--residual-grip-checkpoint-dir",
        default="artifacts/residual_grip_td3_checkpoints",
        help="Directory for --residual-grip-train TD3 checkpoints.",
    )
    parser.add_argument(
        "--residual-grip-settle-steps",
        type=int,
        default=90,
        help="Sim control steps to let the part settle after the frozen "
             "policy locks and releases, before grading (default: 90).",
    )
    parser.add_argument(
        "--residual-grip-prefix-steps",
        type=int,
        default=1500,
        help="Max frozen-DP control steps per reset to reach the endpoint "
             "trigger window (default: 1500).",
    )
    parser.add_argument(
        "--residual-grip-max-reset-attempts",
        type=int,
        default=10,
        help="Max sim resets per episode to get the frozen DP into the "
             "trigger window (default: 10).",
    )
    parser.add_argument(
        "--residual-grip-eval-ckpt",
        default=None,
        help="With --residual-grip-train: skip training, load this TD3 "
             "checkpoint's actor and roll --td3-eval-episodes greedy episodes. "
             "Add --record-video PATH.mp4 to also write one video of the run.",
    )
    # --- Grouped-vision DP GRASP residual (TD3 through set_grasp_residual()) ---
    parser.add_argument(
        "--residual-grasp-train",
        action="store_true",
        help="Train a TD3 xy residual on the frozen grouped-vision DP's PICK "
             "target via its set_grasp_residual() hook. Requires "
             "DP_GRASP_RES_M>0. Placement is left to BC (use DP_CL_MODE=off). "
             "Reuses the --td3-* knobs.",
    )
    parser.add_argument(
        "--residual-grasp-log",
        default="artifacts/residual_grasp_td3_training.jsonl",
    )
    parser.add_argument(
        "--residual-grasp-checkpoint-dir",
        default="artifacts/residual_grasp_td3_checkpoints",
    )
    parser.add_argument(
        "--residual-grasp-max-steps",
        type=int,
        default=24,
        help="Max RL steps of grasp-approach correction per episode (default: 24).",
    )
    parser.add_argument(
        "--residual-grasp-lift-steps",
        type=int,
        default=120,
        help="Max BC control steps to roll the post-grasp lift while checking "
             "whether the part tracks the gripper (default: 120).",
    )
    parser.add_argument(
        "--residual-grasp-lift-rise-m",
        type=float,
        default=0.03,
        help="Roll BC's lift until the ee has risen this far, then score the "
             "move-together check (part must rise within move-tol of it). "
             "Higher = the grasp must survive a bigger lift (default: 0.03).",
    )
    parser.add_argument(
        "--residual-grasp-place-cap",
        type=int,
        default=400,
        help="Max BC control steps to let the place run for the optional "
             "seat bonus (only after a good lift). 0 disables it (default: 400).",
    )
    parser.add_argument(
        "--residual-grasp-prefix-steps",
        type=int,
        default=400,
        help="Max frozen-DP control steps per reset to reach the grasp "
             "window (default: 400).",
    )
    parser.add_argument(
        "--residual-grasp-max-reset-attempts",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--residual-grasp-ckpt-interval",
        type=int,
        default=200,
        help="Save a grasp checkpoint every N training episodes (must be a "
             "multiple of --td3-eval-interval; default 200). Eval still runs "
             "at --td3-eval-interval.",
    )
    parser.add_argument(
        "--residual-grasp-eval-ckpt",
        default=None,
        help="With --residual-grasp-train: skip training, load this TD3 "
             "checkpoint's actor and roll --td3-eval-episodes greedy episodes. "
             "Add --record-video PATH.mp4 to also write one video of the run.",
    )
    # SimulationApp consumes argv too; tolerate unknown args so the runner
    # can be launched as ${ISAAC_SIM}/python.sh run_pick_place.py --policy ...
    args = parser.parse_known_args()[0]
    diagnostic_mode_count = sum(
        bool(value)
        for value in (
            args.pi05_one_step_dry_run,
            args.pi05_ik_dry_run,
            args.pi05_five_request_diagnostic,
        )
    )
    if diagnostic_mode_count > 1:
        parser.error("pi0.5 dry-run diagnostic modes are mutually exclusive")
    if args.residual_random_eval and args.residual_td3_train:
        parser.error("residual evaluation and TD3 training modes are mutually exclusive")
    _residual_modes = sum(bool(v) for v in (
        args.residual_random_eval, args.residual_td3_train,
        args.residual_grip_train, args.residual_grasp_train,
    ))
    if _residual_modes > 1:
        parser.error(
            "choose one residual workflow: --residual-random-eval / "
            "--residual-td3-train / --residual-grip-train / --residual-grasp-train"
        )
    if args.residual_grip_train and (
        args.residual_grip_settle_steps <= 0
        or args.residual_grip_prefix_steps <= 0
        or args.residual_grip_max_reset_attempts <= 0
    ):
        parser.error("--residual-grip-* step counts must be positive")
    if args.residual_grasp_train and (
        args.residual_grasp_max_steps <= 0
        or args.residual_grasp_lift_steps <= 0
        or args.residual_grasp_prefix_steps <= 0
        or args.residual_grasp_max_reset_attempts <= 0
        or args.residual_grasp_place_cap < 0
        or args.residual_grasp_lift_rise_m <= 0.0
    ):
        parser.error("--residual-grasp-* counts must be positive (place-cap >= 0)")
    if args.policy_control_hz is not None and args.policy_control_hz <= 0:
        parser.error("--policy-control-hz must be positive")
    if args.pi05_continuous_episode:
        if args.max_steps is None:
            parser.error("--pi05-continuous-episode requires --max-steps")
        if args.policy_control_hz is None:
            args.policy_control_hz = 10.0
    if args.record_video_deferred and not args.record_video:
        parser.error("--record-video-deferred requires --record-video")
    if args.record_video_deferred and DeferredFrameVideoRecorder is None:
        parser.error(
            "--record-video-deferred needs the 'deferred_video' module, which "
            "is not present in this checkout; use --record-video without it "
            "for live ffmpeg encoding"
        )
    if (
        args.residual_episodes <= 0
        or args.residual_env_max_steps <= 0
        or args.residual_prefix_horizon <= 0
        or args.residual_max_reset_attempts <= 0
        or args.residual_delta_max_pos <= 0.0
        or args.residual_delta_max_ori_deg <= 0.0
        or not 0.0 < args.residual_gate_enter_ori_deg < 60.0
        or args.residual_smooth_weight < 0.0
    ):
        parser.error(
            "residual counts/limits must be positive, smooth weight non-negative, "
            "and gate-enter orientation must lie in (0, 60) degrees"
        )
    if not -1.0 <= args.residual_fixed_x <= 1.0:
        parser.error("--residual-fixed-x must lie in [-1, 1]")
    if (
        args.td3_train_episodes <= 0
        or args.td3_warmup_steps < 0
        or args.td3_batch_size <= 0
        or args.td3_buffer_capacity < args.td3_batch_size
        or args.td3_eval_interval <= 0
        or args.td3_eval_episodes <= 0
        or args.td3_exploration_noise < 0.0
    ):
        parser.error("TD3 counts/noise must be valid and buffer >= batch size")
    for attr in ("max_steps", "max_sim_seconds", "max_parts"):
        value = getattr(args, attr, None)
        if value is not None and value <= 0:
            setattr(args, attr, None)
    for attr in (
        "results_json",
        "record_video",
        "pi05_dry_run_log",
        "pi05_ik_dry_run_log",
        "pi05_five_request_log",
        "residual_log",
        "td3_log",
        "td3_checkpoint_dir",
        "residual_grip_log",
        "residual_grip_checkpoint_dir",
        "residual_grip_eval_ckpt",
        "residual_grasp_log",
        "residual_grasp_checkpoint_dir",
        "residual_grasp_eval_ckpt",
    ):
        value = getattr(args, attr, None)
        if value and not os.path.isabs(value):
            setattr(args, attr, os.path.abspath(os.path.join(_LAUNCH_CWD, value)))
    return args


def _load_policy_class(dotted_path: str):
    """Resolve `module.path.ClassName` -> the class object."""
    if "." not in dotted_path:
        raise ValueError(
            f"--policy must be dotted (module.ClassName), got {dotted_path!r}"
        )
    module_name, _, class_name = dotted_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"could not import policy module {module_name!r}: {e}"
        ) from e
    if not hasattr(module, class_name):
        raise AttributeError(
            f"policy module {module_name!r} has no attribute {class_name!r}"
        )
    return getattr(module, class_name)


def main():
    args = _parse_args()
    residual_workflow = args.residual_random_eval or args.residual_td3_train
    grip_residual_workflow = bool(args.residual_grip_train)
    grasp_residual_workflow = bool(args.residual_grasp_train)
    residual_part_name = None
    if residual_workflow:
        if len(pc.part_order) != 1:
            raise ValueError(
                "residual workflows require exactly one ROCO_PART_ORDER part"
            )
        residual_part_name = pc.part_order[0]
        residual_part_cfg = pc.get_part_config(residual_part_name)
        if args.residual_task == "snap_insertion" and not residual_part_cfg.get("snap"):
            raise ValueError(
                f"residual workflows require a snap-configured part, got "
                f"{residual_part_name!r}"
            )
    pi05_policy_enabled = args.policy.endswith(".Pi05LeRobotPolicy")
    residual_policy_enabled = any(
        args.policy.endswith(suffix)
        for suffix in (
            ".Pi05LeRobotPolicy",
            ".DiffusionLeRobotPolicy",
            ".DiffusionStateOnlyPolicy",
        )
    )
    if residual_workflow and not residual_policy_enabled:
        raise ValueError(
            "residual workflows require Pi05LeRobotPolicy, "
            "DiffusionLeRobotPolicy, or DiffusionStateOnlyPolicy"
        )
    if residual_workflow and pi05_policy_enabled:
        # Residual RL is trained and evaluated against the local frozen policy.
        # Do not inherit a PI05_REMOTE=1 setting from older eval shells.
        os.environ["PI05_REMOTE"] = "0"
        os.environ["PI05_EXEC_HORIZON"] = "1"
        os.environ["PI05_SAFETY_FILTER"] = "0"
        # Checkpoints were trained with the literal LeRobot part name as task
        # label.  Preserve an explicit caller override for prompt ablations.
        os.environ.setdefault("PI05_TASK", residual_part_name)
    fix_task_board_enabled = (
        residual_policy_enabled
        if args.fix_task_board is None
        else bool(args.fix_task_board)
    )
    fixed_head_camera_enabled = bool(
        pi05_policy_enabled
        or args.pi05_one_step_dry_run
        or args.pi05_five_request_diagnostic
        or args.pi05_continuous_episode
    )
    recorder_class = (
        DeferredFrameVideoRecorder
        if args.record_video_deferred
        else FfmpegVideoRecorder
    )
    video_recorder = recorder_class(
        args.record_video,
        fps=args.record_video_fps,
        camera=args.record_video_camera,
    )
    record_period_s = 1.0 / float(max(1, args.record_video_fps))
    next_record_time_s = 0.0
    camera_output_enabled = bool(pc.enable_camera_output or video_recorder.enabled)
    camera_viewports_enabled = bool(pc.enable_camera_viewports and not _HEADLESS)
    auto_play = bool(
        _HEADLESS
        or args.pi05_one_step_dry_run
        or args.pi05_five_request_diagnostic
        or args.pi05_continuous_episode
    )
    exit_on_complete = bool(auto_play or video_recorder.enabled)
    run_complete = False
    finalized = False
    total_task_steps = 0
    completed_parts = 0
    five_request_reset_sent = False
    continuous_loop_steps = 0
    next_policy_time_s = None
    held_merged_action = None
    fixed_head_camera_ready = not fixed_head_camera_enabled

    # The task signature still requires L/R object prim paths. Point both
    # at a STATIC prim so the task's SingleRigidPrim wrapper never aliases
    # a part that snap_attach later joint-locks — that aliasing was what
    # invalidated the physics tensor view mid-snap. L/R_target_position
    # are stored as observation labels we never query, so a dummy zero
    # vector is fine.
    _DUMMY_TARGET = np.zeros(3, dtype=np.float64)
    (my_world, my_controller, my_robots,
     head_depth_camera, L_wrist_camera, R_wrist_camera,
     articulation_controller, task_params, reset_needed) = setup_pick_place_sim(
        L_object_prim_path=pc.L_object_prim_path,
        R_object_prim_path=pc.R_object_prim_path,
        L_target_position=_DUMMY_TARGET,
        R_target_position=_DUMMY_TARGET,
        joint_opened_position=np.array([pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=camera_viewports_enabled,
        enable_camera_output=camera_output_enabled,
        fixed_head_camera=fixed_head_camera_enabled,
        fix_task_board=fix_task_board_enabled,
    )

    # Spawn any pc.part_order entries that aren't already in the loaded scene.
    import_missing_parts()

    L_controller = my_controller["L"]
    R_controller = my_controller["R"]
    L_robot = my_robots["L"]

    dof_names = list(L_robot.dof_names)
    R_arm_dof_indices = np.array(
        [dof_names.index(j) for j in R_ARM_JOINT_NAMES], dtype=np.int64
    )
    L_gripper_dof_index = dof_names.index("L_gripper_joint")
    L_arm_joint_names = [j for j in dof_names if j.startswith("L_arm_j")]

    def _apply_init_joint_targets():
        """Override the live joint state with pc.INIT_JOINT_TARGETS.

        Called at startup and after every World.reset() (stop+play). Velocities
        are zeroed too so PD doesn't carry residual motion through the teleport.
        """
        targets = getattr(pc, "INIT_JOINT_TARGETS", None)
        if not targets:
            return
        full_q = np.asarray(L_robot.get_joint_positions(),
                            dtype=np.float64).copy()
        for jname, val in targets.items():
            if jname in dof_names:
                full_q[dof_names.index(jname)] = float(val)
        L_robot.set_joint_positions(full_q)
        L_robot.set_joint_velocities(np.zeros(len(dof_names)))

    _apply_init_joint_targets()

    # R: latch init pose, command those joints every step.
    R_arm_hold_q = np.asarray(L_robot.get_joint_positions())[R_arm_dof_indices].astype(np.float64)

    # The learned policy only commands the left arm and gripper. In continuous
    # mode, keep the shared base and head-camera chain at the post-reset pose so
    # uncommanded joints cannot drift while one action is held between updates.
    continuous_hold_joint_names = (
        "Lift", "torso_flip", "head_j1", "head_j2", "head_j3",
    )
    initial_full_q = np.asarray(
        L_robot.get_joint_positions(), dtype=np.float64
    )
    continuous_hold_targets = {
        name: float(initial_full_q[dof_names.index(name)])
        for name in continuous_hold_joint_names
        if name in dof_names
    }

    # Snapshot the L arm's c-space joint vector at startup. The baseline
    # policy uses this as the return-home target between parts; other
    # policies can use it for whatever (or ignore it).
    L_arm_init_q = np.asarray(
        L_controller.current_cspace_q(), dtype=np.float64
    ).copy()

    # Build EnvInfo and load the chosen policy. `L_controller` is stashed
    # on env_info so the BaselinePolicy can wrap it in EEPathFollower;
    # participant policies should ignore that attribute.
    env_info = EnvInfo(
        dof_names=dof_names,
        L_arm_joints=L_arm_joint_names,
        R_arm_joints=list(R_ARM_JOINT_NAMES),
        L_gripper_joint="L_gripper_joint",
        L_arm_init_q=L_arm_init_q.copy(),
        physics_dt=1.0 / 200.0,
        enable_camera_output=camera_output_enabled,
        L_controller=L_controller,
        R_controller=R_controller,
    )

    policy_class = _load_policy_class(args.policy)
    policy = policy_class(env_info)
    print(f"[setup] policy: {policy_class.__module__}.{policy_class.__name__}")

    # Snap attacher lifecycle (env-owned success detector).
    stage = omni.usd.get_context().get_stage()
    physx_iface = omni.physx.get_physx_interface()

    parts_iter = iter(pc.part_order)
    current_part = None
    current_snap_attacher = None
    current_snap_sub = None
    continuous_snap_attachers = {}
    continuous_snap_subs = []
    snap_fired_parts = set()
    part_step_count = 0
    PER_PART_TIMEOUT_STEPS = int(getattr(pc, "PER_PART_TIMEOUT_STEPS", 3000))

    def _clear_snap_state():
        nonlocal current_snap_attacher, current_snap_sub
        # Drop the subscription first so the dying attacher can't be ticked
        # by a stray physx event between the two None assignments.
        current_snap_sub = None
        current_snap_attacher = None
        continuous_snap_subs.clear()
        continuous_snap_attachers.clear()

    def _build_observation(with_timings=False):
        observation_start = time.perf_counter()
        full_q = np.asarray(L_robot.get_joint_positions(), dtype=np.float64)
        try:
            full_qd = np.asarray(L_robot.get_joint_velocities(), dtype=np.float64)
        except Exception:
            full_qd = np.zeros_like(full_q)

        # EE pose via Lula FK at the last commanded q, composed with the
        # URDF<->stage frame offset so we return stage-frame quaternions.
        ee_pos = np.zeros(3, dtype=np.float64)
        ee_orn = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        try:
            ik = getattr(L_controller, "_ik", None)
            fk_fn = getattr(ik, "fk_for_last_command", None) if ik else None
            if fk_fn is not None:
                p, o = fk_fn()
                if p is not None:
                    ee_pos = np.asarray(p, dtype=np.float64)
                if o is not None:
                    ee_orn = np.asarray(
                        _quat_mul(o, _R_OFFSET_FK_TO_STAGE), dtype=np.float64
                    )
        except Exception:
            pass

        rgb = {"head": None, "L_wrist": None, "R_wrist": None}
        depth = {"head": None, "L_wrist": None, "R_wrist": None}
        intrinsics = {"head": None, "L_wrist": None, "R_wrist": None}
        camera_start = time.perf_counter()
        if camera_output_enabled:
            for key, cam in (("head", head_depth_camera),
                             ("L_wrist", L_wrist_camera),
                             ("R_wrist", R_wrist_camera)):
                if cam is None:
                    continue
                try:
                    rgba = cam.get_rgba()
                    if rgba is not None and rgba.size > 0:
                        rgb[key] = np.asarray(rgba[..., :3])
                except Exception:
                    pass
                try:
                    frame = cam.get_current_frame()
                    if frame and frame.get("distance_to_image_plane") is not None:
                        depth[key] = np.asarray(frame["distance_to_image_plane"],
                                                dtype=np.float32)
                except Exception:
                    pass
                try:
                    K = cam.get_intrinsics_matrix()
                    if K is not None:
                        intrinsics[key] = np.asarray(K, dtype=np.float64)
                except Exception:
                    pass

        camera_capture_ms = (time.perf_counter() - camera_start) * 1000.0
        if args.pi05_continuous_episode:
            snap_fired = any(
                attacher.attached
                for attacher in continuous_snap_attachers.values()
            )
        else:
            snap_fired = bool(current_snap_attacher is not None
                              and current_snap_attacher.attached)

        obs = Observation(
            step_idx=int(my_world.current_time_step_index),
            joint_positions=full_q,
            joint_velocities=full_qd,
            L_gripper_position=float(full_q[L_gripper_dof_index]),
            ee_pose_L=(ee_pos, ee_orn),
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            snap_fired=snap_fired,
            target_part=(
                None
                if args.pi05_continuous_episode
                else (current_part if isinstance(current_part, str) else None)
            ),
        )
        if with_timings:
            return obs, {
                "camera_capture": camera_capture_ms,
                "observation_construction": (
                    time.perf_counter() - observation_start
                ) * 1000.0,
            }
        return obs

    def _build_part_target(name):
        cfg = pc.get_part_config(name)
        snap = cfg.get("snap") or {}
        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).copy()
        return PartTarget(
            name=name,
            release_mode=cfg.get("release_mode", "open"),
            pick_pos=_arr(cfg.get("pick_pos")),
            spawn_orn=_arr(cfg.get("spawn_orn")),
            place_pos=_arr(cfg.get("place_pos")),
            grade_pos=_arr(cfg.get("grade_pos")),
            snap_target_pos=_arr(snap.get("target_pos")),
            snap_target_rot=_arr(snap.get("target_rot")),
            snap_pos_tol=_arr(snap.get("pos_tol_axes")),
            snap_rot_tol_deg=(None if snap.get("rot_tol_deg") is None
                              else float(snap["rot_tol_deg"])),
            gripper_open=float(cfg.get("gripper_open", 0.0)),
            gripper_close=float(cfg.get("gripper_close", 0.0)),
            ee_orientation=_arr(cfg.get("ee_orientation")),
            extra=dict(cfg),
        )

    def _finalize_iteration(reason="complete"):
        nonlocal finalized
        if finalized:
            return
        finalized = True
        final_full_q = np.asarray(
            L_robot.get_joint_positions(), dtype=np.float64
        )
        continuous_hold_actual = {
            name: float(final_full_q[dof_names.index(name)])
            for name in continuous_hold_targets
        }
        metadata = {
            "completion_reason": reason,
            "current_part": current_part,
            "completed_parts": int(completed_parts),
            "total_task_steps": int(total_task_steps),
            "sim_time_s": (
                float(my_world.current_time_step_index * env_info.physics_dt)
                if my_world is not None else None
            ),
            "max_steps": args.max_steps,
            "max_sim_seconds": args.max_sim_seconds,
            "max_parts": args.max_parts,
            "snap_fired_parts": sorted(snap_fired_parts),
            "pi05_continuous_episode": bool(args.pi05_continuous_episode),
            "fixed_head_camera": bool(fixed_head_camera_enabled),
            "camera_viewports": bool(camera_viewports_enabled),
            "task_board_fixed": bool(fix_task_board_enabled),
            "policy_control_hz": args.policy_control_hz,
            "continuous_loop_steps": int(continuous_loop_steps),
            "continuous_hold_targets": (
                continuous_hold_targets if args.pi05_continuous_episode else {}
            ),
            "continuous_hold_actual": (
                continuous_hold_actual if args.pi05_continuous_episode else {}
            ),
        }
        _grade_task(stage, snap_fired_parts,
                    results_json_path=args.results_json,
                    metadata=metadata)
        save_path = getattr(pc, "SAVE_FINAL_STAGE_PATH", None)
        if save_path:
            _save_stage_snapshot(save_path)

    def _start_next_part():
        """Advance to the next part: build snap attacher, call policy.reset()."""
        nonlocal current_part, current_snap_attacher, current_snap_sub
        nonlocal part_step_count, run_complete, completed_parts
        nonlocal five_request_reset_sent

        # Record previous part's snap status before clearing.
        if (current_part is not None
                and current_snap_attacher is not None
                and current_snap_attacher.attached):
            snap_fired_parts.add(current_part)
        if current_part is not None:
            completed_parts += 1
        _clear_snap_state()

        if args.max_parts and completed_parts >= args.max_parts:
            current_part = None
            run_complete = True
            print(f"[setup] reached max-parts={args.max_parts}; ending early.")
            _finalize_iteration("max_parts")
            return None

        try:
            current_part = next(parts_iter)
        except StopIteration:
            current_part = None
            run_complete = True
            print("[setup] All parts done.")
            _finalize_iteration("complete")
            return None

        cfg = pc.get_part_config(current_part)
        release_mode = cfg.get("release_mode", "open")
        snap_cfg = cfg.get("snap")
        if release_mode == "snap":
            if snap_cfg is None:
                raise ValueError(
                    f"part {current_part!r} has release_mode='snap' but no "
                    f"'snap' config dict in PART_CONFIG."
                )
            current_snap_attacher = build_snap_attacher(
                stage, current_part, snap_cfg,
            )
            attacher = current_snap_attacher
            current_snap_sub = physx_iface.subscribe_physics_step_events(
                lambda dt, a=attacher: a.update()
            )
            print(f"[setup] {current_part}: snap mode  "
                  f"movable={snap_cfg['movable_path']}  "
                  f"target_pos={snap_cfg['target_pos']}")

        obs = _build_observation()
        target = _build_part_target(current_part)
        if not args.pi05_five_request_diagnostic or not five_request_reset_sent:
            policy.reset(obs, target)
            if args.pi05_five_request_diagnostic:
                five_request_reset_sent = True
        part_step_count = 0
        print(f"now working on the part: {current_part}", flush=True)
        return current_part

    def _start_continuous_episode():
        """Reset pi0.5 once and keep every snap detector active."""
        nonlocal current_part, part_step_count
        current_part = None
        for part_name in pc.part_order:
            cfg = pc.get_part_config(part_name)
            if cfg.get("release_mode", "open") != "snap":
                continue
            snap_cfg = cfg.get("snap")
            if snap_cfg is None:
                raise ValueError(
                    f"part {part_name!r} has release_mode='snap' but no snap config"
                )
            attacher = build_snap_attacher(stage, part_name, snap_cfg)
            continuous_snap_attachers[part_name] = attacher
            continuous_snap_subs.append(
                physx_iface.subscribe_physics_step_events(
                    lambda dt, a=attacher: a.update()
                )
            )
            print(
                f"[pi05-continuous] snap detector part={part_name} "
                f"movable={snap_cfg['movable_path']}",
                flush=True,
            )

        first_part = next(iter(pc.part_order))
        policy.reset(_build_observation(), _build_part_target(first_part))
        part_step_count = 0
        print(
            f"[pi05-continuous] started one episode control_hz="
            f"{args.policy_control_hz:g} max_policy_steps={args.max_steps}",
            flush=True,
        )
        print(
            f"[pi05-continuous] holding support joints "
            f"{continuous_hold_targets}",
            flush=True,
        )

    def _merge_left_with_right_hold(L_action):
        R_action_positions = [None] * len(dof_names)
        for j_idx, val in zip(R_arm_dof_indices, R_arm_hold_q.tolist()):
            R_action_positions[j_idx] = float(val)
        R_action = ArticulationAction(joint_positions=R_action_positions)
        merged = merge_bimanual_actions(L_action, R_action, dof_names)
        if args.pi05_continuous_episode:
            merged_positions = list(merged.joint_positions)
            for name, value in continuous_hold_targets.items():
                merged_positions[dof_names.index(name)] = value
            merged = ArticulationAction(joint_positions=merged_positions)
        return merged

    def _restart_iteration():
        nonlocal parts_iter, current_part
        nonlocal next_record_time_s, run_complete
        nonlocal total_task_steps, completed_parts, finalized
        nonlocal continuous_loop_steps, next_policy_time_s
        nonlocal held_merged_action
        nonlocal fixed_head_camera_ready
        _clear_snap_state()
        snap_fired_parts.clear()
        run_complete = False
        finalized = False
        total_task_steps = 0
        completed_parts = 0
        next_record_time_s = 0.0
        continuous_loop_steps = 0
        next_policy_time_s = None
        held_merged_action = None
        fixed_head_camera_ready = not fixed_head_camera_enabled
        # Remove any FixedJoints that snap_attach authored on previous
        # iterations. Joints live in USD and persist across my_world.stop()
        # / play(), so without cleanup the bolt (and any other snap part)
        # stays anchored to wherever the previous run's snap pinned it.
        _stage = omni.usd.get_context().get_stage()
        if _stage is not None:
            for _name in pc.PART_CONFIG.keys():
                _joint_path = f"/World/_snap_joint_{_name}"
                if is_prim_path_valid(_joint_path):
                    _stage.RemovePrim(_joint_path)
                    print(f"[setup] removed stale snap joint at {_joint_path}")
        # Restore scene-resident parts' xformOps to startup snapshot.
        restore_scene_part_xforms()
        parts_iter = iter(pc.part_order)
        current_part = None
        if args.pi05_continuous_episode:
            _start_continuous_episode()
        else:
            _start_next_part()

    def _run_residual_workflow():
        from policies.residual_injector import ResidualInjector
        from residual_env import ResidualEnv, ResidualEnvConfig
        from residual_isaac_backend import IsaacResidualBackend
        from residual_policy import make_residual_policy_adapter
        from residual_task import (
            BoundedSnapRewardConfig,
            SnapInsertionTaskConfig,
            make_residual_reward,
            make_residual_task,
        )

        policy_adapter = make_residual_policy_adapter(policy)

        control_hz = float(args.policy_control_hz or 10.0)
        # ``World.step`` advances one rendering interval, not one raw PhysX
        # substep.  This world renders at 10 Hz while PhysX substeps at 200 Hz,
        # so using ``env_info.physics_dt`` here would accidentally hold every
        # policy target for 20 rendering frames (2 seconds at the default
        # control rate).
        rendering_dt = float(my_world.get_rendering_dt())
        render_steps_per_action = max(1, int(round(1.0 / (control_hz * rendering_dt))))
        contact_view = None
        contact_view_error = None

        def reset_episode(part_name):
            nonlocal fixed_head_camera_ready, contact_view, contact_view_error
            if part_name != residual_part_name:
                raise ValueError(
                    f"residual target mismatch: {part_name!r} != "
                    f"{residual_part_name!r}"
                )
            contact_view = None
            contact_view_error = None
            _clear_snap_state()
            try:
                my_world.stop()
            except Exception:
                pass
            my_world.reset()
            _apply_init_joint_targets()
            L_controller.reset()
            R_controller.reset()
            _restart_iteration()
            my_world.play()

            # The task scene has no pre-authored contact sensor.  Create a
            # force view for stuck-event diagnostics after the selected
            # rigid body has been restored and the physics timeline restarted.
            try:
                from isaacsim.core.prims import RigidPrim

                rigid_body_path = (
                    current_snap_attacher.get_resolved_rigid_body_path(part_name)
                )
                contact_view = RigidPrim(
                    prim_paths_expr=rigid_body_path,
                    name=f"residual_{part_name}_contact_view",
                    track_contact_forces=True,
                    prepare_contact_sensors=True,
                )
                contact_view.initialize()
            except Exception as exc:
                contact_view = None
                contact_view_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[residual.eval] contact-force telemetry unavailable: "
                    f"{contact_view_error}",
                    flush=True,
                )

            warmup_steps = int(getattr(pc, "WARMUP_STEPS", 0))
            for _ in range(warmup_steps):
                _apply_init_joint_targets()
                if fixed_head_camera_enabled:
                    sync_fixed_camera_to_source(
                        head_depth_camera,
                        "/World/robotics/vega_1u_gripper/zed_depth_frame/headcam",
                    )
                my_world.step(render=True)
            if fixed_head_camera_enabled:
                sync_fixed_camera_to_source(
                    head_depth_camera,
                    "/World/robotics/vega_1u_gripper/zed_depth_frame/headcam",
                )
                fixed_head_camera_ready = True
                my_world.step(render=True)

        def get_attacher(part_name):
            if part_name != current_part:
                return None
            return current_snap_attacher

        def apply_cartesian_action(position, quaternion_wxyz, gripper):
            L_action = policy.L.forward(
                np.asarray(position, dtype=np.float64),
                np.asarray(quaternion_wxyz, dtype=np.float64),
                float(np.clip(gripper, 0.0, 0.6649704)),
            )
            articulation_controller.apply_action(_merge_left_with_right_hold(L_action))

        def advance():
            for render_step in range(render_steps_per_action):
                my_world.step(render=(render_step == render_steps_per_action - 1))

        def get_contact_force(part_name):
            del part_name
            if contact_view is None:
                return None
            try:
                forces = contact_view.get_net_contact_forces(
                    dt=float(env_info.physics_dt)
                )
                if hasattr(forces, "cpu"):
                    forces = forces.cpu()
                if hasattr(forces, "numpy"):
                    forces = forces.numpy()
                array = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
                return array[0].copy() if len(array) else None
            except Exception as exc:
                print(
                    f"[residual.eval] contact-force query failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                return None

        backend = IsaacResidualBackend(
            policy_adapter=policy_adapter,
            reset_episode=reset_episode,
            get_observation=_build_observation,
            get_attacher=get_attacher,
            apply_cartesian_action=apply_cartesian_action,
            advance=advance,
            get_contact_force=get_contact_force,
            prefix_exec_horizon=args.residual_prefix_horizon,
            part_config_provider=pc.get_part_config,
        )
        residual_part_cfg = pc.get_part_config(residual_part_name)
        residual_config_kwargs = {}
        task_config = None
        if args.residual_task == "snap_insertion":
            residual_snap_cfg = residual_part_cfg["snap"]
            if float(residual_snap_cfg.get("rot_tol_deg", 5.0)) < 0.0:
                # Axis-symmetric parts deliberately disable the snap orientation
                # gate.  Their full SO(3) error contains an irrelevant axial
                # rotation, so it must not reject or exit the residual window.
                residual_config_kwargs.update(
                    gate_enter_ori_rad=math.pi + 1e-6,
                    gate_exit_ori_rad=math.pi + 2e-6,
                )
            else:
                residual_config_kwargs.update(
                    gate_enter_ori_rad=math.radians(
                        args.residual_gate_enter_ori_deg
                    ),
                )
            task_config = SnapInsertionTaskConfig(
                target_name=residual_part_name,
                gate_enter_ori_rad=residual_config_kwargs.get(
                    "gate_enter_ori_rad", math.radians(45.0)
                ),
                gate_exit_ori_rad=residual_config_kwargs.get(
                    "gate_exit_ori_rad", math.radians(60.0)
                ),
            )
        if args.residual_reward == "bounded_snap":
            reward = make_residual_reward(
                args.residual_reward,
                config=BoundedSnapRewardConfig(
                    w_pos=1.0,
                    w_ori=1.0,
                    smooth_weight=args.residual_smooth_weight,
                ),
            )
        else:
            reward = make_residual_reward(args.residual_reward)
        if args.residual_task == "snap_insertion":
            residual_task = make_residual_task(
                args.residual_task,
                config=task_config,
                reward=reward,
            )
        else:
            residual_task = make_residual_task(
                args.residual_task,
                target_name=residual_part_name,
                part_config=residual_part_cfg,
                reward=reward,
            )
        env_config = ResidualEnvConfig(
            part_name=residual_part_name,
            physical_obs_dim=policy_adapter.observation_dim,
            w_pos=1.0,
            w_ori=1.0,
            smooth_weight=args.residual_smooth_weight,
            max_steps=args.residual_env_max_steps,
            max_reset_attempts=args.residual_max_reset_attempts,
            **residual_config_kwargs,
        )
        residual_env = ResidualEnv(
            backend,
            env_config,
            injector=ResidualInjector(
                delta_max_pos=args.residual_delta_max_pos,
                delta_max_ori=math.radians(args.residual_delta_max_ori_deg),
            ),
            task=residual_task,
        )

        if args.residual_td3_train:
            from residual_td3 import ReplayBuffer, TD3, TD3Config

            def wilson_interval(successes, trials, z=1.959963984540054):
                if trials <= 0:
                    return 0.0, 1.0
                probability = successes / trials
                denominator = 1.0 + z * z / trials
                center = (
                    probability + z * z / (2.0 * trials)
                ) / denominator
                half_width = z / denominator * math.sqrt(
                    probability * (1.0 - probability) / trials
                    + z * z / (4.0 * trials * trials)
                )
                return center - half_width, center + half_width

            td3_config = TD3Config(
                gamma=0.99,
                tau=0.005,
                learning_rate=3e-4,
                policy_freq=2,
            )
            agent = TD3(
                observation_dim=residual_env.observation_space.shape[0],
                action_dim=6,
                config=td3_config,
                device=args.td3_device,
                seed=args.residual_seed,
            )
            replay_buffer = ReplayBuffer(
                observation_dim=residual_env.observation_space.shape[0],
                action_dim=6,
                capacity=args.td3_buffer_capacity,
                seed=args.residual_seed,
            )
            rng = np.random.default_rng(args.residual_seed)
            os.makedirs(os.path.dirname(args.td3_log) or ".", exist_ok=True)
            os.makedirs(args.td3_checkpoint_dir, exist_ok=True)
            total_steps = 0
            training_done_reasons = {}

            def run_evaluation(marker_episode, log_file):
                successes = 0
                reasons = {}
                returns = []
                steps = []
                for eval_episode in range(1, args.td3_eval_episodes + 1):
                    observation, _ = residual_env.reset(
                        seed=(
                            args.residual_seed
                            + 1_000_000
                            + marker_episode * 1000
                            + eval_episode
                        )
                    )
                    episode_return = 0.0
                    while True:
                        action = agent.select_action(observation)
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = residual_env.step(action)
                        episode_return += reward
                        observation = next_observation
                        if terminated or truncated:
                            break
                    reason = info["done_reason"]
                    reasons[reason] = reasons.get(reason, 0) + 1
                    successes += int(reason == residual_task.success_reason)
                    returns.append(episode_return)
                    steps.append(info["rl_steps"])
                    record = {
                        "type": "evaluation_episode",
                        "training_episode": marker_episode,
                        "evaluation_episode": eval_episode,
                        "return": episode_return,
                        "steps": info["rl_steps"],
                        "done_reason": reason,
                    }
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    print(
                        f"[residual.td3.eval.episode] "
                        f"train_episode={marker_episode} "
                        f"episode={eval_episode:02d}/"
                        f"{args.td3_eval_episodes} "
                        f"return={episode_return:+.3f} "
                        f"steps={info['rl_steps']:02d} "
                        f"done={reason}",
                        flush=True,
                    )

                interval_low, interval_high = wilson_interval(
                    successes, args.td3_eval_episodes
                )
                summary = {
                    "type": "evaluation_summary",
                    "training_episode": marker_episode,
                    "episodes": args.td3_eval_episodes,
                    "success_count": successes,
                    "success_rate": successes / args.td3_eval_episodes,
                    "success_reason": residual_task.success_reason,
                    # Compatibility aliases used by HDMI telemetry tools.
                    "snap_count": successes,
                    "snap_rate": successes / args.td3_eval_episodes,
                    "wilson_95_low": interval_low,
                    "wilson_95_high": interval_high,
                    "done_reasons": reasons,
                    "mean_return": float(np.mean(returns)),
                    "mean_steps": float(np.mean(steps)),
                    "total_steps": total_steps,
                }
                log_file.write(json.dumps(summary) + "\n")
                log_file.flush()
                checkpoint_path = os.path.join(
                    args.td3_checkpoint_dir,
                    f"td3_episode_{marker_episode:04d}.pt",
                )
                agent.save(checkpoint_path, metadata=summary)
                print(
                    f"[residual.td3.eval] train_episode={marker_episode} "
                    f"snap={successes}/{args.td3_eval_episodes} "
                    f"rate={summary['snap_rate']:.3f} "
                    f"wilson95=[{interval_low:.3f},{interval_high:.3f}] "
                    f"reasons={reasons}",
                    flush=True,
                )
                return summary

            with open(args.td3_log, "w", encoding="utf-8") as log_file:
                configuration = {
                    "type": "configuration",
                    "algorithm": "TD3",
                    "base_policy_adapter": policy_adapter.name,
                    "residual_task": residual_task.name,
                    "residual_reward": args.residual_reward,
                    "observation_dim": residual_env.observation_space.shape[0],
                    "train_episodes": args.td3_train_episodes,
                    "warmup_steps": args.td3_warmup_steps,
                    "warmup_distribution": "normal",
                    "exploration_noise": args.td3_exploration_noise,
                    "batch_size": args.td3_batch_size,
                    "buffer_capacity": args.td3_buffer_capacity,
                    "eval_interval": args.td3_eval_interval,
                    "eval_episodes": args.td3_eval_episodes,
                    "delta_max_pos": args.residual_delta_max_pos,
                    "delta_max_ori_rad": math.radians(
                        args.residual_delta_max_ori_deg
                    ),
                    "residual_env": {
                        "max_steps": residual_env.config.max_steps,
                        "gate_enter_pos_m": residual_env.config.gate_enter_pos_m,
                        "gate_exit_pos_m": residual_env.config.gate_exit_pos_m,
                        "gate_enter_ori_rad": residual_env.config.gate_enter_ori_rad,
                        "gate_exit_ori_rad": residual_env.config.gate_exit_ori_rad,
                        "smooth_weight": residual_env.config.smooth_weight,
                        "terminal_reward": residual_env.config.terminal_reward,
                        "gate_exit_reward": residual_env.config.gate_exit_reward,
                        "stuck_reward": residual_env.config.stuck_reward,
                    },
                    "base_policy_task": os.environ.get(
                        "PI05_TASK", residual_part_name
                    ),
                    "td3": {
                        "gamma": td3_config.gamma,
                        "tau": td3_config.tau,
                        "learning_rate": td3_config.learning_rate,
                        "policy_freq": td3_config.policy_freq,
                        "policy_noise": td3_config.policy_noise,
                        "noise_clip": td3_config.noise_clip,
                        "hidden_dim": td3_config.hidden_dim,
                        "hidden_layers": td3_config.hidden_layers,
                    },
                }
                log_file.write(json.dumps(configuration) + "\n")

                for training_episode in range(1, args.td3_train_episodes + 1):
                    observation, reset_info = residual_env.reset(
                        seed=args.residual_seed + training_episode
                    )
                    episode_return = 0.0
                    critic_losses = []
                    actor_losses = []
                    while True:
                        if total_steps < args.td3_warmup_steps:
                            action = rng.normal(
                                0.0, args.td3_exploration_noise, size=6
                            )
                            action_source = "warmup_normal"
                        else:
                            action = agent.select_action(observation)
                            action += rng.normal(
                                0.0, args.td3_exploration_noise, size=6
                            )
                            action_source = "actor_plus_normal"
                        action = np.clip(action, -1.0, 1.0)
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = residual_env.step(action)
                        episode_done = bool(terminated or truncated)
                        replay_buffer.add(
                            observation,
                            action,
                            next_observation,
                            reward,
                            episode_done,
                        )
                        observation = next_observation
                        episode_return += reward
                        total_steps += 1

                        # TD3 warm-up is data collection only.  Starting
                        # gradient updates as soon as one batch is available
                        # would train the actor on a small, strongly biased
                        # prefix of the replay buffer while actions are still
                        # labeled as warm-up exploration.
                        if (
                            total_steps >= args.td3_warmup_steps
                            and len(replay_buffer) >= args.td3_batch_size
                        ):
                            losses = agent.train(
                                replay_buffer, batch_size=args.td3_batch_size
                            )
                            critic_losses.append(losses["critic_loss"])
                            if losses["actor_updated"]:
                                actor_losses.append(losses["actor_loss"])
                            update_record = {
                                "type": "update",
                                "training_episode": training_episode,
                                "total_steps": total_steps,
                                "action_source": action_source,
                                "critic_loss": losses["critic_loss"],
                                "actor_loss": (
                                    losses["actor_loss"]
                                    if losses["actor_updated"]
                                    else None
                                ),
                                "actor_updated": losses["actor_updated"],
                            }
                            log_file.write(json.dumps(update_record) + "\n")

                        if episode_done:
                            break

                    reason = info["done_reason"]
                    training_done_reasons[reason] = (
                        training_done_reasons.get(reason, 0) + 1
                    )
                    episode_record = {
                        "type": "training_episode",
                        "episode": training_episode,
                        "return": episode_return,
                        "steps": info["rl_steps"],
                        "done_reason": reason,
                        "snap": reason == "snap",
                        "gate_exit": reason == "gate_exit",
                        "total_steps": total_steps,
                        "buffer_size": len(replay_buffer),
                        "mean_critic_loss": (
                            float(np.mean(critic_losses))
                            if critic_losses
                            else None
                        ),
                        "mean_actor_loss": (
                            float(np.mean(actor_losses))
                            if actor_losses
                            else None
                        ),
                        "reset_elapsed_s": reset_info["reset_elapsed_s"],
                    }
                    log_file.write(json.dumps(episode_record) + "\n")
                    log_file.flush()
                    print(
                        f"[residual.td3.train] episode={training_episode:03d} "
                        f"steps={info['rl_steps']:02d} return={episode_return:+.3f} "
                        f"done={reason} buffer={len(replay_buffer)} "
                        f"critic={episode_record['mean_critic_loss']} "
                        f"gate_exit_rate="
                        f"{training_done_reasons.get('gate_exit', 0) / training_episode:.3f}",
                        flush=True,
                    )

                    if training_episode % args.td3_eval_interval == 0:
                        run_evaluation(training_episode, log_file)

                final_record = {
                    "type": "training_summary",
                    "episodes": args.td3_train_episodes,
                    "total_steps": total_steps,
                    "buffer_size": len(replay_buffer),
                    "done_reasons": training_done_reasons,
                    "gate_exit_rate": (
                        training_done_reasons.get("gate_exit", 0)
                        / args.td3_train_episodes
                    ),
                }
                log_file.write(json.dumps(final_record) + "\n")
                log_file.flush()
                print(
                    f"[residual.td3] complete={json.dumps(final_record)}",
                    flush=True,
                )
            return

        rng = np.random.default_rng(args.residual_seed)
        os.makedirs(os.path.dirname(args.residual_log) or ".", exist_ok=True)
        episode_summaries = []
        total_bottlenecks = {"x": 0, "y": 0, "z": 0}
        reset_times = []
        false_gate_entries = 0

        with open(args.residual_log, "w", encoding="utf-8") as log_file:
            for episode_index in range(1, args.residual_episodes + 1):
                rejection_counts_before = dict(residual_env.rejection_reasons)
                reset_started = time.perf_counter()
                try:
                    _, reset_info = residual_env.reset(
                        seed=args.residual_seed + episode_index
                    )
                except RuntimeError as exc:
                    reset_elapsed_s = time.perf_counter() - reset_started
                    reset_times.append(reset_elapsed_s)
                    rejection_delta = {
                        reason: count - rejection_counts_before.get(reason, 0)
                        for reason, count in residual_env.rejection_reasons.items()
                        if count - rejection_counts_before.get(reason, 0) > 0
                    }
                    prefix_failure_reason = max(
                        rejection_delta,
                        key=rejection_delta.get,
                        default="prefix_failure",
                    )
                    prefix_success = (
                        prefix_failure_reason
                        == f"prefix_{residual_task.success_reason}"
                    )
                    episode_summary = {
                        "type": "episode_summary",
                        "part_name": residual_part_name,
                        "episode": episode_index,
                        "action_mode": args.residual_action_mode,
                        "fixed_x": args.residual_fixed_x,
                        "delta_max_pos": args.residual_delta_max_pos,
                        "prefix_steps": None,
                        "steps": 0,
                        "done_reason": (
                            residual_task.success_reason
                            if prefix_success
                            else prefix_failure_reason
                        ),
                        "return": 100.0 if prefix_success else 0.0,
                        "dense_return": 0.0,
                        "terminal_return": 100.0 if prefix_success else 0.0,
                        "gate_exit_return": 0.0,
                        "stuck_return": 0.0,
                        "terminal_dominates_dense": None,
                        "gate_reactivations": 0,
                        "bottleneck_counts": {"x": 0, "y": 0, "z": 0},
                        "reset_elapsed_s": reset_elapsed_s,
                        "gate_entry_gripper_closed": False,
                        "gate_entry_part_grasped": False,
                        "gripper_hold_activated": (
                            residual_env.last_prefix_hold_activated
                        ),
                        "gripper_hold_prefix_step": (
                            residual_env.last_prefix_hold_step
                        ),
                        "reset_error": str(exc),
                    }
                    episode_summaries.append(episode_summary)
                    log_file.write(json.dumps(episode_summary) + "\n")
                    log_file.flush()
                    print(
                        f"[residual.eval] episode={episode_index:02d} "
                        f"prefix_result={prefix_failure_reason} "
                        f"elapsed_s={reset_elapsed_s:.3f}",
                        flush=True,
                    )
                    continue
                reset_times.append(reset_info["reset_elapsed_s"])
                if not (
                    reset_info["gate_entry_gripper_closed"]
                    and reset_info["gate_entry_part_grasped"]
                ):
                    false_gate_entries += 1
                dense_return = 0.0
                terminal_return = 0.0
                gate_exit_return = 0.0
                stuck_return = 0.0
                episode_return = 0.0
                episode_bottlenecks = {"x": 0, "y": 0, "z": 0}
                gate_reactivations = 0
                saw_inactive_after_entry = False

                while True:
                    if args.residual_action_mode == "zero":
                        action = np.zeros(6, dtype=np.float64)
                    elif args.residual_action_mode == "fixed-x":
                        action = np.array(
                            [args.residual_fixed_x, 0.0, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float64,
                        )
                    else:
                        action = rng.uniform(-1.0, 1.0, size=6)
                    _, reward, terminated, truncated, info = residual_env.step(action)
                    normalized = np.asarray(info["normalized_error"])
                    dense_reward = sum(
                        float(value)
                        for name, value in info["reward_components"].items()
                        if name not in {"terminal", "gate_exit", "stuck"}
                    )
                    dense_return += dense_reward
                    terminal_return += info["reward_terminal"]
                    gate_exit_return += info["reward_gate_exit"]
                    stuck_return += info["reward_stuck"]
                    episode_return += reward
                    axis = info["bottleneck_axis"]
                    episode_bottlenecks[axis] = (
                        episode_bottlenecks.get(axis, 0) + 1
                    )
                    total_bottlenecks[axis] = total_bottlenecks.get(axis, 0) + 1
                    if not info["gate_active"] and not (terminated or truncated):
                        saw_inactive_after_entry = True
                    if saw_inactive_after_entry and info["gate_active"]:
                        gate_reactivations += 1

                    record = {
                        "type": "step",
                        "part_name": residual_part_name,
                        "episode": episode_index,
                        "step": info["rl_steps"],
                        "action_mode": args.residual_action_mode,
                        "se3_error": np.asarray(info["se3_error"]).tolist(),
                        "se3_error_delta": np.asarray(
                            info["se3_error_delta"]
                        ).tolist(),
                        "normalized_error": normalized.tolist(),
                        "gate_active": info["gate_active"],
                        "reward_position": info["reward_position"],
                        "reward_orientation": info["reward_orientation"],
                        "reward_smooth": info["reward_smooth"],
                        "reward_terminal": info["reward_terminal"],
                        "reward_gate_exit": info["reward_gate_exit"],
                        "reward_stuck": info["reward_stuck"],
                        "reward_components": info["reward_components"],
                        "reward": reward,
                        "bottleneck_axis": axis,
                        "done_reason": info["done_reason"],
                        "bc_gripper": info["bc_gripper"],
                        "commanded_gripper": info["commanded_gripper"],
                        "gripper_override_active": info[
                            "gripper_override_active"
                        ],
                        "measured_gripper_open": info["measured_gripper_open"],
                        "position_error_change_m": info[
                            "position_error_change_m"
                        ],
                        "stagnant_steps": info["stagnant_steps"],
                        "contact_force": (
                            None
                            if info["contact_force"] is None
                            else np.asarray(info["contact_force"]).tolist()
                        ),
                        "part_pose": (
                            None
                            if info["part_pose"] is None
                            else np.asarray(info["part_pose"]).tolist()
                        ),
                    }
                    log_file.write(json.dumps(record) + "\n")
                    print(
                        f"[residual.eval] episode={episode_index:02d} "
                        f"step={info['rl_steps']:02d} "
                        f"normalized={np.array2string(normalized, precision=4, separator=',')} "
                        f"gate={int(info['gate_active'])} "
                        f"reward_pos={info['reward_position']:+.4f} "
                        f"reward_ori={info['reward_orientation']:+.4f} "
                        f"reward_smooth={info['reward_smooth']:+.6f} "
                        f"reward_stuck={info['reward_stuck']:+.1f} "
                        f"reward_gate_exit={info['reward_gate_exit']:+.1f} "
                        f"reward_term={info['reward_terminal']:+.1f} "
                        f"bottleneck={axis} done={info['done_reason'] or '-'}",
                        flush=True,
                    )
                    if terminated or truncated:
                        break

                terminal_dominates_dense = (
                    terminal_return > abs(dense_return)
                    if terminal_return > 0.0
                    else None
                )
                episode_summary = {
                    "type": "episode_summary",
                    "part_name": residual_part_name,
                    "episode": episode_index,
                    "action_mode": args.residual_action_mode,
                    "fixed_x": args.residual_fixed_x,
                    "delta_max_pos": args.residual_delta_max_pos,
                    "prefix_steps": reset_info["prefix_steps"],
                    "steps": info["rl_steps"],
                    "done_reason": info["done_reason"],
                    "return": episode_return,
                    "dense_return": dense_return,
                    "terminal_return": terminal_return,
                    "gate_exit_return": gate_exit_return,
                    "stuck_return": stuck_return,
                    "terminal_dominates_dense": terminal_dominates_dense,
                    "gate_reactivations": gate_reactivations,
                    "bottleneck_counts": episode_bottlenecks,
                    "reset_elapsed_s": reset_info["reset_elapsed_s"],
                    "gate_entry_gripper_closed": reset_info[
                        "gate_entry_gripper_closed"
                    ],
                    "gate_entry_part_grasped": reset_info[
                        "gate_entry_part_grasped"
                    ],
                    "gripper_hold_activated": True,
                    "gripper_hold_prefix_step": reset_info[
                        "gripper_hold_prefix_step"
                    ],
                }
                episode_summaries.append(episode_summary)
                log_file.write(json.dumps(episode_summary) + "\n")

            aggregate = {
                "type": "aggregate_summary",
                "part_name": residual_part_name,
                "base_policy_adapter": policy_adapter.name,
                "residual_task": residual_task.name,
                "residual_reward": args.residual_reward,
                "episodes": len(episode_summaries),
                "action_mode": args.residual_action_mode,
                "fixed_x": args.residual_fixed_x,
                "delta_max_pos": args.residual_delta_max_pos,
                "success_count": sum(
                    item["done_reason"] == residual_task.success_reason
                    for item in episode_summaries
                ),
                "success_reason": residual_task.success_reason,
                # Compatibility alias used by existing snap eval scripts.
                "snap_count": sum(
                    item["done_reason"] == residual_task.success_reason
                    for item in episode_summaries
                ),
                "gate_exit_count": sum(
                    item["done_reason"] == "gate_exit" for item in episode_summaries
                ),
                "gripper_open_count": sum(
                    item["done_reason"] == "gripper_open" for item in episode_summaries
                ),
                "stuck_count": sum(
                    item["done_reason"] == "stuck" for item in episode_summaries
                ),
                "max_steps_count": sum(
                    item["done_reason"] == "max_steps" for item in episode_summaries
                ),
                "prefix_failure_count": sum(
                    item["steps"] == 0
                    and item["done_reason"] != residual_task.success_reason
                    for item in episode_summaries
                ),
                "prefix_failure_reasons": {
                    reason: sum(
                        item["steps"] == 0
                        and item["done_reason"] != residual_task.success_reason
                        and item["done_reason"] == reason
                        for item in episode_summaries
                    )
                    for reason in sorted(
                        {
                            item["done_reason"]
                            for item in episode_summaries
                            if item["steps"] == 0
                            and item["done_reason"] != residual_task.success_reason
                        }
                    )
                },
                "gripper_hold_activation_count": sum(
                    item["gripper_hold_activated"] for item in episode_summaries
                ),
                "reset_attempts": residual_env.reset_attempts,
                "rejected_initializations": residual_env.rejected_initializations,
                "initialization_rejection_rate": (
                    residual_env.initialization_rejection_rate
                ),
                "rejection_reasons": residual_env.rejection_reasons,
                "mean_reset_elapsed_s": float(np.mean(reset_times)),
                "total_reset_elapsed_s": float(np.sum(reset_times)),
                "false_gate_entries": false_gate_entries,
                "pose_gate_blocked_ungrasped": (
                    residual_env.pose_gate_blocked_ungrasped
                ),
                "contact_force_setup_error": contact_view_error,
                "gate_reactivations": sum(
                    item["gate_reactivations"] for item in episode_summaries
                ),
                "bottleneck_counts": total_bottlenecks,
                "terminal_dominance": [
                    item["terminal_dominates_dense"]
                    for item in episode_summaries
                    if item["terminal_dominates_dense"] is not None
                ],
            }
            log_file.write(json.dumps(aggregate) + "\n")
            print(f"[residual.eval] aggregate={json.dumps(aggregate)}", flush=True)

    _rl_workflow = residual_workflow or grip_residual_workflow or grasp_residual_workflow
    if not _rl_workflow:
        _restart_iteration()
    if auto_play and not _rl_workflow:
        try:
            my_world.play()
        except Exception:
            pass

    try:
        if grip_residual_workflow or grasp_residual_workflow:
            if grip_residual_workflow:
                from residual_grip_train import (
                    run_grip_residual_training as _run_rl,
                )
            else:
                from residual_grasp_train import (
                    run_grasp_residual_training as _run_rl,
                )
            _run_rl(
                policy=policy,
                my_world=my_world,
                articulation_controller=articulation_controller,
                build_observation=_build_observation,
                merge_left_with_right_hold=_merge_left_with_right_hold,
                apply_init_joint_targets=_apply_init_joint_targets,
                restart_iteration=_restart_iteration,
                clear_snap_state=_clear_snap_state,
                L_controller=L_controller,
                R_controller=R_controller,
                args=args,
                video_recorder_class=FfmpegVideoRecorder,
            )
            return
        if residual_workflow:
            _run_residual_workflow()
            return
        while simulation_app.is_running():
            my_world.step(render=True)

            if run_complete and exit_on_complete:
                break

            if not my_world.is_playing():
                if my_world.is_stopped():
                    reset_needed = True
                if not auto_play:
                    continue

            if reset_needed:
                my_world.reset()
                reset_needed = False
                _apply_init_joint_targets()
                L_controller.reset()
                R_controller.reset()
                _restart_iteration()
                if auto_play:
                    try:
                        my_world.play()
                    except Exception:
                        pass

            if my_world.current_time_step_index == 0:
                _apply_init_joint_targets()
                L_controller.reset()
                R_controller.reset()
                _restart_iteration()
                if auto_play:
                    try:
                        my_world.play()
                    except Exception:
                        pass

            # Warmup: skip task logic until PhysX has had time to cook
            # colliders (SDF meshes in particular) and joints have settled to
            # their init drive targets.
            _warmup_steps = int(getattr(pc, "WARMUP_STEPS", 0))
            if (_warmup_steps > 0
                    and my_world.current_time_step_index < _warmup_steps):
                _apply_init_joint_targets()
                if args.pi05_continuous_episode:
                    empty_action = ArticulationAction(
                        joint_positions=[None] * len(dof_names)
                    )
                    articulation_controller.apply_action(
                        _merge_left_with_right_hold(empty_action)
                    )
                if fixed_head_camera_enabled:
                    sync_fixed_camera_to_source(
                        head_depth_camera,
                        "/World/robotics/vega_1u_gripper/"
                        "zed_depth_frame/headcam",
                    )
                if my_world.current_time_step_index == _warmup_steps - 1:
                    print(f"[setup] warmup done ({_warmup_steps} steps); "
                          f"starting task.")
                continue

            if fixed_head_camera_enabled and not fixed_head_camera_ready:
                position, orientation = sync_fixed_camera_to_source(
                    head_depth_camera,
                    "/World/robotics/vega_1u_gripper/zed_depth_frame/headcam",
                )
                fixed_head_camera_ready = True
                print(
                    f"[setup] froze pi0.5 head camera after warmup "
                    f"position={position.tolist()} "
                    f"orientation_usd_wxyz={orientation.tolist()}",
                    flush=True,
                )
                # Render the newly frozen pose before constructing the first
                # policy observation.
                continue

            if current_part is None and not args.pi05_continuous_episode:
                continue

            if args.pi05_five_request_diagnostic:
                obs, observation_timings = _build_observation(with_timings=True)
                diagnostic = getattr(policy, "five_request_diagnostic", None)
                if not callable(diagnostic):
                    raise RuntimeError(
                        "selected policy does not support five-request diagnostic"
                    )
                diagnostic(obs, args.pi05_five_request_log, observation_timings)
                print(
                    "[diagnostic] 5 predictions recorded; no robot action applied.",
                    flush=True,
                )
                run_complete = True
                break

            if args.pi05_one_step_dry_run:
                obs, observation_timings = _build_observation(with_timings=True)
                dry_run = getattr(policy, "one_step_dry_run", None)
                if not callable(dry_run):
                    raise RuntimeError(
                        "selected policy does not support one-step dry-run"
                    )
                dry_run(obs, args.pi05_dry_run_log, observation_timings)
                print(
                    "[dry-run] prediction recorded; no robot action applied.",
                    flush=True,
                )
                run_complete = True
                break

            if args.pi05_ik_dry_run:
                obs, observation_timings = _build_observation(with_timings=True)
                ik_dry_run = getattr(policy, "ik_dry_run", None)
                if not callable(ik_dry_run):
                    raise RuntimeError(
                        "selected policy does not support IK dry-run"
                    )
                ik_dry_run(
                    obs,
                    args.pi05_ik_dry_run_log,
                    observation_timings,
                )
                print(
                    "[dry-run] IK diagnostics recorded; action was not applied.",
                    flush=True,
                )
                run_complete = True
                break

            if args.pi05_continuous_episode:
                sim_time_s = (
                    my_world.current_time_step_index * env_info.physics_dt
                )
                policy_period_s = 1.0 / float(args.policy_control_hz)
                policy_due = (
                    next_policy_time_s is None
                    or sim_time_s + 1e-9 >= next_policy_time_s
                )
                video_due = (
                    video_recorder.enabled
                    and sim_time_s + 1e-9 >= next_record_time_s
                )
                obs = _build_observation() if policy_due or video_due else None

                if video_due:
                    video_recorder.write(obs.rgb.get(video_recorder.camera))
                    while next_record_time_s <= sim_time_s + 1e-9:
                        next_record_time_s += record_period_s

                for part_name, attacher in continuous_snap_attachers.items():
                    if attacher.attached:
                        snap_fired_parts.add(part_name)

                if (args.max_sim_seconds
                        and sim_time_s + 1e-9 >= args.max_sim_seconds):
                    run_complete = True
                    print(
                        f"[pi05-continuous] reached max-sim-seconds="
                        f"{args.max_sim_seconds:g}; ending.",
                        flush=True,
                    )
                    _finalize_iteration("max_sim_seconds")
                    break

                if policy_due:
                    if total_task_steps >= args.max_steps:
                        run_complete = True
                        print(
                            f"[pi05-continuous] reached max-policy-steps="
                            f"{args.max_steps}; ending.",
                            flush=True,
                        )
                        _finalize_iteration("max_policy_steps")
                        break
                    if policy.is_done(obs):
                        run_complete = True
                        print("[pi05-continuous] policy reported done.", flush=True)
                        _finalize_iteration("policy_done")
                        break

                    held_merged_action = _merge_left_with_right_hold(policy.act(obs))
                    total_task_steps += 1
                    part_step_count += 1
                    if next_policy_time_s is None:
                        next_policy_time_s = sim_time_s + policy_period_s
                    else:
                        while next_policy_time_s <= sim_time_s + 1e-9:
                            next_policy_time_s += policy_period_s
                    if total_task_steps == 1 or total_task_steps % 25 == 0:
                        print(
                            f"[pi05-continuous] policy_step={total_task_steps} "
                            f"sim_time_s={sim_time_s:.3f} "
                            f"snap_fired={sorted(snap_fired_parts)}",
                            flush=True,
                        )

                if held_merged_action is not None:
                    articulation_controller.apply_action(held_merged_action)
                continuous_loop_steps += 1
                continue

            obs = _build_observation()
            sim_time_s = my_world.current_time_step_index * env_info.physics_dt
            if video_recorder.enabled:
                if sim_time_s + 1e-9 >= next_record_time_s:
                    video_recorder.write(obs.rgb.get(video_recorder.camera))
                    next_record_time_s += record_period_s

            total_task_steps += 1
            if args.max_steps and total_task_steps >= args.max_steps:
                run_complete = True
                print(f"[setup] reached max-steps={args.max_steps}; "
                      "ending early.")
                _finalize_iteration("max_steps")
                break
            if (args.max_sim_seconds
                    and sim_time_s + 1e-9 >= args.max_sim_seconds):
                run_complete = True
                print(f"[setup] reached max-sim-seconds="
                      f"{args.max_sim_seconds:g}; ending early.")
                _finalize_iteration("max_sim_seconds")
                break

            # Latch snap-fired into the per-iteration record as soon as the
            # attacher reports it (so a snap that fires on the very last tick
            # before timeout still counts as pass at grading time).
            if (current_snap_attacher is not None
                    and current_snap_attacher.attached):
                snap_fired_parts.add(current_part)

            cfg = pc.get_part_config(current_part)
            is_snap_done = (cfg.get("release_mode") == "snap"
                            and current_snap_attacher is not None
                            and current_snap_attacher.attached)
            snap_tick_count = (
                int(getattr(current_snap_attacher, "_tick", 0))
                if current_snap_attacher is not None else 0
            )
            is_timeout = (
                part_step_count >= PER_PART_TIMEOUT_STEPS
                or snap_tick_count >= PER_PART_TIMEOUT_STEPS
            )

            if policy.is_done(obs) or is_snap_done or is_timeout:
                if is_timeout:
                    print(f"[setup] {current_part}: per-part timeout "
                          f"({PER_PART_TIMEOUT_STEPS} steps) — advancing.")
                _start_next_part()
                continue

            L_action = policy.act(obs)

            merged = _merge_left_with_right_hold(L_action)
            articulation_controller.apply_action(merged)

            part_step_count += 1
    finally:
        try:
            close_policy = getattr(policy, "close", None)
            if callable(close_policy):
                close_policy()
        finally:
            if getattr(video_recorder, "deferred", False):
                try:
                    simulation_app.close()
                finally:
                    video_recorder.close()
            else:
                try:
                    video_recorder.close()
                finally:
                    simulation_app.close()


if __name__ == "__main__":
    main()
