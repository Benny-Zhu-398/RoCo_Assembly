"""GROUPED vision DP: gripper quantization + hold-latch, plus an optional
END-POINT POSITION CLOSED-LOOP near the place target, switchable off /
heuristic / residual so the same hook can later carry a learned RL residual.

Gripper: quantize action[6] to DATA-space {close,open} (tunable boundary via
DP_GRIP_BIAS); snap parts latch closed until snap_fired.

End-point closed-loop (open parts only): once grasped AND the ee is within
DP_CL_TRIGGER_M of place_pos (XY-plane), correct the BC position toward
place_pos in XY only (z left to BC/gravity):
  mode=off       -> no correction (pure BC; RL/heuristic baseline)
  mode=heuristic -> pos += clip(gain*(place_pos-ee), max_step)  [hand-written]
  mode=residual  -> pos += clip(external delta, max_step)        [RL hook]

LOCK-AND-RELEASE: once dist_xy drops below DP_CL_LOCK_M, freeze the current
pose and force the gripper OPEN, ignoring all subsequent BC commands. This
exploits that the closed loop reaches sub-mm xy momentarily but oscillates
(BC keeps re-pulling to its own off-target landing each chunk): we lock at
the first entry into the lock window before it swings back, then let the
part drop vertically from the aligned xy. z is never pushed (open parts
settle by gravity).

Env:
  DP_CKPT_GROUPED, DP_SERVER_PY, DP_TARGET_PARTS, DP_N_ACTION_STEPS,
  DP_NUM_INFERENCE_STEPS, DP_SEED, DP_SERVER_LOG, DP_GRIP_DEBUG, DP_GRIP_BIAS
  DP_CL_MODE      off | heuristic | residual   (default off)
  DP_CL_TRIGGER_M xy-distance to place that arms the loop (default 0.05 m)
  DP_CL_GAIN      proportional gain, heuristic mode (default 0.5)
  DP_CL_MAX_STEP  per-step correction cap in meters (default 0.005)
  DP_CL_LOCK_M    xy-distance at which to lock pose + release (default 0.003;
                  set 0 to disable locking, keep pure proportional loop)
  DP_CL_DEBUG     "1" -> print loop arm/correction/lock
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)
_DP_DIR = os.path.join(os.path.dirname(_TASK_DIR), "training", "diffusion_policy")
if _DP_DIR not in sys.path:
    sys.path.insert(0, _DP_DIR)

from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from constants import CAMERA_KEYS, LEFT_STATE_IDX  # noqa: E402

_SIM_RGB_KEY = {"head": "head", "left_hand": "L_wrist", "right_hand": "R_wrist"}

_GRIPPER_DATA = {
    "gear_20teeth":  (0.0977, 0.1805),
    "gear_60teeth":  (0.1600, 0.3008),
    "rod_16mm":      (0.0602, 0.3008),
    "bolt_8mm":      (0.0602, 0.2256),
    "usb_a":         (0.0602, 0.2256),
    "hdmi":          (0.0602, 0.2256),
    "pin":           (0.0827, 0.3008),
    "battery_size1": (0.1053, 0.3008),
    "battery_size5": (0.0752, 0.3008),
}  # (close, open)


def _euler_xyz_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _to_uint8_hwc(img):
    if img is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.astype(np.uint8)


class DiffusionVisionGroupedGripPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        self.R = env_info.R_controller
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("DP_CKPT_GROUPED")
        if not ckpt:
            raise ValueError("set DP_CKPT_GROUPED to the grouped vision checkpoint .pt path")
        server_py = os.environ.get("DP_SERVER_PY")
        if not server_py:
            raise ValueError("set DP_SERVER_PY to the training venv's python")

        target_parts_env = os.environ.get("DP_TARGET_PARTS")
        if not target_parts_env:
            raise ValueError("set DP_TARGET_PARTS to the parts this checkpoint should drive.")
        self._target_parts = {p.strip() for p in target_parts_env.split(",") if p.strip()}
        n_action_steps_env = os.environ.get("DP_N_ACTION_STEPS")
        self._n_action_steps = int(n_action_steps_env) if n_action_steps_env else None

        self._grip_debug = os.environ.get("DP_GRIP_DEBUG", "0") == "1"
        self._grip_bias = float(os.environ.get("DP_GRIP_BIAS", "0.5"))

        # --- end-point closed-loop config ---
        self._cl_mode = os.environ.get("DP_CL_MODE", "off").strip().lower()
        if self._cl_mode not in ("off", "heuristic", "residual"):
            raise ValueError(f"DP_CL_MODE must be off|heuristic|residual, got {self._cl_mode}")
        self._cl_trigger = float(os.environ.get("DP_CL_TRIGGER_M", "0.05"))
        self._cl_gain = float(os.environ.get("DP_CL_GAIN", "0.5"))
        self._cl_max_step = float(os.environ.get("DP_CL_MAX_STEP", "0.005"))
        self._cl_lock = float(os.environ.get("DP_CL_LOCK_M", "0.003"))
        self._cl_debug = os.environ.get("DP_CL_DEBUG", "0") == "1"
        # DP_CL_BAN_BC_M > 0 (residual mode only): once ee xy is within this
        # distance of the place target, stop letting BC contribute to xy.
        # The RL residual is integrated into a frozen anchor (BC only keeps
        # its live z), so BC can no longer drag the closed loop back to its
        # own landing every action chunk. 0 = disabled (BC + per-step
        # residual, the original behaviour).
        self._cl_ban_m = float(os.environ.get("DP_CL_BAN_BC_M", "0.0"))
        # DP_CL_Z_DESCEND=1 (residual + ban-BC only): once handed off, the anchor
        # z is no longer slaved to BC's live z (BC for gear_60teeth hovers ~20cm
        # above the socket and never descends). Instead ramp anchor z toward a
        # fixed ee target at DP_CL_Z_DESCEND_RATE m/step. Target = DP_CL_Z_TARGET
        # if set, else place_pos.z + ee_offset.z (the pick-time ee/part gap).
        # Lock then also requires ee z within DP_CL_Z_LOCK_M of that target.
        self._cl_zdesc = os.environ.get("DP_CL_Z_DESCEND", "0") == "1"
        self._cl_zdesc_rate = float(os.environ.get("DP_CL_Z_DESCEND_RATE", "0.006"))
        _zt = os.environ.get("DP_CL_Z_TARGET")
        self._cl_zdesc_target = float(_zt) if _zt else None
        self._cl_zlock = float(os.environ.get("DP_CL_Z_LOCK_M", "0.01"))
        self._ee_off_z = 0.0
        self._bc_anchor = None
        self._residual = np.zeros(3, dtype=np.float64)  # RL injects via set_residual()
        self._last_bc_pos = None  # raw BC target xyz of the most recent act() step

        # --- grasp-phase residual (independent of the place-phase closed loop) ---
        # DP_GRASP_RES_M > 0 arms a separate xy residual on BC's pick target,
        # active only while NOT yet grasped and while ee xy is within this
        # distance of the canonical grasp xy (pick_pos + ee_offset). Injected
        # via set_grasp_residual(); trained by residual_grasp_train.py. It does
        # not touch the place phase, so DP_CL_MODE can be "off" here.
        self._grasp_res_m = float(os.environ.get("DP_GRASP_RES_M", "0.0"))
        self._grasp_max_step = float(
            os.environ.get("DP_GRASP_MAX_STEP", os.environ.get("DP_CL_MAX_STEP", "0.005"))
        )
        # Commit = stop correcting and PIN the xy target absolutely; BC then
        # supplies only z, so the descent is genuinely vertical from that
        # point (no BC xy drift, no shear on the part). Commit fires at the
        # first real Z-descent (ee has dropped DP_GRASP_COMMIT_DROP below its
        # peak while aligning) or, as a backstop, once within
        # DP_GRASP_COMMIT_Z of the grasp z -- whichever comes first.
        self._grasp_commit_drop = float(os.environ.get("DP_GRASP_COMMIT_DROP", "0.005"))
        # Don't let the Z-descent commit fire until the grasp xy is actually
        # aligned to within this (else the pin freezes a bad offset and the
        # gripper closes off-centre -> knock). The z backstop still forces a
        # commit if the actor never converges.
        self._grasp_commit_dxy = float(os.environ.get("DP_GRASP_COMMIT_DXY", "0.006"))
        self._grasp_commit_z = float(os.environ.get("DP_GRASP_COMMIT_Z", "0.02"))
        self._grasp_offset_max = float(os.environ.get("DP_GRASP_OFFSET_MAX", "0.03"))
        # Optional descent floor (fallback, OFF by default). DP_GRASP_Z_FLOOR=1
        # clamps the commanded grasp z to >= (baseline grasp z -
        # DP_GRASP_Z_FLOOR_MARGIN) so BC can't dip below the pick depth and
        # mash the part into the board. baseline grasp z = pick_pos.z +
        # ee_offset.z. Prints `[grasp] <part> z-floor: <bc> -> <floor>` (needs
        # DP_CL_DEBUG=1) when it clamps.
        self._grasp_z_floor_on = os.environ.get("DP_GRASP_Z_FLOOR", "0").lower() in (
            "1", "true", "yes", "on"
        )
        self._grasp_z_floor_margin = float(
            os.environ.get("DP_GRASP_Z_FLOOR_MARGIN", "0.002")
        )
        self._grasp_res = np.zeros(3, dtype=np.float64)
        self._grasp_offset = np.zeros(2, dtype=np.float64)  # integrated xy correction
        self._grasp_committed = False
        self._grasp_peak_z = None   # highest ee z seen while aligning
        self._grasp_hold_xy = None  # absolute xy target pinned at commit
        self._pick_ee_xy = None  # canonical grasp xy target, set in reset()
        self._grasp_ee_z = None  # canonical grasp z, set in reset()
        # DP_GRASP_ACTOR: path to a frozen grasp TD3 checkpoint. When set, the
        # policy runs that actor internally every pick-phase step (builds the
        # 11-D grasp obs itself) instead of waiting for set_grasp_residual().
        # Lets a trained grasp residual run during place training with no
        # trainer-side wiring.
        self._grasp_actor = None
        self._grasp_actor_scale = np.array(
            [50., 50., 50., 50., 1., 1., 1., 50., 50., 50., 1.], dtype=np.float64
        )
        self._grasp_prev_action = np.zeros(2, dtype=np.float64)
        self._gear_rest_xy = None
        _grasp_ckpt = os.environ.get("DP_GRASP_ACTOR", "").strip()
        if _grasp_ckpt and self._grasp_res_m > 0.0:
            self._load_grasp_actor(_grasp_ckpt)

        # --- hold-on-seat: once the placed part has settled within tol of
        # grade_pos for N consecutive steps, freeze the arm and open the
        # gripper -- stop sending BC/residual commands so nothing disturbs it.
        # DP_HOLD_ON_SEAT=1 to enable (default off). is_done() then reports
        # True so the harness advances immediately instead of waiting out the
        # per-part timeout.
        self._hold_on_seat = os.environ.get("DP_HOLD_ON_SEAT", "0").lower() in (
            "1", "true", "yes", "on"
        )
        self._hold_on_seat_steps = int(os.environ.get("DP_HOLD_ON_SEAT_STEPS", "5"))
        self._seat_tol_m = float(os.environ.get("DP_HOLD_ON_SEAT_TOL_M", "0.01"))
        self._seated = False
        self._released = False
        self._seat_count = 0
        self._seat_hold = None      # (pos, quat) frozen at seat
        self._grade_use_aabb = False

        cmd = [server_py, os.path.join(_TASK_DIR, "dp_server_vision_grouped.py"), ckpt]
        num_inf = os.environ.get("DP_NUM_INFERENCE_STEPS")
        if num_inf:
            cmd += ["--num-inference-steps", num_inf]
        seed = os.environ.get("DP_SEED")
        if seed:
            cmd += ["--seed", seed]

        keep = ("HOME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES", "UV_PYTHON_INSTALL_DIR")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        for k in keep:
            if k in os.environ:
                env[k] = os.environ[k]
        _log_path = os.environ.get("DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server_vision_grouped.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp-vision-grouped-qgrip] spawned server (ckpt={ckpt}, "
              f"targets={sorted(self._target_parts)}, n_action_steps={self._n_action_steps}, "
              f"grip_bias={self._grip_bias}, cl_mode={self._cl_mode}, "
              f"cl_trigger={self._cl_trigger}, cl_max_step={self._cl_max_step}, "
              f"cl_lock={self._cl_lock})", flush=True)
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server_vision_grouped died on startup (exit {self._proc.returncode}); see {_log_path}")

        self._skip = True
        self._queue = []
        self._grip_open = 0.0
        self._grip_close = 0.0
        self._cur_part = None
        self._is_snap = False
        self._grip_latched = False
        self._place_pos = None
        self._grasped_once = False
        # lock-and-release state
        self._place_locked = False
        self._locked_pos = None

    # ---- RL hook: inject the residual for the NEXT act() step ----
    def set_residual(self, delta_xyz):
        """RL training loop calls this each step with the residual policy's
        output (task-space xyz correction, meters). Only used when
        DP_CL_MODE=residual and the loop is armed (grasped + near place)."""
        self._residual = np.asarray(delta_xyz, dtype=np.float64).reshape(3)

    def set_grasp_residual(self, delta_xy):
        """Grasp-phase RL hook: per-step xy correction rate (metres) for the
        NEXT act(). Integrated into a slew-limited offset on BC's pick target
        while ungrasped, near the grasp xy, and above the commit height.
        Accepts a 2- or 3-vector; z is ignored. Ignored when DP_GRASP_ACTOR
        is loaded (the policy drives the grasp residual itself)."""
        v = np.asarray(delta_xy, dtype=np.float64).reshape(-1)
        self._grasp_res = np.array(
            [float(v[0]), float(v[1]), 0.0], dtype=np.float64
        )

    def _load_grasp_actor(self, path):
        import torch
        from residual_td3 import Actor

        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {}) or {}
        actor = Actor(
            11, 2,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            hidden_layers=int(cfg.get("hidden_layers", 3)),
        )
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        self._grasp_actor = actor
        print(f"[grasp] loaded frozen grasp actor from {path}", flush=True)

    def _gear_mesh_xy(self):
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(f"/World/parts/{self._cur_part}")
            if not root or not root.IsValid():
                return None
            deepest, depth = None, -1
            for p in Usd.PrimRange(root):
                if p.GetTypeName() != "Mesh":
                    continue
                d = p.GetPath().pathString.count("/")
                if d > depth:
                    depth, deepest = d, p
            if deepest is None:
                return None
            t = UsdGeom.XformCache().GetLocalToWorldTransform(deepest).ExtractTranslation()
            return np.array([float(t[0]), float(t[1])], dtype=np.float64)
        except Exception:
            return None

    def _part_world_pos(self, use_aabb=False):
        """3-D world position of the target part's deepest mesh -- translation,
        or world-AABB midpoint when use_aabb (batteries). Matches
        run_pick_place._grade_task so the seat check agrees with grading."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(f"/World/parts/{self._cur_part}")
            if not root or not root.IsValid():
                return None
            deepest, depth = None, -1
            for p in Usd.PrimRange(root):
                if p.GetTypeName() != "Mesh":
                    continue
                dd = p.GetPath().pathString.count("/")
                if dd > depth:
                    depth, deepest = dd, p
            if deepest is None:
                return None
            if use_aabb:
                bb = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
                ).ComputeWorldBound(deepest)
                m = bb.ComputeAlignedRange().GetMidpoint()
                return np.array([float(m[0]), float(m[1]), float(m[2])], dtype=np.float64)
            t = UsdGeom.XformCache().GetLocalToWorldTransform(deepest).ExtractTranslation()
            return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
        except Exception:
            return None

    def _grasp_actor_rate(self, obs, ee, d):
        """Run the frozen grasp actor -> 2-D xy rate in [-1, 1]."""
        import torch

        bc = self._last_bc_pos if self._last_bc_pos is not None else ee
        gear_xy = self._gear_mesh_xy()
        gear_off = (0.0 if (gear_xy is None or self._gear_rest_xy is None)
                    else float(np.linalg.norm(gear_xy - self._gear_rest_xy)))
        raw = np.array([
            ee[0] - self._pick_ee_xy[0], ee[1] - self._pick_ee_xy[1], d,
            ee[2] - self._grasp_ee_z, float(obs.L_gripper_position),
            float(self._grasp_prev_action[0]), float(self._grasp_prev_action[1]),
            bc[0] - self._pick_ee_xy[0], bc[1] - self._pick_ee_xy[1],
            gear_off, 1.0 if self._grasp_committed else 0.0,
        ], dtype=np.float64)
        x = (raw * self._grasp_actor_scale).astype(np.float32)
        with torch.no_grad():
            out = self._grasp_actor(torch.as_tensor(x).unsqueeze(0)).squeeze(0).numpy()
        return np.clip(np.asarray(out, dtype=np.float64), -1.0, 1.0)

    def _grasp_correction(self, obs, pos):
        """One job: align the gripper xy onto the gear centre while the arm
        is still high. The RL rate is integrated (slew-limited) into an xy
        offset on BC's pick target. At the first real Z-descent the xy is
        PINNED absolutely -- from then on BC supplies only z, so the descent
        is vertical and the residual never shears the part. No-op once
        grasped, disarmed, or ee xy outside the window."""
        if self._grasp_res_m <= 0.0 or self._grasped_once or self._pick_ee_xy is None:
            return pos
        ee = np.asarray(obs.ee_pose_L[0], dtype=np.float64)
        d = float(np.linalg.norm(ee[:2] - self._pick_ee_xy))
        if d > self._grasp_res_m:
            return pos
        if self._grasp_actor is not None and self._gear_rest_xy is None:
            self._gear_rest_xy = self._gear_mesh_xy()

        # Optional descent floor (DP_GRASP_Z_FLOOR=1). Off -> z_out == BC z.
        z_out = float(pos[2])
        if self._grasp_z_floor_on and self._grasp_ee_z is not None:
            z_floor = self._grasp_ee_z - self._grasp_z_floor_margin
            if z_out < z_floor:
                if self._cl_debug:
                    print(f"[grasp] {self._cur_part} z-floor: {z_out:.4f} -> "
                          f"{z_floor:.4f}", flush=True)
                z_out = z_floor

        if not self._grasp_committed:
            if self._grasp_peak_z is None or ee[2] > self._grasp_peak_z:
                self._grasp_peak_z = float(ee[2])
            descending = (ee[2] <= self._grasp_peak_z - self._grasp_commit_drop
                          and d <= self._grasp_commit_dxy)
            backstop = (self._grasp_ee_z is not None
                        and ee[2] <= self._grasp_ee_z + self._grasp_commit_z)
            if descending or backstop:
                self._grasp_committed = True
                self._grasp_hold_xy = (pos[:2] + self._grasp_offset).copy()
                if self._cl_debug:
                    print(f"[grasp] {self._cur_part} COMMIT ee_z={ee[2]:.4f} "
                          f"({'descent' if descending else 'backstop'}) "
                          f"hold_xy={self._grasp_hold_xy.round(4).tolist()}", flush=True)

        if self._grasp_committed:
            if self._cl_debug:
                print(f"[grasp] {self._cur_part} d_xy={d*1000:.1f}mm z={ee[2]:.4f} "
                      f"committed=1 hold_xy={self._grasp_hold_xy.round(4).tolist()}",
                      flush=True)
            return np.array(
                [self._grasp_hold_xy[0], self._grasp_hold_xy[1], z_out],
                dtype=np.float64,
            )

        if self._grasp_actor is not None:
            rate = self._grasp_actor_rate(obs, ee, d)
            self._grasp_prev_action = rate
            step = rate * self._grasp_max_step
        else:
            step = self._grasp_res[:2].copy()
        n = float(np.linalg.norm(step))
        if n > self._grasp_max_step:  # per-step slew limit
            step = step * (self._grasp_max_step / n)
        self._grasp_offset = self._grasp_offset + step
        on = float(np.linalg.norm(self._grasp_offset))
        if on > self._grasp_offset_max:  # total offset cap
            self._grasp_offset = self._grasp_offset * (self._grasp_offset_max / on)
        if self._cl_debug:
            print(f"[grasp] {self._cur_part} d_xy={d*1000:.1f}mm z={ee[2]:.4f} "
                  f"committed=0 offset={self._grasp_offset.round(4).tolist()}", flush=True)
        return np.array([
            pos[0] + self._grasp_offset[0],
            pos[1] + self._grasp_offset[1],
            z_out,
        ], dtype=np.float64)

    # ---- length-prefixed pickle pipe ----
    def _send(self, obj):
        b = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(b)) + b)
        self._proc.stdin.flush()

    def _recv(self):
        h = self._proc.stdout.read(4)
        if len(h) < 4:
            raise RuntimeError("dp_server_vision_grouped closed")
        n = struct.unpack(">I", h)[0]
        buf = b""
        while len(buf) < n:
            buf += self._proc.stdout.read(n - len(buf))
        return pickle.loads(buf)

    # ---- Policy API ----
    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._skip = target.name not in self._target_parts
        self._queue = []
        if self._skip:
            return
        self._send({"cmd": "reset", "part": target.name})
        self._recv()
        if target.name in _GRIPPER_DATA:
            self._grip_close, self._grip_open = _GRIPPER_DATA[target.name]
        else:
            self._grip_open = float(getattr(target, "gripper_open", 0.0) or 0.0)
            self._grip_close = float(getattr(target, "gripper_close", 0.0) or 0.0)
            print(f"[qgrip] WARNING: {target.name} not in _GRIPPER_DATA; using config", flush=True)
        self._cur_part = target.name
        self._is_snap = (target.release_mode == "snap")
        self._grip_latched = False
        # end-point target: prefer grade_pos then place_pos
        place = getattr(target, "grade_pos", None)
        if place is None:
            place = getattr(target, "place_pos", None)
        self._place_pos = None if place is None else np.asarray(place, dtype=np.float64)
        self._grasped_once = False
        self._residual = np.zeros(3, dtype=np.float64)
        self._bc_anchor = None
        self._grasp_res = np.zeros(3, dtype=np.float64)
        self._grasp_offset = np.zeros(2, dtype=np.float64)
        self._grasp_committed = False
        self._grasp_peak_z = None
        self._grasp_hold_xy = None
        self._grasp_prev_action = np.zeros(2, dtype=np.float64)
        self._gear_rest_xy = None
        self._seated = False
        self._released = False
        self._seat_count = 0
        self._seat_hold = None
        self._grade_use_aabb = bool(
            (target.extra or {}).get("grade_use_aabb", False)
            if getattr(target, "extra", None) else False
        )
        # canonical grasp pose = pick object pos + ee_offset (baseline convention)
        pp = getattr(target, "pick_pos", None)
        if pp is not None:
            pp = np.asarray(pp, dtype=np.float64).reshape(-1)
            eo = (target.extra or {}).get("ee_offset") if getattr(target, "extra", None) else None
            eo = np.asarray(eo, dtype=np.float64).reshape(-1) if eo is not None else np.zeros(3)
            self._pick_ee_xy = (pp[:2] + eo[:2]).copy()
            self._grasp_ee_z = float(pp[2] + eo[2])
            self._ee_off_z = float(eo[2])
        else:
            self._pick_ee_xy = None
            self._grasp_ee_z = None
            self._ee_off_z = 0.0
        # reset lock state each episode
        self._place_locked = False
        self._locked_pos = None
        if self._grip_debug or self._cl_debug:
            boundary = self._grip_close + self._grip_bias * (self._grip_open - self._grip_close)
            print(f"[qgrip] reset part={target.name} snap={self._is_snap} "
                  f"open={self._grip_open:.4f} close={self._grip_close:.4f} "
                  f"bias={self._grip_bias} boundary={boundary:.4f} "
                  f"cl_mode={self._cl_mode} lock={self._cl_lock} "
                  f"place={None if self._place_pos is None else self._place_pos.round(4).tolist()}",
                  flush=True)

    def _build_state(self, obs: Observation) -> np.ndarray:
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        Lp, Lq = self.L.end_effector.get_world_pose()
        Rp, Rq = self.R.end_effector.get_world_pose()
        full = np.concatenate([
            np.asarray(Lp).reshape(-1)[:3], np.asarray(Lq).reshape(-1)[:4],
            np.asarray(Rp).reshape(-1)[:3], np.asarray(Rq).reshape(-1)[:4],
            q[self._Li], q[self._Ri], qd[self._Li], qd[self._Ri],
            [float(q[self._Lg])],
            [float(q[self._Rg]) if self._Rg is not None else 0.0],
        ]).astype(np.float64)
        assert full.shape == (44,), f"expected 44-D full state, got {full.shape}"
        return full[LEFT_STATE_IDX].astype(np.float32)

    def _quantize_gripper(self, grip_pred: float) -> float:
        boundary = self._grip_close + self._grip_bias * (self._grip_open - self._grip_close)
        return self._grip_close if grip_pred < boundary else self._grip_open

    def _endpoint_correction(self, obs, pos, is_closed):
        """Returns (pos, keep_closed). keep_closed=False means the caller
        should force the gripper OPEN (used after lock, to release the part)."""
        if self._cl_mode == "off" or self._is_snap or self._place_pos is None:
            return pos, is_closed
        if not self._grasped_once:
            return pos, is_closed

        # Already locked: freeze pose, force release, ignore all BC commands.
        if self._place_locked:
            return self._locked_pos, False

        # Correction only while actually holding the part.
        if not is_closed:
            return pos, is_closed

        ee = np.asarray(obs.ee_pose_L[0], dtype=np.float64)
        dist_xy = float(np.linalg.norm(ee[:2] - self._place_pos[:2]))

        # BC-ban hand-off (residual mode): once inside DP_CL_BAN_BC_M, BC no
        # longer touches xy. The residual is integrated into a frozen anchor
        # (BC keeps only its live z), so BC can't drag the loop back to its
        # own landing each chunk. Once handed off, stay handed off.
        anchor_pos = None
        z_tgt = None
        if (self._cl_mode == "residual" and self._cl_ban_m > 0.0
                and (self._bc_anchor is not None or dist_xy < self._cl_ban_m)):
            if self._bc_anchor is None:
                self._bc_anchor = pos.copy()
            step = self._residual.copy()
            step[2] = 0.0
            nrm = float(np.linalg.norm(step))
            if nrm > self._cl_max_step:
                step = step * (self._cl_max_step / nrm)
            self._bc_anchor[0] += step[0]
            self._bc_anchor[1] += step[1]
            if self._cl_zdesc:
                # Scripted descent: BC's z is ignored; ramp toward a fixed ee z.
                z_tgt = (self._cl_zdesc_target if self._cl_zdesc_target is not None
                         else float(self._place_pos[2] + self._ee_off_z))
                dz = z_tgt - self._bc_anchor[2]
                dz = max(-self._cl_zdesc_rate, min(self._cl_zdesc_rate, dz))
                self._bc_anchor[2] = self._bc_anchor[2] + dz
                # IK-paced: never command z more than one ramp-step below the
                # live ee, so the descent tracks real arm motion (as BC's own
                # z would) instead of running open-loop ahead of the arm.
                z_lead_floor = float(ee[2]) - max(2.0 * self._cl_zdesc_rate, 0.01)
                if self._bc_anchor[2] < z_lead_floor:
                    self._bc_anchor[2] = z_lead_floor
            else:
                self._bc_anchor[2] = pos[2]  # follow BC's live z descent
            # Keep the integrator bounded: never let the RL anchor run more
            # than _cl_trigger from the place target (e.g. when it commands
            # an unreachable pose and IK fails, so the ee never catches up).
            off = self._bc_anchor[:2] - self._place_pos[:2]
            off_n = float(np.linalg.norm(off))
            if off_n > self._cl_trigger:
                self._bc_anchor[:2] = (
                    self._place_pos[:2] + off * (self._cl_trigger / off_n)
                )
            anchor_pos = self._bc_anchor.copy()

        # Lock the moment xy is close enough: freeze current pose + release,
        # before the loop can oscillate back out.
        z_ok = True
        if self._cl_zdesc and z_tgt is not None:
            z_ok = abs(float(ee[2]) - z_tgt) <= self._cl_zlock
        if self._cl_lock > 0.0 and dist_xy < self._cl_lock and z_ok:
            self._place_locked = True
            self._locked_pos = (anchor_pos if anchor_pos is not None else pos).copy()
            if self._cl_debug:
                print(f"[cl] {self._cur_part} LOCKED at dist_xy={dist_xy*1000:.2f}mm "
                      f"pos={self._locked_pos.round(4).tolist()} -> hold & release",
                      flush=True)
            return self._locked_pos, False

        # Handed off to the RL anchor: BC's xy is ignored from here.
        if anchor_pos is not None:
            if self._cl_debug:
                print(f"[cl] {self._cur_part} mode=residual-banbc "
                      f"dist_xy={dist_xy*1000:.1f}mm "
                      f"anchor_xy={anchor_pos[:2].round(4).tolist()} "
                      f"anchor_z={anchor_pos[2]:.4f} "
                      f"z_tgt={'-' if z_tgt is None else round(z_tgt,4)} "
                      f"ee_z={float(ee[2]):.4f} "
                      f"step={step[:2].round(4).tolist()}", flush=True)
            return anchor_pos, is_closed

        # Not yet armed by trigger window -> no correction.
        if dist_xy > self._cl_trigger:
            return pos, is_closed

        # Proportional (heuristic) or injected (residual) xy-only correction.
        if self._cl_mode == "heuristic":
            delta = self._cl_gain * (self._place_pos - ee)
        else:
            delta = self._residual.copy()
        delta[2] = 0.0  # xy-only; z left to BC/gravity
        nrm = float(np.linalg.norm(delta))
        if nrm > self._cl_max_step:
            delta = delta * (self._cl_max_step / nrm)
        if self._cl_debug:
            print(f"[cl] {self._cur_part} mode={self._cl_mode} "
                  f"dist_xy={dist_xy*1000:.1f}mm delta={delta.round(4).tolist()}", flush=True)
        return pos + delta, is_closed

    def act(self, obs: Observation):
        if self._skip:
            return None

        # Already seated: hold the frozen pose, gripper open. Stop consuming
        # BC actions entirely so nothing disturbs the placed part.
        if self._seated and self._seat_hold is not None:
            hp, hq = self._seat_hold
            return self.L.forward(hp, hq, self._grip_open)

        if not self._queue:
            state = self._build_state(obs)
            images = {cam: _to_uint8_hwc(obs.rgb.get(_SIM_RGB_KEY[cam]))
                      for cam in CAMERA_KEYS}
            self._send({"state": state, "images": images})
            horizon_action = np.asarray(self._recv()["action_horizon"], np.float64)
            n = self._n_action_steps or horizon_action.shape[0]
            n = min(n, horizon_action.shape[0])
            self._queue = list(horizon_action[:n])

        a = self._queue.pop(0)
        pos = np.asarray(a[:3], dtype=np.float64)
        self._last_bc_pos = pos.copy()  # pre-correction BC target (RL obs term)
        pos = self._grasp_correction(obs, pos)  # grasp-phase xy nudge (no-op post-grasp)
        quat = _euler_xyz_to_quat_wxyz(a[3], a[4], a[5])
        grip_pred = float(a[6])
        grip = self._quantize_gripper(grip_pred)

        if self._is_snap:
            if grip == self._grip_close:
                self._grip_latched = True
            if self._grip_latched:
                if bool(getattr(obs, "snap_fired", False)):
                    grip = self._grip_open
                    self._grip_latched = False
                else:
                    grip = self._grip_close
        else:
            if grip == self._grip_close:
                self._grasped_once = True

        is_closed = (grip == self._grip_close)

        # end-point correction / lock (open parts). Returns possibly-frozen pos
        # and whether to keep the gripper closed.
        pos, keep_closed = self._endpoint_correction(obs, pos, is_closed)
        if (not self._is_snap) and self._place_locked and not keep_closed:
            grip = self._grip_open  # forced release after lock

        action = self.L.forward(pos, quat, grip)

        # Released = grasped at some point and the gripper is now open. Seat
        # detection only starts AFTER release -- a still-held part passing
        # near grade_pos is not "placed".
        if (not self._is_snap) and self._grasped_once and (grip != self._grip_close):
            self._released = True

        # Seat detection: post-release, part settled within tol of grade_pos
        # for N consecutive steps.
        if (self._hold_on_seat and not self._seated and self._released
                and self._place_pos is not None and not self._is_snap):
            pw = self._part_world_pos(self._grade_use_aabb)
            if pw is not None and float(np.linalg.norm(pw - self._place_pos)) < self._seat_tol_m:
                self._seat_count += 1
                if self._seat_count >= self._hold_on_seat_steps:
                    self._seated = True
                    self._seat_hold = (
                        np.asarray(pos, dtype=np.float64).copy(),
                        np.asarray(quat, dtype=np.float64).copy(),
                    )
                    print(f"[seat] {self._cur_part} settled "
                          f"(<{self._seat_tol_m*1000:.0f}mm x{self._hold_on_seat_steps}) "
                          f"-> hold pose + gripper open", flush=True)
            else:
                self._seat_count = 0

        return action

    def is_done(self, obs: Observation) -> bool:
        if self._skip:
            return True
        return bool(self._seated)

    def __del__(self):
        try:
            self._proc.terminate()
        except Exception:
            pass