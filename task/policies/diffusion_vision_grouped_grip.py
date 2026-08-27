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
        self._residual = np.zeros(3, dtype=np.float64)  # RL injects via set_residual()

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

        # Lock the moment xy is close enough: freeze current pose + release,
        # before the loop can oscillate back out.
        if self._cl_lock > 0.0 and dist_xy < self._cl_lock:
            self._place_locked = True
            self._locked_pos = pos.copy()
            if self._cl_debug:
                print(f"[cl] {self._cur_part} LOCKED at dist_xy={dist_xy*1000:.2f}mm "
                      f"pos={self._locked_pos.round(4).tolist()} -> hold & release",
                      flush=True)
            return self._locked_pos, False

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

        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        if self._skip:
            return True
        return False

    def __del__(self):
        try:
            self._proc.terminate()
        except Exception:
            pass