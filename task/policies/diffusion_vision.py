"""Deploy this repo's own VISION-conditioned per-part Diffusion Policy
checkpoint (training/diffusion_policy/{model.py,vision.py,train.py} with
cfg.model.use_vision=True) closed-loop in task/run_pick_place.py.

Sibling of diffusion_stateonly.py -- same process-isolation pattern (model
runs in dp_server_vision.py in a SEPARATE process, talked to over a
stdin/stdout pickle pipe, for the same numpy/torch-version-conflict reason
documented there), same state-construction / rotation / gripper-unit
handling (all copied verbatim, not re-derived, so this file inherits those
fixes rather than re-risking the bugs they fixed -- see the ACTION ROTATION
CONVENTION and GRIPPER UNITS notes below). The only new thing here is
grabbing two camera frames off `obs.rgb` each step and sending them
alongside state.

This is NOT policies/diffusion_lerobot.py -- that file deploys **lerobot's**
vision-based DiffusionPolicy and has two landmines this file deliberately
does NOT share (flagged during the state-only-vs-vision migration
discussion, 2026-08-13): it decodes action rotation as rotvec (wrong for
tools/roco2026_by_part -- see below) and assumes gripper is a [0,1] ratio
via GRIPPER_OPEN_LIMIT (also wrong for this dataset). Neither mistake is
repeated here.

=== CAMERA KEY MAPPING -- dataset side vs. sim side use different names ===

training/diffusion_policy/constants.py::CAMERA_KEYS = ("head", "left_hand")
is the dataset's naming (matches tools/roco2026_by_part's
observation.images.{head,left_hand} video keys). The Isaac harness's
Observation.rgb dict instead uses ("head", "L_wrist", "R_wrist") -- see
diffusion_lerobot.py's existing usage. _SIM_RGB_KEY below is the one place
that mapping lives; if CAMERA_KEYS ever grows to include right_hand, add its
entry here too rather than guessing "R_wrist" is right by analogy.

=== STATE CONSTRUCTION / GRIPPER UNITS / ACTION ROTATION CONVENTION ===

Identical to task/policies/diffusion_stateonly.py -- see that file's module
docstring for the full explanation (byte-for-byte state layout via
constants.LEFT_STATE_IDX, why gripper is raw joint radians not a [0,1]
ratio, why action rotation is Euler XYZ extrinsic not rotvec). Copied here
rather than imported so this file has no runtime dependency on
diffusion_stateonly.py, matching this package's existing per-adapter
self-containment style.

Env vars:
  DP_CKPT_VISION      path to the vision checkpoint .pt (required, e.g.
                       training/diffusion_policy/outputs_vision/hdmi/final.pt)
  DP_SERVER_PY        python for the training venv (required, e.g.
                       .venv-train/Scripts/python.exe)
  DP_TARGET_PARTS     comma-separated part names to actually drive with the
                       model; every other pc.part_order entry is
                       skipped (default: "battery_size1")
  DP_N_ACTION_STEPS   how many steps of each predicted horizon to execute
                       before re-querying the model (default: horizon)
  DP_NUM_INFERENCE_STEPS  DDIM steps override, forwarded to
                       dp_server_vision.py's --num-inference-steps
                       (default: checkpoint's own config value)
  DP_SEED             sampling seed forwarded to the server (default 0)
  DP_SERVER_LOG        stderr log path for the server subprocess (default:
                       task/dp_server_vision.log)
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

# dataset camera_key -> Observation.rgb dict key (see module docstring's
# CAMERA KEY MAPPING note).
_SIM_RGB_KEY = {"head": "head", "left_hand": "L_wrist", "right_hand": "R_wrist"}


def _euler_xyz_to_quat_wxyz(rx, ry, rz):
    """Euler XYZ EXTRINSIC angles -> wxyz quat -- see diffusion_stateonly.py's
    ACTION ROTATION CONVENTION note. NOT axis-angle/rotvec (that was the bug
    in diffusion_lerobot.py); scipy's lowercase 'xyz' is the extrinsic
    convention tools/roco2026_by_part's action rotation dims actually use."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _to_uint8_hwc(img):
    """Observation.rgb entries -> (H,W,3) uint8, None -> zeros (same
    fallback convention as diffusion_lerobot.py's _resize_rgb -- a missing
    camera degrades to a black frame rather than crashing the rollout).
    Actual resizing to the checkpoint's training resolution happens
    server-side (dp_server_vision.py), not here -- this adapter sends
    whatever resolution the harness's camera actually renders at."""
    if img is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.astype(np.uint8)


class DiffusionVisionPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        self.R = env_info.R_controller
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("DP_CKPT_VISION")
        if not ckpt:
            raise ValueError("set DP_CKPT_VISION to the vision checkpoint .pt path")
        server_py = os.environ.get("DP_SERVER_PY")
        if not server_py:
            raise ValueError("set DP_SERVER_PY to the training venv's python "
                              "(see training/diffusion_policy/README.md)")

        self._target_parts = {
            p.strip() for p in os.environ.get("DP_TARGET_PARTS", "battery_size1").split(",") if p.strip()
        }
        n_action_steps_env = os.environ.get("DP_N_ACTION_STEPS")
        self._n_action_steps = int(n_action_steps_env) if n_action_steps_env else None

        cmd = [server_py, os.path.join(_TASK_DIR, "dp_server_vision.py"), ckpt]
        num_inf = os.environ.get("DP_NUM_INFERENCE_STEPS")
        if num_inf:
            cmd += ["--num-inference-steps", num_inf]
        seed = os.environ.get("DP_SEED")
        if seed:
            cmd += ["--seed", seed]

        # CLEAN env -- see diffusion_lerobot.py for why (Isaac's interpreter
        # exports PYTHONPATH/LD_LIBRARY_PATH/CARB_* that would make the
        # training-venv subprocess load Isaac's own torch/libs and crash).
        keep = ("HOME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES", "UV_PYTHON_INSTALL_DIR")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        for k in keep:
            if k in os.environ:
                env[k] = os.environ[k]
        _log_path = os.environ.get("DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server_vision.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp-vision] spawned inference server (ckpt={ckpt}, "
              f"targets={sorted(self._target_parts)}, n_action_steps={self._n_action_steps})",
              flush=True)
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server_vision died on startup (exit {self._proc.returncode}); see {_log_path}")

        self._skip = True
        self._queue = []  # list of (7,) raw actions still to execute from the last predicted horizon

    # ---- length-prefixed pickle pipe ----
    def _send(self, obj):
        b = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(b)) + b)
        self._proc.stdin.flush()

    def _recv(self):
        h = self._proc.stdout.read(4)
        if len(h) < 4:
            raise RuntimeError("dp_server_vision closed")
        n = struct.unpack(">I", h)[0]
        buf = b""
        while len(buf) < n:
            buf += self._proc.stdout.read(n - len(buf))
        return pickle.loads(buf)

    # ---- Policy API ----
    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._skip = target.name not in self._target_parts
        self._queue = []
        if not self._skip:
            self._send({"cmd": "reset"})
            self._recv()

    def _build_state(self, obs: Observation) -> np.ndarray:
        """Byte-for-byte identical to diffusion_stateonly.py::_build_state --
        see that file for the full field-order rationale."""
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)

        Lp, Lq = self.L.end_effector.get_world_pose()
        Rp, Rq = self.R.end_effector.get_world_pose()

        # Raw joint radians, not a ratio -- see diffusion_stateonly.py's GRIPPER UNITS note.
        full = np.concatenate([
            np.asarray(Lp).reshape(-1)[:3], np.asarray(Lq).reshape(-1)[:4],
            np.asarray(Rp).reshape(-1)[:3], np.asarray(Rq).reshape(-1)[:4],
            q[self._Li], q[self._Ri], qd[self._Li], qd[self._Ri],
            [float(q[self._Lg])],
            [float(q[self._Rg]) if self._Rg is not None else 0.0],
        ]).astype(np.float64)
        assert full.shape == (44,), f"expected 44-D full state, got {full.shape}"
        return full[LEFT_STATE_IDX].astype(np.float32)

    def act(self, obs: Observation):
        if self._skip:
            return None

        if not self._queue:
            state = self._build_state(obs)
            images = {
                cam: _to_uint8_hwc(obs.rgb.get(_SIM_RGB_KEY[cam]))
                for cam in CAMERA_KEYS
            }
            self._send({"state": state, "images": images})
            horizon_action = np.asarray(self._recv()["action_horizon"], np.float64)  # (horizon, 7)
            n = self._n_action_steps or horizon_action.shape[0]
            n = min(n, horizon_action.shape[0])
            self._queue = list(horizon_action[:n])

        a = self._queue.pop(0)
        pos = a[:3]
        quat = _euler_xyz_to_quat_wxyz(a[3], a[4], a[5])
        grip = float(a[6])  # raw joint radians -- see diffusion_stateonly.py's GRIPPER UNITS note
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        if self._skip:
            return True
        return False  # no learned termination signal; harness advances on snap fire / per-part timeout

    def __del__(self):
        try:
            self._proc.terminate()
        except Exception:
            pass
