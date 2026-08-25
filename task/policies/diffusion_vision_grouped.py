"""Deploy this repo's own GROUPED VISION-conditioned Diffusion Policy
checkpoint (training/diffusion_policy/{grouped_model.py,train.py}'s
--group path, e.g. outputs_grouped/connectors/final.pt) closed-loop in
task/run_pick_place.py.

Sibling of diffusion_vision.py -- same process-isolation pattern, same
state-construction / rotation / gripper-unit handling (copied verbatim, see
that file's module docstring for the full explanation), same camera-key
mapping. The one real difference: a grouped checkpoint answers for every
part in its group (e.g. connectors = {usb_a, hdmi}) through ONE shared
model, not a single part fixed at process startup, so `reset()` now tells
the server which part the upcoming episode is for (dp_server_vision_grouped
.py's "reset" message grows a "part" field -- see that file's module
docstring and grouped_model.py's "SCOPE OF THIS PASS" note, which flagged
this exact protocol gap when GroupedDiffusionPolicyNet was added).

Env vars:
  DP_CKPT_GROUPED      path to the grouped vision checkpoint .pt (required,
                       e.g. training/diffusion_policy/outputs_grouped/
                       connectors/final.pt)
  DP_SERVER_PY        python for the training venv (required, e.g.
                       .venv-train/Scripts/python.exe)
  DP_TARGET_PARTS     comma-separated part names to actually drive with the
                       model; every other pc.part_order entry is skipped.
                       Required (no default -- unlike diffusion_vision.py's
                       single-checkpoint "battery_size1" default, there is
                       no part that makes sense across every group). Must
                       be a subset of the checkpoint's own ckpt["parts"] --
                       dp_server_vision_grouped.py raises on the first
                       reset() for any part outside that list.
  DP_N_ACTION_STEPS   how many steps of each predicted horizon to execute
                       before re-querying the model (default: horizon)
  DP_NUM_INFERENCE_STEPS  DDIM steps override, forwarded to
                       dp_server_vision_grouped.py's --num-inference-steps
                       (default: checkpoint's own config value)
  DP_SEED             sampling seed forwarded to the server (default 0)
  DP_SERVER_LOG        stderr log path for the server subprocess (default:
                       task/dp_server_vision_grouped.log)
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
# CAMERA KEY MAPPING note in diffusion_vision.py).
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
    server-side (dp_server_vision_grouped.py), not here -- this adapter
    sends whatever resolution the harness's camera actually renders at."""
    if img is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a.astype(np.uint8)


class DiffusionVisionGroupedPolicy(Policy):
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
            raise ValueError("set DP_SERVER_PY to the training venv's python "
                              "(see training/diffusion_policy/README.md)")

        target_parts_env = os.environ.get("DP_TARGET_PARTS")
        if not target_parts_env:
            raise ValueError(
                "set DP_TARGET_PARTS to the parts this grouped checkpoint should drive -- "
                "no default (unlike diffusion_vision.py) since there is no part that makes "
                "sense across every group."
            )
        self._target_parts = {p.strip() for p in target_parts_env.split(",") if p.strip()}
        n_action_steps_env = os.environ.get("DP_N_ACTION_STEPS")
        self._n_action_steps = int(n_action_steps_env) if n_action_steps_env else None

        cmd = [server_py, os.path.join(_TASK_DIR, "dp_server_vision_grouped.py"), ckpt]
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
        _log_path = os.environ.get("DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server_vision_grouped.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp-vision-grouped] spawned inference server (ckpt={ckpt}, "
              f"targets={sorted(self._target_parts)}, n_action_steps={self._n_action_steps})",
              flush=True)
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server_vision_grouped died on startup (exit {self._proc.returncode}); see {_log_path}")

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
        if not self._skip:
            self._send({"cmd": "reset", "part": target.name})
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
