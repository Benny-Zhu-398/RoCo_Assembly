"""Deploy this repo's own state-only per-part Diffusion Policy checkpoint
(training/diffusion_policy/{model.py,train.py}) closed-loop in
task/run_pick_place.py.

This is NOT policies/diffusion_lerobot.py -- that file deploys **lerobot's**
vision-based DiffusionPolicy (44-D state incl. images, HF checkpoint
format). This adapter is for the state-only, no-vision `DiffusionPolicyNet`
checkpoints under training/diffusion_policy/outputs/<part>/final.pt, which
had no closed-loop integration at all before this file.

Same process-isolation pattern as diffusion_lerobot.py and the same reason:
the training venv (torch + diffusers, see training/diffusion_policy/
requirements.txt) can't share Isaac's interpreter (numpy<2 pin, Isaac's own
torch build), so the model runs in dp_server_stateonly.py in a SEPARATE
process and this adapter talks to it over a stdin/stdout pickle pipe.

Only ONE part is driven by the model per run; every other part in
pc.part_order is skipped immediately (act() returns None, is_done() is True
right after reset()) so a run reaches the target part in a handful of
sim-seconds regardless of where it sits in part_order. Point
--max-parts at (1-based index of target part in pc.part_order) to also stop
the harness right after grading it, e.g. battery_size1 is the 8th entry ->
--max-parts 8.

=== STATE CONSTRUCTION (must byte-for-byte match training/diffusion_policy/
dataset.py's PartSequenceDataset, which reads observation.state columns in
the order training/diffusion_policy/constants.py::STATE_NAMES_FULL and then
slices to the 22-D left-arm-only vector via LEFT_STATE_IDX = range(0,7) +
range(14,21) + range(28,35) + [42]) ===

  full 44-D order: left_ee_xyz(3), left_ee_quat_wxyz(4),
                    right_ee_xyz(3), right_ee_quat_wxyz(4),
                    left_jpos(7), right_jpos(7),
                    left_jvel(7), right_jvel(7),
                    left_gripper(1), right_gripper(1)

This file builds that exact 44-D vector every step and slices it with
`constants.LEFT_STATE_IDX` (imported, not re-derived) before sending it to
the server -- so a future change to that slice only has to happen in one
place. constants.py has zero heavy deps (no numpy/torch requirement beyond
what Isaac's env already provides), so importing it directly here is safe.

=== GRIPPER UNITS -- checked against norm_stats.json, not guessed ===

Both STATE and ACTION gripper dims are RAW joint radians here, NOT a [0,1]
open-ratio like policies/diffusion_lerobot.py's convention for the
lerobot-trained vision policy. Confirmed straight from
training/diffusion_policy/norm_stats.json: state "left_gripper" has
mean=0.146, std=0.081 -- squarely inside the same raw range as
action_min/max for "left_gripper" (0.060 to 0.301) and
param_config.py's PART_CONFIG gripper_open/gripper_close values (~0.0-0.2
rad) -- not a 0-1-centered ratio. So: no GRIPPER_OPEN_LIMIT rescale
anywhere in this file, for either building state or consuming the
predicted action's gripper dim. (An earlier version of this file assumed
the ratio convention by analogy with diffusion_lerobot.py; that analogy
was wrong for this dataset -- re-check norm_stats.json yourself if this
checkpoint is ever retrained against a different data export.)

Env vars:
  DP_CKPT_STATEONLY   path to the checkpoint .pt (required, e.g.
                       training/diffusion_policy/outputs/battery_size1/final.pt)
  DP_SERVER_PY        python for the training venv (required, e.g.
                       .venv-dp/bin/python -- see training/diffusion_policy/README.md)
  DP_TARGET_PARTS     comma-separated part names to actually drive with the
                       model; every other pc.part_order entry is
                       skipped (default: "battery_size1")
  DP_N_ACTION_STEPS   how many steps of each predicted horizon to execute
                       before re-querying the model (receding-horizon
                       control -- this repo had NO n_action_steps concept
                       before this file; default: horizon, i.e. execute the
                       whole predicted chunk before re-querying, same as
                       offline evaluate.py implicitly does per-chunk)
  DP_NUM_INFERENCE_STEPS  DDIM steps override, forwarded to
                       dp_server_stateonly.py's --num-inference-steps
                       (default: checkpoint's own config value, 16)
  DP_SEED             sampling seed forwarded to the server (default 0)
  DP_SERVER_LOG       stderr log path for the server subprocess (default:
                       task/dp_server_stateonly.log)
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
from constants import LEFT_STATE_IDX  # noqa: E402


def _rotvec_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_rotvec([rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class DiffusionStateOnlyPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        self.R = env_info.R_controller
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("DP_CKPT_STATEONLY")
        if not ckpt:
            raise ValueError("set DP_CKPT_STATEONLY to the state-only checkpoint .pt path")
        server_py = os.environ.get("DP_SERVER_PY")
        if not server_py:
            raise ValueError("set DP_SERVER_PY to the training venv's python "
                              "(see training/diffusion_policy/README.md)")

        self._target_parts = {
            p.strip() for p in os.environ.get("DP_TARGET_PARTS", "battery_size1").split(",") if p.strip()
        }
        n_action_steps_env = os.environ.get("DP_N_ACTION_STEPS")
        self._n_action_steps = int(n_action_steps_env) if n_action_steps_env else None

        cmd = [server_py, os.path.join(_TASK_DIR, "dp_server_stateonly.py"), ckpt]
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
        _log_path = os.environ.get("DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server_stateonly.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp-stateonly] spawned inference server (ckpt={ckpt}, "
              f"targets={sorted(self._target_parts)}, n_action_steps={self._n_action_steps})",
              flush=True)
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server_stateonly died on startup (exit {self._proc.returncode}); see {_log_path}")

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
            raise RuntimeError("dp_server_stateonly closed")
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
        """Rebuild the 44-D training-time observation.state, in
        STATE_NAMES_FULL order, then slice to the 22-D left-arm vector the
        checkpoint expects via the imported LEFT_STATE_IDX (single source of
        truth shared with training/diffusion_policy/constants.py)."""
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)

        Lp, Lq = self.L.end_effector.get_world_pose()
        Rp, Rq = self.R.end_effector.get_world_pose()

        # Raw joint radians, not a ratio -- see module docstring's GRIPPER UNITS note.
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
            self._send({"state": state})
            horizon_action = np.asarray(self._recv()["action_horizon"], np.float64)  # (horizon, 7)
            n = self._n_action_steps or horizon_action.shape[0]
            n = min(n, horizon_action.shape[0])
            self._queue = list(horizon_action[:n])

        a = self._queue.pop(0)
        pos = a[:3]
        quat = _rotvec_to_quat_wxyz(a[3], a[4], a[5])
        grip = float(a[6])  # raw joint radians -- see module docstring's GRIPPER UNITS note
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
