"""Deploy a LeRobot Pi0.5 policy in the Isaac Sim harness."""
from __future__ import annotations

from collections import deque
import os
import pickle
import shlex
import struct
import subprocess
import time

import numpy as np

from policy_api import EnvInfo, Observation, PartTarget, Policy

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_REMOTE_DEFAULTS = {
    "PI05_REMOTE_HOST": "yudongluo@128.2.178.27",
    "PI05_SSH_EXE": r"C:\Windows\System32\OpenSSH\ssh.exe",
    "PI05_REMOTE_PYTHON": (
        "/media/iam-lab/strange_external/yudongluo/pi05/"
        "envs/lerobot-py312/bin/python"
    ),
    "PI05_REMOTE_SERVER": "/home/yudongluo/user/Roco/RoCo_Assembly/task/pi05_server.py",
    "PI05_REMOTE_CHECKPOINT": (
        "/media/iam-lab/strange_external/yudongluo/pi05/outputs/"
        "roco_pi05_short_20260723_1520/checkpoints/000050/pretrained_model"
    ),
    "PI05_REMOTE_CUDA_VISIBLE_DEVICES": "2",
}

GRIPPER_OPEN_LIMIT = 0.6649704
_IMG_H, _IMG_W = 240, 320
_SAFETY_MAX_TRANSLATION_M = 0.005
_SAFETY_MAX_ROTATION_RAD = np.deg2rad(5.0)
_SAFETY_HOLD_TRANSLATION_M = 0.10
_SAFETY_HOLD_ROTATION_RAD = np.deg2rad(45.0)
_SAFETY_MAX_JOINT_STEP_RAD = np.deg2rad(5.0)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def _positive_env_int(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return value


def _required_env(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"set {name}")
    return value


def _remote_setting(name):
    return os.environ.get(name, _REMOTE_DEFAULTS[name])


def _display_command(command):
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def build_pi05_sidecar_launch():
    """Return (command, environment, checkpoint, mode) for the sidecar."""
    if _env_flag("PI05_REMOTE"):
        checkpoint = _remote_setting("PI05_REMOTE_CHECKPOINT")
        cuda_devices = shlex.quote(_remote_setting("PI05_REMOTE_CUDA_VISIBLE_DEVICES"))
        server_command = " ".join(
            [
                "exec",
                "env",
                f"CUDA_VISIBLE_DEVICES={cuda_devices}",
                "PI05_DEVICE=cuda",
                "TOKENIZERS_PARALLELISM=false",
                shlex.quote(_remote_setting("PI05_REMOTE_PYTHON")),
                shlex.quote(_remote_setting("PI05_REMOTE_SERVER")),
                shlex.quote(checkpoint),
            ]
        )
        remote_command = '. "$HOME/pi05_env.sh" 1>&2 && ' + server_command
        command = [
            _remote_setting("PI05_SSH_EXE"),
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            _remote_setting("PI05_REMOTE_HOST"),
            remote_command,
        ]
        return command, os.environ.copy(), checkpoint, "remote"

    checkpoint = _required_env("PI05_CKPT")
    server_py = _required_env("PI05_SERVER_PY")
    server_script = os.environ.get("PI05_SERVER", os.path.join(_TASK_DIR, "pi05_server.py"))

    keep = (
        "HOME",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PI05_DEVICE",
        "PI05_TASK",
        "TOKENIZERS_PARALLELISM",
    )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key in keep:
        if key in os.environ:
            env[key] = os.environ[key]
    cuda_devices = os.environ.get("PI05_CUDA_VISIBLE_DEVICES")
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    return [server_py, server_script, checkpoint], env, checkpoint, "local"


def _resize_rgb(img):
    if img is None:
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[0] == _IMG_H and a.shape[1] == _IMG_W:
        return a.astype(np.uint8, copy=False)
    try:
        import cv2

        return cv2.resize(a, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        ys = max(1, a.shape[0] // _IMG_H)
        xs = max(1, a.shape[1] // _IMG_W)
        out = a[::ys, ::xs, :3]
        return out[:_IMG_H, :_IMG_W].astype(np.uint8)


def _euler_xyz_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _filter_left_action(action, current_position, current_quat_wxyz, current_gripper):
    """Limit one absolute left-arm action relative to the measured EE pose."""
    from scipy.spatial.transform import Rotation

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    current_position = np.asarray(current_position, dtype=np.float64).reshape(3)
    current_quat_wxyz = np.asarray(current_quat_wxyz, dtype=np.float64).reshape(4)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("safety filter requires one finite 14-D action")
    if not np.isfinite(current_position).all() or not np.isfinite(current_quat_wxyz).all():
        raise ValueError("safety filter requires a finite current EE pose")
    quat_norm = float(np.linalg.norm(current_quat_wxyz))
    if quat_norm < 1e-8:
        raise ValueError("current EE quaternion has zero norm")
    current_quat_wxyz = current_quat_wxyz / quat_norm

    current_rotation = Rotation.from_quat(
        [
            current_quat_wxyz[1],
            current_quat_wxyz[2],
            current_quat_wxyz[3],
            current_quat_wxyz[0],
        ]
    )
    target_rotation = Rotation.from_euler("xyz", action[3:6])
    translation_delta = action[:3] - current_position
    translation_norm = float(np.linalg.norm(translation_delta))
    rotation_delta = target_rotation * current_rotation.inv()
    rotation_delta_rotvec = rotation_delta.as_rotvec()
    rotation_angle = float(np.linalg.norm(rotation_delta_rotvec))

    hold_reason = None
    if translation_norm > _SAFETY_HOLD_TRANSLATION_M:
        hold_reason = "translation"
    if rotation_angle > _SAFETY_HOLD_ROTATION_RAD:
        hold_reason = "rotation" if hold_reason is None else "translation+rotation"

    filtered = action.copy()
    if hold_reason is not None:
        filtered[:3] = current_position
        filtered[3:6] = current_rotation.as_euler("xyz")
        filtered[6] = float(np.clip(current_gripper, 0.0, 1.0))
    else:
        translation_scale = min(
            1.0,
            _SAFETY_MAX_TRANSLATION_M / max(translation_norm, 1e-12),
        )
        rotation_scale = min(
            1.0,
            _SAFETY_MAX_ROTATION_RAD / max(rotation_angle, 1e-12),
        )
        filtered[:3] = current_position + translation_scale * translation_delta
        safe_rotation = (
            Rotation.from_rotvec(rotation_scale * rotation_delta_rotvec)
            * current_rotation
        )
        filtered[3:6] = safe_rotation.as_euler("xyz")

    safe_translation_norm = float(np.linalg.norm(filtered[:3] - current_position))
    safe_rotation = Rotation.from_euler("xyz", filtered[3:6])
    safe_rotation_angle = float(
        np.linalg.norm((safe_rotation * current_rotation.inv()).as_rotvec())
    )
    return filtered, {
        "held": hold_reason is not None,
        "hold_reason": hold_reason,
        "raw_translation_m": translation_norm,
        "raw_rotation_deg": float(np.degrees(rotation_angle)),
        "safe_translation_m": safe_translation_norm,
        "safe_rotation_deg": float(np.degrees(safe_rotation_angle)),
    }


def _guard_left_joint_action(
    joint_positions,
    current_positions,
    left_indices,
    gripper_index,
):
    """Hold the left arm if IK requests an unsafe one-step joint jump."""
    targets = list(joint_positions)
    current = np.asarray(current_positions, dtype=np.float64).reshape(-1)
    left_indices = list(left_indices)
    if len(targets) != current.size:
        raise ValueError(
            f"joint action has {len(targets)} values, current state has {current.size}"
        )

    deltas = []
    invalid_indices = []
    for index in left_indices:
        target = targets[index]
        if target is None or not np.isfinite(float(target)):
            invalid_indices.append(index)
            deltas.append(np.inf)
        else:
            deltas.append(abs(float(target) - float(current[index])))

    max_delta_rad = float(max(deltas, default=0.0))
    held = bool(invalid_indices or max_delta_rad > _SAFETY_MAX_JOINT_STEP_RAD)
    if held:
        for index in left_indices:
            targets[index] = float(current[index])
        targets[gripper_index] = float(current[gripper_index])

    return targets, {
        "held": held,
        "reason": "invalid_ik" if invalid_indices else ("joint_jump" if held else None),
        "max_delta_rad": max_delta_rad,
        "max_delta_deg": float(np.degrees(max_delta_rad)),
        "invalid_indices": invalid_indices,
    }


class Pi05LeRobotPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self._proc = None
        self._err = None
        self._command_display = None
        self._log_path = None
        self._mode = None
        self._exec_horizon = _positive_env_int("PI05_EXEC_HORIZON", 1)
        self._safety_filter = _env_flag("PI05_SAFETY_FILTER")
        self._action_cache = deque()
        self._control_step = 0
        self._replan_index = 0
        self._client_log_path = os.path.abspath(
            os.environ.get(
                "PI05_CLIENT_LOG",
                os.path.join(os.path.dirname(_TASK_DIR), "artifacts", "pi05_action_cache.log"),
            )
        )
        os.makedirs(os.path.dirname(self._client_log_path), exist_ok=True)
        with open(self._client_log_path, "w", encoding="utf-8") as client_log:
            client_log.write(f"PI05_EXEC_HORIZON={self._exec_horizon}\n")
            client_log.write(f"PI05_SAFETY_FILTER={int(self._safety_filter)}\n")
        self.L = env_info.L_controller
        self.R = getattr(env_info, "R_controller", None)
        if self.L is None:
            raise ValueError("Pi05LeRobotPolicy requires env_info.L_controller")

        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        command, env, ckpt, mode = build_pi05_sidecar_launch()
        log_path = os.environ.get("PI05_SERVER_LOG", os.path.join(_TASK_DIR, "pi05_server.log"))
        self._command_display = _display_command(command)
        self._log_path = os.path.abspath(log_path)
        self._mode = mode
        try:
            self._err = open(self._log_path, "w")
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._err,
                env=env,
                cwd=_TASK_DIR,
            )
        except Exception as exc:
            error = self._sidecar_error("failed to start pi0.5 sidecar", "not started")
            self.close()
            raise error from exc
        if self._proc.stdin is None or self._proc.stdout is None:
            error = self._sidecar_error("failed to open pi0.5 sidecar pipes")
            self.close()
            raise error
        print(f"[pi05] spawned {mode} inference server (ckpt={ckpt})", flush=True)

        time.sleep(2)
        if self._proc.poll() is not None:
            error = self._sidecar_error("pi0.5 sidecar died on startup")
            self.close()
            raise error

    def _sidecar_error(self, message, returncode=None):
        if returncode is None:
            proc = getattr(self, "_proc", None)
            returncode = proc.poll() if proc is not None else "not started"
        label = "ssh command" if self._mode == "remote" else "command"
        return RuntimeError(
            f"{message}\n"
            f"{label}: {self._command_display}\n"
            f"return code: {returncode}\n"
            f"log file: {self._log_path}"
        )

    def _log_cache(self, message):
        with open(self._client_log_path, "a", encoding="utf-8") as client_log:
            client_log.write(message + "\n")

    def _send(self, obj):
        self._send_payload(pickle.dumps(obj))

    def _send_payload(self, payload):
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("pi05_server is not running")
        try:
            self._proc.stdin.write(struct.pack(">I", len(payload)) + payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._sidecar_error("pi0.5 sidecar closed while receiving request") from exc

    def _recv(self):
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("pi05_server is not running")
        header = self._proc.stdout.read(4)
        if len(header) < 4:
            raise self._sidecar_error("pi0.5 sidecar closed before sending response")
        size = struct.unpack(">I", header)[0]
        buf = b""
        while len(buf) < size:
            chunk = self._proc.stdout.read(size - len(buf))
            if not chunk:
                raise self._sidecar_error("pi0.5 sidecar closed while sending response")
            buf += chunk
        return pickle.loads(buf)

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._action_cache.clear()
        self._control_step = 0
        self._replan_index = 0
        self._send({"cmd": "reset", "task": os.environ.get("PI05_TASK")})
        reply = self._recv()
        if not reply.get("ok"):
            raise RuntimeError(f"pi05 reset failed: {reply!r}")
        self._log_cache("reset client_cache=0 server_queue=cleared")

    def _build_state(self, obs: Observation) -> np.ndarray:
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        if self.L is not None:
            Lp, Lq = self.L.end_effector.get_world_pose()
        else:
            Lp, Lq = obs.ee_pose_L
        if self.R is not None:
            Rp, Rq = self.R.end_effector.get_world_pose()
        else:
            Rp, Rq = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def ratio(v):
            return float(np.clip(v / GRIPPER_OPEN_LIMIT, 0, 1))

        return np.concatenate(
            [
                np.asarray(Lp).reshape(-1)[:3],
                np.asarray(Lq).reshape(-1)[:4],
                np.asarray(Rp).reshape(-1)[:3],
                np.asarray(Rq).reshape(-1)[:4],
                q[self._Li],
                q[self._Ri],
                qd[self._Li],
                qd[self._Ri],
                [ratio(q[self._Lg])],
                [ratio(q[self._Rg]) if self._Rg is not None else 0.0],
            ]
        ).astype(np.float32)

    def predict_raw(self, obs: Observation, exec_horizon=1):
        if exec_horizon <= 0:
            raise ValueError(f"exec_horizon must be positive, got {exec_horizon}")
        total_start = time.perf_counter()
        state = self._build_state(obs)
        if state.shape != (44,):
            raise RuntimeError(f"expected 44-D state, got {state.shape}")
        images = {
            "head": _resize_rgb(obs.rgb.get("head")),
            "left": _resize_rgb(obs.rgb.get("L_wrist")),
            "right": _resize_rgb(obs.rgb.get("R_wrist")),
        }
        for name, image in images.items():
            if image.shape != (240, 320, 3):
                raise RuntimeError(
                    f"expected {name} image shape (240, 320, 3), got {image.shape}"
                )
        request = {
            "state": state,
            **images,
            "task": os.environ.get("PI05_TASK", "assemble parts onto the task board"),
        }
        if exec_horizon > 1:
            request["exec_horizon"] = exec_horizon

        serialize_start = time.perf_counter()
        payload = pickle.dumps(request)
        serialization_ms = (time.perf_counter() - serialize_start) * 1000.0

        round_trip_start = time.perf_counter()
        self._send_payload(payload)
        reply = self._recv()
        round_trip_ms = (time.perf_counter() - round_trip_start) * 1000.0
        if not reply.get("ok", True):
            raise RuntimeError(f"pi0.5 inference failed: {reply.get('error', reply)!r}")

        actions = np.asarray(reply.get("actions", [reply["action"]]), np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"pi0.5 action chunk has shape {actions.shape}, expected (K, 14)")
        if not 1 <= actions.shape[0] <= exec_horizon:
            raise RuntimeError(
                f"pi0.5 action chunk has length {actions.shape[0]}, expected 1..{exec_horizon}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("pi0.5 action chunk contains non-finite values")
        action = actions[0]

        return {
            "state": state,
            "action": action,
            "actions": actions,
            "images": images,
            "request_bytes": len(payload),
            "timings_ms": {
                "pickle_serialization": serialization_ms,
                "ssh_server_round_trip": round_trip_ms,
                "total_policy_call": (time.perf_counter() - total_start) * 1000.0,
            },
        }

    def predict_cached_action(self, obs: Observation):
        self._control_step += 1
        replanned = False
        chunk_request_ms = None
        chunk_length = None
        if not self._action_cache:
            prediction = self.predict_raw(obs, exec_horizon=self._exec_horizon)
            self._action_cache.extend(action.copy() for action in prediction["actions"])
            self._replan_index += 1
            replanned = True
            chunk_request_ms = prediction["timings_ms"]["ssh_server_round_trip"]
            chunk_length = len(prediction["actions"])
            self._log_cache(
                f"replan_index={self._replan_index} control_step={self._control_step} "
                f"chunk_request_ms={chunk_request_ms:.3f} chunk_length={chunk_length} "
                f"cache_remaining={len(self._action_cache)}"
            )

        action = self._action_cache.popleft()
        self._log_cache(
            f"control_step={self._control_step} source=local_cache "
            f"cache_remaining={len(self._action_cache)}"
        )
        return {
            "action": action,
            "replanned": replanned,
            "replan_index": self._replan_index,
            "control_step": self._control_step,
            "chunk_request_ms": chunk_request_ms,
            "chunk_length": chunk_length,
            "cache_remaining": len(self._action_cache),
        }

    def predict_next_action_raw(self):
        total_start = time.perf_counter()
        serialize_start = time.perf_counter()
        payload = pickle.dumps({"cmd": "next_action"})
        serialization_ms = (time.perf_counter() - serialize_start) * 1000.0

        round_trip_start = time.perf_counter()
        self._send_payload(payload)
        reply = self._recv()
        round_trip_ms = (time.perf_counter() - round_trip_start) * 1000.0
        if not reply.get("ok"):
            raise RuntimeError(f"pi0.5 next_action failed: {reply.get('error', reply)!r}")

        action = np.asarray(reply["action"], np.float64).reshape(-1)
        if action.shape != (14,):
            raise RuntimeError(f"pi0.5 action has shape {action.shape}, expected (14,)")
        if not np.isfinite(action).all():
            raise RuntimeError(f"pi0.5 action contains non-finite values: {action}")
        return {
            "action": action,
            "request_bytes": len(payload),
            "timings_ms": {
                "pickle_serialization": serialization_ms,
                "ssh_server_round_trip": round_trip_ms,
                "total_policy_call": (time.perf_counter() - total_start) * 1000.0,
            },
        }

    def one_step_dry_run(self, obs, log_path, observation_timings):
        from scipy.spatial.transform import Rotation

        prediction = self.predict_raw(obs)
        state = prediction["state"]
        action = prediction["action"]
        images = prediction["images"]

        log_path = os.path.abspath(log_path)
        artifact_dir = os.path.dirname(log_path)
        os.makedirs(artifact_dir, exist_ok=True)

        import cv2

        image_paths = {}
        image_stats = {}
        for name, image in images.items():
            path = os.path.join(artifact_dir, f"pi05_dry_run_{name}.png")
            if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write dry-run image: {path}")
            image_paths[name] = path
            arr = image.astype(np.float64)
            image_stats[name] = {
                "shape": tuple(image.shape),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
                "black": bool(arr.max() <= 5.0),
                "blank": bool(arr.std() < 1.0),
            }

        pairwise_mad = {}
        for first, second in (("head", "left"), ("head", "right"), ("left", "right")):
            delta = np.abs(
                images[first].astype(np.float64) - images[second].astype(np.float64)
            )
            pairwise_mad[f"{first}_vs_{second}"] = float(delta.mean())

        def rotation_diagnostics(current_quat_wxyz, predicted_euler_xyz):
            current_quat_wxyz = np.asarray(current_quat_wxyz, dtype=np.float64)
            current = Rotation.from_quat(
                [
                    current_quat_wxyz[1],
                    current_quat_wxyz[2],
                    current_quat_wxyz[3],
                    current_quat_wxyz[0],
                ]
            )
            predicted = Rotation.from_euler("xyz", predicted_euler_xyz)
            difference = predicted * current.inv()
            delta_rotvec = difference.as_rotvec()
            return current.as_rotvec(), delta_rotvec, float(np.linalg.norm(delta_rotvec))

        left_current_rotvec, left_delta_rotvec, left_angle_rad = rotation_diagnostics(
            state[3:7], action[3:6]
        )
        right_current_rotvec, right_delta_rotvec, right_angle_rad = rotation_diagnostics(
            state[10:14], action[10:13]
        )
        left_translation_delta = action[0:3] - state[0:3]
        right_translation_delta = action[7:10] - state[7:10]
        q = np.asarray(obs.joint_positions, dtype=np.float64)

        def array_text(values):
            return np.array2string(
                np.asarray(values),
                precision=9,
                separator=", ",
                suppress_small=False,
                threshold=1000,
                max_line_width=240,
            )

        lines = [
            "pi0.5 one-step dry-run",
            "safety: prediction recorded only; no IK/controller action was created or applied",
            f"sim_step: {obs.step_idx}",
            f"request_bytes: {prediction['request_bytes']}",
            "",
            "images:",
        ]
        for name in ("head", "left", "right"):
            stats = image_stats[name]
            lines.append(f"  {name}: path={image_paths[name]}")
            lines.append(
                "    shape={shape} min={min:.3f} max={max:.3f} mean={mean:.3f} "
                "std={std:.3f} nonzero_fraction={nonzero_fraction:.6f} "
                "black={black} blank={blank}".format(**stats)
            )
        lines.append(f"  pairwise_mean_absolute_difference: {pairwise_mad}")
        lines.extend(
            [
                "",
                "current_left:",
                f"  xyz: {array_text(state[0:3])}",
                f"  quaternion_wxyz: {array_text(state[3:7])}",
                f"  rotation_rotvec: {array_text(left_current_rotvec)}",
                f"  gripper_joint: {q[self._Lg]:.9f}",
                f"  gripper_normalized: {state[42]:.9f}",
                "current_right:",
                f"  xyz: {array_text(state[7:10])}",
                f"  quaternion_wxyz: {array_text(state[10:14])}",
                f"  rotation_rotvec: {array_text(right_current_rotvec)}",
                f"  gripper_joint: {q[self._Rg] if self._Rg is not None else 0.0:.9f}",
                f"  gripper_normalized: {state[43]:.9f}",
                "",
                f"state_44d: {array_text(state)}",
                f"prediction_14d: {array_text(action)}",
                "",
                "left_prediction_delta:",
                f"  xyz: {array_text(left_translation_delta)}",
                f"  translation_norm_m: {np.linalg.norm(left_translation_delta):.9f}",
                f"  predicted_rotation_euler_xyz: {array_text(action[3:6])}",
                f"  rotation_delta_rotvec: {array_text(left_delta_rotvec)}",
                f"  rotation_delta_rad: {left_angle_rad:.9f}",
                f"  rotation_delta_deg: {np.degrees(left_angle_rad):.9f}",
                f"  gripper_prediction: {action[6]:.9f}",
                "right_prediction_delta:",
                f"  xyz: {array_text(right_translation_delta)}",
                f"  translation_norm_m: {np.linalg.norm(right_translation_delta):.9f}",
                f"  predicted_rotation_euler_xyz: {array_text(action[10:13])}",
                f"  rotation_delta_rotvec: {array_text(right_delta_rotvec)}",
                f"  rotation_delta_rad: {right_angle_rad:.9f}",
                f"  rotation_delta_deg: {np.degrees(right_angle_rad):.9f}",
                f"  gripper_prediction: {action[13]:.9f}",
                "",
                "timings_ms:",
                f"  camera_capture: {observation_timings['camera_capture']:.3f}",
                f"  observation_construction: {observation_timings['observation_construction']:.3f}",
                f"  pickle_serialization: {prediction['timings_ms']['pickle_serialization']:.3f}",
                f"  ssh_server_round_trip: {prediction['timings_ms']['ssh_server_round_trip']:.3f}",
                f"  total_policy_call: {prediction['timings_ms']['total_policy_call']:.3f}",
            ]
        )
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(lines) + "\n")
        print(f"[dry-run] wrote diagnostics -> {log_path}", flush=True)
        return prediction

    def ik_dry_run(self, obs, log_path, observation_timings):
        """Solve one filtered left-arm IK target without applying its action."""
        prediction = self.predict_raw(obs)
        raw_action = prediction["action"].copy()
        current_position, current_quat = self.L.end_effector.get_world_pose()
        current_position = np.asarray(current_position, dtype=np.float64)
        current_quat = np.asarray(current_quat, dtype=np.float64)
        current_joints = np.asarray(obs.joint_positions, dtype=np.float64)

        filtered_action = raw_action.copy()
        ee_safety = None
        if self._safety_filter:
            filtered_action, ee_safety = _filter_left_action(
                filtered_action,
                current_position,
                current_quat,
                float(obs.L_gripper_position) / GRIPPER_OPEN_LIMIT,
            )

        target_quat = _euler_xyz_to_quat_wxyz(*filtered_action[3:6])
        target_gripper = (
            float(np.clip(filtered_action[6], 0.0, 1.0)) * GRIPPER_OPEN_LIMIT
        )
        ik_action = self.L.forward(
            filtered_action[:3],
            target_quat,
            target_gripper,
        )
        raw_joint_targets = list(
            ik_action.joint_positions
            or [None] * len(current_joints)
        )
        guarded_joint_targets, joint_safety = _guard_left_joint_action(
            raw_joint_targets,
            current_joints,
            self._Li,
            self._Lg,
        )

        ik_left_targets = np.array(
            [
                np.nan if raw_joint_targets[index] is None else raw_joint_targets[index]
                for index in self._Li
            ],
            dtype=np.float64,
        )
        current_left_joints = current_joints[self._Li]
        joint_delta = ik_left_targets - current_left_joints
        guarded_left_targets = np.asarray(
            [guarded_joint_targets[index] for index in self._Li],
            dtype=np.float64,
        )
        ik_backend = getattr(self.L, "ik", None)
        ik_success = getattr(ik_backend, "ik_ok", None)

        def array_text(values):
            return np.array2string(
                np.asarray(values),
                precision=9,
                separator=", ",
                suppress_small=False,
                max_line_width=240,
            )

        lines = [
            "pi0.5 one-step IK dry-run",
            "safety: IK was solved, but no articulation action was applied",
            f"sim_step: {obs.step_idx}",
            f"ik_success: {ik_success}",
            f"raw_prediction_14d: {array_text(raw_action)}",
            f"filtered_prediction_14d: {array_text(filtered_action)}",
            f"current_left_xyz: {array_text(current_position)}",
            f"current_left_quaternion_wxyz: {array_text(current_quat)}",
            f"target_left_xyz: {array_text(filtered_action[:3])}",
            f"target_left_quaternion_wxyz: {array_text(target_quat)}",
            f"current_left_joints_rad: {array_text(current_left_joints)}",
            f"ik_left_targets_rad: {array_text(ik_left_targets)}",
            f"ik_left_delta_rad: {array_text(joint_delta)}",
            f"ik_left_delta_deg: {array_text(np.degrees(joint_delta))}",
            f"ik_left_max_abs_delta_deg: {np.degrees(np.nanmax(np.abs(joint_delta))):.9f}",
            f"joint_guard_held: {joint_safety['held']}",
            f"joint_guard_reason: {joint_safety['reason']}",
            f"joint_guard_max_delta_deg: {joint_safety['max_delta_deg']:.9f}",
            f"guarded_left_targets_rad: {array_text(guarded_left_targets)}",
        ]
        if ee_safety is not None:
            lines.extend(
                [
                    f"ee_filter_held: {ee_safety['held']}",
                    f"ee_filter_reason: {ee_safety['hold_reason']}",
                    f"ee_raw_translation_m: {ee_safety['raw_translation_m']:.9f}",
                    f"ee_raw_rotation_deg: {ee_safety['raw_rotation_deg']:.9f}",
                    f"ee_safe_translation_m: {ee_safety['safe_translation_m']:.9f}",
                    f"ee_safe_rotation_deg: {ee_safety['safe_rotation_deg']:.9f}",
                ]
            )
        lines.extend(
            [
                f"camera_capture_ms: {observation_timings['camera_capture']:.3f}",
                f"observation_construction_ms: {observation_timings['observation_construction']:.3f}",
                f"ssh_server_round_trip_ms: {prediction['timings_ms']['ssh_server_round_trip']:.3f}",
                "controller_forward_called: 1 (IK solve only)",
                "articulation_apply_action_called: 0",
            ]
        )
        log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(lines) + "\n")
        print(f"[dry-run] wrote IK diagnostics -> {log_path}", flush=True)
        return {
            "prediction": prediction,
            "filtered_action": filtered_action,
            "joint_safety": joint_safety,
            "ik_success": ik_success,
        }

    def five_request_diagnostic(
        self, obs, log_path, observation_timings, request_count=5
    ):
        log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        records = []
        previous_action = None
        for request_index in range(1, request_count + 1):
            request_kind = "full_observation" if request_index == 1 else "next_action"
            prediction = (
                self.predict_raw(obs)
                if request_index == 1
                else self.predict_next_action_raw()
            )
            action = prediction["action"].copy()
            difference = (
                None if previous_action is None else action - previous_action
            )
            records.append(
                {
                    "request_index": request_index,
                    "request_kind": request_kind,
                    "request_bytes": prediction["request_bytes"],
                    "round_trip_ms": prediction["timings_ms"][
                        "ssh_server_round_trip"
                    ],
                    "total_policy_call_ms": prediction["timings_ms"][
                        "total_policy_call"
                    ],
                    "action": action,
                    "difference": difference,
                }
            )
            previous_action = action

        def array_text(values):
            return np.array2string(
                np.asarray(values),
                precision=9,
                separator=", ",
                suppress_small=False,
                threshold=1000,
                max_line_width=240,
            )

        lines = [
            "pi0.5 full-observation plus next_action remote diagnostic",
            "safety: 5 predictions recorded only; no reset between requests and no robot action applied",
            f"sim_step: {obs.step_idx}",
            f"request_count: {request_count}",
            f"camera_capture_ms: {observation_timings['camera_capture']:.3f}",
            f"observation_construction_ms: {observation_timings['observation_construction']:.3f}",
            "queue_diagnostics: see PI05_SERVER_LOG stderr entries",
            "",
        ]
        for record in records:
            lines.extend(
                [
                    f"request_index: {record['request_index']}",
                    f"request_kind: {record['request_kind']}",
                    f"request_bytes: {record['request_bytes']}",
                    f"ssh_server_round_trip_ms: {record['round_trip_ms']:.3f}",
                    f"total_policy_call_ms: {record['total_policy_call_ms']:.3f}",
                    f"action_14d: {array_text(record['action'])}",
                ]
            )
            if record["difference"] is None:
                lines.append("difference_from_previous: N/A")
            else:
                difference = record["difference"]
                lines.extend(
                    [
                        f"difference_from_previous: {array_text(difference)}",
                        f"difference_l2_norm: {np.linalg.norm(difference):.9f}",
                        f"difference_max_abs: {np.max(np.abs(difference)):.9f}",
                    ]
                )
            lines.append("")

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(lines))
        print(f"[diagnostic] wrote 5-request timing -> {log_path}", flush=True)
        return records

    def act(self, obs: Observation):
        action = self.predict_cached_action(obs)["action"]
        if self._safety_filter:
            current_position, current_quat = self.L.end_effector.get_world_pose()
            action, safety = _filter_left_action(
                action,
                current_position,
                current_quat,
                float(obs.L_gripper_position) / GRIPPER_OPEN_LIMIT,
            )
            self._log_cache(
                f"safety control_step={self._control_step} held={int(safety['held'])} "
                f"reason={safety['hold_reason'] or 'none'} "
                f"raw_translation_m={safety['raw_translation_m']:.9f} "
                f"raw_rotation_deg={safety['raw_rotation_deg']:.6f} "
                f"safe_translation_m={safety['safe_translation_m']:.9f} "
                f"safe_rotation_deg={safety['safe_rotation_deg']:.6f}"
            )
        pos = action[:3]
        quat = _euler_xyz_to_quat_wxyz(action[3], action[4], action[5])
        grip = float(np.clip(action[6], 0, 1)) * GRIPPER_OPEN_LIMIT
        control_action = self.L.forward(pos, quat, grip)
        if self._safety_filter:
            guarded_positions, joint_safety = _guard_left_joint_action(
                control_action.joint_positions,
                obs.joint_positions,
                self._Li,
                self._Lg,
            )
            control_action.joint_positions = guarded_positions
            self._log_cache(
                f"joint_safety control_step={self._control_step} "
                f"held={int(joint_safety['held'])} "
                f"reason={joint_safety['reason'] or 'none'} "
                f"max_delta_deg={joint_safety['max_delta_deg']:.6f} "
                f"invalid_indices={joint_safety['invalid_indices']}"
            )
        return control_action

    def is_done(self, obs: Observation) -> bool:
        return False

    def close(self):
        self._action_cache.clear()
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            finally:
                try:
                    if proc.stdout is not None:
                        proc.stdout.close()
                except OSError:
                    pass
            self._proc = None
        try:
            if self._err is not None:
                self._err.close()
        except Exception:
            pass
        self._err = None

    def __del__(self):
        self.close()
