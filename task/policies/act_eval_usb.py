"""ACT inference policy for the single-part `usb_a` model.

Run with:
    uv run python task/run_pick_place.py --policy policies.act_eval_usb.ACTUsbEvalPolicy

Requirements:
  1. lerobot==0.4.4 + torch installed.
  2. param_config.enable_camera_output = True
  3. Checkpoint under outputs/train/act_usb_a/checkpoints/*/pretrained_model/
     (override with ROCO_ACT_USB_CHECKPOINT env var)

State / action layout (matches training/build_gear_act_dataset.py):
    observation.state (22,) = [
        left_ee_xyz(3), left_ee_quat_wxyz(4),
        left_jpos(7), left_jvel(7), left_gripper(1),
    ]
    action (7,) = left_ee_xyz(3), left_ee_rotvec(3), left_gripper(1)

    observation.images.head      <- obs.rgb["head"]
    observation.images.left_hand <- obs.rgb["L_wrist"]
"""
from __future__ import annotations

import os.path
import sys

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np
from PIL import Image

from omni.isaac.core.utils.types import ArticulationAction
from policy_api import EnvInfo, Observation, PartTarget, Policy

try:
    import torch
except ImportError as e:
    raise ImportError("ACTUsbEvalPolicy requires torch + lerobot==0.4.4") from e


def _import_lerobot_act():
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy as ACTModel
    except ImportError as e:
        raise ImportError("Could not import lerobot ACTPolicy") from e
    return ACTModel, PreTrainedConfig


def _import_make_pre_post_processors():
    try:
        from lerobot.processor.factory import make_pre_post_processors
        return make_pre_post_processors
    except ImportError:
        pass
    try:
        from lerobot.policies.factory import make_pre_post_processors
        return make_pre_post_processors
    except ImportError as e:
        raise ImportError("Could not find make_pre_post_processors") from e


_REPO_ROOT = os.path.dirname(_TASK_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "train", "act_usb_a",
    "checkpoints", "020000", "pretrained_model",
)
CHECKPOINT_DIR = os.environ.get("ROCO_ACT_USB_CHECKPOINT", DEFAULT_CHECKPOINT_DIR)

IMAGE_HW = (240, 320)


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


class ACTUsbEvalPolicy(Policy):

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)

        L_controller = getattr(env_info, "L_controller", None)
        if L_controller is None:
            raise ValueError("ACTUsbEvalPolicy requires env_info.L_controller")
        self._L_controller = L_controller
        self._L_arm_idx = np.array(
            [env_info.dof_names.index(j) for j in env_info.L_arm_joints],
            dtype=np.int64,
        )
        self._act_step = 0

        weights_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found. Set ROCO_ACT_USB_CHECKPOINT to the "
                "pretrained_model directory."
            )

        ACTModel, PreTrainedConfig = _import_lerobot_act()
        make_pre_post_processors = _import_make_pre_post_processors()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_cfg = PreTrainedConfig.from_pretrained(CHECKPOINT_DIR)
        policy_cfg.device = str(self._device)

        self._model = ACTModel.from_pretrained(CHECKPOINT_DIR)
        self._model.to(self._device)
        self._model.eval()

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg, pretrained_path=CHECKPOINT_DIR,
        )
        self._current_target: PartTarget = None

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._current_target = target
        self._model.reset()
        self._act_step = 0

    def act(self, obs: Observation) -> ArticulationAction:
        self._act_step += 1

        state = self._build_state(obs)

        if self._act_step % 200 == 1:
            print(f"[usb_eval] step={self._act_step} ee_pos={state[:3].tolist()}", flush=True)

        batch = {
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(self._device),
            "observation.images.head": torch.from_numpy(
                self._build_image(obs, "head")
            ).float().unsqueeze(0).to(self._device),
            "observation.images.left_hand": torch.from_numpy(
                self._build_image(obs, "L_wrist")
            ).float().unsqueeze(0).to(self._device),
        }
        batch = self._preprocessor(batch)
        with torch.no_grad():
            action = self._model.select_action(batch)
        action = self._postprocessor(action)
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float64)

        pos = action[0:3]
        quat = _rotvec_to_quat_wxyz(action[3:6])
        gripper = float(action[6])

        result = self._L_controller.forward(pos, quat, gripper)

        if self._act_step % 200 == 1:
            print(f"[usb_eval] pred_pos={pos.tolist()} gripper={gripper:.3f}", flush=True)

        return result

    def is_done(self, obs: Observation) -> bool:
        return bool(obs.snap_fired)

    def _build_state(self, obs: Observation) -> np.ndarray:
        L_ee_pos, L_ee_quat = obs.ee_pose_L
        L_jpos = obs.joint_positions[self._L_arm_idx]
        L_jvel = obs.joint_velocities[self._L_arm_idx]
        L_grip = float(obs.L_gripper_position)
        return np.concatenate([
            np.asarray(L_ee_pos, dtype=np.float32),
            np.asarray(L_ee_quat, dtype=np.float32),
            np.asarray(L_jpos, dtype=np.float32),
            np.asarray(L_jvel, dtype=np.float32),
            np.array([L_grip], dtype=np.float32),
        ])

    def _build_image(self, obs: Observation, cam_name: str) -> np.ndarray:
        frame = obs.rgb.get(cam_name) if obs.rgb else None
        if frame is None:
            return np.zeros((3, IMAGE_HW[0], IMAGE_HW[1]), dtype=np.float32)
        arr = np.asarray(frame)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr.astype(np.uint8))
        img = img.resize((IMAGE_HW[1], IMAGE_HW[0]), Image.BILINEAR)
        chw = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))
        return chw
