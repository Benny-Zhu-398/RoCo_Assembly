"""ACT (lerobot) inference policy for the single-part `gear_20teeth` model.

Sibling of `policies/act_eval.py`, kept as a separate file (not a copy-over)
because the underlying checkpoint has a different, simpler feature layout:
this one is trained left-arm-only on `dataset/lerobot_v4_gear_20teeth/`
(merged via `training/build_gear_act_dataset.py` + `training/train_gear_act.json`),
instead of the two-arm, 9-part model `act_eval.py` targets.

Run with:

    uv run python task/run_pick_place.py --policy policies.act_eval_gear.ACTGearEvalPolicy

Requirements before this will run:
  1. `lerobot==0.4.4` + `torch` installed in the environment.
  2. `param_config.enable_camera_output = True` -- this policy needs both
     the head camera and the L wrist camera every step.
  3. A checkpoint directory under `outputs/train/act_gear_20teeth/checkpoints/*/pretrained_model/`
     that actually contains `model.safetensors` (override the default via
     `ROCO_ACT_GEAR_CHECKPOINT`).

State / action layout, matching `training/build_gear_act_dataset.py`'s
merged dataset schema (which in turn mirrors `policy_api.Observation`
exactly, so there is no right-arm placeholder to fake here):

    observation.state (22,) = [
        left_ee_xyz(3), left_ee_quat_wxyz(4),
        left_jpos(7), left_jvel(7), left_gripper(1),
    ]
    action (7,) = left_ee_xyz(3), left_ee_rotvec(3), left_gripper(1)

    observation.images.head      <- obs.rgb["head"]
    observation.images.left_hand <- obs.rgb["L_wrist"]

Unlike `act_eval.py`, there is no `R_EE_HOME_POSE` placeholder: the right
arm is never part of this model's input or output, since the harness
holds it fixed and it isn't observable through `policy_api.Observation`
anyway. That mismatch in `act_eval.py` (identity-placeholder right-arm
pose biasing every prediction) is exactly the class of bug training a
left-arm-only, single-part model avoids.
"""
from __future__ import annotations

import os.path
import sys

# Ensure `task/` is on sys.path so `policy_api` imports work.
_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ACTGearEvalPolicy requires `torch` + `lerobot==0.4.4` in this "
        "environment (see task/collect_lerobot_v3.py's docstring)."
    ) from e


def _import_lerobot_act():
    """Import the ACT model class + config loader.

    Split out so a version mismatch fails with one clear message instead
    of a bare ImportError deep in __init__.
    """
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy as ACTModel
    except ImportError as e:
        raise ImportError(
            "Could not import lerobot's ACTPolicy / PreTrainedConfig from "
            "the expected 0.4.x locations (lerobot.policies.act.modeling_act, "
            "lerobot.configs.policies). Check your installed lerobot version "
            "and adjust these imports in policies/act_eval_gear.py."
        ) from e
    return ACTModel, PreTrainedConfig


def _import_make_pre_post_processors():
    """Locate the factory that rebuilds the normalizer pre/post-processor
    pipelines described by policy_preprocessor.json / policy_postprocessor.json.
    Tried in two plausible locations since this is a newer (~lerobot 0.4)
    API surface that moved around across releases.
    """
    try:
        from lerobot.processor.factory import make_pre_post_processors
        return make_pre_post_processors
    except ImportError:
        pass
    try:
        from lerobot.policies.factory import make_pre_post_processors
        return make_pre_post_processors
    except ImportError as e:
        raise ImportError(
            "Could not find `make_pre_post_processors` in "
            "lerobot.processor.factory or lerobot.policies.factory. This "
            "checkpoint's policy_preprocessor.json / policy_postprocessor.json "
            "use the processor-pipeline API introduced around lerobot 0.4 -- "
            "check your installed version's module layout and adjust the "
            "import in policies/act_eval_gear.py."
        ) from e


# Checkpoint to load. Override with the ROCO_ACT_GEAR_CHECKPOINT env var
# without editing this file (e.g. to point at a different training run / step).
#
# Defaults to the final numbered step directory (matching train_gear_act.json's
# "steps": 20000), not "checkpoints/last" -- in the previous act_stage1 run,
# checkpoints/last/pretrained_model saved its weights as model_10.safetensors
# instead of model.safetensors, which ACTModel.from_pretrained can't find.
# The numbered step dirs (002000, 004000, ...) did not have this problem.
# If your run also mis-names checkpoints/last, either point this at a
# numbered step dir or rename/symlink the weights file to model.safetensors.
_REPO_ROOT = os.path.dirname(_TASK_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "train", "act_gear_20teeth",
    "checkpoints", "020000", "pretrained_model",
)
CHECKPOINT_DIR = os.environ.get("ROCO_ACT_GEAR_CHECKPOINT", DEFAULT_CHECKPOINT_DIR)

# (height, width) -- matches train_gear_act.json's observation.images.*
# shape (3, 240, 320). The cameras themselves render 480x640; we resize down.
IMAGE_HW = (240, 320)


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Inverse of collect_lerobot_v3.py's `_quat_wxyz_to_rotvec`."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


class ACTGearEvalPolicy(Policy):
    """Runs a trained left-arm-only lerobot ACT checkpoint for `gear_20teeth`.

    Needs `env_info.L_controller` (an `EEPoseController`) to turn the
    model's predicted absolute Cartesian target into joint positions via
    Lula IK -- the harness sets this for every policy by default, not
    just BaselinePolicy (see run_pick_place.py's EnvInfo construction).
    """

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)

        L_controller = getattr(env_info, "L_controller", None)
        if L_controller is None:
            raise ValueError(
                "ACTGearEvalPolicy requires env_info.L_controller (an "
                "EEPoseController) to convert predicted EE poses into "
                "joint targets. The harness sets this automatically; if "
                "you see this error you may be running outside the "
                "provided harness."
            )
        self._L_controller = L_controller
        self._L_arm_idx = np.array(
            [env_info.dof_names.index(j) for j in env_info.L_arm_joints],
            dtype=np.int64,
        )

        weights_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found. Point ROCO_ACT_GEAR_CHECKPOINT at "
                "a checkpoint dir (outputs/train/act_gear_20teeth/checkpoints/"
                "<step>/pretrained_model) that actually has model.safetensors."
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

        self._current_target: PartTarget = None  # type: ignore[assignment]

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._current_target = target
        # Clears the model's internal action-chunk queue so the first
        # act() after a part switch runs a fresh forward pass instead of
        # draining stale queued actions computed for the previous part.
        self._model.reset()

    def act(self, obs: Observation) -> ArticulationAction:
        batch = {
            "observation.state": torch.from_numpy(
                self._build_state(obs)
            ).float().unsqueeze(0).to(self._device),
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

        return self._L_controller.forward(pos, quat, gripper)

    def is_done(self, obs: Observation) -> bool:
        # ACT has no explicit stop signal in its output -- let the
        # harness's snap-fired / per-part timeout advance us, same as
        # policies/template.py's stub.
        return False

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
            raise RuntimeError(
                f"ACTGearEvalPolicy needs the '{cam_name}' camera frame but "
                f"obs.rgb[{cam_name!r}] is None. Set "
                "param_config.enable_camera_output = True before running "
                "this policy."
            )
        arr = np.asarray(frame)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr.astype(np.uint8))
        # PIL resize takes (width, height); IMAGE_HW is (height, width).
        img = img.resize((IMAGE_HW[1], IMAGE_HW[0]), Image.BILINEAR)
        chw = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))
        return chw
