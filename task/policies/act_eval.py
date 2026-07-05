"""ACT (lerobot) inference policy for the IROS 2026 vega_1u assembly challenge.

Wraps a trained lerobot ACT checkpoint (resnet18 vision backbone, 4-layer
transformer encoder, chunk_size=20, see the checkpoint's config.json) as a
`policy_api.Policy` so it can be run through the standard harness:

    uv run python task/run_pick_place.py --policy policies.act_eval.ACTEvalPolicy

Requirements before this will run:
  1. `lerobot==0.4.4` + `torch` installed in the environment (see
     collect_lerobot_v3.py's docstring for the pinned version).
  2. `param_config.enable_camera_output = True`. The baseline scripted
     policy doesn't need cameras so this defaults to False; ACT needs the
     head camera frame every step.
  3. A checkpoint directory that actually contains `model.safetensors`.
     **As of this writing, none of the checkpoints under
     outputs/train/act_stage1/checkpoints/*/pretrained_model/ have one** —
     each only has config.json / train_config.json and the pre/post-
     processor normalizer files (policy_preprocessor*.safetensors,
     policy_postprocessor*.safetensors). That means the actual network
     weights were never saved (or were saved somewhere else). Loading
     will fail with a clear FileNotFoundError until you point
     ROCO_ACT_CHECKPOINT at a checkpoint dir that has the weights file,
     or re-run/resume training far enough that one gets written.

State / action layout, confirmed against the training dataset's
meta/info.json (rocochallenge2025/rocochallenge2026_Industrial_Assembly on
the HF Hub) rather than guessed from this repo's collect_lerobot_v3.py:

    observation.state (44,) = [
        left_ee_xyz(3), left_ee_quat_wxyz(4),
        right_ee_xyz(3), right_ee_quat_wxyz(4),
        left_jpos(7), right_jpos(7),
        left_jvel(7), right_jvel(7),
        left_gripper(1), right_gripper(1),
    ]
    action (14,) = [
        left_ee_xyz(3), left_ee_rotvec(3), left_gripper(1),
        right_ee_xyz(3), right_ee_rotvec(3), right_gripper(1),
    ]

Only the left half of the predicted action is ever applied. The harness
always drives R from its own held init pose and ignores R dofs in
whatever ArticulationAction a policy returns (see
`policy_api.Policy.act` / `run_pick_place.merge_bimanual_actions`), so
there's no point converting the right half at all.

Known gap -- right arm EE pose isn't observable through the Policy API:
`Observation` only carries `ee_pose_L` (see policy_api.py), but the
trained state vector wants right_ee_xyz/quat too. Since R is held fixed
at `param_config.INIT_JOINT_TARGETS` for the entire run (run_pick_place.py
never moves it), that value is a constant -- not something we need FK
for at runtime. Measure it once on the Isaac Sim box and fill in
`R_EE_HOME_POSE` below: temporarily add
`print(R_controller.end_effector.get_world_pose())` next to the
`_build_observation` call in run_pick_place.py (same call
collect_lerobot_v3.py's `_actual_ee_pose(r_controller)` makes), run once,
copy the printed (pos, quat) here. Left as an identity placeholder for
now, which will bias every prediction until corrected.
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
        "ACTEvalPolicy requires `torch` + `lerobot==0.4.4` in this "
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
            "and adjust these imports in policies/act_eval.py."
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
            "import in policies/act_eval.py."
        ) from e


# Checkpoint to load. Override with the ROCO_ACT_CHECKPOINT env var without
# editing this file (e.g. to point at a different training run / step).
_REPO_ROOT = os.path.dirname(_TASK_DIR)
DEFAULT_CHECKPOINT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "train", "act_stage1",
    "checkpoints", "010000", "pretrained_model",
)
CHECKPOINT_DIR = os.environ.get("ROCO_ACT_CHECKPOINT", DEFAULT_CHECKPOINT_DIR)

# (height, width) -- matches config.json's observation.images.head shape
# (3, 240, 320). The head camera itself renders larger; we resize down.
IMAGE_HW = (240, 320)

# TODO: replace with the real measured (pos_xyz, quat_wxyz) -- see the
# module docstring's "Known gap" section. Identity placeholder for now.
R_EE_HOME_POSE = (
    np.array([0.0, 0.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
)


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Inverse of collect_lerobot_v3.py's `_quat_wxyz_to_rotvec`."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = angle / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


class ACTEvalPolicy(Policy):
    """Runs a trained lerobot ACT checkpoint as the L-arm policy.

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
                "ACTEvalPolicy requires env_info.L_controller (an "
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
        self._R_arm_idx = np.array(
            [env_info.dof_names.index(j) for j in env_info.R_arm_joints],
            dtype=np.int64,
        )
        self._R_gripper_idx = env_info.dof_names.index("R_gripper_joint")

        weights_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found. This checkpoint directory is "
                "missing its model weights (only config.json / "
                "train_config.json and the pre/post-processor normalizer "
                "files are present). Point ROCO_ACT_CHECKPOINT at a "
                "checkpoint dir that actually has model.safetensors."
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
                self._build_image(obs)
            ).float().unsqueeze(0).to(self._device),
        }
        batch = self._preprocessor(batch)
        with torch.no_grad():
            action = self._model.select_action(batch)
        action = self._postprocessor(action)
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float64)

        left = action[:7]
        left_pos = left[0:3]
        left_quat = _rotvec_to_quat_wxyz(left[3:6])
        left_gripper = float(left[6])

        return self._L_controller.forward(left_pos, left_quat, left_gripper)

    def is_done(self, obs: Observation) -> bool:
        # ACT has no explicit stop signal in its output -- let the
        # harness's snap-fired / per-part timeout advance us, same as
        # policies/template.py's stub.
        return False

    def _build_state(self, obs: Observation) -> np.ndarray:
        L_ee_pos, L_ee_quat = obs.ee_pose_L
        R_ee_pos, R_ee_quat = R_EE_HOME_POSE
        L_jpos = obs.joint_positions[self._L_arm_idx]
        R_jpos = obs.joint_positions[self._R_arm_idx]
        L_jvel = obs.joint_velocities[self._L_arm_idx]
        R_jvel = obs.joint_velocities[self._R_arm_idx]
        L_grip = float(obs.L_gripper_position)
        R_grip = float(obs.joint_positions[self._R_gripper_idx])
        return np.concatenate([
            np.asarray(L_ee_pos, dtype=np.float32),
            np.asarray(L_ee_quat, dtype=np.float32),
            np.asarray(R_ee_pos, dtype=np.float32),
            np.asarray(R_ee_quat, dtype=np.float32),
            np.asarray(L_jpos, dtype=np.float32),
            np.asarray(R_jpos, dtype=np.float32),
            np.asarray(L_jvel, dtype=np.float32),
            np.asarray(R_jvel, dtype=np.float32),
            np.array([L_grip, R_grip], dtype=np.float32),
        ])

    def _build_image(self, obs: Observation) -> np.ndarray:
        frame = obs.rgb.get("head") if obs.rgb else None
        if frame is None:
            raise RuntimeError(
                "ACTEvalPolicy needs the head camera frame but "
                "obs.rgb['head'] is None. Set "
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
