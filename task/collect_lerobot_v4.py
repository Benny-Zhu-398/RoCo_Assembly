import argparse
from collections import deque
import json
import shutil
import traceback
from pathlib import Path
from textwrap import dedent

import numpy as np
from PIL import Image

import run_pick_place as rp
from isaacsim.core.utils.types import ArticulationAction
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policy_api import EnvInfo, Observation, PartTarget


TASK_DESCRIPTION = (
    "TaskBoardAssembly baseline scripted rollout collected from Isaac Sim. "
    "actions.* are absolute Cartesian targets encoded as xyz + rotation-vector + gripper."
)
DEPTH_SCALE_MM = 1000.0
DEPTH_MAX_MM = np.iinfo(np.uint16).max


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect a single-part TaskBoardAssembly scripted rollout into a local LeRobot v4-style dataset "
            "using a temporary part-order override, physics-callback sampling, labels, and optional EE disturbances."
        ),
        epilog=dedent(
            """
            Runtime requirements:
              - Run with the Isaac Sim Python environment used by this repository.
              - Install lerobot 0.4.4 in that environment.
              - Keep numpy on a 1.x build compatible with Isaac Sim camera annotators.
              - Pillow is required for 16-bit depth PNG encoding.

            Example:
              OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_HEADLESS=1 \
              ./.conda-isaacsim/bin/python task/collect_lerobot_v4.py \
                --overwrite \
                --output-root artifacts/lerobot_taskboard_episode \
                --sample-hz 30 \
                --include-depth
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/lerobot_taskboard_v4_episode"),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="taskboard/vega_1u_lerobot_v4_episode",
    )
    parser.add_argument(
        "--part-name",
        type=str,
        required=True,
        help=(
            "Single part to collect. v4 temporarily overrides rp.pc.part_order "
            "to [part_name], so the selected part is treated as the first and "
            "does not get a return_home waypoint."
        ),
    )
    parser.add_argument(
        "--label-json",
        type=Path,
        default=None,
        help=(
            "Path for the single-part label JSON. Default is a sibling file "
            "named <output_root.name>_label.json."
        ),
    )
    parser.add_argument(
        "--policy",
        default="policies.baseline_scripted.BaselinePolicy",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=30.0,
        help=(
            "Target recording rate in Hz. Default is 30 Hz. "
            "Ignored when --sample-stride is set."
        ),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help=(
            "Record one frame every N physics steps. Overrides --sample-hz. "
            "At 200 Hz physics, 200 means 1 Hz."
        ),
    )
    parser.add_argument(
        "--max-recorded-frames",
        type=int,
        default=0,
        help="Stop after recording this many frames. 0 disables the cap.",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=0,
        help="Stop after this many parts have become active. 0 means the full task order.",
    )
    parser.add_argument(
        "--max-sim-steps",
        type=int,
        default=0,
        help=(
            "Hard safety cap on physics steps. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("/tmp/taskboard_lerobot_collect_results.json"),
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--include-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store depth as 16-bit PNG images in millimeters. Enabled by default; use "
            "--no-include-depth to disable."
        ),
    )
    parser.add_argument(
        "--disturb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, occasionally inject a short EE-space disturbance into "
            "the executed left-arm command while keeping the stored expert action "
            "label unchanged."
        ),
    )
    parser.add_argument(
        "--disturbance-prob",
        type=float,
        default=0.004,
        help="Per-control-step probability of starting a disturbance when none is active.",
    )
    parser.add_argument(
        "--disturbance-duration-steps",
        type=int,
        default=20,
        help="Number of physics/control steps for each EE disturbance.",
    )
    parser.add_argument(
        "--disturbance-pos-max",
        type=float,
        default=0.005,
        help="Maximum translational EE disturbance magnitude in meters.",
    )
    parser.add_argument(
        "--disturbance-rot-max-deg",
        type=float,
        default=2.0,
        help="Maximum rotational EE disturbance magnitude in degrees.",
    )
    parser.add_argument(
        "--disturbance-seed",
        type=int,
        default=None,
        help="Random seed for disturbance generation.",
    )
    parser.add_argument(
        "--skip-non-ee-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, skip cspace_target and joint_lerp_t frames so "
            "actions.left_arm_action remains a pure EE target label."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def _quat_wxyz_to_rotvec(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return (axis * angle).astype(np.float32)


def _pack_pose(pos, quat):
    pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
    quat_arr = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.concatenate([pos_arr, quat_arr], axis=0)


def _pack_cartesian_action(pos, quat, gripper):
    pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
    rotvec = _quat_wxyz_to_rotvec(quat)
    grip = np.array([float(gripper)], dtype=np.float32)
    return np.concatenate([pos_arr, rotvec, grip], axis=0)


def _as_rgb(frame):
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _as_depth(frame):
    if frame is None:
        return np.zeros((480, 640, 1), dtype=np.float32)
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr.astype(np.float32, copy=False)


def _encode_depth_image(frame):
    depth = _as_depth(frame)[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(np.rint(depth * DEPTH_SCALE_MM), 0.0, float(DEPTH_MAX_MM)).astype(np.uint16)
    return Image.fromarray(depth_mm)


def _feature_spec(include_depth):
    features = {
        "actions.left_arm_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "actions.right_arm_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "observations.left_arm_ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "observations.right_arm_ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "observations.left_arm_joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"left_joint_{i}" for i in range(7)],
        },
        "observations.right_arm_joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"right_joint_{i}" for i in range(7)],
        },
        "observations.left_arm_joint_velocity": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"left_joint_{i}" for i in range(7)],
        },
        "observations.right_arm_joint_velocity": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"right_joint_{i}" for i in range(7)],
        },
        "observations.left_gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["left_gripper"],
        },
        "observations.right_gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["right_gripper"],
        },
        "observations.rgb_head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "observations.rgb_left_hand": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "observations.rgb_right_hand": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
    }
    if include_depth:
        for key in (
            "observations.depth_head",
            "observations.depth_left_hand",
            "observations.depth_right_hand",
        ):
            features[key] = {
                "dtype": "image",
                "shape": (480, 640, 1),
                "names": ["height", "width", "channel"],
            }
    return features


def _resolve_gripper_cmd(command, cfg, current_value):
    if command is None:
        return float(current_value)
    if isinstance(command, str):
        cmd = command.lower()
        if cmd == "open":
            return float(cfg.get("gripper_open", current_value))
        if cmd == "close":
            return float(cfg.get("gripper_close", current_value))
    try:
        return float(command)
    except Exception:
        return float(current_value)


def _rotvec_to_quat_wxyz(rotvec):
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rv / angle
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], axis * np.sin(half)]).astype(np.float64)


def _quat_multiply_wxyz(q1, q2):
    a = np.asarray(q1, dtype=np.float64).reshape(4)
    b = np.asarray(q2, dtype=np.float64).reshape(4)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    out = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
    n = np.linalg.norm(out)
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out / n


def _random_vector_with_max_norm(rng, max_norm):
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return np.zeros(3, dtype=np.float64)
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return direction * float(rng.uniform(0.0, max_norm))


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _build_label_path(args, output_root):
    if args.label_json is not None:
        return args.label_json
    return output_root.parent / f"{output_root.name}_label.json"


def _left_action_target(policy, obs, current_cfg, skip_non_ee_actions=True):
    wp = getattr(policy, "current_waypoint", None)
    if callable(wp):
        wp = wp()
    ee_pos, ee_orn = obs.ee_pose_L
    if wp is None:
        return ee_pos, ee_orn, float(obs.L_gripper_position)

    # cspace_target (return_home) has no EE target — always skip.
    if getattr(wp, "cspace_target", None) is not None:
        return None

    # lock_pose is kept as an EE hold label. For the current v4 collector,
    # that means target=current EE pose + gripper command. This matches the
    # intended practical behavior: hold the current end-effector pose while
    # the gripper closes/opens or while snap_wait is active.
    if getattr(wp, "lock_pose", False):
        return ee_pos, ee_orn, _resolve_gripper_cmd(
            getattr(wp, "gripper", None), current_cfg, obs.L_gripper_position
        )

    # joint_lerp_t waypoints (descend_pick midpoints, transit midpoints) DO
    # have a wp.pos target — use it as the action label instead of skipping
    # or falling back to the current EE pose. This ensures the dataset covers
    # the full descend trajectory (z from hover down to pick_z).
    if getattr(wp, "joint_lerp_t", None) is not None:
        pos = wp.pos if getattr(wp, "pos", None) is not None else ee_pos
        orn = wp.orn if getattr(wp, "orn", None) is not None else ee_orn
        grip = _resolve_gripper_cmd(getattr(wp, "gripper", None), current_cfg, obs.L_gripper_position)
        return pos, orn, grip

    pos = wp.pos if getattr(wp, "pos", None) is not None else ee_pos
    orn = wp.orn if getattr(wp, "orn", None) is not None else ee_orn
    grip = _resolve_gripper_cmd(getattr(wp, "gripper", None), current_cfg, obs.L_gripper_position)
    return pos, orn, grip


def _build_summary_path(args):
    if args.save_summary is not None:
        return args.save_summary
    return args.output_root / "taskboard_lerobot_summary.json"


def _resolve_sample_stride(args, physics_hz):
    if args.sample_stride is not None:
        if args.sample_stride <= 0:
            raise ValueError("--sample-stride must be > 0")
        stride = int(args.sample_stride)
    else:
        if args.sample_hz <= 0:
            raise ValueError("--sample-hz must be > 0 when --sample-stride is unset")
        stride = max(1, int(round(float(physics_hz) / float(args.sample_hz))))
    return stride


def _resolve_recording_config(args, physics_hz):
    if args.sample_stride is not None:
        stride = _resolve_sample_stride(args, physics_hz)
        fps = max(1, int(round(float(physics_hz) / float(stride))))
        return {
            "mode": "stride",
            "sample_stride": stride,
            "sample_fps": fps,
            "sample_period_s": float(stride) / float(physics_hz),
        }

    target_hz = float(args.sample_hz)
    return {
        "mode": "time",
        "sample_stride": None,
        "sample_fps": max(1, int(round(target_hz))),
        "sample_period_s": 1.0 / target_hz,
    }


def main():
    args = _parse_args()
    if args.max_recorded_frames < 0:
        raise ValueError("--max-recorded-frames must be >= 0")
    if args.max_parts < 0:
        raise ValueError("--max-parts must be >= 0")
    if args.disturbance_prob < 0.0 or args.disturbance_prob > 1.0:
        raise ValueError("--disturbance-prob must be in [0, 1]")
    if args.disturbance_duration_steps <= 0:
        raise ValueError("--disturbance-duration-steps must be > 0")
    if args.disturbance_pos_max < 0.0:
        raise ValueError("--disturbance-pos-max must be >= 0")
    if args.disturbance_rot_max_deg < 0.0:
        raise ValueError("--disturbance-rot-max-deg must be >= 0")

    original_part_order = list(rp.pc.part_order)
    known_parts = set(getattr(rp.pc, "PART_CONFIG", {}).keys())
    if args.part_name not in known_parts:
        raise ValueError(
            f"unknown --part-name {args.part_name!r}; known parts: {sorted(known_parts)}"
        )
    rp.pc.part_order = [args.part_name]

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    physics_hz = 200.0
    physics_dt = 1.0 / physics_hz
    recording_cfg = _resolve_recording_config(args, physics_hz)
    sample_stride = recording_cfg["sample_stride"]
    sample_fps = recording_cfg["sample_fps"]
    effective_max_parts = 1  # v4 always collects exactly one overridden part per episode
    effective_max_frames = None if args.max_recorded_frames == 0 else int(args.max_recorded_frames)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=sample_fps,
        robot_type="vega_1u_taskboard",
        features=_feature_spec(args.include_depth),
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        vcodec="h264",
    )

    camera_output_enabled = True
    enable_camera_viewports = bool(rp.pc.enable_camera_viewports and not rp._HEADLESS)
    render_each_step = True

    dummy_target = np.zeros(3, dtype=np.float64)
    (
        my_world,
        my_controller,
        my_robots,
        head_depth_camera,
        l_wrist_camera,
        r_wrist_camera,
        articulation_controller,
        _task_params,
        reset_needed,
    ) = rp.setup_pick_place_sim(
        L_object_prim_path=rp.pc.L_object_prim_path,
        R_object_prim_path=rp.pc.R_object_prim_path,
        L_target_position=dummy_target,
        R_target_position=dummy_target,
        joint_opened_position=np.array([rp.pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([rp.pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=enable_camera_viewports,
        enable_camera_output=camera_output_enabled,
    )
    rp.import_missing_parts()

    l_controller = my_controller["L"]
    r_controller = my_controller["R"]
    l_robot = my_robots["L"]

    dof_names = list(l_robot.dof_names)
    r_arm_dof_indices = np.array([dof_names.index(j) for j in rp.R_ARM_JOINT_NAMES], dtype=np.int64)
    l_arm_dof_indices = np.array([dof_names.index(j) for j in sorted(j for j in dof_names if j.startswith("L_arm_j"))], dtype=np.int64)
    l_gripper_dof_index = dof_names.index("L_gripper_joint")
    r_gripper_dof_index = dof_names.index("R_gripper_joint")
    l_arm_joint_names = [j for j in dof_names if j.startswith("L_arm_j")]

    def _apply_init_joint_targets():
        targets = getattr(rp.pc, "INIT_JOINT_TARGETS", None)
        if not targets:
            return
        full_q = np.asarray(l_robot.get_joint_positions(), dtype=np.float64).copy()
        for jname, val in targets.items():
            if jname in dof_names:
                full_q[dof_names.index(jname)] = float(val)
        l_robot.set_joint_positions(full_q)
        l_robot.set_joint_velocities(np.zeros(len(dof_names)))

    _apply_init_joint_targets()
    r_arm_hold_q = np.asarray(l_robot.get_joint_positions())[r_arm_dof_indices].astype(np.float64)
    l_arm_init_q = np.asarray(l_controller.current_cspace_q(), dtype=np.float64).copy()

    env_info = EnvInfo(
        dof_names=dof_names,
        L_arm_joints=l_arm_joint_names,
        R_arm_joints=list(rp.R_ARM_JOINT_NAMES),
        L_gripper_joint="L_gripper_joint",
        L_arm_init_q=l_arm_init_q.copy(),
        physics_dt=physics_dt,
        enable_camera_output=True,
        L_controller=l_controller,
    )
    policy_class = rp._load_policy_class(args.policy)
    policy = policy_class(env_info)
    print(f"[collect] policy: {policy_class.__module__}.{policy_class.__name__}")

    stage = rp.omni.usd.get_context().get_stage()
    physx_iface = rp.omni.physx.get_physx_interface()
    parts_iter = iter(rp.pc.part_order)
    current_part = None
    current_snap_attacher = None
    current_snap_sub = None
    snap_fired_parts = set()
    part_step_count = 0
    run_complete = False
    part_activation_count = 0
    recorded_frames = 0
    policy_action_ready = False
    last_recorded_step = None
    next_record_time_s = 0.0
    record_time_origin_s = None
    pending_record_queue = deque()
    record_sub = None
    recorded_parts = []
    part_completion_reasons = []
    final_completion_reason = None
    per_part_timeout_steps = int(getattr(rp.pc, "PER_PART_TIMEOUT_STEPS", 3000))
    warmup_steps = int(getattr(rp.pc, "WARMUP_STEPS", 0))
    effective_max_sim_steps = (
        None if args.max_sim_steps == 0 else int(args.max_sim_steps)
    )
    rng = np.random.default_rng(args.disturbance_seed)
    disturbance_events = []
    active_disturbance = None

    def _clear_snap_state():
        nonlocal current_snap_attacher, current_snap_sub
        current_snap_sub = None
        current_snap_attacher = None

    def _actual_ee_pose(controller):
        pos, orn = controller.end_effector.get_world_pose()
        return np.asarray(pos, dtype=np.float64), np.asarray(orn, dtype=np.float64)

    def _camera_payload():
        rgb = {"head": None, "L_wrist": None, "R_wrist": None}
        depth = {"head": None, "L_wrist": None, "R_wrist": None}
        for key, cam in (("head", head_depth_camera), ("L_wrist", l_wrist_camera), ("R_wrist", r_wrist_camera)):
            if cam is None:
                continue
            try:
                rgba = cam.get_rgba()
                if rgba is not None and rgba.size > 0:
                    rgb[key] = np.asarray(rgba[..., :3], dtype=np.uint8)
            except Exception:
                pass
            if args.include_depth:
                try:
                    frame = cam.get_current_frame()
                    if frame and frame.get("distance_to_image_plane") is not None:
                        depth[key] = np.asarray(frame["distance_to_image_plane"], dtype=np.float32)
                except Exception:
                    pass
        return rgb, depth

    def _build_observation(step_idx_override=None):
        full_q = np.asarray(l_robot.get_joint_positions(), dtype=np.float64)
        try:
            full_qd = np.asarray(l_robot.get_joint_velocities(), dtype=np.float64)
        except Exception:
            full_qd = np.zeros_like(full_q)

        ee_pos, ee_orn = _actual_ee_pose(l_controller)
        rgb, depth = _camera_payload()

        snap_fired = bool(current_snap_attacher is not None and current_snap_attacher.attached)
        return Observation(
            step_idx=(int(my_world.current_time_step_index) if step_idx_override is None else int(step_idx_override)),
            joint_positions=full_q,
            joint_velocities=full_qd,
            L_gripper_position=float(full_q[l_gripper_dof_index]),
            ee_pose_L=(ee_pos, ee_orn),
            rgb=rgb,
            depth=depth,
            intrinsics={"head": None, "L_wrist": None, "R_wrist": None},
            snap_fired=snap_fired,
            target_part=current_part if isinstance(current_part, str) else None,
        )

    def _build_part_target(name):
        cfg = rp.pc.get_part_config(name)
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
            snap_rot_tol_deg=(None if snap.get("rot_tol_deg") is None else float(snap["rot_tol_deg"])),
            gripper_open=float(cfg.get("gripper_open", 0.0)),
            gripper_close=float(cfg.get("gripper_close", 0.0)),
            ee_orientation=_arr(cfg.get("ee_orientation")),
            extra=dict(cfg),
        )

    def _start_next_part():
        nonlocal current_part, current_snap_attacher, current_snap_sub
        nonlocal part_step_count, run_complete, part_activation_count, policy_action_ready

        if current_part is not None and current_snap_attacher is not None and current_snap_attacher.attached:
            snap_fired_parts.add(current_part)
        _clear_snap_state()
        policy_action_ready = False

        if part_activation_count >= effective_max_parts:
            current_part = None
            run_complete = True
            return None

        try:
            current_part = next(parts_iter)
        except StopIteration:
            current_part = None
            run_complete = True
            return None

        part_activation_count += 1
        recorded_parts.append(current_part)
        cfg = rp.pc.get_part_config(current_part)
        if cfg.get("release_mode", "open") == "snap":
            snap_cfg = cfg.get("snap")
            current_snap_attacher = rp.build_snap_attacher(stage, current_part, snap_cfg)
            attacher = current_snap_attacher
            current_snap_sub = physx_iface.subscribe_physics_step_events(lambda dt, a=attacher: a.update())

        obs = _build_observation()
        target = _build_part_target(current_part)
        policy.reset(obs, target)
        part_step_count = 0
        print(f"[collect] now working on the part: {current_part}", flush=True)
        return current_part

    def _restart_iteration():
        nonlocal parts_iter, current_part, last_recorded_step, next_record_time_s
        nonlocal policy_action_ready, record_time_origin_s
        _clear_snap_state()
        snap_fired_parts.clear()
        pending_record_queue.clear()
        last_recorded_step = None
        next_record_time_s = 0.0
        record_time_origin_s = None
        policy_action_ready = False
        if stage is not None:
            for name in rp.pc.PART_CONFIG.keys():
                joint_path = f"/World/_snap_joint_{name}"
                if rp.is_prim_path_valid(joint_path):
                    stage.RemovePrim(joint_path)
        rp.restore_scene_part_xforms()
        parts_iter = iter(rp.pc.part_order)
        current_part = None
        _start_next_part()

    def _current_waypoint():
        wp = getattr(policy, "current_waypoint", None)
        return wp() if callable(wp) else wp

    def _waypoint_name(wp):
        if wp is None:
            return None
        return getattr(wp, "name", None)

    def _disturbance_allowed(wp):
        if wp is None:
            return False
        if getattr(wp, "cspace_target", None) is not None:
            return False
        if getattr(wp, "joint_lerp_t", None) is not None:
            return False
        return True

    def _apply_optional_disturbance(l_action, obs, current_cfg):
        nonlocal active_disturbance
        if not args.disturb:
            return l_action

        step_idx = int(my_world.current_time_step_index)
        wp = _current_waypoint()
        if not _disturbance_allowed(wp):
            if active_disturbance is not None and step_idx > active_disturbance["end_step"]:
                active_disturbance = None
            return l_action

        if active_disturbance is not None and step_idx > active_disturbance["end_step"]:
            active_disturbance = None

        if active_disturbance is None and rng.random() < args.disturbance_prob:
            offset_xyz = _random_vector_with_max_norm(rng, args.disturbance_pos_max)
            rotvec = _random_vector_with_max_norm(
                rng, np.deg2rad(args.disturbance_rot_max_deg)
            )
            active_disturbance = {
                "start_step": int(step_idx),
                "end_step": int(step_idx + args.disturbance_duration_steps - 1),
                "duration_steps": int(args.disturbance_duration_steps),
                "offset_xyz_m": offset_xyz,
                "rotvec_rad": rotvec,
                "waypoint_name": _waypoint_name(wp),
            }
            disturbance_events.append(_json_safe(active_disturbance.copy()))

        if active_disturbance is None:
            return l_action

        target = _left_action_target(
            policy, obs, current_cfg, skip_non_ee_actions=args.skip_non_ee_actions
        )
        if target is None:
            return l_action

        base_pos, base_orn, grip = target
        disturbed_pos = np.asarray(base_pos, dtype=np.float64).reshape(3) + np.asarray(
            active_disturbance["offset_xyz_m"], dtype=np.float64
        )
        disturbed_orn = _quat_multiply_wxyz(
            _rotvec_to_quat_wxyz(active_disturbance["rotvec_rad"]),
            np.asarray(base_orn, dtype=np.float64).reshape(4),
        )
        return l_controller.forward(disturbed_pos, disturbed_orn, grip)

    def _infer_task_label(completion_reason):
        cfg = rp.pc.get_part_config(args.part_name)
        release_mode = cfg.get("release_mode", "open")
        snap_success = bool(args.part_name in snap_fired_parts)
        if release_mode == "snap":
            success = snap_success or completion_reason == "snap_done"
        else:
            success = completion_reason == "policy_done"
        if completion_reason == "timeout":
            success = False

        label = {
            "label_type": "single_part_action_chunk",
            "part_name": args.part_name,
            "success": bool(success),
            "score": 1.0 if success else 0.0,
            "score_source": "fallback_completion_reason",
            "completion_reason": completion_reason,
            "release_mode": release_mode,
            "snap_fired": snap_success,
            "disturb": bool(args.disturb),
            "disturbance_config": {
                "prob": args.disturbance_prob,
                "duration_steps": args.disturbance_duration_steps,
                "pos_max_m": args.disturbance_pos_max,
                "rot_max_deg": args.disturbance_rot_max_deg,
                "seed": args.disturbance_seed,
            },
            "disturbance_events": _json_safe(disturbance_events),
        }

        # If the original task runner exposes an official scoring helper, use
        # it. The fallback above remains the source of truth when no helper is
        # available in the local repository version.
        for fn_name in (
            "compute_task_score",
            "score_task",
            "evaluate_task_score",
            "grade_task",
            "compute_score",
        ):
            fn = getattr(rp, fn_name, None)
            if not callable(fn):
                continue
            for call in (
                lambda: fn(args.part_name),
                lambda: fn(args.part_name, stage),
                lambda: fn(stage, args.part_name),
                lambda: fn(),
            ):
                try:
                    official = call()
                except TypeError:
                    continue
                except Exception as exc:
                    label["official_score_error"] = f"{fn_name}: {exc}"
                    continue
                if isinstance(official, dict):
                    label.update(_json_safe(official))
                    label.setdefault("score_source", fn_name)
                else:
                    try:
                        label["score"] = float(official)
                        label["score_source"] = fn_name
                    except Exception:
                        label["official_score_raw"] = repr(official)
                        label["score_source"] = fn_name
                return label
        return label

    def _queue_record_frame(step_idx):
        nonlocal last_recorded_step, next_record_time_s, record_time_origin_s
        sim_time_s = step_idx * physics_dt
        if last_recorded_step is not None and step_idx < last_recorded_step:
            last_recorded_step = None
            next_record_time_s = 0.0
            record_time_origin_s = None

        if run_complete or current_part is None:
            return
        if warmup_steps > 0 and step_idx < warmup_steps:
            return

        if recording_cfg["mode"] == "stride":
            if last_recorded_step is not None and step_idx - last_recorded_step < sample_stride:
                return
        else:
            if sim_time_s + 1e-9 < next_record_time_s:
                return

        if record_time_origin_s is None:
            record_time_origin_s = sim_time_s

        obs = _build_observation(step_idx_override=step_idx)
        l_pos, l_orn = obs.ee_pose_L
        r_pos, r_orn = _actual_ee_pose(r_controller)
        full_q = np.asarray(obs.joint_positions, dtype=np.float32)
        full_qd = np.asarray(obs.joint_velocities, dtype=np.float32)
        r_grip = float(full_q[r_gripper_dof_index])
        l_grip = float(full_q[l_gripper_dof_index])

        # Use actual EE pose as action label (behavior cloning from real trajectory).
        # This captures the full pick trajectory including cspace_target phases.
        # One-step lag is acceptable for imitation learning.
        frame = {
            "actions.left_arm_action": _pack_cartesian_action(l_pos, l_orn, l_grip),
            "actions.right_arm_action": _pack_cartesian_action(r_pos, r_orn, r_grip),
            "observations.left_arm_ee_pose": _pack_pose(l_pos, l_orn),
            "observations.right_arm_ee_pose": _pack_pose(r_pos, r_orn),
            "observations.left_arm_joint_position": full_q[l_arm_dof_indices].astype(np.float32),
            "observations.right_arm_joint_position": full_q[r_arm_dof_indices].astype(np.float32),
            "observations.left_arm_joint_velocity": full_qd[l_arm_dof_indices].astype(np.float32),
            "observations.right_arm_joint_velocity": full_qd[r_arm_dof_indices].astype(np.float32),
            "observations.left_gripper_position": np.array([float(full_q[l_gripper_dof_index])], dtype=np.float32),
            "observations.right_gripper_position": np.array([float(full_q[r_gripper_dof_index])], dtype=np.float32),
            "observations.rgb_head": _as_rgb(obs.rgb.get("head")),
            "observations.rgb_left_hand": _as_rgb(obs.rgb.get("L_wrist")),
            "observations.rgb_right_hand": _as_rgb(obs.rgb.get("R_wrist")),
            "task": TASK_DESCRIPTION,
        }
        if args.include_depth:
            frame.update({
                "observations.depth_head": _encode_depth_image(obs.depth.get("head")),
                "observations.depth_left_hand": _encode_depth_image(obs.depth.get("L_wrist")),
                "observations.depth_right_hand": _encode_depth_image(obs.depth.get("R_wrist")),
            })
        pending_record_queue.append({
            "frame": frame,
            "sim_time_s": float(sim_time_s - record_time_origin_s),
            "step_idx": int(step_idx),
        })
        last_recorded_step = step_idx
        if recording_cfg["mode"] == "time":
            while next_record_time_s <= sim_time_s + 1e-9:
                next_record_time_s += recording_cfg["sample_period_s"]

    def _flush_record_queue():
        nonlocal recorded_frames
        while pending_record_queue:
            if effective_max_frames is not None and recorded_frames >= effective_max_frames:
                pending_record_queue.clear()
                break
            item = pending_record_queue.popleft()
            dataset.add_frame(item["frame"])
            if dataset.episode_buffer is not None and dataset.episode_buffer["timestamp"]:
                dataset.episode_buffer["timestamp"][-1] = item["sim_time_s"]
            recorded_frames += 1
            frame_cap_label = ("unbounded" if effective_max_frames is None else str(effective_max_frames))
            print(f"[collect] recorded frame {recorded_frames}/{frame_cap_label} at sim step {item['step_idx']}")

    def _on_physics_step(dt):
        del dt
        _queue_record_frame(int(my_world.current_time_step_index))

    summary = {
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "part_name": args.part_name,
        "original_part_order": original_part_order,
        "effective_part_order": list(rp.pc.part_order),
        "label_json": str(_build_label_path(args, output_root)),
        "requested_sample_hz": args.sample_hz,
        "recording_mode": recording_cfg["mode"],
        "sample_stride": sample_stride,
        "sample_fps": sample_fps,
        "sample_period_s": recording_cfg["sample_period_s"],
        "max_recorded_frames": args.max_recorded_frames,
        "effective_max_recorded_frames": effective_max_frames,
        "max_parts": args.max_parts,
        "effective_max_parts": effective_max_parts,
        "max_sim_steps": args.max_sim_steps,
        "effective_max_sim_steps": effective_max_sim_steps,
        "include_depth": bool(args.include_depth),
        "disturb": bool(args.disturb),
        "skip_non_ee_actions": bool(args.skip_non_ee_actions),
        "joint_dof_per_arm": 7,
        "action_encoding": "absolute_cartesian_target_xyz_rotvec_gripper",
        "notes": [
            "right arm is held at its initial pose by the current runner",
            "depth is stored as 16-bit PNG images in millimeters",
            "frames are selected from physics-step callbacks instead of the outer control loop",
            "stored timestamps are episode-relative physics times written explicitly into the dataset",
            "sample-hz mode uses simulation-time scheduling so 30 Hz is not rounded down to a fixed 7-step stride",
            "vega_1u exposes 7 arm joints per side, not 6",
            "v4 temporarily overrides rp.pc.part_order to a single selected part",
            "optional EE disturbances perturb executed commands but not stored expert labels",
        ],
    }
    max_sim_steps_label = "unbounded" if effective_max_sim_steps is None else str(effective_max_sim_steps)
    print(f"[collect] effective max sim steps: {max_sim_steps_label}")

    try:
        record_sub = physx_iface.subscribe_physics_step_events(_on_physics_step)
        _restart_iteration()
        while True:
            my_world.step(render=render_each_step)
            _flush_record_queue()
            if run_complete:
                break
            if not my_world.is_playing():
                if my_world.is_stopped():
                    reset_needed = True
                continue
            if reset_needed:
                my_world.reset()
                reset_needed = False
                _apply_init_joint_targets()
                l_controller.reset()
                r_controller.reset()
                _restart_iteration()
            if my_world.current_time_step_index == 0:
                _apply_init_joint_targets()
                l_controller.reset()
                r_controller.reset()
                _restart_iteration()

            if warmup_steps > 0 and my_world.current_time_step_index < warmup_steps:
                _apply_init_joint_targets()
                continue

            if effective_max_sim_steps is not None and int(my_world.current_time_step_index) >= effective_max_sim_steps:
                raise RuntimeError(
                    "collector exceeded max sim steps before finishing the requested episode"
                )

            if current_part is None:
                continue

            obs = _build_observation()
            if current_snap_attacher is not None and current_snap_attacher.attached:
                snap_fired_parts.add(current_part)

            cfg = rp.pc.get_part_config(current_part)
            is_snap_done = (
                cfg.get("release_mode") == "snap"
                and current_snap_attacher is not None
                and current_snap_attacher.attached
            )
            is_timeout = part_step_count >= per_part_timeout_steps
            if policy.is_done(obs) or is_snap_done or is_timeout:
                completion_reason = (
                    "snap_done" if is_snap_done else
                    "timeout" if is_timeout else
                    "policy_done"
                )
                final_completion_reason = completion_reason
                if current_part is not None:
                    part_completion_reasons.append({
                        "part": current_part,
                        "reason": completion_reason,
                        "step_count": part_step_count,
                    })
                _start_next_part()
                continue

            l_action = policy.act(obs)
            policy_action_ready = True
            if effective_max_frames is not None and recorded_frames >= effective_max_frames:
                final_completion_reason = "frame_cap"
                run_complete = True
                break

            l_action = _apply_optional_disturbance(l_action, obs, cfg)

            r_action_positions = [None] * len(dof_names)
            for j_idx, val in zip(r_arm_dof_indices, r_arm_hold_q.tolist()):
                r_action_positions[j_idx] = float(val)
            r_action = ArticulationAction(joint_positions=r_action_positions)
            merged = rp.merge_bimanual_actions(l_action, r_action, dof_names)
            articulation_controller.apply_action(merged)
            part_step_count += 1

        if recorded_frames == 0:
            raise RuntimeError("no frames were recorded")
        _flush_record_queue()
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()

        if final_completion_reason is None:
            final_completion_reason = (
                part_completion_reasons[-1]["reason"] if part_completion_reasons else "unknown"
            )
        label = _infer_task_label(final_completion_reason)
        label.update({
            "output_root": str(output_root),
            "repo_id": args.repo_id,
            "recorded_frames": recorded_frames,
            "recorded_parts": recorded_parts,
            "part_completion_reasons": part_completion_reasons,
            "sample_fps": sample_fps,
            "action_encoding": "absolute_cartesian_target_xyz_rotvec_gripper",
        })
        label_path = _build_label_path(args, output_root)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(json.dumps(_json_safe(label), indent=2), encoding="utf-8")

        summary.update({
            "recorded_frames": recorded_frames,
            "recorded_parts": recorded_parts,
            "part_completion_reasons": part_completion_reasons,
            "disturbance_events": _json_safe(disturbance_events),
            "task_label": _json_safe(label),
            "results_json": str(args.results_json),
            "label_json": str(label_path),
        })
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        summary_path = _build_summary_path(args)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        print(json.dumps(_json_safe(summary), indent=2))
    except Exception:
        print("[collect] unhandled exception:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        try:
            rp.pc.part_order = original_part_order
        except Exception:
            pass
        record_sub = None
        try:
            rp.simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()