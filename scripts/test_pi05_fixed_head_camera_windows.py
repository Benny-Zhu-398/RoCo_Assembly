"""Verify the pi0.5 world-fixed head camera without applying robot actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-steps", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="artifacts/pi05_fixed_head_camera",
    )
    args, _ = parser.parse_known_args()
    if args.render_steps < 2:
        parser.error("--render-steps must be at least 2")
    return args


ARGS = _parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

from isaacsim.sensors.camera import Camera

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))

import param_config as pc
from controllers.vega_1u_setup import (
    setup_pick_place_sim,
    sync_fixed_camera_to_source,
)
from deferred_video import _rgb8


def _quat_distance_deg(q0, q1):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(q0 @ q1), 0.0, 1.0))))


def _write_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def main():
    output_dir = Path(ARGS.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_target = np.zeros(3, dtype=np.float64)
    (
        world,
        _,
        robots,
        head_camera,
        _,
        _,
        _,
        _,
        _,
    ) = setup_pick_place_sim(
        L_object_prim_path=pc.L_object_prim_path,
        R_object_prim_path=pc.R_object_prim_path,
        L_target_position=dummy_target,
        R_target_position=dummy_target,
        joint_opened_position=np.array([pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=False,
        enable_camera_output=True,
        fixed_head_camera=True,
    )
    if head_camera is None:
        raise RuntimeError("fixed head camera was not created")

    robot = robots["L"]
    dof_names = list(robot.dof_names)

    def _apply_init_joint_targets():
        full_q = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
        for name, value in getattr(pc, "INIT_JOINT_TARGETS", {}).items():
            if name in dof_names:
                full_q[dof_names.index(name)] = float(value)
        robot.set_joint_positions(full_q)
        robot.set_joint_velocities(np.zeros(len(dof_names)))

    _apply_init_joint_targets()
    source_camera_path = (
        "/World/robotics/vega_1u_gripper/zed_depth_frame/headcam"
    )
    source_camera = Camera(
        prim_path=source_camera_path,
        name="Pi05SourceHeadCamDiagnostic",
        resolution=(640, 480),
        frequency=30.0,
    )
    source_camera.initialize()
    warmup_steps = int(getattr(pc, "WARMUP_STEPS", 0))
    initial_position = None
    initial_orientation = None
    camera_frozen = False

    max_position_drift_m = 0.0
    max_rotation_drift_deg = 0.0
    captured_frames = []
    source_frames = []
    valid_frames = []
    world.play()
    for step in range(ARGS.render_steps):
        world.step(render=True)
        if warmup_steps > 0 and world.current_time_step_index < warmup_steps:
            _apply_init_joint_targets()
            sync_fixed_camera_to_source(head_camera, source_camera_path)
            continue
        if not camera_frozen:
            _apply_init_joint_targets()
            initial_position, initial_orientation = sync_fixed_camera_to_source(
                head_camera, source_camera_path
            )
            initial_position = np.asarray(initial_position, dtype=np.float64)
            initial_orientation = np.asarray(initial_orientation, dtype=np.float64)
            camera_frozen = True
            continue

        position, orientation = head_camera.get_world_pose(camera_axes="usd")
        max_position_drift_m = max(
            max_position_drift_m,
            float(np.linalg.norm(np.asarray(position) - initial_position)),
        )
        max_rotation_drift_deg = max(
            max_rotation_drift_deg,
            _quat_distance_deg(initial_orientation, orientation),
        )

        frame = head_camera.get_rgba()
        if frame is None:
            continue
        if np.asarray(frame).size == 0:
            continue
        rgb = _rgb8(frame)
        if rgb is None or rgb.shape != (480, 640, 3):
            continue
        captured_frames.append((step, rgb.copy()))
        source_frame = source_camera.get_rgba()
        source_rgb = None if source_frame is None else _rgb8(source_frame)
        if source_rgb is not None and source_rgb.shape == (480, 640, 3):
            source_frames.append((step, source_rgb.copy()))
        if float(rgb.mean()) <= 1.0 or float(rgb.std()) <= 1.0:
            continue
        valid_frames.append((step, rgb.copy()))

    first_step, first_rgb = captured_frames[0] if captured_frames else (None, None)
    last_step, last_rgb = captured_frames[-1] if captured_frames else (None, None)
    source_first_step, source_first_rgb = (
        source_frames[0] if source_frames else (None, None)
    )
    source_last_step, source_last_rgb = (
        source_frames[-1] if source_frames else (None, None)
    )
    first_path = output_dir / "head_first.png"
    last_path = output_dir / "head_last.png"
    if first_rgb is not None:
        _write_rgb(first_path, first_rgb)
        _write_rgb(last_path, last_rgb)
    source_first_path = output_dir / "source_head_first.png"
    source_last_path = output_dir / "source_head_last.png"
    if source_first_rgb is not None:
        _write_rgb(source_first_path, source_first_rgb)
        _write_rgb(source_last_path, source_last_rgb)

    source_position, source_orientation = source_camera.get_world_pose(
        camera_axes="usd"
    )
    fixed_position, fixed_orientation = head_camera.get_world_pose(camera_axes="usd")

    report = {
        "render_steps": ARGS.render_steps,
        "warmup_physics_steps": warmup_steps,
        "captured_frame_count": len(captured_frames),
        "valid_frame_count": len(valid_frames),
        "source_frame_count": len(source_frames),
        "first_captured_step": first_step,
        "last_captured_step": last_step,
        "first_valid_step": None if not valid_frames else valid_frames[0][0],
        "last_valid_step": None if not valid_frames else valid_frames[-1][0],
        "source_first_step": source_first_step,
        "source_last_step": source_last_step,
        "initial_position": initial_position.tolist(),
        "initial_orientation_usd_wxyz": initial_orientation.tolist(),
        "max_position_drift_m": max_position_drift_m,
        "max_rotation_drift_deg": max_rotation_drift_deg,
        "final_source_fixed_position_delta_m": float(
            np.linalg.norm(np.asarray(source_position) - np.asarray(fixed_position))
        ),
        "final_source_fixed_rotation_delta_deg": _quat_distance_deg(
            source_orientation, fixed_orientation
        ),
        "first_rgb_mean": None if first_rgb is None else float(first_rgb.mean()),
        "first_rgb_std": None if first_rgb is None else float(first_rgb.std()),
        "last_rgb_mean": None if last_rgb is None else float(last_rgb.mean()),
        "last_rgb_std": None if last_rgb is None else float(last_rgb.std()),
        "mean_abs_first_last": (
            None
            if first_rgb is None
            else float(
                np.abs(
                    first_rgb.astype(np.float32) - last_rgb.astype(np.float32)
                ).mean()
            )
        ),
        "first_image": None if first_rgb is None else str(first_path),
        "last_image": None if last_rgb is None else str(last_path),
        "source_first_image": (
            None if source_first_rgb is None else str(source_first_path)
        ),
        "source_last_image": (
            None if source_last_rgb is None else str(source_last_path)
        ),
        "mean_abs_fixed_source_first": (
            None
            if first_rgb is None or source_first_rgb is None
            else float(
                np.abs(
                    first_rgb.astype(np.float32)
                    - source_first_rgb.astype(np.float32)
                ).mean()
            )
        ),
    }
    report["passed"] = bool(
        max_position_drift_m <= 1e-7
        and max_rotation_drift_deg <= 1e-6
        and len(captured_frames) >= ARGS.render_steps - 5
        and len(valid_frames) >= ARGS.render_steps - 5
    )
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


try:
    main()
except BaseException as exc:
    output_dir = Path(ARGS.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    error = {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with (output_dir / "error.json").open("w", encoding="utf-8") as stream:
        json.dump(error, stream, indent=2)
    print(json.dumps(error, indent=2), file=sys.stderr, flush=True)
finally:
    simulation_app.close()
