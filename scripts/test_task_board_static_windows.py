"""Inspect task-board physics and measure zero-control pose drift in Isaac Sim."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Physics steps to observe after the normal warmup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/task_board_static_20260729.json"),
    )
    parser.add_argument(
        "--fix-task-board",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_known_args()[0]


ARGS = parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import param_config as pc
from controllers.vega_1u_setup import setup_pick_place_sim


BOARD_ROOT = "/World/task_board"
BOARD_COLOR = "/World/task_board/task_board_color"


def _json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return list(value)
    except TypeError:
        return str(value)


def _pose(prim):
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return {
        "translation": [float(v) for v in translation],
        "quaternion_wxyz": [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
    }


def _pose_delta(start, end):
    start_t = np.asarray(start["translation"], dtype=np.float64)
    end_t = np.asarray(end["translation"], dtype=np.float64)
    start_q = np.asarray(start["quaternion_wxyz"], dtype=np.float64)
    end_q = np.asarray(end["quaternion_wxyz"], dtype=np.float64)
    start_q /= np.linalg.norm(start_q)
    end_q /= np.linalg.norm(end_q)
    dot = float(np.clip(abs(np.dot(start_q, end_q)), 0.0, 1.0))
    return {
        "translation_m": float(np.linalg.norm(end_t - start_t)),
        "rotation_deg": float(np.degrees(2.0 * np.arccos(dot))),
    }


def _physics_attributes(prim):
    values = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if name.startswith("physics:") or name.startswith("physx"):
            try:
                values[name] = _json_value(attribute.Get())
            except Exception as exc:
                values[name] = f"<read failed: {exc}>"
    return values


def _inspect_board(stage):
    root = stage.GetPrimAtPath(BOARD_ROOT)
    if not root or not root.IsValid():
        raise RuntimeError(f"missing task-board prim: {BOARD_ROOT}")

    records = []
    for prim in Usd.PrimRange(root):
        schemas = list(prim.GetAppliedSchemas())
        physics_attributes = _physics_attributes(prim)
        is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_mass = prim.HasAPI(UsdPhysics.MassAPI)
        if schemas or physics_attributes or is_rigid or is_collision or has_mass:
            records.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "applied_schemas": schemas,
                    "rigid_body": bool(is_rigid),
                    "collision": bool(is_collision),
                    "mass": bool(has_mass),
                    "physics_attributes": physics_attributes,
                }
            )
    return records


def main():
    if ARGS.steps <= 0:
        raise ValueError("--steps must be positive")

    output = ARGS.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy_target = np.zeros(3, dtype=np.float64)
    world, _, robots, _, _, _, _, _, _ = setup_pick_place_sim(
        L_object_prim_path=pc.L_object_prim_path,
        R_object_prim_path=pc.R_object_prim_path,
        L_target_position=dummy_target,
        R_target_position=dummy_target,
        joint_opened_position=np.array([pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=False,
        enable_camera_output=False,
        fixed_head_camera=False,
        fix_task_board=ARGS.fix_task_board,
    )
    robot = robots["L"]
    dof_names = list(robot.dof_names)

    def hold_initial_robot_state():
        positions = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
        for name, value in pc.INIT_JOINT_TARGETS.items():
            if name in dof_names:
                positions[dof_names.index(name)] = float(value)
        robot.set_joint_positions(positions)
        robot.set_joint_velocities(np.zeros(len(dof_names)))

    stage = omni.usd.get_context().get_stage()
    board_prim = stage.GetPrimAtPath(BOARD_ROOT)
    color_prim = stage.GetPrimAtPath(BOARD_COLOR)
    if not color_prim or not color_prim.IsValid():
        raise RuntimeError(f"missing task-board color prim: {BOARD_COLOR}")

    physics_records = _inspect_board(stage)
    before_play = {
        BOARD_ROOT: _pose(board_prim),
        BOARD_COLOR: _pose(color_prim),
    }

    hold_initial_robot_state()
    world.play()
    for _ in range(int(getattr(pc, "WARMUP_STEPS", 60))):
        hold_initial_robot_state()
        world.step(render=False)

    after_warmup = {
        BOARD_ROOT: _pose(board_prim),
        BOARD_COLOR: _pose(color_prim),
    }
    max_drift = {path: {"translation_m": 0.0, "rotation_deg": 0.0} for path in after_warmup}
    for _ in range(ARGS.steps):
        hold_initial_robot_state()
        world.step(render=False)
        for path, prim in ((BOARD_ROOT, board_prim), (BOARD_COLOR, color_prim)):
            delta = _pose_delta(after_warmup[path], _pose(prim))
            max_drift[path]["translation_m"] = max(
                max_drift[path]["translation_m"], delta["translation_m"]
            )
            max_drift[path]["rotation_deg"] = max(
                max_drift[path]["rotation_deg"], delta["rotation_deg"]
            )

    final_pose = {
        BOARD_ROOT: _pose(board_prim),
        BOARD_COLOR: _pose(color_prim),
    }
    report = {
        "safety": (
            "No policy, SSH, IK, controller forward, or articulation apply_action; "
            "robot joints were held at configured initial state using direct state setters."
        ),
        "warmup_steps": int(getattr(pc, "WARMUP_STEPS", 60)),
        "observation_steps": ARGS.steps,
        "render": False,
        "camera_output": False,
        "task_board_fix_requested": bool(ARGS.fix_task_board),
        "before_play": before_play,
        "after_warmup": after_warmup,
        "final_pose": final_pose,
        "warmup_delta": {
            path: _pose_delta(before_play[path], after_warmup[path])
            for path in before_play
        },
        "final_delta_from_warmup": {
            path: _pose_delta(after_warmup[path], final_pose[path])
            for path in after_warmup
        },
        "max_delta_from_warmup": max_drift,
        "task_board_physics": physics_records,
        "rigid_body_paths": [
            record["path"] for record in physics_records if record["rigid_body"]
        ],
        "collision_prim_count": sum(
            record["collision"] for record in physics_records
        ),
        "apply_action_calls": 0,
        "policy_requests": 0,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "rigid_body_paths": report["rigid_body_paths"],
                "collision_prim_count": report["collision_prim_count"],
                "warmup_delta": report["warmup_delta"],
                "max_delta_from_warmup": report["max_delta_from_warmup"],
                "apply_action_calls": 0,
                "policy_requests": 0,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


try:
    main()
except BaseException as exc:
    output = ARGS.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    raise
finally:
    simulation_app.close()
