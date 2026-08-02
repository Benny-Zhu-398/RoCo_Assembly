"""Solve saved pi0.5 actions with Isaac IK without applying robot actions."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action-log",
        type=Path,
        default=Path("artifacts/pi05_first_action_variance_3000_20260728.log"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pi05_ik_no_apply_saved_actions_20260729.json"),
    )
    return parser.parse_known_args()[0]


ARGS = parse_args()
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": True})

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))

import param_config as pc
from controllers.vega_1u_setup import setup_pick_place_sim
from policies.pi05_lerobot import (
    GRIPPER_OPEN_LIMIT,
    _euler_xyz_to_quat_wxyz,
    _filter_left_action,
    _guard_left_joint_action,
)


def load_actions(path):
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"action_14d:\s*\[(.*?)\]", text, flags=re.DOTALL)
    actions = [np.fromstring(match.replace(",", " "), sep=" ") for match in matches]
    if not actions or any(action.shape != (14,) for action in actions):
        raise RuntimeError(f"could not parse 14-D actions from {path}")
    return actions


def main():
    action_log = ARGS.action_log
    if not action_log.is_absolute():
        action_log = REPO_ROOT / action_log
    output = ARGS.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    actions = load_actions(action_log)
    dummy_target = np.zeros(3, dtype=np.float64)
    (
        world,
        controllers,
        robots,
        _,
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
        enable_camera_output=False,
        fixed_head_camera=False,
    )
    controller = controllers["L"]
    robot = robots["L"]
    dof_names = list(robot.dof_names)
    left_indices = [
        dof_names.index(name)
        for name in sorted(name for name in dof_names if name.startswith("L_arm_j"))
    ]
    gripper_index = dof_names.index("L_gripper_joint")

    def apply_init_targets():
        q = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
        for name, value in pc.INIT_JOINT_TARGETS.items():
            if name in dof_names:
                q[dof_names.index(name)] = float(value)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(len(dof_names)))

    apply_init_targets()
    world.play()
    for _ in range(int(getattr(pc, "WARMUP_STEPS", 60))):
        apply_init_targets()
        world.step(render=False)

    current_q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    current_position, current_quat = controller.end_effector.get_world_pose()
    current_position = np.asarray(current_position, dtype=np.float64)
    current_quat = np.asarray(current_quat, dtype=np.float64)
    current_gripper = float(current_q[gripper_index]) / GRIPPER_OPEN_LIMIT

    records = []
    for index, raw_action in enumerate(actions, start=1):
        filtered, ee_safety = _filter_left_action(
            raw_action,
            current_position,
            current_quat,
            current_gripper,
        )
        target_quat = _euler_xyz_to_quat_wxyz(*filtered[3:6])
        target_gripper = float(np.clip(filtered[6], 0.0, 1.0)) * GRIPPER_OPEN_LIMIT
        ik_action = controller.forward(filtered[:3], target_quat, target_gripper)
        targets = list(ik_action.joint_positions or [None] * len(current_q))
        _, joint_safety = _guard_left_joint_action(
            targets,
            current_q,
            left_indices,
            gripper_index,
        )
        ik_left = np.array(
            [np.nan if targets[i] is None else targets[i] for i in left_indices],
            dtype=np.float64,
        )
        delta_deg = np.degrees(ik_left - current_q[left_indices])
        records.append(
            {
                "request_index": index,
                "raw_action": raw_action.tolist(),
                "filtered_action": filtered.tolist(),
                "ee_filter": ee_safety,
                "ik_success": bool(controller.ik.ik_ok),
                "current_left_joints_rad": current_q[left_indices].tolist(),
                "ik_left_targets_rad": ik_left.tolist(),
                "ik_left_delta_deg": delta_deg.tolist(),
                "ik_left_max_abs_delta_deg": float(np.nanmax(np.abs(delta_deg))),
                "joint_guard": joint_safety,
            }
        )

    report = {
        "safety": "IK solved only; no articulation action applied",
        "action_count": len(records),
        "apply_action_calls": 0,
        "physics_steps_after_ik": 0,
        "current_left_xyz": current_position.tolist(),
        "current_left_quaternion_wxyz": current_quat.tolist(),
        "records": records,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "action_count": len(records),
                "ik_success_count": sum(record["ik_success"] for record in records),
                "joint_guard_hold_count": sum(
                    record["joint_guard"]["held"] for record in records
                ),
                "max_joint_delta_deg": max(
                    record["ik_left_max_abs_delta_deg"] for record in records
                ),
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
