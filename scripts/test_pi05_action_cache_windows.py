"""Dry-run ten cached Pi0.5 actions without Isaac or robot control."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from test_pi05_next_action_windows import _array_text, _load_rgb, _load_state


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_DIR = _REPO_ROOT / "task"

import sys

sys.path.insert(0, str(_TASK_DIR))

from policies.pi05_lerobot import (  # noqa: E402
    GRIPPER_OPEN_LIMIT,
    Pi05LeRobotPolicy,
)
from policy_api import EnvInfo, Observation, PartTarget  # noqa: E402


_DIAGNOSTIC_SERVER = (
    "/media/iam-lab/strange_external/yudongluo/pi05/scripts/"
    "pi05_server_action_chunk_diagnostic.py"
)


class _FakeEndEffector:
    def __init__(self, position, quaternion):
        self._position = np.asarray(position, dtype=np.float64)
        self._quaternion = np.asarray(quaternion, dtype=np.float64)

    def get_world_pose(self):
        return self._position.copy(), self._quaternion.copy()


class _NoActionController:
    def __init__(self, position, quaternion):
        self.end_effector = _FakeEndEffector(position, quaternion)
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("robot controller must not be called during cache dry-run")


def _build_fixture(state):
    left_joints = [f"L_joint_{index}" for index in range(1, 8)]
    right_joints = [f"R_joint_{index}" for index in range(1, 8)]
    dof_names = left_joints + right_joints + ["L_gripper_joint", "R_gripper_joint"]

    joint_positions = np.zeros(len(dof_names), dtype=np.float64)
    joint_velocities = np.zeros(len(dof_names), dtype=np.float64)
    joint_positions[:7] = state[14:21]
    joint_positions[7:14] = state[21:28]
    joint_positions[14] = state[42] * GRIPPER_OPEN_LIMIT
    joint_positions[15] = state[43] * GRIPPER_OPEN_LIMIT
    joint_velocities[:7] = state[28:35]
    joint_velocities[7:14] = state[35:42]

    left_controller = _NoActionController(state[0:3], state[3:7])
    right_controller = _NoActionController(state[7:10], state[10:14])
    env_info = EnvInfo(
        dof_names=dof_names,
        L_arm_joints=left_joints,
        R_arm_joints=right_joints,
        L_gripper_joint="L_gripper_joint",
        L_arm_init_q=joint_positions[:7].copy(),
        physics_dt=1.0 / 60.0,
        enable_camera_output=True,
        L_controller=left_controller,
        R_controller=right_controller,
    )

    images = {
        "head": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_head.png"),
        "L_wrist": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_left.png"),
        "R_wrist": _load_rgb(_REPO_ROOT / "artifacts" / "pi05_dry_run_right.png"),
    }
    observation = Observation(
        step_idx=0,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        L_gripper_position=float(joint_positions[14]),
        ee_pose_L=(state[0:3].copy(), state[3:7].copy()),
        rgb=images,
        depth={"head": None, "L_wrist": None, "R_wrist": None},
        intrinsics={"head": None, "L_wrist": None, "R_wrist": None},
    )
    target = PartTarget(name="diagnostic", release_mode="open")
    return env_info, observation, target, left_controller, right_controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-log",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pi05_one_step_dry_run.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "pi05_ten_action_cache_dry_run.log",
    )
    args = parser.parse_args()

    state = _load_state(args.observation_log.resolve())
    env_info, observation, target, left_controller, right_controller = _build_fixture(state)

    os.environ["PI05_REMOTE"] = "1"
    os.environ["PI05_EXEC_HORIZON"] = "5"
    os.environ.setdefault("PI05_REMOTE_SERVER", _DIAGNOSTIC_SERVER)
    os.environ.setdefault(
        "PI05_SERVER_LOG",
        str(_REPO_ROOT / "artifacts" / "pi05_action_chunk_server.log"),
    )
    os.environ.setdefault(
        "PI05_CLIENT_LOG",
        str(_REPO_ROOT / "artifacts" / "pi05_action_cache_client.log"),
    )

    policy = None
    records = []
    try:
        policy = Pi05LeRobotPolicy(env_info)
        policy.reset(observation, target)
        for action_index in range(1, 11):
            started = time.perf_counter()
            result = policy.predict_cached_action(observation)
            call_ms = (time.perf_counter() - started) * 1000.0
            action = np.asarray(result["action"], dtype=np.float64)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid action {action_index}: {action}")
            records.append({**result, "call_ms": call_ms, "action": action})
    finally:
        if policy is not None:
            policy.close()

    replan_positions = [record["control_step"] for record in records if record["replanned"]]
    if replan_positions != [1, 6]:
        raise RuntimeError(f"expected replans at actions [1, 6], got {replan_positions}")
    if left_controller.forward_calls or right_controller.forward_calls:
        raise RuntimeError("a robot controller was called during the dry-run")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "pi0.5 ten-action client cache dry-run",
        "safety: no Isaac, IK, controller.forward, or robot action",
        "PI05_EXEC_HORIZON=5",
        f"replan_positions: {replan_positions}",
        f"left_controller_forward_calls: {left_controller.forward_calls}",
        f"right_controller_forward_calls: {right_controller.forward_calls}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"action_index: {record['control_step']}",
                f"replanned: {record['replanned']}",
                f"replan_index: {record['replan_index']}",
                f"call_ms: {record['call_ms']:.3f}",
                f"chunk_request_ms: {record['chunk_request_ms']}",
                f"chunk_length: {record['chunk_length']}",
                f"cache_remaining: {record['cache_remaining']}",
                f"action_14d: {_array_text(record['action'])}",
                "",
            ]
        )
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print("ten-action cache dry-run passed")
    print(f"replan positions: {replan_positions}")
    print("controller forward calls: 0")
    print(f"log file: {args.output}")


if __name__ == "__main__":
    main()
