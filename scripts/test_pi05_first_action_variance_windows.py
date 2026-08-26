"""Measure pi0.5 first-action variance for one fixed real observation."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from test_pi05_action_cache_windows import _build_fixture
from test_pi05_next_action_windows import _array_text, _load_rgb, _load_state


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))

from policies.pi05_lerobot import Pi05LeRobotPolicy  # noqa: E402


FORMAL_CHECKPOINT = (
    "/media/iam-lab/strange_external/yudongluo/pi05/outputs/"
    "roco_pi05_expert_3000_20260728_180527/checkpoints/003000/pretrained_model"
)
PRODUCTION_SERVER = "/home/yudongluo/user/Roco/RoCo_Assembly/task/pi05_server.py"


def rotation_from_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat([x, y, z, w])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument(
        "--observation-log",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "pi05_fixed_head_remote_dry_run_20260728"
            / "pi05_one_step.log"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "pi05_fixed_head_remote_dry_run_20260728",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "pi05_first_action_variance.log",
    )
    args = parser.parse_args()
    if args.requests < 2:
        parser.error("--requests must be at least 2")

    state = _load_state(args.observation_log.resolve())
    env_info, observation, target, left_controller, right_controller = _build_fixture(
        state
    )
    observation.rgb = {
        "head": _load_rgb(args.image_dir / "pi05_dry_run_head.png"),
        "L_wrist": _load_rgb(args.image_dir / "pi05_dry_run_left.png"),
        "R_wrist": _load_rgb(args.image_dir / "pi05_dry_run_right.png"),
    }

    os.environ["PI05_REMOTE"] = "1"
    os.environ["PI05_EXEC_HORIZON"] = "1"
    os.environ.setdefault("PI05_REMOTE_SERVER", PRODUCTION_SERVER)
    os.environ.setdefault("PI05_REMOTE_CHECKPOINT", FORMAL_CHECKPOINT)
    os.environ.setdefault(
        "PI05_SERVER_LOG",
        str(REPO_ROOT / "artifacts" / "pi05_first_action_variance_server.log"),
    )
    os.environ.setdefault(
        "PI05_CLIENT_LOG",
        str(REPO_ROOT / "artifacts" / "pi05_first_action_variance_client.log"),
    )

    current_rotation = rotation_from_wxyz(state[3:7])
    records = []
    policy = None
    try:
        policy = Pi05LeRobotPolicy(env_info)
        for request_index in range(1, args.requests + 1):
            policy.reset(observation, target)
            started = time.perf_counter()
            prediction = policy.predict_raw(observation)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            action = np.asarray(prediction["action"], dtype=np.float64)
            translation_m = float(np.linalg.norm(action[:3] - state[:3]))
            predicted_rotation = Rotation.from_euler("xyz", action[3:6])
            rotation_deg = float(
                np.degrees(
                    np.linalg.norm(
                        (predicted_rotation * current_rotation.inv()).as_rotvec()
                    )
                )
            )
            records.append(
                {
                    "request_index": request_index,
                    "action": action,
                    "translation_m": translation_m,
                    "rotation_deg": rotation_deg,
                    "hard_hold": translation_m > 0.10 or rotation_deg > 45.0,
                    "elapsed_ms": elapsed_ms,
                }
            )
    finally:
        if policy is not None:
            policy.close()

    if left_controller.forward_calls or right_controller.forward_calls:
        raise RuntimeError("a robot controller was called during variance diagnostic")

    translations = np.array([record["translation_m"] for record in records])
    rotations = np.array([record["rotation_deg"] for record in records])
    hold_count = sum(record["hard_hold"] for record in records)
    lines = [
        "pi0.5 fixed-observation first-action variance",
        "safety: no Isaac, IK, controller.forward, or robot action",
        f"requests: {args.requests}",
        f"hard_hold_count: {hold_count}",
        f"translation_m_min_median_max: {translations.min():.9f} "
        f"{np.median(translations):.9f} {translations.max():.9f}",
        f"rotation_deg_min_median_max: {rotations.min():.9f} "
        f"{np.median(rotations):.9f} {rotations.max():.9f}",
        f"left_controller_forward_calls: {left_controller.forward_calls}",
        f"right_controller_forward_calls: {right_controller.forward_calls}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"request_index: {record['request_index']}",
                f"elapsed_ms: {record['elapsed_ms']:.3f}",
                f"translation_m: {record['translation_m']:.9f}",
                f"rotation_deg: {record['rotation_deg']:.9f}",
                f"hard_hold: {record['hard_hold']}",
                f"action_14d: {_array_text(record['action'])}",
                "",
            ]
        )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]), flush=True)
    print(f"log_file: {args.output}", flush=True)


if __name__ == "__main__":
    main()
