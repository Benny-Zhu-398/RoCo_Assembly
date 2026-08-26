"""Compare the RoCo training state/action distribution with an Isaac reset."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_ROOT = (
    "/media/iam-lab/strange_external/yudongluo/pi05/datasets/"
    "rocochallenge2026_Industrial_Assembly"
)
REPO_ID = "rocochallenge2025/rocochallenge2026_Industrial_Assembly"
CURRENT_LEFT_XYZ = np.array([0.25116101, -0.09056653, 1.3754114])
CURRENT_LEFT_QUAT_WXYZ = np.array(
    [-0.19758347, -0.88201094, -0.29920083, 0.30577192], dtype=np.float64
)


def rotation_from_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat([x, y, z, w])


def angle_deg(first, second):
    return float(np.degrees(np.linalg.norm((first * second.inv()).as_rotvec())))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, revision="main")
table = dataset.hf_dataset.select_columns(["observation.state", "action"]).with_format(
    "numpy"
)
print(f"loaded length={len(dataset)} episodes={dataset.num_episodes}", flush=True)
current_rotation = rotation_from_wxyz(CURRENT_LEFT_QUAT_WXYZ)

episode_starts = {
    int(dataset.meta.episodes[index]["dataset_from_index"])
    for index in range(dataset.num_episodes)
}
translation_deltas = []
euler_rotation_deltas = []
rotvec_rotation_deltas = []
start_translation_deltas = []
start_euler_rotation_deltas = []
nearest = None

for frame_index in range(len(dataset)):
    frame = table[frame_index]
    state = np.asarray(frame["observation.state"], dtype=np.float64)
    action = np.asarray(frame["action"], dtype=np.float64)

    translation_delta = float(np.linalg.norm(action[:3] - state[:3]))
    state_rotation = rotation_from_wxyz(state[3:7])
    euler_delta = angle_deg(Rotation.from_euler("xyz", action[3:6]), state_rotation)
    rotvec_delta = angle_deg(Rotation.from_rotvec(action[3:6]), state_rotation)
    translation_deltas.append(translation_delta)
    euler_rotation_deltas.append(euler_delta)
    rotvec_rotation_deltas.append(rotvec_delta)

    if frame_index in episode_starts:
        start_translation_deltas.append(translation_delta)
        start_euler_rotation_deltas.append(euler_delta)

    state_distance = float(np.linalg.norm(state[:3] - CURRENT_LEFT_XYZ))
    candidate = (state_distance, frame_index, state, action)
    if nearest is None or candidate[0] < nearest[0]:
        nearest = candidate

sample = table[0]
sample_state = np.asarray(sample["observation.state"], dtype=np.float64)
sample_action = np.asarray(sample["action"], dtype=np.float64)
print(f"length={len(dataset)} episodes={dataset.num_episodes}")
print(f"frame0_state={np.array2string(sample_state, precision=9, max_line_width=240)}")
print(f"frame0_action={np.array2string(sample_action, precision=9, max_line_width=240)}")
print(f"all_translation_delta_m={quantiles(translation_deltas)}")
print(f"episode_start_translation_delta_m={quantiles(start_translation_deltas)}")
print(f"all_euler_action_vs_state_deg={quantiles(euler_rotation_deltas)}")
print(f"all_rotvec_action_vs_state_deg={quantiles(rotvec_rotation_deltas)}")
print(f"episode_start_euler_action_vs_state_deg={quantiles(start_euler_rotation_deltas)}")
print(
    "current_left_euler_xyz="
    f"{np.array2string(current_rotation.as_euler('xyz'), precision=9)}"
)
assert nearest is not None
print(
    "nearest_training_state: "
    f"frame={nearest[1]} xyz_distance_m={nearest[0]:.9f} "
    f"state_xyz={np.array2string(nearest[2][:3], precision=9)} "
    f"action_xyz={np.array2string(nearest[3][:3], precision=9)} "
    f"action_minus_state={np.array2string(nearest[3][:3] - nearest[2][:3], precision=9)} "
    f"action_rotation={np.array2string(nearest[3][3:6], precision=9)}"
)
