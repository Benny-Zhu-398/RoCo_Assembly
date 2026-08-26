"""Export one RoCo dataset observation for visual comparison."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_ROOT = (
    "/media/iam-lab/strange_external/yudongluo/pi05/datasets/"
    "rocochallenge2026_Industrial_Assembly"
)
REPO_ID = "rocochallenge2025/rocochallenge2026_Industrial_Assembly"
OUTPUT_DIR = Path(
    os.environ.get(
        "PI05_REFERENCE_OUTPUT",
        "/media/iam-lab/strange_external/yudongluo/pi05/logs/"
        "pi05_dataset_reference_20260728",
    )
)


def to_uint8_hwc(value):
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, revision="main")
sample = dataset[0]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for feature, filename in (
    ("observation.images.head", "dataset_frame0_head.png"),
    ("observation.images.left_hand", "dataset_frame0_left.png"),
    ("observation.images.right_hand", "dataset_frame0_right.png"),
):
    image = to_uint8_hwc(sample[feature])
    path = OUTPUT_DIR / filename
    Image.fromarray(image).save(path)
    print(
        f"{path} shape={image.shape} min={image.min()} max={image.max()} "
        f"mean={image.mean():.3f} std={image.std():.3f}",
        flush=True,
    )
