from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "rocochallenge2025/rocochallenge2026_Industrial_Assembly",
    revision="main"   # ← 加这一行
)

print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)
print("fps:", ds.fps)
for k, v in ds.features.items():
    print(k, v)

sample = ds[0]
for k, v in sample.items():
    print(k, getattr(v, "shape", type(v)))