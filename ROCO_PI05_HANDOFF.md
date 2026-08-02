# RoCo pi0.5 部署、训练与验证交接

更新时间：2026-07-23（America/New_York）

## 新对话使用方法

上传本文件，然后发送：

```text
请完整阅读 ROCO_PI05_HANDOFF.md，直接在 iam-strange 本机核对实际状态并继续。
不要 SSH 连接本机，不要重复已完成的下载和训练，不要删除现有数据、缓存或 checkpoint。
以服务器文件、日志和 checkpoint 为准，从“下一步”继续。
```

## 总体目标

```text
Windows：Isaac Sim + RoCo 仿真 + 相机 + IK + 评分
iam-strange：LeRobot + pi0.5 + checkpoint + GPU 训练/推理
最终：Windows → SSH/网络协议 → iam-strange pi05_server.py
```

目前没有修改 Windows 通信代码。正式 checkpoint 训练并验证完成前不要开始该部分。

## 主机和目录

```text
hostname: iam-strange
user: yudongluo
GPU: 3 × NVIDIA RTX A6000 48 GB
推荐 GPU: 物理 GPU 2（进程内映射为 cuda:0）
PI05_ROOT: /media/iam-lab/strange_external/yudongluo/pi05
RoCo repo: /home/yudongluo/user/Roco/RoCo_Assembly
```

每次执行：

```bash
source ~/pi05_env.sh
export CUDA_VISIBLE_DEVICES=2
```

大型文件只能放进 `$PI05_ROOT`。不要递归修改整块外接盘权限，不要使用 sudo。

## LeRobot 和 Python

```text
LeRobot tag: v0.6.0
commit: 30da8e687a6dfc617fcd94afc367ac7071c376ce
source: $PI05_ROOT/src/lerobot
```

detached HEAD 是正常状态。

旧环境 `$PI05_ROOT/envs/lerobot` 使用 Python 3.14.6，与当前 `draccus/argparse` 不兼容：训练 CLI 和 checkpoint 配置加载都会失败。不要用于训练/推理，也不要删除。

有效环境：

```text
$PI05_ROOT/envs/lerobot-py312
Python 3.12.13
LeRobot 0.6.0
PyTorch 2.11.0+cu128
```

该环境已通过训练 CLI、CUDA kernel、数据集、checkpoint 和 server 测试。后续显式使用：

```bash
PY312="$PI05_ROOT/envs/lerobot-py312/bin"
```

## CUDA 和 Hugging Face

物理 GPU 2 的真实 FP16 kernel 已验证：RTX A6000、CUDA available、矩阵乘成功。

用户已完成 Hugging Face 登录，并验证可访问 gated repo：

```text
google/paligemma-3b-pt-224
```

检查身份时不要打印 token：

```bash
"$PI05_ROOT/envs/lerobot-py312/bin/hf" auth whoami
```

## RoCo 数据集

```text
repo_id: rocochallenge2025/rocochallenge2026_Industrial_Assembly
revision: main
commit: dc03b003f94d184b2b20465ed986456ee1bf2a3c
root: $PI05_ROOT/datasets/rocochallenge2026_Industrial_Assembly
episodes: 200
frames: 121454
fps: 10
task: assemble parts onto the task board
```

Hub 只有 `main`，没有 dataset tag。必须显式使用 `root=` 和 `revision="main"`，不能依赖默认 revision。

真实 features：

```text
observation.images.head          RGB 240×320×3；sample 为 float32 3×240×320
observation.images.left_hand     RGB 240×320×3
observation.images.right_hand    RGB 240×320×3
observation.state                float32, 44-D
action                           float32, 14-D
```

44-D state 顺序：左 EE xyz+qwxyz、右 EE xyz+qwxyz、左右各 7 个 joint position、左右各 7 个 joint velocity、左右 gripper。

14-D action 顺序：

```text
left xyz + left rx/ry/rz + left gripper
right xyz + right rx/ry/rz + right gripper
```

动作是 absolute EE pose，`use_relative_actions=False`。state/action 使用 checkpoint 的 quantile normalization。

## 必需的相机 rename_map

`lerobot/pi05_base` 与 RoCo 相机键不同，训练必须加入：

```bash
--rename_map='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_hand":"observation.images.left_wrist_0_rgb","observation.images.right_hand":"observation.images.right_wrist_0_rgb"}'
```

映射会保存进 checkpoint preprocessor；已验证现有 `pi05_server.py` 使用 RoCo 原始键时能正确处理。

## pi05_base

```text
model: lerobot/pi05_base
revision: 7de663972b7817d2c4cf2d84c821153dfea772e9
size: 约 13.47 GiB
cache: $HF_HOME/hub
```

已成功加载到 GPU 2：约 15.45 GB 显存，`chunk_size=50`，`n_action_steps=50`，absolute actions。

## 已完成训练

### 1-step smoke

```text
output: $PI05_ROOT/outputs/roco_pi05_smoke
checkpoint: checkpoints/000001/pretrained_model
loss: 0.663
gradient norm: 6.127
peak memory log: 12.85 GB
```

### 50-step short run

```text
output: $PI05_ROOT/outputs/roco_pi05_short_20260723_1520
checkpoint: checkpoints/000050/pretrained_model
expert-only, bfloat16, batch 1, workers 0
gradient checkpointing true, compile false
WandB false, Hub push false
speed: 约 2.5 step/s（模型加载后）
peak memory log: 12.85 GB
loss min/max/mean: 0.398 / 3.707 / 1.5181
step 50 loss: 1.674
```

所有记录的 loss、gradient 和 LR 都是有限值。50 步会把默认 scheduler 自动压缩到 50 步，只能说明链路稳定，不能说明模型收敛。

当前推荐 checkpoint：

```text
$PI05_ROOT/outputs/roco_pi05_short_20260723_1520/checkpoints/000050/pretrained_model
```

最新路径记录：

```text
$PI05_ROOT/logs/latest_checkpoint_path.txt
```

## checkpoint/server 验证

1-step 和 50-step checkpoint 都通过：

- `PI05Policy.from_pretrained()`；
- 权重 keys 全部匹配；
- `config.json`、`model.safetensors`、pre/postprocessor 文件齐全；
- `/home/yudongluo/user/Roco/RoCo_Assembly/task/pi05_server.py` 加载成功；
- length-prefixed pickle 协议成功；
- `reset` 返回成功；
- 三相机 + 44-D 合成观测完整推理成功；
- 返回有限的 shape `(14,)` action；
- 测试结束后 GPU 已释放。

当前 RoCo runner 只执行左侧 7-D，右臂保持固定。

## 辅助脚本

位于 `$PI05_ROOT/scripts`：

```text
check_environment.sh
check_cuda.py
check_dataset.py
inspect_dataset_schema.py
load_pi05_base.py
train_roco_pi05_smoke.sh
train_roco_pi05_short.sh
find_latest_checkpoint.sh
test_pi05_server.sh
```

## 关键日志

```text
$PI05_ROOT/logs/roco_pi05_interface_report.md
$PI05_ROOT/logs/check_cuda.log
$PI05_ROOT/logs/lerobot_train_help.txt
$PI05_ROOT/logs/load_pi05_base.log
$PI05_ROOT/logs/train_roco_pi05_smoke.log
$PI05_ROOT/logs/load_smoke_checkpoint.log
$PI05_ROOT/logs/pi05_protocol_smoke.log
$PI05_ROOT/logs/pi05_server_protocol.log
$PI05_ROOT/logs/train_roco_pi05_short.log
$PI05_ROOT/logs/pi05_short_protocol_smoke.log
$PI05_ROOT/logs/pi05_server_short_protocol.log
$PI05_ROOT/logs/latest_checkpoint_path.txt
```

## 磁盘状态

最后检查：外接盘剩余约 73 GB；有效环境约 11 GB；HF cache 约 14 GB；数据集约 1.1 GB；smoke 和 50-step 输出各约 11 GB；单个 `model.safetensors` 约 9.35 GB。

不要擅自删除 HF cache、数据集、环境或 checkpoint。需要清理时先列出精确目标、大小、用途和可恢复性，并获得用户确认。

曾保留的失败诊断目录（不是有效 checkpoint）：

```text
$PI05_ROOT/outputs/roco_pi05_smoke_failed_feature_names_20260723_1449
$PI05_ROOT/outputs/roco_pi05_smoke_failed_hf_auth_20260723
```

## 已知问题（不要重复调查）

1. Python 3.14 与 draccus/argparse 不兼容；使用 Python 3.12。
2. 数据集没有 tag；显式 `revision=main`。
3. 相机 feature 名不一致；使用上述 rename_map。
4. PaliGemma gated 401 已通过 HF 登录和权限解决。
5. 非 resume 训练要求 output dir 事先不存在；脚本已经检查，不能预创建最终目录。

## 下一步

全链路已经完成。下一阶段是正式 expert-only 训练，建议起点：

```text
steps: 3000
batch_size: 1
dtype: bfloat16
train_expert_only: true
gradient_checkpointing: true
compile: false
GPU: physical GPU 2
wandb/push_to_hub: false
```

启动前必须决定 checkpoint 策略。每 1000 步保存可能消耗约 30 GB；只保存最终 checkpoint 更安全。未经用户确认不要删除当前 smoke/50-step 输出。

建议新输出目录：

```text
$PI05_ROOT/outputs/roco_pi05_expert_3000_<date>
```

正式训练前保存 Git commit、完整 CLI、环境版本、dataset/base revision、GPU 和磁盘状态。完成后重复 checkpoint 独立加载和 server 完整协议测试，再开始 Windows 远程 sidecar。

## 新对话首先执行

直接在本机执行，不要 SSH：

```bash
hostname
whoami
pwd
source ~/pi05_env.sh
echo "$PI05_ROOT"

cd "$PI05_ROOT/src/lerobot"
git status --short --branch
git describe --tags --always
git rev-parse HEAD

"$PI05_ROOT/envs/lerobot-py312/bin/python" --version
"$PI05_ROOT/envs/lerobot-py312/bin/hf" auth whoami
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

cat "$PI05_ROOT/logs/latest_checkpoint_path.txt"
cat "$PI05_ROOT/logs/roco_pi05_interface_report.md"
ls -l "$PI05_ROOT/scripts"
du -sh "$PI05_ROOT/outputs"/* 2>/dev/null
df -h "$PI05_ROOT"
```

如检查结果与本文不同，以服务器当前状态为准，先解释差异，不要自动覆盖或删除。

## 工作约束

- Codex 已运行在 iam-strange，本机不要 `ssh iam-strange`。
- 不使用 sudo，不修改其他用户文件，不打印 token。
- 大文件只写 `$PI05_ROOT`。
- 不覆盖已有训练输出；使用新目录或明确 resume。
- 训练前检查 GPU，不抢占他人任务。
- 未经批准不删除数据、缓存、环境或 checkpoint。
- 正式 checkpoint 验证前不修改 Windows 通信代码。
- 第一版远程通信优先复用 stdin/stdout + pickle，不要一开始重写 FastAPI。
