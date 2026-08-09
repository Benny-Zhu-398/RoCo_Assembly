# pi0.5 Isaac Sim 验证 — Linux 部署交接

调查时间：2026-08-08。目标机器：Windows 11 笔记本（RTX 5090 Laptop 24 GB，
Intel Ultra 9 275HX）。结论：**在这台机器的当前驱动下 Isaac Sim 跑不起来**，
改为等原生 Linux 环境。本文记录已验证的部分、已知的坑，以及在 Linux 上
应该怎么走。

## 一句话架构

```
Isaac Sim（本地）── ssh 常驻连接 / 长度前缀 pickle ──> iam-strange 的 pi05_server.py
  task/run_pick_place.py                                  PI05Policy.from_pretrained(ckpt)
  └ task/policies/pi05_lerobot.py                         └ GPU 推理，返回 14-D action
```

pi0.5 推理**留在服务器**，本地只跑仿真。这是 `ROCO_PI05_HANDOFF.md` 定下的
设计，代码里的 `_REMOTE_DEFAULTS` 也是照这个写的。

## 已验证通过（不需要重做）

- **远端推理全链路**。`scripts/test_pi05_remote_windows.py` 在 Windows 上用
  真实 checkpoint 跑通：reset → 44-D state + 三路 240×320 图像 → 有限的
  14-D action。首次生成 50 步 chunk 约 694 ms。
- **SSH 免密**到 `iam-strange`（`~/.ssh/config` 已配，指向 128.2.178.27）。
- **服务器侧完好**：`pi05_env.sh`、`envs/lerobot-py312`、`pi05_server.py`、
  三组 checkpoint 都在；3×A6000 空闲。

## 阻塞点：Isaac Sim + Blackwell + 新驱动

Isaac Sim 5.1.0（Kit 107.3.3）在 `app ready` 之后立刻崩，原生栈：

```
000: rtx.scenedb.plugin.dll!carbOnPluginStartup+0x252db
006: carb.scenerenderer-rtx.plugin.dll!...
009: omni.hydra.rtx.plugin.dll!+0x553c
```

两次运行完全复现。这是 sm_120（Blackwell）+ 新驱动的**已知问题**，多个
GPU 型号（5060 Ti / 5070 Ti / 5080 / 5090）都有相同报告。

**官方验证过的驱动版本：**

| 平台 | 版本 |
|---|---|
| Linux | **580.65.06** |
| Windows | 580.88 |

本机是 610.88。**在 Linux 上务必钉住 580 系列**，不要装 latest ——
驱动 595 / 610 分支会在 `librtx.scenedb.plugin.so` 崩在同一位置。

参考：
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252
- https://github.com/isaac-sim/IsaacSim/issues/651
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335

## WSL2 为什么不行（别再试了）

WSL2 通过 `/dev/dxg` → D3D12 → Windows 驱动访问 GPU，只映射了 CUDA，
**没有映射 Vulkan/OpenGL**。`/usr/lib/wsl/lib` 里没有任何 Vulkan ICD，
`vulkaninfo` 只能看到 Mesa 的软件/转译驱动。Mesa 的 dozen(dzn) 能把 Vulkan
转到 D3D12，但不支持光追扩展，而 Omniverse RTX 渲染器必须要。

也不能用 headless 绕开：pi0.5 是视觉策略，必须要三路相机图像，相机渲染
走的就是 RTX 渲染器。

## Linux 上的部署步骤

1. 装 NVIDIA 驱动 **580.65.06**（不要 latest）。
2. `uv sync` —— 仓库根目录的 `pyproject.toml` / `.python-version` /
   `uv.lock` 就是为 linux x86_64 锁的，直接可用（约 18 GB）。
   `uv` 会自己管 Python 3.11，不需要 conda。
3. `export OMNI_KIT_ACCEPT_EULA=YES`
4. 冒烟：`uv run python -c "import isaacsim; print('ready')"`
5. 基线：`uv run python task/run_pick_place.py`（scripted policy），
   确认仿真 + 相机 + IK + 评分正常，顺便量显存。

## 评测口径（已和用户确认）

对比实验对象是 `training/diffusion_policy/` 的 state-only Diffusion Policy。

- **重规划节奏对齐到 16 步**：`PI05_EXEC_HORIZON=16`，DP 保持默认
  （`DP_N_ACTION_STEPS` 默认 = horizon = 16）。
  两边默认值原本不匹配（pi0.5 默认 1，DP 默认 16），重规划频率会成为
  混淆变量，所以固定住它。
- **不开 safety filter**：`PI05_SAFETY_FILTER` 保持关闭。它会把模型输出
  限幅到单步 5mm/5°，等于加了个外部控制器，对比不公平。
  （注意：`_guard_left_joint_action` 那个 IK 关节跳变保护是常开的，
  防的是数值爆炸，不属于策略层干预。）
- **只跑训练过的零件**：每个 checkpoint 只见过一个 part group，rollout
  必须只迭代该组零件。为此加了 `ROCO_PART_ORDER` 环境变量覆盖
  （见 `task/param_config.py`，不设时行为完全不变）。
- **只用完整跑完的 group**：A 组在 step 8000/17137 因磁盘满挂了，只有
  半程的 4000（LR 没退完），暂不使用。

## 要跑的两次 rollout

```bash
# Group B — usb_a, hdmi
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupB_20260807_003521/checkpoints/007082/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
PI05_EXEC_HORIZON=16 \
ROCO_PART_ORDER=usb_a,hdmi \
ISAACSIM_HEADLESS=1 \
uv run python task/run_pick_place.py \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video artifacts/pi05_groupB_head.mp4 \
  --results-json artifacts/pi05_groupB_results.json

# Group C — rod_16mm, bolt_8mm, pin
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
PI05_EXEC_HORIZON=16 \
ROCO_PART_ORDER=rod_16mm,bolt_8mm,pin \
ISAACSIM_HEADLESS=1 \
uv run python task/run_pick_place.py \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video artifacts/pi05_groupC_head.mp4 \
  --results-json artifacts/pi05_groupC_results.json
```

在 Linux 上 `PI05_SSH_EXE` 要覆盖成 `/usr/bin/ssh`（默认值是 Windows 的
`C:\Windows\System32\OpenSSH\ssh.exe`）。

时间估算：`PER_PART_TIMEOUT_STEPS = 3000`，horizon 16 时每零件约 2 分钟
推理开销（3000/16 × 694 ms），仿真本身另计。

## 三组 checkpoint 与训练参数

统一配方（`$PI05_ROOT/scripts/train_group.sh`）：base `lerobot/pi05_base`，
batch 16，5 epochs，bfloat16，gradient checkpointing，全量微调
（`train_expert_only=false`，`freeze_vision_encoder=false`），
`decay_steps = steps`（LR 在本次 run 内走完退火），
数据划分 `val_fraction=0.1, split_seed=0`（与 DP 一致）。

| 组 | 零件 | train frames | steps | 状态 | 终点 checkpoint |
|---|---|---|---|---|---|
| A | gear_20teeth, gear_60teeth, battery_size1, battery_size5 | 54838 | 17137 | ✗ 8000 步 ENOSPC | 仅半程 004000 |
| B | usb_a, hdmi | 22661 | 7082 | ✓ | `checkpoints/007082` |
| C | rod_16mm, bolt_8mm, pin | 31844 | 9951 | ✓ | `checkpoints/009951` |

离线 VAL 分数（`VAL.left_trans_m.all.mean`，越小越好）：
A@4000 = 0.00926，B@6000 = 0.01081，B@7082 = 0.01116。
B 已 plateau；仍建议用 7082 以保持「5 epoch」口径统一，选 6000 属于
cherry-pick。

## ⚠️ 未解决：服务器上的 pi05_server.py 与仓库不一致

远端模式执行的是**服务器上那份**
（`/home/yudongluo/user/Roco/RoCo_Assembly/task/pi05_server.py`），
它和仓库里的 `task/pi05_server.py` 有一处实质差异：

- **仓库版**：只有 `exec_horizon > 1` 时才 `policy.reset()`。horizon=1 时
  新观测被忽略，直接从已有队列弹出 → 每 50 步才真正重规划一次。
- **服务器版**：收到完整观测**一律** `policy.reset()` 重规划
  （注释写明 "A full observation is always a closed-loop replan"）。
  另外多了命令白名单，日志标签是 `cmd=replan`。

服务器上留有 `logs/pi05_server_before_replan_20260802.py` 备份，说明服务器版
是有意的修复、仓库版是旧的。**正式跑之前应该把仓库同步到服务器那份**，
并在结果里记录用的是哪个 commit，否则复现不了。

## 旋转表示：本分支的 pi0.5 路径已验证正确

数据集的 action 旋转三维是 **Euler XYZ extrinsic**，不是 rotvec/轴角
（尽管列名叫 `left_ee_rx/ry/rz`）。仓库自采的 `collect_lerobot_v3.py/
v4.py` 数据集才是真 rotvec，两者不能混用同一个解码器 ——
`act_eval_usb.py` / `act_eval_gear.py` 用的 rotvec 是**正确**的，
不要跟着改。

在 pi0.5 实际训练用的那份数据（200 episodes / 121454 帧）上的实测
（`$PI05_ROOT/logs/diagnose_pi05_rotation_representation_20260802_201700.log`，
两种解码各自与同帧 state 四元数的测地距离）：

```
Euler XYZ 解码:   p50=0.605°   p90= 3.40°   mean= 3.98°
rotvec   解码:   p50=1.038°   p90=60.93°   mean=16.12°
```

另一个独立证据：`raw_action_rotation_norm_rad` 的 p50 恰好是 π，且 83%
的帧模长 > π（最大 7.98）。若这三维是旋转向量，模长即旋转角，必须在
[0, π] 内 —— 作为轴角表示不成立。

**结论：已训练的 pi0.5 checkpoint 不受影响。** 这个 bug 只存在于部署时
的解码环节，训练阶段模型只是回归数据里的原始数值，与约定无关。本分支
`task/policies/pi05_lerobot.py` 用 `from_euler("xyz")` 解码，与数据一致，
不需要重训。

分支分工（有意为之，不要合并）：

| 文件 | main（跑 DP） | pi05-remote-inference（跑 pi0.5） |
|---|---|---|
| `task/policies/pi05_lerobot.py` | ✓ 已修（a6c437f） | ✓ `from_euler("xyz")` |
| `task/policies/diffusion_stateonly.py` | ✓ 已修 | 仍 `from_rotvec`（本分支不跑 DP） |
| `task/policies/gt_replay.py` | ✓ 已修 | 仍 `from_rotvec` |

两个分支的 pi0.5 解码现在一致。DP 侧只在 main 上跑，本分支那两个文件
未同步不影响任何实际运行的路径。合并 main 还会触发 1ed97bb
「Delete unnecessary documents」对本分支若干文件的删除
（`sanity_check.py`、`export_val_episodes.py`、`precheck_right_arm.py`
等），没有必要。

`task/policies/diffusion_lerobot.py`（LeRobot 版 DP adapter，与
state-only 那个不同）在两个分支上都仍是 `from_rotvec` + 警告注释，
c400fc9 也没有动它。如果之后要用它，需要先确认它的 checkpoint 训练在
哪个数据集上。

副作用（对两个 baseline 对称，不影响对比）：Euler 角在 ±π 处回绕不
连续，上述日志里 1.5% 的相邻动作步跳变 > 45°，会略微增加回归难度。

## 本机遗留物（可清理）

- conda 环境 `py311`（约 18 GB，Isaac Sim 5.1 Windows 版）—— 驱动问题没解决
  之前没用；`conda env remove -n py311` 可删。
- WSL2 发行版 `Ubuntu-24.04` —— Isaac Sim 用不了，但如果以后想在本地跑
  pi0.5 推理（LeRobot 在 Linux 是官方测过的路径），它还能用。
  `wsl --unregister Ubuntu-24.04` 可删。
- Windows 长路径支持已启用（`LongPathsEnabled = 1`），无副作用，建议保留。
