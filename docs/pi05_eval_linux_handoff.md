# pi0.5 Isaac Sim 验证 — 全新 Linux 环境部署与运行

面向对象：一台刚装好 Linux、什么都没有的机器。目标是跑通 pi0.5 checkpoint
在 Isaac Sim 里的闭环 rollout，产出每个零件的 pass/fail 和 rollout 视频，
用于和 Diffusion Policy baseline 对比。

分支：`pi05-remote-inference`（pi0.5 跑这个分支；DP baseline 跑 `main`）。

---

## 1. 架构

```
Isaac Sim（本机）── ssh 常驻连接 / 长度前缀 pickle ──> iam-strange 的 pi05_server.py
  task/run_pick_place.py                                 PI05Policy.from_pretrained(ckpt)
  └ task/policies/pi05_lerobot.py                        └ GPU 推理，返回 14-D action
```

**pi0.5 推理跑在服务器上，本机只跑仿真。** 本机不需要装 LeRobot、不需要下
checkpoint（单个 8.8 GB）、不需要 HF 认证。Isaac 侧通过 `subprocess` 起一个
`ssh` 进程，把观测用 pickle 写进它的 stdin、从 stdout 读回 action。

观测：44-D state + 三路 240×320 RGB（head / left_hand / right_hand）。
动作：14-D = 左 xyz + 左 Euler XYZ + 左夹爪 + 右同样。runner 目前只执行左臂
那 7 维，右臂保持初始位姿。

---

## 2. 前置条件

### ⚠️ 驱动版本（最重要的一条）

**必须装 NVIDIA 驱动 580.65.06，不要装 latest。**

Isaac Sim 5.1.0（Kit 107.3.3）在 Blackwell（sm_120，RTX 50 系）配 595 / 610
分支驱动时，会在 `app ready` 之后立刻崩在 RTX 渲染器里：

```
000: librtx.scenedb.plugin.so   (Windows 上是 rtx.scenedb.plugin.dll)
006: carb.scenerenderer-rtx
009: omni.hydra.rtx
```

这个坑在 Windows 上已经实测撞到过（RTX 5090 Laptop + 驱动 610.88，两次运行
完全复现），换 Linux 不会自动消失——**它是驱动版本问题，不是操作系统问题。**

官方验证过的版本：Linux `580.65.06`，Windows `580.88`。

同类报告：
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252
- https://github.com/isaac-sim/IsaacSim/issues/651 （RTX 5080 + 驱动 610）
- https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335

### 其它

- GPU：需要带 RT Core，≥ 16 GB 显存。
- 磁盘：Isaac Sim 依赖约 18 GB。若 home 分区小，把 `UV_CACHE_DIR` 指到和仓库
  **同一个文件系统**的大盘上（uv 靠硬链接省空间，跨文件系统会复制两份）。
- 到 `iam-strange` 的免密 SSH（见第 3 节）。

---

## 3. 部署步骤

### 3.1 SSH 到推理服务器

```bash
# ~/.ssh/config
Host iam-strange
    HostName 128.2.178.27
    User yudongluo
    IdentityFile ~/.ssh/id_ed25519
```

验证（必须免密、无交互）：

```bash
ssh -o BatchMode=yes iam-strange 'hostname; nvidia-smi --query-gpu=index,memory.free --format=csv,noheader'
```

### 3.2 拉仓库

```bash
git clone git@github.com:Benny-Zhu-398/RoCo_Assembly.git
cd RoCo_Assembly
git checkout pi05-remote-inference
git lfs pull          # *.usd / *.usdc / *.obj / *.mp4 走 LFS
```

确认 `scene_base.usd`（约 36 MB）和 `scene_init.usd`（约 39 MB）是真实文件而
不是 LFS 指针（`head -c 40 scene_init.usd` 不应出现 `version https://git-lfs`）。

### 3.3 建环境

```bash
uv sync                    # 从 uv.lock 复现，约 18 GB
```

仓库根目录的 `pyproject.toml` / `.python-version` / `uv.lock` 就是为
linux x86_64 锁的，**uv 会自己下载并管理 CPython 3.11，不需要 conda**。
（反过来说：`uv.lock` 只解析了 linux，Windows 上用不了这条路。）

首次 `import isaacsim` 会打印 EULA 并等待确认，非交互场景先设：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

---

## 4. 分三步验证（每步失败都不要往下走）

### 4.1 Isaac Sim 能起来

```bash
uv run python -c "import isaacsim; print('import ok')"
```

再起一次真正的 app（这一步才会暴露驱动问题，首次启动要编译 shader，慢）：

```bash
uv run python - <<'PY'
import os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("app up")
app.close()
print("OK")
PY
```

日志里应出现 `Graphics API: Vulkan` 和你的 GPU 那一行 `Active | Yes: 0`。
如果崩在 `librtx.scenedb.plugin.so`——回到第 2 节，是驱动版本。

### 4.2 远端 pi0.5 推理链路（不需要 Isaac Sim）

这一步独立验证 SSH sidecar，可以和 4.1 并行排查。

```bash
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_SSH_EXE=/usr/bin/ssh \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
python scripts/test_pi05_remote_windows.py --timeout 900
```

脚本名字里的 `_windows` 只是历史遗留，代码本身是跨平台的（`_display_command`
按 `os.name` 分支）。**Linux 上必须显式设 `PI05_SSH_EXE=/usr/bin/ssh`**，
因为 `_REMOTE_DEFAULTS` 里的默认值是 Windows 的
`C:\Windows\System32\OpenSSH\ssh.exe`。

期望输出：

```
remote pi0.5 smoke test passed (checkpoint=...)
action shape: (14,), all finite: true
```

第一次推理约 700 ms（生成 50 步 action chunk）。checkpoint 从外接盘加载到
显存要等一会儿，`--timeout 900` 是留给它的。

### 4.3 scripted baseline 跑通仿真

```bash
uv run python task/run_pick_place.py --max-parts 1 --results-json /tmp/base.json
```

确认仿真 + 相机 + IK + 评分都正常。顺便另开一个终端 `nvidia-smi` 量一下
Isaac Sim 实际占多少显存——如果以后想把推理也挪到本地，这个数决定够不够。

---

## 5. 正式 rollout

每个 checkpoint 只见过一个零件组，`Pi05LeRobotPolicy` 一次运行只加载一个
checkpoint，所以**分两次跑**。

```bash
# ---- Group B: usb_a, hdmi ----
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_SSH_EXE=/usr/bin/ssh \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupB_20260807_003521/checkpoints/007082/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
PI05_EXEC_HORIZON=16 \
ROCO_PART_ORDER=usb_a,hdmi \
ISAACSIM_HEADLESS=1 \
./scripts/run_roco.sh \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video artifacts/pi05_groupB_head.mp4 \
  --record-video-camera head \
  --results-json artifacts/pi05_groupB_results.json

# ---- Group C: rod_16mm, bolt_8mm, pin ----
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_SSH_EXE=/usr/bin/ssh \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
PI05_EXEC_HORIZON=16 \
ROCO_PART_ORDER=rod_16mm,bolt_8mm,pin \
ISAACSIM_HEADLESS=1 \
./scripts/run_roco.sh \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video artifacts/pi05_groupC_head.mp4 \
  --record-video-camera head \
  --results-json artifacts/pi05_groupC_results.json
```

**不要用 `scripts/eval_pi05_roco.sh`** —— 那个封装是给**本地** sidecar 用的
（它会设 `PI05_SERVER_PY` 指向本机的 LeRobot venv），走远端模式请直接用
`run_roco.sh`，它只负责设 `UV_CACHE_DIR` / `OMNI_KIT_ACCEPT_EULA` /
`ISAACSIM_HEADLESS` / `VK_ICD_FILENAMES` 然后 `uv run python task/run_pick_place.py`。

### 关键环境变量

| 变量 | 作用 |
|---|---|
| `PI05_REMOTE=1` | 走 SSH 远端 sidecar；不设则是本地模式，需要 `PI05_CKPT` + `PI05_SERVER_PY` |
| `PI05_SSH_EXE` | **Linux 上必须设成 `/usr/bin/ssh`** |
| `PI05_EXEC_HORIZON` | 一次取多少步 action 再重新查询。默认 1 |
| `PI05_SAFETY_FILTER` | 单步限幅。**对比实验保持关闭**（默认就是关） |
| `ROCO_PART_ORDER` | 逗号分隔，限制本次迭代的零件；不设则跑完整 9 件 |
| `PI05_TASK` | 语言指令，默认 `assemble parts onto the task board` |
| `PI05_SERVER_LOG` | sidecar 的 stderr 落盘位置，默认 `task/pi05_server.log` |
| `PI05_CLIENT_LOG` | 客户端 action 缓存日志，默认 `artifacts/pi05_action_cache.log` |
| `ISAACSIM_HEADLESS` / `ISAACSIM_ACTIVE_GPU` / `ISAACSIM_PHYSICS_GPU` | Isaac 侧 |

### 耗时估算

`PER_PART_TIMEOUT_STEPS = 3000`（`task/param_config.py`）。`PI05_EXEC_HORIZON=16`
时每 16 步一次推理，即每零件最多 3000/16 × 0.7 s ≈ **2 分钟**推理开销，仿真
本身另计。horizon 设成 1 的话是每零件约 35 分钟。

### 服务器礼仪

`iam-strange` 是共享机器，跑之前先看 `nvidia-smi`，别抢别人的卡，
**绝不要 kill 别人的进程**。推理约占 15.5 GB。已知的其他用户：`sumo`、`xinyiy`。

---

## 6. 评测口径（与 DP baseline 对比的约定）

对比对象是 `main` 分支上 `training/diffusion_policy/` 的 state-only
Diffusion Policy。

1. **重规划节奏对齐到 16 步**：`PI05_EXEC_HORIZON=16`，DP 用默认
   （`DP_N_ACTION_STEPS` 默认 = horizon = 16）。
   两边的原始默认值不匹配（pi0.5 默认 1，DP 默认 16），重规划频率会成为
   混淆变量，所以固定住它。
2. **不开 safety filter**：它会把输出限幅到单步 5 mm / 5°，相当于加了个外部
   控制器，对比不公平。（`_guard_left_joint_action` 那个 IK 关节跳变保护是
   常开的，防的是数值爆炸，不属于策略层干预。）
3. **只跑该 checkpoint 训练过的零件**：用 `ROCO_PART_ORDER`。
4. **只用完整跑完的 group**：A 组半途挂了，暂不使用（见第 7 节）。
5. **按 per-part 成功率报数**：DP 是每零件一个模型，pi0.5 是每组一个模型，
   粒度不同，只有按零件报才对得上。

### 两边模型本身的差异（写进论文说明，不是要"修"的）

| | DP baseline | pi0.5 |
|---|---|---|
| 粒度 | per-part，9 个模型 | per-group，3 个模型 |
| 观测 | state-only，**无视觉** | state + 三路 RGB |
| batch / epochs | 64 / 200 | 16 / 5 |
| action chunk | horizon 16 | chunk_size 50 |
| 数据划分 | val_fraction 0.1, split_seed 0 | **同上，已对齐** |

epochs 差 40 倍是模型体量决定的（DP 是小的 state-only 网络，pi0.5 是 3B 全量
微调，5 epoch 已经 11~14 小时），不是可调的自变量。

### 注意 group C 的语义

`rod_16mm` 和 `bolt_8mm` 在 `param_config.py` 里属于 `PERMANENT_PARTS`
（预装在板上、harness 只做原地刷新，不是从零抓取），它们的"成功"含义和
`usb_a` / `hdmi` / `pin` 不一样，报数据时要单独说明。

---

## 7. checkpoint 清单

统一训练配方（`$PI05_ROOT/scripts/train_group.sh`）：base `lerobot/pi05_base`，
batch 16，5 epochs，bfloat16，gradient checkpointing，全量微调
（`train_expert_only=false`、`freeze_vision_encoder=false`），
`decay_steps = steps`（LR 在本次 run 内走完退火），
`warmup = min(1000, steps*0.1)`，`save_freq = steps`。

`PI05_ROOT = /media/iam-lab/strange_external/yudongluo/pi05`

| 组 | 零件 | train frames | steps | 状态 | 路径（`$PI05_ROOT/outputs/` 下） |
|---|---|---|---|---|---|
| A | gear_20teeth, gear_60teeth, battery_size1, battery_size5 | 54838 | 17137 | ✗ 8000 步 ENOSPC | `roco_pi05_groupA_20260806_225051/checkpoints/004000`（半程，LR 未退完） |
| B | usb_a, hdmi | 22661 | 7082 | ✓ 完成 | `roco_pi05_groupB_20260807_003521/checkpoints/007082` |
| C | rod_16mm, bolt_8mm, pin | 31844 | 9951 | ✓ 完成 | `roco_pi05_groupC_20260807_011246/checkpoints/009951` |

离线 VAL 分数（`VAL.left_trans_m.all.mean`，越小越好）：
A@4000 = 0.00926、B@6000 = 0.01081、B@7082 = 0.01116、C@9951 见
`$PI05_ROOT/logs/eval/C_step9951.json`。

B 已 plateau（6000 优于终点 7082）。**仍建议用 7082**，保持"5 epoch 走完
完整 LR 退火"的口径统一；挑 6000 属于 cherry-pick。

A 组要补的话，`$PI05_ROOT/scripts/RESUME.md` 里有完整的重跑说明和当初
磁盘打爆的原因分析。

---

## 8. 参考

### 8.1 旋转约定（已验证，不用再查）

数据集 action 的旋转三维是 **Euler XYZ extrinsic**，不是 rotvec/轴角，
尽管列名叫 `left_ee_rx/ry/rz`。

在 pi0.5 实际训练用的那份数据（200 episodes / 121454 帧）上实测
（`$PI05_ROOT/logs/diagnose_pi05_rotation_representation_20260802_201700.log`，
两种解码各自与同帧 state 四元数的测地距离）：

```
Euler XYZ 解码:   p50=0.605°   p90= 3.40°   mean= 3.98°
rotvec   解码:   p50=1.038°   p90=60.93°   mean=16.12°
```

独立佐证：`raw_action_rotation_norm_rad` 的 p50 恰好是 π，83% 的帧模长 > π
（最大 7.98）。旋转向量的模长即旋转角，不可能超过 π —— 排除轴角。

小幅旋转下两种解码看着差不多（所以错误解码的 median 有欺骗性），大幅重定向
时差几十度，足以让 Lula IK 拒绝目标位姿、机械臂冻结在上一条 good command 上、
耗光 per-part 超时。

`pi05_lerobot.py` 在 `main` 和本分支上都已用 `from_euler("xyz")`。
**checkpoint 不受影响** —— 这个约定只在部署解码时起作用，训练阶段模型只是
回归原始数值，与约定无关。

反例：仓库自采的 `collect_lerobot_v3.py/v4.py` 数据集（metadata 写着
`absolute_cartesian_target_xyz_rotvec_gripper`）是**真 rotvec**，
`act_eval_usb.py` / `act_eval_gear.py` 对它们的解码是正确的，不要跟着改。
`diffusion_lerobot.py` 两个分支上都仍是 rotvec + 警告注释，要用它之前先确认
它的 checkpoint 训练在哪个数据集上。

### 8.2 为什么不是 Windows / WSL2

- **WSL2 不可能**：WSL 通过 `/dev/dxg` → D3D12 → Windows 驱动访问 GPU，只映射
  了 CUDA，**没有映射 Vulkan/OpenGL**。`/usr/lib/wsl/lib` 里没有任何 Vulkan
  ICD。Mesa 的 dozen(dzn) 能把 Vulkan 转译到 D3D12，但不支持光追扩展，而
  Omniverse RTX 渲染器必须要。也不能靠 headless 绕开——pi0.5 是视觉策略，
  必须要相机图像，相机渲染走的就是 RTX 渲染器。
- **Windows 原生可行但撞驱动**：`isaacsim` 有 `cp311-win_amd64` wheel，
  装得上（需要先开长路径支持 `LongPathsEnabled=1`，否则 `extscache` 解压超
  260 字符会失败），但随后崩在第 2 节说的驱动问题上。

### 8.3 sidecar 协议

长度前缀 pickle：4 字节大端长度 + pickle payload，双向。

| 请求 | 含义 |
|---|---|
| `{"cmd": "reset"}` | 清空 policy 的 action 队列，每个零件开始时发一次 |
| `{"state","head","left","right","task","exec_horizon"}` | 完整观测，**强制重规划**，返回 1 个或 `exec_horizon` 个 action |
| `{"cmd": "next_action"}` | 不带观测，从队列里弹一个（队列空则报错） |

服务器 stderr 的每行日志带 `queue_before` / `queue_after` / `select_action_ms`
等，排查时先看 `PI05_SERVER_LOG`。

---

## 8.4 下次训练：把右臂从 action 空间里去掉

runner 只执行左臂 7 维，右臂全程固定，所以数据集里 action 的右半 7 维是常量
（只有浮点噪声）。但 LeRobot 的分位数归一化 `2*(x-q01)/(q99-q01) - 1` 的
epsilon 保护只在 `denom == 0` **精确为零**时才触发
（`lerobot/processor/normalize_processor.py`），而右臂的 denom 是 1e-6 量级的
非零值，保护不生效。结果是这 7 维被除以约 1e-6：

| 维度 | denom = q99−q01 | 归一化后 min ~ max |
|---|---|---|
| 左臂 7 维 | 0.177 ~ 0.736 | 主体 [−1,1]，尾部 −13.5 ~ +21.2 |
| Rx | 0.00000073 | −8.2 ~ +35.9 |
| Ry | 0.00066382 | −121.4 ~ +1.4 |
| Rz | 0.00113142 | −1.2 ~ +48.5 |
| Rrx | 0.00130153 | −121.7 ~ +1.4 |
| Rry | 0.00000184 | −57.6 ~ +18.5 |
| Rrz | 0.00002426 | −44.3 ~ +9.7 |
| Rgrip | 0.00000098 | −7.2 ~ +27.0 |

q01→−1、q99→+1 是归一化的定义，所以右臂那 7 维的**整个 [−1,1] 核心区间被
浮点噪声填满**，尺度与左臂真实信号相同，外加冲到 ±120 的离群尾部。14 维回归
目标里有 7 维是纯噪声。

实际损害程度**未经 ablation 验证**，不要当成定论：噪声梯度零均值，模型对这些
维度只能输出均值并吃一个固定的损失底噪，后果更接近"收敛变慢 + 浪费容量 +
共享表征被扰动"，而非左臂预测被系统性带偏；±120 的离群帧有梯度裁剪兜底。

**建议下次训练把 action 砍成左臂 7 维**，或至少在 loss 里 mask 掉右半。
state 的右臂维度优先级低一些（是输入不是回归目标，常量输入模型可以学会忽略），
但同样被噪声放大。

已有 checkpoint 不必因此重训——现有离线指标 `VAL.left_trans_m.all.mean` 只看
左臂平移，未被这些维度污染；部署侧 runner 也只执行左臂 7 维。

## 9. 未决事项

1. **服务器上的 `pi05_server.py` 是游离状态。**
   `/home/yudongluo/user/Roco/RoCo_Assembly` 这个 checkout 停在
   `server-pi05-formal-backup` 分支，而 `task/pi05_server.py` 是直接在文件
   系统上改的、从未 commit 到任何分支——远端模式实际执行的就是这份游离代码。
   本仓库的 `dd13444` 已经把它的内容收进 git，但**服务器那个 checkout 还没有
   对齐**。建议把服务器切到本分支或至少 commit 掉，否则下次改动又会悄悄失同步。
   （这次是靠日志里 `cmd=replan` 和仓库代码打的 `cmd=observation` 对不上才发现的。）

2. **A 组 checkpoint 未完成**，见第 7 节。重跑前先读下面第 5 条。

   ⚠️ **不要只把右臂修复用在 A 组的重跑上。** A 组无论如何都要重来，很容易
   顺手在它上面改用 8.4 节说的 7 维 action 空间——但那样 A 就和 B/C 不是同一个
   配方了，**跨组不可比**，对比表里 A 那一行没法和 B/C 并列。要改就三组一起改。

3. **DP baseline 的 checkpoint 不在本仓库**。训练日志里的路径是
   `C:\Users\haozh\repos\RoCo_Assembly\training\diffusion_policy\outputs\`，
   要向 haozh 索取。而且 DP 只训了 7 个零件，缺 `gear_20teeth` 和
   `battery_size5`。

4. **本机显存能否同时跑 Isaac Sim + pi0.5** 尚未测量。pi0.5 推理约 15.5 GB，
   24 GB 卡上留给 Isaac Sim 约 6.7 GB。第 4.3 步跑 baseline 时顺便量一下，
   够的话可以把推理也搬到本地（`PI05_REMOTE` 不设，改用 `PI05_CKPT` +
   `PI05_SERVER_PY`，代码已支持，不需要改）。本地跑的好处是不受服务器上
   其他用户抢卡影响。

5. **要不要为右臂噪声重训——决策路径。**

   先分清哪些问题需要重训、哪些不需要：

   | 问题 | 性质 | 需要重训？ |
   |---|---|---|
   | 旋转按 rotvec 解码 | 部署侧 | ✗ 已改 adapter（a6c437f / 本分支） |
   | 夹爪乘除 `GRIPPER_OPEN_LIMIT` | 部署侧 | ✗ 已改 adapter（1f7752a） |
   | 右臂 7 维噪声进了 loss | **训练侧** | 见下 |

   只有第三条是训练侧的，但它的实际损害**没有做过 ablation**（见 8.4 节）。

   **建议顺序：先跑 rollout，再决定。** 在拿到 group B / C 的 per-part 结果
   之前重训属于盲目优化——现有 checkpoint 若在仿真里够用，30~40 小时就白花了；
   若表现很差，也未必是这个原因。判据：

   - 左臂精度可接受 → 右臂噪声写成 limitation，不重训。
   - 左臂精度明显不行 → 右臂 mask 是第一个该试的改动，但**三组一起重训**。

   重训前必须确认的成本：约 11~14 小时/组、三组 35~40 小时；外接盘当时只剩
   59 GB，而 A 组第一次就是被 ENOSPC 打死的（`$PI05_ROOT/scripts/RESUME.md`
   记录了原因：三个 run 并发 + 中途 checkpoint 撑爆了约 94 GB 分区，两个并发
   约 66 GB 才是安全的）。清空间和并发数要先规划好。

   **不要出现"A 组用新配方、B/C 用旧配方"的混合状态。**
