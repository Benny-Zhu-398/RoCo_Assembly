# pi0.5 Isaac Sim 验证 — Linux 部署与运行

在 Isaac Sim 里跑 pi0.5 checkpoint 的闭环 rollout，产出每个零件的 pass/fail
和视频，用于和 Diffusion Policy baseline 对比。

分支：`pi05-remote-inference`（pi0.5 跑这个分支；DP baseline 跑 `main`）。

**必须在 Linux 上跑。** Windows 上仿真结果和参照对不上（两台不同 CPU 的
Windows 机器都是同样的错误结果，Linux 正确），见第 7 节。

---

## 1. 架构

```
Isaac Sim（本机）── ssh 常驻连接 / 长度前缀 pickle ──> iam-strange 的 pi05_server.py
  task/run_pick_place.py                                 PI05Policy.from_pretrained(ckpt)
  └ task/policies/pi05_lerobot.py                        └ GPU 推理，返回 14-D action
```

**pi0.5 推理在服务器上，本机只跑仿真。** 本机不装 LeRobot、不下 checkpoint
（单个 8.8 GB）、不需要 HF 认证。Isaac 侧起一个 `ssh` 子进程，把观测用 pickle
写进 stdin、从 stdout 读回 action。

观测：44-D state + 三路 240×320 RGB。
动作：14-D 绝对末端位姿（左 xyz + 左 Euler XYZ + 左夹爪 + 右同构），经 Lula IK
反解成关节目标。runner 只执行左臂 7 维，右臂固定。

---

## 2. 前置条件

### ⚠️ 驱动必须钉 580.65.06，不要装 latest

Isaac Sim 5.1 对驱动极挑剔：595 / 610 分支会在 `librtx.scenedb.plugin.so` 里
崩溃（启动到 `app ready` 之后立刻 access violation）。官方 Linux 验证版本是
**580.65.06**。

```bash
sudo apt install nvidia-driver-580
sudo apt-mark hold nvidia-driver-580 nvidia-dkms-580 libnvidia-gl-580   # 防止自动升级
nvidia-smi   # 确认版本
```

其它：GPU 需带 RT Core、≥16 GB 显存；磁盘约 20 GB。
**不要用 WSL** —— WSL 不映射 Vulkan，Omniverse RTX 渲染器起不来（已实测确认）。

### SSH 到推理服务器

```
# ~/.ssh/config
Host iam-strange
    HostName 128.2.178.27
    User yudongluo
    IdentityFile ~/.ssh/id_ed25519
```

```bash
ssh -o BatchMode=yes iam-strange 'hostname'    # 必须免密、无交互
```

---

## 3. 安装

```bash
git clone git@github.com:Benny-Zhu-398/RoCo_Assembly.git
cd RoCo_Assembly
git checkout pi05-remote-inference
git lfs pull                      # scene_init.usd 应是约 39 MB 的真实文件
uv sync                           # 约 18 GB，uv 自己管 Python 3.11
export OMNI_KIT_ACCEPT_EULA=YES
```

`uv.lock` 就是为 linux x86_64 锁的，直接可用。若 home 分区小，把 `UV_CACHE_DIR`
指到和仓库**同一文件系统**的大盘上（uv 靠硬链接省空间）。

---

## 4. 三步验证

### 4.1 Isaac Sim 能起来

```bash
uv run python -c "from isaacsim import SimulationApp; a=SimulationApp({'headless':True}); print('up'); a.close()"
```

首次启动编译 shader，1~3 分钟。日志应有 `Graphics API: Vulkan`。
崩在 `librtx.scenedb.plugin.so` → 回第 2 节查驱动。

### 4.2 远端推理（不需要 Isaac Sim，可并行）

```bash
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_SSH_EXE=/usr/bin/ssh \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
python scripts/test_pi05_remote_windows.py --timeout 900
```

期望 `action shape: (14,), all finite: true`。首次推理约 700 ms。

脚本名里的 `_windows` 是历史遗留，代码跨平台。**Linux 上必须设
`PI05_SSH_EXE=/usr/bin/ssh`**（默认值是 Windows 路径）。

### 4.3 ⚠️ baseline 必须先对上参照结果

**这是硬性关卡。baseline 对不上之前，pi0.5 的任何成绩都无法解释。**

```bash
ISAACSIM_HEADLESS=1 uv run python task/run_pick_place.py --results-json artifacts/baseline_check.json
```

对照 [`docs/baseline_reference_6of9.json`](baseline_reference_6of9.json)——队友
Linux 机器上的已知正确结果（同版本、同代码、同场景）：

| part | 参照 |
|---|---|
| gear_20teeth | 184.57mm **FAIL** ← 官方 baseline 固有短板 |
| gear_60teeth | 1.84mm PASS |
| rod_16mm | snap=False **FAIL** ← 官方 baseline 固有短板 |
| bolt_8mm | snap=False **FAIL** ← 官方 baseline 固有短板 |
| usb_a | snap=True PASS |
| hdmi | snap=True PASS |
| pin | snap=True PASS |
| battery_size1 | 5.62mm PASS |
| battery_size5 | 7.20mm PASS |

**pass=6 fail=3，566 步，56.81 秒仿真时间。**

接近 6/9 → 环境正确，继续。差很远 → 停下，别往下跑。

---

## 5. 正式 rollout

**一个零件一次**。训练数据每条 episode 都是"从初始场景开始只做一个零件"，
一次跑多个的话后面的零件处于分布外状态，成绩被低估。

```bash
# Group B: usb_a（换 ROCO_PART_ORDER=hdmi 跑第二个）
PI05_REMOTE=1 \
PI05_REMOTE_HOST=iam-strange \
PI05_SSH_EXE=/usr/bin/ssh \
PI05_REMOTE_CHECKPOINT=/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupB_20260807_003521/checkpoints/007082/pretrained_model \
PI05_REMOTE_CUDA_VISIBLE_DEVICES=2 \
PI05_EXEC_HORIZON=16 \
ROCO_PART_ORDER=usb_a \
ISAACSIM_HEADLESS=1 \
uv run python task/run_pick_place.py \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video artifacts/pi05_groupB_usb_a.mp4 --record-video-camera head \
  --results-json artifacts/pi05_groupB_usb_a.json

# Group C: 换 checkpoint 成 roco_pi05_groupC_20260807_011246/checkpoints/009951，
#          ROCO_PART_ORDER 依次取 pin / rod_16mm / bolt_8mm
```

**先跑 group B**（`usb_a`、`hdmi` 在参照里都成功，结果干净）。
**group C 谨慎**：三个零件里 `rod_16mm`、`bolt_8mm` 连官方 baseline 都完成不了，
策略失败和任务不可完成分不开；重点看 `pin`。

耗时：`PER_PART_TIMEOUT_STEPS=3000`，horizon 16 时每零件最多约 2 分钟推理开销。

### 环境变量

| 变量 | 作用 |
|---|---|
| `PI05_REMOTE=1` | 走 SSH 远端 sidecar |
| `PI05_SSH_EXE` | Linux 上必须 `/usr/bin/ssh` |
| `PI05_EXEC_HORIZON` | 一次取多少步 action 再重新查询，默认 1 |
| `PI05_SAFETY_FILTER` | 单步限幅，**对比实验保持关闭**（默认关） |
| `ROCO_PART_ORDER` | 逗号分隔，限制本次迭代的零件 |
| `PI05_SERVER_LOG` | sidecar stderr，默认 `task/pi05_server.log` |

**不要用 `scripts/eval_pi05_roco.sh`** —— 那是本地 sidecar 模式的封装，
远端模式直接调 `task/run_pick_place.py`。

### 服务器礼仪

`iam-strange` 是共享机器。跑前看 `nvidia-smi`，别抢卡，**绝不 kill 别人的进程**。
推理占约 15.5 GB。已知其他用户：`sumo`、`xinyiy`。

---

## 6. 评测口径（已确认）

1. **重规划节奏对齐 16 步**：`PI05_EXEC_HORIZON=16`，DP 用默认（也是 16）。
   两边原始默认值不匹配（pi0.5 是 1），重规划频率会成为混淆变量。
2. **不开 safety filter**：它会把输出限幅到单步 5mm/5°，等于加外部控制器。
   （`_guard_left_joint_action` 那个 IK 跳变保护常开，防数值爆炸，不算策略干预。）
3. **只跑该 checkpoint 训练过的零件**，用 `ROCO_PART_ORDER`。
4. **一个零件一次 rollout**。
5. **只用完整跑完的 group**（A 组半途 ENOSPC 挂了，暂不用）。
6. **按 per-part 成功率报数**（DP 每零件一个模型，pi0.5 每组一个）。
7. **pi0.5 和 DP 必须在同一个 Isaac Sim 版本上评测**，和 haozh 对齐后再出终稿。

模型本身的差异（写进说明，不是要修的）：DP 是 state-only 无视觉、per-part、
batch 64 跑 200 epoch；pi0.5 是 3B 全量微调、per-group、batch 16 跑 5 epoch。
数据划分两边一致（`val_fraction 0.1, split_seed 0`）。

---

## 7. checkpoint 清单

统一配方：base `lerobot/pi05_base`，batch 16，5 epochs，bfloat16，
gradient checkpointing，全量微调，`decay_steps = steps`。

`PI05_ROOT = /media/iam-lab/strange_external/yudongluo/pi05`，路径在 `$PI05_ROOT/outputs/` 下：

| 组 | 零件 | steps | 状态 | checkpoint |
|---|---|---|---|---|
| A | gear_20teeth, gear_60teeth, battery_size1, battery_size5 | 17137 | ✗ 8000 步 ENOSPC | `roco_pi05_groupA_20260806_225051/checkpoints/004000`（半程） |
| B | usb_a, hdmi | 7082 | ✓ | `roco_pi05_groupB_20260807_003521/checkpoints/007082` |
| C | rod_16mm, bolt_8mm, pin | 9951 | ✓ | `roco_pi05_groupC_20260807_011246/checkpoints/009951` |

离线 VAL（`left_trans_m.all.mean`，越小越好）：A@4000=0.00926、
B@6000=0.01081、B@7082=0.01116。B 已 plateau，**仍用 7082** 以保持"5 epoch
走完 LR 退火"口径统一，挑 6000 属于 cherry-pick。

A 组重跑说明见 `$PI05_ROOT/scripts/RESUME.md`。

---

## 8. 为什么不能用 Windows

同一份代码、场景、Isaac Sim 5.1.0.0（都是 pip 装），Windows 上 scripted baseline
稳定 **0/9**，Linux 上 6/9。两台**不同 CPU** 的 Windows 机器给出基本相同的错误
结果，所以不是 CPU。

已排除：分支改动（main 与本分支逐位相同）、Isaac 版本（5.1 和 6.0 都 0/9）、
驱动、GPU 物理（场景 `enableGPUDynamics=False`，运行时 `world.device=cpu`）、
缺失资产（0 个未解析 reference）、线程/确定性设置（`numThreads=1` 与
`enableEnhancedDeterminism=True` 均已验证生效，同进程两次预热逐位一致）。

症状是零控制下零件就自己漂：`gear_60teeth` 在 60 步预热内漂 390mm，
`bolt_8mm` 105mm（容差 10mm）。两个齿轮用 `sdf` 碰撞近似、其余用
`convexDecomposition`，而漂得最厉害的正是齿轮——指向 **Windows/Linux 的 PhysX
碰撞烘焙（尤其 SDF）数值路径不同**。零件本身没有 authored mass/density
（`/World/table` 的 mass 是 0），初始状态处于临界稳定，放大了平台差异。

---

## 9. 已修的 bug（本分支，已推送）

| commit | 内容 |
|---|---|
| `dd13444` | 把服务器上实际在跑的 `pi05_server.py` 收进 git |
| `a6c437f`（main）/ 本分支 | 动作旋转按 **Euler XYZ extrinsic** 解码，不是 rotvec |
| `1f7752a` | 夹爪维去掉 `GRIPPER_OPEN_LIMIT` 缩放（数据本就是原始关节弧度） |

**都不影响已训练的 checkpoint**，纯部署侧解码问题，不需要重训。

证据：旋转——在训练数据 121454 帧上，Euler 解码与同帧 state 四元数的测地距离
p50=0.605°/p90=3.40°，rotvec 解码 p50=1.038°/p90=60.93°；且旋转三元组模长
中位数恰为 π、83% 超过 π，作为轴角不成立。
夹爪——checkpoint 自带的归一化统计里 state 与 action 的夹爪分位数逐点重合，
量程 0.06~0.30 弧度，是关节弧度而非 [0,1] 比例。

（自采的 v3/v4 数据集是**真 rotvec**，`act_eval_usb.py` / `act_eval_gear.py`
的解码正确，不要跟着改。）

---

## 10. 未决事项

1. **服务器上的 `pi05_server.py` 是游离状态**：那个 checkout 停在
   `server-pi05-formal-backup` 分支，文件是直接改的、从未 commit。`dd13444`
   已把内容收进本仓库，但**服务器那份还没对齐**，建议切分支或 commit 掉。
2. **Windows 平台差异值得反馈给官方（Haichao Liu）**：同一份代码和场景在
   Windows/Linux 上结果不同，benchmark 不可跨平台复现。证据见第 8 节。
3. **A 组 checkpoint 未完成**，见第 7 节。
4. **DP checkpoint 不在仓库**，在 haozh 的 `C:\Users\haozh\repos\...`，且只训了
   7 个零件（缺 gear_20teeth、battery_size5）。
5. **下次训练把右臂移出 action 空间**：右臂全程固定，action 右半 7 维是常量
   噪声，但 LeRobot 的分位数归一化只在 `q99-q01` **精确为 0** 时才用 epsilon
   保护，而右臂 denom 是 1e-6 量级，噪声被放大到填满 [-1,1]（尾部到 ±120），
   14 维目标里 7 维是纯噪声。实际损害未做 ablation，**已有 checkpoint 不必
   重训**（离线指标只看左臂，部署也只执行左臂）。**要改就三组一起改**，
   否则跨组不可比。
