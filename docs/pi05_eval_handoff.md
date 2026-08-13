# pi0.5 Isaac Sim 验证 — 部署与运行交接

目标：在 Isaac Sim 里跑 pi0.5 checkpoint 的闭环 rollout，产出每个零件的
pass/fail 和 rollout 视频，用于和 Diffusion Policy baseline 对比。

分支：`pi05-remote-inference`（pi0.5 跑这个分支；DP baseline 跑 `main`）。

**目标机器：Windows + RTX 3060（Ampere）那台旧电脑。** 为什么不是新的
RTX 5090 笔记本，见第 8 节——那台机器上仿真结果和别人对不上，已放弃。

---

## 0. 三条会浪费你一整天的禁令

1. **不要用 WSL 装 Isaac Sim。** 那台机器有 WSL，很容易顺手就用了。WSL2 通过
   `/dev/dxg` → D3D12 访问 GPU，**只映射 CUDA，不映射 Vulkan/OpenGL**，
   `/usr/lib/wsl/lib` 里一个 NVIDIA 的 Vulkan ICD 都没有。Mesa 的 dozen(dzn)
   能把 Vulkan 转译到 D3D12 但不支持光追，而 Omniverse RTX 渲染器必须要。
   也不能靠 headless 绕开——pi0.5 是视觉策略，必须要相机图像，相机渲染走的
   就是 RTX 渲染器。**要装就装 Windows 原生。**
2. **不要升级显卡驱动。** 那台机器以前跑通过，说明它现在的驱动是好的。
   Isaac Sim 5.1 对驱动版本极其挑剔（见第 8 节），升上去大概率崩。
   **先记下当前版本号再动任何东西。**
3. **不要改 `scene_init.usd` 或零件的物理属性。** 那是官方资产，队友能跑通，
   改了就没法和别人对比了。

---

## 1. 架构

```
Isaac Sim（旧电脑，Windows）── ssh 常驻连接 / 长度前缀 pickle ──> iam-strange 的 pi05_server.py
  task/run_pick_place.py                                            PI05Policy.from_pretrained(ckpt)
  └ task/policies/pi05_lerobot.py                                   └ GPU 推理，返回 14-D action
```

**pi0.5 推理跑在服务器上，本机只跑仿真。** 本机不需要装 LeRobot、不需要下
checkpoint（单个 8.8 GB）、不需要 HF 认证。Isaac 侧起一个 `ssh` 子进程，
把观测用 pickle 写进它的 stdin、从 stdout 读回 action。

观测：44-D state + 三路 240×320 RGB（head / left_hand / right_hand）。
动作：14-D 绝对末端位姿 = 左 xyz + 左 Euler XYZ + 左夹爪 + 右同构，
经 Lula IK 反解成关节目标。runner 只执行左臂那 7 维，右臂保持初始位姿。

---

## 2. 部署步骤（Windows 原生）

### 2.1 先记录现状，别急着装

```powershell
nvidia-smi --query-gpu=name,driver_version --format=csv
```

**把这个版本号记下来。** 如果之后出问题，要能回到这个状态。

### 2.2 开启长路径支持（管理员 PowerShell，需要重启）

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

不开这个，`isaacsim` 的 `extscache` 解压到超过 260 字符的路径时会失败，
报 `OSError: [Errno 2] No such file or directory`，pip 自己会提示是长路径问题。

### 2.3 Python 环境

Isaac Sim 5.1 要 **Python 3.11**。用 conda 或 venv 都行：

```powershell
conda create -n py311 python=3.11 -y
conda activate py311
pip install "isaacsim[all,extscache]==5.1.0.0"
```

约 18 GB，20~40 分钟。装完验证：

```powershell
python -m pip list | findstr isaacsim   # 应有 isaacsim-extscache-kit / kit-sdk / physics
python -c "import isaacsim, numpy; print(numpy.__version__)"   # 1.26.x
```

**注意不能用 `uv sync`**——仓库的 `uv.lock` 只解析了 linux x86_64，Windows 上用不了。

### 2.4 仓库

```powershell
git clone git@github.com:Benny-Zhu-398/RoCo_Assembly.git
cd RoCo_Assembly
git checkout pi05-remote-inference
git lfs pull
```

确认 `scene_init.usd` 是真实文件（约 39 MB）而不是 LFS 指针。

### 2.5 SSH 到推理服务器

```
# %USERPROFILE%\.ssh\config
Host iam-strange
    HostName 128.2.178.27
    User yudongluo
    IdentityFile C:/Users/<你>/.ssh/id_ed25519
```

验证（必须免密、无交互）：

```powershell
ssh -o BatchMode=yes iam-strange "hostname; nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"
```

Windows 上 `PI05_SSH_EXE` 用默认值即可（`C:\Windows\System32\OpenSSH\ssh.exe`）。
**Linux 上才需要覆盖成 `/usr/bin/ssh`。**

---

## 3. 三步验证（每步失败都不要往下走）

### 3.1 Isaac Sim 能起来

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"
python -c "from isaacsim import SimulationApp; app=SimulationApp({'headless':True}); print('app up'); app.close(); print('OK')"
```

首次启动要编译 shader，慢（1~3 分钟）。日志里应出现
`Graphics API: Vulkan` 和你的 GPU 那一行 `Active | Yes: 0`。

如果崩在 `rtx.scenedb.plugin.dll`（access violation），是驱动版本问题，见第 8 节。

### 3.2 远端 pi0.5 推理（不需要 Isaac Sim，可并行排查）

```powershell
$env:PI05_REMOTE="1"
$env:PI05_REMOTE_HOST="iam-strange"
$env:PI05_REMOTE_CHECKPOINT="/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model"
$env:PI05_REMOTE_CUDA_VISIBLE_DEVICES="2"
python scripts\test_pi05_remote_windows.py --timeout 900
```

期望输出：

```
remote pi0.5 smoke test passed (checkpoint=...)
action shape: (14,), all finite: true
```

第一次推理约 700 ms（生成 50 步 action chunk）。checkpoint 从外接盘加载到显存
要等一会儿，`--timeout 900` 是留给它的。

### 3.3 ⚠️ scripted baseline 必须对上参照结果

**这是最重要的一步。在 baseline 对不上之前，任何 pi0.5 的成绩都没有意义。**

```powershell
$env:ISAACSIM_HEADLESS="1"
python task\run_pick_place.py --results-json artifacts\baseline_check.json
```

拿结果和 `docs/baseline_reference_6of9.json` 对照——那是队友 Linux 机器上跑出的
已知正确结果（Isaac Sim 5.1.0.0，同一份代码和场景）：

| part | 参照结果 | 说明 |
|---|---|---|
| gear_20teeth | 184.57mm **FAIL** | 官方 baseline 本来就过不了 |
| gear_60teeth | 1.84mm **PASS** | |
| rod_16mm | snap=False **FAIL** | 官方 baseline 本来就过不了 |
| bolt_8mm | snap=False **FAIL** | 官方 baseline 本来就过不了 |
| usb_a | snap=True **PASS** | |
| hdmi | snap=True **PASS** | |
| pin | snap=True **PASS** | |
| battery_size1 | 5.62mm **PASS** | |
| battery_size5 | 7.20mm **PASS** | |

**合计 pass=6 fail=3，566 个 task step，56.81 秒仿真时间。**

- 你的结果接近 6/9 → 环境正确，继续
- 你的结果是 0/9 或差很远 → **停下**，环境和参照不一致，往下跑没有意义。
  对照第 8 节的调查记录，别重复已经排除过的方向。

注意三个 FAIL 是官方 baseline 的固有短板，不是环境问题。

---

## 4. 正式 rollout

### 一个零件一次，不要一次跑一组

训练数据的每条 episode 都是**从初始场景开始、只做一个零件**
（采集元数据：`effective_part_order: ["gear_20teeth"]`、`effective_max_parts: 1`）。
如果一次 rollout 连做两个零件，第二个零件开始时的场景状态在训练数据里从未
出现过，策略会在分布外被评测，成绩被低估。

```powershell
# ---- Group B: usb_a ----
$env:PI05_REMOTE="1"
$env:PI05_REMOTE_HOST="iam-strange"
$env:PI05_REMOTE_CHECKPOINT="/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupB_20260807_003521/checkpoints/007082/pretrained_model"
$env:PI05_REMOTE_CUDA_VISIBLE_DEVICES="2"
$env:PI05_EXEC_HORIZON="16"
$env:ROCO_PART_ORDER="usb_a"
$env:ISAACSIM_HEADLESS="1"
python task\run_pick_place.py `
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy `
  --record-video artifacts\pi05_groupB_usb_a.mp4 --record-video-camera head `
  --results-json artifacts\pi05_groupB_usb_a.json

# ---- Group B: hdmi ----  （同上，只改这两行）
$env:ROCO_PART_ORDER="hdmi"
# --record-video artifacts\pi05_groupB_hdmi.mp4 --results-json artifacts\pi05_groupB_hdmi.json

# ---- Group C: rod_16mm / bolt_8mm / pin ----  换 checkpoint
$env:PI05_REMOTE_CHECKPOINT="/media/iam-lab/strange_external/yudongluo/pi05/outputs/roco_pi05_groupC_20260807_011246/checkpoints/009951/pretrained_model"
$env:ROCO_PART_ORDER="pin"     # 然后 rod_16mm、bolt_8mm
```

### 建议先跑 group B

`usb_a` 和 `hdmi` 在参照结果里都是 snap 成功的，pi0.5 的成绩干净可解释。

**group C 要谨慎**：它的三个零件里 `rod_16mm` 和 `bolt_8mm` **连官方 scripted
baseline 都完成不了**（参照结果里也是 FAIL）。在这两个上评测 pi0.5，策略失败
和任务本身不可完成分不开。group C 重点看 `pin`。

### 关键环境变量

| 变量 | 作用 |
|---|---|
| `PI05_REMOTE=1` | 走 SSH 远端 sidecar；不设则是本地模式，需要 `PI05_CKPT` + `PI05_SERVER_PY` |
| `PI05_SSH_EXE` | Windows 用默认值；**Linux 上必须设成 `/usr/bin/ssh`** |
| `PI05_EXEC_HORIZON` | 一次取多少步 action 再重新查询。默认 1 |
| `PI05_SAFETY_FILTER` | 单步限幅。**对比实验保持关闭**（默认就是关） |
| `ROCO_PART_ORDER` | 逗号分隔，限制本次迭代的零件；不设则跑完整 9 件 |
| `PI05_TASK` | 语言指令，默认 `assemble parts onto the task board` |
| `PI05_SERVER_LOG` | sidecar 的 stderr，默认 `task/pi05_server.log` |
| `PI05_CLIENT_LOG` | 客户端 action 缓存日志，默认 `artifacts/pi05_action_cache.log` |

### 耗时

`PER_PART_TIMEOUT_STEPS = 3000`。`PI05_EXEC_HORIZON=16` 时每 16 步一次推理，
每零件最多约 3000/16 × 0.7s ≈ **2 分钟**推理开销，仿真本身另计。
horizon 设成 1 的话是每零件约 35 分钟。

### 服务器礼仪

`iam-strange` 是共享机器，跑之前先看 `nvidia-smi`，别抢别人的卡，
**绝不要 kill 别人的进程**。推理约占 15.5 GB。已知其他用户：`sumo`、`xinyiy`。

---

## 5. 评测口径（已确认）

对比对象是 `main` 分支上 `training/diffusion_policy/` 的 state-only Diffusion Policy。

1. **重规划节奏对齐到 16 步**：`PI05_EXEC_HORIZON=16`，DP 用默认
   （`DP_N_ACTION_STEPS` 默认 = horizon = 16）。两边原始默认值不匹配
   （pi0.5 默认 1，DP 默认 16），重规划频率会成为混淆变量，所以固定住它。
2. **不开 safety filter**：它会把输出限幅到单步 5mm/5°，相当于加了个外部
   控制器，对比不公平。（`_guard_left_joint_action` 那个 IK 关节跳变保护是
   常开的，防的是数值爆炸，不属于策略层干预。）
3. **只跑该 checkpoint 训练过的零件**，用 `ROCO_PART_ORDER`。
4. **一个零件一次 rollout**，见第 4 节。
5. **只用完整跑完的 group**：A 组半途 ENOSPC 挂了，只有半程的 4000，暂不使用。
6. **按 per-part 成功率报数**：DP 是每零件一个模型，pi0.5 是每组一个模型，
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

### ⚠️ Isaac 版本必须两边一致

pi0.5 和 DP 必须在**同一个 Isaac Sim 版本**上评测，否则对比被仿真器版本混淆。
和 haozh 对齐后再出最终数字。

---

## 6. checkpoint 清单

统一配方（`$PI05_ROOT/scripts/train_group.sh`）：base `lerobot/pi05_base`，
batch 16，5 epochs，bfloat16，gradient checkpointing，全量微调
（`train_expert_only=false`、`freeze_vision_encoder=false`），
`decay_steps = steps`，`warmup = min(1000, steps*0.1)`，`save_freq = steps`。

`PI05_ROOT = /media/iam-lab/strange_external/yudongluo/pi05`

| 组 | 零件 | train frames | steps | 状态 | 路径（`$PI05_ROOT/outputs/` 下） |
|---|---|---|---|---|---|
| A | gear_20teeth, gear_60teeth, battery_size1, battery_size5 | 54838 | 17137 | ✗ 8000 步 ENOSPC | `roco_pi05_groupA_20260806_225051/checkpoints/004000`（半程，LR 未退完） |
| B | usb_a, hdmi | 22661 | 7082 | ✓ 完成 | `roco_pi05_groupB_20260807_003521/checkpoints/007082` |
| C | rod_16mm, bolt_8mm, pin | 31844 | 9951 | ✓ 完成 | `roco_pi05_groupC_20260807_011246/checkpoints/009951` |

离线 VAL 分数（`VAL.left_trans_m.all.mean`，越小越好）：
A@4000 = 0.00926、B@6000 = 0.01081、B@7082 = 0.01116。

B 已 plateau（6000 优于终点 7082）。**仍建议用 7082**，保持"5 epoch 走完完整
LR 退火"的口径统一；挑 6000 属于 cherry-pick。

A 组要补的话，`$PI05_ROOT/scripts/RESUME.md` 里有完整的重跑说明。

---

## 7. 已修的三个 bug（都在本分支，已推送）

| commit | 内容 |
|---|---|
| `dd13444` | 把服务器上实际在跑的 `pi05_server.py` 收进 git（见第 9 节第 1 条） |
| `a6c437f`（main）/ 本分支 | pi0.5 动作旋转按 **Euler XYZ extrinsic** 解码，不是 rotvec |
| `1f7752a` | 夹爪维去掉 `GRIPPER_OPEN_LIMIT` 缩放，数据本来就是原始关节弧度 |

**这三个都不影响已训练的 checkpoint**，纯部署侧解码问题，不需要重训。

### 旋转约定的证据

数据集 action 的旋转三维是 Euler XYZ extrinsic，不是 rotvec，尽管列名叫
`left_ee_rx/ry/rz`。在训练用的那份数据（200 episodes / 121454 帧）上实测
（`$PI05_ROOT/logs/diagnose_pi05_rotation_representation_20260802_201700.log`）：

```
Euler XYZ 解码:   p50=0.605°   p90= 3.40°   mean= 3.98°
rotvec   解码:   p50=1.038°   p90=60.93°   mean=16.12°
```

独立佐证：`raw_action_rotation_norm_rad` 的 p50 恰好是 π，83% 的帧模长 > π
（最大 7.98）。旋转向量的模长即旋转角，不可能超过 π —— 排除轴角。

反例：自采的 `collect_lerobot_v3.py/v4.py` 数据集（metadata 写着
`absolute_cartesian_target_xyz_rotvec_gripper`）是**真 rotvec**，
`act_eval_usb.py` / `act_eval_gear.py` 对它们的解码是正确的，不要跟着改。

### 夹爪单位的证据

来自 checkpoint 自带的归一化统计量（B 和 C 一致）：

```
action[6] (左夹爪)  q01=0.0903  q50=0.0993  q90=0.2148  q99=0.2670  max=0.3008
state[42] (左夹爪)  q01=0.0898  q50=0.0996  q90=0.2149  q99=0.2517
```

state 和 action 的分位数逐点重合，量程约 0.06~0.30 弧度——是原始关节弧度，
不是 [0,1] 比例。若是比例，量程该铺满 0~1。

---

## 8. RTX 5090 那台机器为什么放弃（别重复这些调查）

新笔记本（Windows 11 + RTX 5090 Laptop，Blackwell sm_120，
Core Ultra 9 275HX）上，scripted baseline 稳定 **0/9**，和参照的 6/9 对不上。
排查记录：

| 假设 | 结论 |
|---|---|
| 分支改动引入回归 | ❌ `main` 和本分支结果**逐位相同** |
| Isaac Sim 版本 | ❌ 5.1 和 6.0 都 0/9 |
| 显卡驱动 | ❌ 换驱动修好了崩溃，但成绩没变 |
| Blackwell GPU 物理 | ❌ 物理**根本没用 GPU**：运行时 `is_gpu_dynamics_enabled=False`、`use_gpu_pipeline=False`、`world.device=cpu` |
| 缺几何/碰撞体资产 | ❌ 0 个缺失 reference/payload（只缺 3 张贴图 + 2 个内置 MDL，纯外观） |
| 场景文件被改写 | ❌ git 干净 |
| 线程/确定性设置没生效 | ❌ 已验证 `numThreads=1`、`enableEnhancedDeterminism=True` 都生效；同进程内两次相同预热**逐位一致** |

**剩下的唯一变量是 CPU。** PhysX 的 CPU 求解器按指令集选 SIMD 代码路径，
浮点舍入随之不同。而这个场景处于临界稳定状态——零件手工摆放、9 个零件
**全都没有 authored mass/density**（`/World/table` 的 mass 甚至是 0）、初始位置
很可能有轻微穿模。零控制下的实测漂移：

```
gear_60teeth  60 步预热内自己漂了 390.06 mm
bolt_8mm      105.30 mm
battery_size1  26.62 mm
```

容差是 10 mm。`gear_60teeth` 在 0.3 秒内飞出 390mm 是穿模被大力弹开的特征。
在这种场景里浮点末位差异足以放大成"过 / 不过"。

**这不是配置能修的**，那台机器的 Isaac Sim 一切正常，是官方场景对 CPU 太敏感。

### 驱动版本（如果旧机器也撞到崩溃）

Isaac Sim 5.1 对驱动挑剔。已知情况：

| 来源 | 版本 |
|---|---|
| 官方文档（Windows） | 580.88 |
| 官方文档（Linux） | 580.65.06 |
| 维护者实测 | "591.86 是最后一个能用的，>591.x 都崩" |
| 实测有效（RTX 5090 Laptop） | **591.86**（CUDA 13.1） |

崩溃特征：启动到 `app ready` 之后立刻 access violation，栈顶是
`rtx.scenedb.plugin.dll` / `librtx.scenedb.plugin.so`。

Isaac Sim **6.0** 的驱动窗口宽得多（实测 610.74 可用，维护者也推荐升 6.0 解决
驱动兼容），而且本 harness 用到的 11 个模块路径在 6.0 里全部存在、API 零改动
（`isaacsim.sensors.camera` 废弃但未移除）。但 6.0 改变物理行为，换版本前
必须和 haozh 对齐，见第 5 节。

### WSL2 为什么不行

见第 0 节第 1 条。已在新机器上实测确认：`/usr/lib/wsl/lib` 没有任何 Vulkan
ICD，`vulkaninfo` 只能看到 Mesa 的软件/转译驱动。这是
[microsoft/wslg#1254](https://github.com/microsoft/wslg/issues/1254) /
[#1312](https://github.com/microsoft/wslg/issues/1312) 长期未解的架构问题。

---

## 9. 未决事项

1. **服务器上的 `pi05_server.py` 是游离状态。**
   `/home/yudongluo/user/Roco/RoCo_Assembly` 这个 checkout 停在
   `server-pi05-formal-backup` 分支，而 `task/pi05_server.py` 是直接在文件系统上
   改的、**从未 commit 到任何分支**——远端模式实际执行的就是这份游离代码。
   本仓库的 `dd13444` 已经把它的内容收进 git，但**服务器那个 checkout 还没对齐**。
   建议把服务器切到本分支或至少 commit 掉，否则下次改动又会悄悄失同步。
   （这次是靠日志里 `cmd=replan` 和仓库代码打的 `cmd=observation` 对不上才发现的。）

2. **官方场景的 CPU 敏感性值得反馈给 Haichao Liu**（官方）。同一份代码和场景，
   在不同 CPU 上给出不同的确定性结果，意味着 benchmark 成绩不可跨机器复现。
   证据见第 8 节。根因指向零件缺少 authored mass/density。

3. **A 组 checkpoint 未完成**，见第 6 节。

4. **DP baseline 的 checkpoint 不在本仓库**。训练日志里的路径是
   `C:\Users\haozh\repos\RoCo_Assembly\training\diffusion_policy\outputs\`，
   要向 haozh 索取。而且 DP 只训了 7 个零件，缺 `gear_20teeth` 和 `battery_size5`。

5. **右臂 7 维是噪声，下次训练应该去掉。**
   runner 只执行左臂，右臂全程固定，所以 action 的右半 7 维是常量（只有浮点
   噪声）。但 LeRobot 的分位数归一化 `2*(x-q01)/(q99-q01)-1` 的 epsilon 保护只在
   `denom == 0` **精确为零**时才触发，而右臂的 denom 是 1e-6 量级的非零值，
   保护不生效。结果这 7 维被除以约 1e-6，噪声被放大到填满整个 [-1,1] 核心区间，
   尺度与左臂真实信号相同，外加冲到 ±120 的离群尾部。**14 维回归目标里有 7 维
   是纯噪声。**

   | 维度 | denom = q99−q01 | 归一化后 min ~ max |
   |---|---|---|
   | 左臂 7 维 | 0.177 ~ 0.736 | 主体 [−1,1] |
   | Rx | 0.00000073 | −8.2 ~ +35.9 |
   | Ry | 0.00066382 | −121.4 ~ +1.4 |
   | Rrx | 0.00130153 | −121.7 ~ +1.4 |
   | Rgrip | 0.00000098 | −7.2 ~ +27.0 |

   实际损害**未经 ablation 验证**：噪声梯度零均值，后果更接近"收敛变慢 +
   浪费容量"，而非左臂预测被系统性带偏。**已有 checkpoint 不必因此重训**——
   离线指标 `VAL.left_trans_m` 只看左臂，部署侧也只执行左臂 7 维。

   **要改就三组一起改。** A 组反正要重跑，很容易顺手只在它上面用 7 维 action
   空间——那样 A 和 B/C 不是同一配方，**跨组不可比**，对比表会塌。
   重训成本：约 11~14 小时/组、三组 35~40 小时；外接盘当时只剩 59 GB，
   A 组第一次就是被 ENOSPC 打死的（`$PI05_ROOT/scripts/RESUME.md` 有记录）。

---

## 10. sidecar 协议（排查用）

长度前缀 pickle：4 字节大端长度 + pickle payload，双向。

| 请求 | 含义 |
|---|---|
| `{"cmd": "reset"}` | 清空 policy 的 action 队列，每个零件开始时发一次 |
| `{"state","head","left","right","task","exec_horizon"}` | 完整观测，**强制重规划**，返回 1 个或 `exec_horizon` 个 action |
| `{"cmd": "next_action"}` | 不带观测，从队列里弹一个（队列空则报错） |

服务器 stderr 每行带 `queue_before` / `queue_after` / `select_action_ms` 等，
排查时先看 `PI05_SERVER_LOG`。

**不要用 `scripts/eval_pi05_roco.sh`** —— 那是给**本地** sidecar 用的封装
（会设 `PI05_SERVER_PY` 指向本机 LeRobot venv），远端模式请直接调
`task/run_pick_place.py`。
