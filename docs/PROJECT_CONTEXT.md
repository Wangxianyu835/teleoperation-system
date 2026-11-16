# 项目交接文档 / AI 对话上下文

> **用途**：新开一个 AI 对话时，先让它读本文件，它就能立刻了解项目全貌、环境陷阱和历史决策。
> **文件位置**：`F:\simulation_platform\docs\PROJECT_CONTEXT.md`
> **最后更新**：2026-11-10（新增 **§20「两周冲刺计划：论文复现收口（11-10 → 11-23）」** —— 逐日排期 + 里程碑 gate + 现场采集 SOP + 降级方案；§8.1 追加第 38 条：把 `main`（`1a1984b`）上丢了 110 KB 交接日志恢复到本分支，并把今天三机器人 VR 的腕限位修复写进 §20.1.1）| 上一条 2026-10-15（§8.1 追加第 37 条：现场「GR1-T2 的手臂怎么拐就怎么拐」逐层取证定位到**方向口径搬的是操作者的绝对方向**（gr1_t2 加载姿势下前臂朝后下方，同一句「指向前下方」对它等于「先拆开 97 度」），口径落成**每台的属性** `ArmProfile.posture_frame` + 旗子 `--arm-posture-frame {auto,mapped,delta}`；同一份 200 帧上 gr1_t2 左臂肘任务残差中位 196.4 -> 28.6 mm、h1_2 / g1 六个读数逐位不变）| 上一条 2026-10-14（§0.1 新增第 6 条「改动落点：优先在既有文件里增量修改」+ §3.2(6) / §7.6 新增「30 个环境一次跑通 + 批量截图」实测 + §8.1 追加本轮记录）

---

## 0. 给 AI 的第一段话（复制这段给新对话）

```
请先读这三个文件，再开始工作：
  F:\simulation_platform\docs\CONVENTIONS.md      （★ 工程约定 / 硬约束，动手前必读，含一键自检）
  F:\simulation_platform\docs\PROJECT_CONTEXT.md   （项目背景、技术栈、决策历史）
  F:\simulation_platform\docs\ENVIRONMENT_SETUP.md （环境配置、工具链、踩过的坑）

关键前提：项目用 E:\python3.11.7\python.exe 运行（不是项目里的 .venv，那是空壳）；
.py 文件禁止出现非 GBK 字符（提交前跑 python scripts/check_gbk_safe.py）；
**任何东西都不许写进 C 盘**（依赖/缓存/临时文件，必须先跑 scripts/env_e_drive_cache.ps1）；
git 走 SSH（不需要代理），看 GitHub 网页要开代理。
**不要改队友已入库的算法代码**（retargeting/ 下的训练/推理/数据/模型/config/inspect/dual_teleop/
simulation/__main__/visionpro/mediapipe）；要兼容就新增自己的文件（见 §0.1 第 3、5 条）。
**开工前先跑一次自检**：powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1
```

---

## 0.1 ★ 硬约束（违反即返工，AI 助手必须逐条遵守）

> **本节是摘要**；完整规则（含每条的理由、命令、检查方式、翻车记录）在
> [`CONVENTIONS.md`](CONVENTIONS.md)（工程约定，唯一来源），AI 自动加载版在仓库根目录 `.clinerules`。
> **一键自检**：`powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1`（`-Full` 加跑测试与流水线）。

| # | 约束 | 为什么 / 怎么守 |
|---|---|---|
| 1 | **禁止往 C 盘写任何东西**<br>（依赖、pip/HF/matplotlib/torch 缓存、临时文件）| 用户明确要求（`C:` 只剩 18.7 GB）。默认缓存目录**全在 C 盘**，必须靠环境变量改写：<br>`PIP_CACHE_DIR` / `MPLCONFIGDIR` / `HF_HOME` / `TORCH_HOME` / `XDG_CACHE_HOME` / `TEMP` / `TMP` -> `E:\cache\*`<br>**每条新终端先跑** `. .\scripts\env_e_drive_cache.ps1`；自检 `python scripts/check_no_c_drive.py --strict`（详见 ENVIRONMENT_SETUP §2.6）|
| 2 | **跑项目只用 `E:\python3.11.7\python.exe`** | 项目内 `.venv` 是**空壳**，用它必然 `No module named 'torch'`；曾经因此误判"环境缺 roboticstoolbox"（ENVIRONMENT_SETUP §2.2）|
| 3 | **不改队友已入库的算法代码** | `retargeting/` 下 `arm.py`、`tracking.py`、`inference.py`、`training.py`、`data.py`、`model.py`、`config.py`、`inspect.py`、`dual_teleop.py`、`simulation.py`、`__main__.py`、`visionpro/`、`mediapipe/` **本轮零改动**。需要兼容就在**自己的新增文件**里做转发层 / conftest 钩子 / 包装脚本，并把问题清单交给队友（RETARGETING_PIPELINE §9.1）|
| 4 | **`.py` 里禁止非 GBK 字符** | 中文控制台 GBK 会直接 `UnicodeEncodeError` **崩溃**；提交前 `python scripts/check_gbk_safe.py --strict`（ENVIRONMENT_SETUP §6）|
| 5 | **改前先跑一遍基线，改后再跑同一套** | 自检清单见 RETARGETING_PIPELINE §10；结论必须附**退出码 + 不变量**，不要凭"看起来对"就下结论。<br>血泪教训：本轮曾因空壳 `.venv` 报错就断言"缺依赖会连累整仓"，改了队友 `arm.py`，事后实测**完全不必要**、全部回退 |
| 6 | （★★ 规范）**改动落点：优先在既有文件里增量修改**，不要新建 `xxx_fix.py` / `xxx_v2.py` 这类平行文件；确实要新建时，必须在**同一次提交**里登记进文档索引 | **用户 2026-10-14 明确要求**："不然你写的有时候总会这一块那一块，后面如果要改的话，会遗漏很多地方，容易出问题"。同一功能的逻辑散在多个文件里 = 本项目返工主因（对照第 3 条翻车记录）。<br>新建的三个例外：① 约定 3 禁止改的队友文件（在自己这侧写转发层 / 钩子）；② 现有文件里确实没有可挂靠的位置；③ 用户要求单独成文件。测试优先加到既有测试文件（详见 CONVENTIONS 约定 14）|

---

## 1. 项目概况

| 项目 | 内容 |
|---|---|
| **中文名** | 大创 —— 遥操作系统 |
| **性质** | 中山大学 · 大学生创新创业训练计划项目 |
| **核心目标** | **用 PyBullet 复现 TeleOpBench 论文**（双臂灵巧手遥操作仿真基准平台）|
| **本地路径** | `F:\simulation_platform` |
| **GitHub 仓库** | `https://github.com/Wangxianyu835/teleoperation-system` |
| **GitHub 用户名** | `Wangxianyu835` |
| **提交邮箱** | `328205274+wangxianyu835@users.noreply.github.com`（GitHub 隐私邮箱）|
| **作者** | 王宪雨（中山大学）|

### 项目规模

- **源码 2,392 行 / 23 个 Python 文件**（整个仓库仅 ~115 KB）
- **30 个分层任务**（Level 1~4）
- **3 种人形机器人模型**

---

## 2. 复现目标论文

| 项目 | 内容 |
|---|---|
| **标题** | TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation |
| **arXiv** | 2505.12748v2 [cs.RO]，2025-09-15 |
| **机构** | 上海AI实验室、浙大、港中文、港科大(广州)、港大、Feeling AI、清华深研院 |
| **作者** | Hangyu Li\*, Qin Zhao\*（共同一作）；Junting Dong†, Jiangmiao Pang（通讯）|
| **项目主页** | https://gorgeous2002.github.io/TeleOpBench/ |
| **本地 PDF** | `F:\26年大创 双臂遥操作平台\2025TeleOpBench.pdf`（13 页）|
| **官方参考源码** | `F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\TeleOpBench-main`（1.4 GB）|
| **许可证** | Apache-2.0（Unitree Robotics / HangZhou YuShu TECHNOLOGY CO.,LTD.）|

### 论文核心内容（速览）

**一句话**：为「双臂灵巧遥操作」建立第一个公平、可复现的基准测试平台 —— 用**仿真器固定住机器人和任务环境**，消除硬件差异，让四种遥操作接口能在完全相同条件下被对比。

**三大贡献**：
1. **30 个高保真任务环境**（覆盖运动学与力交互难度广谱）
2. **4 种代表性遥操作管线**，统一模块化接口（MoCap / VR / 外骨骼 / 单目视觉）
3. **仿真 + 真机对照实验**，10 个任务上强相关 → 证明仿真能预测真机表现

**论文的四种遥操作接口**：

| # | 接口 | 关键硬件/算法 |
|---|---|---|
| ① | Vision-based | SMPL 体型估计 + SMPLer-X 姿态 + **MediaPipe + Dex-Retargeting**（手部）+ **PINK** IK + 卡尔曼滤波 |
| ② | MoCap-based | **Xsens MVN**（17 IMU）+ **Xsens Metagloves by Manus**（每手 20 DoF）+ 关节级重缩放 + **CLIK 闭链IK** |
| ③ | VR-based | **Apple Vision Pro**（OpenXR）+ **PINK** IK + AnyTeleop dex-retargeting |
| ④ | Exoskeleton-based | **HOMIE** 同构外骨骼 + 伺服关节 + 霍尔传感器手套（每手 15 DoF）+ **直接映射绕过 IK** |

**论文的 3 种机器人**：

| 机器人 | 厂商 | 手臂 | 灵巧手 | 手 DoF |
|---|---|---|---|---|
| H1-2 | Unitree | 7 DoF/臂 | Inspire 灵巧手 | 5指 6 DoF |
| GR1-T2 | Fourier | 7 DoF/臂 | Fourier 原生灵巧手 | 5指 6 DoF |
| G1 | Unitree | — | 轻量三指手 | 3指 4 DoF |

**论文的评估协议**：从 30 个任务中选 **10 个代表性任务**，指标 = **成功率(%) + 完成时间(s)**，**4 名被试**。

**论文实验结论（各方法性格）**：

| 方法 | 优势 | 短板 |
|---|---|---|
| Vision | 部署最简单、成本最低 | 帧率低、腕部朝向估计粗糙、遮挡 → 只能做简单任务；Task 4/7 差 |
| VR | 抓取强 | **Task 7（ball bimanual 双手传递）完全失败**（手挡手致位姿估计失效）|
| Exoskeleton | 控制平滑，多数任务好 | 侧向肘部运动受限 → Task 1 耗时增加 |
| Xsens | 最平滑最精确、耗时最少 | **最贵** |

**论文局限性**：只覆盖上半身+桌面场景；**无触觉反馈**；未来方向 = loco-manipulation + haptics。

---

## 3. 项目现状（代码结构 + 已验证功能）

### 3.1 目录结构

```
F:\simulation_platform\
├── main.py                  137行  主入口（交互选任务/指定任务/demo/benchmark/无头）
├── demo_teleop.py           119行  无控制器演示
├── show_all.py              137行  ⭐ 三机器人并排展示
├── show_hand.py              73行  ⭐ LinkerHand **L21** 灵巧手展示
├── test_camera.py            22行  摄像头测试
├── test_import.py            50行  模块导入自检
├── requirements.txt                 pybullet/numpy/h5py/scipy
│
├── envs\                           仿真环境
│   ├── simulation_env.py    176行  环境核心类（集成机器人/任务/记录/随机化）
│   ├── robot_loader.py       46行  机器人模型加载
│   ├── sensor_recorder.py   123行  观测数据记录（状态/相机流/物体元数据）
│   └── domain_randomizer.py  54行  域随机化 ★论文没有，本项目额外加的
│
├── tasks\
│   ├── all_tasks.py         682行  ⭐ 30 个分层任务实现
│   ├── base_task.py         173行  任务基类
│   └── task_registry.py      32行  任务注册表
│
├── teleop\
│   ├── teleop_pipeline.py   244行  完整遥操作流水线
│   ├── vr_interface.py      140行  Vision Pro 手腕追踪 → 关节角度
│   ├── hand_interface.py    117行  灵巧手驱动（L21 的 17 个关节 + apply_to_robot）
│   ├── camera_interface.py   50行  摄像头采集（联想电脑 RGB）
│   └── pipeline_data.py     143行  流水线数据结构
│
├── utils\metrics.py          53行  成功率 / 完成时间统计
├── docs\PROJECT_CONTEXT.md         ⭐ 本交接文档
│
├── robots\from_teleopbench\        ★机器人模型（181MB，**已入库**，普通 Git）
├── lib\                            依赖目录（335MB，未入库）
├── linkerhand_sdk\                 灵巧手 SDK（1013MB，未入库，来自 gitee）
└── .venv\                          虚拟环境（未入库）
```

**源码总计：2,392 行 / 23 个 Python 文件**

### 3.2 ✅ 已验证可运行的功能

**（1）三种机器人全部能被 PyBullet 成功加载**（2026-09-12 实测，无界面 `p.DIRECT` 模式）

| 机器人 | URDF 路径 | 结果 | 总关节 | 可动关节 |
|---|---|---|---|---|
| **H1-2** | `robots/from_teleopbench/h1_2/h1_2.urdf` | ✅ OK | **55** | 51 |
| **GR1-T2** | `robots/from_teleopbench/gr1/urdf/robot.urdf` | ✅ OK | **70** | 54 |
| **G1** | `robots/from_teleopbench/g1/g1_29dof_with_hand_lock_waist.urdf` | ✅ OK | **53** | 41 |

> 55 关节与 `envs/robot_loader.py` 注释「H1-2 (55关节人形)」完全吻合。
> 加载时的 `b3Warning: No inertial data for link` 是无害警告（URDF 部分 link 未写惯性参数，PyBullet 用默认值兜底）。

**（2）`show_all.py` 三机器人并排展示**
- H1-2（红）x=-2.2 / GR1-T2（蓝）x=0 / G1（绿）x=+2.2
- 各配彩色地面圆盘 + 头顶方块 + 3D 文字标签
- 按正则 `(shoulder_pitch|upper_arm_pitch|arm_pitch)` 动态查找左右肩关节，做正弦摆动演示
- 240 Hz 仿真 + `useFixedBase=True`

**（3）传感器数据记录**（`envs/sensor_recorder.py`）对应论文观测三类：(a) 机器人状态向量 (b) 相机流（头戴第一人称 + 固定第三人称）(c) 任务级物体位姿元数据

**（4）评估指标**（`utils/metrics.py`）对应论文：成功率 + 完成时间

**（5）★ 三台机器人都能用同一份采集数据演示手部动作**（2026-09-30 实测，`p.DIRECT`）

数据 = `datasets/raw/my_recording_angles.h5`（809 帧，摄像头采集 + 队友重定向输出）。

| 机器人 | 可表达自由度 | 会动的映射关节 | 限位截断（左/右） | 稳定后跟踪误差 | 结论 |
|---|---|---|---|---|---|
| H1-2 | 12/17 | 24 | 4.4% / 6.5% | 0.0000 rad | PASS |
| GR1-T2 | 11/17 | 22 | **0.0% / 0.0%** | 0.0000 rad | PASS |
| G1 | 7/17 | 14 | 2.8% / 4.9% | 0.0000 rad | PASS |

> 「会动」= 整段回放中该关节实际转过 ≥ 0.05 rad。三台机器人**没有一根卡死的关节**。
> GR1-T2 的原装手行程最宽，L21 数据完全不需要截断，贴合度反而比 H1-2 更好。
> 工具：`scripts/check_native_hand_motion.py`（量化）+ `scripts/render_hand_snapshots.py`（离屏出图）。

**（6）★ 30 个任务环境能一次全部跑通并离屏出图**（2026-10-14 实测）

工具：`scripts/capture_task_envs.py`（一个命令把 Level 1~4 的 30 个环境逐个建起来、
跑 480 步物理、离屏截图，并输出实测报告）。

| 项 | 实测结果 |
|---|---|
| 建成并跑完的环境数 | **30 / 30**（`成功 30 / 30；失败 0`，退出码 0）|
| 机器人 | H1-2（`--robot h1_2`，动作维度 38）|
| 截图 | 30 张 + `00_overview_all30.png`（6x5 总览拼图），共 1.69 MiB |
| 截图内容自检 | 像素标准差最低 **47.74**、颜色数最低 **217**（不是空图/纯色图）|
| 产物目录 | `outputs/task_env_shots/`（含 `report.json` 逐环境数据，**不入库**）|

> ⚠️ 能跑通 ≠ 能做对：这一轮用的是 `main.py` 的演示正弦控制器（只动关节、不做任务），
> 所以 27/30 任务在 2 秒内未成功属预期；**暴露出的 3 类环境缺陷**（抽屉/门掉落、判据过松、
> 缺任务控制器）见 §7.6，那才是「搭建 30 个环境」还没做完的部分。

### 3.3 ⚠️ PyCharm 中有一个失效的运行配置

`.idea/workspace.xml` 里注册了 4 个运行入口，其中 **`show_robot.py` 在本仓库中不存在**（可能已删除或只是计划）：

```
demo_teleop.py   ✅ 存在
show_all.py      ✅ 存在
show_hand.py     ✅ 存在
show_robot.py    ❌ 文件不存在
WORKING_DIRECTORY = $PROJECT_DIR$
```

---

## 4. ★ 运行环境（有坑，必读）

### 4.1 依赖装在哪里（★ 不是标准 venv！）

本项目用 **`pip install --target`** 把依赖装进了**项目内的 `lib\` 目录**：

```
F:\simulation_platform\lib\
├── bin\
├── pybullet.cp311-win_amd64.pyd        ← pybullet 3.2.7 的本地扩展模块
├── pybullet_data\ pybullet_envs\ pybullet_robots\ pybullet_utils\
├── numpy\   numpy-2.4.6.dist-info\   numpy.libs\
├── scipy\   scipy-1.17.1.dist-info\  scipy.libs\
└── h5py\    h5py-3.16.0.dist-info\
```

**⚠️ 后果**：`lib\` **既不在 `sys.path` 里，也不在 venv 的 site-packages 里**。
直接运行 `python main.py` 会报：
```
ModuleNotFoundError: No module named 'pybullet'
```

### 4.2 ✅ 两种实测可用的运行方式

**方式 A：设置 `PYTHONPATH`（命令行推荐）**
```powershell
cd F:\simulation_platform
$env:PYTHONPATH = 'F:\simulation_platform\lib'
python main.py
```
实测通过：
```
pybullet: F:\simulation_platform\lib\pybullet.cp311-win_amd64.pyd
numpy: 2.4.6    scipy: 1.17.1    h5py: 3.16.0
```

**方式 B：直接用 `E:\python3.11.7` 解释器（全局已装齐）**
```powershell
& 'E:\python3.11.7\python.exe' main.py
```
实测通过：`ALL OK pybullet=pybullet.cp311-win_amd64.pyd numpy=2.4.6 scipy=1.17.1 h5py=3.16.0`

> PyCharm 项目 SDK 名 = `Python 3.11 (simulation_platform) (2)`，`WORKING_DIRECTORY = $PROJECT_DIR$`

### 4.3 本机 Python 环境清单

| 版本 | 路径 | 有 pybullet？ |
|---|---|---|
| 3.11（项目 venv）| `F:\simulation_platform\.venv\Scripts\python.exe` | ❌ 只有 pip/setuptools/pypdf |
| **3.11** | **`E:\python3.11.7\python.exe`** | ✅ **齐全** |
| 3.14 | `E:\python\python.exe` | ✅ pybullet（cp314）|
| 3.12 | `C:\Users\王宪雨\AppData\Local\Programs\Python\Python312\python.exe` | ❌ |

### 4.4 requirements.txt 与实际版本

```
pybullet>=3.2.0   → 实际 3.2.7
numpy>=1.24.0     → 实际 2.4.6
h5py>=3.10.0      → 实际 3.16.0
scipy>=1.10.0     → 实际 1.17.1
```

---

## 5. Git / GitHub 配置

| 项目 | 值 |
|---|---|
| Git 安装位置 | `E:\Git`（v2.55.0.windows.3）|
| `user.name` | `Wangxianyu835` |
| `user.email` | `328205274+wangxianyu835@users.noreply.github.com` |
| 其他全局配置 | `core.quotepath=false`、`core.autocrlf=true`、`init.defaultBranch=main` |
| **SSH 私钥** | **`E:\ssh\id_ed25519`** ⚠️ **唯一一份，务必备份到 U 盘/网盘！** |
| SSH 公钥 | `E:\ssh\id_ed25519.pub` |
| `known_hosts` | `E:\ssh\known_hosts` |
| **`core.sshCommand`** | `C:/Windows/System32/OpenSSH/ssh.exe -i E:/ssh/id_ed25519 -o UserKnownHostsFile=E:/ssh/known_hosts -o IdentitiesOnly=yes` |
| 远程 `origin` | `git@github.com:Wangxianyu835/teleoperation-system.git` |
| 浏览器 HTTPS 代理 | Clash Verge（**看网页**时需要开，**推代码不需要**）|

> **⚠️ 副作用**：因为密钥不在默认的 `~/.ssh` 位置，直接敲 `ssh -T git@github.com` 会失败。
> **只有 `git` 命令能正常使用**（因为配置了 `core.sshCommand`）。日常 clone/pull/push 不受影响。

### 5.1 仓库当前状态

```
main (05eddb7)
├── 9e0a94e  初始化仓库：双臂遥操作仿真平台源码
├── 071cd31  fix: show_hand.py 去除硬编码盘符路径，改为相对脚本定位
└── 05eddb7  Merge pull request #1: 修正 show_hand.py 打印的左手型号标签
```

### 5.2 `.gitignore` 已排除

```
__pycache__/  *.pyc  .venv/  venv/  .idea/  .vscode/
lib/              ← 335MB 本地依赖
linkerhand_sdk/   ← 1013MB 第三方 SDK（来自 gitee.com/ericbrunt/linkerhand_telop_python）
robots/*          ← 排除 robots 下全部子项…（共约 274MB，代码未引用）
!robots/from_teleopbench/  ← …但【放行】from_teleopbench（181MB，已入库，见第 13 节）
data/ outputs/ logs/ *.h5 *.pkl *.pt *.ckpt ...
```

### 5.3 提交历史（供参考）

| 提交 | 内容 | 大小 |
|---|---|---|
| `9e0a94e` | 首次提交：27 个文件（源码，排除了 1.8GB 依赖与资产）| 115.4 KB |
| `071cd31` | 修复 `show_hand.py` 硬编码盘符路径 | — |
| `1cad50e` | 修复打印文字 `l10v7` → `l7`（经 PR #1 合并）| 369 字节 |

---

## 6. ★ 环境陷阱（换机器 / 重装前必读）

这些都是**已经踩过并解决**的真实问题，记录下来避免重复踩坑。

### 6.1 中文用户名导致 Git 自带 ssh 失效

**现象**：
```
Could not create directory '/c/Users/\315\365\317\334\323\352/.ssh' (No such file or directory).
Failed to add the host to the list of known hosts (.../known_hosts).
git@github.com: Permission denied (publickey).
```
> `\315\365\317\334\323\352` 是「**王宪雨**」的 GBK 八进制编码（`CD F5`=王，`CF DC`=宪，`D3 EA`=雨）。

**原因**：Git 自带的 `E:\Git\usr\bin\ssh.exe` 是 **MSYS 版**，把中文用户名按 GBK 字节串去查目录，找不到真实的 `C:\Users\王宪雨\.ssh`。

**对照测试证据**：
| 使用的 ssh | 结果 |
|---|---|
| 默认（Git 自带 MSYS，`OpenSSH_10.3p1`）| ❌ `Host key verification failed` |
| 强制 Windows 自带（`OpenSSH_for_Windows_9.5p2`）| ✅ 成功 |

**解决**：设置 `core.sshCommand` 强制用 Windows 原生 ssh（见第 5 节）。

### 6.2 E 盘目录权限「不继承」的坑

**现象**：在 `E:\` 下新建的文件夹，**里面写不进文件**（访问被拒绝）。

**原因**：`E:\` 根目录给用户的 `(M)` 权限**没有 `(OI)(CI)` 继承标记**，所以新建子目录不继承写权限。

**解决**（你是目录所有者，**不需要管理员权限**）：
```powershell
icacls "E:\目标目录" /grant "*<用户SID>:(OI)(CI)M"
```
> 本机已修复：`E:\ssh`、`E:\projects`、`E:\cache`（2026-10-01，缓存目录用）
> **一劳永逸方案**（需**管理员**运行，因为 `E:\` 根目录所有者是 `NT AUTHORITY\SYSTEM`）：
> ```powershell
> icacls "E:\" /grant "*<用户SID>:(OI)(CI)M"
> ```

### 6.3 SSH 私钥必须收紧 ACL

**原因**：Windows OpenSSH 有安全检查 —— 私钥若对其他用户可读，会直接拒绝使用（报 `UNPROTECTED PRIVATE KEY FILE`）。`E:\` 下默认权限较宽松，必须收紧。

**解决**：
```powershell
icacls "E:\ssh\id_ed25519" /inheritance:r /grant:r "*<用户SID>:F"
```

### 6.4 ★ GitHub 网页被 SNI 层阻断，但 SSH 通

**现象**：浏览器打不开 `github.com`（提示「响应时间太长」），但 `git push` 完全正常。

**实测证据**：
| 测试 | 结果 |
|---|---|
| TCP 连 GitHub 各 IP 的 **443** 端口 | ✅ 全部 OPEN |
| HTTPS 访问 `github.com`（**换任意 IP 都一样**）| ❌ 全部超时（TLS 握手被阻断）|
| HTTPS 访问 `api.github.com` | ✅ 200 |
| **SSH（22 端口）访问 `github.com`** | ✅ **OPEN** |

**结论**：这是针对 `github.com` 域名的 **SNI 定向干扰**，**不是 IP 问题** → **改 hosts / 换 IP 都无效**。

**应对策略**：
| 目的 | 方案 |
|---|---|
| **推/拉代码** | 走 **SSH**，**无需代理**，稳定可用 ✅ |
| **看 GitHub 网页** | 开 **Clash Verge** 代理（加密隧道绕过 SNI 检测）|

> ⚠️ **代理节点选择**：**不要用香港机房节点**（DataDome 会拦截），要选节点名带「**住宅 / 家宽 / 原生IP / Residential**」的。

### 6.5 DataDome 风控（注册/登录时）

GitHub 用 **DataDome** 保护 `/signup` 等接口。响应头特征：`x-datadome: protected`、`Set-Cookie: datadome=...; Max-Age=31536000`（**有效期 1 年**）。

**触发特征**：机房/代理 IP（`hosting:true`）、**脏的 `datadome` cookie**、广告拦截扩展（拦掉 `captcha-delivery.com` 验证脚本）。

**应对**：
1. **清除 `github.com` 站点数据**（F12 → Application → Storage → Clear site data）← 关键！
2. 关闭 uBlock/AdGuard 等广告拦截扩展
3. 换住宅类节点（不要机房 IP）
4. 从 `github.com` 首页自然点击进入，不要直接输 `/signup`
5. 失败后**不要连续重试**，等 24 小时

### 6.6 其他已知现象

- **pybullet 加载时的 `b3Warning: No inertial data for link`** → 无害警告，URDF 部分 link 未写惯性参数，PyBullet 用默认值兜底
- **`git log --oneline` 易打成 `--online`** → 会报 `fatal: unrecognized argument`，用 Tab 补全可避免
- **PowerShell 敲中文提交信息不会乱码**（已实测 UTF-8 正常），但**引号必须用英文 `"`**

---

## 7. ★ 与论文的差距清单（写报告/答辩直接用）

### 7.1 30 个任务 —— ✅ **完全对应**

| Level | 分类 | 数量 | 任务名（`tasks/all_tasks.py` 注册名）|
|---|---|---|---|
| **1** | 基础拾取放置 | 5 | `pushcube` `pickcube` `pickplacecube` `uprearcup` `balltrashcan` |
| **2** | 工具操作 | 11 | `rotatefaucet` `rotatehearth` `openmicrowave` `closemicrowave` `opendrawer` `closedrawer` `liftmug` `openlaptop` `ballmug` `pourwater` `breadtoaster` |
| **3** | 双手协作 | 9 | `ballbimanual` `potbimanual` `pottomato` `pottray` `stackboxes` `panhearth` `tidyuptable` `pottomatoout` `plateoven` |
| **4** | 长时域序列 | 5 | `pottomatoplate` `penbrushpot` `drawerbook` `twistbottlecaps` `stacktoyblocks` |

**与论文 Table 1 的 30 个任务 100% 对应**，唯一差异：
> 论文写作 `twist bottle cap`（**单数**），代码注册名是 `twistbottlecaps`（**复数，多了个 s**）

### 7.2 仿真器 —— ⚠️ **根本性差异（最重要的偏离）**

| 关键项 | 论文 | 本项目 |
|---|---|---|
| **仿真器** | **NVIDIA Isaac Sim** | **PyBullet** |
| 物理引擎 | PhysX | Bullet |
| 渲染 | 照片级（photorealistic）| 基础 OpenGL |
| 模型格式 | USD | URDF |

> 🚨 论文标题就是 **「Simulator-Centric」** —— **仿真器本身是论文的核心贡献**。论文明确说明选 Isaac Sim 是因为「高性能 PhysX 引擎 + 照片级渲染器」，才能让仿真结果预测真机表现。

**★ 已确认的项目决策（2026-09-12，用户明确表示）**：
> **「我就是要用 PyBullet 做」**
>
> **理由**：PyBullet 轻量、无需 GPU、无需 Isaac Sim 授权/许可，且已成功加载 3 种机器人。
>
> **⚠️ 写报告/答辩时必须声明**：本项目是 **PyBullet 简化复现**，聚焦「任务体系 + 遥操作流程 + 评测协议」，**不复现论文的物理/视觉保真度结论**（不强声称「仿真能预测真机」）。

> 💡 项目里同时存在 USD 资产 `robots/h1_with_hand/h1_with_hand_rt.usd`（59.5 MB），未来若升级到 Isaac Sim 可用。

### 7.3 四种遥操作接口 —— 覆盖 **2 / 4**

| # | 论文接口 | 本项目对应文件 | 状态 |
|---|---|---|---|
| ① | **Vision-based**<br>SMPL + SMPLer-X + MediaPipe + Dex-Retargeting + PINK + 卡尔曼 | `teleop/camera_interface.py`<br>（注释：联想电脑 RGB 摄像头采集）| ⚠️ **实现不同**（只是摄像头采集，非视觉遥操作）|
| ② | **MoCap-based**<br>Xsens MVN（17 IMU）+ Manus 手套（20 DoF）| ❌ 无 | ❌ **缺失**（需硬件）|
| ③ | **VR-based**<br>Apple Vision Pro + PINK + AnyTeleop | `teleop/vr_interface.py`<br>（Vision Pro 手腕追踪 → 关节角度）| ⚠️ **部分实现** |
| ④ | **Exoskeleton-based**<br>HOMIE 同构外骨骼 + 霍尔手套（15 DoF）| ❌ 无 | ❌ **缺失**（需硬件）|

> 论文整个实验设计是「**同一套任务 × 4 种接口**」。目前只有 1~2 种可用，**对照实验暂时无法成立**。

### 7.4 关键技术组件 —— 缺失清单

| 组件 | 用途 | 状态 |
|---|---|---|
| **dex-retargeting** | 向量优化手部重定向（论文 Vision + VR **都靠它**）| ❌ 未使用（`pip install dex-retargeting` 可装）|
| **PINK** | 逆运动学求解 | ❌ 未使用（`pip install pink`）|
| **SMPL / SMPLer-X** | 人体姿态与体型估计 | ❌ 未使用 |
| **MediaPipe** | 手部关键点检测 | ❌ 未使用（`pip install mediapipe`）|
| **卡尔曼滤波** | 关节平滑（减少抖动）| ❌ 未使用 |
| **CLIK** | 闭链逆运动学（MoCap 用）| ❌ 未使用 |
| **灵巧手 DoF 匹配** | 论文：H1-2=6DoF / GR1-T2=6DoF / G1=4DoF | ⚠️ 待核实 |

### 7.5 本项目「额外」有的（论文没有）

| 组件 | 说明 |
|---|---|
| `envs/domain_randomizer.py` | **域随机化**（光照/摩擦/纹理/位置）—— 论文明确**没有**做 sim2real 随机化，这是本项目的扩展 |
| `README.md` / `NOTICE` / `.gitignore` / `docs/` | 工程化配套文档 |
| Git / GitHub 版本管理 | 论文是发布源码包，本项目用 Git 管理全过程 |
| `linkerhand_sdk` 集成 | 引入 LinkerHand 灵巧手（论文用 Inspire / Fourier / Unitree 手）|

### 7.6 30 个任务环境「搭建进度」盘点（2026-10-14 实测）

> 命令：`E:\python3.11.7\python.exe scripts\capture_task_envs.py --robot h1_2 --steps 480`
> 结果：**30 / 30 环境全部建成并出图**（`成功 30 / 30；失败 0`，退出码 0）。
> 30 张截图 + 6x5 总览拼图 + 逐环境实测数据在 `outputs/task_env_shots/`（不入库）。

**逐层数量与 §7.1 完全一致**：Level 1 = 5 / Level 2 = 11 / Level 3 = 9 / Level 4 = 5 = 30。

**已经具备的（本次实测证据）**

| 项 | 状态 |
|---|---|
| 30 个任务类都能被 `envs/SimulationEnv` 建起（机器人 + 桌面 + 任务物体） | ✅ 30/30 |
| 每个环境能连续跑 480 步物理（240 Hz = 2 秒）无异常 | ✅ `steps=480` |
| 离屏渲染出图（`DIRECT` + `getCameraImage`，无需显示器） | ✅ 30/30 |
| 截图内容自检（像素标准差 / 颜色数，防「空图当成功」） | ✅ 最低 std 47.74、最少 217 色 |

**还没做完的：3 类环境缺陷（同一份报告可复查，逐条附源码判据）**

| # | 现象（实测终点） | 任务 | 源码层原因 |
|---|---|---|---|
| 1 | 抽屉 / 门是**自由刚体**，初始又嵌在固定体里，被物理挤开后掉到地面 | `closedrawer` drawer 落到 z=0.100（正好半高 = 躺地面）<br>`drawerbook` drawer 落到 (1.045, 0.3, 0.100)（掉出桌面）<br>`plateoven` oven_door 落到 (1.111, 0.300, 0.010)（掉出桌面 + 落地）| `drawer`（mass=0.15）/ `oven_door`（mass=0.2）是自由 box，`cabinet` / `oven` 是 `mass=0` 固定体；初始位置与固定体或桌面**穿模**，没有铰链/滑轨约束（PyBullet 里应加 `createConstraint` 或用关节做滑块） |
| 2 | **判据过松：什么都不做，t=0 就判成功** | `openmicrowave`（门 x=0.6508，本体 0.5 -> 0.151 > 0.12 初始即成立）<br>`opendrawer`（抽屉 0.54 - 柜 0.4 = 0.14 > 0.12 初始即成立）<br>`liftmug`（盖 0.475 - 杯 0.42 = 0.055 > 0.05 初始即成立）| 判据只比「位移量阈值」，阈值小于初始位移量；另一侧 `closemicrowave`（要求门与本体 x 距离 < 0.02）与 `closedrawer`（< 0.17）在物理上**几乎不可达**，同样需要重写 |
| 3 | 30 个环境**都还没有能完成任务的控制器** | 全部 30 个 | 本轮用的是 `main.py` 的演示正弦控制器（只让关节摆动，不做任务）；任务级评测需要每个任务的动作序列 / 策略，对应 §9 待办里的评测协议 |

> **一处自我更正（留档）**：自动体检脚本一开始把 `panhearth` 的 `pan`/`lid` 判成「初始穿模」，
> 逐行核对后确认是**误报** —— 盖子（z=0.445）正常落在锅（z=0.430）上，间距 1.5 cm，
> 是锅盖稳定后的正常状态。同理 `cabinet` 的 z=0.300 是设计值（固定体），不是掉落。

**结论**：「30 个仿真环境」的**可跑性**这一层已经打通（能建、能跑、能看图、有数据），
剩下三件事按优先级是：① 修物体放置/约束（第 1 类）→ ② 改成功判据（第 2 类）→
③ 给每个任务配控制器（第 3 类，工作量最大，属评测协议范畴）。

---

## 8. 历史会话要点（2026-09-12 首次会话）

### 8.1 时间线（按顺序）

| # | 内容 |
|---|---|
| 1 | **GitHub 注册被 DataDome 拦截** → 诊断根因（香港机房 IP `89.185.26.180` + 有效期 1 年的脏 `datadome` cookie）→ 清站点数据后注册成功 |
| 2 | 确认 Git 已安装：`E:\Git`，v2.55.0.windows.3 |
| 3 | 配置 Git 全局身份（使用 GitHub 隐私邮箱，不暴露学校邮箱）|
| 4 | 生成 **ed25519** SSH 密钥并添加到 GitHub；核对主机密钥指纹与官方一致（无中间人）|
| 5 | 分析项目体积：**1.8 GB / 12,462 文件**，发现 3 大体量目录 + 一个 **119 MB 的 pack 文件**（超 GitHub 100MB 硬限）|
| 6 | 编写 `.gitignore`（排除 lib / robots / linkerhand_sdk / .venv / .idea / __pycache__）+ `README.md` + `NOTICE`（第三方开源声明）|
| 7 | `git init` → 首次提交（**27 文件 / 115.4 KB**）→ push 成功 |
| 8 | 修复 `show_hand.py` 硬编码盘符路径（改为基于 `__file__` 动态计算）|
| 9 | **完整走完 PR 流程**：`checkout -b` → 改代码 → `diff` → `add` → `commit` → `push -u` → 网页 Create PR → Merge → `checkout main` + `pull` + `branch -d` |
| 10 | 排查并解决 4 个环境问题：**中文用户名 SSH 失效 / E 盘权限不继承 / SNI 阻断 / 私钥 ACL** |
| 11 | 把 SSH 密钥从 `C:\Users\王宪雨\.ssh` **迁移到 `E:\ssh`**（用户不想在 C 盘存东西）|
| 12 | 完整精读论文（13 页全文 + 68 篇参考文献）|
| 13 | **实测验证 3 种机器人能被 PyBullet 成功加载**（H1-2=55关节 / GR1-T2=70关节 / G1=53关节）|
| 14 | 生成本交接文档（`docs/PROJECT_CONTEXT.md`）|
| 15 | 排查「仿真环境到底能不能跑」，**发现 8 个问题**（详见第 12 节）|
| 16 | **修复接口不匹配**：重写 `RobotLoader`（3 种机器人 + 关节映射表）、修 `apply_action` 动作错位、修 `SimulationEnv`、修 `main.py` |
| 17 | **首次跑通端到端**：`test_import.py` 6 项全绿（`EXIT=0`），H1-2 + pushcube 跑 300 步无异常 |
| 18 | 新增 `docs/INTERFACE_CONTRACT.md`（接口契约）+ `docs/TEAM_ONBOARDING.md`（队友上手指南）|
| 19 | 用户告知：**重定向算法由队友（肖奕阳）负责，数据采集由另一位队友负责** |
| 20 | **设计并实现「离线 Vision 流水线」**：新增契约 G/H、`scripts/replay_actions.py`、`scripts/make_sample_data.py`（详见第 14 节）|
| 21 | **接入队友的重定向输出**，用「维度范围 vs 关节限位」二分图匹配推导出目标手是 **l21**（非 l7），新增 `scripts/replay_hand_angles.py`（详见第 15 节）|
| 22 | **✅ 手部链路视觉验证通过** —— GUI 回放肉眼确认「四指大幅屈伸 / 拇指基本不动 / 向掌心弯曲」，与预测完全一致，坐实映射与型号判断（详见第 15.8 节）|
| 23 | **交付平滑滤波模块**（`teleop/filters.py`）+ **两处重要修正**：① 34 个"跳变"其实是真实运动（非噪声）② 发现 3 个「整帧异常」坏帧且 `*_valid` 漏标；新增 `scripts/check_gbk_safe.py`（详见第 16 节）|
| 24 | **项目决策：先只做手部（不含手臂）**；新增 `scripts/verify_hand_pipeline.py` **一键验收**（5 项全 PASS）；重新评估队友问题清单（8 项 → **2 项需处理**）；补充「阶段限制声明」与「与论文差异表」（详见 OFFLINE_PIPELINE 8.8 / 8.9）|
| 25 | 【2026-09-29】补全 **Apache-2.0 LICENSE 正文**（第三方模型资产再分发合规要求；正文取自本机某个 Python 包自带的副本，因未开代理下载不到 apache.org）|
| 26 | 【2026-09-30】新增 **`docs/ENVIRONMENT_SETUP.md`**（环境配置与工具链备忘）；修复 `teleop/native_hand.py` 里残留的 `U+26A0 U+FE0F`（位于 docstring，潜伏雷），`check_gbk_safe.py --strict` 退出码归零 |
| 28 | 【2026-09-30】**三台机器人演示打通（本轮真正的主任务）**：用户指出「文档收尾不重要，要让 GR1-T2 和 G1 也能用手部数据演示」。① 新增 `scripts/check_native_hand_motion.py` 逐关节实测运动量 → 三台全部 PASS（24/22/14 个关节都会动，稳定后跟踪误差 0.0000 rad，GR1-T2 限位截断 0.0%）；② 查出真正卡住演示的 bug —— `replay_hand_native.py --view hands` 用「整机包围盒」猜手的位置且**写死 yaw=135 / 距离 1.1 m**，而三台机器人手在世界里的方位不同（右手 x：H1-2 +0.36 / GR1-T2 **-0.19** / G1 +0.25），GR1-T2 的手在画面里只剩几个像素；已改为按【手部 AABB】取景 + 按「躯干 → 手」方位自动选 yaw，并在 `teleop/native_hand.py` 新增正式接口 `link_extent()`；③ 新增 `scripts/render_hand_snapshots.py` 用 `getCameraImage` 离屏出图（无需显示器），并内置「相邻快照像素差」自检 —— 该自检当场抓出我自己写的一个 bug：hold 排除写成「名字比索引」，导致手部关节被 500 力矩锁死、4 张快照全一样；④ README 增「三台机器人演示验收」表 + PROJECT_CONTEXT 3.2(5) |
| 27 | 【2026-09-30】**文档可发现性收尾**：README 新增「📚 文档索引」、补齐目录结构（`scripts/` `datasets/` `native_hand.py` `filters.py`）、修正「`robots/` 未入库」的旧说法；本文件「附录：相关文件索引」与第 0 节开场白同步补上 ENVIRONMENT_SETUP；`.gitignore` 补 `tmp_*/` 并入库（此前只存在于工作区，队友 clone 后无效）|
| 29 | 【2026-10-01】**① 撤回对队友代码的全部改动**：`retargeting/arm.py`、`retargeting/tracking.py`、`tests/test_dual_arm.py`、`inspect_angle_h5.py` 已 `git checkout 3e4e763` 还原，`git diff 3e4e763 HEAD -- retargeting/ ...` **为空**；兼容逻辑改放新增文件（`input_adapters/hand_keypoints.py` 转发层 + 缺常量兜底、`tests/conftest.py` 缺 URDF 时跳过）<br>**② 修掉「往 C 盘写东西」**：pip 把 **216.78 MB** 下载缓存写进 `C:\Users\王宪雨\AppData\Local\pip\Cache`、`huggingface_hub/hf_xet` 写进 `C:\Users\王宪雨\.cache\huggingface\xet\logs`。已建 `E:\cache\{pip,matplotlib,huggingface,torch,xdg,tmp}`、设 7 个用户级环境变量、把 **1560.54 MB / 889 个文件**的 pip 缓存整体 `robocopy /MOVE` 到 `E:\cache\pip`（缓存仍可复用，`pip cache list` 可见）；新增 `scripts/env_e_drive_cache.ps1`（含 E 盘 ACL 自愈）与 `scripts/check_no_c_drive.py`；规则成文为 §0.1 硬约束 + ENVIRONMENT_SETUP §2.6 |
| 30 | 【2026-10-01】**更正一处误判（留档）**：上一轮我称「`roboticstoolbox 1.4.4` 本来就装着」——**错**，它是 2026-10-01 14:49 才由 `pip install` 装上的（mtime 可查）。结论（`arm.py` 无需改）仍成立，但**理由换成**：① `roboticstoolbox-python` 本来就在 `requirements-retargeting.txt` 里声明，缺依赖应装依赖、不应改队友源码；② 手部链路不导入该模块（实测还原后 `pytest` 40 passed / 3 skipped、一键流水线 7/7 全绿）|
| 31 | 【2026-10-01】**把要求固化成「工程约定」**（用户要求：加上之前所有要求 + 不许在 C 盘存东西）：新增 **`docs/CONVENTIONS.md`**（13 条约定的唯一来源，每条含为什么 / 怎么守 / 怎么查 / 翻车记录）+ **`.clinerules`**（AI 自动加载版，以后新对话自动生效）+ **`scripts/preflight.ps1`**（一键自检 6 项：解释器 / GBK / C 盘零写入 / 队友代码零改动 / 大资产不入库 / 规则文件齐备；`-Full` 再加 pytest 与重定向流水线）。实测：正常路径 `RESULT: [OK]` 退出码 0；故意改 `retargeting/arm.py` 一行 → 第 4 项抓到 `M retargeting/arm.py` 并提示还原命令、退出码 1；`-Full` 下 pytest `40 passed, 3 skipped`、流水线 `7/7 PASS`；还原后 `git status` 干净。README / ENVIRONMENT_SETUP / TEAM_ONBOARDING / RETARGETING_PIPELINE §10.1 / 本节均已接入索引与指针 |
| 32 | 【2026-10-10】**把第 9 节的复核命令变成队友真能跑的**（上一轮遗留项收尾）：原表格第 1 条写「`python scripts/validate_retarget_input.py tmp_motion/npy_replay_sample.npy` → `payloads=10 result=ok`」，但**样本文件不在仓库**（`tmp_*/` 被 .gitignore 排除），生成它的脚本也只躺在 `tmp_motion/` 里 —— 别人 clone 后跑这条必然 `FileNotFoundError`（"数字是对的、命令是死的"）。① 先**实测复现**，确认表格数字无误：双手 12 帧 → `payloads=10 / source=mediapipe_approx / left_shape=(3, 25, 3) / right_shape=(3, 25, 3) / result=ok`，**退出码 0**；单手 8 帧（`--drop-right`）→ `payloads=6 / right_shape=None / result=ok`，**退出码 0**；顺手测出规律 **payload 数 = 帧数 - 2**（`HandWindowBuffer` 攒满 3 帧才吐 payload）；② 新增 **`scripts/make_npy_replay_sample.py`**（只依赖 numpy，不导入 `retargeting` 包，`--frames/--fps/--drop-right/--out` 可选，固定种子可复现），把生成器正式入库；③ 表格改成「先造样本 → 再校验」两步并附实测数字与日期，§8 文件表 + README 目录树同步登记；`check_gbk_safe.py --strict` 与 `check_no_c_drive.py --strict` 均 `[OK]` 退出码 0 |
| 33 | 【2026-10-10】**收尾时抓到一个"看着像回归"的口径 bug（已被本次自检抓到）**：跑 `preflight.ps1 -Full` 时流水线显示 `6/7 PASS`（本会话早先是 `7/7`），一度怀疑改坏了东西。逐行读源码 + 重跑定位：`run_retargeting_pipeline.py` 第 0 步（造原始数据）在 `tmp_motion/pipeline/raw_twohand.h5` 已存在时**按设计跳过**，但旧版**不把这一步记入汇总**（于是只在汇总里剩 6 行），而 `preflight.ps1` 又硬编码打印 `($pass/7 PASS)` —— 两处叠加就成了"7 项里像挂了一项"。① 修 `run_retargeting_pipeline.py`：跳过时记入 `skipped`，汇总新增 `[SKIP]` 行与 `统计：PASS=.. FAIL=.. SKIP=..`（共 N 项），docstring 写明**判据是 `FAIL=0`，不是"必须 7/7"**；② 修 `preflight.ps1`：不再写死 `/7`，改印 `(PASS=.. FAIL=0 SKIP=..; 6 or 7 steps ...)`；③ 两分支实测（均 `结论：全链路跑通`、退出码 0）：复用已有 raw → `PASS=6 FAIL=0 SKIP=1`；全新目录 `--work-dir tmp_motion/pipeline_fresh0` → `PASS=7 FAIL=0 SKIP=0`；④ 口径同步到 CONVENTIONS 约定 5（含新翻车记录 + 版本表第 2 版）、自检清单第 7 项、`.clinerules`、`RETARGETING_PIPELINE` §5/§9/§10 —— 免得下一个接手的人重踩 |

| 34 | 【2026-10-14】**用户新增一条长期要求 -> 落成「约定 14」**：以后改代码**尽量在原有代码的基础上增加代码**，不要动不动新添一个文件 —— "不然你写的有时候总会这一块那一块，后面如果要改的话，会遗漏很多地方，容易出问题"。落地三处（约定 12 要求的同步）：① `docs/CONVENTIONS.md` 新增**约定 14**（6 条怎么守：先找落点 / 就地增量 / 不建 `xxx_fix.py` 这类平行文件 / 3 个新建例外 / 新建必须在同一次提交里登记进文档 / 测试加进既有测试文件；附"为什么"）；② `.clinerules`「工程习惯」同步同一段；③ 本节 §0.1 新增第 6 条（标为 ★★ 规范）。另：`F:\simulation_platform_cs` 侧的实时手臂入口本轮实测可用（`devtools\run_realtime_arm_windows.py` 两个窗口：摄像头预览 + PyBullet 三机器人；`--check` 退出码 0，摄像头窗口 `position=(20,60)` 置顶，`neutral mode track`）|
| 35 | 【2026-10-14】**现场三症状定位到「默认口径」这一层，并改掉**：用户反馈「改代码前手臂向上平举三台机器人的大臂能抬起来、现在抬不起来 / 张开手臂时机器人张不开怀抱 / GR1-T2 的大臂与小臂夹角是反的」。逐层还原后真正的变量是**那一场跑在实时入口的历史默认 `--arm-map chain` 上**（该场日志里没有 `arm map:` 那一行，因为历史上只有 `segments/posture` 才打印；`outputs/tmp_realtime` 里 68 行启动日志全是 posture、一行 chain 都没有 —— 之前几场都是显式带 `--arm-map posture` 跑的），而 §8.10-§8.13 修好的肘位置目标 / 方向口径 / 刚性肩锚点（肘父连杆）/ IK 只动本侧手臂全在 posture 那一支。**同一份真实录制（`live_long_frames.npz`，200 帧）、同样代码与默认，只差 `--arm-map` 的 A/B**：肩行程（度）h1_2 50.11/44.48 -> 119.1/186.83、gr1_t2 78.65/88.51 -> 168.28/149.97、g1 37.24/25.56 -> **243.6**/164.38；肘跨距（mm）g1 123.0/85.1 -> 395.8/392.2、h1_2 右 250.4 -> 663.7；逐轴夹断帧数 370 / 460 -> **0 / 0**；前臂反向帧占比 chain 下**不可测**（chain 不要求肘，报告里没有\"要的方向\"）-> posture 0.000-0.028。改动（按约定 14，全部落在既有文件）：① `realtime_teleop.py` 的 `--arm-map` CLI 默认 `chain` -> `posture`（库 / 桥默认仍 `chain`，探针与老 A/B 逐字不变）+ chain 时也打印启动提示 + help 改写；② `devtools/report_table.py` 加「肩行程 / 上臂方向行程 / 肘跨距 / 夹断帧数」四列、缺字段打 `-`（改前读 chain 报告会 `TypeError` 崩，这也是这一层一直没被看见的原因）；③ `tests/test_arm_realtime_bridge.py` 新增「入口默认是 posture」1 条；④ `docs/multi-robot-arm.md` 新增 **§8.15**（原话 / 根因 / A/B 表 / 改动清单 / 验收 / 如实说）、`docs/RETARGETING_PIPELINE.md` 实时入口小节同步。验收：`preflight.ps1 -Full -Base d8bca93` 退出码 **0**（`RESULT: [OK]`；第 4 项 `809 teammate-owned files: 0 modified/deleted`；第 8 项 `Ran 353 tests [OK (skipped=3)]` + CLI 冒烟 0）、`check_gbk_safe.py --strict` 退出码 0、默认与 `--arm-map chain` 两条 5 帧冒烟 0/0（启动行分别为 `arm map: posture | shoulder end = upper ...` 与 `arm map: chain (no elbow task ...)`）。**如实说未解决**：GR1-T2 仍是六侧最差（前臂方向误差中位 43.0/35.6 度、肘任务残差中位 175.8 mm），`multi-robot-arm.md` §8.13.8 那两条（肘任务世界点与腕任务 link 帧点是否同一口径、关节限位）仍未查。落地提交：`F:\simulation_platform_cs` 分支 `cs/multi-robot-arm` 的 **`2170cfb`**（5 文件 / +213 -45，未 push）|
| 36 | 【2026-10-14】**用户要求「把 30 个仿真环境都搭好、逐个跑出来、每个截图放到指定图片文件夹」-> 本轮先把「能跑 + 能看 + 有数据」这一层做实**：① 新增 `scripts/capture_task_envs.py`（按约定 14 的三个例外之②：现有文件里没有可挂靠的位置）。一个命令把 Level 1~4 的 30 个环境**逐个建起来 -> 跑 480 步物理 -> 离屏截图**（`DIRECT` + `getCameraImage`，无需显示器），相机按【场景真实包围盒 8 角点投影】自动取景（第一版球拟合取景太远，实测后改掉），并复用 `main.py` 的演示控制器（只把 `_ACTION_DIM` 换成环境真实维度）而不是另抄一份，避免两份正弦波。产物 = 30 张图 + 6x5 总览拼图 + `report.json`/`report.txt`（不入库），`--copy-to` 可复制到用户指定目录。② **实测**：`--robot h1_2 --steps 480` -> **30/30 建成并出图，`成功 30 / 30；失败 0`，退出码 0**；截图自检最低 std 约 47.7 / 最少 217 色（防「空图当成功」）；31 个文件（1.69 MiB）已复制到用户目录。③ **过程中修掉两个坑**：本机 PyBullet `ER_TINY_RENDERER` 的扁平缓冲**行序是自上而下**，照抄 `render_hand_snapshots.py` 的 `[::-1]` 会画出**上下颠倒**的图（第一张图桌子悬在机器人头顶）-> 默认不翻、留 `--vflip`；相机球拟合取景太远 -> 改 8 角点投影贴边。④ **如实报告 3 类环境缺陷**（新增 §7.6 盘点 + §3.2(6) 实测 + §9 待办细分）：`drawer`/`oven_door` 是自由刚体又初始穿模，480 步内掉到地面或掉出桌面（`closedrawer` z=0.100 / `drawerbook` x=1.045 / `plateoven` z=0.010 & x=1.111）；`openmicrowave`/`opendrawer`/`liftmug` 的成功判据在 t=0 就成立（源码判据阈值 < 初始位移量），`closemicrowave`/`closedrawer` 判据物理上几乎不可达；30 个任务都还没有能完成任务的控制器。**并留档一处自我更正**：体检脚本起初把 `panhearth` 的锅/盖判成"初始穿模"，逐行核对后确认是误报（盖子 1.5 cm 正常落在锅上）。⑤ **顺手修掉一个会坑死用户目录的坑**：OpenCV 在 Windows 上按 ANSI 代码页打开文件路径 —— 用户目录 `C:\Users\王宪雨\...` 含中文，`cv2.imwrite` 会**静默返回 False、什么都不写**（校验时也读不出来）；已改为 `cv2.imencode`/`imdecode` + Python 二进制读写（`save_cv_image` / `read_cv_image`），用 `--out "E:\cache\tmp\中文截图测试"` 实测退出码 0、图正常写出 |
| 37 | 【2026-10-15】**用户现场反馈「GR1-T2 的手臂角度没有任何规律，怎么拐就怎么拐，达不到 H1-2 那样准 —— 是机器人手臂的问题还是算法的问题？」-> 逐层取证后判定是「映射层（算法侧）的口径问题」，并且把口径落成「每台机器人加载姿势的属性」**。① **先把「机器人跟不上」排除掉**（同一份 `live_long_frames.npz`、200 帧、`--arm-map posture --arm-direction-rate-deg-s 0 --arm-slew-mm 40`）：报告里 `elbow_task_mm`（实际到达）与 `solution_elbow_mm`（IK 自己解出来的）**六个机位逐位相同**（gr1_t2 左 197.5 / 197.5、右 137.6 / 137.6）-> 物理忠实执行了解算结果；逐帧关节转储（探针钩住 `record_posture`/`apply`）显示 achieved 与 commanded **逐列相同**（肘 0.490..2.272 vs 0.491..2.279 rad）且肘**没有**翻进 URDF 允许的 ±2.27 反向区；两个目标点的半径逐帧等于本机自己的连杆长（`|cmd_elbow-upper|` 中位 178.6 vs 实际 178.7 mm）-> 锚点口径没问题（§8.13 那条已修好）；真正的病在**上臂方向**（gr1_t2 左 65.0 度 / 右 42.9 度，而 h1_2 18.3/14.5、g1 17.0/17.6）。② **根因**：方向口径搬的是操作者的**绝对**方向（`u = axis_map(你的上臂方向)`），而 gr1_t2 加载姿势下前臂朝**后**下方（基座系 `(-0.80,+0.09,-0.60)`、肘弯 111.2 度），h1_2 / g1 是朝前下方（`+0.56` / `+0.57`）—— 同一句「把手前臂指向前下方」对另两台是「本来就差不多」，对 gr1_t2 是「先拆开 97 度」（报告里那对不变量 `operator_fore_deg` 47.7 -> `robot_fore_deg` 96.7 早就写着，模块注释也写着「gain=1 时两个数应当几乎相等」），它于是被顶进自己限位的一角（肩 pitch -2.785/限位 -2.79、肩 roll -0.560/-0.570、肘 2.272/2.27，合计 27/200 帧贴死），残差全摊到肘上。③ **三种口径的 A/B**（探针 `tmp_ik_frame/probe_posture_frame.py`，只在探针里替换 `_posture_terms`，库不动）：`mapped`（绝对）-> `delta`（把你的旋转套在本机自己的中性方向上）-> `align`（把「你的中性 -> 本机中性」那个固定旋转套在你的绝对方向上）在 gr1_t2 左是 65.8 -> **8.2** -> 14.9 度（肘残差 197.5 -> **25.6** -> 46.0 mm），但 h1_2 右是 9.9 -> 45.5 -> 43.8 度、g1 右 18.8 -> 26.1 -> 23.8 度 —— **绝对档对 h1_2/g1 是对的、对 gr1_t2 是错的**，所以口径是**每台**的属性、不能全局换。④ **改动（按约定 14，全部落在既有文件里）**：`arm_profiles.py` 加 `POSTURE_FRAMES` 与 `ArmProfile.posture_frame`（默认 `mapped`，**gr1_t2 填 `delta`**）+ 校验 + `describe()`；`arm_realtime.py` 加 `posture_frame`（`None` = 逐台按 profile）、`posture_frame_of()`、`_turned_direction()`、模块级 `_rotation_between_directions()`（最小旋转，含平行/反向兜底）、`_posture_terms` 的 `delta` 分支、每帧 `info["posture_frame"]`、`summary()` 的 `posture_frame`/`posture_frame_by_robot`、新增 `elbow_bend_neutral()`；`realtime_teleop.py` 加旗子 `--arm-posture-frame {auto,mapped,delta}`（**默认 `auto` = 逐台**）、启动行 `posture frame = auto (h1_2:mapped, gr1_t2:delta, g1:mapped)`、报告 `arm_posture_frame`、每帧按台算 `elbow_bend_baseline`；`arm_scene_driver.apply_each(elbow_bend_baseline=...)` 与 `arm_ik.posture_from_bend(neutral_bend=...)` 让 `delta` 档把**零空间参考的基准**换成操作者的中性肘弯（不换基就是 §8.17.4 那个常数）；测试 +6（bridge 逐台默认/旗子强制 + `delta` 的两条不变量、并把既有「搬绝对方向」那条**显式钉 `mapped`**；ik_multi_robot 入口默认 `auto`、坏档被拒、换基两种写法等价、`apply_each` 只对点名的台换基）。⑤ **实测（命令 + 退出码 + 关键不变量）**：默认档 **gr1_t2 左**肘任务残差中位 196.4 -> **28.6** mm、上臂方向误差 65.0 -> **9.2** 度、前臂 49.4 -> **7.2** 度，**gr1_t2 右** 129.6 -> **47.6** mm、42.9 -> 15.4 度；**h1_2 / g1 六个读数逐位不变**；`--arm-posture-frame mapped` 与**改前逐位相同**（= 历史行为仍可达）；`--arm-posture-frame delta` 强制全部时 h1_2 右退化（46.5 -> 78.4 mm）—— 这就是"不能全局默认 delta"的证据；`pytest tests -q` 退出码 0（`377 passed, 3 skipped, 376 subtests passed`，基线 371/3/347）、`check_gbk_safe.py --strict` 0、`check_no_c_drive.py --strict` 0、`preflight.ps1 -Full -Base d8bca93` 0（`RESULT: [OK]`；`unittest discover -> Ran 380 tests [OK (skipped=3)]`；第 4 项 `809 teammate-owned files: 0 modified/deleted`）、现场固化档入口 `--check` 0。⑥ **如实说**：`delta` 的「转角幅度」不是逐字相等（同一次旋转作用在方向不同的两条骨头上，实测前臂 47.7 -> 31.8/66.4 度），要让幅度也逐字相等得走 `align`（数字留在 §8.19.4，但它对 gr1_t2 右臂没修好）；h1_2 右 / g1 右在绝对档下为什么仍然准（它们的前臂偏移是 -22~-25 度）这一层**没查** —— 所以这次只动 gr1_t2 的默认，另两台一个数都不动。落地提交：`F:\simulation_platform_cs` 分支 `cs/multi-robot-arm` 的 **`04e8660`**（10 文件 / +546 -34，未 push；同一次提交带上上一轮 §8.18 的收尾 —— `--arm-slew-pair-floor` 那条 CLI 与它的报告字段/用例此前只在工作区，与本轮改动同文件纠缠）|

| 38 | 【2026-11-10】**用户要求「按现有进度 + 论文复现要求，在日志里写一份 1-2 周内完成所有内容的详细规划」，并顺带找回丢失的交接日志**。① **日志本身**：本文件在 `main`（`1a1984b`）里仍然存在（111,838 字节 / 1807 行，最后更新 2026-10-15），但当前分支 `restore/arm-1aaabb2` 的工作树里**没有**它（该分支从更早的树分出，`docs/` 只带了 CONVENTIONS / RETARGETING_PIPELINE / multi-robot-arm / task-evaluation 等）—— 已用 `git checkout 1a1984b -- docs/PROJECT_CONTEXT.md` 恢复到本分支，改前打 tag `checkpoint-20261110-1404-pre-project-context-restore`（HEAD `0f96d2b`）。② **今天的实测（三机器人 VR「手臂冻结」）**：同一份 900 帧录制里 209 次 `nonfinite_or_limit` 拒绝**全部**落在腕关节，超限量级 0.1-7.8 mrad（纯数值噪声）；`arm_ik.py` 加 `ARM_LIMIT_TOLERANCE = 1e-2`（`solve()` 门槛 / `_arm_values()` 裁剪 / 新增 `_limit_violation()` 诊断）后 headless 复跑：拒绝 **209 -> 2**、h1_2 位置误差 p50 **4.35 -> 3.00 mm**、姿态 p95 **17.24 -> 11.65 度**（分支 `fix/limit-clamp-and-guard-margin`，commit `d9fb470`，已推 origin）。③ 新增 **§20 两周冲刺计划**（11-10 → 11-23，14 天逐日排期 + 4 个里程碑 gate + 现场采集 SOP + 风险降级表）|
| 39 | 【2026-11-16】**用户现场反馈「翻转手腕时机器人的手腕并不会翻转，而且它们的手腕会有点抽搐的感觉 —— 是你改的代码让手臂不动吗」**。① **先排掉「改坏了」**：`--arm-ik-mode pose` 只**多加**一遍姿态求解，位置那一遍逐字没动（两次现场的位置目标与腕关节值都在正常跟随）；`--wrist-trace` 两场（`track` 3090 行 / `hold` 885 行）里 `ori_err` 只有 4-9 度 —— IK 到位，不是被锁住。② **根因（两个症状同一个机制）**：掌面基依赖米制 21 点手，而现场只有 **15-25%** 的帧给了完整手点，`ori_source` 于是逐帧在 `palm_normal` / `three_point` 之间掉；`three_point` 对**旋前完全不敏感**，于是丢一帧掌面基 = 把腕目标从「拧了 90 度」瞬间改成「没拧」（= 手腕不转），而两套不同口径的目标交替下发，3 个腕关节就在**对称解**之间跳 —— 实测 g1 左 `left_wrist_pitch_joint` 停在 **-1.512 / +1.619 rad** 两颗限位上反复换边（URDF 限位 **±1.614429558**），CSV 里第 237 帧正好是 `palm_normal` 中间夹了一帧 `three_point`，腕先到 -1.700、下一帧落到 +1.619 = 现场看到的「抽搐」。顺带记：`ori_turn` 113-140 度、腕顶在限位上而 `ori_err` 只有 4-9 度 = 「人的腕行程比机器人宽」，不是 IK 算错（§8.19.3 的 `wrist_usage` 就是量它的）。③ **另记一条口径**：默认 `--arm-calibrate-mode track` 会在拧腕过程中**每帧重新平均所有样本**（`_refresh_neutral`，含 `orientation_palm`），中性自己跟着漂 -> `track` 那场 `ori_turn` 冲到 178.8 度 / `ori_err` 116-150 度，`hold` 同一套代码是 113-140 / 4-9 度；**带姿态的现场验收必须用 `hold`**。④ **改动（约定 14，既有文件增量）**：`arm_realtime.py` 加 `PALM_HOLD_S`（默认 0.5 s）+ 构造参数 `palm_hold_s`（`0` = 关，A/B 基线）+ `_palm_hold` 每侧 `(四元数, 距今秒数)` + `update()` 按真实 `frame_dt` 累加 + `_orientation_frame()` 三分支（`palm_normal` / **`palm_hold`** / `three_point`）+ `counts["palm_held"]` + `reset()` 清理；`realtime_teleop.py` 加 `--arm-palm-hold-ms`（默认 500）、桥构造换算成秒、`_one_source_label()` 把 `palm_hold` 排在 `palm` 之前（否则被吞成 `palm_normal`，「手点丢了多少帧」就看不见了）。⑤ **验收**：`pytest tests -q` 0（新增 6 条：丢一帧四元数逐位保持且计 `palm_held`；0.5 s 内保持、0.6 s 退回三点基且此时手臂未动 -> 单位阵；`palm_hold_s=0` 立刻退回；无掌面基的场次不凭空造；`reset()` 清空；旗子默认/可设 0 且 `palm_hold` 不被 `startswith("palm")` 吞掉）、`check_gbk_safe.py --strict` 0。文档：`multi-robot-arm.md` **§8.20**。⑥ **如实说**：本缺陷有一半根子在摄像头（75-85% 的帧没有手点），代码侧只能如实统计（`palm_hold` vs `palm_normal` 比值）；朝向限幅（§8.19.4）与旋前平滑仍未做 |
| 40 | 【2026-11-16】**接着 #39 那一场的另一半：`palm_hold` 落地后重放同一份 CSV（`wrist_live_hold2.csv`，885 帧级）发现「抽搐」还有第二个源，于是给腕姿态目标加角速度上限**。① **取证**：进 `palm_hold` 的帧本身是干净的（相邻帧 `|d腕|` 仅 **0.000-0.010 rad**）-> 「换基那一下」确实治住了；但掌面基**逐帧角速度**是 p50 13-50、p90 114-210、p99 **740-833**、max **1252 度/秒**（人前臂旋前/旋后峰值 500-700）-> 有 **1-2%** 的掌面帧自己就是坏的，一帧把 g1 左腕目标推到 **3.14 rad**（= 行程尽头），下一帧拉回 **-0.49**，4 s 内出现 0.75 -> 3.14 -> -0.49 三段跳 = 现场说的「抽搐」。同源还有第二种：掌面基丢超 `PALM_HOLD_S` 退回 `three_point` 时两套基的目标差 **2.6-3.0 rad**（f=246 / 439 / 550），也是一帧完成。② **修法**：`MAX_ORIENTATION_RATE_DEG_S`（默认 **540 度/秒**）+ `--arm-orientation-rate-deg-s`（`0` = 关，A/B 基线），每帧按真实 `frame_dt` 放行 `rate*dt` 度、**轴不变**（球面插值缩放，不是 lerp）；链式基准 = 上一帧**放行**值（削掉的角度不累积成追赶，尖峰后一帧回 0、不过冲）；真手速之内**逐位透传**（不是低通，无相位滞后）；对 `three_point` 同样生效；`counts["orientation_clamped"]` 计数、`reset()` 清基准；`_palm_hold` 里仍存**原始测量**（限速只喂 IK，不写回保持值，否则坏帧会污染下一次保持）。与 `_guard_direction` **刻意不同**：`arm` 断流时不清限速基准（跳变护栏判「是不是手在动」要重新起算，限速管「命令连不连续」）。③ **为什么不是保持/低通**：保持会把真的在拧腕也冻住（现场第一需求就是「翻转手腕机器人要跟着转」），低通在每次快转上都给固定滞后与过冲；限速只在超出人体能力的帧上动手，代价是那 1-2% 的帧晚到 1-3 帧。④ **落点（约定 14）**：`arm_realtime.py` 的常量 + 三个纯函数（`quaternion_angle_deg` 与 `--wrist-trace` 的 `ori_turn` **同一口径**、`_quaternion_multiply`、`_quaternion_scaled`）+ 构造参数 `max_orientation_rate_deg_s` + `_last_orientation`，`_orientation_frame()` 拆成 `_raw_orientation()`（测量）+ `_clamp_orientation()`（限速）；`realtime_teleop.py` 的旗子与桥构造传参。⑤ **验收**：`pytest tests -q` -> **420 passed / 3 skipped**，退出码 0（`OrientationSlewTests` 新增 8 条：一帧 150 度只放行 54 度且计 `orientation_clamped`、30 度/帧逐位透传不计数、`frame_dt` 0.5 s 时 150 度原样过去 / 1/30 s 时只剩 18 度、尖峰回中性一步到 0 不反弹、丢手点后的 `palm_hold` 用原始 150 度而非削过的 54 度、`=0` 关闭、`reset()` 清基准、旗子默认 540 可设 0）；`check_gbk_safe.py --strict` 0。同时 `test_rotating_the_forearm_sends_that_relative_rotation` 显式关掉限速（它钉的是基的数学：35 度一帧 > 默认 18 度/帧）。文档：`multi-robot-arm.md` **§8.20.7 / §8.20.8**。⑥ **离线 A/B（真入口、无相机、可复现）**：`--headless --input-npz outputs\tmp_probe\wrist_fixture_spike.npz`（150 帧 dump 的第 90 帧左手拧 150 度，探针不入库），限速关 -> 那一帧 `wrist_b` **-0.0559 -> +2.0364**（Δ 2.09 rad）、`ori_err` **36.31** 度；`--arm-orientation-rate-deg-s 540` -> 同一帧 `wrist_b` **+0.8667**（Δ 0.92 rad）、`ori_turn` **150.00 -> 52.09** 度、`ori_err` 8.12 度；两次退出码 0。未注入尖峰的同一 fixture 上 `ori_turn` / `ori_source` / `target_*` 两份 CSV **逐行相同**（在动的右臂上帧间速率 p50 4.3 / max 17.4 度/秒）= 限速对真手速零影响。⑦ **如实说**：限速**不是**单帧野值驱逐器 —— 一帧就回来的坏帧只能被削到 `rate*dt`（2.09 -> 0.92 rad，仍会抽一下，但已离 g1 的 ±1.61 rad 限位很远），彻底清掉需要「缓一帧回看」= 给每帧加 83-125 ms 延迟，本轮没做；实机 A/B 仍需人在相机前跑 §8.20.7 那两条命令确认；手点检测率（75-85% 帧无手点）、朝向限幅（§8.19.4）、旋前平滑仍未做 |
| 41 | 【2026-11-16 晚】**用户盯着现场那两个窗口（`run_realtime_arm_windows.py` 的「摄像头预览 + PyBullet」）说「效果还是不行」，并问「这三个机器人本身的条件都能做个很流畅的手腕翻转吗」—— 查下来是「旧代码 + 低帧率 + 一台硬件翻不动」三件事叠在一起，于是给腕姿态再加一条与帧长无关的每帧步长上限，并把帧率花销拆开量**。① **先排掉「修没生效」**：那份 16 分钟实跑（`outputs/tmp_realtime/arm_run_20261116_213650.json`，5940 帧 / 967.7 s，21:36:50 启动）的 `bridge.counts` 里**没有** `palm_held` / `orientation_clamped`（当前 `summary()` 是导出的），而 `arm_realtime.py` 的 mtime 是 **21:40:52**、晚于启动，HEAD `ab77371` 里连 `PALM_HOLD_S` 都没有 -> 那两轮改动**一次都没进过窗口**（未提交），所以「还是不行」是旧行为。② **低帧率把速率上限吃掉**：同一份报告 `fps` **5.92** / `median_frame_interval_ms` **155.4** -> `540 x 0.155 = 84 度/帧`，一帧 150 度的坏掌面帧照样放行 84 度；同一场还有 `held_reason.left.no_pose_landmarks` **2251/5940 = 38%**（右 27%）、`arm.residual_mm` **49-329 mm**（目标根本不可达）、`h1_2` 右 `wrist_pitch usage 1.0 at_limit`、`gr1_t2` 左 `wrist_roll`/`wrist_pitch` 两个 `usage 0.0 at_limit`。③ **修法**：`MAX_ORIENTATION_STEP_DEG`（默认 **30 度/帧**，与 `dt` 无关）+ `--arm-orientation-step-deg`（`0` = 只按速率，旧行为）；合并点 `_orientation_limit_deg()`：`rate<=0` -> 0（整套关，A/B 基线逐位透传，步长那条也不能单独接管），否则 `min(rate*frame_dt, step)`；负数按 0；分工是「速率管这个速度人体可能吗 / 步长管这一帧别走太远」，30 fps 下速率只放 18 度，所以 **>=18 fps 两条上限一个数都不改**。④ **三台能不能翻（URDF 实测，正面回答）**：`g1` 的 `wrist_roll` 轴 `(1,0,0)`、**±1.97 rad = ±113 度**（能翻，手的安装点不在转轴上 -> 45 度拧腕时腕位置残差最大 49.8 mm）；`h1_2` 的旋前轴是**肘上的** `elbow_roll`、**-3.01..+2.75 rad = -172..+158 度（跨度 330）**（最适合翻，腕只剩 pitch ±27 / yaw ±73 度）；`gr1_t2` 的 `wrist_roll` **只有 ±0.61 rad = ±35 度** -> **机械上翻不了**（实测右腕 `usage 1.000 at_limit`、朝向残差 **58.53 度**），软件无解；三台**全是 revolute、没有 continuous 关节**，谁都做不到 360 度连续旋转。⑤ **帧率花销**：新增只读探针 `--frame-budget-json`（`FrameBudget`，记 `capture/retarget/bridge/ik/step/trace/preview/rest` 八段的 mean/p50/p90/p99/max + 占比，同时写进报告 `frame_budget`）；60 帧回放实测 **`ik` 46.77 ms = 71.8%**、`step` 13.46 ms = 20.7%、其余 < 8%；把迭代次数当旋钮：**60 -> 15.29 fps、30 -> 22.85、20 -> 27.56**（`ik` 46.9 -> 25.2 -> 17.9 ms）；**真摄像头**（画面里没有人）再跑一跑量回放测不到的 `capture`：**33.94 ms = 40.4%**、`ik` 36.0 = 42.9%、`step` 13.8 = 16.4%，整帧 84 ms = **11.91 fps**（p50 才准：capture 均值被第一帧 361 ms 的模型加载拉高；退出码 1 是「`raw=1/25` 没手点、标定没完成」，不带任何新旗子跑同一条件也是 1），残差在这份不可达 fixture 上 312-402 mm、差别是噪声（**如实说：不构成「30 次迭代一样准」的证明**）。22.9 fps 时速率上限只放 23.6 度/帧、`step 30` 基本不介入 = 回到「真手速逐位透传」的设计点。⑥ **落点（约定 14）**：`arm_realtime.py` 常量 + 构造参数 `max_orientation_step_deg` + `_orientation_limit_deg()`；`realtime_teleop.py` 两条旗子（`--arm-orientation-step-deg` / `--frame-budget-json`）+ `FrameBudget` 类 + 循环里八处计时（不给旗子时一处都不构造）。⑦ **验收**：`pytest tests -q` -> **433 passed / 3 skipped**，退出码 0（新增 `OrientationStepCapTests` 7 条 + `FrameBudgetTests` 6 条；三个老的速率测试与 `_calibrated_with_a_palm_basis` 显式关掉步长那条，各自只钉一个机制）；`check_gbk_safe.py --strict` 0；离线三档 A/B（`outputs/tmp_probe/ab_off.csv` / `ab_rateonly.csv` / `ab_step30.csv`）同一坏帧一帧 Δ **2.09 -> 0.68 -> 0.53 rad**，900 行里 `ori_source` 全同、`ori_turn` **只有第 91 帧不同**。文档：`multi-robot-arm.md` **§8.20.9（步长上限 + 三台机械判决）/ §8.20.10（帧率预算）**。⑧ **未做**：现场 A/B 未跑（要人 + 相机）；`gr1_t2` 的 ±35 度是硬件；手点缺失 38%、目标不可达 128-329 mm、「缓一帧回看」驱逐单帧野值（+1 帧延迟）都留给后续。
| 42 | 【2026-11-16 晚】**现场第三轮反馈（同一跑、同一套手势）：「H1-2 抽得最凶、转腕最不准；GR1-T2 转的方向和我相反；G1 最好但也会抽」，于是把「抽」从朝向上限挪到关节空间去量、去修**。① **先量**：`outputs/tmp_realtime/live2.csv`（9264 行，`outputs/tmp_probe/analyze_joints.py`）逐关节统计 `--wrist-trace` 的 `wrist_a/b/c` —— **朝向上限确实守住了**（`\|Δori\|` max **29.98°**，一帧没超），但**单个腕链关节一帧能跳 193-238°**：gr1_t2 左 `wrist_roll`（±35°）**64.0%** 的帧钉在 -35° 限位上、`wrist_pitch` 42.4%（右 55.4%），残余朝向由 `wrist_yaw`（±170°）接走、单帧最大 **213.2°**；h1_2 左 `elbow_roll` 单帧 **202.3°**（右 193.1°）；g1 左 `wrist_roll` **237.8°**。② **根因**：`PyBulletArmIK._ik_sequence` 在几遍之间用 `_teleport`（`resetJointState`）**直接把关节写成解**，渲染出来的就是那个解 —— 所以「限电机目标」等于没限；现场看到的「抽」是关节跳，不是朝向跳。三句话各有归属：**gr1_t2「转反」不是符号错**，是唯一能拧的轴 35° 就位满、剩下的由方向不对的 `yaw` 去追（这就是画面上的「往回拧」）；**h1_2 最抽**因为腕链第一个关节是 `elbow_roll`（整个小臂 + 手一起转，力臂最大）；**g1 最好**因为行程最大（roll ±113°）。③ **修法（约定 14，既有文件增量）**：`arm_ik.MAX_WRIST_JOINT_STEP_DEG = 20.0` + `PyBulletArmIK._limit_wrist_step`（构造参数 `max_wrist_joint_step_deg`，**adapter 默认 0 = 关**，免得演示扫掠一次走 200 mm 被拖住）+ `last_commanded`/`wrist_step_clamped`，在 `solve()` 末尾**同时限「解」与「关节状态」**（只限解没用，见 ②），基准取上一帧**放行**值（误差不累积，与 `arm_realtime._clamp_orientation` 同构），**只限腕链**（`self.wrist[side]` = `orientation_scope="wrist"` 放开的同一组），肩/肘仍归位置任务；`arm_scene_driver` 加 `wrist_joint_step_deg` 透传；`realtime_teleop.py` 加 `--arm-wrist-joint-step-deg`（**默认 20**，0 = 关）+ 横幅 `wrist joint step = N deg/frame`；报告 `arm.ik[*].max_wrist_joint_step_deg` / `wrist_step_clamped`。④ **验收**：`pytest tests -q` -> **438 passed / 3 skipped**，退出码 0（`WristJointStepTests` 新增 5 条 / 58 个子断言：90° 拧腕在 20° 上限下 ≤ 20° 且计数 > 0 / 关掉时同一拧腕 > 20° 且计数 0 / **状态**也被限（`joint_values` 与放行值逐个相等）/ 肩肘一个数字都不许改 / 库默认 0、负值按 0、`describe()` 落两字段）；`check_gbk_safe.py --strict` 0。⑤ **离线 A/B**（同一份 `wrist_fixture_spike.npz`，只差旗子，`outputs/tmp_probe/compare_joint_cap.py`）：最坏单关节 `\|Δ\|` **44.79° -> 20.05°**，`\|Δori\|` **30.00° 不变**，g1/gr1_t2 残差不变，h1_2 残差中位 433.9/445.1 -> 487.1/501.0 mm（fixture 目标不可达，残差本就在 300-500 mm 噪声区，`\|Δori\|` 中位 15.44 -> 16.84°）。⑥ **顺带修**：点掉 PyBullet 窗口那一场（`Not connected to physics server` 抛在 `finally` 之前）把 93 KB 的 `arm_run_*.json` 吃掉了 —— 现在 `finally` 里补一份最小报告（`aborted: true` + 错误 + 帧数 + `bridge_counts` + `frame_budget`，`realtime_teleop._write_abort_report`），完整报告仍是正常路径唯一权威。⑦ **如实说**：这条上限只把「一帧跳 200°」压成「每帧 ≤ 20°」，**不解决饱和**（gr1_t2 的 roll 仍然只有 ±35°，只是不再让 yaw 甩 213°）；现场还得再跑一轮（`outputs/tmp_realtime/live3.csv` + `budget_live3.json`）才算验收；「整帧只用一个关节承接」与「等一帧再判尖峰」仍未做。文档：`multi-robot-arm.md` **§8.20.11** |
| 43 | 【2026-11-16 深夜】**#42 那条腕关节上限现场验收过了（live3），同时挖出「整场丢旋前」这个更靠前的坑**。① **现场验收**（`outputs/tmp_realtime/arm_run_live3.json` + `live3.csv`，1227 帧 / **8.11 fps**，旗子与 live2 完全一致 + `--arm-wrist-joint-step-deg 20`）：**全部 18 个腕链关节的最坏单帧 `\|Δ\|` 从 237.81° 掉到 21.21°**（g1 左 roll 237.81 -> 20.18、g1 左 pitch 130.51 -> 20.34、g1 右 roll 215.39 -> 20.53、gr1_t2 左 yaw 213.20 -> 21.21、h1_2 左 `elbow_roll` 202.32 -> 20.31、h1_2 右 193.07 -> 20.34），`orientation_clamped` 115/149 次；帧预算 `ik` 67.97 ms = 49.5% / `capture` 38.13 = 27.8% / `step` 16.74 = 12.2% / `preview` 10.64 = 7.7%（整帧 138.3 ms）。② **但跑完发现「跟手」根本没被验证**：`ori_source` 全程 `three_point`（1197/1227）、`palm_held` **0/0** —— 而 22:16 那跑同样代码是 69% `palm_normal`。根因：`arm_realtime._raw_orientation` 的掌面基要跟**中性里的 `orientation_palm`** 相减，中性是标定那 30 帧的均值 —— 标定期间一帧掌面基都没拿到（手没进画面 / 掌面不可见），它就一直是 `None`，**之后再把手伸进来也补不回来**，整场只能退回 `three_point`（对旋前不敏感）。这条以前完全静默（`orientation_palm` 没进报告序列化），现场只看到「手腕不跟」。③ **修（可见性，不改控制量）**：`_palm_samples[side]` 记标定期间真拿到掌面基的帧数 -> `neutral[side].palm_samples`；报告加 `bridge.neutral[*].palm_samples` / `orientation_palm_recorded`；`realtime_teleop._warn_missing_palm_basis` 在标定完成后一次性喊 `[WARN] <side>: the neutral has no palm basis (...) -> pronation/supination is NOT tracked this run ... keep both palms facing the camera for the first 30 frames`。④ **顺手补**：`ArmSceneDriver.summary()` 现在带 `wrist_joint_step_deg` / `wrist_step_clamped`（#42 的计数器原来只挂在 adapter 的 `describe()` 上，报告里读不到，等于没法验收）。⑤ **验收**：`pytest tests -q` -> **442 passed / 3 skipped**（`WristJointStepTests` 6 条 + `PalmBasisVisibilityTests` 3 条）、`check_gbk_safe.py --strict` 0。⑥ **下一步（按顺序）**：重开一跑并**在前 30 帧让两手掌心朝镜头站定**（看到 §③ 那条 WARN 就说明还得重开），再看 `ori_source` 是否出现 `palm_normal`/`palm_hold` —— 那才是「手腕跟着拧」的验收；随后 A/B `--arm-orientation-step-deg 0` / `--arm-wrist-joint-step-deg 0` 各一次，确认这两条上限的观感收益。文档：`multi-robot-arm.md` **§8.20.11.4 / §8.20.12** |
| 44 | 【2026-11-16 深夜】**live4 现场第四轮：「GR1-T2 / H1-2 都是右手跟着拧、左手反着拧，左边还抖」—— 定位到掌面法向在左手上指向掌心内侧（180 度共轭），并顺手回答了「为什么改手腕这么复杂」「能不能只转手腕不转小臂」**。① **现场条件终于齐了**（`arm_run_live4.json` + `live4.csv`，2836 帧 / 7.14 fps，`cal=ok L=P R=P`，`ori_source` 100% `palm_normal`，`palm_turn` 左 131.5° / 右 12.1°）—— 旋前真的在跟，所以「左边反」只能是符号问题。② **离线取证**（`outputs/tmp_probe/left_right_palm_sign.py`，合成两双镜像的手 + 同一个世界旋转）：`cross(hand[5]-hand[0], hand[17]-hand[0])` 在右手给出掌面法向 z 分量 **-1**（掌心朝外），在左手给出 **+1**（掌心**内侧**）；因为 `y = z x x` 是推算的，那个基形式上仍是正交旋转，所以差别等价于**整帧绕前臂多转 180 度**。**关键且反直觉**：共轭 `D R D` 不改变绕前臂轴的分量（两侧旋前都给 **+40.0°**），翻的是另外两维 —— 我最初按「旋前翻号」猜，被探针当场否掉，别再往那个方向找。③ **现场口径的后果**（同一跑 `arm.wrist_usage`，usage = 行程归一化位置）：**左边一路顶着行程、右边全在中段** —— h1_2 左 `wrist_pitch` 1.000 顶满 / `wrist_yaw` 0.000 顶死另一端（右 0.283 / 0.248）、g1 左 `wrist_roll` 1.000（右 0.184）、gr1_t2 左 `wrist_pitch` 1.000；`direction_glitch` 左 77 / 右 39；`command_jerk_deg` p90 左 27-31° / 右 18-21°；`forearm_opposite_fraction` 左 0.058-0.078 / 右 **0.000**。左手目标整体偏 180 度 -> 关节顶死 -> 「反着拧 + 更抖」，而右手目标一直在行程中间所以顺。④ **修**：`MIRRORED_PALM_SIDES = ("left",)`，`palm_orientation_frame(hand_world, arm_world, side)` 里按侧取反法向；`side` 是**必填无默认**（漏传 = 把 bug 装回去），两处调用点（`_record_neutral` / `_raw_orientation`）都带上。⑤ **验收**：`PalmBasisChiralityTests` 5 条（掌面法向 z 分量两侧都须 -1、镜像姿势两侧基逐位相同、同一世界旋转两侧旋前同号、整条手臂摆动两侧相对旋转逐位相同、**接线**走整条桥比对含/不含 `side` 的两份四元数）；把 `MIRRORED_PALM_SIDES` 改回 `()` 后 **3 红 2 绿**（绿的两条正是「旋前本来就对」与接线），把调用换成丢 `side` 的假实现则接线那条红；全量 `pytest tests -q` -> **447 passed / 3 skipped**。⑥ **「小臂跟着转」的答案**：GR1-T2 的 `wrist_roll` parent 是 `hand_yaw_link`、child 是 `hand_roll_link`，在**前臂远端之外**，`elbow_pitch` 是 pitch —— 腕链不可能转小臂；动的是**肩**（同一跑 `gr1_t2.left.shoulder_range_deg`：pitch **82.9°** / yaw **56.0°** / roll 13.6°），即位置任务 + 七自由度余度在重新摆姿势。要连位置那一半也钉住，只能锁同侧上游关节（代价：腕一饱和就彻底不动），这次**没做**，先看符号修完的现场手感。⑦ **下一步**：重开一跑（**前 30 帧双掌朝镜头**，看 `palm_normal`）验证左手是否与右手同向；仍抖再考虑给姿态目标加低通（`_clamp_orientation` 是限速不是平滑，7 fps 下 20°/帧 = 140°/s 仍可见）；最后决定是否加「锁上游关节」开关。文档：`multi-robot-arm.md` **§8.20.13 / §8.20.14** |


### 8.2 已掌握的 Git 工作流

```powershell
cd F:\simulation_platform              # 1. 进项目
git pull                               # 2. 同步最新
git checkout -b feature/新功能名        # 3. 开分支
#   ... 改代码 ...
git status                             # 4. 看改了啥
git diff                               # 5. 确认改对了
git add .                              # 6. 暂存
git commit -m "feat: 做了xxx"           # 7. 提交
git push -u origin feature/新功能名     # 8. 推送
#   ... 网页开 PR → Merge ...
git checkout main                      # 9. 切回主线
git pull                               # 10. 同步合并结果 ⭐ 最容易漏
git branch -d feature/新功能名          # 11. 删本地分支
```

**提交信息规范**：`feat:` / `fix:` / `docs:` / `style:` / `refactor:` / `test:` / `chore:` + **一个空格** + 说明

---

## 9. 待办清单

### 🔴 高优先级

- [ ] **两周冲刺（2026-11-10 -> 2026-11-23）按 §20 执行**：终点 = 10 个代表任务 × {Vision, VR} 的成功率 + 完成时间表（交付物清单见 §20.2）；每天收尾回填 §20.10
- [ ] **⚠️ 备份 SSH 私钥** `E:\ssh\id_ed25519` 到 **U 盘 / 网盘（加密压缩）** ← 最紧急，目前全世界只有这一份
- [ ] 确认 GitHub 仓库可见性是 **Private** 还是 Public（Settings → 最下方 Danger Zone）
- [x] ~~跑通 `main.py` 的 30 个任务~~ → **2026-10-14 已跑通**：30/30 环境逐个建成 + 480 步物理 + 离屏截图（`scripts/capture_task_envs.py`，见 §3.2(6) / §7.6）。**但下面 3 条才是「搭建完 30 个环境」的真正剩余工作**：
  - [ ] 修**自由刚体**物体放置/约束：`drawer` / `oven_door` 会掉到地面或掉出桌面（`closedrawer` `drawerbook` `plateoven`；应加 `createConstraint` 或用关节做真实抽屉/铰链门）← §7.6 第 1 类
  - [ ] 改**成功判据**：`openmicrowave` / `opendrawer` / `liftmug` 在 t=0 就判成功；`closemicrowave` / `closedrawer` 的判据物理上几乎不可达 ← §7.6 第 2 类
  - [ ] 给 30 个任务各配一个**能完成任务的控制器**（或示教动作序列），才能出「成功率 / 完成时间」← §7.6 第 3 类
- [ ] 核实三种机器人挂载的灵巧手 DoF 是否与论文一致（H1-2=6 / GR1-T2=6 / G1=4）

### 🟠 中优先级

- [ ] 深化 `teleop/vr_interface.py`，对齐论文的 **PINK IK + AnyTeleop dex-retargeting**
- [ ] 引入 `dex-retargeting` 做手部重定向（论文 Vision + VR 都靠它）
- [ ] 补齐 `show_robot.py`（PyCharm 里有运行配置，但文件不存在）
- [ ] 修正 `twistbottlecaps` → 与论文一致的命名
- [ ] 处理 `teleop/camera_interface.py`：目前只是摄像头采集，还不是论文的视觉遥操作

### 🟢 低优先级 / 长期

- [ ] 用 **Git LFS** 把 `robots/from_teleopbench/`（181 MB）纳入版本管理
- [ ] 用 **GitHub Issues** 管理任务（答辩时可展示工程管理能力）
- [ ] 实现论文的评测协议：**10 个代表性任务 × 4 种接口 × 4 名被试**
- [ ] 考虑迁到 **Isaac Sim**（项目里已有 USD 资产），以对齐论文的仿真保真度

---

## 10. 常用命令速查

### 运行项目

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = 'F:\simulation_platform\lib'   # ★ 必须！否则找不到 pybullet

python main.py                        # 交互式选任务
python main.py --task pushcube        # 运行指定任务
python main.py --demo                 # 演示模式
python main.py --benchmark            # Level 1 任务基准测试
python main.py --task pushcube --no-render   # 无头模式

python show_all.py                    # 三机器人并排展示
python show_hand.py                   # LinkerHand 灵巧手展示
python test_import.py                 # 模块导入自检

# 30 个任务环境一次跑通 + 批量截图（离屏，无需显示器；产物在 outputs/task_env_shots/）
python scripts/capture_task_envs.py                     # 全部 30 个（每个跑 480 步，960x720）
python scripts/capture_task_envs.py --levels 1 2         # 只跑 Level 1/2
python scripts/capture_task_envs.py --tasks pushcube pickcube
python scripts/capture_task_envs.py --robot gr1_t2       # 换机器人
python scripts/capture_task_envs.py --gui                # 开窗口一个个看（不写 PNG）
```

### Git

```powershell
git status                            # 看状态（最常用）
git log --oneline --graph --all --decorate   # 分支图
git diff                              # 看未暂存改动
git branch -a                         # 所有分支
git remote -v                         # 远程地址
```

### 排障

```powershell
# 报 Permission denied (publickey)
git config --global --get core.sshCommand     # 应显示 Windows OpenSSH 路径
git ls-remote origin                          # 测试 SSH 通道是否通

# GitHub 网页打不开（不是代码问题）
#   → 开 Clash Verge 代理看网页；推代码不需要代理

# 报 ModuleNotFoundError: No module named 'pybullet'
$env:PYTHONPATH = 'F:\simulation_platform\lib'   # 设置后重试
```

---

## 附：相关文件索引

| 文件 | 说明 |
|---|---|
| `F:\simulation_platform\docs\PROJECT_CONTEXT.md` | **本文件**（AI 交接文档）|
| `F:\simulation_platform\docs\CONVENTIONS.md` | ★ **工程约定（本项目规则的唯一来源）**：13 条约定 + 每条的理由/命令/检查/翻车记录 + 一键自检 + 改约定的同步流程 |
| `F:\simulation_platform\.clinerules` | 上面那份约定的 **AI 自动加载版**（与 CONVENTIONS.md 必须一致）|
| `F:\simulation_platform\scripts\preflight.ps1` | **一键自检**（解释器 / GBK / C 盘 / 队友代码零改动 / 大资产 / 规则文件齐备，`-Full` 再加测试与流水线）|
| `F:\simulation_platform\scripts\env_e_drive_cache.ps1` | 把 7 个缓存/临时变量重定向到 `E:\cache\*`（含 E 盘 ACL 自愈；`-Persist` 写用户级）|
| `F:\simulation_platform\scripts\check_no_c_drive.py` | C 盘零写入自检（`--strict` 有 FAIL 即退出码 1）|
| `F:\simulation_platform\docs\ENVIRONMENT_SETUP.md` | **环境配置与工具链备忘**（解释器 / SSH / 代理 / E 盘权限 / GBK 约定 / §2.6 C 盘零写入）|
| `F:\simulation_platform\docs\INTERFACE_CONTRACT.md` | 接口契约 A~H（队友对接必读）|
| `F:\simulation_platform\docs\OFFLINE_PIPELINE.md` | 离线数据流水线（契约 G / H）|
| `F:\simulation_platform\docs\TEAM_ONBOARDING.md` | 队友从零上手（约 30 分钟）|
| `F:\simulation_platform\README.md` | 项目说明（会展示在 GitHub 首页），含「三台机器人演示验收」实测表 |
| `F:\simulation_platform\scripts\replay_hand_native.py` | ★ 原装手回放（**演示入口**；`--view hands` 按手部 AABB 自动取景）|
| `F:\simulation_platform\scripts\check_native_hand_motion.py` | 逐关节实测运动量（三台机器人是否真的会动 + 是否卡死）|
| `F:\simulation_platform\scripts\render_hand_snapshots.py` | 离屏渲染三台机器人手部快照（`getCameraImage`，无需显示器）|
| `F:\simulation_platform\scripts\capture_task_envs.py` | ★ **30 个任务环境一次跑通 + 批量截图**（含 `--copy-to` 复制到指定目录；实测数据见 §7.6）|
| `F:\simulation_platform\scripts\show_hands_all.py` | 三台机器人并排 + 同一份数据同步屈伸（答辩用）|
| `F:\simulation_platform\NOTICE` | 第三方开源声明（TeleOpBench Apache-2.0 / LinkerHand SDK）|
| `F:\simulation_platform\.gitignore` | 已排除 1.8GB 依赖与资产（`lib/` `linkerhand_sdk/` `.venv/` `tmp_*/` 等，放行 `datasets/`）|
| `F:\simulation_platform\.gitattributes` | 把 `robots/**` 下的 `.stl` `.obj` `.usd` 等标记为 `binary`，防止 `autocrlf` 改坏模型 |
| `F:\26年大创 双臂遥操作平台\2025TeleOpBench.pdf` | 论文原文（13 页）|
| `F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\` | 论文官方参考源码（1.4 GB）|
| `E:\ssh\id_ed25519` | ⚠️ **SSH 私钥（唯一一份，务必备份）** |

---

---

## 11. ★ 文档维护约定（后续 AI 请务必遵守）

> 本文档是**活文档**，需要随项目进展持续更新。**每次对话结束前，请检查是否需要更新以下内容。**

### 11.1 更新清单

| 章节 | 何时更新 |
|---|---|
| 文首 **「最后更新」日期** | ⚠️ **每次修改本文件都必须改** |
| **0.1 硬约束** | 新增/修改任何规则时（**必须与 `docs/CONVENTIONS.md` 和 `.clinerules` 同步**）|
| **3.2 已验证可运行的功能** | 新验证成功的功能（**附实测数据**）|
| **5.1 仓库当前状态 / 5.3 提交历史** | 有新提交时 |
| **6. 环境陷阱** | 踩到新坑并解决后 |
| **7. 与论文的差距清单** | 补齐了某个接口/组件后（把 ❌ 改成 ✅）|
| **8.1 时间线** | **每次对话结束时追加一行** |
| **附：相关文件索引** | 新增 / 重命名 / 删除任何 `docs/` 文件或根目录配置时 |
| **9. 待办清单** | 完成一项就勾掉；有新任务就加上 |
| **20. 两周冲刺计划** | 每天收尾回填 §20.10；gate 被触发（砍范围 / 换任务 / n 变化）时同步改 §20.4 / §20.5，**不要只改口头计划** |
| **10. 常用命令速查** | 发现新的常用命令时 |

### 11.2 更新原则

1. **只记录事实和已实测的结论**，不要写猜测或未验证的内容
2. **命令和路径必须可直接复制使用**
3. **踩坑记录写清「现象 → 原因 → 解决」三段式**
4. **绝对不要写入任何密钥、密码、Token、SID 的实际内容**
5. 保持**中文**，风格与现有章节一致
6. 表格优先（可读性最好）

### 11.3 提交约定

- 文档类改动提交信息用 `docs:` 前缀，例如：
  ```
  docs: 更新 PROJECT_CONTEXT 的待办清单与时间线
  ```
- **文档类改动** → 可直接提交到 `main`（单人项目，简单高效）
- **功能 / 修复类改动** → 走分支 + PR（保留评审记录）

```powershell
# 文档更新的标准三步
cd F:\simulation_platform
git add docs/
git commit -m "docs: 更新了 xxx"
git push
```

---

> **说明**：本文件已纳入 Git 版本管理（提交到 `main` 分支）。
> 修改后请及时 commit + push，这样在**任何设备、任何新对话**中都能获取到最新上下文。

---

## 12. 【2026-09-12】仿真环境修复记录（★ 重要）

### 12.1 修复前的真实状态

> **各模块单独都写好了，但核心的 `SimulationEnv` 从未成功运行过一次。**

此前的「环境已搭好」是**错觉** —— 验证过的两个脚本**都没有真正用到 `SimulationEnv`**：

| 验证手段 | 为什么没发现问题 |
|---|---|
| `show_all.py`（能显示三机器人）| **绕过 `SimulationEnv`**，自己直接调 pybullet |
| `test_import.py`（全绿）| 只检查「能不能 import」，`None` 也能 import 成功 |

> 类比：发动机、轮子、方向盘都分别测试过能转，但**从来没把车拼起来开过一次**。

### 12.2 发现并修复的 8 个问题

| # | 问题 | 位置 | 症状 |
|---|---|---|---|
| 1 | **相对导入越界** | `envs/simulation_env.py:11-12` `from ..tasks import` | `ImportError: attempted relative import beyond top-level package`（项目根目录无 `__init__.py`）|
| 2 | **静默吞异常** | `envs/__init__.py` 的 `try/except → SimulationEnv = None` | 致命错误被隐藏成 `None`，运行时只报 `TypeError: 'NoneType' object is not callable` |
| 3 | **变量未定义** | `main.py:116` `if choice in ('quit','exit')` | `UnboundLocalError` → **所有命令行参数模式全崩** |
| 4 | **假阳性自检** | `test_import.py` 只检查 import | 组件是 `None` 也报 ✓ |
| 5 | **函数签名不匹配** | `SimulationEnv.reset()` 调 `load_robot(self.robot_type)`，但 `load_robot()` 不收参数 | `TypeError: takes 1 positional argument but 2 were given` |
| 6 | **返回值类型不匹配** | `load_robot()` 返回 **dict**，`SimulationEnv` 当 **int** 用 | `p.getNumJoints(dict)` 崩溃 |
| **7** | **⭐ 动作映射错位（最严重）** | `BaseTask.apply_action()` 按「第 i 个动作 → 关节索引 i」映射 | URDF **前 12 个关节是腿部**！`action[0]`（本意左肩）被送到 `left_hip_yaw_joint`（左髋）→ **手臂不动、腿乱动** |
| 8 | **渲染判断错误** | `simulation_env.py` 的 `if self.client == p.GUI` | `client` 是连接 id（0），`p.GUI` 是类型常量（1）→ 永不相等 |

### 12.3 修复后的实测结果

**三种机器人的动作空间（实测 2026-09-12）：**

| robot_type | 总关节 | 可动 | 左臂 | 右臂 | 左手 | 右手 | **动作维度** | `action[0]` 对应的关节索引 |
|---|---|---|---|---|---|---|---|---|
| `h1_2` | 55 | 51 | 7 | 7 | **12** | **12** | **38** | **13** `left_shoulder_pitch_joint` |
| `gr1_t2` | 70 | 54 | 7 | 7 | **11** | **11** | **36** | **16** `left_shoulder_pitch_joint` |
| `g1` | 53 | 41 | 7 | 7 | **7** | **7** | **28** | **22** `left_shoulder_pitch_joint` |

> ✅ `action[0]` 现在指向**肩关节**（索引 13 / 16 / 22），不再是索引 0（腿）
> ✅ 自动检查「动作是否误触腿部关节」→ **否**

**端到端测试（首次成功）：**
```
$ python test_import.py
测试6: ✓ 端到端跑通（robot_id=1, action_dim=38）
     任务对象 = ['table', 'cube', 'target']
所有测试完成！
EXIT=0                                   ← 退出码 0（真·全绿）
```

### 12.4 本次新增的文件

| 文件 | 用途 |
|---|---|
| `docs/INTERFACE_CONTRACT.md` | ⭐ **接口契约**（动作空间 / 观测空间 / 控制器接口 / HDF5 格式）—— **队友对接必读** |
| `docs/TEAM_ONBOARDING.md` | 队友上手指南（装环境 → 拿模型 → 验证，含 7 个 FAQ）|

### 12.5 新增的关键 API

```python
# RobotLoader（envs/robot_loader.py）
loader.load_robot(robot_type)          # 'h1_2' | 'gr1_t2' | 'g1'，返回 dict
loader.all_joints                      # {关节名: 索引}
loader.arm_joints['left'|'right']      # 手臂关节名列表（各 7 个）
loader.hand_joints['left'|'right']     # 灵巧手关节名列表
loader.action_joint_names              # ★ 动作向量每个位置的关节名
loader.action_joint_indices            # ★ 动作向量每个位置对应的 pybullet 关节索引
loader.describe_action_space()         # 打印完整映射表

# SimulationEnv（envs/simulation_env.py）
env.action_dim                         # ★ 动作维度（不是固定 28！）
env.action_joint_names / action_joint_indices
env.robot_id                           # int（已修正类型）

# BaseTask（tasks/base_task.py）
task.apply_action(action, joint_indices=env.action_joint_indices)   # ★ 必须传映射
```

### 12.6 ⚠️ 仍然存在的问题（待办）

| 问题 | 来源 | 影响 |
|---|---|---|
| `teleop_pipeline.py` 用 **TRON2(逐际动力) / xArm7** 的硬编码索引 | 早期代码 | 与论文 3 种机器人不符，需重写或废弃 |
| `teleop_pipeline.py` **未接入 `SimulationEnv`** | 架构缺口 | 重定向算法没有标准入口（契约 C 已定义接口，待实现）|
| `main.py` 非交互路径**仍用 `demo_controller`** | 功能缺口 | `--task xxx` 跑的不是真实遥操作 |
| obs **缺少机器人状态向量和相机流** | 与论文的差距 | 队友 B 的 P0 任务 |
| 相机第一人称 **eye 硬编码** `[0,0,1.5]`，未跟随头部 | 与论文的差距 | 队友 B 的 P0 任务 |
| 相机流**只存首尾帧** | 与论文的差距 | 队友 B 的 P0 任务 |

---

## 13. 【2026-09-12】模型资产入库方式决策（★ 重要）

### 13.1 结论

> **`robots/from_teleopbench/`（181 MB）用【普通 Git】入库，不使用 Git LFS。**

### 13.2 为什么不用 Git LFS

排查时发现一个关键事实：**Git LFS 的文件传输强制走 HTTPS**：
```
Endpoint=https://github.com/Wangxianyu835/teleoperation-system.git/info/lfs
                              ↑ HTTPS（不是 SSH）
```

而本项目网络环境的实测情况是：

| 通道 | 实测结果 |
|---|---|
| `github.com` 的 **HTTPS（443）** | ❌ **SNI 定向干扰**，直连超时；**必须开代理**才能通 |
| `github.com` 的 **SSH（22）** | ✅ **稳定可用，无需代理** |

**若用 LFS，代价是：**
- 你**和两位队友**都必须**常开 Clash 代理**才能 push/pull 模型
- 受 LFS 免费额度限制：**1 GB 存储 + 1 GB/月流量**
  （3 人各 clone 一次 ≈ 543 MB，**很容易超**，超额要付费买流量包）
- 队友还要额外安装 `git-lfs`、配置代理

**用普通 Git 的收益：**
- 走 SSH → **完全不需要代理** ✅
- 队友 **`git clone` 一步到位**，不需要 git-lfs ✅
- **无配额限制** ✅
- 代价：仓库体积 181 MB（远低于 GitHub **1 GB 警告线 / 5 GB 硬限**）

### 13.3 模型体积构成

| 子目录 | 体积 | 说明 |
|---|---|---|
| **`gr1`** | **117.4 MB** | GR1-T2（含 32 MB 的 `gr1.usd`）← **体积主体** |
| `h1_2` | 25.5 MB | H1-2 |
| `g1` | 17.4 MB | G1 |
| `unitree_hand` | 15.8 MB | |
| `inspire_hand` | 5.0 MB | |
| **合计** | **181.1 MB** | 512 个文件 |

### 13.4 `.gitignore` 的处理要点

只放行代码实际使用的 `from_teleopbench`，其余 7 个未引用的子目录（274 MB）继续排除：

```gitignore
robots/*
!robots/from_teleopbench/
```

> ⚠️ **不能用 `robots/` 整体排除** —— gitignore 规定「**父目录被排除后，无法再重新包含其子文件**」。
> 必须用 `robots/*`（只排除直接子项）+ `!robots/from_teleopbench/` 重新放行。

### 13.5 排查过程记录（供参考）

1. `git lfs version` → 已装（git-lfs 3.7.1，Git for Windows 自带）
2. `git lfs install` → 安装钩子成功
3. 创建 `.gitattributes`（按扩展名跟踪 stl/usd/glb/onnx/obj）
4. `git lfs track` → 18 条规则注册成功
5. `git check-attr filter -- ...` → 正确显示 `filter: lfs`
6. ⚠️ **测试 HTTPS**：直连 `github.com` 超时；**通过 Clash 代理返回 200**
7. → 结论：LFS 需要代理 + 有配额风险，**改用普通 Git**
8. 删除 `.gitattributes`，确认 `filter: unspecified`（LFS 规则已失效）

### 13.6 若将来必须改用 LFS

（例如模型膨胀到几十 GB）步骤：
```powershell
git lfs install
# 创建 .gitattributes 写 LFS 过滤规则
# 让 LFS 走代理：
git config --global http.https://github.com.proxy http://127.0.0.1:7897
# 然后重新 git add 目标文件（首次会被转成指针）
```

> ⚠️ **已在历史中的普通文件不会自动转成 LFS**，需要重写历史（`git filter-repo`），
> 且所有协作者都要重新 clone。**趁早决定比事后迁移便宜得多。**

---

## 14. 【2026-09-12】离线 Vision 流水线（契约 G/H + 回放脚本）

### 14.1 目标

实现 **Vision-based（单目视觉）的「离线」测试**：
```
队友B 采集手部数据  →  队友A 做重定向  →  仿真平台回放
```

**为什么离线优先**：可以「录好慢慢处理」、无性能压力、
**改算法不用重新采集**（最大优势）。论文评测协议本身就是先录制后评测，
所以**离线符合论文思路，不是简化版**。

### 14.2 新增两个契约

| 契约 | 内容 | 谁 → 谁 | 载体 |
|---|---|---|---|
| **G** | **人类手部数据文件** | 队友B → 队友A | `human_hand.h5`<br>`keypoints_3d (T,21,3)` + `keypoints_2d` + `wrist_pose` + `timestamps` |
| **H** | **动作序列文件** | 队友A → 仿真平台 | `actions.h5`<br>`actions (T,action_dim)` + attrs(robot_type/task_name/fps) |

> 📄 完整定义见 `docs/INTERFACE_CONTRACT.md`（契约 G/H）
> 与 `docs/OFFLINE_PIPELINE.md`（详细规范 + 分阶段计划）
>
> **核心动机**：如果不先定义**文件格式**就交换数据，**100% 接不上**。

### 14.3 新增脚本

| 脚本 | 作用 |
|---|---|
| **`scripts/replay_actions.py`** | ⭐ **离线回放**：读契约H 的动作文件 → 灌进 `SimulationEnv` → 输出结果 |
| `scripts/make_sample_data.py` | 生成契约 G/H 的**示例文件**（可执行的文档）|

```powershell
# 查看某机器人的动作空间定义（38/36/28 维的完整映射）
python scripts/replay_actions.py --describe --robot h1_2

# 用假数据验证链路（不需要真实数据）
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render

# 导出「契约H 示例文件」给队友A 照着写
python scripts/replay_actions.py --dummy --save-actions datasets/samples/actions_demo_h1_2.h5

# 回放真实数据（自动读取文件里的 robot_type / task_name）
python scripts/replay_actions.py --file datasets/actions/pushcube_h1_2.h5

# 带可视化（弹 pybullet 窗口）
python scripts/replay_actions.py --file xxx.h5 --render

# 生成契约 G/H 示例文件
python scripts/make_sample_data.py --kind all
```

### 14.4 实测验证（2026-09-12）

| 测试 | 结果 |
|---|---|
| `--describe --robot h1_2` | ✅ 打印完整 38 维动作空间映射 |
| `--dummy --robot h1_2`（480 步）| ✅ 回放成功，0.58s（**825 步/秒**）|
| `--file actions_demo_g1.h5` | ✅ 正确读取文件内 `robot_type=g1` / `task_name=pickcube` |
| 维度不匹配测试（h1_2 + 28维）| ✅ 正确抛 `ValueError`，提示期望 38 |
| `make_sample_data.py --kind all` | ✅ 生成 G/H 示例文件（h5）|

### 14.5 本轮顺带修复的 3 个问题

| # | 问题 | 症状 | 修复 |
|---|---|---|---|
| 1 | `import *` 不能放函数内 | `SyntaxError: import * only allowed at module level` | 改为 `import tasks.all_tasks` |
| 2 | **`env.action_dim` 构造后不存在** | `AttributeError: 'SimulationEnv' object has no attribute 'action_dim'` | `__init__` 加占位属性；调用方需在 `reset()` 后访问 |
| 3 | **任务注册表依赖调用方导入** | `KeyError: Task 'pickcube' not found. Available: []` | `simulation_env.py` **自己** `import tasks.all_tasks`；`get_task` 增加提示 |
| 4 | **print 非 GBK 字符直接崩溃** | `UnicodeEncodeError: 'gbk' codec can't encode character '\u2713'` | 全项目 **31 处** `✓`/`✗`/`⚠️` 替换为 `[OK]` / `[FAIL]` / `注意` |

> ⚠️ 问题 3 是**设计脆弱点**：30 个任务靠 `import tasks.all_tasks` 的**副作用**注册，
> 任何不导入它的调用方都会遇到「注册表为空」，而错误信息（`Available: []`）毫无提示性。
> 已改为 **`SimulationEnv` 自己负责导入**，从根上消除这类问题。

> 🔴 **问题 4 会直接坑到队友**：中文 Windows 控制台默认 **GBK（cp936）**，
> 而 `✓`(U+2713) / `✗`(U+2717) / `⚠️`(U+26A0) **不在 GBK 字符集**里，
> Python 一 `print` 就抛 `UnicodeEncodeError` **直接崩溃**。
> **队友在自己电脑上跑 `python test_import.py` 就会挂。**
>
> **修复方式**：写了扫描脚本找出全项目 41 处（去重后 31 处替换），
> 把符号换成 ASCII 安全写法，覆盖 9 个文件：
> ```
> 已修复   1 处  ./demo_teleop.py
> 已修复   1 处  ./envs/__init__.py
> 已修复   2 处  ./envs/robot_loader.py
> 已修复   4 处  ./envs/simulation_env.py
> 已修复   2 处  ./main.py
> 已修复   5 处  ./scripts/replay_actions.py
> 已修复   1 处  ./tasks/base_task.py
> 已修复   2 处  ./teleop/teleop_pipeline.py
> 已修复  13 处  ./test_import.py
> ```
> **验证**：`chcp 936` 模拟队友环境实测 → `EXIT=0`，不再崩溃；
> 全量编译 `OK=25 FAIL=0`。

### 14.6 下一步（队友可以开工了）

| 谁 | 任务 | 产出 |
|---|---|---|
| **队友B** | 按**契约G**采集 3~5 秒手部数据 | `datasets/raw/human_hand_xxx.h5` → commit |
| **队友A** | 按**契约H**输出动作序列 | `datasets/actions/xxx.h5`；可用 `replay_actions.py` 自测 |
| **你** | 收到数据后回放 | `python scripts/replay_actions.py --file xxx.h5` |

### 14.7 分阶段计划（来自 OFFLINE_PIPELINE.md）

- **阶段 1（先做）**：只做**手部**（MediaPipe 关键点 → dex-retargeting → 手部关节角）
- **阶段 2**：加上**手臂**（手腕 6DoF → IK(PINK) → 7 个手臂关节）
- **阶段 3**：对齐论文（SMPLer-X 替代 MediaPipe、PINK IK、卡尔曼滤波）

---

## 15. 【2026-09-12】接入队友的重定向输出（★ 目标手是 l21，不是 l7）

### 15.1 收到的数据

| 项 | 值 |
|---|---|
| 项目内路径 | **`datasets/raw/retarget_twohand_153542.h5`**（96 KB）|
| 原始来源 | `D:\xwechat_files\...\twohand_angles_153542_none_warmstart - 副本.h5` |
| 队友的模型 | `mytrans/checkpoint/models/twohand_h5/linker/none_warmstart/model_best.pth` |
| 说明 | **自训练模型**（不是论文的 dex-retargeting）；目标是 `linker` 灵巧手 |

文件结构：
```
left_angles / right_angles  (557, 18)  float32
left_valid  / right_valid   (557,)     bool     ← 无效帧 = 全零帧
timestamps                  (557,)     float64  0.174 ~ 18.99 s
attrs: checkpoint / input_file / left_coordinate_mode / output_shape=[18]
```

### 15.2 数据质量

| 项 | 结果 |
|---|---|
| 总帧 / 有效帧 | 557 / **512（91.9%）** |
| **有效帧范围** | **第 25 ~ 536 帧**（首尾各约 25 帧无效 + 全零）|
| 时长 / 帧率 | 17.00 s / 30.3 FPS |
| 有变化的维度 | 16 / 18 |
| ⚠️ 帧间跳变（>0.3 rad）| **34/511，最大 1.4634 rad（83.8°）** → **建议加平滑（卡尔曼滤波）** |

> ✅ `*_valid` 标记与全零帧**完全对应**，设计规范，可直接用于过滤。

### 15.3 ★★★ 关键发现：目标手是 `l21`，**不是** `l7`

用「**每个维度的实测范围** vs **每个关节的限位**」做**二分图最优匹配**（`scipy.optimize.linear_sum_assignment`），
遍历 LinkerHand 全部 15 个型号：

| 型号 | 关节数 | 可行匹配 | 越界维度 |
|---|---|---|---|
| **l21_left / l21_right** | **17** | **17/18** | **1** ← **最吻合** |
| g20 / l25 | 21 | 17/18 | 1 |
| l20_right | 21 | 14/18 | 4 |
| l10 / l10v6 / l10v7 | 20 | 11/18 | 7 |
| **l7_left / l7_right** | 17 | **11/18** | **7** ← **差很多** |

**决定性证据**（数据范围精确落在 l21 关节限位内）：

```
dim 13 [-0.5997, -0.5875] ↔ thumb_cmc_roll    (±0.600)   精确到 0.0003  ★
dim 14 [+0.0067, +1.5953] ↔ thumb_cmc_yaw     (0~1.600)  精确到 0.002   ★
dim 15 [+0.0002, +0.9549] ↔ thumb_cmc_pitch   (0~1.000)
dim  1 [+0.1771, +0.1800] ↔ index_mcp_roll    (±0.180)
dim  4 [-0.1258, +0.1798] ↔ middle_mcp_roll   (±0.180)
```

> ✅ **已于 2026-09-12 解决（队友确认统一到 L21）**：
> 项目里 `teleop/hand_interface.py` 原先用的是 **l7 的关节名**
> （`index_dip/middle_dip/ring_dip/pinky_dip`，且**没有** `*_mcp_roll` 侧摆关节），
> 而**队友的重定向目标是 `l21`**。现两边已统一：
>
> | 文件 | 原状态 | 现状态 |
> |---|---|---|
> | `teleop/hand_interface.py` | l7 关节名，只能命中 13/17 | ✅ 已改为 L21 的 17 个关节名<br>（新增 `L21_JOINTS` 常量 + 补齐缺失的 `apply_to_robot()`）|
> | `show_hand.py` | 加载 `l7_left` / `l7_right` URDF | ✅ 已改为 `l21_left` / `l21_right` |
>
> **验证**：17 项自检全 PASS —— `get_both_hands()` 的 17 个关节名与
> `l21_*.urdf` 的关节名**集合与顺序都完全一致**，且 `apply_to_robot()` 命中 17/17。
>
> 若将来要用**真实 L7 硬件**，再单独实现明确的 `L21 -> L7` 降维映射
> （会丢掉 4 个 `*_mcp_roll` 侧摆维度，需声明）。

### 15.4 推导出的映射（18 维 → l21 的 17 关节）

```
dim 0                → 占位（恒 0）
dim 1,2,3            → index:  mcp_roll, mcp_pitch, pip
dim 4,5,6            → middle: mcp_roll, mcp_pitch, pip
dim 7,8,9            → ring:   mcp_roll, mcp_pitch, pip
dim 10,11,12         → pinky:  mcp_roll, mcp_pitch, pip
dim 13,14,15,16,17   → thumb:  cmc_roll, cmc_yaw, cmc_pitch, mcp, ip
```

**顺序 = l21 的 URDF 顺序**（即「每 3 个一组 = (侧摆, MCP屈曲, PIP屈曲)」）

**交叉验证**：`scripts/replay_hand_angles.py` 加载 `linkerhand_l21_right.urdf` 后报告
```
可驱动的动作维度: 17 / 17
```
→ 映射表里的 17 个关节名在 URDF 中**全部命中** ✅
（若误按 l7 映射，会有关节名找不到，数量就会少于 17）

### 15.5 新增脚本：`scripts/replay_hand_angles.py`

```powershell
# ① 静态检查（维度 / 范围 / 越界 / 跳变）—— 不渲染
python scripts/replay_hand_angles.py --check --hand right

# ② GUI 可视化回放（弹 pybullet 窗口，能看见手指动）
python scripts/replay_hand_angles.py --hand right --render
python scripts/replay_hand_angles.py --hand both --render --smooth 5

# ③ 无头回放（自动验证，不弹窗）
python scripts/replay_hand_angles.py --hand both --headless-replay
```

### 15.6 实测结果（2026-09-12）

```
【静态检查】right   维度 18   总帧 557   有效帧 512
  映射检查: OK=17  越界=0  未映射=1
  帧间跳变(>0.3rad): 34/511   最大 1.4634 rad (83.8°)

【回放】l21_right
  已加载 linkerhand_l21_right.urdf：17 个关节
  可驱动的动作维度: 17
  回放完成：512 帧
  末帧关节角: index_mcp_roll +0.1794 / index_pip +1.1378 / middle_pip +0.6117 ...
结论: 所有映射维度都落在 l21 关节限位内  [OK]
```

### 15.7 ⚠️ 仍然存在的差距（待解决）

| # | 问题 | 说明 |
|---|---|---|
| 1 | **只有手指、没有手臂** | 契约H 需要 38/36/28 维（含双臂各 7），当前输出只有手部 |
| 2 | **维度 18 ≠ 契约H** | 需要「适配器」把 18 维映射成目标机器人的完整 `action` |
| 3 | ~~**手型号不一致**：队友目标是 **l21**，项目里手接口是 **l7**~~ | ✅ **已解决**（2026-09-12 队友确认统一到 L21；`hand_interface.py` + `show_hand.py` 均已改为 L21，17 项自检通过）|
| 4 | **跳变需平滑** | 34/511 帧 >0.3 rad，最大 83.8° → 按论文建议加**卡尔曼滤波** |
| 5 | 与论文不同 | 队友用的是**自训练模型**（`mytrans`）+ LinkerHand，论文是 **dex-retargeting** + Inspire/Fourier/Unitree 手 |

### 15.8 ✅ 视觉验证通过（2026-09-12）

**用 `--render` 打开 GUI 窗口，肉眼观察 `l21_right` 的回放，结果与预测完全一致。**

回放前的预测（基于数据统计）与实际观察：

| 部位 | 预测 | 实际 | 结论 |
|---|---|---|---|
| **四根手指** | 反复「握紧→张开」，幅度接近 90° | ✅ 一致 | 正确 |
| **四指侧摆** | 几乎不动（±0.18 rad 很小）| ✅ 一致 | 正确 |
| **大拇指** | 基本保持固定姿势，轻微动作 | ✅ 一致 | 正确 |
| **弯曲方向** | 向掌心（URDF 限位全为正值，不可能反关节）| ✅ 一致 | 正确 |

**这一次验证同时坐实了 4 件事：**

1. ✅ **映射推导 100% 正确** ——「维度范围 vs 关节限位」的二分图匹配结果是对的
2. ✅ **目标手型号确实是 `l21`**
3. ✅ **队友的重定向算法输出有效**（能驱动真实 l21 模型做合理抓握）
4. ✅ **回放脚本端到端工作正常**（数据 → 映射 → URDF → 渲染）

> 📌 **意义**：从「一个来历不明的 18 维 h5 文件」到「看得见的正确动作」，
> 中间没有任何人提供格式说明 —— 全靠**推理 + 交叉验证**打通。
> 这条**手部链路已可作为后续工作的可靠基础**。

---

## 16. 【2026-09-12】平滑滤波模块 + 两处重要修正

### 16.1 交付内容

**新增 `teleop/filters.py`：**

| 函数 / 类 | 用途 |
|---|---|
| `KalmanSmoother` | 常速度(CV)卡尔曼滤波（论文 §3.2.1 对齐）|
| `moving_average` | 移动平均（对照用）|
| **`detect_bad_frames`** | ⭐ 检测「整帧异常」坏帧 |
| **`repair_bad_frames`** | ⭐ 用前后帧线性插值修复坏帧 |
| **`classify_jumps`** | ⭐ 诊断「孤立尖峰（噪声）vs 成片真实运动」|
| `motion_metrics` | 抖动量化指标（maxd / jumps / rms_accel）|
| `compare_methods` / `print_metrics_table` | 多方法对比与打印 |
| `print_jump_report` / `print_bad_frames_report` | 报告打印 |

**`scripts/replay_hand_angles.py` 新增选项：**
`--analyze` `--compare` `--kalman` `--kalman-q` `--kalman-r` `--gate` `--no-repair` `--fps`

### 16.2 ★ 修正一：那 34 个「跳变」**不是噪声，是真实运动**

| 检查项 | 结果 |
|---|---|
| 跳变帧（单帧变化 > 0.3 rad）| 34 / 511（6.7%）|
| **孤立尖峰**（前后都小 + 下一帧弹回）| **0** |
| **成片快速运动** | **34（全部）** |
| 分布 | **前 8 秒完全没有**；全部集中在后 9 秒 |

**旁证（三条都指向同一结论）：**
1. 视觉验证时用户反馈「完美符合预期」、**未提到抽搐**
2. **Hampel 滤波（专门去尖峰）几乎无效** —— maxd 仅从 1.463 降到 1.264
3. 把卡尔曼的**创新门控设为 3** 时，**436/512 帧被判为离群点** → 误杀合法快动作，
   导致 maxd 从 1.46 **恶化**到 5.22

> ⚠️ **修正**：之前写的「34/511 帧跳变 → 建议加卡尔曼滤波」**不准确**，已更正。

### 16.3 ★ 修正二（已二次更正）：所谓「3 个整帧异常坏帧」是误报

> ⚠️ **2026-09-12 第 2 次更正**：本节原先断言「发现 3 个整帧异常坏帧、
> `*_valid` 漏标」，**这是错的**。队友核对后指出数据正常 —— **他是对的**。
> 完整更正说明见 `OFFLINE_PIPELINE.md` 第 8.8 节。以下是更正后的真实情况。

**原先的报告（错误）**：

```
left  : 帧 [497]          <-- 这是【压缩下标】，不是绝对下标
right : 帧 [497, 510]     <-- 同上
```

**错在两处**：

1. **下标错误**：`detect_bad_frames(arr[valid])` 在**压缩数组**上运行
   （剔掉了 47/45 个无效帧），返回的是**压缩下标**。
   换算成**绝对下标**是 `left[522]`、`right[522, 535]`。
   队友去查绝对下标 497/510，数据当然正常
   （范数 4.3940 / 4.3283，非零维数 17/18）。
   **SHA256 已核对：双方是同一个文件**（`65851D74...75A342`）。

2. **结论错误**：旧判据是「当前帧相对前后帧均值偏离 > 0.4 rad 的维度 >= 3 个」，
   这在**快速运动上必然误报**。决定性反例：

   ```
   left 单帧最大变化：中位数 0.0304 rad(1.7 度)，最大 1.4823 rad(84.9 度)
     432 -> 433 : 1.4823 rad (84.9 度)   <-- 没被命中
     434 -> 435 : 1.2825 rad (73.5 度)   <-- 被命中
   ```

   同一量级、命中与否纯属偶然 → 该判据**不是异常检测**。
   旧版把函数文档注释里描述的「典型场景」当成了判据名，
   **实际根本没有检测「塌零」**。

**更正后的真实情况**：

| 手 | 塌零坏帧 |
|---|---|
| `left` | **0 个** |
| `right` | **1 个：`right[535]`**（绝对下标）|

**`right[535]`**（`valid=True`）：

```
范数 0.8125，|v|<0.05 的维度数 = 10/18
534 -> 535 = 1.4546 rad (83.3 度)
535 -> 536 = 1.4634 rad (83.8 度)      <- 掉下去又弹回（孤立单帧）

三个侧摆关节（限位 ±0.18）一帧内翻到反侧：
  534: middle_roll=+0.1785  ring_roll=+0.1786  pinky_roll=+0.1790
  535: middle_roll=-0.1126  ring_roll=-0.1620  pinky_roll=-0.1027
  536: middle_roll=+0.1796  ring_roll=+0.1798  pinky_roll=+0.1798

frame_id = 535，t = 18.2880 s
```

→ 已请队友用**原始采集数据**（`visual_hand_data_20260912_153542.h5`）核对，
以区分责任在**采集端**还是**模型端**。**核对清楚前不做单方面断言。**

**已修复（本仓库）**：`teleop/filters.py` 的 `detect_bad_frames` **重写**为
「孤立塌零」判据（本帧 >= 10 维 `|v|<0.05`，且前后**有效**帧 <= 4 维），
**必须传完整数组 + `valid`、返回绝对下标**；3 处调用点同步修正；
`verify_hand_pipeline.py` 的验收项改为**检测器自检**
（注入塌零须精确检出 + 真实快动作须零误报）。

**修复效果**（线性插值，新判据下只作用于 `right[535]`）：

| 手 | rms_accel | max_delta |
|---|---|---|
| `right` | 72.30 -> **58.95** | 1.4634 -> 1.2640 |
| `left` | 79.39 -> **79.39**（不变）| 1.4823 -> **1.4823**（不变）|

> ★ 旧版错误数字是 `left 79.39 -> 76.00`、`right 72.30 -> 55.29` ——
> **旧版把真实快动作也一起"修"掉了**，`left` 的 `max_delta` 被压低就是证据。

### 16.4 平滑效果实测对比（队友数据，512 帧，30 FPS）

| 方法 | 最大单帧变化 | 跳变帧数 | rms加速度 | 与原始偏差 |
|---|---|---|---|---|
| 原始 raw | 1.4634 | 34 | 72.30 | 0.0000 |
| ma3（移动平均 3）| 0.5120 | 31 | 26.04 | 0.0281 |
| **ma7（移动平均 7）** | **0.2235** | **0** | 12.17 | **0.0503** |
| kf（q=1, r=0.01）| 0.4217 | 21 | 18.95 | 0.0874 |
| kf（q=1, r=0.1）| 0.2560 | 0 | 11.46 | 0.1255 |

**结论**：
1. **卡尔曼（CV 模型）在这份数据上不如移动平均** —— `ma7` 达到同样平滑度时，
   偏差只有 `kf` 的 **57%**
2. **原因**：CV 模型假设运动平滑，而数据含**真实的大幅快速运动** → 滤波器跟不上或滞后
3. **强滤波会破坏真实动作**（偏差 0.126 rad ≈ **7.2°**）

### 16.5 本项目的滤波策略

| 场景 | 建议 |
|---|---|
| **当前离线数据** | ✅ **不需要滤波**（只需修复坏帧）|
| 未来**实时遥操作**（MediaPipe / Vision Pro）| ✅ 用 `KalmanSmoother`，按实际噪声调 q/r |
| 论文对齐 | ✅ 模块已实现，报告里可写「采用论文的卡尔曼滤波方案」|

**`KalmanSmoother` 默认 `gate=0`（关闭创新门控）** —— 因为门控会误杀真实快动作。
仅当 `classify_jumps()` 判定为 `noise`（孤立尖峰为主）时才开启。

### 16.6 新增命令

```powershell
# ① 诊断"该不该滤波"
python scripts/replay_hand_angles.py --check --analyze

# ② 对比平滑方法
python scripts/replay_hand_angles.py --check --compare

# ③ 启用卡尔曼（默认温和参数 q=1, r=0.01）
python scripts/replay_hand_angles.py --hand right --render --kalman

# ④ 调参（更平滑但偏差更大）
python scripts/replay_hand_angles.py --hand right --render --kalman --kalman-r 0.1

# ⑤ 关闭坏帧自动修复
python scripts/replay_hand_angles.py --hand right --render --no-repair
```

### 16.7 新增 `scripts/check_gbk_safe.py`（提交前检查）

新写的代码又引入了 10 类非 GBK 字符（`U+2194 ↔ / U+2192 → / U+1D40 ᵀ / U+00B2² / U+00B3³ / U+25C0◀ …`），
已再次全项目清扫：**11 个文件、66 处**。

为避免再犯，把扫描脚本正式收进仓库：

```powershell
python scripts/check_gbk_safe.py            # 只报告
python scripts/check_gbk_safe.py --strict   # 有违规退出码 1（可接 CI）
```

> ⚠️ **教训**：中文 Windows 控制台下，代码里（尤其 `print()` 的字符串）
> **不要使用** `✓ ✗ ⚠️ ↔ → ← ᵀ ² ³ ◀ ▶ ≥ ≤ ≈` 等符号 —— 会直接崩溃。
> 替代写法见 `check_gbk_safe.py` 的 `SUGGEST` 表。



---

---

## 17. 【2026-09-12】方案B：把 l21 装到机器人腕部（★ 内含一个 pybullet 大坑）

### 17.1 交付内容

新增 `scripts/replay_hand_on_robot.py`：让画面里出现「论文机器人 + l21 灵巧手」，
比孤立的一只手更接近真实，也为将来接入手臂数据做准备（**手臂一动，手会自动跟随**）。

技术做法：

1. `RobotLoader` 加载机器人（`h1_2` / `gr1_t2` / `g1`）
2. 自动定位「手基座 link」（见 17.2 表）
3. 隐藏机器人自带的手（改为全透明，避免与 l21 重叠）
4. 用 pybullet **固定约束**（`createConstraint` + `JOINT_FIXED`）把 l21 锁到「手基座 link」
5. 每帧用队友的 18 维数据驱动 l21 的 17 个关节

### 17.2 各种机器人的「手基座 link」与腕部安装变换（实测自 URDF）

| 机器人 | 侧 | 手基座 link | 父级（腕部 link）| 腕部→手基座 xyz | rpy |
|---|---|---|---|---|---|
| `h1_2` | left | `L_hand_base_link` | `left_wrist_yaw_link` | `0.054 0 0` | `0 0 1.5708` |
| `h1_2` | right | `R_hand_base_link` | `right_wrist_yaw_link` | `0.054 0 0` | `3.14159 0 -1.5708` |
| `gr1_t2` | left | `l_hand_base_link` | `left_end_effector_link` | `-0.00144 -0.00006 -0.021` | `0 0 0` |
| `gr1_t2` | right | `r_hand_base_link` | `right_end_effector_link` | `-0.00142 0 -0.021` | `0 0 0` |
| `g1` | left | `left_hand_palm_link` | `left_wrist_yaw_link` | `0.0415 0.003 0` | `0 0 0` |
| `g1` | right | `right_hand_palm_link` | `right_wrist_yaw_link` | `0.0415 -0.003 0` | `0 0 0` |

隐藏自带手的数量：h1_2 = 26 个 link，gr1_t2 = 40 个，g1 = 16 个。

> 💡 这些三字型号**都把手基座单独做成了一个 link**，而 l21 的基座 link 也叫
> `hand_base_link` —— 命名语义天然对齐，所以「挂到哪个 link」这一步没有歧义。

### 17.3 ★★★ pybullet 大坑：`loadURDF` 是按「质心」摆放 `basePosition` 的

**这是本轮排查耗时最久的问题，务必记住。**

```python
p.loadURDF(手.urdf, basePosition=[0.286, 0.2095, 0.095], baseOrientation=orn)
p.getBasePositionAndOrientation(手)   # 实际返回 (0.2773, 0.2372, 0.1713) !!!
#                                                    ^^^^^^^^^^^^^^^^^^ 偏移 0.0816 m
```

对**自由刚体**（`useFixedBase=False`），pybullet 把 `basePosition` 当作
**base link 的质心（COM）**，而不是 link 原点。l21 的实测偏移：

| 手 | 偏移量 |
|---|---|
| `l21_left` | 0.0816 m |
| `l21_right` | 0.0763 m |

**为什么这个坑很隐蔽**：手的 base 位姿差了 8 cm，
用它当参考系去测量手的坐标系时，**手指方向会被算歪约 15 度**，
最终表现为「安装朝向不对，但 rpy 看起来又没错」。

**排查关键线索**：残差（左 15.16deg / 右 4.88deg）在 `h1_2` 与 `gr1_t2`
上**完全相同** —— 既然两种机器人算出完全一样的误差，
说明误差根本不在机器人侧，**而在 l21 侧**。

**修复**：

1. 测量手的坐标系时，用 `p.getBasePositionAndOrientation()` 读到的
   **实际** base 位姿当参考系，而不是 `loadURDF` 的传入值
2. `loadURDF` 之后立即 `resetBasePositionAndOrientation()` 到目标位姿
   （`reset*` 用的是 **link 原点** 约定，已验证精确），
   保证固定约束建立在正确位置上

### 17.4 自动安装朝向的原理（不用人工盲试）

「装得正不正」可以完全用几何推出来，不需要逐个试 rpy：

1. **测出机器人自带手的坐标系** `M_robot`（表达在该侧手基座 link 系里）
   - `e1` = 四指指尖均值方向（手指伸展方向）
   - `e2` = 拇指尖方向去掉 `e1` 分量后归一化（拇指侧）
   - `e3` = `e1 x e2`
2. **同法测出 l21 的坐标系** `M_l21`（表达在它自己的 `hand_base_link` 系里）
3. **安装旋转** `R = M_robot * M_l21^T`，按 `Rz(yaw)*Ry(pitch)*Rx(roll)`
   约定反解成 rpy（与 `p.getQuaternionFromEuler` 一致，已做往返自检，误差 1.2e-16）

实测参考值（脚本会自动算，这里仅供参考）：

| 机器人 | 侧 | 自动算出 rpy |
|---|---|---|
| `h1_2` | left | `(-1.7165, +0.3019, +2.9226)` |
| `h1_2` | right | `(-1.5872, -0.0602, -3.0305)` |
| `gr1_t2` | left | `(+3.0403, -0.3327, -1.6037)` |
| `gr1_t2` | right | `(-3.0901, +0.0000, +1.2792)` |
| `g1` | left | `(-3.0046, +1.3663, -2.9047)` |
| `g1` | right | `(+1.2687, +1.1753, +1.2329)` |

### 17.5 验证结果：6 / 6 全部精确对齐

脚本内置**安装后自检**（重新测量再比对），实测：

```
h1_2   left   手指夹角 0.000deg  拇指夹角 0.000deg  最大分量差 1.67e-07  [OK]
h1_2   right  0.000deg  0.000deg  9.94e-08  [OK]
gr1_t2 left   0.000deg  0.000deg  1.27e-07  [OK]
gr1_t2 right  0.000deg  0.000deg  1.14e-07  [OK]
g1     left   0.000deg  0.000deg  2.34e-07  [OK]
g1     right  0.000deg  0.000deg  1.34e-07  [OK]
```

6 个组合的 `det(R)` 均为 `+1.0000`（**没有左右手镜像问题**）。

### 17.6 用法

```powershell
# 基本用法（朝向自动算）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand both --render
python scripts/replay_hand_on_robot.py --robot gr1_t2 --hand both --render
python scripts/replay_hand_on_robot.py --robot g1 --hand both --render

# 无头（自动验证，不弹窗）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand both

# 手动覆盖朝向（非零时优先于自动值）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand right --render `
    --mount-offset 0 0 0.05 --mount-rpy 0 0 1.5708

# 关闭自动朝向，完全手动
python scripts/replay_hand_on_robot.py --robot h1_2 --hand both --render --no-auto-mount
```

### 17.7 本轮顺手修掉的两个 pybullet API 细节

1. **`getJointInfo` 不返回 child link 索引** —— `info[12]` 是 child link 的
   **名字**（bytes），且树形结构中「关节 `i` 的 child link 索引就是 `i` 本身」。
   （`info[16]` 是 *parent* link 索引）
2. **link 侧别命名有两套**：`h1_2` / `gr1_t2` 用 `L_*` / `R_*`，
   `g1` 用 `left_*` / `right_*`。匹配时必须兼容，否则会「一个都没匹配上」
   （表现为「已隐藏 0 个 link」，但**不报错**，很隐蔽）。







---

## 18. 【2026-09-12】★ 决策：仿真采用机器人【原装手】（不换手）

### 18.1 决策与理由

**在仿真里使用机器人自带的灵巧手**，不把 LinkerHand l21 装到腕部。

| | **采用** | 未采用（保留备用）|
|---|---|---|
| 方案 | **原装手 + L21 数据降维映射** | 把 l21 装到腕部 |
| 脚本 | `scripts/replay_hand_native.py` | `scripts/replay_hand_on_robot.py` |
| 映射模块 | `teleop/native_hand.py` | —— |

**理由**：
1. 机器人保持**完全原装**，与论文的 H1-2 基线更可比
2. 使用者反馈：换成 LinkerHand **一直"手抖"**（根因见 18.3；虽已修，但增加了复杂度）
3. 原装手的运动学/动力学参数是厂商调过的，不需要额外治理

**代价（必须在报告里声明）**：这是**有损**映射

| 机器人 | 可表达 | 覆盖率 | 丢弃的自由度 |
|---|---|---|---|
| **H1-2** | **12/17** | 71% | 4 个 `*_mcp_roll`（侧摆）+ 拇指 `cmc_roll` |
| GR1-T2 | 11/17 | 65% | 再丢拇指 1 个 |
| G1 | 7/17 | 41% | 另丢无名指/小指整根 |

### 18.2 映射表（`teleop/native_hand.py`）

原则：**语义 1:1，不做无依据的"合并"**

| L21 数据维度 | 原装手关节 | 语义 |
|---|---|---|
| `*_mcp_pitch` | `proximal` | 近端指节屈曲 |
| `*_pip` | `intermediate`（GR1-T2 拇指为 `distal`）| 远端指节屈曲 |
| `*_mcp_roll` | **丢弃** | 原装手没有侧摆自由度 |
| 拇指 `cmc_yaw` / `cmc_pitch` | `proximal_yaw` / `proximal_pitch` | 基座偏摆 / 俯仰 |
| 拇指 `mcp` / `ip` | `intermediate` / `distal` | 指节屈曲 |

**★ 符号方向自动判定**（不处理的话手指会**反着弯**）：

```
H1-2   [0.000, +1.700]   屈曲为正  -> sign = +1
GR1-T2 [-1.570, 0.000]   屈曲为负  -> sign = -1
G1     left 负 / right 正（左右手还不一样）

规则：sign = +1 if |hi| >= |lo| else -1
```

### 18.3 「手抖」根因（已查明；两个方案都受益）

l21 的 URDF 在 pybullet 里**动力学参数退化**：

```
base link 质量 = 1.5785e-07 kg         惯量对角 = (0.0, 0.0, 0.0)
各指节质量     = 0.0004 ~ 0.003 kg     关节 damping = friction = 0
（而 URDF 自己声明 effort=100, velocity=1 —— 作者是按真实伺服设计的）
```

→ 位置控制的增益相对这么小的惯量**过大** → 过冲 / 数值发散 → 肉眼就是"手抖"。
（实测：不加阻尼电机追不上，稳态误差 0.4380；加了阻尼反而数值爆炸，速度 RMS 0.003 → 29）

修法（`replay_hand_on_robot.py` **已默认开启**）：

```python
p.changeDynamics(h, j, mass=0.02, localInertiaDiagonal=[1e-6]*3)
p.setTimeStep(1/1000)
```

实测：稳态误差 0.4380 → **0.0007**，超调 0.4800 → **0.0000**。

> 注：质量/惯量重标只是为了让控制稳定，**不代表真实质量**；
> 将来做接触/抓取实验时需要重新评估。

### 18.4 ★★ 顺带发现的普遍问题：仿真时间 ≠ 数据时间

原来**所有**回放脚本都是**每帧只调一次 `stepSimulation()`**：

```
dt = 1/240  ->  每帧推进 4.17 ms，而数据是 33 ms/帧
             ->  仿真以 1/8 速度慢放，关节必然滞后
```

现在两个脚本都支持 `--substeps`：

| 脚本 | 默认 |
|---|---|
| `replay_hand_native.py` | 自动 = `round(33ms / (1/240))` = **8 步** |
| `replay_hand_on_robot.py` | **8 步**（`--substeps 0` 走自动）|

**验收实测**：给足步数后跟踪误差 **0.0018 rad**；不给（20 步/采样）则 **0.1878 rad** —— 差 100 倍。

### 18.5 新增/修改的文件

| 文件 | 说明 |
|---|---|
| `teleop/native_hand.py` | **新增**：原装手映射模块（映射表 + 符号判定 + `map_frame` 等工具）|
| `scripts/replay_hand_native.py` | **新增**：原装手回放（GUI / 无头 / `--report`）|
| `scripts/verify_hand_pipeline.py` | 新增验收项 **[6]**：三类机器人覆盖率 + 真回放跟踪误差 |
| `scripts/replay_hand_on_robot.py` | 修手抖 + `--dt` / `--substeps` / `--no-fix-jitter` |

### 18.6 用法

```powershell
# 原装手（本项目主用）
python scripts/replay_hand_native.py --robot h1_2   --hand both --render
python scripts/replay_hand_native.py --robot gr1_t2 --hand both --render
python scripts/replay_hand_native.py --robot g1     --hand both --render

# 只看映射报告（不开仿真）
python scripts/replay_hand_native.py --robot h1_2 --hand both --report

# 一键验收（含原装手映射 + 真回放跟踪误差）
python scripts/verify_hand_pipeline.py
```

### 18.7 仍然存在的限制（未变）

**手臂不会动** —— 队友的数据里**只有手部 18 维**，没有手臂关节角、也没有手基座位姿。
**这与用哪只手无关。** 要显示完整机器人的动作，需要队友补：

1. **双臂各 7 个关节角**（最直接），或
2. 至少 **hand_base 的世界位姿**（可用 IK 反解手臂）

### 18.8 ★★ 坏帧 `right[535]` 的真正根因：MediaPipe 左右手身份切换

> 队友（重定向算法作者）核对**原始关键点**后给出根因，我们**用角度数据独立验证通过**。

**事件**：`right_angles[535]` 单帧塌陷（范数 0.8125，10/18 维接近 0），前后帧正常。

**根因：不是模型输出坏值，而是 MediaPipe 把左右手标签搞混了。**

队友从原始关键点找到的证据：

```
frame 534:  left 中心 x ≈ 0.731    right 中心 x ≈ 0.304
frame 535:  left 全零（消失）       right 中心 x ≈ 0.747

R535 vs L534 的 RMSE = 0.0216   <- 非常像
R535 vs R534 的 RMSE = 0.2654   <- 不像
```

**我们用角度数据独立验证（不需要原始关键点）：**

| 验证项 | 结果 |
|---|---|
| `left_valid[535]` 是否为 False | ✅ **是**（`[536]` 也是 False；537 起两侧都无效 = 录制结束）|
| `right[535]` 最像谁 | ✅ 除自己外**第二像就是 `L[534]`（RMSE 0.0441）**，前 5 名里有 4 个是 `L[53x]` |
| 对照：`right[534]` 的邻居 | 全是 `R[53x]`（0.0366 / 0.0455 / 0.0618）—— 正常的右手轨迹 |
| `right[536]` 是否也可疑 | ✅ **是**（与 `R[534]` 的 RMSE 0.316，正常相邻帧只有 ~0.03）|

**所以 `right[535]` 和 `[536]` 实际都在跟踪之前的左手。**

**为什么原来的 valid 检查抓不到**：采集侧只判断
`np.isfinite(points).all() and not np.all(points == 0)` ——
只要关键点非零且无 NaN 就是 valid，**身份跳变完全检不出来**。

#### 修复（本仓库已实现）

`teleop/filters.py` 新增：

| 函数 | 作用 |
|---|---|
| `detect_identity_swaps(L, R, LV, RV)` | 检测身份切换，返回绝对下标 + 证据 |
| `hold_last_valid(seq, bad, valid)` | 用【保持上一有效姿态】替代插值 |
| `repair_stream(...)` | 组合：身份切换帧 hold、塌零帧插值（**顺序有讲究**：先 hold 改干净，后续插值才会用到正确邻居）|

**判据**（不需要原始关键点）：

```
d_same  = RMSE(X_s[t], X_s[t-1])        本侧突变
d_cross = RMSE(X_s[t], X_other[t-1])    与另一侧前一帧的距离
条件：d_cross < 0.5 * d_same  且  d_same > 0.05  且  另一侧本帧【消失】
```

**★ 为什么必须加「另一侧消失」这个门控**：只用 RMSE 比值会**误报 7 处**
（345/346/353/378/383/429 —— 那些都是之前确认的**真实快速运动**）。
加上门控后**只命中 535，0 误报**。

**★ 为什么必须用「保持」而不是「插值」**：`right[536]` 同样受污染，插值会被它拉偏：

```
原始          范数 0.8125
插值 后        范数 2.3135   <- 被 536 拉偏
保持 534 后    范数 2.1213   <- 与帧 534 完全相同（正确）
```

**验收新增第 [3b] 项**（`verify_hand_pipeline.py`）：
检出数必须为 1 且是 `right[535]`，同时证明关掉门控会误报 7 处。
一键验收现为 **7/7 PASS**。

**给数据提供方的建议（根治要放在采集/加载侧）**：

1. 检测**手掌中心瞬移** + **左右身份交换**
2. 发生跳变后**清空 3 帧历史**，等重新积满 3 帧再恢复 `valid=True`
3. 按此规则，`535`、`536` 都会保持上一有效姿态

> 这一条已经写进 `OFFLINE_PIPELINE.md` 第 8.12 节，可直接转给队友。

---

## 19. 【2026-09-12】阶段 1 验收通过 + 范围锁定（只做手部）

### 19.1 用户验收结果

> 「运行后，那个机器人的表现完全正常。」

**阶段 1（手部链路）验收通过**。画面为：

- **H1-2 原装整机**（头 / 躯干 / 双臂 / 双腿全在，相机自动取景）
- **两只原装 Inspire 手**按队友数据屈伸
- 手臂保持中性姿态**静止**（数据里没有手臂关节）
- 机器人头顶 HUD：型号 / 数据帧号 / 映射覆盖率

**一键复现**：

```powershell
python scripts/replay_hand_native.py --robot h1_2 --hand both --render --view full --loop 0
```

### 19.2 范围锁定：手臂数据暂不可得

用户确认：**重定向算法目前还无法输出手臂数据**，因此**先只做手部**。

| 项 | 状态 |
|---|---|
| 手指驱动 | ✅ 已实现并验收（原装手 12/17 = 71%）|
| 手臂驱动 | ⏸️ **暂停** —— 队友算法暂未输出手臂关节角 |
| 任务级评测（论文 30 个任务）| ⏸️ 依赖手臂数据 |

**一旦拿到双臂各 7 个关节角，现有代码可直接接上**：
`RobotLoader` 已识别各机器人的 7 个手臂关节（`_ARM_KEYWORDS`），
且两个回放脚本里已经**把手部关节与其余关节分开处理**
（手部由数据驱动、其余用位置控制锁住），补一张手臂映射表即可。

### 19.3 新增：三台机器人并排演示（答辩 / 报告用）

`scripts/show_hands_all.py`：H1-2 / GR1-T2 / G1 **并排站立**，
各自用**原装手**按**同一份数据**同步屈伸 —— 一张图就能说明
「三种论文机器人都跑通了手部链路」。

```powershell
python scripts/show_hands_all.py --render --loop 0
python scripts/show_hands_all.py --render --view front --loop 0
```

实测输出：

```
身份切换检测：1 处 -> [535]（用 hold 修复，不用插值）
  h1_2     摆位 x=-2.4   手部 L 12/17(71%)  R 12/17(71%)   锁住 27 个非手部关节
  gr1_t2   摆位 x=+0.0   手部 L 11/17(65%)  R 11/17(65%)   锁住 32 个非手部关节
  g1       摆位 x=+2.4   手部 L 7/17(41%)   R 7/17(41%)    锁住 27 个非手部关节
```

**相机按 16:9 窗口自动取景**（三台横向跨度 5.24 m）。
注意：用正方形的 60 度 FOV 估距离会退得太远、机器人显小，
所以按**水平 / 垂直两个约束取更远的那个**。

### 19.4 阶段 1 交付物清单

| 文件 | 作用 |
|---|---|
| `teleop/native_hand.py` | **L21 -> 原装手 降维映射**（正式接口，可复用）|
| `teleop/filters.py` | 塌零检测 / 身份切换检测 / hold / 插值 / 卡尔曼 |
| `scripts/replay_hand_native.py` | **单台机器人**手部回放（整机取景 + HUD + 循环 + 4 视角）|
| `scripts/show_hands_all.py` | **三台并排**手部回放（答辩展示）|
| `scripts/replay_hand_on_robot.py` | 备用：把 l21 装到腕部（无损，需声明换手）|
| `scripts/verify_hand_pipeline.py` | 一键验收（**7 / 7 PASS**）|
| `docs/OFFLINE_PIPELINE.md` 8.11 / 8.12 | 原装手决策 + 身份切换根因（可转队友）|

### 19.5 写报告时可直接引用的一句话

> 本项目在 PyBullet 中复现了 TeleOpBench 的三种机器人（H1-2 / GR1-T2 / G1），
> 并将重定向算法输出的 18 维手部动作**降维映射到各机器人自带的灵巧手**上
> （覆盖率 12/17、11/17、7/17），可在仿真中实时回放；
> 手臂部分因重定向算法暂未输出对应数据，按已确认的阶段范围保持中性姿态。
> 数据侧另发现并修复了 MediaPipe 左右手身份切换导致的单帧污染（详见 18.8 节）。

---

## 20. 【2026-11-10】两周冲刺计划：论文复现收口（11-10 → 11-23）

> **本节由用户 2026-11-10 明确要求写入**：「按照我们的进度和这个论文复现的要求在日志里写一个规划，要详细一点，1-2 周内完成所有的内容，越早越好。」
> 执行口径：**每天只问一句 —— 今天结束时，交付物里多了哪一个可复核的文件？** 没有文件的「完成」不算完成。
> 依赖的既有结论：本文 §2（论文要什么）、§7.3 / §7.4（覆盖度与缺失组件）、§9（待办）；
> `docs/task-evaluation.md`（批次 1/2/3 的实测数字与批次 4 的输入）。

### 20.0 一句话目标

**2026-11-23（周一）之前，拿出一张「同一批 10 个任务 × 同一套判据 × Vision / VR 两条接口」的成功率 + 完成时间表，并附一份写清边界的偏差声明。**

论文的 4 条遥操作接口里，**MoCap / 外骨骼两条没有硬件**，用同格式的**代理档**（离线动作回放）占位并标 `proxy` —— 这个边界必须在动手前讲清（见 20.3 第 1 条），否则「两周做完论文」是不成立的承诺。

### 20.1 起点：现在到底在哪（2026-11-10 盘点，全部有据可查）

| 面 | 现状 | 证据 / 出处 |
|---|---|---|
| 平台 | PyBullet + 三台机器人（h1_2 / gr1_t2 / g1）并排，手臂 7+7 可独立驱动 | `applications/three_robot_scene.py` + `simulation/arm_scene_driver.py` |
| 手部 | 18 维 NN 重定向（共享权重 `3eb1dd8`）-> 原装手 12/11/7 DoF 映射，端到端验收通过 | `RETARGETING_PIPELINE.md` §4.6 / §4.7 |
| 手臂（VR 线） | Vision Pro 三机器人同帧跑通；今天定位并修掉「腕限位噪声把整臂冻结」 | 分支 `fix/limit-clamp-and-guard-margin`，commit `d9fb470`（见 20.1.1）|
| 手臂（视觉线） | MediaPipe pose -> 相对重定向 -> 每侧 7 DoF IK（腕 + 肘双位置任务）；拧腕只动腕（§8.18），`--arm-ik-mode pose` 让三台一起转腕 + `wrist_usage` 判定「腕到限位」（§8.19）；掌面基丢帧时短暂保持（`--arm-palm-hold-ms`，防「换基抽搐」）+ 腕姿态目标角速度上限（`--arm-orientation-rate-deg-s`，防单帧坏帧与换基那一跳）（§8.20） | `multi-robot-arm.md` §8.15-§8.20 |
| 任务（批次 1） | 30 任务可评测性审计：**19/30 通过**；三台机器人 **0/57** 个可搬物体在手臂可达半径内 | `task-evaluation.md` §3 |
| 任务（批次 2） | 场景层落地：**72/90 格可评测**（每台 24 OK / 3 CRITERIA_INVALID / 2 CRITERIA_ABSOLUTE / 1 SCENE_UNSUPPORTED） | `task-evaluation.md` §4.3；`devtools/task_eval.py --verify --robot all` |
| 任务（批次 3） | `--policy camera` 现场档已接：链路通（238 帧 / `joint_travel_rad` 1.16），**任务未解**（pushcube 需要 92% 臂展） | `task-evaluation.md` §4.6 |
| 任务（批次 4） | **未开始**：还没有「10 任务 × N trial」的成功率 / 完成时间表 | `task-evaluation.md` §2 的批次表 |
| 论文覆盖度 | 4 条接口覆盖 **2/4**（Vision 近似、VR 部分）；PINK / dex-retargeting / SMPLer-X 未接 | 本文 §7.3 / §7.4 |

#### 20.1.1 今天（11-10）已定案的一处修复（写报告可直接引用）

- **现象**：三机器人 VR 同帧跑 900 帧，209 次 `nonfinite_or_limit` 拒绝**全部**落在腕关节，超限量级 0.1-7.8 mrad（纯数值噪声），后果是整条手臂被 hold 住 —— 现场看就是「手臂冻结」。
- **处理**：`arm_ik.py` 加 `ARM_LIMIT_TOLERANCE = 1e-2`，三处共用（`solve()` 门槛 / `_arm_values()` 裁剪 / 新增 `_limit_violation()` 诊断，结果记在 `ik.limit_violation`）。
- **实测**（headless 复跑，同一份录制）：拒绝 **209 -> 2**；h1_2 拒动帧 371/381 -> 303/318；位置误差 p50 **4.35/4.91 -> 3.00/3.95 mm**；姿态 p95 **17.24 -> 11.65 度**（h1_2 左）。
- **回归**：`python -m unittest discover -s tests` -> `Ran 612 tests / OK (skipped=5)`，退出码 0。

#### 20.1.2 本仓今天的基线实测（改前状态，规划里的"起点"必须是真的）

| 命令（`cd F:\simulation_platform`，`$env:PYTHONPATH="$pwd\src"`） | 退出码 | 结果 |
|---|---|---|
| `python -m teleoperation --help` | 0 | 正常出帮助 |
| `python -m unittest discover -s tests` | 0 | **Ran 397 tests / OK (skipped=3)**，63.9 s，日志 `outputs/baseline_1110/unittest.log` |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1` | **1** | **2 项 FAIL**（详见下），日志 `outputs/baseline_1110/preflight.log` |

两项 FAIL 的真实成因（今天已查明，属于**分支状态**，不是本轮改动引入）：

1. **`rule 3` 49 个队友文件相对 `10ebd9f` 有 M/D** —— 本分支相对 `10ebd9f` 有 **76 个提交**，其中 `src/teleoperation` 60 处差异（`A` 与 `M` 混排）。判据本身没被违反，是**基线选错了**：`preflight.ps1` 默认 `-Base 10ebd9f`，而本线的分叉点在更后面。D1 要做的不是"还原 49 个文件"，而是**用 `-Base <本线分叉点>` 重跑并把结论写进 CONVENTIONS**。
2. **根目录 7 个旧布局空壳目录**（`retargeting/` `model/` `teleop/` `input_adapters/` `envs/` `tasks/` `scripts/`）—— 实测**除 `__pycache__/` 外 0 个文件**，是旧布局删源码后留下的空壳；删除不丢任何源码（可先用 `-WhatIf` 确认）。

> 这两项直接决定后面两周"改前/改后跑同一套"能不能成立（约定 5），所以排在 D1 第一件。


### 20.2 终点：11-23 必须拿出的 5 件交付物

| # | 交付物 | 判据（可复核，不靠"看起来对"） |
|---|---|---|
| 1 | **论文 Table 2 样式的评估表**：10 个代表任务 × {Vision, VR} × N trial，成功率 + 完成时间 | 表格由脚本从 trial JSON **自动生成**，不是手填；每格标明 trial 数 |
| 2 | **失败归因表**：每任务失败原因分类（够不着 / 判据不可触发 / 场景不稳 / 人被遮挡 / 超时 / 其他） | 与批次 1/2 的 18 格排除项对得上，不重复计入成功率分母 |
| 3 | **观测记录样例**：状态向量 + 第一视角 + 第三视角 + 物体位姿，一条 trial 一份 H5 + 一段视频 | 能被 `docs/contracts.md` 的读取器读回，帧数 = trial 的 `steps` |
| 4 | **一键复现入口**：一条命令从原始 JSON 重算整张表（不重新采集） | 退出码 0，产物落 `outputs/paper_eval/`，两次重算逐字相同 |
| 5 | **偏差声明 + 报告章节草稿**（4 接口覆盖 2/4、仿真器差异、被试人数） | 可直接粘进大创报告，§20.9 有模板 |

### 20.3 范围与硬边界（这两周**不做**什么）

1. **不采购 Xsens / HOMIE 外骨骼**：MoCap 与外骨骼两条接口用 `--policy replay`（离线动作回放）占位，表格里标 `proxy`，**不参与"四条接口横向对比"的结论**。
2. **不迁 Isaac Sim**（约定 6 已冻结）：仿真器差异写进偏差声明（§20.9）。
3. **不改 `src/teleoperation/**`**（约定 2/3）：任务判据 / 场景摆法的修补一律走 `devtools/task_eval.py` 的场景层；队友代码的问题进问题清单，交回队友。
4. **不追 4 名被试**：先 1 人 × 3 trial × 10 任务跑通，再补第 2 人；n < 4 在表头注明（论文是 4 人，这点必须写在偏差里）。
5. **不做 Loco-manipulation / 触觉反馈**（论文自己的 future work）。

### 20.4 逐日排期（D1 = 2026-11-10 周二，D14 = 2026-11-23 周一）

> 每格三段：**做什么 / 当天必须落地的文件 / 验收命令与判据**。
> 标 **现场** 的需要人站在摄像头前或戴 Vision Pro，其余全部可无头复算。
> 现场时段集中在周末（D6 / D8）与周四（D10），工作日留给代码与文档 —— 这是"越早越好"的关键：**把等人拆成两天，不要每天等**。

| 天 | 日期 | 内容（做什么） | 当天必须落地的文件 | 验收命令 / 判据 |
|---|---|---|---|---|
| **D1** | 11-10 周二 | ⓪ **先让基线可判**：删 7 个旧布局空壳目录、把 `preflight.ps1 -Base` 改到本线分叉点（见 20.1.2）；① 三个仓收敛成一条交付主线（见 20.4.1）；② 今天的腕限位修复合进主线；③ 本节入库；④ 跑全量基线 | 本节 + 合并后的主线分支 + `outputs/baseline_1110/*.log` | `python -m unittest discover -s tests`（不劣化，今天 397 OK）+ `preflight.ps1` **退出码 0**（今天 1 项 rule 3 + 1 项空壳目录 = 2 FAIL） |
| **D2** | 11-11 周三 | 批次 3 收口：pushcube 端到端（真摄像头），先解决"够得着"（台面再收 / 物体前移 / 或换 `uprearcup`） | `outputs/task_eval/camera_pushcube.json` + 快照 PNG + 60 s 录屏 | `task_eval.py --task pushcube --policy camera --render --preview --max-wall-s 60`；判据 `joint_travel_rad > 1.0` 且 `camera.hand_frames > 0` |
| **D3** | 11-12 周四 | 给 `task_eval.py` 加 **`--policy vr`** 现场档（Vision Pro 录制 -> 同一 trial runner），与 camera 档做 pushcube A/B | `devtools/task_eval.py`（加档位，不新建文件）+ `outputs/task_eval/vr_pushcube.json` | 两档同任务各 3 trial，`success` / `completion_time_s` 都落盘；`tests/test_task_audit.py` 加 1 个 VR 档用例 |
| **D4** | 11-13 周五 | 10 个代表任务逐个过"能不能解"（idle / dummy / camera 各 1 trial），把"够不着""判据不可触发"钉死 | `outputs/task_eval/solvable_10.json` + 问题清单（§9 + `task-evaluation.md` §4.5） | 10 任务 × 3 策略 = 30 条 trial 全部落盘，每条带 `scene.in_reach` 与 `final_objects` |
| **D5** | 11-14 周六 | trial runner 增强（`--trials N` / 超时 / 失败归因分类）+ 汇总器 `--summary`（多 JSON -> markdown + csv） | `devtools/task_eval.py`（增强）+ `outputs/task_eval/summary_demo.md` | `task_eval.py --task pushcube --trials 3 --summary` 退出码 0；同一批 JSON 重算两次逐字相同 |

| 天 | 日期 | 内容（做什么） | 当天必须落地的文件 | 验收命令 / 判据 |
|---|---|---|---|---|
| **D6** | 11-15 周日 **现场 ~2 h** | **采集 A —— Vision 接口**：10 任务 × 3 trial，1 名被试；每个任务前先等 `cal=ok` | `outputs/paper_eval/vision_<task>_<n>.json` + 每 trial 1 张快照 | 10 任务全部有 `success` 记录（成功失败都留）；采集日志无连续 > 30 帧 `held_reason=no_frame` |
| **D7** | 11-16 周一 | 数据清洗 + 第一版表（Vision）；"假成功 / 判据不可触发"的格子标 `criteria_invalid` 并从分母剔除 | `outputs/paper_eval/table_vision.md` + `.csv` | 每格 = 3 trial；剔除项与批次 1/2 的 18 格对得上 |
| **D8** | 11-17 周二 **现场 ~2 h** | **采集 B —— VR 接口**：同 10 任务 × 3 trial、同一名被试、同一顺序（减少学习效应） | `outputs/paper_eval/vr_<task>_<n>.json` | 同 D6 判据 |
| **D9** | 11-18 周三 | 两接口对比表 + 与论文"Vision 在 Task 4 / Task 7 明显差"的结论对照（我们是否同方向） | `outputs/paper_eval/table_two_iface.md` + 一页对照分析 | 表里每个差值都能追到 trial 级原始数据 |
| **D10** | 11-19 周四 **现场 30-40 min** | 补第 2 名被试（10 任务 × 1 trial，够做趋势）+ Wilson 95% 区间 | `outputs/paper_eval/table_subject2.md` | n=2 写明；区间计算函数进 `task_eval.py`（不新建文件） |
| **D11** | 11-20 周五 | 观测记录对齐论文（(a) 状态向量 (b) 第一视角 + 第三视角 (c) 物体位姿），导 H5 + 拼视频 | `outputs/paper_eval/obs/<trial>.h5` + `outputs/paper_eval/videos/*.mp4` | H5 用 `docs/contracts.md` 读取器读回，帧数 = trial 的 `steps` |
| **D12** | 11-21 周六 | 偏差声明 + 三处文档同步（本文件 / `task-evaluation.md` / `RETARGETING_PIPELINE.md`）+ 报告章节草稿 | `docs/task-evaluation.md` 新增"两周结果"节 + `outputs/report/` 草稿 | `python devtools/check_gbk_safe.py --strict` 退出码 0；§7.3 / §7.4 覆盖度表按实际更新 |
| **D13** | 11-22 周日 | 全量回归 + 一键复现（从原始 JSON 重算表）+ 清理 354 MB smoke 权重等遗留物 | `outputs/baseline_1122/*.log` + 清理记录 | `preflight.ps1 -Full` 退出码 0；重算表与 D9 的表逐字一致 |
| **D14** | 11-23 周一 | 交付（表 + 图 + 视频 + 数据包 + 报告段落）、打 tag、写 handoff | `outputs/paper_eval/`（冻结）+ 本文件 §20.10 回填 | 20.2 的 5 件交付物齐；tag 可回溯到当天 HEAD |

#### 20.4.1 D1 的分支收敛（**必须先做**，否则两周后又是三份代码）

现状：三个工作副本各有一段**别人没有**的提交，且都不完整。

| 副本 | 独有内容（截至 2026-11-10） | 处理 |
|---|---|---|
| `F:\sim_wt_wrist` | 腕限位容差修复 `d9fb470`（已推 `fix/limit-clamp-and-guard-margin`）+ 612 条用例；三机器人 VR 逐帧分析产物 | 合并进交付主线 |
| `F:\simulation_platform` | 任务评估批次 1/2/3（`c926b42` / `ed85d87` / `f008869`）+ `docs/task-evaluation.md` + 本文件 | 合并进交付主线 |
| `F:\simulation_platform_cs` | `pose_body.py`、`reference_task_controller.py`、`run_task_eval.py` 等（**大量未提交改动**） | 先提交，再作为主线候选 |

主线建议（**需要用户拍板**）：以 `F:\simulation_platform_cs` 新建 `release/paper-eval` 作为交付主线，另两个副本只作来源分支；合并后立刻跑一次全量回归。

> **为什么不合并就不行**：这两周里每修一个 bug 都要在 2-3 个副本各改一遍，且"哪份是最新"会反复说不清 —— 这是本计划**最大的进度风险**，比任何技术点都更容易拖垮排期。

### 20.5 里程碑与 gate（不过 gate 就砍范围，不许延期硬扛）

| 里程碑 | 时间 | 内容 | gate 条件 |
|---|---|---|---|
| **M1** | D3 末（11-12） | pushcube 双接口可跑 | 若 VR 档 11-12 没通 -> 用已录 900 帧离线回放当 VR 档（标 `proxy`），不阻塞 M2 |
| **M2** | D7 末（11-16） | Vision 接口 10 任务 × 3 trial 表 | 若某任务物理不可解 -> 从批次 2 全 OK 的 24 个里换一个，并在报告写明换了哪个 |
| **M3** | D10 末（11-19） | 双接口对比 + n=2 | 若第 2 人不可得 -> n=1 并在表头注明，不延期 |
| **M4** | D14（11-23） | 5 件交付物齐 | 时间不够先保 1 / 2 / 5，观测记录（3）可延 3 天 |

### 20.6 现场采集 SOP（照这个做，别临场发挥）

1. 开跑前：`. .\devtools\env_e_drive_cache.ps1`；`python devtools\check_no_c_drive.py --strict` 必须是 `[OK]`（约定 1）。
2. 摄像头档：先看预览窗左上角 `L=1 R=1`，等控制台打出 `cal=ok` 再动；标定期间机器人不动是**正常**的。
3. 每个任务固定 3 trial；**失败也要落 JSON** —— 失败数据是交付物 2（失败归因表）的唯一原料。
4. 每 trial 结束立刻看 `held_frames` 的 `held_reason`：若 > 30% 是 `no_frame`，该 trial 作废重采（不许事后"补一个成功"）。
5. 采集完立刻把 `outputs/paper_eval/` 整目录复制一份带日期的备份（`outputs/paper_eval_bak_1115/`），不要只留一份。
6. 现场只做"采集"，**不做调参**：发现参数要改，记下来，回来改完再单独约下一次现场。

### 20.7 风险与降级（触发信号 -> 动作，不要临场想）

| 风险 | 触发信号 | 降级动作 |
|---|---|---|
| 三个副本不合并，改一处漏一处 | 同一个 bug 改了两遍 / 两份报告数字对不上 | D1 当天定主线，其余副本转只读 |
| 任务够不着 | `scene.in_reach=false` 或 trial 全 `not_reached` | 用批次 2 的 `solve_shift` 把台面再收 0.1 m；仍不行换任务（20.5 M2） |
| 判据假成功 | `--policy idle` 也判 `success` | 标 `criteria_invalid`，从分母剔除并在报告写明（3 个已知：liftmug / opendrawer / openmicrowave） |
| 摄像头检不到手 | `camera.hand_frames = 0` | 先查光照 / 距离 / 站姿；先跑 `--policy idle` 确认环境本身没问题 |
| VR 掉线或握持遮挡 | 报告里 `frames` 突然归零 | 用已录 900 帧离线回放当 VR 档，标 `proxy` |
| 第 2 名被试约不到 | D10 前没确认 | n=1 + 表头注明，D10 改做 D11（观测记录） |
| 时间不够 | D10 结束还没出 M3 表 | 砍 D11/D12 到 11-25，先保表（交付物 1/2/5） |
| 手部/手臂参数又被现场改坏 | 同一任务两次结果差 > 30% | 采集前先跑 `preflight.ps1 -Full`，退出码必须 0 |

### 20.8 每日收尾 checklist（30 秒能过一遍）

1. 今天的产物落在**哪个文件**？（写进当天 commit 信息，不写"优化了一些细节"）
2. `git status` 干净吗？没提交的改动，明天还认得出来吗？
3. 关键命令的**退出码**记了吗？（§0.1 第 5 条：结论必须附退出码 + 不变量）
4. 有没有把 `outputs/` 里的东西误 `git add` 进去？（约定 5：临时产物一律 `outputs/tmp_*/`）
5. 明天第一件事是什么？（写在 §20.4 对应格子里，别留在脑子里）

### 20.9 对外材料可直接抄的偏差声明

> 本项目在 PyBullet 中复现 TeleOpBench 的 **30 个任务**与 **3 台机器人**，评估口径（从 30 个中选 10 个代表任务、指标为成功率 + 完成时间）与论文一致；
> 但仿真器（PyBullet vs NVIDIA Isaac Sim）、上体估计（MediaPipe pose vs SMPLer-X + 体型标定）、IK（PyBullet `calculateInverseKinematics` vs PINK）、
> 手部重定向（自研 18 维 NN + 原装手语义映射 vs dex-retargeting）均与论文不同；
> 四条遥操作接口中 **2 条（MoCap / 外骨骼）因无硬件只做代理档**（proxy，不参与接口对比结论），被试人数 **n=2**（论文 4 人）。
> 因此本表用于**同条件下的横向比较与趋势复现**，不作为论文数值的逐项对比。

### 20.10 结果回填（每完成一天/一个里程碑，往这里补一行）

| 日期 | 天 / 里程碑 | 产出文件 | 关键数字 | 是否达 gate |
|---|---|---|---|---|
| 2026-11-10 | D1 | 本节 + 日志恢复 + 主线合并方案 | 腕限位拒绝 209 -> 2；h1_2 位置 p50 4.35 -> 3.00 mm；本仓 `unittest` **Ran 397 / OK (skipped=3)**；`preflight.ps1` 退出码 **1**（rule 3 的 49 文件 + 7 个空壳目录） | 部分：测试达 gate，preflight 未达（D1 待修） |

> **回填规则**：只写**已实测**的数字（§11.2 原则 1）；没跑出来就写"未跑"，不要写"预计"。









## 21. 【2026-11-16】批次 5：无人脚本档（没有人、也不等人，把任务做完）

> 需求原话：「你自己一个人完成那两个目标（手腕转动没问题、完成 30 个测试任务），我不会坐在电脑前」。
> 手腕那条的**离线可复现验收**与 30 任务那条的**无人执行**都落在这一节；详细数据在 `docs/task-evaluation.md` §7。

### 21.1 手腕：把「只能现场验」变成「离线可复现」

| 项 | 结果 |
|---|---|
| 左侧反向的根因（§8.20.13） | `palm_orientation_frame` 的 `cross(index-wrist, pinky-wrist)` 在左手朝掌心内 -> 左基 = 右基乘 `R_x(180)`；扭腕不变（两侧 +40.0°），pitch/yaw 反号 |
| 修法 | `MIRRORED_PALM_SIDES = ("left",)`，`side` 设为**必填位置参数**（两个调用点都传） |
| 用例 | `tests/test_arm_realtime_bridge.py::PalmBasisChiralityTests`（5 条）：去掉镜像 -> 3 红 2 绿；漏传 `side` -> 接线用例红 |
| 离线 A/B | 纯旋前夹具 `wrist_fixture_spike.npz` 开/关**逐行相同**（证明共轭数学）；真实动作夹具 `wrist_fixture.npz` 的 `on vs on` 噪声（0.0274/0.0250 rad）**大于**开/关差（0.0253/0.0232），噪声来自墙上时钟 `frame_dt` 喂 `_orientation_limit_deg` -> **离线暂时无法判定**，验收仍以现场为准 |

### 21.2 30 个任务：无人脚本档 9/30（失败原因逐条可查）

命令：`devtools/task_eval.py --policy scripted --task <30 个> --max-sim-s 30`（DIRECT，无窗口、无摄像头）。

| 成功（9） | 时间 | 失败（21）的归类 |
|---|--:|---|
| pushcube / pickcube / pickplacecube | 4.71 / 2.78 / 2.89 s | 判据失效 7（`CRITERIA_INVALID` 3 + `CRITERIA_ABSOLUTE_Z` 2 + 布局后 reset 即真 2） |
| ballmug / pourwater / pottomato | 3.07 / 3.96 / 2.87 s | 需要接触式推压 4（开/关门、拉抽屉、翻屏幕） |
| potbimanual / panhearth / twistbottlecaps | 3.77 / 4.18 / 3.55 s | 抓取路径碰撞或落点精度 10（物体彼此贴着） |

**四个根因（都已修，且都留了实测数字）**：
1. `measure_arm` 的可达半径是假的（0.5527 m 是「肩 link 原点→肘」+「肘→腕」之和，实测伸直只有 0.4067 m）
   -> 新增 `sample_reach()` 用 FK 采样量，h1_2 = **0.4084 m**；场景按这个半径重摆。
2. `TaskScene._settle` 空转 600 步**没给电机**，手臂垂进腿里自己卡死（`shoulder_pitch` 指令 −1.03 rad、
   实测停在 **+0.44 rad**；力 200/500/1200 一样、关掉重力也一样）-> settle 前 `_hold_motors()`。
3. `PyBulletArmIK.solve()` 内部 `resetJointState` 会让机械臂**瞬移穿过**桌面物体（把方块扫飞 0.5 m）
   -> 策略里求解前后存/还原整机角度，只把解交给位置电机。
4. 桌面比肩关节还高 5 cm、桌子+落地支撑在 x>=0.18 是一堵墙 -> 航点固定四段（收肘到胸前 -> 抬过桌沿 ->
   前移到物体上方 -> 下探），并夹进肩关节 0.36 m 球内。

### 21.3 诚实边界（对外材料必须一起给）

* **抓取是约束辅助**（手任一 link 进物体 10 cm 才允许挂 `JOINT_FIXED`，够不着就记 `grasp MISSED`），
  不是指尖摩擦抓取；`--no-grasp-assist` 可复现「纯接触抓不住」的对照。
* **场景摆法被改过**：物体质心在肩前 0.26 m、桌面比肩低 0.22 m、相对布置压缩进 0.12 m —— 每一条都在
  trial 报告的 `scene` 段里（`layout_for_arm` / `ik_reach` / `moved`），报成功率时必须一并给出。
* **判据在 reset 即为真 -> 记失败**（`criteria_at_reset`），不刷成功率。
* 还没做：三台机器人都跑一遍（本轮只跑 h1_2）、推压原语、抓取局部避障、落点二次校正、
  离线 `frame_dt` 固定值（有了它离线 A/B 才能判定左手修复）。
