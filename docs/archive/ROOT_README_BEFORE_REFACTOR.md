# 大创 —— 遥操作系统

> 中山大学 · 大学生创新创业训练计划项目

**阶段汇报与答辩材料**：[present 成果包](present/README.md)（10 张图片、4 段短视频及 GIF 预览），
完整汇报正文见 [项目阶段成果](present/项目阶段成果.md)。

基于 **PyBullet** 的双臂灵巧手遥操作仿真平台，面向双臂人形机器人的
**遥操作数据采集** 与 **动作重定向算法** 验证。

**最终方向**：实现真实输入 → 手部/双臂重定向 → 明确的机器人命令与关节映射 →
仿真任务执行 → 同步记录 → 可复现评估的闭环。当前手部核心已完成 PR1 合并，
实时整机连接、command 统一、真实设备/物理验证仍需推进。

请先阅读 [总体架构与端到端链路](docs/SYSTEM_ARCHITECTURE.md)：包含当前代码内容、
逐文件职责、重点链路、人工验证和已确认问题。所有说明文档统一从
[docs 文档索引](docs/README.md)进入。

---

## 一、项目简介

本项目搭建了一个面向双臂人形机器人的遥操作仿真平台，提供 **30 个分层操作任务**，
用于验证「VR / 数据手套 → 动作重定向 → 双臂机器人 → 任务执行 → 数据采集」这条完整链路。

下面的遥操作图是项目方向，当前旧 VR/手套接口使用 mock，正式 hand realtime
只输出模型预测；各模块的实际连接状态以总体架构说明为准。

### 遥操作流水线

```
Apple Vision Pro（手腕追踪） ──┐
                              ├──► 坐标转换 / IK ──► 双臂关节角度
LinkerHand 数据手套 ───────────┘
                                    │
                                    ▼
                            灵巧手关节（重定向）
                                    │
                                    ▼
              PyBullet 仿真环境 ──► 任务执行 ──► 传感器数据记录
```

### 分层任务体系（共 30 个）

| 层级 | 类别 | 任务数 |
|------|------|--------|
| Level 1 | 基础拾取放置 | 5 |
| Level 2 | 工具操作 | 11 |
| Level 3 | 双手协作 | 9 |
| Level 4 | 长时域序列 | 5 |

### 支持的机器人模型

| 机器人 | 说明 |
|--------|------|
| **H1-2** | Unitree 人形机器人（55 关节） |
| **GR1-T2** | Fourier 人形机器人 |
| **G1** | Unitree 人形机器人 |

---

## 二、目录结构

```
teleoperation-system/
├── main.py / main_*         # 应用入口、hand CLI wrappers、双臂入口
├── retargeting/             # canonical hand 与保留的 arm/command 服务
├── model/                   # PoseTransformer、L21 FK、loss
├── input_adapters/          # NPY replay 等应用输入
├── config/                  # 应用契约、TRON2A 示例 calibration
├── envs/                    # benchmark 环境、robot loader、记录、随机化
├── teleop/                  # 旧遥操作框架、原生手映射与回放滤波
├── simulation/              # TRON2A + L21 PyBullet 场景
├── tasks/                   # 30 个任务类、基类与注册
├── utils/                   # 成功率、完成时间统计
├── scripts/                 # 回放、诊断和可视化
├── tests/                   # hand/coordinate/应用边界回归
├── docs/                    # 架构、使用、协议与验收说明
├── robots/from_teleopbench/ # 已入库机器人模型
├── dataset/robot/           # 已有 L21 FK/手部资产
├── datasets/                # 系统协作数据
├── data/ / outputs/         # 本地记录、训练与推理产物（忽略）
└── requirements*.txt        # 按 application/hand 用途分类的依赖
```

---

## 三、环境要求

- 正式环境为 **Conda `teleoperation` / Python 3.10.20**；本机解释器：
  `D:\Anaconda\envs\teleoperation\python.exe`，PyCharm 也使用此路径。
- 从仓库根按 [environment.yml](environment.yml) 重建。完整安装包括 application、
  hand runtime/training、camera/tools 与 Robotics Toolbox；requirements 保持原约束。
- 已验证 CPU hand 导出、完整 tests、PyBullet、双臂 FK/IK 与 camera imports。
  当前 Torch 为 `2.14.1+cu130`，RTX 4050 上 CUDA 导出与 1 epoch 训练 smoke 通过；
  完整 CUDA training 与真实摄像头未验收。
- 旧 TransHandR 保留为 reference baseline；仓库 `.venv` 不是正式环境。
  详细版本、验证结果和资源要求见 [环境说明](docs/ENVIRONMENT.md)。

```powershell
# 本机已创建环境，只需 activate；首次安装执行 env create。
conda env create -f environment.yml
conda activate teleoperation
python -c "import sys; print(sys.executable)"
python -m pip check
python -m retargeting --help
```

---

## 四、使用方法

### 基础入口

先激活正式环境 `conda activate teleoperation`。PR1.5 已修复 `main.py` 的动作维度
初始化、单独 `--demo` 不执行以及 inspect wrapper 的 import 错误，见架构说明第 8 节。
`--demo` 默认运行 pushcube 三次，可用 `--task` 指定任务；`--benchmark` 与 demo/task 互斥。
入口验证建议加 `--no-record`。默认逐步缓存双相机图像的长程 smoke 未通过验收，
出现高内存占用与一次 native crash，详见 [PR1.5 结果](docs/PR15_APPLICATION_ENTRY_RESULT.md)。

```bash
python test_import.py                       # 环境自检（6 项，建议先跑这个）
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render --steps 40
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --render --steps 240
python scripts/replay_actions.py --describe --robot gr1_t2
python -m retargeting --help                # 正式手部 CLI
python main.py --help                       # 查看已有应用入口参数
python main.py --demo --no-render --no-record # 实际运行三次演示
python main.py --task pushcube --robot h1_2 --no-render --no-record
python inspect_angle_h5.py --angle-h5 outputs/env_checks/palm_angles.h5

python show_all.py                          # 并排展示三种机器人
python show_hand.py                         # 展示 LinkerHand 灵巧手
python test_camera.py                       # 测试摄像头
```

### 离线 Vision 流水线（核心工作流）

> 📄 详见 [`docs/OFFLINE_PIPELINE.md`](docs/OFFLINE_PIPELINE.md)　|　接口定义见 [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md)

本节回放的是系统 actions 协议。当前正式 hand H5 → 模型 → angles H5 链路见
[总体架构第 3 节](docs/SYSTEM_ARCHITECTURE.md)和 [hand 使用说明](docs/HAND_RETARGETING.md)。
早期契约 G 的单手示例不是 canonical 两手训练输入。

```bash
# 查看某机器人的动作空间定义（38/36/28 维的完整关节映射）
python scripts/replay_actions.py --describe --robot h1_2

# 用假数据验证整条链路（不需要真实数据）
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render

# 回放重定向算法输出的动作序列（契约H 格式）
python scripts/replay_actions.py --file datasets/actions/xxx.h5

# 带可视化
python scripts/replay_actions.py --file xxx.h5 --render

# 生成契约 G/H 的示例数据文件
python scripts/make_sample_data.py --kind all
```

### 手部回放（队友重定向输出 → 机器人手）★ 本项目主用

> 映射模块：`teleop/native_hand.py`　|　决策记录：`docs/PROJECT_CONTEXT.md` 第 18 节

```bash
# ★ 用机器人【原装】灵巧手回放（本项目采用，不换手）
python scripts/replay_hand_native.py --robot h1_2   --hand both --render
python scripts/replay_hand_native.py --robot gr1_t2 --hand both --render
python scripts/replay_hand_native.py --robot g1     --hand both --render

# 演示用：整机视角 + 无限循环（相机自动框住整个机器人）
python scripts/replay_hand_native.py --robot h1_2 --hand both --render --view full --loop 0
#   --view full(默认,整机) / front(正面) / side(侧面) / hands(手部特写)
#   --loop 0 = 无限循环；--loop 3 = 播 3 遍

# ★ 答辩/报告用：三台机器人并排，各自原装手按同一份数据同步屈伸
python scripts/show_hands_all.py --render --loop 0
python scripts/show_hands_all.py --render --view front --loop 0

# 只看映射报告（覆盖率 / 丢弃哪些自由度 / 每个关节的符号方向）
python scripts/replay_hand_native.py --robot h1_2 --hand both --report

# 备用：把 LinkerHand l21 装到机器人腕部（无损，但需要"换手"）
python scripts/replay_hand_on_robot.py --robot h1_2 --hand both --render

# 一键验收（6 项：数据结构 / 有效帧 / 坏帧检测器 / 限位 / 回放 / 原装手映射）
python scripts/verify_hand_pipeline.py
```

**各机器人原装手能表达多少自由度**（数据是 L21 的 17 维）：

| 机器人 | 原装手 | 可表达 | 覆盖率 |
|---|---|---|---|
| H1-2 | Inspire | 12/17 | 71% |
| GR1-T2 | Fourier 原生手 | 11/17 | 65% |
| G1 | 轻量三指手 | 7/17 | 41% |

> ⚠️ 这是**有损**映射（丢弃 `*_mcp_roll` 侧摆等原装手没有的自由度），
> 详情与必须声明的限制见 `docs/OFFLINE_PIPELINE.md` 第 8.11 节。

### 离线流水线三方分工

```
队友B 采集手部数据  →  队友A 做重定向  →  仿真平台回放
  (契约G)               (契约H)          (replay_actions.py)
 human_hand.h5          actions.h5
```

---

## 五、常见问题

**Q：`show_hand.py` 报路径错误？**
以前是硬编码绝对路径，现已改为**基于脚本位置动态定位**（见 `HAND_DIR`）。
若仍报错，说明本地缺少 `linkerhand_sdk`，按第六节获取即可。

**Q：克隆后运行报找不到 URDF？**
`robots/from_teleopbench/` 与已有 L21 FK 资产已入库；先检查具体失败路径。
TRON2A description、LinkerHand SDK 等外部资产仍需按使用分支自行配置，见第六节与总体架构说明。

---

## 六、机器人模型资产

> **好消息：`robots/from_teleopbench/`（181 MB）已纳入 Git 仓库**，
> `git clone` 后**自动获得**，无需额外下载。
>
> （早期版本未入库、需要手动获取 —— 现已改为直接入库。原因见下文说明。）

### 关于「为什么不用 Git LFS」

**Git LFS 的文件传输强制走 HTTPS**，而本项目的网络环境对 `github.com` 的 HTTPS
存在**SNI 定向干扰**（需要代理才能通），但 **SSH（22 端口）稳定可用**。

| 方案 | 传输协议 | 需要代理？ | 队友上手 |
|---|---|---|---|
| Git LFS | HTTPS | ⚠️ **必须常开代理** + 有 1GB/月流量配额 | 需装 git-lfs + 配代理 |
| **普通 Git（本项目采用）** | **SSH** | ✅ **不需要** | `git clone` 一步到位 |

**代价**：仓库体积约 181 MB（远低于 GitHub 的 1 GB 警告线 / 5 GB 硬限）。

### 仍然需要自行获取的部分

以下内容体积过大或属第三方，**未入库**：

| 目录 | 体积 | 获取方式 |
|---|---|---|
| `lib/` | 335 MB | `pip install -r requirements.txt` |
| `linkerhand_sdk/` | 1013 MB | ⬇️ 见下方 |
| `robots/` 的其他子目录 | 274 MB | 代码未引用，**不需要** |

#### `linkerhand_sdk/` —— 灵巧手 SDK

`show_hand.py` 从该 SDK 读取灵巧手模型：

```bash
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git linkerhand_sdk
```

### `robots/` 目录内容说明

| 子目录 | 体积 | 是否入库 | 说明 |
|---|---|---|---|
| **`from_teleopbench/`** | **181 MB** | ✅ **已入库** | H1-2 / GR1-T2 / G1 三种模型（代码实际使用）|
| `arms/` `assembly/` `h1_paper/` `h1_with_hand/` `hands/` `linker_hand/` `xarm7_ability/` | 274 MB | ❌ | 代码未引用，可由 `.gitignore` 排除 |


---

## 七、第三方开源声明

本项目的部分设计参考与机器人模型资产来自以下开源项目，在此致谢：

| 项目 | 作者 / 组织 | 许可证 |
|------|-------------|--------|
| TeleOpBench | Unitree Robotics（HangZhou YuShu TECHNOLOGY CO.,LTD.） | Apache-2.0 |
| LinkerHand SDK | ericbrunt (brunt888) | 见原仓库 |

详见仓库根目录的 [`NOTICE`](NOTICE) 文件。

TeleOpBench 建立在以下开源代码库之上，请访问链接查看各自的许可证：

1. https://github.com/OpenTeleVision/TeleVision
2. https://github.com/dexsuite/dex-retargeting
3. https://github.com/vuer-ai/vuer
4. https://github.com/stack-of-tasks/pinocchio
5. https://github.com/casadi/casadi
6. https://github.com/meshcat-dev/meshcat-python
7. https://github.com/zeromq/pyzmq
8. https://github.com/unitreerobotics/unitree_dds_wrapper
9. https://github.com/tonyzhaozh/act
10. https://github.com/facebookresearch/detr
11. https://github.com/Dingry/BunnyVisionPro
12. https://github.com/unitreerobotics/unitree_sdk2_python

---

## 八、作者

**Wangxianyu835** —— 中山大学 · 大学生创新创业训练计划

---

## 九、Hand retargeting 子系统（PR1）

本仓库现为 production hand retargeting 的唯一源码维护入口，合入了
`mytrans@138fc2d` 的 canonical hand 核心。系统原有双臂、机器人、仿真、任务、
benchmark 和 recording 继续沿用本仓库的应用架构。

手部链路为 MediaPipe21 → Hand25 → 坐标对齐 → `[B,3,25,3]` → PoseTransformer →
18D（root placeholder + 17 个真实 hand DOF）。支持 legacy `source_to_l21_xyz` 与
palm-local `palm_local_to_l21_v1`，H5 / Dataset / training / checkpoint / inference
必须严格使用同一标识；缺失或不匹配直接失败。

正式手部入口为 `python -m retargeting {align,train,export,inspect,realtime}`。
安装、H5 schema、训练/推理、MediaPipe realtime、原生手回放及旧 wrapper 行为见
[Hand retargeting 使用说明](docs/HAND_RETARGETING.md)。
[PR1 合并结果](docs/PR1_HAND_CONSOLIDATION_RESULT.md)记录测试与迁移边界；
[PR2 follow-up](docs/PR2_FOLLOW_UP.md)记录整机 command、48DOF 与 arm mapping 后续工作。

