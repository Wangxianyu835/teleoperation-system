# 大创 —— 遥操作系统

> 中山大学 · 大学生创新创业训练计划项目

基于 **PyBullet** 的双臂灵巧手遥操作仿真平台，面向双臂人形机器人的
**遥操作数据采集** 与 **动作重定向算法** 验证。

---

## 一、项目简介

本项目搭建了一个面向双臂人形机器人的遥操作仿真平台，提供 **30 个分层操作任务**，
用于验证「VR / 数据手套 → 动作重定向 → 双臂机器人 → 任务执行 → 数据采集」这条完整链路。

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
simulation_platform/
├── main.py                  # 主入口（任务选择 / demo / benchmark）
├── demo_teleop.py           # 无控制器演示模式
├── demo_hands_three_robots.py  # ★ 一键入口：三台机器人手部演示（PyCharm 右键 Run）
├── show_all.py              # 并排展示 H1-2 / GR1-T2 / G1 三种机器人
├── show_hand.py             # LinkerHand 灵巧手独立展示
├── test_camera.py           # 摄像头采集测试
├── test_import.py           # 模块导入自检
├── requirements.txt         # Python 依赖
│
├── envs/                    # 仿真环境
│   ├── simulation_env.py    #   环境核心类（集成机器人/任务/记录/随机化）
│   ├── robot_loader.py      #   机器人模型加载
│   ├── sensor_recorder.py   #   观测数据记录（状态/相机流/物体元数据）
│   └── domain_randomizer.py #   域随机化（光照/摩擦/纹理/位置）
│
├── tasks/                   # 任务库
│   ├── all_tasks.py         #   全部 30 个分层任务实现
│   ├── base_task.py         #   任务基类
│   └── task_registry.py     #   任务注册表
│
├── teleop/                  # 遥操作接口
│   ├── teleop_pipeline.py   #   完整遥操作流水线
│   ├── vr_interface.py      #   Vision Pro 手腕追踪 → 关节角度
│   ├── hand_interface.py    #   灵巧手驱动接口
│   ├── camera_interface.py  #   摄像头采集
│   ├── pipeline_data.py     #   流水线数据结构
│   ├── native_hand.py       #   ★ L21 数据 → 机器人【原装手】降维映射
│   └── filters.py           #   关节角平滑滤波
│
├── scripts/                 # 离线回放 / 验收工具（10 个）
│   ├── replay_hand_native.py    # ★ 原装手回放（本项目主用）
│   ├── show_hands_all.py        # ★ 三机器人并排同步屈伸（答辩用）
│   ├── check_native_hand_motion.py  # ★ 逐关节实测"到底动了没、动了多少"
│   ├── render_hand_snapshots.py # ★ 离屏渲染三台机器人手部快照（无需显示器）
│   ├── replay_hand_on_robot.py  # 备用：把 l21 装到机器人腕部
│   ├── replay_hand_angles.py    # l21 直接回放
│   ├── replay_actions.py        # 契约 H 动作序列回放
│   ├── make_sample_data.py      # 生成契约 G/H 示例数据
│   ├── verify_hand_pipeline.py  # 手部链路一键验收
│   └── check_gbk_safe.py        # 提交前 GBK 安全检查
│
├── utils/
│   └── metrics.py           # 评估指标（成功率 / 完成时间）
│
├── docs/                    # 文档（见下方「文档索引」）
├── datasets/                # 数据交换目录（已入库，契约 G/H 靠它传递）
│   ├── raw/                 #   队友采集的原始手部数据
│   └── samples/             #   示例数据（可直接跑回放）
├── robots/from_teleopbench/ # ★ 机器人模型资产（已入库，见第六节）
├── lib/                     # 本地依赖目录（不入库）
└── linkerhand_sdk/          # 第三方 SDK（不入库，见第六节）
```

### 📚 文档索引（`docs/`）

| 文档 | 内容 | 什么时候看 |
|------|------|-----------|
| [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) | **环境配置与工具链备忘**（解释器 / SSH / 代理 / E 盘权限 / GBK 编码约定）| ★ 换机器、重装、报环境错时 |
| [`docs/TEAM_ONBOARDING.md`](docs/TEAM_ONBOARDING.md) | 队友从零上手（约 30 分钟）| 新队友加入 |
| [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md) | 接口契约 A~H（动作 / 观测 / 控制器 / HDF5）| 三方对接前必读 |
| [`docs/OFFLINE_PIPELINE.md`](docs/OFFLINE_PIPELINE.md) | 离线数据流水线（契约 G / H）| 跑离线回放 |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | 项目背景 / 决策历史 / 待办清单 | 了解「为什么这么做」|

---

## 三、环境要求

- **Python 3.11**
- 依赖（见 `requirements.txt`）：

```
pybullet>=3.2.0
numpy>=1.24.0
h5py>=3.10.0
scipy>=1.10.0
```

安装：

```bash
pip install -r requirements.txt
```

> ⚠️ **开发机上的实际跑法**：用的是全局 Python `E:\python3.11.7\python.exe`，
> **不是**仓库里的 `.venv`（那是个空壳，只有 pip/Pillow/pypdf）。
> 环境细节、踩过的坑、常用命令速查见
> [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md)。
>
> PyCharm 里跑不起来（`ModuleNotFoundError: No module named 'pybullet'`）=
> 项目解释器选到了空壳 `.venv`：`Settings > Project > Python Interpreter`
> 换成 `E:\python3.11.7\python.exe` 即可。
> 另外 `demo_hands_three_robots.py` 自带兜底：解释器缺依赖时会自动改用
> 仓库内 `lib/`（`pip install -r requirements.txt -t lib` 的产物，不入库）。

---

## 四、使用方法

### 基础入口

```bash
python test_import.py                       # 环境自检（6 项，建议先跑这个）
python main.py                              # 列出所有任务并交互式选择
python main.py --task pushcube              # 运行指定任务
python main.py --demo                       # 演示模式（无控制器）
python main.py --benchmark --trials 3       # 对 Level 1 任务做基准测试
python main.py --task pushcube --no-render  # 无头模式
python main.py --robot gr1_t2 --task pickcube   # 切换机器人

python demo_hands_three_robots.py           # ★ 一键：三台机器人手部演示（PyCharm：右键本文件 -> Run）
python show_all.py                          # 并排展示三种机器人
python show_hand.py                         # 展示 LinkerHand 灵巧手
python test_camera.py                       # 测试摄像头
```

### 离线 Vision 流水线（核心工作流）

> 📄 详见 [`docs/OFFLINE_PIPELINE.md`](docs/OFFLINE_PIPELINE.md)　|　接口定义见 [`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md)

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

# ★ 演示推荐：手部特写 + 无限循环（三台机器人都能演，实测通过）
#   单侧特写最清楚：能看清每根手指的关节在屈伸
python scripts/replay_hand_native.py --robot gr1_t2 --hand right --render --view hands --loop 0
python scripts/replay_hand_native.py --robot g1     --hand right --render --view hands --loop 0
python scripts/replay_hand_native.py --robot h1_2   --hand right --render --view hands --loop 0
#   想两只手同框（相机自动拉到 0.5~0.7 m，两手左右并排）
python scripts/replay_hand_native.py --robot gr1_t2 --hand both --render --view hands --loop 0
#   --view hands 按【手部 AABB】自动取景，并按「躯干 -> 手」方位自动选 yaw
#   三台机器人的手在世界里的位置完全不同（右手 x：H1-2 +0.36 / GR1-T2 -0.19 / G1 +0.25），
#   早期版本写死 yaw=135 + 距离 1.1 m，GR1-T2 的手在画面里只剩几个像素（已修）
#   --view full(默认,整机) / front(正面) / side(侧面) / hands(手部特写)
#   --loop 0 = 无限循环；--loop 3 = 播 3 遍
#   --file 指定数据：默认队友数据（557 帧）；自己的采集加
#          --file datasets/raw/my_recording_angles.h5（809 帧）

# ★ 答辩/报告用：三台机器人并排，各自原装手按同一份数据同步屈伸
python scripts/show_hands_all.py --render --loop 0
python scripts/show_hands_all.py --render --view front --loop 0

# ★ PyCharm 用户看这里：不用记上面这串参数，直接右键仓库根目录的
#   demo_hands_three_robots.py -> Run，默认就是上面第一条的效果
#   （已内置 --render --loop 0，并自动挑 datasets/raw 下已采好的数据）
#   想换视角/速度就在 Run Configuration 里加参数，例如 --view front
python demo_hands_three_robots.py
python demo_hands_three_robots.py --view front

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

### 三台机器人演示验收（2026-09-30 实测，同一份采集数据）

数据 = `datasets/raw/my_recording_angles.h5`（809 帧，摄像头采集 + 重定向输出）。

| 机器人 | 可表达 | 会动的映射关节 | 限位截断 | 稳定后跟踪误差 | 结论 |
|---|---|---|---|---|---|
| H1-2 | 12/17 | 24（左右各 12） | 4.4% / 6.5% | 0.0000 rad | PASS |
| GR1-T2 | 11/17 | 22（左右各 11） | **0.0% / 0.0%** | 0.0000 rad | PASS |
| G1 | 7/17 | 14（左右各 7） | 2.8% / 4.9% | 0.0000 rad | PASS |

> GR1-T2 的行程最宽，L21 数据**一点都不用截断**，是三者里贴合度最好的。
> 「会动」判据：该关节整段回放中实际转过 ≥ 0.05 rad（肉眼可见）。

```bash
# 量化验收：逐关节测"实际转了多少弧度"，报告有没有卡死的关节
python scripts/check_native_hand_motion.py --file datasets/raw/my_recording_angles.h5

# 离屏出图（不需要显示器）：三台机器人的原装手快照条带 -> outputs/hand_snapshots/
python scripts/render_hand_snapshots.py --file datasets/raw/my_recording_angles.h5
python scripts/render_hand_snapshots.py --robots gr1_t2 g1 --n 5 --hand left

# 出视频：每台一段 MP4；加 --stacked 再拼成【三台同框、帧同步】一段
python scripts/render_hand_snapshots.py --n 6 --video --stacked --file datasets/raw/my_recording_angles.h5
#   --video     相机固定（按取样帧姿态【并集】取景，张开/握紧都不出画）
#   --stacked   把三台按帧横向拼成 three_robots.mp4（第 k 帧 = 同一数据帧）
#   --no-paint  不刷浅灰：H1-2 网格材质在 getCameraImage 下是【纯黑剪影】，
#               默认刷成浅灰才看得清手指（只改外观，不动物理）
#   自检：视频抽「最张开 / 中 / 最握紧」三帧算像素平均差，接近 0 会告警
```

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
代码实际使用的 `robots/from_teleopbench/`（181 MB）**已随仓库入库**（普通 Git，非 LFS），
`git clone` 完成即自动获得，**无需额外下载**（见第六节）。
若本地缺失，说明 clone 不完整（浅克隆 / 中途中断），重新 clone 即可。

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

