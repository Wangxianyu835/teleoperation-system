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
│   └── pipeline_data.py     #   流水线数据结构
│
├── utils/
│   └── metrics.py           # 评估指标（成功率 / 完成时间）
│
├── robots/                  # 机器人模型资产（不入库，见第六节）
├── lib/                     # 本地依赖目录（不入库）
└── linkerhand_sdk/          # 第三方 SDK（不入库，见第六节）
```

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

---

## 四、使用方法

```bash
python main.py                              # 列出所有任务并交互式选择
python main.py --task pushcube              # 运行指定任务
python main.py --demo                       # 演示模式（无控制器）
python main.py --benchmark                  # 对 Level 1 任务做基准测试
python main.py --task pushcube --no-render  # 无头模式

python show_all.py                          # 并排展示三种机器人
python show_hand.py                         # 展示 LinkerHand 灵巧手
python test_import.py                       # 自检所有模块能否正常导入
python test_camera.py                       # 测试摄像头
```

---

## 五、常见问题

**Q：`show_hand.py` 报路径错误？**
该脚本中的 `HAND_DIR` 目前是硬编码的绝对路径，换机器后需要改成自己的
`linkerhand_sdk` 实际路径（建议改为相对路径）。

**Q：克隆后运行报找不到 URDF？**
`robots/` 目录未入库，需按第六节自行获取模型资产。

---

## 六、模型资产获取（重要）

本仓库**只托管源码**，以下大型资产未入库，需要自行获取后才能完整运行：

### 1. `robots/from_teleopbench/` —— 机器人模型（约 181 MB）

代码通过相对路径读取 `robots/from_teleopbench/` 下的 URDF 与 mesh：

| 用途 | 路径 |
|------|------|
| H1-2 模型 | `robots/from_teleopbench/h1_2/h1_2.urdf` |
| GR1-T2 模型 | `robots/from_teleopbench/gr1/urdf/robot.urdf` |
| G1 模型 | `robots/from_teleopbench/g1/g1_29dof_with_hand_lock_waist.urdf` |

> 这些模型来自 **TeleOpBench**（Unitree Robotics，Apache-2.0）。
> 请从 TeleOpBench 官方仓库获取后，放到 `robots/from_teleopbench/` 目录下。

### 2. `linkerhand_sdk/` —— 灵巧手 SDK（约 1013 MB）

`show_hand.py` 从该 SDK 读取灵巧手模型：

```bash
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git linkerhand_sdk
```

### 3. `lib/` —— 本地依赖目录（约 335 MB）

由 `pip install -r requirements.txt` 生成，不入库。

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

