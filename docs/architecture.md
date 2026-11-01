# 架构与模块边界

生产包是 `src/teleoperation`。完整工作流仅在 `applications` 中组装，CLI 使用静态命令注册并延迟导入应用 handler。

相机机械臂链路使用 `CameraTeleopSource`：一次 RGB 采集分别进入 MediaPipe Hands 和独立
SMPLer-X 工作进程。`inputs/body_tracking` 管理新身体源的时序有效性；实时与新 H5 回放通过
`applications/camera_scene_arm` 共用既有映射、腕姿态与 IK 控制。模型依赖只进入独立环境。
旧 Pose 身体 reader 和 Hands 驱动机械臂的回退已移除。配置见 [SMPLer-X 链路](smplerx-arm.md)。

MediaPipe Hands 仅输出手部关键点，`RawHandFrame.wrists=None`；不再生成、录制或恢复
`camera_palm` 腕朝向。机械臂读取 `RawTeleopObservation.wrists` 中的 SMPLer-X 腕姿态，
Vision Pro 继续使用原始腕矩阵。手指局部坐标仍调用 `build_palm_basis` 和
`align_mediapipe_hand`，与被删除的独立腕观测分支用途不同。
变更范围及验证见 [MediaPipe 腕观测退役](mediapipe-wrist-removal.md)。

```mermaid
flowchart LR
  I[Input] --> O[Observation Contract]
  O --> R[Retargeting]
  R --> S[Robot-specific Semantic Result]
  S --> A[Robot Adapter]
  A --> N[Native Action]
  N --> E[Simulation / Hardware]
```

## 模块职责

| 模块 | 职责 | 边界 |
|---|---|---|
| contracts | 简单类型、声明、shape 与 validation | 仅标准库和 NumPy；不处理模型或文件工作流 |
| inputs | 获取和解析传感器原始结构 | 不变换坐标，不做 21→25、跟踪、窗口或推理 |
| retargeting | 坐标、拓扑、跟踪、时间算法、网络、FK、IK、安全控制 | 不运行设备，不选择任务，不依赖 learning |
| robots | 特定机器人的规格、手部映射、native action 编码 | 不读取人体 landmarks，不复制跨机器人手臂角 |
| simulation | 加载和执行机器人、任务、随机化、状态采集 | 不加载手部 checkpoint 或训练模块 |
| learning | Dataset 构建、loss、优化、epoch 计算 | 可用手部算法与数据读取；运行组装在 apps/training |
| data | schema、读写、序列化、事务、记录缓存 | 不调用对齐、滤波、推理、IK 或设备采集 |
| apps | 创建组件、按顺序调用、循环、同步、关闭资源 | 不实现坐标、IK、网络或关节映射公式 |
| tools | 绘图、显示、报告格式化 | 不独立组织业务流程 |

`retargeting/arm/rotations.py` 保存机械臂共享的 NumPy 矩阵/四元数转换；
回归测试位于 `tests/test_arm_rotations.py`。旧 `wrist_targets.py` 的手部到末端近似映射
及其 `devtools/verify_arm_from_capture.py` 入口已退役。

`inputs/replay` 只逐帧读取 NPY；`learning/dataset` 允许调用单帧处理与时间缓冲来构建样本。二者都不调用应用层。

## 手部链路

当前训练、控制与导出的操作说明见 [统一手部重定向](hand-retargeting-unified.md)。

```text
MediaPipe / aligned world H5
    -> CanonicalHandFrame (metres, canonical right)
    -> TemporalBuffer (3 frames, per-side reset)
    -> normalize_hand_window (3-frame median palm length, dimensionless)
    -> HandWindow
    -> JointNNRetargeter / NormalizedDexRetargeter
    -> dict[(hand_model,side), HandJointResult] (native independent q[D])
    -> JointCommandPipeline (filter, effective limits, speed, optional L21 guard)
    -> HandJointCommand (expanded named joints)
    -> HandExecutor -> scene.apply_native -> PyBullet
```

`HandSpec` 定义目标资产身份、独立/从动关节、参考姿态、固定机器人掌长和根坐标变换。新 NN 与 dex 共用 `build_hand_targets`；NN 用可微 FK 训练。`learning/trainer.py` 只训练目标手原生 D 维模型，人体米制 H5 不带机器人角度标签。多手型顺序训练，数据和划分相同，权重按手型/目标侧独立保存。

公共 profile 是 `camera_palm_length_window3_v1`，新窗口 units=1；H5 原始人体点仍为米。每侧三帧使用该窗口腕点到中指 MCP 掌长中位数，不使用未来/全局统计。缺手或退化后仅清空该侧历史。Dataset 和实时调用同一归一化函数。应用层组装流程，data 层只序列化。

历史 L21 NN、几何和 VR 也返回公共 HandJointResult，旧 18D 占位只存在于实际权重、历史 H5 和明确诊断边界。旧 NN 不输入新版归一化数据；其输入必须匹配权重真实声明。VR 保留独立上游轴变换与三帧数值链路，左手目标在输出边界转换为原生侧角度，保留原始输出诊断。

TRON2A/双臂既有相机流程保持其显式旧输入规范。旧 L21 到 H1/GR1/G1 的降维显示属于历史有损映射，新 `hand control --hand-model ...` 直接使用目标手原生结果，不经过该映射。

## 双臂链路

```text
UpperBodyObservation
    → Calibration + ArmRetargeter
    → ArmTargetPose[xyz + xyzw]
    → ArmIK
    → ArmIKResult[q7, success, errors]
    → ArmSafetyController
```

上身关键点、末端 pose、IK 关节角不能以裸数组互相冒充。实时观测严格检查结构和有限值；离线无效观测保留上一目标。IK 失败保留 previous q 和 inf 误差；安全控制仍按原有成功标志与时间戳处理。

`apps/dual.py` 接收原始组合观测，手部与双臂分别处理，再组装 `Tron2AL21Command`。机器人层不解释人体 landmarks。

## 机器人动作与执行

TRON2A/L21 固定 `[LA7 | LH17 | RA7 | RH17]`，使用严格拼接或 `Tron2AL21ActionAdapter`。H1/GR1/G1 使用自己的原生关节语义目标和各自动作适配器，动作顺序仍由实际加载的 `action_joint_names` 表达。

`pack_partial_limb_vector()` 只返回 `PartialLimbVector`，没有自动 NumPy 转换。正式适配器拒绝 partial；`SimulationEnv.step` 在关节下发和物理步进之前拒绝 partial、错误维度与非有限动作。`None` 保留原行为。

## 文件与记录流程

```text
apps/hand_align
    → data/hand_h5 读取 schema 与声明
    → retargeting/hand/alignment 和 coordinates 计算
    → data/hand_h5 写入
    → data/transaction 关闭后 os.replace
```

事务使用同目录临时文件，失败清理自己的临时文件；输入与输出不能指向同一文件。历史 schema、metadata 与数据精度沿用已有实现。

```text
simulation/sensors.collect
    → SimulationSample
apps/simulation.EpisodeRecording
    → data/recorder 缓存与保存
```

仿真环境通过注入的 recording observer 发出原有采集时点事件；应用 observer 连接采集与持久化。`data/recorder` 不导入 PyBullet，也不查询机器人或相机。使用 `apps/simulation.create_environment()` 创建带录制的应用环境；直接构造 `SimulationEnv(record=True)` 需提供 recording observer。episode、evaluation、benchmark 循环位于 `apps/simulation.py`。

checkpoint 的文件序列化位于 `data/checkpoint`，state_dict 选择、严格加载和坐标 guard 位于 `retargeting/hand/checkpoint`。仿真只使用机器人和资源配置。

## 入口迁移

| 原入口 | 当前命令 |
|---|---|
| main.py | sim run |
| demo_teleop.py | sim demo-joints |
| main_train_twohand.py | hand train |
| main_offline_twohand.py | hand export |
| inspect_angle_h5.py | hand inspect |
| record_world_landmarks.py | hand record-world |
| scripts/align_h5_coordinates.py | hand align |
| main_offline_dual_teleop.py | dual export |
| main_realtime_dual_teleop.py | dual realtime |
| main_pybullet_dual_teleop.py | dual replay |
| scripts/replay_actions.py | replay actions |
| scripts/replay_hand_native.py | replay native-hand |
| scripts/replay_hand_angles.py | replay l21-hand |
| scripts/replay_hand_on_robot.py | replay mounted-hand |
| show_all.py / show_hand.py / scripts/show_hands_all.py | tools show-robots / show-hand / show-all-hands |
| test_import.py / test_camera.py | tools check-environment / check-camera |
| 其他 scripts 工具 | tools 文件名去扩展名、下划线转连字符 |

不保留旧路径 Python wrapper。静态架构测试检查直接导入、重导出后的传递依赖、受控动态导入和 partial 的生产调用位置。

## 目录迁移时的验证范围（历史）

按用户最新要求不使用冻结基线进行验收。仅执行语法、导入、CLI 帮助、包构建与架构规则检查；没有执行迁移后的数值或物理回归。现有行为测试保留并迁移导入、mock 目标和命令，容差不放宽。后续由用户运行完整 unittest 和真实工作流。摄像头、CUDA 与外部 TRON2A URDF/mesh 的运行情况仍需实际验证。

后续 current-state 数值回归、物理 smoke 与实时坐标契约修复的实际结果见 [2026-10-09 验收报告](current-state-validation-2026-10-09.md)。
