# PR2 follow-up: application command and arm integration

当前总体链路、文件职责及随后发现的入口问题见 [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)。

PR1 仅合并 canonical hand implementation。以下均为目标 baseline 已存在的系统边界，
保持原有行为；PR2 应在 canonical 手部链路之上统一应用协议，避免再次复制 hand 算法。

## ACTION_ORDER、48DOF 与 RobotCommand

| 当前模块 | 已有语义 |
|---|---|
| `retargeting/contracts.py` | `[left_arm(7),left_hand(17),right_arm(7),right_hand(17)]`，固定 48D、缺臂拒绝 |
| `retargeting/command.py` | 已有 `RobotCommand`，同一 48D 顺序、四个 validity flags |
| `config/retarget_io.py` | 同一 ACTION_ORDER，但允许缺少 parts；只有双手时输出 34D |
| `envs/robot_loader.py` | 按关节名 `[left_arm,right_arm,left_hand,right_hand]`，H1=38 / GR1=36 / G1=28D |
| `teleop/pipeline_data.py` | 另一 `RobotJointCommand`，arm arrays + hand joint-name dictionaries |

表中模块/语义为现状。PR2 应定义一个 application command 边界与
明确消费端转换；48D L21/TRON2A 命令不能直接送入 benchmark action space。
PR2 应明确 flatten order、schema/version、timestamp 与各侧 validity/hold policy，
并用实际机器人 joint-name 列表验证映射。PR1 未改动已有 `RobotCommand`，未宣称
本仓库已经完成全系统 48DOF 统一。

## Validity 与文件消费者

canonical angle H5 为每侧 `(T,18)` angles + valid；网络第 0 维是 root placeholder，
L21 command adapter 去掉 root 得到 17DOF。native-hand consumer 仍按 18D 索引映射，
不能先裁成 17D 后仍沿用原映射表。PR1 新增回归覆盖 H1/GR1/G1 的原有索引行为。

离线导出沿用无效帧 hold_previous；需要区分持有角度与新有效预测。
`simulation/pybullet_world.py::apply()` 现直接施加数组，未将 validity 用作控制策略。
PR2 应明确 missing-side、失手恢复、失效 arm IK 时的安全保持与有效性传播。
系统 `actions.h5`、dual-teleop structured H5、canonical hand angles H5 不应只靠维度猜协议。

## Joint indices、arm IK 与 TRON2A

`teleop/teleop_pipeline.py::TRON2_CONFIG` 的左右 arm indices 都是 `[0,1,2,3,4,5,6]`，
两侧 EE index 均为 7，注释仍按 xArm7 占位说明。PR2 应按实际加载 robot 的关节名解析，
消除模型/左右侧的索引歧义。

另一方面，`simulation/pybullet_world.py` 已按 TRON2A 关节名映射，
`retargeting/arm.py` 已有双侧 7DOF ETS/FK/IK、URDF chain 检查与 calibration。
保留这些目标实现，统一 VR arm observation、calibration、IK、command 与 benchmark
robot adapters 的入口，并验证左右 arm limits、坐标/末端安装和失败时行为。

TRON2A description 是外部资产，本次不迁移。测试通过 `TRON2A_TEST_URDF` 指向现有资源；
PR2 应记录可靠的资产获取方式与路径配置。旧 `scripts/replay_hand_angles.py` 单独依赖
`linkerhand_sdk/ros1/src/assets/robots/hands/linker_hand`，canonical FK 使用仓库内的
`dataset/robot/l21_left` / `l21_right`。资源入口的统一需要和 consumer 边界一起处理。

## 本次实际覆盖与后续验收

PR1 覆盖 VisionPro 三帧输入、NPY 缺帧恢复、旧 dual-arm 单测与 root CLI、native-hand
映射，以及已有 benchmark 无头仿真路径。真实摄像头、VR、手套、硬件控制与物理尺度
尚无实机验收；生成的 synthetic checkpoint 只验证工程链路。

PR2 至少应增加：48D 与各机器人 action 的显式转换测试、joint-name/index 一致性、
validity 传播、左右 arm 独立控制、真实 TRON2A scene replay 与 recorder round trip。
