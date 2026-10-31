# 多机器人机械臂控制

相机身体输入使用 SMPLer-X S32，MediaPipe Hands 提供手指关键点。
环境、资产、录制和回放命令见 [SMPLer-X 链路](smplerx-arm.md)。

## 控制流程

`CameraTeleopSource -> BodyTracker -> apply_camera_arm_frame -> RealtimeArmBridge / WristSceneController -> ArmSceneDriver -> PyBulletArmIK`。

H1-2、GR1-T2、G1 的参数由 `simulation/arm_profiles.py` 管理。
每台机器人使用独立 IK 适配器和自身连杆几何。TRON2A 使用独立的标定与控制入口。

## 映射方式

实时入口默认 `--arm-map posture`，用机器人刚体肩锚点和实测连杆长度构造目标：

```text
elbow = shoulder + upper_length * upper_direction
wrist = elbow + forearm_length * forearm_direction
```

`mapped` 使用轴向映射后的操作者方向；`delta` 将操作者相对中性的旋转作用到机器人中性方向。
`--arm-posture-frame auto` 按 profile 选择：H1-2、G1 为 `mapped`，GR1-T2 为 `delta`。
方向增益通过球面插值控制中性方向到目标方向的比例。

`chain` 使用肩到腕的相对位移；`segments` 分别按机器人与操作者的上臂、前臂长度比例缩放。
两种位移方式受配置的行程限制。`posture` 不使用位移行程框，但仍受关节限位、IK 与控制速率约束。

## 标定、腕姿态和执行

中性样本按侧收集并取中位数。实时默认 `track` 在窗口收集期间使用已有样本；
`hold` 等待窗口完整。身体失效时保持对应侧目标，恢复由跟踪器判断。
相机轴向映射是配置约定，不代表已完成相机外参标定。

腕旋转使用 SMPL-X FK，经 `WristSceneController` 相对操作者中性姿态对齐每台机器人的腕姿态。
`ArmSceneDriver` 将肘腕目标随当前肩位置平移，限制目标速度，并在刚性肘腕目标对过度缩短时保持上一对命令。

IK 支持腕位置、腕姿态和肘位置任务、关节限位、零空间参考及带滞回和冷却的肘支路恢复。
求解临时状态在返回前恢复；实际显示运动由 PyBullet 位置控制和物理步进产生。

## 诊断

`--arm-trace` 输出操作者位移和机器人目标；`--report` 保存映射、求解、腕状态及仿真时间统计。
离线检查见 [机械臂诊断](arm-diagnostics.md)，腕部人工检查见 [腕输入与验收](wrist-input-and-acceptance.md)。
