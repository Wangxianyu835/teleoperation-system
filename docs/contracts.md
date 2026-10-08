# 观测、结果与动作 contracts

类型采用简单 dataclass，避免将传感器观测、模型输出和执行动作混为同一个数组。手部类型不自动改变 dtype、裁剪角度或归一化数值；原有 TRON2A 命令封装仍保留其 float32 行为。

## 输入协议

```python
InputSource.next_observation() -> RawObservation | None
InputSource.close() -> None
```

`None` 表示本次没有新样本；EOF 使用 `StopIteration`。未检测到双手仍返回 `RawHandFrame`，其双侧值为 `None`。读取错误明确失败。close 幂等，仅关闭采集资源。

| 类型 | 数据与声明 |
|---|---|
| CameraFrame | RGB 图像、timestamp、frame_id |
| RawHandFrame | hands 左右侧、timestamp、source、metadata、geometry_encoding |
| UpperBodyObservation | arms、valid、timestamp、coordinate_frame、units |
| RawTeleopObservation | hands: RawHandFrame、upper_body: UpperBodyObservation |

`RawHandFrame.geometry_encoding` 表达每侧 `landmarks21`、`landmarks25` 或 `transforms25`。一般 landmarks 编码也支持历史 21/25 点。MediaPipe 原始点为 `(21,3)`，Vision Pro 原始变换矩阵为 `(25,4,4)`，Replay 保留文件内的点值。输入不执行轴交换、左右符号变换、归一化或拓扑转换。

缺失、非有限数据、退化几何分别在设备解析、单帧处理和正式算法入口按原有策略处理；不能将它们伪装成窗口未就绪。

## 手部结果

| 类型 | 语义 |
|---|---|
| CanonicalHandFrame | 每侧 `(25,3)` 或缺失；当前帧有效性与原因 |
| HandWindow | 每侧 `(frames,25,3)` 或缺失；生产模型三帧 |
| L21HandAngles | 单侧 `values.shape == (18,)`；保留 root placeholder |
| HandRetargetResult | 左右侧 L21HandAngles、左右有效标志、timestamp |
| L21HandDOFs | 单侧有限的 `values.shape == (17,)`，真正可动关节 |

正式接口：

```python
canonical = processor.process_frame(raw)
window = temporal.append(canonical)
result = retargeter.retarget(window)  # 仅在 window 非空时
```

单帧处理器拥有跟踪状态，时间缓冲独立。缓冲器只接收 CanonicalHandFrame，不重新做坐标或 21→25。

18D 顺序为占位维，加食指、中指、无名指、小指各三维，再加拇指五维；映射常量位于 `robots/l21.py` 和 `robots/native_hand.py`。root placeholder 不作为 actuator 下发。

```python
# 原 TRON2A/L21 路径：保持旧转换的 float32 行为
L21HandDOFs(angle18_to_dofs(angles.values))

# 原生回放：保留 float64 修复和映射精度，只切除占位维
angles.native_mapping_dofs()
```

两条路径不能合并成隐式 dtype 转换。原生映射的符号、裁剪、rescale 和缺失关节行为保持既有策略。

## 双臂三阶段

| 类型 | 字段和校验 |
|---|---|
| UpperBodyObservation | 每侧 shoulder/elbow/wrist 固定顺序 `(3,3)` 或 None；valid；坐标与单位未知时保持 unknown |
| ArmTargetPose | side、robot、coordinate_frame、pose `(7,)`、valid；xyz + qx qy qz qw；有限值 |
| ArmIKResult | side、robot、q `(7,)`、success、position_error、orientation_error；q 有限；失败允许 inf 误差 |

```python
target = arm_retargeter.update(observation, side)
result = arm_ik.solve(target, q_previous, q_reference)
```

实时应用调用 `UpperBodyObservation.validate(strict=True)`；非法点明确失败。离线 valid=False 或非有限观测保持上一目标。ArmTargetPose 不自动归一化或翻转四元数。标定输出属于 TRON2A `base_Link`。

IK 的 FK 矩阵仍可参与原有计算；进入正式 IK 接口前用 `target_pose_from_matrix(matrix, side)` 显式构造目标。IK 失败返回 previous q、success=False、两项误差 inf。安全控制继续使用原来的 q、success、timestamp 更新限制和保持策略。

## 机器人命令

| 类型 / 适配器 | 动作顺序 | 维度 |
|---|---|---|
| Tron2AL21Command / Tron2AL21ActionAdapter | LA7、LH17、RA7、RH17 | 48 |
| H1JointTargets / H1ActionAdapter | LA7、RA7、LH12、RH12 | 38 |
| GR1JointTargets / GR1ActionAdapter | 既有 native action_joint_names，双侧手各 11 | 36 |
| G1JointTargets / G1ActionAdapter | 既有 native action_joint_names，双侧手各 7 | 28 |

后三者的语义目标为原生关节名到角度的 Mapping。adapter 的 joint_names 必须使用对应 RobotLoader 的实际 action_joint_names，验证维度、唯一性、所需关节与有限值。输入手臂属于该目标机器人；不接受 Tron2AL21Command 作为另一台机器人的动作。

`NativeHandAdapter` 接收 L21HandDOFs，仅映射手部目标。其返回关节名称与角度，不解释人体 landmarks，不承担手臂转换。

辅助与正式拼接：

```python
pack_partial_limb_vector(...) -> PartialLimbVector
pack_tron2a_l21_action(...) -> np.ndarray[48]
```

PartialLimbVector 包含 values 与 limbs，没有 `__array__` 或隐式升级。其 values 保留旧的忽略缺失 limb 拼接结果，partial 对象即使偶然等长也不能编码或执行。正式拼接要求左右臂各 7、左右手各 17，任何缺失明确失败。

SimulationEnv.step 保留 action=None；非空动作必须为有限、正确维度的一维 ndarray。错误在调用 task.apply_action 和 stepSimulation 之前抛出。

## 双臂输入 factory

推荐返回 `RawTeleopObservation`。应用边界也显式接受下面的原始字典：

```python
{
    "hands": {"left": raw_left, "right": raw_right},
    "arms": {"left": shoulder_elbow_wrist_left, "right": None},
    "timestamp": timestamp_seconds,
    "source": "my_sensor",
    "metadata": {...},
    "geometry_encoding": {"left": "landmarks21", "right": "landmarks21"},
}
```

Vision Pro raw 必须声明 transforms25。HandWindow 或模型窗口不能冒充输入；采集 factory 不应隐藏 processor、buffer 或模型。

## 文件格式与声明

`data/hand_h5.py` 保留 root 格式的 `left_hand_keypoints/right_hand_keypoints`，以及历史 group 格式的 `l_glove_pos/r_glove_pos`。训练输入要求显式 `coordinate_frame=l21` 和受支持的 coordinate_alignment；未知声明不自动补全。

坐标标识：

- `source_to_l21_xyz`：既有 legacy 数学。
- `palm_local_to_l21_v1`：当前 palm-local 数学。

checkpoint 必须使用原有严格 state_dict 加载和相同 coordinate_alignment guard；保留参数名称和网络结构。不会通过迁移重新训练或补写历史声明。

严格 canonical angle 文件包含 frame_ids、timestamps、左右 angles `(T,18)` 与 valid `(T,)`。reader 校验 shape、时间线和有限值；后续保持算法由 retargeting 执行，`apps/replay/canonical.py` 连接读取与保持。

原生手 reader 保留 float64 和缺失 valid 时默认全有效的既有行为，独立于严格 canonical reader。诊断工具的旧可选 timestamps schema 也独立保留。

原生动作 H5/NPZ 使用原有 actions 和 metadata，不改为统一机器人协议。动作文件按实际目标机器人验证。

TRON2A command H5 保留 robot_command、左右 arm_q/hand_q、arm_ee_pose、valid、frame_ids、timestamps 和既有 metadata；`robot_command` 数据集名称为历史文件协议，与已退役的 Python RobotCommand 类型无关。

标定使用原有 JSON；模型文件序列化位于 data/checkpoint。记录器保留 episode H5 的关节位置/速度、时间戳、物体轨迹、相机首尾帧与成功 metadata。它接收 SimulationSample，不查询设备或 PyBullet。
