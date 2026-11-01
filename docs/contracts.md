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

MediaPipe 的原始手部观测保持 `wrists=None`，不再通过关键点推算独立腕朝向。
`RawHandFrame.wrists` 仍为可选公共字段，供 Vision Pro 使用；相机组合观测的
`RawTeleopObservation.wrists` 来自 SMPLer-X。腕控制不接受旧 `camera_palm` 来源或
`mediapipe_hand_world_axes` 参考系。

缺失、非有限数据、退化几何分别在设备解析、单帧处理和正式算法入口按原有策略处理；不能将它们伪装成窗口未就绪。

## 手部结果

| 类型 | 语义 |
|---|---|
| CanonicalHandFrame | 每侧 `(25,3)` 或缺失；声明坐标和单位 |
| HandWindow | 每侧 `(frames,25,3)` 或缺失；新相机 profile 严格三帧 |
| HandJointResult | hand_model、side、spec_hash、有序独立 joint_names、positions[D] rad、timestamp、valid/error、solve_ms |
| HandJointCommand | 已接受的原生具名展开关节命令；与原始解分开 |

```python
normalized = normalize_hand_window(metric_window)
results = retargeter.retarget(normalized)  # dict[(hand_model,side), HandJointResult]
command = pipeline.update(results[(hand_model,side)], timestamp=normalized.timestamp)
executor.apply(command)
```

新掌长 profile 为 `camera_palm_length_window3_v1`，坐标框架 `hand_canonical_right`，units/length_unit 为 `1`，原始来源仍为 `mediapipe_world_meters`。分母单独记录为 `palm_length_m`。目标 specs 确定 D、关节顺序与 hash；每个 hand_model 的 shared NN 使用右侧 canonical 输出语义，运行时转换为实际 left/right 的原生结果，dex 仍按目标侧独立求解。无效结果 positions=None；缺手保持命令，但不把保持值标记为有效求解。

L21 为 D=17，H1/GR1 为 D=6（展开 12/11），G1 为 D=7。旧 L21 18D 占位布局只由明确文件/FK/权重边界读取；`legacy_l21_dofs` 保留 float64 精度，旧 TRON2A 消费端显式选择 float32。没有旧 L21 和 Native 两套公共结果包装。

新 H5 schema `hand_joint_results_v1` 将 raw_positions[D] 和 command_positions[展开数] 分开；组中记录 hand_model/side/spec_hash 和两个有序关节表，根部保留 timestamps/frame_ids、输入及算法 metadata。回放验证完整身份、有限性、有效界和耦合，不重新滤波。旧 L21 和 `dex_native_results_v1` 仅走明确旧格式读取器；有损旧映射对照文件不能作为原生命令回放。

权重 schema `hand_joint_nn_v1` 严格校验目标与输入契约，不依赖维数相同就认定兼容。新版 shared checkpoint 声明 `side=shared`、`hand_sides=[left,right]`、`training_robot=right`、`output_coordinates=canonical_right` 和 `angle_convention=l21_canonical_right_angles_v1`，路径为 `<hand_model>/shared/nn/<profile>/<run>/`。旧 side-specific checkpoint 仍保留明确的单侧兼容边界。路径与字段详见 [统一流程](hand-retargeting-unified.md)。

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

新相机身体输入由独立 SMPLer-X worker 提供。`source=smplerx`、
`source_coordinate_frame=smplx_camera_root_relative_m`、`source_units=m` 为必需声明。
身体与手部时间为独立单调秒，不能把缓存身体结果标成当前相机帧。
跟踪器对每侧管理 valid、fresh、sample_id、age 和状态；无效侧保持，不阻止另一侧或手指。
新 H5 保存原始模型包及最终观测，不接受旧 Pose 文件或缺少来源/模型 hash 的数据。
回放入口和实时时间规则见 [SMPLer-X 链路](smplerx-arm.md)。

IK 的 FK 矩阵仍可参与原有计算；进入正式 IK 接口前用 `target_pose_from_matrix(matrix, side)` 显式构造目标。IK 失败返回 previous q、success=False、两项误差 inf。安全控制继续使用原来的 q、success、timestamp 更新限制和保持策略。

## 机器人命令

| 类型 / 适配器 | 动作顺序 | 维度 |
|---|---|---|
| Tron2AL21Command / Tron2AL21ActionAdapter | LA7、LH17、RA7、RH17 | 48 |
| H1JointTargets / H1ActionAdapter | LA7、RA7、LH12、RH12 | 38 |
| GR1JointTargets / GR1ActionAdapter | 既有 native action_joint_names，双侧手各 11 | 36 |
| G1JointTargets / G1ActionAdapter | 既有 native action_joint_names，双侧手各 7 | 28 |

后三者的语义目标为原生关节名到角度的 Mapping。adapter 的 joint_names 必须使用对应 RobotLoader 的实际 action_joint_names，验证维度、唯一性、所需关节与有限值。输入手臂属于该目标机器人；不接受 Tron2AL21Command 作为另一台机器人的动作。

历史 `NativeHandAdapter` 接收有限的 L21 17D NumPy 坐标，仅映射手部目标。其返回关节名称与角度，不解释人体 landmarks，不承担手臂转换。

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

`dual realtime --hand-preprocessing legacy|palm-local` 显式选择模型输入契约，默认保留 legacy。
MediaPipe 原始输入需要 `source="mediapipe_approx"`、`metadata.source_landmark_space="mediapipe_normalized"` 和 landmarks21 编码，使用 `palm-local` 及相同坐标声明的 checkpoint。
已经对齐的 landmarks 必须声明 `metadata.coordinate_frame="l21"` 与 `metadata.coordinate_alignment`；palm-local 已对齐输入必须为 `(25,3)`，直接进入缓冲，不再对齐。
Legacy 保留原 CanonicalHandProcessor 的拓扑、wrist-relative 与 identity tracking 行为；接受明确声明的 legacy landmarks，或 `source="visionpro"` 的原始 transforms25。
未知输入空间、预处理/数据/checkpoint 不匹配均报错。单侧 palm-local 非有限或退化输入清除该侧历史，恢复须重新积累三帧。
加载 checkpoint 后，`TwoHandRetargeter.predict` 在模型调用前逐次核对 HandWindow 声明与加载时确认的 alignment 完全相同。
手部 frame timestamp 与 timestamp_unit 原样进入 HandWindow；hand 摄像头应用仍把 raw Unix 毫秒另存为 raw_timestamp_ms 并产生 relative seconds。dual 的 command timestamp 沿用上身观测的秒时间戳，adapter 应明确声明并同步两个时间线。

## 文件格式与声明

`data/hand_h5.py` 保留 root 格式的 `left_hand_keypoints/right_hand_keypoints`，以及历史 group 格式的 `l_glove_pos/r_glove_pos`。训练输入要求受支持的 coordinate_alignment 和对应 coordinate_frame；旧原生根坐标声明 l21，新 world 声明 hand_canonical_right。未知声明不自动补全。

坐标标识：

- `source_to_l21_xyz`：既有 legacy 数学。
- `palm_local_to_l21_v1`：历史四 MCP 归一化文件，仅用于匹配旧权重。
- `palm_local_to_l21_v2`：新的腕点/食指/中指 SVD 掌面，归一化点对齐至各侧原生 L21 根坐标。
- `mediapipe_world_palm_local_to_l21_v1`：历史四 MCP world 原生根坐标，仅用于匹配旧权重。
- `mediapipe_world_wrist_index_middle_canonical_right_v1`：新默认 world，腕点归零、0/5/9 三点 SVD、左右规范为右手表示，尺度固定 1.0。

### MediaPipe world 手部契约

`RawHandFrame.hands` 保持归一化 21 点，`world_hands` 保存同一次检测、同一手索引的米制 21 点。输入层 `include_world=True` 不替换 hands，不做空间转换。
raw metadata 的 `source_landmark_space=mediapipe_normalized` 描述 hands；`world_landmark_space=mediapipe_world_meters`、`world_length_unit=m`、`world_origin=hand_geometric_center` 描述 world_hands。
归一化数据和 world 数据独立判定有效性，因此坏 world 点不会被归一化点替代，归一化预览也可保留。

`hand record` 原始 H5 的主字段 `left_hand_keypoints/right_hand_keypoints` 为归一化 `(T,21,3)`，新增 `left_world_landmarks/right_world_landmarks` 为米制 `(T,21,3)`。
world 字段的 dataset attrs 声明 `source_landmark_space=mediapipe_world_meters`、`length_unit=m`、`origin=hand_geometric_center`；root 使用 `hand_world_landmark_space`、`world_length_unit`、`hand_world_origin`。
`hand_processing_landmark_space` 用 `mediapipe_world_meters` 或 `mediapipe_normalized` 指定默认处理来源，**不改变主字段的含义**。raw 缺手保存全零行，时间戳为 Unix 秒。
`record-world` 保留身体字段、NaN 缺失编码和 Unix 毫秒时间戳，只补充默认处理空间 world 的声明。

`hand align --landmark-space world|normalized` 可显式选择来源；省略参数时读取上述 root 声明，旧文件没有声明则选择 normalized。

相机手部录制不再生成 `left/right_wrist_orientation_xyzw`、
`left/right_wrist_orientation_valid` 及其相机腕版本/坐标元数据。
旧 H5 中的这些附加字段不作为腕控制输入；手部 reader 继续读取关键点与时间戳。
对齐复制可以保留旧附加字段，原始文件不被重写。
组合身体录制在写入和回放时过滤旧 MediaPipe 腕观测，SMPLer-X 与 Vision Pro 腕数据保留。
Vision Pro 的 `wrist_reference_frame` 等同名声明仍由其专用 reader 校验。

world 对齐要求两套 world dataset 均存在，并声明米制来源；单侧零行、非有限值和退化掌面变为无效零行，整套字段缺失时报错。
新输出另存为训练 H5：主字段成为所选来源的 `(T,25,3)`；原始 world 21 点及身体字段原样复制。原始文件保持不变，拒绝对已声明 alignment 的输入再次对齐。

world 对齐 H5、CanonicalHandFrame、HandWindow 和 checkpoint 必须共同声明：

```text
coordinate_frame = hand_canonical_right
coordinate_alignment = mediapipe_world_wrist_index_middle_canonical_right_v1
palm_basis_version = wrist_index_middle_svd_v1
canonical_hand = right
handedness_normalization = dex_left_then_y_reflection
source_landmark_space = mediapipe_world_meters
length_unit = m
units = m
source_scale = 1.0
```

坐标是腕部相对的米制坐标，保留原人手尺寸。中间姿态 R 是行列式 +1 的旋转；右侧乘 dex 右矩阵，左侧乘 dex 左矩阵再做 Y 镜像。左侧总变换行列式 -1，明确用于统一手性；旋转及镜像均保持距离。最终 +Y 朝食指侧，+Z 为腕点到中指 MCP，+X 为右手规范掌面法向。MediaPipe 手部几何中心原点不能当作相机或身体的全局原点。
world Dataset 不重新对齐、重新缩放或身份重分配；实时窗口独立按侧清空，失效恢复后需要连续三帧。
world 数据的额外输入缩放、非米制单位和不完整 checkpoint 声明均报错。normalized palm-local 允许历史 checkpoint 缺少来源声明，但拒绝将显式 world 来源配到 normalized 契约。
训练和导出只认文件声明，初始化模型和实时模型都必须与输入契约匹配；交叉使用在加载权重或调用模型前拒绝。

新 shared checkpoint 还声明 `side=shared`、`hand_sides=[left,right]`、`training_robot=right`、`output_coordinates=canonical_right`、`angle_convention=l21_canonical_right_angles_v1`。左右窗口都参与训练，但使用右侧目标 spec 的 FK 和 canonical 输出语义；运行时按目标手型转换到原生侧角度。L21 左侧四个 MCP roll 反号，其他角度不变；其他登记手型按其左右 URDF 限位方向适配。

checkpoint 必须使用原有严格 state_dict 加载和相同 coordinate_alignment guard；保留参数名称和网络结构。不会通过迁移重新训练或补写历史声明。

严格 canonical angle 文件包含 frame_ids、timestamps、左右 angles `(T,18)` 与 valid `(T,)`。reader 校验 shape、时间线和有限值；后续保持算法由 retargeting 执行，`apps/replay/canonical.py` 连接读取与保持。

原生手 reader 保留 float64 和缺失 valid 时默认全有效的既有行为，独立于严格 canonical reader。诊断工具的旧可选 timestamps schema 也独立保留。

原生动作 H5/NPZ 使用原有 actions 和 metadata，不改为统一机器人协议。动作文件按实际目标机器人验证。

TRON2A command H5 保留 robot_command、左右 arm_q/hand_q、arm_ee_pose、valid、frame_ids、timestamps 和既有 metadata；`robot_command` 数据集名称为历史文件协议，与已退役的 Python RobotCommand 类型无关。

标定使用原有 JSON；模型文件序列化位于 data/checkpoint。记录器保留 episode H5 的关节位置/速度、时间戳、物体轨迹、相机首尾帧与成功 metadata。它接收 SimulationSample，不查询设备或 PyBullet。
