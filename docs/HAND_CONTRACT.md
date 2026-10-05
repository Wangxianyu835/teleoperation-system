# L21 手部数据契约与处理流程

本文记录当前实现的数据格式与行为。物理坐标和单位证据见 [坐标说明](COORDINATE_SYSTEMS.md)，关节轴与限位见 [L21 关节契约](L21_JOINT_CONTRACT.md)。软件约定不代表已经完成物理标定。

## 数据形状

| 层级 | 形状 / 类型 | 含义 |
|---|---|---|
| MediaPipe 检测 | `(21,3)` / float32 | `hand_landmarks` 的 x/y/z；未使用 world landmarks |
| Canonical 单帧 | `(25,3)` / float32 | 腕部归零，插入四个掌根中点，乘 source scale |
| Canonical 窗口 | `(3,25,3)` / float32 | 按时间从旧到新排列 |
| 模型输入 | `(B,3,25,3)` / torch.float32 | 左右手分别调用同一个共享模型 |
| 模型输出 | `(B,18)` / torch.float32 | 弧度；dim 0 固定为零，dim 1..17 为可动关节 |
| 训练 FK 输入 / 输出 | `(B,23)` / `(B,23,3)` | 18 维末尾补五个固定指尖零节点，得到 URDF 位置 |
| 仿真适配 | `(17,)` 或 `(23,)` | `angle18_to_dofs` 去掉 dim 0；`angle18_to_nodes` 补五个零 |

18 维顺序：根占位；食指、中指、无名指、小指各依次为 MCP roll、MCP pitch、PIP；最后是拇指 CMC roll、CMC yaw、CMC pitch、MCP、IP。完整名称与限位见 [17 个 URDF 关节表](L21_JOINT_CONTRACT.md#17-个可动-urdf-关节)。内部顺序尚无经验证的硬件协议映射。

## 21 → 25 拓扑

| 规范化点索引 | 来源 / 构造 |
|---:|---|
| 0 | MediaPipe 0（腕部） |
| 1–4 | MediaPipe 1–4（拇指） |
| 5 | `(MP0 + MP5) / 2` |
| 6–9 | MediaPipe 5–8（食指） |
| 10 | `(MP0 + MP9) / 2` |
| 11–14 | MediaPipe 9–12（中指） |
| 15 | `(MP0 + MP13) / 2` |
| 16–19 | MediaPipe 13–16（无名指） |
| 20 | `(MP0 + MP17) / 2` |
| 21–24 | MediaPipe 17–20（小指） |

`ensure_hand25` 接受 21 或 25 点；`wrist_relative` 减去点 0 并乘 `scale_factor`，默认 1.0。插值点不提供额外观测信息。

## 输入和训练 H5

`load_twohand_h5` 接受以下根数据集，或唯一一个包含 `l_glove_pos` / `r_glove_pos` 的旧格式组：

```text
left_hand_keypoints    (N,21,3) 或 (N,25,3)
right_hand_keypoints   (N,21,3) 或 (N,25,3)
frame_ids             (N,) 可选，默认 arange(N)
timestamps            (N,) 可选，默认 arange(N)
```

旧格式的可选向量位于同一组。输入转换为 float32，手部非有限值由后续逐帧处理判断有效性。默认加载要求根属性 `coordinate_frame="l21"`；对齐脚本还写入 `coordinate_alignment="source_to_l21_xyz"`，但 loader 不校验后者。

训练不读取目标关节角。`TwoHandH5Dataset` 生成 `(B,3,25,3)` 输入、`(B,1,25,3)` 最新帧几何目标和逐侧布尔 valid mask；无效侧使用零占位并被 mask 排除。

## 两条输入路径

离线：原始 H5 → 显式对齐 CLI → aligned H5 → `CanonicalHandProcessor` → 三帧窗口 → 共享模型 → 训练 FK/loss 或角度导出。对齐 CLI 先验证，再写同目录临时文件，关闭后原子替换；拒绝同文件输入输出及已经声明对齐的输入。失败保留已有输出。

实时：`MediaPipeCameraAdapter` 从 BGR 帧得到 RGB 检测，读取 handedness 和 `hand_landmarks` → `CanonicalHandProcessor.update_detections` → 身份关联 → 25 点转换 → 三帧窗口 → `TwoHandRetargeter.predict`。该实验适配器未接入手部 CLI，也没有执行离线 source-to-L21 旋转。

身份关联默认门限为掌心位移 0.08、腕部相对形状 RMSE 0.05。通过门限的候选按接受数量、标签匹配数量、连续性代价排序；当两手都满足门限时，标签仍可能决定分配。腕部相对 H5 丢失绝对位置，不能保证纠正标签互换。

缺失或被拒绝的侧由 processor/调用方清空三帧缓存，恢复需要三帧连续有效观测；`HandWindowBuffer` 本身不统一处理缺失重置。实时模型对缺失侧返回 `None`，保持上一姿态由下游控制器负责。

PoseTransformer 使用学习得到的三帧加权聚合，并非硬编码中心帧。训练才计算 FK；离线推理检查有限值和角度限位。六个损失的尺度行为见坐标说明，稳定性与训练安全门见 [验证报告](P0_VERIFICATION_REPORT.md)。

## 输出角度 H5

```text
frame_ids, timestamps  (N,)
left_angles            (N,18), float32, rad
right_angles           (N,18), float32, rad
left_valid             (N,), bool
right_valid            (N,), bool
```

属性包含输入、checkpoint、`output_shape=(18,)`、坐标/身份设置及 `invalid_angle_policy="hold_previous"`。无效行保持上一有效角度，开头无效行使用全零；valid 仍为 false。`iter_angle_h5` 检查六个数据集、形状和有限值。

手部 CLI 仅提供 train/export/inspect。双臂 48D 协议是额外消费者，共享 `contracts.py` 仍包含 arm 字段，`build_retarget_output` 要求双臂和双手。扩展入口及专用 H5 格式仅在 [README](../README.md#tron2a-双臂-teleoperation) 维护。
