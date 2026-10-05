# 坐标系与单位

本文记录当前手部离线路径的坐标、单位及证据，不改变数值处理流程。

## 状态含义

- `CODE_DEFINED`（代码已定义）：仓库代码直接实现或声明。
- `DOCUMENTED`（文档已说明）：上游或官方文档中的说明。
- `VERIFIED`（已验证）：代码、文档与实际执行的可复现实验一致。
- `UNRESOLVED`（尚未确认）：现有证据不足。

只有描述了实际执行的实验，才可标记为“已验证”。下方保留官方 MediaPipe 链接作为参考；前次审计访问 `ai.google.dev` 受到限制，相关上游说明仍需人工浏览核对，不能直接升级为本项目的实测结论。

## MediaPipe 输入约定

已审计的本地采集脚本（独立采集项目中的 `hand capture media.py`）使用 MediaPipe Tasks Hand Landmarker 的 `VIDEO` 模式，保存 `result.hand_landmarks` 中的 `[lm.x, lm.y, lm.z]`，不读取 `result.hand_world_landmarks`。身体采集脚本使用相同路径。

| 核对内容 | 状态 | 证据与影响 |
|---|---|---|
| x 是按图像宽度归一化的水平位置 | 文档已说明 | [官方 Python 指南](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)；采集脚本绘图使用 `int(lm.x * width)` |
| y 是按图像高度归一化的垂直位置 | 文档已说明 | 同一官方指南；绘图使用 `int(lm.y * height)` |
| z 是以腕部为参考的相对深度，数量级大致与归一化图像坐标相当 | 文档已说明 | 输出约定；这不是经过标定的物理距离 |
| hand_landmarks 的单位是米或其他物理长度单位 | 尚未确认 | 官方约定描述归一化坐标和相对深度；仓库没有执行长度标定 |
| hand_world_landmarks 是以米表示、以手部为中心的世界三维坐标 | 文档已说明 | 官方输出说明；当前采集路径没有使用该输出 |
| 腕部索引为 0 | 文档已说明 | 官方 21 点拓扑；`wrist_relative()` 减去点 0 |
| 当前代码是否使用世界关键点 | 代码已定义：未使用 | 两个采集器仅读取 `result.hand_landmarks` |
| 当前非镜像采集下的左右手标签是否正确 | 尚未确认 | 上游 handedness 说明涉及自拍镜像假设；采集脚本明确不镜像，但尚无真实左右手摄像头实验记录 |

采集代码按 `result.handedness[i][0].category_name` 分配：`"Left"` 写入 `left_hand`，其他标签写入 `right_hand`。脚本注释“不镜像！左右手正确”表达实现意图，不是物理验证证据。

## 关键点规范化与离线对齐

离线 loader 默认要求 H5 根属性 `coordinate_frame="l21"`。显式预处理脚本 `scripts/align_h5_coordinates.py` 依次执行：

1. `ensure_hand25()`：21 点转为项目的 25 点拓扑，并减去腕部位置。
2. 对双手调用 `align_source_hand_coordinates()`。
3. 写入 `coordinate_frame="l21"` 和 `coordinate_alignment="source_to_l21_xyz"`。

loader 只校验 coordinate_frame，不校验 coordinate_alignment；这是当前代码行为。

`retargeting/coordinates.py` 对行向量定义：

```text
p_aligned = p_source @ M.T
M = [[ 0, -1,  0],
     [ 0,  0,  1],
     [-1,  0,  0]]

x' = -y
y' =  z
z' = -x
```

变换及名称已由代码定义。Git 提交 `319e9f5`（2026-09-29，“加入坐标系对齐”）引入该固定矩阵；提交说明和仓库检索未提供论文、上游 SDK 或标定来源。此前重构提交 `767c2df` 也未提供外部推导，因此物理正确性仍未确认，不能将提交历史作为验证证据。

## L21 基坐标证据

当前 URDF：

- `dataset/robot/l21_left/linkerhand_l21_left.urdf`
- `dataset/robot/l21_right/linkerhand_l21_right.urdf`

两者的 hand_base_link 可视和碰撞原点均为 `xyz="0 0 0"`、`rpy="0 0 0"`，使用 `meshes/hand_base_link.STL`。这是代码定义的几何信息。URDF 没有声明 +X/+Y/+Z 分别对应指向、掌面、手背等解剖方向；单靠网格轴和关节轴不能唯一确定这些含义，需要网格可视化与已知姿态实验。

完整关节证据见 [L21 关节契约](L21_JOINT_CONTRACT.md)。

## 尺度与单位

| 阶段 | 数值行为 | 状态 |
|---|---|---|
| MediaPipe 输入 | 复制 hand_landmarks 的归一化 x/y/z | 代码已定义；物理单位尚未确认 |
| 腕部相对化 | `(points - points[0:1]) * scale_factor`，默认 1.0 | 代码已定义 |
| H5 数据集 | CanonicalHandProcessor 转换 21→25，减去腕部并应用配置尺度 | 代码已定义 |
| 训练源尺度 | `L21.training.source_scale = 1.0` | 默认值已定义；物理合理性尚未确认 |
| FK 几何 | URDF 原点长度约为 0.018 到 0.141，文件未声明长度单位 | 数值已定义；是否为米尚未确认 |
| 训练机器人尺度 | `L21.training.robot_scale = 1.0`，传入 FK | 默认值已定义；与源尺度是否匹配尚未确认 |

没有已记录的标定步骤将归一化人手跨度映射到 URDF 长度。1.0 只是默认系数，不代表已证明的物理尺度匹配。

### 损失对尺度的敏感性

| 损失 | 实现 | 尺度影响 |
|---|---|---|
| `vec_inter_loss` | MCP/DIP 相关向量差后使用 F.normalize | 归一化后主要比较方向；退化或近零向量例外 |
| `tip_pos_loss` | 指尖减 MCP 后使用 F.normalize | 主要比较方向 |
| `thumb_loss` | 点到平面距离，机器人目标距离乘 0.9 | 对相对长度尺度敏感 |
| `tip_distance_loss` | 成对指尖距离各乘 1000.0 | 对绝对长度及两侧尺度匹配敏感；该乘数不能证明单位是毫米 |
| `thumb_loss2` | 归一化拇指节段之间的角度 | 主要比较角度；退化向量例外 |
| `CollisionLoss` | FK 点距与 threshold=0.010 比较 | 对 FK 长度单位和尺度敏感 |

没有损失将全部笛卡尔距离统一归一化到手部尺寸。源尺度与机器人尺度的兼容性尚未确认。

## 已自动验证的软件性质

`test_coordinate_modes.py`、`test_coordinate_contracts.py` 和 `test_coordinate_diagnostics.py` 覆盖：

- 基向量 +X→-Z、+Y→-X、+Z→+Y，矩阵正交、行列式为 +1，并识别人为注入的行列式 -1 反射。
- 非共面合成左右手在转换、腕部相对化、旋转和带标签窗口中保留有向体积。有向体积只检查方向性，不是物理左右手分类器。
- 腕部相对化与旋转保留距离；source_scale 传到模型窗口和最新帧目标，两侧 FK 都应用 robot_scale。
- 共同缩放位置使指尖 MSE 乘以系数平方；碰撞阈值仍为缩放后 FK 单位中的 0.010。

零长度或数值无法分辨的拇指节段按 [验证报告](P0_VERIFICATION_REPORT.md) 中的约定屏蔽。只读诊断报告单位声明、缺失元数据、代表距离、有向体积与 FK 几何，不修改输入：

```text
python scripts/diagnose_hand_coordinates.py --input input/aligned_visual_hand_data_20260912_153542.h5
```

## 物理验证所需证据

最小实验见 [待验证清单](UNRESOLVED_VERIFICATION_PLAN.md)。实验完成前，不将当前矩阵、左右手标签或尺度标记为物理正确。
