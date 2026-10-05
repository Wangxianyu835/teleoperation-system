# Canonical hand-retargeting subsystem

teleoperation-system 是 production hand retargeting 的唯一源码维护仓库。本子系统合入自
`mytrans@138fc2d9cc11580421c7094c5f4b8c703a690b82`；mytrans 只保留历史参考，后续
production 修复、训练协议和推理接口均在本仓库维护。机器人、双臂、任务、benchmark、
recording 和应用级 teleop 继续由本仓库原有模块负责。

## 协议与模块

正式链路：MediaPipe 原始单帧 → Hand25 → 声明的坐标变换 → 每侧独立三帧窗口 →
共享 PoseTransformer → 每侧 18D L21 角度 → 消费端按机器人映射。

| Hand25 索引 | 语义 |
|---|---|
| 0 | wrist |
| 1–4 | thumb |
| 5 / 6–9 | index palm root / index |
| 10 / 11–14 | middle palm root / middle |
| 15 / 16–19 | ring palm root / ring |
| 20 / 21–24 | pinky palm root / pinky |

模型输入固定为 `[B,3,25,3]`。网络输出为 `[B,18]`，第 0 维是 root placeholder，
其余 17 维为真实 L21 hand DOF。`retargeting.simulation.angle18_to_dofs()` 去掉 root；
`angle18_to_nodes()` 添加 5 个固定指尖节点用于原有 FK。不要把模型输出改成 17D。
`teleop.native_hand` 继续按原来的 18D 索引映射到 H1-2 / GR1-T2 / G1 原生手。

| 模块 | 职责 |
|---|---|
| `tracking.py`, `hand_core.py` | MediaPipe21 → Hand25、缺帧重置、三帧窗口 |
| `coordinates.py` | legacy 和 palm-local 几何与标识验证 |
| `preprocessing.py` | 唯一正式 H5 preprocessing，事务式写入 |
| `data.py`, `training.py` | H5 Dataset、训练与 checkpoint metadata |
| `model.py`, `inference.py` | 严格 checkpoint 加载与离线导出 |
| `retargeter.py` | `HandWindow` → `HandCommand` 可复用接口 |
| `inputs/mediapipe.py` | 原始相机输入与资源生命周期 |
| `mediapipe.py`, `visualization.py` | palm-local realtime 与可选可视化 |
| `cli.py`, `runtime.py`, `config.py` | hand CLI、设备选择、hand-specific 配置 |

`HandCommand` 的 validity 只描述手部输出；整机 RobotCommand 与 48DOF 协议留待 PR2。
`config/retarget_io.py` 的 hand 校验/构建委托 canonical 实现；原有 arm 与 action 打包行为保留。
NPY replay 同样复用 `CanonicalHandProcessor`，缺失侧立即清空窗口，恢复后重新积累三帧。

## 两种坐标模式

| 标识 | 变换 |
|---|---|
| `source_to_l21_xyz` | legacy `points @ SOURCE_TO_L21_MATRIX.T`，保留 `align_source_hand_coordinates()` |
| `palm_local_to_l21_v1` | wrist-relative Hand25 `points @ B_source @ B_robot.T` |

palm-local 的 source MCP 为 **6 / 11 / 16 / 21**。左右手分别从各自 L21 零位 FK 构建
robot basis。每帧独立构建 source basis，正交且 determinant +1，保持点间距离。
source 退化帧输出 zeros 并作为无效帧；robot reference 退化直接失败。不叠加 legacy
matrix，不归一化尺度，不平滑，不复用上一帧 basis。Dataset/模型原有显式 source scale
仍然保留；它不是 palm-local 对齐时的尺度归一化。

## 安装与资源

应用依赖仍在 `requirements.txt`。根据用途安装附加文件：

```powershell
python -m pip install -r requirements-retargeting.txt
python -m pip install -r requirements-retargeting-training.txt
# 相机或诊断可视化按需：
python -m pip install -r requirements-retargeting-camera.txt
python -m pip install -r requirements-retargeting-tools.txt
# 完整测试，包括双臂 FK / IK：
python -m pip install -r requirements-dev.txt
```

runtime 包含实际 FK/模型所需 numpy、scipy、h5py、torch、einops、timm、urchin、trimesh；
TensorBoard 单独归 training；MediaPipe 与 OpenCV contrib 归 camera；Matplotlib 归 tools；
PyBullet 仍归 application，roboticstoolbox/spatialmath 归完整开发测试。OpenCV 选择 contrib
发行包，避免同一环境同时安装多个提供 `cv2` 的 OpenCV 包。

L21 FK 使用仓库已有 `dataset/robot/l21_left/` 与 `dataset/robot/l21_right/` 资产。相机 Tasks asset、训练输入与
checkpoint 需要自行提供，不随此次源码合并迁入。离线 export 保留目标仓库历史默认
`checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth`；默认路径存在与否
不构成 metadata，仍须声明正确 alignment。新训练输出默认在忽略的
`outputs/hand_retargeting/checkpoint/`。realtime 默认查找该目录的 `palm_local_v2` run；
可通过 `--checkpoint` 指定任何严格声明 palm-local 的权重。

## H5 → train → export

原始两手 H5 的 root schema：

```text
frame_ids              (T,)
timestamps             (T,)
left_hand_keypoints    (T,21,3) 或 (T,25,3)
right_hand_keypoints   (T,21,3) 或 (T,25,3)
```

亦支持既有 group 中的 `l_glove_pos` / `r_glove_pos` schema。未检测到的手用全零帧表示。
原始 palm-local preprocessing 接受 `source_landmark_space=mediapipe_normalized` 或未声明；
显式声明其他 landmark space 会失败。

```powershell
python -m retargeting align --input input/raw.h5 --output outputs/hand_retargeting/palm.h5
python -m retargeting train --input outputs/hand_retargeting/palm.h5 --run-name palm_local_v2 --device cpu
python -m retargeting export --input outputs/hand_retargeting/palm.h5 --checkpoint outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_local_v2/model_best.pth --output outputs/hand_retargeting/palm_angles.h5 --device cpu
python -m retargeting inspect --angle-h5 outputs/hand_retargeting/palm_angles.h5
```

`align` 只生成 palm-local H5；输入文件不得覆盖，已声明 alignment 的输入拒绝二次对齐。
输出先写入同目录临时文件，关闭 HDF5 后原子替换，失败时保留已有输出。
旧 `scripts/align_h5_coordinates.py` 现在是该命令的兼容 wrapper，**其输出模式是 palm-local**。
legacy 已对齐数据继续受支持；若需要新建 legacy 数据，显式调用保留的 legacy 坐标函数
并写入真实 metadata，不能只修改 palm-local 文件的属性来伪装另一模式。

已对齐 H5 必须在 root 声明 `coordinate_frame=l21` 和受支持的 `coordinate_alignment`。
完整链路严格要求：

```text
H5 alignment == Dataset alignment == training alignment
             == checkpoint alignment == inference expected alignment
```

legacy H5 + legacy checkpoint、palm H5 + palm checkpoint 通过；交叉组合失败。
missing / empty / unknown H5 alignment、checkpoint missing alignment、checkpoint mismatch
均失败。没有 force、ignore、warning 后继续或根据路径猜测模式的选项。
`--init-checkpoint` warm start 同样校验，训练默认不隐式加载历史权重。

导出包含 `frame_ids` / `timestamps`、每侧 `(T,18)` angles 与 `(T,)` valid，以及显式
坐标 metadata。窗口初始两帧无效；缺帧/退化帧重置窗口。导出无效角度沿用既有
`hold_previous` 策略，消费端必须读取 valid，不能把持有值当成新预测。

## MediaPipe realtime 与应用入口

```powershell
python -m retargeting realtime --model-asset-path hand_landmarker.task --checkpoint outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_local_v2/model_best.pth --device cpu --frames 120
# 可选可视化：追加 --visualize；无窗口截图再追加 --headless --snapshot outputs/hand_retargeting/realtime.png
```

realtime 按 MediaPipe handedness 处理，identity tracking 关闭；每侧独立积累三帧，
失手立即重置，无 previous-frame fallback。checkpoint 必须是 palm-local。
默认打印原始点、窗口、18D 输出与恢复计数；真实相机检测和实物精度仍需人工验证。

root `main_train_twohand.py` / `main_offline_twohand.py` 保留为 package CLI wrapper。
`main.py`、benchmark、dual-arm roots、`teleop/`、`simulation/`、robot adapters、tasks、
recorder 保留目标应用架构。hand angles H5 与系统 `actions.h5` 是不同消费协议；整机
flatten order 与 validity 的统一工作见 [PR2 follow-up](PR2_FOLLOW_UP.md)。

```powershell
python scripts/replay_hand_native.py --file outputs/hand_retargeting/palm_angles.h5 --robot h1_2 --hand both --loop 1 --no-repair --substeps 1
```

## 验证

```powershell
# 完整测试需要外部 TRON2A description；设置为自己已有的 URDF：
$env:TRON2A_TEST_URDF='D:/path/to/tron2a/DACH_TRON2A/urdf/robot.urdf'
python -B -m unittest discover -s tests -v
python -m retargeting --help
```

TRON2A 测试默认仍查找 `third_party/tron2-robot-description/.../robot.urdf`。
本 PR 不迁移第三方资产。未提供 URDF 的环境中，相关三个测试会报缺资源。
真实历史 palm-local checkpoint 等价性测试在权重缺失时明确 skip；不伪造/迁移该权重。
PR1 的完整结果与实际 smoke 见 [合并报告](PR1_HAND_CONSOLIDATION_RESULT.md)。
