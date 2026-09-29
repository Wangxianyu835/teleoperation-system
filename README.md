# LinkerHand L21 双手重定向

本仓库只维护一条正式链路：

```text
MediaPipe / Vision Pro / 双手关键点 H5
-> 左右手身份连续性检查与三帧窗口
-> PoseTransformer
-> L21 正向运动学与训练损失
-> 双手 18 维角度 H5
-> 外部仿真适配
```

旧单手流程、真实硬件控制、Vision Pro 多进程控制、本地 PyBullet 环境和数据清洗脚本位于 `legacy/`，仅供查阅，不保证依赖完整或命令可运行。正式代码不依赖 `legacy/`。

## 环境与命令

推荐使用项目现有 Conda 环境：

```powershell
conda activate TransHandR
python -m pip install -r requirements.txt
python -m pip install -r requirements-teleop.txt
python -m retargeting --help
```

导出双手角度：

```powershell
python -m retargeting export `
  --input input/aligned_visual_hand_data_20260912_153542.h5 `
  --output output/twohand_angles_153542_aligned.h5
```

默认 checkpoint 为：

```text
checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth
```

训练共享左右手模型：

```powershell
python -m retargeting train `
  --input input/aligned_visual_hand_data_20260912_153542.h5 `
  --run-name my_run
```

原始关键点 H5 不能直接用于训练。先生成一个新的坐标对齐文件，原始文件不会被覆盖：

```powershell
python scripts/align_h5_coordinates.py `
  --input input/visual_hand_data_20260912_153542.h5 `
  --output input/aligned_visual_hand_data_20260912_153542.h5
```

该步骤对左右手都执行固定变换 `x'=-y, y'=z, z'=-x`，并在 H5 根属性中写入 `coordinate_frame=l21`。训练、推理和可视化只接受带有该标记的 aligned 文件。

检查角度 H5：

```powershell
python -m retargeting inspect `
  --angle-h5 output/twohand_angles_153542_aligned.h5
```

## 代码结构

```text
retargeting/
  __main__.py        # train / export / inspect
  config.py          # L21、模型、损失和相对路径配置
  contracts.py       # 统一输入协议
  coordinates.py     # 固定 source -> L21 坐标对齐
  data.py            # H5 读取、三帧窗口和 valid 规则
  tracking.py        # 关键点转换、身份关联和窗口缓存
  mediapipe.py       # MediaPipe 输入适配
  visionpro.py       # Vision Pro 输入适配
  model.py           # 双手共享模型封装
  inference.py       # checkpoint 加载、批量推理和导出
  training.py        # 训练、验证和 checkpoint 管理
  inspect.py         # 输出 H5 校验
  simulation.py      # 仿真侧稳定 API

model/
  pose_transformer.py
  kinematics.py
  losses.py
```

## 18 维角度定义

单位为弧度，顺序按手指分组。L21 有 17 个可动关节；dim 0 是与 FK 根节点 `hand_base_link` 对齐的固定零占位，不是手腕自由度。

| dim | 关节名 | 语义 |
|---:|---|---|
| 0 | `hand_base_link` | 固定零占位 |
| 1 | `index_mcp_roll` | 食指侧摆 |
| 2 | `index_mcp_pitch` | 食指 MCP 屈伸 |
| 3 | `index_pip` | 食指 PIP 屈伸 |
| 4 | `middle_mcp_roll` | 中指侧摆 |
| 5 | `middle_mcp_pitch` | 中指 MCP 屈伸 |
| 6 | `middle_pip` | 中指 PIP 屈伸 |
| 7 | `ring_mcp_roll` | 无名指侧摆 |
| 8 | `ring_mcp_pitch` | 无名指 MCP 屈伸 |
| 9 | `ring_pip` | 无名指 PIP 屈伸 |
| 10 | `pinky_mcp_roll` | 小指侧摆 |
| 11 | `pinky_mcp_pitch` | 小指 MCP 屈伸 |
| 12 | `pinky_pip` | 小指 PIP 屈伸 |
| 13 | `thumb_cmc_roll` | 拇指 CMC roll |
| 14 | `thumb_cmc_yaw` | 拇指 CMC yaw |
| 15 | `thumb_cmc_pitch` | 拇指 CMC pitch |
| 16 | `thumb_mcp` | 拇指 MCP |
| 17 | `thumb_ip` | 拇指 IP |

仿真适配 API：

```python
from retargeting.simulation import (
    angle18_to_dofs,
    angle18_to_nodes,
    iter_angle_h5,
)

dofs17 = angle18_to_dofs(angle18)    # 丢弃 dim 0
nodes23 = angle18_to_nodes(angle18)  # 末尾补 5 个固定指尖零
```

## H5 协议

输入关键点 H5：

```text
frame_ids
timestamps
left_hand_keypoints   (N, 21, 3) 或 (N, 25, 3)
right_hand_keypoints  (N, 21, 3) 或 (N, 25, 3)

H5 根属性：
coordinate_frame      l21
coordinate_alignment  source_to_l21_xyz
```

双臂离线重定向额外要求上肢观测 H5 字段：

```text
left_arm_keypoints    (N, 3, 3), shoulder / elbow / wrist
right_arm_keypoints   (N, 3, 3), shoulder / elbow / wrist
left_arm_valid        (N,)
right_arm_valid       (N,)
```

输出角度 H5：

```text
frame_ids
timestamps
left_angles   (N, 18), float32, rad
right_angles  (N, 18), float32, rad
left_valid    (N,), bool
right_valid   (N,), bool
```

身份连续性默认阈值为手掌中心位移 `0.08`、腕部相对形状 RMSE `0.05`。身份交换或异常会清空受影响侧的三帧缓存，重新积满连续有效帧后才恢复输出。无效角度帧不写零，H5 中保持上一有效姿态；开头连续无效帧使用全零中立姿态。相关属性写入 H5，`invalid_angle_policy=hold_previous`。

## TRON2A 双臂 Teleoperation

TRON2A DACH 的官方资产位于 `third_party/tron2-robot-description`，当前验证版本为 `9939c22e69d27653ec0ba8a505859a2903dd1a71`。机械臂使用 Robotics Toolbox 从实际 URDF 解析 7 DOF 链、限位与速度，再通过数值 IK 输出关节角；双手继续使用现有 PoseTransformer。

先根据 `config/tron2a_dach_calibration.example.json` 创建并填写实际中立位、坐标变换和 L21 安装偏置。示例文件中的零值仅用于说明格式，不能直接当作真实机器人标定。

离线导出固定 48 维命令：

```powershell
python main_offline_dual_teleop.py `
  --observations input/arm_observations.h5 `
  --angle-h5 output/twohand_angles.h5 `
  --calibration config/tron2a_dach_calibration.json `
  --output output/robot_commands.h5
```

命令 H5 保持以下字段：

```text
frame_ids, timestamps
left/right_arm_ee_pose    (N, 7), xyz + qx qy qz qw
left/right_arm_q          (N, 7)
left/right_hand_q         (N, 17)
left/right_arm_valid
left/right_hand_valid
robot_command             (N, 48)
```

在 PyBullet 回放：

```powershell
python main_pybullet_dual_teleop.py `
  --command-h5 output/robot_commands.h5 `
  --calibration config/tron2a_dach_calibration.json `
  --loop
```

实时入口从外部摄像头或 VR 工程加载一个 `module:factory`；factory 返回带 `hands`（3 帧 25 点）和 `arms`（单帧肩肘腕）字段的 canonical payload：

```powershell
python main_realtime_dual_teleop.py `
  --adapter my_camera_adapter:create_payloads `
  --calibration config/tron2a_dach_calibration.json
```

控制器对 IK 失败、无效追踪、超限和非有限值保持上一有效命令；持续超过 `tracking_timeout` 时，以速度和加速度限制回到 `safe_q`。

## 保留权重

| 路径 | 大小（bytes） | SHA256 |
|---|---:|---|
| `checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth` | 待重新训练 | - |
| `checkpoint/models/thumb3/linker/model_final.pth` | 186036239 | `6A163E30EF805BAE79301230E40DA19BE059903CFE30FBD3F29A5012D15F36FF` |

## 测试

```powershell
python -m unittest discover -s tests -v
```
