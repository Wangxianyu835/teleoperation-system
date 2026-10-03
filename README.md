# LinkerHand L21 双手重定向

手部正式链路如下；实时摄像头适配与双臂扩展见后文：

```text
MediaPipe / 双手关键点 H5
-> 左右手身份连续性检查与三帧窗口
-> PoseTransformer
-> L21 正向运动学与损失（仅训练）
-> 双手 18 维角度 H5
-> 外部仿真适配
```

当前手部入口不使用历史单手和旧数据采集实现。

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
checkpoint/models/twohand_h5/linker/my_run/model_best.pth
```

默认输入为 `input/aligned_visual_hand_data_20260912_153542.h5`。两条路径均从仓库根目录解析，不依赖开发机目录。当前本地文件已通过导出验证；录制和权重被 Git 忽略，新 checkout 需要自行提供这两个文件，或用 `--input` / `--checkpoint` 指定其他匹配的已对齐资源。

训练共享左右手模型：

```powershell
python -m retargeting train `
  --input input/aligned_visual_hand_data_20260912_153542.h5 `
  --run-name my_new_run
```

原始关键点 H5 不能直接用于训练。先生成一个新的坐标对齐文件，原始文件不会被覆盖：

```powershell
python scripts/align_h5_coordinates.py `
  --input input/visual_hand_data_20260912_153542.h5 `
  --output input/aligned_visual_hand_data_20260912_153542.h5
```

该步骤对左右手都执行固定变换 `x'=-y, y'=z, z'=-x`，并在 H5 根属性中写入 `coordinate_frame=l21`。训练和导出 loader 默认要求该标记；标记本身不能证明物理坐标正确。

对齐脚本先验证输入，在同目录临时文件中完成写入并关闭，然后替换输出；验证、写入或替换失败时已有输出保持不变。已标记 L21 或声明 `coordinate_alignment` 的输入会被拒绝，避免重复或不明对齐。输入输出不能是同一路径或同一个文件的硬链接。

只读坐标与尺度诊断：

```powershell
python scripts/diagnose_hand_coordinates.py --input input/aligned_visual_hand_data_20260912_153542.h5
```

该工具报告矩阵、行列式、有向体积、关键点距离、声明单位和 FK 尺度。合成测试只验证软件约定，真实坐标、左右手标签和物理尺度仍需人工验证，详见 [验证计划](docs/UNRESOLVED_VERIFICATION_PLAN.md) 和 [P0 报告](docs/P0_VERIFICATION_REPORT.md)。

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

## 数据与仿真接口

完整手部 H5 格式、18 维关节顺序、21→25 拓扑、身份与无效帧行为统一见 [手部数据契约](docs/HAND_CONTRACT.md)。模型输出含一个固定根占位和 17 个可动关节；推理不计算训练用 FK。

```python
from retargeting.simulation import angle18_to_dofs, angle18_to_nodes, iter_angle_h5

dofs17 = angle18_to_dofs(angle18)    # 去掉根占位
nodes23 = angle18_to_nodes(angle18)  # 补五个固定指尖零节点
```

## TRON2A 双臂 Teleoperation

TRON2A DACH 的官方资产位于 `third_party/tron2-robot-description`，当前验证版本为 `9939c22e69d27653ec0ba8a505859a2903dd1a71`。机械臂使用 Robotics Toolbox 从实际 URDF 解析 7 DOF 链、限位与速度，再通过数值 IK 输出关节角；双手继续使用现有 PoseTransformer。

首次 checkout 需单独获取官方资产（不随本次修复提交）：

```powershell
git clone https://github.com/limxdynamics/tron2-robot-description.git third_party/tron2-robot-description
git -C third_party/tron2-robot-description checkout 9939c22e69d27653ec0ba8a505859a2903dd1a71
```

双臂观测额外使用 `left_arm_keypoints` / `right_arm_keypoints`，形状 `(N,3,3)`，顺序 shoulder/elbow/wrist，及 `(N,)` 的 `left_arm_valid` / `right_arm_valid`。

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
| `checkpoint/models/twohand_h5/linker/my_run/model_best.pth` | 186036461 | `73de5f2d9481ecceff0725af7c094b818a707bbf6cdc0131d57f1e62c90af3bf` |
| `checkpoint/models/thumb3/linker/model_final.pth` | 186036239 | `6A163E30EF805BAE79301230E40DA19BE059903CFE30FBD3F29A5012D15F36FF` |

## 测试

```powershell
python -m unittest discover -s tests -v
```

当前本机全量 91 项通过；新 checkout 运行双臂测试前需要上面的官方 URDF 资产，本地录制 smoke test 还需要输入和权重。

## 文档索引

- [数据契约与处理流程](docs/HAND_CONTRACT.md)：形状、拓扑、输入输出及 valid 行为。
- [坐标与单位](docs/COORDINATE_SYSTEMS.md)：软件约定、尺度路径及证据边界。
- [L21 关节契约](docs/L21_JOINT_CONTRACT.md)：URDF 轴、限位和硬件映射差异。
- [验证报告](docs/P0_VERIFICATION_REPORT.md)：本轮修复、测试、checkpoint 和历史指标。
- [待验证清单](docs/UNRESOLVED_VERIFICATION_PLAN.md)：真实数据与物理实验、后续补测。
- [开发环境](docs/DEVELOPMENT_ENVIRONMENT.md)：依赖版本与环境限制。
