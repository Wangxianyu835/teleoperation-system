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

该步骤对左右手应用 `retargeting/coordinates.py` 中当前启用的 `SOURCE_TO_L21_MATRIX`。原矩阵定义固定变换 `x'=-y, y'=z, z'=-x`；若临时改成单位矩阵，生成的数据就会采用单位矩阵。脚本仍会在 H5 根属性中写入 `coordinate_frame=l21`，因此该标记和 `coordinate_alignment` 字符串都不能用于判断实际使用了哪个矩阵。使用现有 `my_run/model_best.pth` 时，应保持原矩阵输入约定。

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

## 原始人手与 L21 URDF FK 同步回放

`D:\2026\code\mytrans\scripts\compare_zuobiaoxi_vs_origin.py` 模仿采集项目的 `read_hand_xyz.py`，在同一窗口并排播放原始 MediaPipe 人手骨架和已有预测角度驱动的 L21 FK 骨架。左栏保留原始图像 XY 投影；右栏显示 URDF 计算出的机器人节点及连线。双手模式中上排是左手，下排是右手。

脚本只读取原始关键点和已导出的角度，不执行模型推理或坐标对齐。机器人侧显示骨架节点，不渲染 URDF 网格外观。运行环境需要 OpenCV GUI 支持；当前 `TransHandR` 环境已有 `cv2`。

### 新录制 H5 的完整操作示例

以下路径对应当前开发机。换一段录制时，保持各步骤中的采集日期与文件名一致；回放的原始 H5 和角度 H5 必须来自同一段采集。

本例原始文件是 `D:\2026\code\MediaPipe\visual_hand_dataset\visual_hand_data_20261004_182606.h5`。在 PowerShell 中进入项目根目录并激活环境：

```powershell
Set-Location "D:\2026\code\mytrans"
conda activate TransHandR
```

**0. 确认坐标矩阵。** 现有 checkpoint 的训练数据使用了原矩阵。若 `D:\2026\code\mytrans\retargeting\coordinates.py` 仍启用了调试用单位矩阵，请将 `SOURCE_TO_L21_MATRIX` 恢复为下面的定义并保存。确保只有一个实际执行的定义，避免后面的单位矩阵赋值将其覆盖。

```python
SOURCE_TO_L21_MATRIX = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ),
    dtype=np.float32,
)
```

修改矩阵只影响之后生成的对齐数据，不会改变已经保存的 H5。单位矩阵实验应使用单独的输入与角度输出文件名。

**1. 从原始录制生成对齐后的模型输入。**

```powershell
python "D:\2026\code\mytrans\scripts\align_h5_coordinates.py" `
  --input "D:\2026\code\MediaPipe\visual_hand_dataset\visual_hand_data_20261004_182606.h5" `
  --output "D:\2026\code\mytrans\input\aligned_visual_hand_data_20261004_182606.h5"
```

这一步进行 21→25 点转换、手腕归零和坐标变换。原始录制文件保持不变。

**2. 用已有 checkpoint 导出对应角度。**

```powershell
python -m retargeting export `
  --device cpu `
  --disable-identity-tracking `
  --input "D:\2026\code\mytrans\input\aligned_visual_hand_data_20261004_182606.h5" `
  --checkpoint "D:\2026\code\mytrans\checkpoint\models\twohand_h5\linker\my_run\model_best.pth" `
  --output "D:\2026\code\mytrans\output\manual_angles_20261004_182606_no_tracking.h5"
```

这里关闭身份跟踪，直接使用采集文件中的左右手标签，便于逐侧对照原始回放。已有导出默认会覆盖同名输出；需要保留其他实验时使用不同文件名。更换录制后必须重新生成匹配的角度文件，不能沿用上一段录制的角度。

**3. 检查角度文件。**

```powershell
python -m retargeting inspect `
  --angle-h5 "D:\2026\code\mytrans\output\manual_angles_20261004_182606_no_tracking.h5"
```

检查两侧形状为 `(总帧数, 18)`，`nonfinite=0`、`out_of_limits=0`。有效帧数可能小于采集检测帧数，因为预测需要连续三帧窗口；这些数值检查不能证明动作重定向准确。

**4. 同步回放原始人手和机器人骨架。**

```powershell
python "D:\2026\code\mytrans\scripts\compare_zuobiaoxi_vs_origin.py" `
  --source-h5 "D:\2026\code\MediaPipe\visual_hand_dataset\visual_hand_data_20261004_182606.h5" `
  --angle-h5 "D:\2026\code\mytrans\output\manual_angles_20261004_182606_no_tracking.h5" `
  --side both `
  --view iso `
  --speed 1
```

`--source-h5` 必须传原始的 21 点采集文件，不能传步骤 1 的 25 点对齐文件。`--angle-h5` 传步骤 2 导出的角度文件。脚本检查两份数据的帧数、帧编号和时间戳，并预计算整段 CPU FK 后打开窗口。

### 回放控制与参数

先点击回放窗口，再操作键盘：

| 按键 | 功能 |
|---|---|
| 空格 | 暂停 / 继续 |
| `a` / `d` | 上一帧 / 下一帧，并暂停 |
| `1` / `2` / `3` / `4` | 机器人 YZ / XZ / XY / 斜视投影 |
| `r` | 回到第一帧；如果当前暂停，再按空格开始播放 |
| `s` | 保存当前并排对照 PNG |
| `q` / Esc | 退出 |

常用参数：

| 参数 | 用法 |
|---|---|
| `--side left` / `--side right` | 只回放一侧；默认 `both` |
| `--speed 0.5` | 相对采集时间戳半速播放；默认 `1` |
| `--frame 100 --paused` | 从数组下标 100 开始并暂停；下标从 0 开始 |
| `--labels` | 显示各自的节点编号；MediaPipe 和 L21 的编号体系不同 |
| `--loop` | 播放结束后循环；默认停留在最后一帧 |
| `--image-width 640 --image-height 480` | 原始采集图像尺寸；不同分辨率时应填写实际尺寸 |
| `--output-dir "D:\2026\code\mytrans\picture\my_replay"` | 指定按 `s` 保存截图的目录 |

默认截图目录是 `D:\2026\code\mytrans\picture\compare_zuobiaoxi_vs_origin`。不打开窗口、只导出单帧对照图的示例：

```powershell
python "D:\2026\code\mytrans\scripts\compare_zuobiaoxi_vs_origin.py" `
  --source-h5 "D:\2026\code\MediaPipe\visual_hand_dataset\visual_hand_data_20261004_182606.h5" `
  --angle-h5 "D:\2026\code\mytrans\output\manual_angles_20261004_182606_no_tracking.h5" `
  --frame 100 --view iso --labels `
  --snapshot "D:\2026\code\mytrans\picture\compare_zuobiaoxi_vs_origin\new_recording_frame_000100.png"
```

机器人栏中的 `VALID` 表示当前有效预测；`HOLD` 表示当前无效，保持最近有效角度；`ZERO` 表示此前还没有有效预测，显示零角度参考。原始栏的 `DETECTED` / `MISSING` 表示当前是否有采集关键点。回放按秒单位的时间戳调度且不丢帧，绘制较慢时会减速；这不是实时推理性能测量。两栏使用不同投影和显示比例，不能将屏幕距离当作物理误差。

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
