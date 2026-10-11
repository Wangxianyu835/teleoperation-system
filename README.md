# 遥操作与重定向

摄像头采集人手关键点，经 L21 重定向输出关节角，在 PyBullet 中回放或实时驱动手部。支持独立 L21 手和 H1-2、GR1-T2、G1 原装手；TRON2A 双臂另需输入适配和标定。

**离线：采集 → 对齐 → 训练 → 推理导出 → 仿真回放。实时：摄像头 → 对齐 → 推理 → 仿真。** 按下面的顺序执行，扩展流程按需选择。

## 1. 环境与路径

以下命令使用 **Windows PowerShell**，从仓库根目录执行。反引号用于续行，后面不要加空格；也可将多行合并为一行。需要 Python 3.10 或更新版本。

### 1.1 安装与检查

已有环境：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
python -m teleoperation --help
python -m teleoperation tools check-environment
```

首次创建环境时，先执行以下命令，再执行上面的激活与安装步骤：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda env create -f environment.yml
```

依赖见 `environment.yml` 和 `requirements*.txt`，已有环境无需重复创建。未安装项目时，可临时直接运行源码：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python -m teleoperation --help
```

### 1.2 准备检测模型与变量

摄像头需要 `datasets/hand_landmarker.task`；身体同步采集还需要 `datasets/pose_landmarker_lite.task`。这两份 MediaPipe 检测模型需自行准备，程序不会自动下载；训练生成的 `.pth` 是重定向模型。

先执行下列变量定义，后续命令沿用它们。重新打开终端后需再次执行；更换采集批次时重新设置 `$session`，避免覆盖已有文件。

```powershell
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$camera = 0
$device = 'cuda'  # 无 CUDA 时改为 'cpu'
$handAsset = 'datasets/hand_landmarker.task'
$poseAsset = 'datasets/pose_landmarker_lite.task'
$raw = "datasets/raw/world_$session.h5"
$aligned = "datasets/raw/aligned_world_$session.h5"
$run = "world_$session"
$checkpointRoot = 'outputs/hand_retargeting/checkpoint'
$checkpoint = "$checkpointRoot/models/twohand_h5/linker/$run/model_best.pth"
$angles = "outputs/hand_retargeting/angles_$session.h5"

Test-Path $handAsset
python -m teleoperation tools check-camera
```

`Test-Path` 应返回 `True`；`$camera` 是后续采集使用的摄像头编号。相对路径按当前工作目录解释。

## 2. 离线：采集到训练、推理和仿真

### 2.1 采集手部数据

保存原始 21 点，同时自动生成训练用 palm-local 25 点文件。保持双手可见，录制张手、握拳和各指弯曲等动作。

```powershell
python -m teleoperation hand record `
  --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset `
  --width 640 --height 480 --fps 30 --frames 900 `
  --output $raw --aligned-output $aligned
```

`--frames 900` 采集 900 帧；改为 `0` 持续录制。按 `q`、`Esc`、关闭预览窗口或 `Ctrl+C` 停止并保存。`--no-preview` 关闭预览，`--hand-side left/right` 只录指定手。

自动命名时，用这一条替代上面的命令：

```powershell
python -m teleoperation hand record `
  --landmark-space world --hand-side both --frames 900 `
  --camera-index $camera --model-asset-path $handAsset --output-dir datasets/raw
```

生成 `visual_hand_data_both_<时间>.h5` 和 `aligned_visual_hand_data_both_<时间>.h5`；将 `$aligned` 改为实际对齐文件路径再训练。`--output` 与 `--output-dir` 二选一。

### 2.2 对齐已有原始数据

2.1 正常结束后已自动对齐，跳过此步。只有已有原始 H5 或自动对齐未完成时，才另存对齐文件：

```powershell
python -m teleoperation hand align `
  --input $raw --output $aligned --landmark-space world
```

输入须含米制 world 字段；不能重复对齐已对齐文件。原始文件保留，输出路径须尚不存在。

### 2.3 训练共享模型

左右手共用一份 PoseTransformer。输入连续三帧 `(3,25,3)`，输出每手 18 维角度；通过机器人 FK 和几何损失训练，无需机器人角度标签。

```powershell
python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-side shared `
  --checkpoint-root $checkpointRoot --device $device `
  --epochs 100 --batch-size 64 --learning-rate 0.0001 `
  --val-ratio 0.2 --early-stopping-patience 20 --seed 1234
```

默认从头训练，按时间划分训练/验证数据，并排除缺手或无效窗口。输出位置：

| 输出 | 位置与用途 |
|---|---|
| 最佳权重 | `$checkpointRoot/models/twohand_h5/linker/$run/model_best.pth`，用于推理 |
| 最后权重 | 同目录 `model_last.pth` |
| 每轮指标 | 同目录 `epoch_metrics.csv` |
| 训练日志 | `$checkpointRoot/logs/twohand_h5/linker/$run/` |

用已有权重初始化新实验时执行：

```powershell
python -m teleoperation hand train `
  --input $aligned --run-name "${run}_finetune" `
  --checkpoint-root $checkpointRoot --device $device `
  --init-checkpoint $checkpoint --epochs 100 --batch-size 64 --learning-rate 0.0001
```

`--init-checkpoint` 只加载网络权重，优化器和轮次重新开始。新权重在 `${run}_finetune` 目录，后续推理需相应修改 `$checkpoint`。

### 2.4 离线推理并导出角度

读取对齐 H5 和训练权重，生成可回放的双手角度 H5：

```powershell
python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles `
  --device $device --batch-size 256 --scale-factor 1.0

python -m teleoperation hand inspect --angle-h5 $angles
```

输出包含左右角度、有效标记、帧号和时间戳。每手 18 维中的第 0 维是固定占位，实际执行关节为 17 个。world 模式的 `--scale-factor` 必须为 `1.0`。

### 2.5 将重定向结果输出到 PyBullet

先检查独立 L21 双手：

```powershell
python -m teleoperation replay l21-hand `
  --file $angles --hand both --render --speed 1.0
```

再显示三种机器人的原装手，逐条执行，每次结束后再运行下一条：

```powershell
python -m teleoperation replay native-hand `
  --file $angles --robot h1_2 --hand both --render --view hands --loop 1

python -m teleoperation replay native-hand `
  --file $angles --robot gr1_t2 --hand both --render --view hands --loop 1

python -m teleoperation replay native-hand `
  --file $angles --robot g1 --hand both --render --view hands --loop 1
```

`--view full` 看整机，`--speed` 调整速度；原装手回放的 `--loop 0` 无限循环。原装手会丢弃无法表达的 L21 自由度；这里只驱动手指，手臂保持中性姿态。

将 L21 挂到机器人上展示，以及不打开窗口时的检查：

```powershell
python -m teleoperation replay mounted-hand `
  --file $angles --robot g1 --hand both --render --speed 1.0

python -m teleoperation replay l21-hand --file $angles --hand both --check
python -m teleoperation replay l21-hand --file $angles --hand both --headless-replay
python -m teleoperation replay native-hand --file $angles --robot h1_2 --hand both --report
```

### 2.6 可选：左右手独立训练与推理

同一双手 H5 分别训练左、右模型。单手文件只训练对应一侧；双手实时推理需两侧权重齐全。

```powershell
$separateRun = "${run}_separate"
$leftCheckpoint = "$checkpointRoot/models/twohand_h5/linker/$separateRun/left/model_best.pth"
$rightCheckpoint = "$checkpointRoot/models/twohand_h5/linker/$separateRun/right/model_best.pth"
$separateAngles = "outputs/hand_retargeting/angles_separate_$session.h5"

python -m teleoperation hand train `
  --input $aligned --run-name $separateRun --hand-side left `
  --checkpoint-root $checkpointRoot --device $device `
  --epochs 100 --batch-size 64 --learning-rate 0.0001

python -m teleoperation hand train `
  --input $aligned --run-name $separateRun --hand-side right `
  --checkpoint-root $checkpointRoot --device $device `
  --epochs 100 --batch-size 64 --learning-rate 0.0001

python -m teleoperation hand export `
  --input $aligned --output $separateAngles --device $device `
  --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint

python -m teleoperation hand inspect --angle-h5 $separateAngles
python -m teleoperation replay l21-hand --file $separateAngles --hand both --render
```

推理选择 `--checkpoint`，或成对提供 `--left-checkpoint`、`--right-checkpoint`；两种方式不能混用。

## 3. 实时：摄像头到推理和仿真

### 3.1 训练模型实时推理与 FK 对比

使用离线训练生成的 `$checkpoint`，显示人手关键点、模型输入和 L21 FK：

```powershell
python -m teleoperation hand realtime `
  --landmark-space world --checkpoint $checkpoint `
  --camera-index $camera --model-asset-path $handAsset --device $device `
  --visualize --view iso --robot-radius 0.16 --frames 0
```

`--frames 0` 持续运行，须配合 `--visualize`；按 `q`、`Esc` 或 `Ctrl+C` 退出。每侧连续 3 帧有效检测后输出角度，丢手后重新积累窗口。

使用 2.6 的双模型时：

```powershell
python -m teleoperation hand realtime `
  --landmark-space world --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint `
  --camera-index $camera --model-asset-path $handAsset --device $device `
  --visualize --frames 0
```

只在终端运行有限帧，或保存对比图：

```powershell
python -m teleoperation hand realtime `
  --landmark-space world --checkpoint $checkpoint --device $device `
  --camera-index $camera --model-asset-path $handAsset --frames 300

python -m teleoperation hand realtime `
  --landmark-space world --checkpoint $checkpoint --device $device `
  --camera-index $camera --model-asset-path $handAsset `
  --visualize --headless --frames 300 --snapshot outputs/hand_retargeting/realtime.png
```

此入口用于推理与 FK 对比，不保存角度序列；实时 PyBullet 使用下一节。

### 3.2 训练模型实时驱动仿真

同时展示 H1-2、GR1-T2、G1 原装手，每帧推理后直接驱动仿真：

```powershell
python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world `
  --camera-index $camera --model-asset-path $handAsset --device $device `
  --scene robots --hand both --view full --sim-steps 8 --frames 0 `
  --report outputs/hand_retargeting/realtime_sim.json
```

仅显示 L21 双手：

```powershell
python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world `
  --camera-index $camera --model-asset-path $handAsset --device $device `
  --scene hands --hand both --frames 0
```

`--frames 0` 持续运行；在摄像头窗口按 `q/Esc`、关闭窗口或按 `Ctrl+C` 停止。`--no-preview` 隐藏摄像头预览，`--headless` 关闭 PyBullet 窗口。此入口目前只支持一份共享 checkpoint；双模型实时推理使用 3.1。报告保存统计信息，不保存角度 H5。

### 3.3 无需训练的几何重定向

没有 `.pth` 时可直接测试摄像头到三机器人仿真。开始时双手张开，完成每侧 15 帧零位标定。

```powershell
python -m teleoperation.applications.realtime_hand_sim `
  --backend geometric --landmark-space world `
  --camera-index $camera --model-asset-path $handAsset `
  --scene robots --hand both --calibration-frames 15 --smoothing 0.0 --frames 0
```

`--calibration-frames` 调整张手标定帧数，`--smoothing` 调整几何角度平滑。几何算法与训练模型是独立方案。默认 `--backend auto` 仅在权重缺失时选择几何后端；权重契约错误时直接报错。

录制几何角度再回放，可用独立采集入口：

```powershell
python -m teleoperation.applications.camera_record `
  --seconds 30 --camera-index $camera --model-asset-path $handAsset `
  --output-h5 "outputs/hand_retargeting/geometric_raw_$session.h5" `
  --angles-h5 "outputs/hand_retargeting/geometric_angles_$session.h5" `
  --calibration-frames 15 --smoothing 0.0

python -m teleoperation replay l21-hand `
  --file "outputs/hand_retargeting/geometric_angles_$session.h5" --hand both --render
```

这个独立入口使用 normalized 采集和几何计算，与第 2 节的 world 模型推理流程不同。

## 4. 扩展：身体采集、双臂和任务仿真

### 4.1 同步采集手部与肩、肘、腕

需要 `$poseAsset` 文件。保存 normalized/world 手部及身体点，不自动对齐；手部训练前执行 `hand align`。

```powershell
Test-Path $poseAsset
$bodyRaw = "outputs/hand_body_capture/body_$session.h5"
$bodyAligned = "outputs/hand_body_capture/aligned_body_$session.h5"

python -m teleoperation hand record-world `
  --camera-index $camera --model-asset-path $handAsset --pose-model-asset-path $poseAsset `
  --width 640 --height 480 --fps 30 --max-fps 5 --frames 300 `
  --pose-min-visibility 0.2 --output $bodyRaw

python -m teleoperation hand align `
  --input $bodyRaw --output $bodyAligned --landmark-space world
```

`--max-fps` 限制实际采集频率，`0` 取消限速。将第 2 节的 `$aligned` 改为 `$bodyAligned` 可训练手部。手与身体 world 原点不同；身体数据还需坐标、单位和时间戳适配才能用于双臂 IK。

### 4.2 TRON2A 双臂离线导出与回放

先准备符合 [数据契约](docs/contracts.md) 的上身观测 H5、同帧号和秒时间戳的手部角度 H5、实际标定 JSON。仓库目前没有 `configs/tron2a_dach_calibration.example.json`，标定字段以 `src/teleoperation/retargeting/arm/calibration.py` 的 `ArmCalibration.load()` 为准。

以下是待提供文件的路径，须改成实际文件；普通 `record-world` 文件不能直接代替已适配的 `$observations`。

```powershell
$observations = 'outputs/dual/observations.h5'
$dualAngles = 'outputs/dual/hand_angles.h5'
$calibration = 'outputs/calibration/tron2a.json'
$urdf = 'third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf'
$commands = 'outputs/dual/commands.h5'

python -m teleoperation dual export `
  --observations $observations --angle-h5 $dualAngles `
  --calibration $calibration --urdf $urdf --output $commands

python -m teleoperation dual replay `
  --command-h5 $commands --calibration $calibration --urdf $urdf --fps 30
```

输出是 TRON2A 双臂 14 维加 L21 双手 34 维的 48 维命令，不能直接用于 H1-2、GR1-T2、G1 原生动作回放。

### 4.3 TRON2A 双臂实时计算

需自行实现 `my_input.py` 的 `factory()`，返回 `InputSource` 或组合观测迭代器。下面使用 normalized palm-local 权重；替换适配器名称和权重路径后执行。

```powershell
$adapter = 'my_input:factory'
$dualCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_local_v2/model_best.pth'

python -m teleoperation dual realtime `
  --adapter $adapter --calibration $calibration --urdf $urdf `
  --hand-preprocessing palm-local --checkpoint $dualCheckpoint --device $device
```

该入口支持 `legacy` 和 normalized `palm-local`，不能直接使用第 2 节的 world 权重。它计算组合命令并打印状态，未自动连接仿真窗口或硬件；命令文件回放使用 4.2。

### 4.4 仿真任务与原生动作回放

任务入口使用演示控制器；手部重定向展示使用 2.5 或 3.2。

```powershell
# PyBullet 任务演示，保存 episode 观测
python -m teleoperation sim run `
  --task pushcube --robot h1_2 --demo --data-dir outputs/sim

# 无窗口、无记录运行单次任务
python -m teleoperation sim run --task pushcube --robot h1_2 --no-render --no-record

# Level 1 任务基准测试
python -m teleoperation sim run --benchmark --robot g1 --trials 5 --no-render --no-record

# 关节展示
python -m teleoperation sim demo-joints

# 回放仓库自带原生动作样例
python -m teleoperation replay actions `
  --file datasets/samples/actions_demo_h1_2.h5 --robot h1_2 --render

# 生成原生动作文件并在无窗口模式运行
python -m teleoperation replay actions `
  --dummy --steps 120 --robot g1 --no-render --save-actions outputs/sim/actions_g1.h5
```

`replay actions` 读取动作序列；`sim run` 记录的 episode 主要保存观测，格式不同。原生动作维数：H1-2 为 38、GR1-T2 为 36、G1 为 28。

## 5. 在哪里修改参数

**临时调整：改命令行参数或 1.2 的变量。永久调整：改下面的默认配置。** 数据坐标契约、权重和网络结构必须匹配。

### 5.1 常用命令参数

| 环节 | 参数 | 默认值或用途 |
|---|---|---|
| 手部采集 | `--camera-index`、`--width`、`--height`、`--fps` | `0 / 640 / 480 / 30` |
| 手部采集 | `--frames`、`--hand-side`、`--output`、`--aligned-output` | 默认持续录制、双手；指定输出文件 |
| 身体采集 | `--frames`、`--max-fps`、`--pose-min-visibility` | `30 / 5.0 / 0.2` |
| 训练 | `--epochs`、`--batch-size`、`--learning-rate` | `100 / 64 / 0.0001` |
| 训练 | `--val-ratio`、`--early-stopping-patience`、`--seed` | `0.2 / 20 / 1234` |
| 训练 | `--hand-side`、`--run-name`、`--checkpoint-root` | 模型手侧、实验名、输出根目录 |
| 推理 | `--checkpoint` 或双侧 checkpoint | 选择实际训练权重 |
| 导出 | `--batch-size`、`--device`、`--output` | 默认批次 `256`，设备 `auto/cpu/cuda` |
| 实时 FK | `--view`、`--robot-radius`、`--frames` | `iso / 0.16 / 300`；world 使用米制显示比例 |
| 实时仿真 | `--backend`、`--scene`、`--sim-steps` | `auto / robots / 8` |
| 几何后端 | `--calibration-frames`、`--smoothing` | `15 / 0.0` |
| 回放 | `--render`、`--hand`、`--speed` | 打开窗口、手侧、播放速度 |
| 原装手回放 | `--limit-mode`、`--view`、`--loop` | `clamp / full / 1` |
| 双臂 | `--calibration`、`--urdf`、`--adapter` | 标定文件、机器人模型、输入适配器 |

`hand train/export` 默认设备 `auto`；实时入口只接受 `cpu/cuda`，默认 `cpu`。本 README 用 `$device` 显式统一设置。

### 5.2 默认配置与实现位置

| 要修改的内容 | 文件与位置 |
|---|---|
| 训练轮次、批次、学习率、验证比例、早停 | [config.py](src/teleoperation/retargeting/hand/config.py) → `TrainingConfig` |
| 优化器、调度器、梯度裁剪、损失权重 | 同文件 → `weight_decay`、`optimizer_eps`、`scheduler_*`、`gradient_clip_norm`、`loss_weights` |
| 网络层数、嵌入维度、注意力头数 | 同文件 → `ModelConfig`；实现见 [transformer.py](src/teleoperation/retargeting/hand/transformer.py) |
| 默认设备、摄像头编号、导出批次、FK 显示范围 | 同文件 → `RuntimeConfig` |
| 统一 CLI 选项、采集分辨率等默认值 | [schemas.py](src/teleoperation/cli/schemas.py) → 对应命令函数 |
| 模型、数据、URDF 默认路径 | [paths.py](src/teleoperation/paths.py) |
| 实时仿真、几何标定、预览默认值 | [realtime_hand_sim.py](src/teleoperation/applications/realtime_hand_sim.py) → `build_parser()` |
| 独立几何采集默认值 | [camera_record.py](src/teleoperation/applications/camera_record.py) → `build_parser()` |
| 数据窗口、训练/验证划分、优化过程 | [dataset.py](src/teleoperation/learning/dataset.py)、[trainer.py](src/teleoperation/learning/trainer.py)、[training.py](src/teleoperation/applications/training.py) |
| 损失公式 | [losses.py](src/teleoperation/learning/losses.py) → `hand_loss()` |
| 三帧、25 点、18 维等共享常量 | [constants.py](src/teleoperation/contracts/constants.py) |
| 坐标对齐、几何求角、角度滤波 | `src/teleoperation/retargeting/hand/` 下的 `alignment.py`、`geometric.py`、`postprocessing.py` |
| 原装手映射和限位处理 | [native_hand.py](src/teleoperation/robots/native_hand.py) |
| 双臂坐标、工作空间、超时、加速度 | 实际标定 JSON；字段见 [calibration.py](src/teleoperation/retargeting/arm/calibration.py) |
| 物理环境、机器人加载、任务 | `src/teleoperation/simulation/` 下的 `environment.py`、`robot_loader.py`、`tasks/` |

`loss_weights` 默认 `(500,500,10,500,0.01,500)`，依次对应 `vec/pos/collision/thumb/tip_distance/thumb2`。未开放 CLI 的设置在配置文件修改；改变网络结构后需重新训练或使用结构兼容权重。

## 6. 坐标兼容与诊断

新采集、实时手部和实时仿真默认 **world 米制模式**，尺度固定 `1.0`。训练/导出由对齐 H5 声明确定模式；world 数据需要 world 权重，旧 normalized 权重不能混用。

| 模式 | 对齐契约 | 实时省略权重时的默认实验目录 |
|---|---|---|
| `world` | `mediapipe_world_palm_local_to_l21_v1` | `world_palm_local_v1` |
| `normalized` | `palm_local_to_l21_v1` | `palm_local_v2` |

默认实验目录在 `outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/` 下。本文显式使用 `$checkpoint`，自定义实验不会自动成为默认权重。

旧 normalized 数据使用独立流程；先将变量改成实际文件路径：

```powershell
$oldRaw = 'datasets/raw/visual_hand_data_left_20261009_200808.h5'
$oldAligned = "outputs/hand_retargeting/aligned_normalized_$session.h5"
$oldCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_local_v2/model_best.pth'

python -m teleoperation hand align `
  --input $oldRaw --output $oldAligned --landmark-space normalized

python -m teleoperation hand export `
  --input $oldAligned --checkpoint $oldCheckpoint `
  --output "outputs/hand_retargeting/angles_normalized_$session.h5" --device $device

python -m teleoperation hand realtime `
  --landmark-space normalized --checkpoint $oldCheckpoint --device $device `
  --camera-index $camera --model-asset-path $handAsset --visualize --frames 0
```

查询完整参数，或在完成双模型训练后检查 FK、损失和梯度：

```powershell
python -m teleoperation hand record --help
python -m teleoperation hand train --help
python -m teleoperation hand export --help
python -m teleoperation hand realtime --help
python -m teleoperation.applications.realtime_hand_sim --help
python -m teleoperation replay native-hand --help
python -m teleoperation dual --help

python -m teleoperation tools verify-p0 `
  --input $aligned --left-checkpoint $leftCheckpoint --right-checkpoint $rightCheckpoint

python -m teleoperation tools show-robots
python -m teleoperation tools show-all-hands --file $angles
```

补充说明：[数据契约](docs/contracts.md)、[米制采集](docs/world-landmark-capture.md)、[架构](docs/architecture.md)、[阶段汇报材料](present/README.md)。生产代码在 `src/teleoperation/`，完整流程入口在 `applications/`。
