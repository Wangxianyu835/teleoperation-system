# 遥操作与重定向

摄像头采集人手关键点，经 L21 重定向输出关节角，在 PyBullet 中回放或实时驱动手部。支持独立 L21 手和 H1-2、GR1-T2、G1 原装手；TRON2A 可用同帧 MediaPipe Pose/Hands world 经真人中性标定实时驱动双臂双手，见第 4.1.2、4.1.3 节。

**离线：采集 → 对齐 → 训练 → 推理导出 → 仿真回放。
实时：摄像头 → 对齐 → 推理 → 仿真。** 
按下面的顺序执行，扩展流程按需选择。

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

### 1.2 这些 `$变量` 是什么，怎么配置

它们是 **PowerShell 当前终端里的变量**，用于保存设备编号、文件路径和实验名，后续命令用 `$变量名` 取值。把下面整段复制到已激活环境的 PowerShell 执行即可；不是写入 Python 文件，也不会自动保存到配置文件。

摄像头需要 MediaPipe 检测模型 `hand_landmarker.task`；身体同步采集还需要 `pose_landmarker_lite.task`。模型需自行准备，程序不会自动下载。先放到 `datasets/`，或将下面路径改为模型所在的完整路径。

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
```

| 变量 | 含义 | 怎么配置 |
|---|---|---|
| `$session` | 本批数据编号，自动取当前时间，如 `20261011_163000` | 新采集时自动生成，也可设为 `'session01'` |
| `$camera` | 摄像头编号 | 默认 `0`；多摄像头时尝试 `1`、`2` |
| `$device` | 重定向模型的训练与推理设备 | CUDA 可用设 `'cuda'`，否则设 `'cpu'` |
| `$handAsset` | MediaPipe 手部检测模型路径 | 指向已存在的 `.task` 文件；采集和实时检测使用 |
| `$poseAsset` | MediaPipe 身体检测模型路径 | 仅 4.1 身体采集使用；只用手部时无需准备 |
| `$raw` | 原始采集 H5 的输出路径 | 新采集使用尚不存在的文件名 |
| `$aligned` | 对齐 H5 路径，后续训练/导出的输入 | 新采集自动生成；使用已有数据时改为实际 `aligned_*.h5` |
| `$run` | 共享训练的实验名，也是输出目录名 | 可改为 `'world_test_v2'`；使用已有权重时填写对应实验名 |
| `$checkpointRoot` | 训练权重和日志的根目录 | 默认保留，也可改为其他可写目录 |
| `$checkpoint` | 共享模型最佳权重路径 | 训练后生成的 `model_best.pth`；已有模型可直接填完整路径 |
| `$angles` | 推理导出的角度 H5 路径 | 共享模型和双模型均导出到此处，供后续仿真回放 |

双引号中的 `$session`、`$run` 会在赋值时展开为实际值。**修改 `$session`、`$run` 或 `$checkpointRoot` 后，要重新执行依赖它们的路径赋值行**；重新打开终端也要重新定义变量。路径可用相对路径或完整路径，相对路径从仓库根目录计算。

首次使用主要检查 `$camera`、`$device`、`$handAsset`，其他路径可先保留。检查 GPU 能否用于 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

返回 `True` 可使用 `'cuda'`，返回 `False` 将 `$device` 改为 `'cpu'`。

再检查检测模型与摄像头：

```powershell
Test-Path $handAsset
python -m teleoperation tools check-camera
```

- `Test-Path $handAsset`：检查该路径是否存在，返回 `True/False`；不会加载模型。返回 `False` 时先放好模型文件或修改 `$handAsset`。
- `python -m teleoperation tools check-camera`：打开 **0 号摄像头**显示画面，按 `q` 关闭。当前命令固定使用 `0`，不读取 `$camera`；实际采集/实时命令通过 `--camera-index $camera` 选择设备。

仅训练或回放已有文件时，无需执行摄像头检查，也无需 MediaPipe 检测模型。

## 2. 离线：采集到训练、推理和仿真

### 2.1 采集手部数据

保存原始 21 点，同时自动生成训练用 palm-local 25 点文件。可只保持一只手可见，录制张手、握拳和各指弯曲等动作；另一只手无需另录训练集。

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

新 world 流程只训练一份模型。只录左手、只录右手或同时录双手都可以：左右人手先归一到同一个右手表示，所有有效窗口汇入同一批次，模型只通过右侧 L21 FK 和六类几何损失训练，无需机器人角度标签。输入为 `(3,25,3)`，输出为 18 维规范角度。

```powershell
python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-side shared `
  --checkpoint-root $checkpointRoot --device $device `
  --epochs 100 --batch-size 64 --learning-rate 0.0001 `
  --val-ratio 0.2 --early-stopping-patience 20 --seed 1234
```

新 world 数据要求 `--hand-side shared`，默认已是 shared。训练不加载左侧 FK；checkpoint 声明 `hand_side=shared`、`training_robot=right`。左手推理使用相同权重，仅四个 MCP roll 输出反号，以匹配本仓库左侧 URDF。左右拇指几何并非严格对称，右侧训练不保证左侧几何误差相同，需要仿真和真人验收。

默认从头训练，按原始时间划分训练/验证并留出窗口间隔，排除无效窗口。模型目录为 `$checkpointRoot/models/twohand_h5/linker/$run/`，包含 `model_best.pth`、`model_last.pth` 和 `epoch_metrics.csv`；日志位于同根的 `logs/twohand_h5/linker/$run/`。CSV 的 right 指训练机器人侧，left 为零；原始人手样本数在启动日志另列。

`--init-checkpoint` 仅接受同坐标、同角度契约的权重。旧四 MCP world 权重不能初始化新模型。新数据需从原始 world 21 点重新对齐，不能改写旧 H5 或 checkpoint 的契约来冒充新格式。

### 2.4 离线推理共享模型

```powershell
python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles `
  --device $device --batch-size 256 --scale-factor 1.0
```

一份权重同时支持左右手。实时和离线导出调用同一输出适配函数，导出的是各侧原生 L21 18 维角度，可继续使用既有回放和 18→17 DOF 映射。输入为单手文件时，另一侧输出为无效；缺手不当作全零真实手势训练。

旧原生根坐标 H5 与匹配的旧独立权重仍可显式离线导出；新 world 链路拒绝 `--left-checkpoint/--right-checkpoint`。新流程的模型输入坐标是 `hand_canonical_right`，不是机器人根坐标。

### 2.5 将重定向结果输出到 PyBullet

共享模型或独立模型都已导出到 `$angles`，后续使用相同回放命令。先检查独立 L21 双手：

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

## 3. 实时：摄像头到推理和仿真

### 3.1 训练模型实时推理与 FK 对比

使用 2.3 的共享 `$checkpoint`。人手关键点和机器人 FK 在同一个规范右手米制坐标中显示；左右模型输入共用同一坐标约定。

**方式 A：共享模型实时推理**

```powershell
python -m teleoperation hand realtime `
  --landmark-space world --checkpoint $checkpoint `
  --camera-index $camera --model-asset-path $handAsset --device $device `
  --visualize --view iso --robot-radius 0.16 --frames 0
```

`--frames 0` 持续运行，须配合 `--visualize`；按 `q`、`Esc` 或 `Ctrl+C` 退出。每侧连续 3 帧有效检测后输出角度，丢手后重新积累窗口。

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

### 4.1.1 TRON2A 摄像头中性姿态标定（第 1 步）

`dual calibrate-camera` 使用同一摄像头帧和同一原始 Unix 毫秒检测时间戳获取 Hands 与 Pose，另外用单调时钟产生相对秒时间。只预览骨架、采集中性姿态和保存 JSON，不驱动机器人；TRON2A 参考末端由 URDF FK 计算，不依赖 mesh。

下面是一组**用于标定流程审核的配置示例**。关节顺序为近端 pitch/roll/yaw、elbow、腕部 yaw/pitch/roll，单位为弧度；旋转按行展开。示例单位旋转和尺度不是测出的相机外参，关节姿态只验证 URDF 限位，不保证碰撞安全。实际坐标映射、工作空间和参考姿态需按安装情况设置。

```powershell
python -m teleoperation dual calibrate-camera --help

python -m teleoperation dual calibrate-camera `
  --camera-index 0 `
  --model-asset-path datasets/hand_landmarker.task `
  --pose-model-asset-path datasets/pose_landmarker_lite.task `
  --urdf third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf `
  --output outputs/calibration/tron2a_camera_neutral.json `
  --safe-q-left 0 0 0 -0.5 0 0 0 --safe-q-right 0 0 0 -0.5 0 0 0 `
  --rotation 1 0 0 0 1 0 0 0 1 --scale 1 1 1 `
  --workspace-min -1 -1 -1 --workspace-max 1 1 1 `
  --samples 60 --pose-min-visibility 0.7 --stability-tolerance 0.015 `
  --min-elbow-bend 10 --max-elbow-bend 100 --max-sample-gap 0.5 `
  --timeout 120 --max-frames 3000

python -c "from teleoperation.retargeting.arm.calibration import ArmCalibration; c=ArmCalibration.load('outputs/calibration/tron2a_camera_neutral.json'); print(c.human_neutral_wrist, c.metadata['extrinsics_calibrated'])"
```

摄像头打开后自动开始计数。左右手臂均需肩、肘、腕 world 数据完整且达到可见度阈值，双肘稍屈。稳定性使用每侧肘、腕相对本侧肩部的位置，窗口内与均值的最大距离不超过 `--stability-tolerance`（米）。遮挡、低可见度、伸直/过度弯肘或采集间断会清空窗口；运动超过阈值从当前帧重新计数。默认连续 60 帧通过后保存并关闭；按 q、Esc、关闭窗口或 Ctrl+C 取消且不保存。输出已存在会在打开摄像头前拒绝；再次运行选择新文件名。`--no-preview` 可关闭窗口。

标定保存 `human_neutral_wrist = mean(wrist_world - shoulder_world)`，保留各侧 FK 参考位姿、配置的旋转/尺度/额外偏移、工作空间、安全参数、URDF 关节限位及采样帧号和两种时间。`extrinsics_calibrated=false` 明确表示没有自动估计完整相机外参；手部安装位姿为未标定的单位占位值。第 2 步的摄像头入口使用肩部相对的 Pose world 米制坐标；原有 `dual realtime` 适配器的默认输入方式保持兼容。

合成验收样例：[tron2a_camera_neutral.synthetic.json](tests/fixtures/tron2a_camera_neutral.synthetic.json)，使用合成肩肘腕及真实 TRON2A URDF FK，不能替代现场标定。其左右参考腕为 `[0, 0.45, -0.15]` 和 `[-0.1, 0.45, -0.1]` 米。测试命令（未安装 editable 包时先执行 `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')`）：

```powershell
python -m unittest tests.test_mediapipe_input tests.test_pose_mediapipe tests.test_mediapipe_tron2a
python -m unittest discover -s tests
```

人工审核：确认未镜像预览的左右标记；遮挡一侧时只有该侧状态失效、恢复后重新计数；稳定稍屈肘能生成并重新加载 JSON；取消不保存；取消或完成后换输出文件名再次启动摄像头。真实摄像头和窗口验收需现场执行。第 1 步仅采集标定；以下第 2 步独立驱动仿真双臂。

### 4.1.2 摄像头实时驱动 TRON2A 仿真双臂（第 2 步）

先用上一节的现场配置采集**真人标定**，然后运行下列命令。`tests/fixtures/tron2a_camera_neutral.synthetic.json` 是自动验收样例，实时入口拒绝它；复制或改名不能将其变为真人标定。`extrinsics_calibrated=false` 表示旋转与尺度仍由参数配置。当前仓库缺少 TRON2A meshes，现场运动学审核须明确选择 `--display simplified`。

本机可用 Python 为 `D:\Anaconda\envs\teleoperation\python.exe`，包含 MediaPipe、Robotics Toolbox、PyBullet、OpenCV。若使用其他环境，应安装现有依赖文件并执行 `python -m pip install -e . --no-deps --no-build-isolation`。以下命令从仓库根目录运行，所有参数均可先通过 `--help` 查看，不会打开设备。

```powershell
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& $python -m teleoperation dual camera --help

& $python -m teleoperation dual camera --arms-only `
  --camera-index 0 `
  --model-asset-path datasets/hand_landmarker.task `
  --pose-model-asset-path datasets/pose_landmarker_lite.task `
  --calibration outputs/calibration/tron2a_camera_neutral.json `
  --urdf third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf `
  --orientation-mode fixed --display simplified --gui --preview `
  --no-attach-hands --frames 0 --simulation-steps 8 `
  --report outputs/dual/camera_report.json
```

| 参数 | 行为与默认值 |
|---|---|
| `--arms-only` | 与 `--with-hands` 二选一；只创建双臂控制，不创建手部重定向模型或手部时序窗口，不要求 checkpoint |
| `--calibration` | 必填的真人中性标定 JSON；验证源、米制单位、本侧肩参考、关节限位及参考 FK 与本次 URDF 匹配 |
| `--orientation-mode fixed/forearm` | 默认 `fixed`：位置跟随，末端方向固定为标定参考方向；`forearm` 复用标定方向对齐偏移 |
| `--display full/simplified` | 默认 `full`，逐个解析并验证 mesh，缺失明确报错；`simplified` 保留运动链、关节参数、惯性及非 mesh 几何，以球/杆替代 mesh，关闭机器人接触碰撞，仅验收运动学 |
| `--gui/--no-gui` | 默认 PyBullet GUI；`--no-gui` 使用 DIRECT，摄像头仍为真实输入 |
| `--preview/--no-preview` | 默认未镜像的摄像头双臂骨架，显示各侧 Pose、IK、工作空间拒绝、失效/超时及方向保持状态，以及实际末端坐标、相对原始目标和限速控制的跟随误差 |
| `--attach-hands/--no-attach-hands` | 默认关闭 L21 挂载；开启时仅将固定零关节的几何随末端定位，不引入手部控制或接触负载；安装偏移仍为标定中的未标定占位值 |
| `--frames` | 默认 `0` 持续运行；如 `300` 限制采集帧数 |
| `--simulation-steps` | 默认每摄像头帧 8 个 `1/240 s` 物理步；按期望采集率选择，报告记录仿真步数 |
| `--width/--height/--fps` | 默认 `640/480/30`；FPS 同时限制处理节奏，过慢推理仍以实际单调时间驱动安全控制 |
| `--pose-min-visibility` | 默认使用标定文件的阈值；可显式设置 `[0,1]` 范围内的值 |
| `--report` | 默认 `outputs/dual/camera_report.json`；每次写入汇总与最后状态，拒绝覆盖任何输入资产 |

控制使用 `world_arms`，逐侧计算 `机器人参考位置 + 配置偏移 + 旋转 @ (尺度 * ((当前腕 world - 当前肩 world) - 标定相对腕))`。世界坐标中的整体平移不影响目标，左右不交换。摄像头目标超出配置工作空间会明确拒绝，旧适配器的绝对腕坐标映射仍保留。固定方向模式允许伸直手臂继续更新位置；前臂模式下近共线（平面夹角正弦小于 0.05）保持上一有效方向，跨过伸直状态时连续选择平面法向符号，位置照常更新。

左右复用现有 7 自由度 IK，优先上一帧控制关节状态；成功标志、有限值、限位和独立 FK 残差均通过后才能下发，容差仍为 `0.01 m / 0.15 rad`。机器人从标定 `safe_q` 初始化，第一帧保持原姿态。有效输入经过关节速度、加速度与位置约束；短时失效先受限减速，残余速度可能产生小幅制动位移，之后保持。超过标定的 `tracking_timeout` 平缓返回 `safe_q`；恢复从当前状态继续跟随。重复、倒退或非有限时间不推进；超过 `0.25 s` 的采样间隔按失效处理、按真实时间计算超时，但最多积分 `0.05 s`，避免积累大步位移。其他可动关节通过固定参考电机保持，报告其实际漂移。

报告分开统计有效/无效 Pose、IK 成功/失败、工作空间拒绝、方向保持、各失效状态，并区分 IK 残差、限速后的控制关节/FK、实际末端相对控制 FK 和原始目标的误差。另列控制约束违例、实际关节限位/速度统计、实际加速度峰值和非控制关节漂移。控制加速度限制约束的是指令；实际物理响应单独测量。原始 Unix 毫秒和控制相对秒保留在最后状态。延迟用单调墙钟测量“开始获取帧到读取仿真实际状态”，单列采集检测、IK 控制、仿真读取耗时，排除预览；这是软件处理延迟，未测量相机曝光至显示的端到端延迟。物理步时长与墙钟时间可能不同，不能把正常退出或平均误差当作持续稳定跟随成功。

人工审核：从与标定一致的中性姿态启动，检查无突跳；分别伸手、收手、抬臂、侧移，确认左右及配置轴向；缓慢小幅移动和同时移动，观察连续跟随与固定末端方向；遮挡一侧确认另一侧独立跟随，短时减速保持、持续遮挡平缓回安全姿态、恢复连续跟随；不可达目标显示 IK 失败或工作空间拒绝；前臂模式下伸直再弯曲不翻腕。用稳定可达目标检查报告中的实际末端位置误差是否达到 `≤0.02 m`，不达标时先定位映射、求解、限速阶段还是仿真阶段。最后用 q、Esc、关闭摄像头/PyBullet 窗口或 Ctrl+C 退出，再次启动确认资源释放。

```powershell
# 有限帧、关闭窗口的真人摄像头检查（仍须真人标定）
& $python -m teleoperation dual camera --arms-only `
  --calibration outputs/calibration/tron2a_camera_neutral.json `
  --display simplified --no-gui --no-preview --frames 300 `
  --report outputs/dual/camera_headless_report.json

# 自动验收，不打开真实摄像头
& $python -m unittest tests.test_mediapipe_input tests.test_pose_mediapipe tests.test_mediapipe_tron2a -v
& $python -m unittest discover -s tests -v
```

自动验收使用合成观测和真实 TRON2A URDF、IK、PyBullet DIRECT；不能替代真人左右方向、延迟、遮挡、连续性和设备重启审核。完整 mesh 显示、GUI 与可选 L21 几何显示的视觉效果也需现场核验（固定参考已通过无头测试）。双手集成使用以下完整模式。

### 4.1.3 摄像头实时驱动 TRON2A 仿真双臂双手（第 3 步）

复用第 1 步真人标定和第 2 步双臂映射、IK、安全控制。先按第 2 节重新对齐 world 数据并训练共享模型。此入口使用新的规范右手米制坐标，权重必须声明新的坐标和规范角度契约、`training_robot=right`；旧 `world_test_v1` 左右权重不能用于新实时链路。坐标不匹配或缺少模型在打开摄像头前报错。

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$calibration = 'outputs/calibration/tron2a_camera_neutral.json'
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_canonical_shared_v1/model_best.pth'

& $python -m teleoperation dual camera --with-hands `
  --camera-index 0 `
  --model-asset-path datasets/hand_landmarker.task `
  --pose-model-asset-path datasets/pose_landmarker_lite.task `
  --calibration $calibration `
  --urdf third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf `
  --checkpoint $checkpoint --device cpu `
  --orientation-mode fixed --display simplified --gui --preview `
  --attach-hands --frames 0 --simulation-steps 8 `
  --report outputs/dual/camera_with_hands_report.json
```

新 world 模式使用一份共享 checkpoint；完整模式需要显式提供 `--checkpoint`。实时默认 CPU，选择 CUDA 后加载和推理均使用 CUDA。坐标变换和短训练测试不能替代真实模型质量与左侧拇指适配的现场检查。

`--with-hands` 默认开启 L21 挂载，显式 `--no-attach-hands` 会报错。场景使用现有 `attach_hands=True, hand_reference_only=False`，通过 `world.apply(command)` 执行每帧完整命令，再执行配置的物理步数。`--arms-only` 保留 `world.apply_arms()`，默认不挂手；显式挂手仍为固定零关节参考展示。安装变换和碰撞执行沿用仿真实现及标定安装参数；标定中的单位安装偏移尚需视觉审核。发现安装或仿真问题时记录并交给仿真负责人。

应用每帧只调用一次 `next_teleop_observation()`。扩展的组合接口如下；它不打开摄像头或创建仿真，手部模型由应用加载后传入：

```python
teleoperator = DualTeleoperator(
    hand_retargeter, device, calibration_path, urdf_path,
    input_mode="camera_world", orientation_mode="fixed",
)
raw = camera.next_teleop_observation()
command = teleoperator.update(raw, control_seconds=clock.next_seconds())
world.apply(command)
```

摄像头模式必须显式传入非负、有限且严格递增的控制秒时间。原始观测和 Unix 毫秒时间戳保留；手部处理时间版本单独构造并带 `raw_timestamp_ms`，输入不原地修改。手部沿用 `MediaPipePalmLocalPipeline("world")` 的对齐和三帧窗口、`PoseTransformerRetargeter`、`angle18_to_dofs()`。旧构造和不传 `control_seconds` 的 `dual realtime` 调用保持 legacy 行为，`hand_preprocessing` 继续用于旧模式；摄像头模式不另设手部空间参数。

命令为独立的 `Tron2AL21Command` 数组，48 维固定顺序为 `[左臂7 | 左手17 | 右臂7 | 右手17]`，以后更新不改变已返回命令。处理顺序仍为手部预处理/推理、左臂、右臂、组装。每侧首次等待三帧时保持初始化手指目标，当前推理有效标志为 false；单侧丢失清空该侧窗口并保持上一有效目标，另一侧继续。恢复需连续三帧有效；保持值不会标为当前帧有效推理。Pose 失效和 IK 失败继续使用原双臂安全输出，手指可以独立更新。`close()` 清空处理器窗口并重置手部适配器。

预览保留原双臂指标，增加左右 world 手部检测、对齐有效、三帧计数、窗口就绪和推理有效状态，以及实际模型路径末段、设备和手部处理耗时；启动信息与 JSON 报告保留完整绝对权重路径。报告补充手部有效帧、窗口等待、无效帧、检测丢失/恢复及推理恢复次数，保留原双臂指标。检测丢失/恢复按处理后的有效对齐状态转换计数；推理恢复仅统计已有有效推理后中断并重新有效的情况，启动等待不计为推理恢复。窗口等待仅计对齐有效但窗口未满的帧。延迟 `hand_processing_ms` 包括手部对齐、缓存和模型推理；`retarget_ik_control_ms` 在完整模式包含手部及双臂处理，不能将其视为纯 IK 耗时。

自动验收命令（不打开真实摄像头）：

```powershell
$env:PYTHONUTF8 = '1'
& $python -m unittest tests.test_mediapipe_input tests.test_pose_mediapipe tests.test_world_hand_pipeline tests.test_mediapipe_tron2a -v
& $python -m unittest discover -s tests -v
```

测试覆盖旧入口回归、时间单位、数组独立性、48 维顺序、失效恢复和共享 checkpoint 生命周期。真实模型集成验收使用新共享权重；旧四 MCP 权重仅验证契约拒绝。合成观测与 PyBullet DIRECT 验证不能替代真实摄像头、模型准确度或挂载视觉检查。

第 3 步提交后由用户人工审核以下项目；未经现场检查均为待审核：

1. 左右手位置、朝向、腕部随动及安装变换正确。
2. 张手、握拳、单指弯曲对应正确，17 关节没有索引错位。
3. 移动双臂同时操作手指，两条链路连续响应。
4. 单侧手部丢失不阻断双臂，Pose 失效不阻断手指。
5. 手部恢复满足连续三帧，双臂恢复保持第 2 步安全行为。
6. 原 arms-only 不退化；完整模式通过 q/Esc、窗口关闭和 Ctrl+C 退出后可再次启动。

本阶段不连接实机，不进行多机器人扩展。完整 mesh/GUI、接触效果、相机和设备重启需现场审核；退出码 0 只说明软件正常运行。

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
$dualCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_three_point_v2/model_best.pth'

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

`loss_weights` 默认 `(500,500,10,500,0.05,500)`，依次对应 `vec/pos/collision/thumb/tip_distance/thumb2`。未开放 CLI 的设置在配置文件修改；改变网络结构后需重新训练或使用结构兼容权重。

## 6. 坐标兼容与诊断

新采集、实时手部和实时仿真默认 **world 米制模式**，尺度固定 `1.0`。训练/导出由对齐 H5 声明确定模式；world 数据需要 world 权重，旧 normalized 权重不能混用。

| 模式 | 对齐契约 | 实时省略权重时的默认实验目录 |
|---|---|---|
| `world` | `mediapipe_world_wrist_index_middle_canonical_right_v1` | `world_canonical_shared_v1` |
| `normalized` | `palm_local_to_l21_v2` | `palm_three_point_v2` |

默认实验目录在 `outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/` 下。本文显式使用 `$checkpoint`，自定义实验不会自动成为默认权重。

normalized 新预处理同样使用三点掌面，但保留各侧原生根坐标。它声明 v2，需要重新训练匹配的权重；旧 v1 文件只可与旧权重显式离线使用。先将变量改成实际文件路径：

```powershell
$oldRaw = 'datasets/raw/visual_hand_data_left_20261009_200808.h5'
$oldAligned = "outputs/hand_retargeting/aligned_normalized_$session.h5"
$oldCheckpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/palm_three_point_v2/model_best.pth'

python -m teleoperation hand align `
  --input $oldRaw --output $oldAligned --landmark-space normalized

python -m teleoperation hand export `
  --input $oldAligned --checkpoint $oldCheckpoint `
  --output "outputs/hand_retargeting/angles_normalized_$session.h5" --device $device

python -m teleoperation hand realtime `
  --landmark-space normalized --checkpoint $oldCheckpoint --device $device `
  --camera-index $camera --model-asset-path $handAsset --visualize --frames 0
```

查询完整参数，或在完成共享模型训练后检查 FK、损失和梯度：

```powershell
python -m teleoperation hand record --help
python -m teleoperation hand train --help
python -m teleoperation hand export --help
python -m teleoperation hand realtime --help
python -m teleoperation.applications.realtime_hand_sim --help
python -m teleoperation replay native-hand --help
python -m teleoperation dual --help

python -m teleoperation tools verify-p0 `
  --input $aligned --checkpoint $checkpoint

python -m teleoperation tools show-robots
python -m teleoperation tools show-all-hands --file $angles
```

补充说明：[数据契约](docs/contracts.md)、[米制采集](docs/world-landmark-capture.md)、[架构](docs/architecture.md)、[阶段汇报材料](present/README.md)。生产代码在 `src/teleoperation/`，完整流程入口在 `applications/`。
