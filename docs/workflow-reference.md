# 完整工作流参考

> 手部训练/控制/导出接口已更新，当前可执行命令见 [统一手部流程](hand-retargeting-unified.md)。本页原有共享18D训练及输出段落是历史记录，不能用于新版模型；采集仍保存米制人体数据。

日常录制与 L21 回放先看 [README](../README.md)。本页保留训练、双臂、仿真和参数的完整操作说明；命令仍从仓库根目录执行。

摄像头采集人手关键点，经 L21 重定向输出关节角，在 PyBullet 中回放或实时驱动手部。支持独立 L21 手和 H1-2、GR1-T2、G1 原装手；TRON2A 可用同帧 SMPLer-X 身体和 MediaPipe Hands 数据，经真人中性标定实时驱动双臂双手，见第 4.1 节。

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

摄像头手部需要 `hand_landmarker.task`；身体使用独立 SMPLer-X 工作进程。资产准备见 [SMPLer-X 链路](smplerx-arm.md)。

```powershell
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$camera = 0
$device = 'cuda'  # 无 CUDA 时改为 'cpu'
$handAsset = 'datasets/hand_landmarker.task'
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

新版 camera world 流程按每个 `--hand-model` 只训练一份模型。左右人手先归一到同一个右手表示，所有有效窗口汇入同一批次，模型通过目标手型的右侧 FK 训练，无需机器人角度标签。输入为 `(3,25,3)`，输出维度由目标手型决定：L21 为 17 个独立关节，原生手型为对应的独立关节数。

```powershell
python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-model l21 `
  --checkpoint-root $checkpointRoot --device $device `
  --epochs 100 --batch-size 64 --learning-rate 0.0001 `
  --val-ratio 0.2 --early-stopping-patience 20 --seed 1234
```

新版 camera world 数据保留 `--hand-model` 作为目标手型选择，但不再选择训练手侧。左右有效窗口汇入同一个 shared NN，checkpoint 声明 `hand_side=shared`、`training_robot=right`；左手推理使用相同权重，并按目标手型转换到左侧原生输出。左右拇指几何并非严格对称，需要仿真和真人验收。

默认从头训练，按原始时间划分训练/验证并留出窗口间隔，排除无效窗口。模型目录为 `$checkpointRoot/<hand_model>/shared/nn/camera_palm_length_window3_v1/$run/`，包含 `model_best.pth`、`model_last.pth`、`config.json`、`dataset.json`、`metrics.jsonl` 和 `training.log`；左右有效窗口数写入数据身份。

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

要在**同一个窗口**中显示双手、掌面同向正对窗口，使用已导出的角度文件：

```powershell
python devtools/replay_capture_to_l21.py `
  --angles-h5 $angles --scene hands --hand both --view front `
  --gui --fps 30 --render-every 0 --hold-seconds 30
```

`--max-frames 900` 只播放前 900 帧，默认播放全部；`--fps` 限制窗口播放速度。
这条路线直接显示模型导出的原生关节角，不经过 VR 入口的输出滤波和碰撞检查。
`aligned_camera_*.h5` 是摄像头关键点文件，需先用匹配的摄像头 checkpoint 执行 `hand export`；
不能将它替换到 `hand vr --input-h5` 中，也不能使用上游 VR 权重。

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

### 4.1 同步身体采集与相机机械臂

相机身体使用 SMPLer-X S32，手指使用 MediaPipe Hands。
录制和三机器人回放命令见 [SMPLer-X 链路](smplerx-arm.md)，映射说明见 [多机器人机械臂控制](multi-robot-arm.md)。

TRON2A 使用 `dual calibrate-camera` 采集真人中性标定，再通过 `dual camera` 控制。
通过 `python -m teleoperation dual calibrate-camera --help` 与 `python -m teleoperation dual camera --help`
查看参数，显式提供安全关节、工作空间、URDF 和匹配当前来源的标定。
合成标定不能用于实时控制。

### 4.2 TRON2A 双臂离线导出与回放

先准备符合 [数据契约](contracts.md) 的上身观测 H5、同帧号和秒时间戳的手部角度 H5、实际标定 JSON。仓库目前没有 `configs/tron2a_dach_calibration.example.json`，标定字段以 `src/teleoperation/retargeting/arm/calibration.py` 的 `ArmCalibration.load()` 为准。

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
| 身体采集 | `--frames`、`--max-fps` | `30 / 5.0` |
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
| 训练轮次、批次、学习率、验证比例、早停 | [config.py](../src/teleoperation/retargeting/hand/config.py) → `TrainingConfig` |
| 优化器、调度器、梯度裁剪、损失权重 | 同文件 → `weight_decay`、`optimizer_eps`、`scheduler_*`、`gradient_clip_norm`、`loss_weights` |
| 网络层数、嵌入维度、注意力头数 | 同文件 → `ModelConfig`；实现见 [transformer.py](../src/teleoperation/retargeting/hand/transformer.py) |
| 默认设备、摄像头编号、导出批次、FK 显示范围 | 同文件 → `RuntimeConfig` |
| 统一 CLI 选项、采集分辨率等默认值 | [schemas.py](../src/teleoperation/cli/schemas.py) → 对应命令函数 |
| 模型、数据、URDF 默认路径 | [paths.py](../src/teleoperation/paths.py) |
| 实时仿真、几何标定、预览默认值 | [realtime_hand_sim.py](../src/teleoperation/applications/realtime_hand_sim.py) → `build_parser()` |
| 独立几何采集默认值 | [camera_record.py](../src/teleoperation/applications/camera_record.py) → `build_parser()` |
| 数据窗口、训练/验证划分、优化过程 | [dataset.py](../src/teleoperation/learning/dataset.py)、[trainer.py](../src/teleoperation/learning/trainer.py)、[training.py](../src/teleoperation/applications/training.py) |
| 损失公式 | [losses.py](../src/teleoperation/learning/losses.py) → `hand_loss()` |
| 三帧、25 点、18 维等共享常量 | [constants.py](../src/teleoperation/contracts/constants.py) |
| 坐标对齐、几何求角、角度滤波 | `src/teleoperation/retargeting/hand/` 下的 `alignment.py`、`geometric.py`、`postprocessing.py` |
| 原装手映射和限位处理 | [native_hand.py](../src/teleoperation/robots/native_hand.py) |
| 双臂坐标、工作空间、超时、加速度 | 实际标定 JSON；字段见 [calibration.py](../src/teleoperation/retargeting/arm/calibration.py) |
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

补充说明：[Vision Pro / VR 同侧双手回放](transhandr-vr.md)、[数据契约](contracts.md)、[米制采集](world-landmark-capture.md)、[架构](architecture.md)、[阶段汇报材料](../present/README.md)。生产代码在 `src/teleoperation/`，完整流程入口在 `applications/`。
