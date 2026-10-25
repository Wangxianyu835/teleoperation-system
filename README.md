# 遥操作与运动重定向

本项目使用摄像头检测人手关键点。
程序将关键点转换为 L21 机械手的关节角。
PyBullet 在回放或实时运行时显示机械手动作。

本文沿用 ASD-STE100 的简明技术写作原则编写中文说明：使用短句，统一术语，每个步骤只说明一个操作。

常用操作流程如下：

**录制手部动作 → 自动对齐关键点 → 使用已有模型导出关节角 → 回放 L21 动作。**

如果已有兼容模型，则无需重新训练。

项目还支持 Vision Pro / VR 输入和双臂仿真，入口见第 5 节。
双臂完整合成场景验收尚未全部通过：H1-2 右臂在大幅运动触发行程夹取后仍有位置误差。
最近的双臂修复尚未完成真人摄像头复测。
具体结果见[双臂诊断与验收](docs/arm-diagnostics.md)。

- [1. 准备环境](#1-准备环境)
- [2. 录制手部动作](#2-录制手部动作)
- [3. 导出与回放关节角](#3-导出与回放关节角)
- [4. 训练模型或实时运行](#4-训练模型或实时运行)
- [5. 查找其他功能与源文件](#5-查找其他功能与源文件)

## 1. 准备环境

本文命令使用 **Windows PowerShell**。
示例路径来自本地开发计算机。

如果已安装 `teleoperation` 环境，请转到第 2 节。
首次安装时，按以下顺序操作：

1. 打开项目目录。
2. 创建 Conda 环境。
3. 激活环境。
4. 安装项目。

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda env create -f environment.yml
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
```

将 MediaPipe 手部检测模型放到 `datasets/hand_landmarker.task`。
第 3 节还需要适用于 `world` 坐标的共享模型权重文件，格式为 `.pth`。
下文用 checkpoint 表示保存模型权重的文件。
程序不会自动下载这些模型。

下方的 `$python` 变量用于指定本地 Conda 环境的 Python 解释器。
如果环境安装在其他位置，请修改该变量。

## 2. 录制手部动作

1. 将下方完整命令块复制到 PowerShell。
2. 执行命令块。
3. 将双手保持在摄像头画面内。
4. 张开双手。
5. 握紧双手。
6. 分别弯曲各根手指。
7. 使用下表中的任意一种方式停止录制并保存数据。

| 停止方式 | 操作 |
|---|---|
| 使用预览窗口按键 | 在预览窗口中按 `q` 或 `Esc`。 |
| 关闭预览窗口 | 单击预览窗口的关闭按钮。 |
| 使用终端按键 | 在终端中按 `Ctrl+C`。 |

本例使用编号为 `0` 的摄像头录制双手。

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'

$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$camera = 0
$handAsset = 'datasets/hand_landmarker.task'
$raw = "datasets/raw/camera_$session.h5"
$aligned = "datasets/raw/aligned_camera_$session.h5"

if (-not (Test-Path $handAsset)) { throw "缺少 MediaPipe 模型：$handAsset" }

& $python -m teleoperation hand record `
  --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset `
  --width 640 --height 480 --fps 30 --frames 0 `
  --output $raw --aligned-output $aligned
if ($LASTEXITCODE -ne 0) { throw '录制失败，请检查上方输出。' }
```

按下表修改录制设置：

| 任务 | 操作 |
|---|---|
| 选择其他摄像头 | 将 `$camera` 设为对应编号，例如 `1` 或 `2`。 |
| 只录制一只手 | 将 `--hand-side both` 改为 `--hand-side left` 或 `--hand-side right`。 |
| 录制指定帧数 | 将 `--frames 0` 改为 `--frames 900`。实际录制时间取决于采集帧率。 |
| 关闭预览 | 添加 `--no-preview`。在终端中按 `Ctrl+C` 停止录制。 |

H5 文件保存**关键点和时间戳**，不保存摄像头视频。
录制正常停止后，程序会自动对齐关键点。
无需单独执行 `hand align` 命令。

| 变量 | 文件内容 | 用途 |
|---|---|---|
| `$raw` | 每只手的 21 个原始关键点；同一帧同时保存 `normalized` 和 `world` 两套坐标 | 检查输入，或重新对齐关键点。 |
| `$aligned` | 每只手的 25 个关键点；由 `world` 坐标对齐到手掌局部坐标系 | 训练模型，或离线计算关节角。 |
| `$angles`（第 3 节） | 每只手的 18 维 L21 关节角，以及左右手各自的有效性标记 | 在仿真中回放动作。 |

手掌局部坐标系是相对于手掌建立的坐标系。
文件名包含录制时间，用于避免覆盖先前的数据。

**第 3 节和第 4 节需继续使用当前终端。**
如果打开新终端，请重新设置 `$python` 和 `PYTHONPATH`。
同时设置所选命令需要的文件路径和参数变量。

## 3. 导出与回放关节角

### 3.1 使用已有模型导出关节角

示例 checkpoint 是本地已有的 `world` 坐标共享模型。
如果该文件不存在，请选择自己的兼容 checkpoint。
如果没有兼容 checkpoint，请先按第 4.1 节训练模型。

录制结束后，执行以下命令块：

```powershell
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_shared_20261014_000112/model_best.pth'
$angles = "outputs/hand_retargeting/camera_angles_$session.h5"

if (-not (Test-Path $checkpoint)) { throw "缺少模型权重文件：$checkpoint" }

& $python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles --device auto
if ($LASTEXITCODE -ne 0) { throw '关节角导出失败，请检查上方输出。' }
```

导出和回放均不使用摄像头。

**`world` 数据必须配合适用于 `world` 坐标的 checkpoint 使用。**
不要将旧的 `normalized` 坐标 checkpoint 或 Vision Pro / VR checkpoint 用于这些数据。
不要通过修改文件元数据使不兼容的文件看起来兼容。

### 3.2 在同一窗口中回放两只 L21 手

执行以下命令，从正面观察双手掌面：

```powershell
& $python devtools/replay_capture_to_l21.py `
  --angles-h5 $angles --scene hands --hand both --view front `
  --gui --fps 30 --render-every 0 --hold-seconds 30
```

窗口只显示两只 L21 手。
两个掌面均朝向观察视角。
回放结束后，窗口将保持最后姿态 30 秒。

| 任务 | 操作 |
|---|---|
| 只显示右手 | 将 `--hand both` 改为 `--hand right`。 |
| 只回放前 900 帧 | 添加 `--max-frames 900`。 |
| 设置回放帧率上限 | 修改 `--fps` 的值。 |

此命令直接显示导出的原生关节角。
它不应用实时程序中的输出滤波或碰撞检查。

也可以使用标准命令回放一只 L21 右手：

```powershell
& $python -m teleoperation replay l21-hand `
  --file $angles --hand right --render --speed 1.0
```

标准命令使用时间戳控制回放速度。
如需显示双手，将 `--hand right` 改为 `--hand both`。
如需半速回放，将 `--speed 1.0` 改为 `--speed 0.5`。

**这两条命令都需要 `$angles` 关节角文件。**
不要直接传入 `$raw` 或 `$aligned`。
下次回放时，将 `$angles` 设为已保存的关节角文件路径。

### 3.3 在机器人模型上显示手部动作（可选）

按下表选择回放命令：

| 命令 | 功能 |
|---|---|
| `native-hand` | 使用机器人原装手。映射时省略原装手无法复现的 L21 自由度。 |
| `mounted-hand` | 将 L21 手安装到机器人模型上。 |

执行以下命令，循环回放 H1-2 原装右手的动作：

```powershell
& $python -m teleoperation replay native-hand `
  --file $angles --robot h1_2 --hand right --view right-hand --render --loop 0
```

窗口显示右手特写。
其他关节保持中性姿态。

执行以下命令，回放安装在 H1-2 上的 L21 右手：

```powershell
& $python -m teleoperation replay mounted-hand `
  --file $angles --robot h1_2 --hand right --render --speed 1.0
```

`--robot` 也支持 `gr1_t2` 和 `g1`。
以上回放命令均使用 PyBullet 仿真，不连接实体机器人。

## 4. 训练模型或实时运行

### 4.1 训练共享模型（可选）

如果没有兼容的模型权重，请使用第 2 节的 `$aligned` 文件训练共享模型。
同一个共享模型可以使用左手、右手或双手数据。
训练不需要机器人关节角标签。

执行以下命令块：

```powershell
$run = "world_$session"
$checkpointRoot = 'outputs/hand_retargeting/checkpoint'

& $python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-side shared `
  --checkpoint-root $checkpointRoot --device auto `
  --epochs 100 --batch-size 64 --learning-rate 0.0001
if ($LASTEXITCODE -ne 0) { throw '训练失败，请检查上方输出。' }

$checkpoint = "$checkpointRoot/models/twohand_h5/linker/$run/model_best.pth"
```

训练成功后，按以下顺序操作：

1. 保持 `$checkpoint` 指向新模型。
2. 将 `$angles` 设为所需的输出文件路径。
3. 使用新 checkpoint 执行第 3.1 节的 `hand export` 命令。
4. 执行第 3.2 节中的回放命令。

不要将新 checkpoint 路径改回第 3.1 节的示例路径。
[工作流参考](docs/workflow-reference.md)说明了数据划分、损失函数和模型兼容性要求。

### 4.2 实时驱动 L21 双手（可选）

此命令使用已设置的 `$checkpoint`、`$camera` 和 `$handAsset`。
无需先录制数据。

执行以下命令，启动实时运行：

```powershell
& $python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
  --camera-index $camera --model-asset-path $handAsset `
  --scene hands --hand both --view front --frames 0
```

使用以下任意一种方式停止运行：

- 在摄像头窗口中按 `q` 或 `Esc`。
- 关闭摄像头窗口。
- 在终端中按 `Ctrl+C`。

此程序不保存关节角 H5 文件。
[手部输出说明](docs/hand-command-safety.md)介绍了滤波、角速度限制和碰撞检查。

## 5. 查找其他功能与源文件

| 任务或主题 | 参考文档 |
|---|---|
| 手动对齐、训练参数、几何重定向、原装手和任务仿真 | [工作流参考](docs/workflow-reference.md) |
| 同步采集米制手部、肩、肘、腕关键点 | [世界坐标关键点采集](docs/world-landmark-capture.md) |
| TRON2A 手臂标定、双臂双手实时控制和组合离线命令 | [身体与手臂工作流](docs/workflow-reference.md#4-扩展身体采集双臂和任务仿真) |
| Vision Pro 录制和离线数据 | [头显录制](docs/visionpro-offline-capture.md) |
| Vision Pro / VR 推理和 L21 回放 | [TransHandR VR 工作流](docs/transhandr-vr.md) |
| H5 格式、坐标和模型兼容性 | [数据契约](docs/contracts.md) |
| 模块职责和依赖关系 | [架构说明](docs/architecture.md) |
| 三机器人双臂诊断、离线验收和已测得的失败项 | [双臂诊断与验收](docs/arm-diagnostics.md) |
| 项目进展演示 | [演示材料](present/README.md) |

临时调整设置时，使用命令行参数。
修改默认设置时，编辑以下文件：

| 默认设置 | 源文件 |
|---|---|
| 训练和模型 | [`config.py`](src/teleoperation/retargeting/hand/config.py) |
| 统一命令行接口（CLI） | [`schemas.py`](src/teleoperation/cli/schemas.py) |
| 资源路径 | [`paths.py`](src/teleoperation/paths.py) |

在命令后添加 `--help`，可查看该命令的全部参数。
例如，执行以下任意一条命令：

```powershell
& $python -m teleoperation hand record --help
& $python -m teleoperation hand export --help
& $python -m teleoperation replay l21-hand --help
```
