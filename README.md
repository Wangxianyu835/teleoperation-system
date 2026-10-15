# 遥操作与重定向

摄像头识别人手关键点，重定向为 L21 关节角，再在 PyBullet 中回放或实时显示。日常使用按下面的顺序操作：

**录制动作 → 自动对齐 → 用已有模型导出角度 → L21 回放。** 已有匹配模型时，无需重新训练。

- [1. 准备环境](#1-准备环境)
- [2. 一键录制摄像头动作](#2-一键录制摄像头动作)
- [3. 导出角度并在 L21 上回放](#3-导出角度并在-l21-上回放)
- [4. 可选：训练与实时显示](#4-可选训练与实时显示)
- [5. 其他功能与实现位置](#5-其他功能与实现位置)

## 1. 准备环境

以下命令使用 **Windows PowerShell**。已有 `teleoperation` 环境可直接进入第 2 节；首次使用先安装：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda env create -f environment.yml
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
```

摄像头检测模型放在 `datasets/hand_landmarker.task`。第 3 节还需要匹配 world 坐标的共享 `.pth` 权重；示例使用本机已有文件，程序不会自动下载模型。后面的 `$python` 指向本机 Conda 解释器，环境安装在其他位置时修改这一行。

## 2. 一键录制摄像头动作

**复制下面整段执行即可开始录制。** 默认使用 0 号摄像头、录制双手，按 `q`、`Esc`、关闭预览窗口或 `Ctrl+C` 结束并保存。做张手、握拳、单指弯曲等动作时，保持手在画面内。

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

if (-not (Test-Path $handAsset)) { throw "Missing MediaPipe model: $handAsset" }

& $python -m teleoperation hand record `
  --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset `
  --width 640 --height 480 --fps 30 --frames 0 `
  --output $raw --aligned-output $aligned
if ($LASTEXITCODE -ne 0) { throw 'Recording failed; check the output above.' }
```

| 要调整什么 | 修改哪里 |
|---|---|
| 摄像头 | `$camera = 0` 改成实际编号，如 `1` 或 `2` |
| 只录一侧 | `--hand-side both` 改成 `left` 或 `right` |
| 录满指定帧数自动停止 | `--frames 0` 改成 `900`；实际时长取决于采集帧率 |
| 不显示摄像头预览 | 添加 `--no-preview`，用 `Ctrl+C` 停止 |

录制保存的是**关键点与时间戳 H5，不是摄像头视频**。正常结束后自动对齐，不需要再执行 `hand align`。

| 变量 | 文件内容 | 用途 |
|---|---|---|
| `$raw` | 同帧 normalized / world 原始 21 点 | 保留原始输入，供检查或重新对齐 |
| `$aligned` | world 对齐后的 palm-local 25 点 | 输入训练或离线推理 |
| `$angles`（下一节生成） | 左右 L21 18 维关节角及有效标志 | 输入仿真回放 |

文件名带录制时间，避免覆盖之前的数据。**第 3、4 节继续在同一个终端执行**；重新开终端时需恢复 `$python`、`PYTHONPATH` 和实际文件路径。

## 3. 导出角度并在 L21 上回放

### 3.1 用已有模型导出角度

录制完成后执行下面整段。`$checkpoint` 使用本机已有的 world 共享权重；如果没有该文件，改为自己的匹配权重，或先按第 4 节训练。

```powershell
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_shared_20261014_000112/model_best.pth'
$angles = "outputs/hand_retargeting/camera_angles_$session.h5"

if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }

& $python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles --device auto
if ($LASTEXITCODE -ne 0) { throw 'Angle export failed; check the output above.' }
```

导出和后续回放不使用摄像头。world 数据必须配 world 权重，不能混用旧 normalized 或 Vision Pro / VR 权重，也不能修改文件元数据来冒充兼容。

### 3.2 L21 双手在同一窗口正面回放

```powershell
& $python devtools/replay_capture_to_l21.py `
  --angles-h5 $angles --scene hands --hand both --view front `
  --gui --fps 30 --render-every 0 --hold-seconds 30
```

只显示 L21 双手，双掌朝向窗口；播放完保持最后姿态 30 秒。`--hand right` 只显示右手，`--max-frames 900` 只播放前 900 帧。`--fps` 限制此入口的播放帧率；它直接显示已导出的原生角度，不经过实时入口的输出滤波和碰撞检查。

使用标准回放入口只看一只 L21 右手，并按时间戳调整播放速度：

```powershell
& $python -m teleoperation replay l21-hand `
  --file $angles --hand right --render --speed 1.0
```

`--hand both` 显示双手，`--speed 0.5` 为半速。两种入口都读取 **`$angles` 角度文件**，不能直接传 `$raw` 或 `$aligned`；需要再次回放时，将 `$angles` 设为已保存文件的实际路径即可。

### 3.3 可选：在一台机器人上展示

两种方式选一条运行。`native-hand` 使用机器人原装手，映射时会丢弃其无法表达的 L21 自由度；`mounted-hand` 将 L21 挂载到机器人上。

```powershell
# H1-2 原装右手特写，循环播放；其他关节保持中性姿态
& $python -m teleoperation replay native-hand `
  --file $angles --robot h1_2 --hand right --view right-hand --render --loop 0

# 将 L21 右手挂到 H1-2 上回放
& $python -m teleoperation replay mounted-hand `
  --file $angles --robot h1_2 --hand right --render --speed 1.0
```

`--robot` 还可选 `gr1_t2`、`g1`。以上均为 PyBullet 仿真，不连接实机。

## 4. 可选：训练与实时显示

### 4.1 没有匹配权重时，训练共享模型

使用第 2 节生成的 `$aligned`。只录左手、只录右手或双手都可训练同一份共享模型，无需机器人角度标签。

```powershell
$run = "world_$session"
$checkpointRoot = 'outputs/hand_retargeting/checkpoint'

& $python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-side shared `
  --checkpoint-root $checkpointRoot --device auto `
  --epochs 100 --batch-size 64 --learning-rate 0.0001
if ($LASTEXITCODE -ne 0) { throw 'Training failed; check the output above.' }

$checkpoint = "$checkpointRoot/models/twohand_h5/linker/$run/model_best.pth"
```

训练成功后，用此 `$checkpoint` 执行 3.1 的 `hand export`，再执行 3.2 回放；无需重新赋值为示例旧实验路径。训练划分、损失和模型契约见[完整工作流参考](docs/workflow-reference.md)。

### 4.2 不录文件，直接实时驱动 L21

使用已选好的 `$checkpoint`、`$camera` 和 `$handAsset`：

```powershell
& $python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
  --camera-index $camera --model-asset-path $handAsset `
  --scene hands --hand both --view front --frames 0
```

按摄像头窗口的 `q/Esc`、关闭窗口或 `Ctrl+C` 退出。此入口不保存角度 H5；滤波、角速度限制和碰撞检查见[手部输出处理](docs/hand-command-safety.md)。

## 5. 其他功能与实现位置

| 需要做什么 | 查看哪里 |
|---|---|
| 手动对齐、详细训练参数、几何后端、原装手与任务仿真 | [完整工作流参考](docs/workflow-reference.md) |
| 同步采集手部与肩肘腕 | [米制采集说明](docs/world-landmark-capture.md) |
| TRON2A 双臂标定、实时双臂双手、离线组合命令 | [完整工作流参考：身体与双臂](docs/workflow-reference.md#4-扩展身体采集双臂和任务仿真) |
| Vision Pro 录制和离线数据 | [头显录制](docs/visionpro-offline-capture.md) |
| Vision Pro / VR 推理与 L21 回放 | [TransHandR VR 链路](docs/transhandr-vr.md) |
| H5 格式、坐标与权重兼容 | [数据契约](docs/contracts.md) |
| 模块职责与依赖关系 | [架构](docs/architecture.md) |
| 汇报材料 | [阶段汇报](present/README.md) |

临时调整使用命令行参数；长期默认值主要在 `src/teleoperation/retargeting/hand/config.py`（训练与模型）、`src/teleoperation/cli/schemas.py`（统一 CLI）、`src/teleoperation/paths.py`（资源路径）。完整参数可通过对应命令的 `--help` 查询，例如：

```powershell
& $python -m teleoperation hand record --help
& $python -m teleoperation hand export --help
& $python -m teleoperation replay l21-hand --help
```
