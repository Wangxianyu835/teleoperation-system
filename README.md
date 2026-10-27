# 遥操作与运动重定向

本项目使用摄像头检测人手关键点。
程序将关键点转换为 L21 机械手的关节角。
也可选择 dex-retargeting，直接计算三种机器人原装手的关节角，见 4.4 节。
PyBullet 在回放或实时运行时显示机械手动作。

本文沿用 ASD-STE100 的简明技术写作原则编写中文说明：使用短句，统一术语，每个步骤只说明一个操作。

常用操作流程如下：

**录制手部动作 → 自动对齐关键点 → 使用已有模型导出关节角 → 回放 L21 动作。**

如果已有兼容模型，则无需重新训练。

项目还支持 Vision Pro / VR 输入和双臂仿真，入口见第 5 节。
合并后的腕部位移路线在 30 Hz 和 13 Hz 合成验收中各通过 89 项检查。
新增默认方向路线已完成六条手臂的离线回放与诊断检查。
真人摄像头效果尚待复测。
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

### 2.1 使用 hand_capture 归一化录制工具（可选）

`hand_capture` 分支的独立工具位于 `MediaPipe/`。
在第 2 节设置的终端中，选择一条命令执行：

```powershell
# 录制双手的 21 点归一化坐标。
& $python 'MediaPipe/hand capture media.py'

# 同步录制双手和肩、肘、腕等 14 个身体关键点。
& $python 'MediaPipe/body capture media.py'
```

在摄像头窗口中按 `q` 停止录制。
手部文件保存在 `outputs/hand_capture/visual_hand_dataset/`。
身体与手部文件保存在 `outputs/hand_capture/body_hand_arm_dataset/`。
模型统一存放在 `datasets/`；这些独立工具在模型缺失时尝试下载。

执行以下命令，回放对应目录中最近保存的录制：

```powershell
& $python 'MediaPipe/read_hand_xyz.py'
& $python 'MediaPipe/body picture.py'
```

这些工具保留上游的 `normalized` 坐标，不输出米制手臂坐标。
训练前，先用 `hand align` 对齐，再选择坐标声明匹配的模型。
不要将这些数据直接传给要求 Pose world 米制输入的双臂程序。
需要同步录制米制关键点时，请使用[世界坐标关键点采集](docs/world-landmark-capture.md)。

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

### 3.4 检查原装手运动或导出快照（可选）

以下工具已从 `hand_capture` 迁入 `devtools/`，使用当前原装手映射接口：

```powershell
& $python devtools/check_native_hand_motion.py --file $angles
& $python devtools/render_hand_snapshots.py --file $angles --out outputs/hand_snapshots
```

第一条命令在无窗口仿真中测量关节运动范围和跟踪误差。
第二条命令生成三台机器人原装手的姿态快照。
它们接收关节角文件，不接收原始关键点录制。

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

### 4.3 实时驱动三机器人的双臂与双手（可选）

此入口已整合 `cs/multi-robot-arm` 的方向映射和肘部跟随。
默认同时驱动 H1-2、GR1-T2 和 G1。
手指继续使用第 3 节中坐标声明匹配的 checkpoint。
手臂使用 MediaPipe Pose world 米制肩、肘、腕关键点。

先将 Pose 模型放到 `datasets/pose_landmarker_lite.task`。
然后在前文设置的终端中执行以下命令：

```powershell
$armSession = Get-Date -Format 'yyyyMMdd_HHmmss'
$poseAsset = 'datasets/pose_landmarker_lite.task'
if (-not (Test-Path $poseAsset)) { throw "缺少 Pose 模型：$poseAsset" }

& $python -m teleoperation.simulation.realtime_teleop `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
  --camera-index $camera --model-asset-path $handAsset --pose-model-asset-path $poseAsset `
  --side both --no-arm-fallback --arm-map posture --arm-elbow-follow operator `
  --diagnostics --diagnostic-trace "outputs/tmp_realtime/arm_trace_$armSession.csv" `
  --report "outputs/tmp_realtime/arm_report_$armSession.json" `
  --dump-npz "outputs/tmp_realtime/arm_raw_$armSession.npz"
```

1. 保持双臂中性姿态约 30 个有效帧，让标定值收敛。
2. 分别小幅移动左右上臂。
3. 分别弯曲左右肘部。
4. 检查机械臂方向和手指动作。
5. 在摄像头窗口中按 `q` 或 `Esc` 停止。

默认 `track` 模式从第一帧有效观测开始跟随，同时收集中性姿态。
若需在标定完成前保持不动，添加 `--arm-calibrate-mode hold`。
命令中的 `--no-arm-fallback` 让 Pose 丢失时保持上一目标。

| 任务 | 参数或说明 |
|---|---|
| 只显示并驱动一台机器人 | 添加 `--robot g1`、`--robot h1_2` 或 `--robot gr1_t2`。 |
| 按人体上臂和前臂方向跟随 | `--arm-map posture`，这是当前实时入口的默认模式。 |
| 使用旧的腕部相对位移映射 | 改为 `--arm-map chain`。 |
| 使用分段位移并加入肘部位置目标 | 改为 `--arm-map segments --arm-elbow-task`。 |
| 关闭肘角参考跟随 | 将 `--arm-elbow-follow operator` 改为 `--arm-elbow-follow off`。标准入口默认是 `off`。 |
| 选择方向参考系 | 默认 `--arm-posture-frame auto`：GR1-T2 使用相对中性旋转 `delta`，H1-2 和 G1 使用 `mapped`。 |
| 选择 IK 关节参考 | 默认 `--arm-rest-poses current` 读取本帧关节角；`live` 是别名，`loaded` 使用加载时姿态，`zero` 使用零角度。 |
| 使用人体腕姿态约束 | 默认 `--arm-orientation off`；选择 `auto` 时按机器人 profile 启用。 |

`posture` 按机器人骨段长度构造方向目标，不使用旧位移模式的逐轴行程夹取。
实际运动仍受 URDF 关节限位和 IK 收敛情况限制。
方向跳变保护、目标变化速率限制和解分支恢复会影响最终跟随。
请结合报告检查效果，不要将源分支的历史现场指标当作本次实测结果。

CSV 保留原有请求目标字段，并增加 `ik_target_base`、`ik_target_world` 和 `ik_position_error_mm`。
新增字段表示经过目标限速和肩部跟随后实际交给 IK 的目标及其误差。
旧 NPZ 回放继续使用固定物理步数；新增方向保护和限速也使用该固定时间间隔。
真人摄像头效果需在本机复测。
整合细节见[分支整合记录](docs/branch-integration-2026-10-27.md)。

### 4.4 使用 dex-retargeting 直接驱动原装手（可选）

新增 `dex` 后端直接优化 H1-2、GR1-T2、G1 的原装手。它使用 Vector 优化，不需要手部网络权重，
也不经过 L21 降维映射。H1、GR1 每手各优化 6 个独立驱动量，再展开 URDF 耦合关节；G1 每手优化 7 个关节。

先按 [dex 环境与验收说明](docs/dex-native.md)建立独立环境，然后加载终端配置：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
. ./devtools/dex_native_env.ps1
& $dexPython -m pip check
```

使用已有 Vision Pro 实录校准、导出并回放六只原装手：

```powershell
$dexSession = Get-Date -Format 'yyyyMMdd_HHmmss'
$vrRaw = 'datasets/raw/visionpro_hand_arm_20261012_215137_live.h5'
$vrCal = "outputs/tmp_dex_native/vr_cal_$dexSession.json"
$vrNative = "outputs/tmp_dex_native/vr_native_$dexSession.h5"

& $dexPython -m teleoperation hand dex-calibrate --source visionpro `
  --input $vrRaw --start 3 --duration 2 --output $vrCal
& $dexPython -m teleoperation hand dex-export --source visionpro `
  --input $vrRaw --calibration $vrCal --output $vrNative
& $dexPython -m teleoperation replay dex-native --file $vrNative --render
```

此处 `--start 3` 仅适用于上面这份实录。其他录制应选双手张开至少两秒的片段。
校准与结果文件不覆盖已有路径。原生结果使用独立的 `dex_native_results_v1` 格式，不能作为 L21 角度文件读取。

相机输入只接受 world 米制数据。先录制一段双手张开、保持 2–3 秒的数据，再执行校准：

```powershell
$cameraRaw = "outputs/tmp_dex_native/camera_open_$dexSession.h5"
$cameraCal = "outputs/tmp_dex_native/camera_cal_$dexSession.json"
& $dexPython -m teleoperation hand record --landmark-space world --hand-side both `
  --camera-index $camera --output $cameraRaw
& $dexPython -m teleoperation hand dex-calibrate --source camera `
  --input $cameraRaw --start 0 --duration 2 --output $cameraCal
& $dexPython -m teleoperation.applications.realtime_hand_sim `
  --backend dex --dex-calibration $cameraCal --landmark-space world `
  --scene robots --hand both --camera-index $camera
```

如果开始录制时双手尚未张开，将 `--start 0` 改为张开片段的相对秒数。
双臂双手入口也支持 `--backend dex --dex-calibration $cameraCal`，其余手臂参数沿用 4.3 节。
相机与 VR 使用不同的校准配置。VR 新路线为 `dex_visionpro_palm_v2`，不能使用修正前的 v1 校准。

连接头显时，在支持 Tracking Streamer 的环境中使用：

```powershell
& $dexPython -m teleoperation hand vr --backend dex `
  --ip '<头显地址>' --dex-calibration $vrCal `
  --scene robots --hand both --hand-mapping same `
  --report "outputs/tmp_dex_native/vr_live_$dexSession.json"
```

独立环境已安装 `avp_stream==2.51+wincontrib` 本地兼容包，保留现有 OpenCV；SDK 导入检查通过。
安装命令与依赖适配说明见 [Vision Pro SDK 安装](docs/dex-native.md#安装-vision-pro-实时-sdk)。
头显实际连接与连续运行验收仍待完成。旧 TransHandR VR 入口仍保留。

已通过的验收包括：600 个原生可达姿态、两份 VR 实录的数值稳定性、原生命令导出/回放和完整回归。
六手求解单测 P95 为 22.1 ms，样本为预热后 1,000 帧，不含检测和渲染。
真实人体目标拟合误差仍为厘米量级，不能据此宣称达到抓取或硬件控制精度。
相机新校准、相机和头显各 5 分钟现场测试仍待完成；原生手本轮没有新增网格避碰。

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
| hand_capture 与双臂分支的迁移位置、兼容处理和验证记录 | [分支整合记录](docs/branch-integration-2026-10-27.md) |
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
