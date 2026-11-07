# 遥操作与运动重定向

想从源码理解“相机数据如何变成机器人手动作”，先读[相机手部链路阅读说明](docs/hand-camera-reading-guide.md)：包含调用关系、数据形状和单位、类保存的状态，以及修改影响与对应测试。

手部输入现按“RGB相机 / Vision Pro → 设备适配 → 同一手部契约 → NN或dex → 原生命令”组织。使用 `hand control --source camera|visionpro --backend nn|dex --hand-model ...`，设备与算法独立选择；`hand vr` 的 NN/dex 也复用此入口。完整命令、数值规范与旧权重边界见[统一手部输入](docs/unified-hand-input.md)。

本项目将摄像头或 Vision Pro 的人体观测转换为机器人关节命令，并在 PyBullet 中显示动作。手部支持 L21、H1-2、GR1-T2 和 G1；摄像头还支持三机器人双臂跟随与 TRON2A 工作流。下列示例用于仿真，不连接实体机器人。

手部正式控制使用设备无关契约与目标手原生关节；相机机械臂使用独立SMPLer-X S32工作进程；TransHandR上游复现需显式选择。当前接口和仍保留的局部分支见[接口说明](docs/current-interfaces.md)。

## 选择工作流

| 目标 | 输入与方法 | 入口 |
|---|---|---|
| 录制、训练、导出和回放目标手 | 相机 world 关键点，NN | [相机数据与训练](#2-录制相机手部数据)、[导出与回放](#4-导出检查与回放) |
| 无权重实时驱动目标手 | 相机 world 关键点，dex | [实时控制](#5-实时控制目标手) |
| 同时训练多种机器人手 | 同一份人体米制 H5，每个手型独立 shared NN | [训练模型](#3-训练目标手模型) |
| 三机器人双臂与双手仿真 | SMPLer-X 身体和腕姿态、MediaPipe Hands 手指 | [相机机械臂](#6-smpler-x-相机机械臂录制与回放) |
| RGB / VR 使用相同 NN 或 dex | 统一三帧掌长手部契约 | [统一手部输入](docs/unified-hand-input.md) |
| 显式复现上游 TransHandR 权重 | Vision Pro 实时流或 VR H5，`--backend transhandr` | [Vision Pro](#7-vision-pro-专用流程) |
| TRON2A 标定与双臂控制 | SMPLer-X 身体观测与显式安全配置 | [SMPLer-X 配置](docs/current-interfaces.md)、[工作流参考](docs/current-interfaces.md) |

首次使用新版相机 NN，按以下顺序操作：**准备环境 → 录制 world 数据 → 训练目标手模型 → 导出 → 回放**。已有严格匹配的新版权重时，可跳过训练。dex 不需要 NN 权重，但需要独立环境。

## 1. 准备环境

### 1.1 安装并设置终端

以下命令使用 Windows PowerShell。在仓库根目录执行，按实际位置修改解释器路径。

首次安装：

```powershell
conda env create -f environment.yml
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
```

`environment.yml` 记录 Windows x86_64 与 NVIDIA GPU 的已验证配置，包含 CUDA 依赖。其他平台需要调整配置。dex 和 SMPLer-X 使用各自的独立环境，见后文。

在每个新终端中设置：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$cacheRoot = Join-Path (Get-Location).Path 'outputs/tmp_readme/cache'
foreach ($cacheName in @('PIP_CACHE_DIR', 'MPLCONFIGDIR', 'HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TEMP', 'TMP')) {
    $cachePath = Join-Path $cacheRoot $cacheName
    New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
    Set-Item -Path "Env:$cacheName" -Value $cachePath
}

if (-not (Test-Path $python)) { throw "找不到 Python 解释器：$python" }
& $python -m teleoperation --help
& $python -m teleoperation hand list-models
```

缓存设置只影响当前终端。生成文件和临时缓存位于 `outputs/`。

### 1.2 准备所需资产

| 资产 | 用途与要求 |
|---|---|
| `datasets/hand_landmarker.task` | MediaPipe Hands 相机检测；主程序不会自动下载 |
| 新版目标手 `.pth` | NN 训练产物，schema 为 `hand_joint_nn_v1`，严格匹配手型、shared canonical right 输出、规格与输入 profile |
| `.venv/dex-native/` | dex 独立解释器和依赖；见 [环境重建](docs/current-interfaces.md) |
| SMPLer-X 官方源码、S32 权重、许可人体模型与工作进程 | 仅相机身体链路需要；见 [环境和资产](docs/current-interfaces.md) |
| 上游 TransHandR 权重与实时 SDK | 仅相应 VR 流程需要；见 [VR 文档](docs/current-interfaces.md) |

本地权重、人体模型和工作进程不随 Git 仓库分发。接入某个手型不表示本机已有经过验收的权重。

### 1.3 手型与数据兼容性

| `--hand-model` | 每侧独立关节数 D | 展开关节数 | 相机 NN / dex |
|---|---:|---:|---|
| `l21` | 17 | 17 | 已接入 |
| `h1_2` | 6 | 12 | 已接入 |
| `gr1_t2` | 6 | 11 | 已接入 |
| `g1` | 7 | 7 | 已接入 |

NN 每个目标手型使用一份左右手共享的 canonical right 权重。dex 直接优化目标手独立关节，再展开 URDF 耦合关系。L21 的原生 17D 与历史模型内部 18D 布局不同。

| 文件 | 内容与入口 |
|---|---|
| 相机原始 H5 | 21 点 `world` 米制坐标及 `normalized` 图像坐标；用于对齐与检查 |
| 相机对齐 H5 | canonical right 掌部坐标中的 25 点，仍保存米制数据；供新版训练与导出 |
| `hand_joint_results_v1` | 目标手原始 D 维解、展开后的命令、有效性与诊断；使用 `replay hand-joints` |
| `smplerx_body_recording_h5_v1` | RGB、原始模型结果及身体、手指和腕观测；使用 `--input-body` 回放 |
| 原始 Vision Pro 矩阵 H5 | `hand control --source visionpro --input-h5`或`hand vr`，不接受旧相机角度文件 |

正式手部控制默认profile为`hand_palm_length_window3_v1`，RGB和VR先各自适配，再建立相同三帧掌长规范。既有相机训练profile`camera_palm_length_window3_v1`数值处理相同，保持严格权重校验与可读取性；人体H5仍为米制。无需人工掌长标定，但换操作者和跨设备效果需单独验收。

不要将图像归一化坐标、机器人根坐标或旧18D权重改标签后作为新版输入。VR正式NN/dex经专用设备适配进入共同掌长规范；显式TransHandR复现仍使用其原始预处理。完整定义见[统一手部输入](docs/unified-hand-input.md)。

## 2. 录制相机手部数据

先完成环境设置，检查 Hand 模型，然后录制：

```powershell
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$camera = 0
$handAsset = 'datasets/hand_landmarker.task'
$raw = "datasets/raw/camera_$session.h5"
$aligned = "datasets/raw/aligned_camera_$session.h5"

if (-not (Test-Path $handAsset)) { throw "缺少 MediaPipe Hand 模型：$handAsset" }
& $python -m teleoperation hand record `
  --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset `
  --width 640 --height 480 --fps 30 --frames 0 `
  --output $raw --aligned-output $aligned
if ($LASTEXITCODE -ne 0) { throw '录制失败，请检查上方输出。' }
```

将双手放入画面，依次张开、握紧并弯曲各根手指。在预览中按 `q` / `Esc`、关闭预览窗口或在终端按 `Ctrl+C` 停止。正常停止后保存原始 H5，并自动生成 `$aligned`，无需再对齐。此入口保存关键点和时间戳，不保存视频。

`--frames 900` 表示录制指定帧数；`--no-preview` 关闭预览；`--hand-side left|right` 选择单侧。`--fps` 是请求帧率，实际速度取决于设备与处理负载。

如果已有原始 world H5，单独执行一次对齐；不要给已对齐文件重复处理：

```powershell
& $python -m teleoperation hand align --input $raw --output $aligned --landmark-space world
```

## 3. 训练目标手模型

以下示例训练 L21 右手，不需要机器人关节角标签：

```powershell
$run = "camera_$session"
$modelRoot = 'outputs/hand_retargeting/models'

& $python -m teleoperation hand train `
  --input $aligned --hand-model l21 --run-name $run `
  --checkpoint-root $modelRoot --device auto `
  --epochs 100 --batch-size 64 --learning-rate 0.0001
if ($LASTEXITCODE -ne 0) { throw '训练失败，请检查上方输出。' }

$checkpoint = "$modelRoot/l21/shared/nn/camera_palm_length_window3_v1/$run/model_best.pth"
```

将 `--hand-model l21` 改为 `--hand-model l21 h1_2 gr1_t2 g1`，可对同一份人体数据依次训练多个手型。每个手型只生成一份 shared checkpoint，训练同时使用左右手窗口；模型输出采用 canonical right 角度语义，运行时转换到实际左手或右手。训练器使用训练与验证分区中可用的有效侧窗口。

保存路径固定为：

```text
outputs/hand_retargeting/models/<hand_model>/shared/nn/
  camera_palm_length_window3_v1/<run_name>/
    model_best.pth
    model_last.pth
    config.json
    dataset.json
    metrics.jsonl
    training.log
```

已有目录拒绝覆盖。`--resume <model_last.pth>` 在同一目标、运行配置与数据下恢复训练；`--epochs` 表示总轮数。`--init-checkpoint` 仅初始化严格匹配的新 shared 模型参数，与 `--resume` 互斥。详细约束见[训练与保存路径](docs/current-interfaces.md)。

短训练只能检查流程。跨操作者效果需要按人划分数据并做独立验收。

## 4. 导出、检查与回放

继续使用训练产生的 `$checkpoint`，或将其设为已有的同手型 shared 新版权重：

```powershell
$results = "outputs/hand_retargeting/nn_results_$session.h5"
if (-not (Test-Path $checkpoint)) { throw "缺少模型权重：$checkpoint" }

& $python -m teleoperation hand export `
  --input $aligned --output $results --hand-model l21 --backend nn `
  --hand right --checkpoint $checkpoint --device auto
if ($LASTEXITCODE -ne 0) { throw '关节角导出失败，请检查上方输出。' }

& $python -m teleoperation hand inspect --angle-h5 $results
& $python -m teleoperation replay hand-joints --input $results --hand-model l21 --hand right
```

导出不打开摄像头。结果保留完整时间线，包括前两帧预热及缺手帧。回放验证手型、关节顺序、限位与耦合后直接执行已接受命令，不再次滤波。结果文件独占创建，不覆盖已有路径。

导出双侧 NN 时使用 shared `--hand both --checkpoint <shared.pth>`，回放也选 `--hand both`。结果统一使用 `replay hand-joints`。

## 5. 实时控制目标手

### 5.1 NN 控制

无需先录制数据；准备匹配的权重与 Hand 模型即可启动：

```powershell
& $python -m teleoperation hand control `
  --hand-model l21 --backend nn --hand both --checkpoint $checkpoint `
  --camera-index $camera --model-asset-path $handAsset `
  --record-results "outputs/hand_retargeting/live_$session.h5"
```

新版 shared NN 左右手使用同一个 `--checkpoint`。程序在打开相机和 GUI 前检查权重身份，不自动寻找最新权重。`Ctrl+C` 或关闭 PyBullet 窗口停止；`--frames N --headless` 可执行有限运行。侧专用或旧相机 checkpoint 不再作为控制入口接受。

### 5.2 dex 控制与离线导出

先按 [dex 环境重建说明](docs/current-interfaces.md)准备独立环境，再执行：

```powershell
. ./devtools/dex_native_env.ps1
if (-not (Test-Path $dexPython)) { throw '缺少 dex 独立环境，请先完成环境重建。' }
& $dexPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'dex 环境依赖检查失败。' }

& $dexPython -m teleoperation hand control `
  --hand-model g1 --backend dex --hand both `
  --camera-index $camera --model-asset-path $handAsset

$dexResults = "outputs/hand_retargeting/dex_results_$session.h5"
& $dexPython -m teleoperation hand export `
  --input $aligned --output $dexResults --hand-model g1 --backend dex --hand both
& $dexPython -m teleoperation replay hand-joints --input $dexResults --hand-model g1 --hand both
```

RGB和Vision Pro的正式dex均无需NN权重或人工掌长标定。将`g1`换为其他登记手型即可；VR用`--source visionpro --ip`或`--input-h5`，后端与命令策略相同。

两种方法共用滤波、URDF 有效限位、关节限速和耦合展开。缺手时保持上一条接受命令，恢复时限制首步时间预算。L21 保留网格路径检查；其他原装手没有等价运行时网格检查。方法比较与指标说明见[统一手部规范](docs/current-interfaces.md)。

## 5.5 统一机器人控制、部位显示与命令回放

新主入口按“输入采集一次 → 手部/机械臂并列求解 → 目标规格约束 → 接受命令帧 → 统一输出 → 仿真或录制”组织。RGB 相机和 Vision Pro 在手部重定向入口处是并列来源；相机机械臂使用 SMPLer-X 的身体与腕观测，Vision Pro 机械臂使用腕姿态，二者都在接受命令边界汇合。

```powershell
# 默认只显示并控制一台完整机器人
& $dexPython -m teleoperation robot control --robot g1 --source camera --backend dex

# 只有右侧机械臂，场景中不加载完整身体
& $dexPython -m teleoperation robot control --robot g1 --parts arms --side right --source camera --backend dex

# 需要对比时才显式打开多机器人
& $dexPython -m teleoperation robot control --robots h1_2 gr1_t2 g1 --source camera --backend dex

# 录制统一接受命令；回放不重新推理或求解
& $dexPython -m teleoperation robot control --robot g1 --input-body $body --parts both --record-results outputs/robot_commands.h5
& $python -m teleoperation robot replay --robot g1 --input outputs/robot_commands.h5 --headless
```

`--parts both` 显示一台目标的整机，`--parts arms` 和 `--parts hands` 只显示所选侧的原装机械臂或手；`--side left|right|both` 控制侧面集合。多目标必须使用 `--robots`，且至少两个目标。TRON2A 机械臂需要显式相机标定；TRON2A 的 `--geometry simplified` 只表示简化显示几何，不能当作原装网格验收。

录制格式为 `robot_command_results_v1`，每帧保存目标规格身份、部位关节顺序、接受位置、来源时间及时基、tracking/holding 状态，以及输出回执中的物理步数和步长。回放先完整校验文件，再按录制的步数直接提交命令。

## 6. SMPLer-X 相机机械臂、录制与回放

相机身体链路使用 SMPLer-X S32；MediaPipe Hands 仅提供手指。一次 RGB 采集共享给 Hands 和独立模型工作进程，保留各自时间。手臂使用根相对米制肩、肘、腕及完整旋转链得到的腕朝向，再进入既有重定向、腕对齐、IK 和安全控制。

先按 [SMPLer-X 配置](docs/current-interfaces.md)准备源码、权重、许可人体模型和独立解释器。默认目录为 `.venv/smplerx-assets/SMPLer-X`、`.venv/smplerx-assets/smpler_x_s32.pth.tar`、`datasets/smpl_models` 和 `.venv/smplerx-worker/Scripts/python.exe`；可用对应 CLI 参数覆盖。

以下三机器人入口使用原生目标手 dex 后端，在 dex 独立环境运行，无需 NN 权重或相机人体标定：

```powershell
$armSession = Get-Date -Format 'yyyyMMdd_HHmmss'
$body = "outputs/smplerx/$armSession/body.h5"

& $dexPython -m teleoperation.simulation.realtime_teleop `
  --robot g1 --backend dex --side both --arm-map posture --arm-calibrate-mode hold `
  --camera-index $camera --model-asset-path $handAsset `
  --dump-body $body --report "outputs/smplerx/$armSession/live.json"

# 无相机、无 S32 推理，按原始采集间隔回放。
& $dexPython -m teleoperation.simulation.realtime_teleop `
  --robot g1 --input-body $body --backend dex --headless --no-preview `
  --report "outputs/smplerx/$armSession/replay.json"

& $python -m teleoperation.applications.arm_validation `
  --input-body $body --output-dir "outputs/smplerx/$armSession/validation"
```

旧三机器人入口现在要求显式选择目标；单目标使用 `--robot g1`，对比使用 `--robot h1_2 --robot gr1_t2 --robot g1`。新主入口推荐使用 `robot control`，默认只构造所选目标。首次加载工作进程约需十余秒，期间机械臂保持。请保持中性姿态等待标定完成。多人、检测框突然换位或离场超过两秒会锁定身体输入，需重开会话并重新标定。

只录制观测、不启动机器人仿真时使用：

```powershell
& $python -m teleoperation hand record-world --frames 300 --max-fps 5 `
  --camera-index $camera --model-asset-path $handAsset `
  --output "outputs/smplerx/$armSession/capture.h5"
```

`record-world` 等待首个身体结果后开始指定数量的采集；加 `--no-arms` 仅采集 Hands。组合 H5 保存无损压缩 RGB、原始模型包及最终观测。录制器在保存前持有本次数据，长录制会占用内存和磁盘。

三机器人手指使用各自原生 HandSpec。NN 后端用重复的 `--hand-checkpoint MODEL PATH` 为每种目标指定 shared 权重；dex 使用掌长规范，无需相机 dex 标定。`--input-npz` 仅支持手指调试，身体与腕观测使用上述身体录制格式。

当前实测的五分钟 S32 单模型静态图片运行得到 1,604 个新结果、5.34 Hz、p95 234 ms，未达到 10 Hz / p95 150 ms 目标。40 个真实模型结果完成 H5 往返和三机器人回放；这些输入是重复静态图片。真实相机、动态动作、遮挡及组合负载连续运行仍待验收，见[回归与验收状态](docs/current-interfaces.md)。

## 7. Vision Pro 专用流程

Vision Pro 与RGB在正式手部重定向中并列：`hand control --source visionpro`或快捷`hand vr --backend nn|dex`使用共同契约和后端。下面的`--backend transhandr`仅用于显式复现上游权重，其左传感器到右L21、三帧和固定缩放语义保留；该专用权重不能冒充统一shared模型。统一NN可复用数值规范匹配的目标手shared权重，详见[兼容性说明](docs/unified-hand-input.md)。

先配置 `$vrCheckpoint` 为真实上游权重、`$vrRaw` 为已有的 Vision Pro 原始 H5：

```powershell
$vrCheckpoint = 'D:/2026/code/TransHandR/TransHandR/checkpoint/models/thumb3/linker/model_final.pth'
$vrRaw = '<Vision Pro 原始录制.h5>'
& $python -m teleoperation hand vr `
  --backend transhandr `
  --input-h5 $vrRaw --checkpoint $vrCheckpoint `
  --source-hand left --hand right --scene hands --device cpu --frames 120 `
  --report "outputs/hand_retargeting/vr_replay_$session.json"
```

实时连接将 `--input-h5 $vrRaw` 换为 `--ip '<头显地址>'`，并安装实时 SDK。数值来源、源码哈希、左右侧边界适配和完整命令见 [TransHandR VR 文档](docs/current-interfaces.md)。VR 手部入口不生成肩肘数据。

训练、三机器人相机和TRON2A仍使用`camera_palm_length_window3_v1`；正式RGB/VR手部控制默认使用`hand_palm_length_window3_v1`。三机器人手指直接输出目标手原生关节。TransHandR VR内部18D数值保留在专用模型边界，对外为L21原生17D；TransHandR仅支持`--scene hands`，统一VR控制按目标手选择场景。

TRON2A 的 `dual camera --with-hands` 使用一份新版 L21 shared checkpoint；`dual export --hand-joints <hand_joint_results_v1.h5>` 读取已接受原生命令。双臂实时输入使用`RawTeleopObservation`。

## 8. 接口与验证

当前保留的 contracts、旧接口清理清单、VR 上游源码路径与哈希见 [当前接口](docs/current-interfaces.md)。任务基准 `sim run` 保留；删除的是旧 `applications/demo` 模拟VR/手套管线、`sim demo-joints` 和旧原装手有损展示入口。

```powershell
& $python -m unittest discover -s tests
& $python -m teleoperation hand control --help
& $python -m teleoperation.simulation.realtime_teleop --help
& $python -m teleoperation dual export --help
```

软件回归和静态图片推理不代替真实相机、头显、操作者变化与实体硬件连续验收。

## 9. 代码维护入口

机器人正式控制与回放使用`robot control/replay`；手部训练、导出和独立手场景继续使用`hand`命令。各入口的实现归属、保留诊断工具和录制格式见[代码归属表](docs/current-interfaces.md#当前代码归属)。本次清理没有改变重定向算法或训练数据契约。

Windows三个启动脚本仍保留给既有IDE配置使用，共用`devtools/windows_runtime.py`，用于旧相机预览/诊断；必须显式`--robot`，`--check`核对当前后端依赖及SMPLer-X路径。手/臂正式验收推荐上面的统一机器人入口。
