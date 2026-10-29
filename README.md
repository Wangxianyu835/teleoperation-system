# 遥操作与运动重定向

> **当前手部入口**：见 [统一手部训练、控制与结果规范](docs/hand-retargeting-unified.md)。使用 `hand train --hand-model ...`、`hand control --hand-model ... --backend nn|dex`、`hand export` 和 `replay hand-joints`。下文历史共享 L21 权重、18D 文件及三机器人降维映射示例仅对应旧配置，不能用来加载新版 D 维模型。

```powershell
python -m teleoperation hand list-models
python -m teleoperation hand train --input aligned_world.h5 --hand-model l21 g1 --hand-side right --run-name experiment01
python -m teleoperation hand control --hand-model l21 --backend nn --hand right --checkpoint <right_target_model.pth>
# dex 使用 devtools/dex_native_env.ps1 中的独立解释器；无需权重或人工掌长标定
python -m teleoperation hand control --hand-model l21 --backend dex --hand right
```

权重路径固定为 `outputs/hand_retargeting/models/<hand_model>/<side>/nn/camera_palm_length_window3_v1/<run_name>/`；同一人体米制 H5 可训练多个目标手。新相机窗口按三帧掌长中位数归一化。新 NN 与 dex 返回同侧同手型的独立原生角度；L21=17、H1/GR1=6、G1=7。接入能力不表示本地已有合格权重或跨操作者效果通过验收。


本项目将人手关键点转换为机器人关节角，并在 PyBullet 中显示动作。
输入设备可以是摄像头或 Vision Pro。
仿真模型包括 L21 手，以及 H1-2、GR1-T2 和 G1 的原装手。
摄像头入口还支持双臂跟随。
本文中的命令不连接实体机器人。

本文参考 [ASD-STE100](https://www.asd-ste100.org/STE_faq.html) 的写作原则：使用短句，统一术语，每个步骤只说明一个操作。
ASD-STE100 面向英语。本文是中文操作说明，不声明符合其英语词汇规则。

## 选择工作流

| 目标 | 输入 | 方法 | 操作入口 |
|---|---|---|---|
| 录制动作，再回放 L21 手 | 摄像头 | 使用模型权重计算 L21 关节角 | [第 2 节](#2-录制手部动作)、[第 3 节](#3-导出与回放关节角) |
| 训练 L21 共享模型 | 已对齐的摄像头数据 | 使用无关节角标签的数据训练模型 | [第 4.1 节](#41-训练共享模型可选) |
| 实时驱动 L21 双手 | 摄像头 | 使用模型权重计算 L21 关节角 | [第 4.2 节](#42-实时驱动-l21-双手可选) |
| 实时驱动三台机器人的双臂与双手 | 摄像头 | 使用 Pose world 数据驱动手臂 | [第 4.3 节](#43-实时驱动三机器人的双臂与双手可选) |
| 直接驱动三台机器人的原装手 | 摄像头或 Vision Pro | 使用 dex-retargeting 优化原装手关节角 | [第 4.4 节](#44-使用-dex-retargeting-直接驱动原装手可选) |
| 使用上游 TransHandR 模型驱动 L21 手 | Vision Pro 或 VR 录制 | 使用专用 VR 预处理和上游权重 | [TransHandR VR 工作流](docs/transhandr-vr.md) |

首次使用摄像头时，按以下顺序操作：

**准备环境 → 录制关键点 → 导出关节角 → 回放动作。**

如果没有兼容的模型权重，请在导出前执行第 4.1 节。
如果选择 dex-retargeting，请先阅读第 4.4 节的独立环境要求。

## 1. 准备环境

本文命令使用 Windows PowerShell。
示例路径来自本地开发计算机。请按实际安装位置修改路径。

### 1.1 设置终端

在每个新终端中，先执行以下命令：

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
```

缓存设置只影响当前终端。
缓存和临时文件保存在项目的 `outputs/` 目录中。

### 1.2 安装项目

如果已有可用的 `teleoperation` 环境，请转到第 1.3 节。
首次安装时，依次执行以下命令：

```powershell
conda env create -f environment.yml
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
```

`environment.yml` 记录 Windows x86_64 与 NVIDIA GPU 的已验证环境。
文件包含 CUDA 依赖。其他平台需要调整环境配置。

### 1.3 检查解释器与模型文件

执行以下命令，检查项目入口：

```powershell
if (-not (Test-Path $python)) { throw "找不到 Python 解释器：$python" }
& $python -m teleoperation --help
if ($LASTEXITCODE -ne 0) { throw '项目入口检查失败。' }
```

按所选工作流准备文件：

| 文件 | 用途 | 放置位置或要求 |
|---|---|---|
| MediaPipe Hand 模型 | 检测摄像头中的手部关键点 | `datasets/hand_landmarker.task` |
| MediaPipe Pose 模型 | 检测肩、肘、腕关键点 | `datasets/pose_landmarker_lite.task`；仅双臂流程需要 |
| 模型权重文件（checkpoint） | 计算 L21 关节角 | `.pth` 文件；坐标声明必须与输入一致 |
| dex 校准文件 | 固定输入尺度，计算原装手关节角 | 在第 4.4 节生成；摄像头与 VR 分别校准 |

本文用 checkpoint 表示保存模型权重的文件。
主程序不会自动下载这些模型文件。
`MediaPipe/` 中的独立采集工具有自己的下载行为，见第 2.1 节。

### 1.4 检查数据是否兼容

| 数据 | 内容 | 使用要求 |
|---|---|---|
| 摄像头原始 H5 | 每只手的 21 个关键点，同时保存 `normalized` 和 `world` 坐标 | 供对齐、检查或 dex 摄像头流程使用 |
| 摄像头对齐 H5 | 每只手的 25 个关键点，位于手掌局部坐标系 | 供 L21 训练或导出使用 |
| L21 关节角 H5 | 每只手的 18 维关节角，以及各侧有效性标记 | 供 `l21-hand`、`native-hand`、`mounted-hand` 回放使用 |
| dex 原装手结果 H5 | 按机器人和手侧保存原装手关节角 | 使用 `replay dex-native` 回放；格式为 `dex_native_results_v1` |
| TransHandR VR 结果 H5 | 保存 VR 模型输出和最终仿真命令 | 专用诊断格式，见 [VR 文档](docs/transhandr-vr.md#结果文件) |

`world` 表示米制手部坐标。`normalized` 表示图像归一化坐标。
对齐后，关键点以手掌为参考，但仍保留其来源坐标声明。

**摄像头 `world` 数据必须使用兼容的 `world` checkpoint。**
使用 VR 数据时，请选择明确的 VR 入口。
请勿将摄像头预处理用于 VR 数据。
请勿修改文件元数据来绕过兼容性检查。
完整要求见[数据契约](docs/contracts.md)。

## 2. 录制手部动作

先完成第 1 节。
然后在同一个终端中执行以下命令：

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

录制开始后，依次完成以下操作：

1. 将双手放在摄像头画面内。
2. 张开双手。
3. 握紧双手。
4. 分别弯曲各根手指。
5. 使用下表中的一种方式停止录制。

| 停止方式 | 操作 |
|---|---|
| 预览窗口按键 | 在预览窗口中按 `q` 或 `Esc`。 |
| 窗口按钮 | 关闭预览窗口。 |
| 终端按键 | 在终端中按 `Ctrl+C`。 |

正常停止后，程序保存原始文件 `$raw`，再自动生成对齐文件 `$aligned`。
无需再次执行 `hand align`。
H5 文件保存关键点和时间戳，不保存摄像头视频。

按需要修改录制设置：

| 目标 | 修改方法 |
|---|---|
| 选择其他摄像头 | 将 `$camera` 改为设备编号，例如 `1`。 |
| 只录制左手 | 将 `--hand-side both` 改为 `--hand-side left`。 |
| 只录制右手 | 将 `--hand-side both` 改为 `--hand-side right`。 |
| 录制 900 帧后停止 | 将 `--frames 0` 改为 `--frames 900`。 |
| 关闭预览 | 添加 `--no-preview`，然后使用 `Ctrl+C` 停止。 |

`--fps 30` 是请求的采集帧率。实际录制时间取决于设备与处理速度。
时间后缀用于区分每次录制的文件。

后续步骤使用当前终端中的变量。
如果打开新终端，请重新执行第 1.1 节，并设置所需文件路径。

### 2.1 使用 hand_capture 归一化录制工具（可选）

如果需要保留上游 `hand_capture` 的归一化数据，请选择以下一种工具：

```powershell
# 录制双手的 21 点归一化坐标。
& $python 'MediaPipe/hand capture media.py'

# 同步录制双手和 14 个身体关键点。
& $python 'MediaPipe/body capture media.py'
```

在摄像头窗口中按 `q` 停止。
手部文件保存在 `outputs/hand_capture/visual_hand_dataset/`。
身体与手部文件保存在 `outputs/hand_capture/body_hand_arm_dataset/`。
这些工具在 `datasets/` 中查找模型。模型缺失时，工具会尝试下载。

选择对应命令，回放最近保存的录制：

```powershell
& $python 'MediaPipe/read_hand_xyz.py'
& $python 'MediaPipe/body picture.py'
```

训练前，请先执行 `hand align`，再选择坐标声明匹配的 checkpoint。
这些工具不输出米制手臂坐标。
需要双臂输入时，请使用[世界坐标关键点采集](docs/world-landmark-capture.md)。

## 3. 导出与回放关节角

### 3.1 使用已有模型导出关节角

先完成第 2 节，或将 `$aligned` 设为已有的兼容对齐文件。

选择一个兼容的 `world` checkpoint。
下方路径是本地示例，不保证其他计算机上存在该文件：

```powershell
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_shared_20261014_000112/model_best.pth'
```

如果没有兼容的 checkpoint，请先执行第 4.1 节。
如果刚完成训练，请保留训练步骤设置的 `$checkpoint`。

执行以下命令，导出关节角：

```powershell
$angles = "outputs/hand_retargeting/camera_angles_$session.h5"
if (-not (Test-Path $checkpoint)) { throw "缺少模型权重文件：$checkpoint" }
if (-not (Test-Path $aligned)) { throw "缺少对齐文件：$aligned" }

& $python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles --device auto
if ($LASTEXITCODE -ne 0) { throw '关节角导出失败，请检查上方输出。' }
```

导出不使用摄像头。
成功后，`$angles` 指向 L21 关节角文件。

### 3.2 在同一窗口中回放两只 L21 手

执行以下命令，从正面观察双手掌面：

```powershell
& $python devtools/replay_capture_to_l21.py `
  --angles-h5 $angles --scene hands --hand both --view front `
  --gui --fps 30 --render-every 0 --hold-seconds 30
```

窗口显示两只 L21 手。
回放结束后，窗口保持最后姿态 30 秒。
此工具直接显示导出的原生关节角，不应用实时程序的输出滤波或碰撞检查。

也可以使用标准回放入口：

```powershell
& $python -m teleoperation replay l21-hand `
  --file $angles --hand both --render --speed 1.0
```

标准入口按录制时间戳控制回放速度。

| 目标 | 修改方法 |
|---|---|
| 只显示右手 | 在任一命令中将 `--hand both` 改为 `--hand right`。 |
| 工具入口只回放前 900 帧 | 添加 `--max-frames 900`。 |
| 调整工具入口的帧率上限 | 修改 `--fps`。 |
| 标准入口半速回放 | 将 `--speed 1.0` 改为 `--speed 0.5`。 |

两条命令都需要 `$angles` 关节角文件。
下次回放时，请将 `$angles` 设为已保存的文件路径。

### 3.3 在机器人模型上显示手部动作（可选）

以下两个入口都读取 L21 关节角文件：

| 入口 | 显示方式 |
|---|---|
| `native-hand` | 将 L21 关节角映射到机器人原装手，省略原装手无法复现的自由度。 |
| `mounted-hand` | 将 L21 手安装到机器人模型上。 |

执行以下命令，循环回放 H1-2 原装右手：

```powershell
& $python -m teleoperation replay native-hand `
  --file $angles --robot h1_2 --hand right --view right-hand --render --loop 0
```

执行以下命令，回放安装在 H1-2 上的 L21 右手：

```powershell
& $python -m teleoperation replay mounted-hand `
  --file $angles --robot h1_2 --hand right --render --speed 1.0
```

`--robot` 也支持 `gr1_t2` 和 `g1`。
需要直接优化原装手关节角时，请使用第 4.4 节的 dex 流程。

### 3.4 检查原装手运动或导出快照（可选）

选择以下工具，检查 L21 映射后的原装手动作：

```powershell
& $python devtools/check_native_hand_motion.py --file $angles
& $python devtools/render_hand_snapshots.py --file $angles --out outputs/hand_snapshots
```

第一条命令测量关节运动范围和跟踪误差，不打开仿真窗口。
第二条命令生成三台机器人原装手的姿态快照。

## 4. 训练模型或实时运行

### 4.1 训练共享模型（可选）

先完成第 2 节，或将 `$aligned` 设为已有的 `world` 对齐文件。
共享模型可以使用左手、右手或双手数据。
训练不需要机器人关节角标签。

执行以下命令：

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

训练成功后，依次完成以下操作：

1. 保留新设置的 `$checkpoint`。
2. 执行第 3.1 节的导出命令块。
3. 执行第 3.2 节的回放命令。

训练参数与模型兼容性要求见[工作流参考](docs/workflow-reference.md)。

### 4.2 实时驱动 L21 双手（可选）

此流程无需先录制数据。
先完成第 1 节，再设置以下变量：

- `$checkpoint`：兼容摄像头 `world` 输入的权重路径。
- `$camera`：摄像头编号，例如 `0`。
- `$handAsset`：MediaPipe Hand 模型路径。

如果刚完成第 2 节和第 3 节，可以继续使用当前变量。
执行以下命令，启动实时仿真：

```powershell
& $python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
  --camera-index $camera --model-asset-path $handAsset `
  --scene hands --hand both --view front --frames 0
```

在摄像头窗口中按 `q` 或 `Esc` 停止。
也可以关闭摄像头窗口，或在终端中按 `Ctrl+C`。
此入口不保存关节角 H5 文件。
输出滤波、角速度限制与碰撞检查见[手部输出说明](docs/hand-command-safety.md)。

### 4.3 实时驱动三机器人的双臂与双手（可选）

先按第 4.2 节准备权重、摄像头和 Hand 模型。
然后准备 MediaPipe Pose 模型。

此入口默认同时显示 H1-2、GR1-T2 和 G1。
手指使用 L21 checkpoint。
手臂使用 Pose world 米制肩、肘、腕关键点。

执行以下命令：

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

窗口打开后，依次完成以下操作：

1. 保持双臂中性姿态约 30 个有效帧。
2. 小幅移动左上臂。
3. 小幅移动右上臂。
4. 分别弯曲左右肘部。
5. 检查手臂方向和手指动作。
6. 在摄像头窗口中按 `q` 或 `Esc` 停止。

默认 `track` 模式从首个有效观测开始跟随，同时收集中性姿态。
如需等待标定完成后再运动，请添加 `--arm-calibrate-mode hold`。
`--no-arm-fallback` 使 Pose 丢失时保持上一目标。

| 目标 | 参数 |
|---|---|
| 只运行一台机器人 | 添加 `--robot h1_2`、`--robot gr1_t2` 或 `--robot g1`。 |
| 按上臂与前臂方向跟随 | 使用 `--arm-map posture`，这是当前默认模式。 |
| 使用旧腕部相对位移映射 | 改为 `--arm-map chain`。 |
| 使用分段位移与肘部位置目标 | 改为 `--arm-map segments --arm-elbow-task`。 |
| 关闭肘角参考跟随 | 改为 `--arm-elbow-follow off`；未指定时默认 `off`。 |
| 选择方向参考系 | 使用 `--arm-posture-frame auto`；GR1-T2 选 `delta`，其余两台选 `mapped`。 |
| 选择 IK 关节参考 | 默认 `--arm-rest-poses current`；其他值为 `live`、`loaded`、`zero`，其中 `live` 是 `current` 的别名。 |
| 启用腕姿态约束 | 改为 `--arm-orientation auto`；默认 `off`。 |

`posture` 按机器人骨段长度构造方向目标。
最终运动受 URDF 关节限位、目标限速和 IK 收敛情况影响。
报告中的 `ik_target_base`、`ik_target_world` 和 `ik_position_error_mm` 对应实际交给 IK 的目标及误差。

方向路线已完成六条手臂的离线回放与诊断检查。
真人摄像头效果仍需复测。
验收范围见[双臂诊断与验收](docs/arm-diagnostics.md)。
相机与 VR 的腕朝向现由输入层提供并自动传给手臂后端；初始对齐、启动命令及待执行的人体验收清单见[腕输入与验收](docs/wrist-input-and-acceptance.md)。本轮未运行自动测试或现场设备验收。
目前单独验证腕旋转请使用[独立彩色方块查看器](docs/wrist-viewer.md)：无需机器人、手指后端或 IK，旋转 1:1，支持相机和 VR。
参数适配与旧 NPZ 回放行为见[分支整合记录](docs/branch-integration-2026-10-27.md)。

### 4.4 使用 dex-retargeting 直接驱动原装手（可选）

`dex` 后端使用 Vector 优化器，直接计算三台机器人的原装手关节角。
它不需要 L21 checkpoint，也不经过 L21 关节角映射。
H1-2 和 GR1-T2 每手各优化 6 个独立驱动量，再展开耦合关节。
G1 每手优化 7 个关节。

#### 准备独立环境

先按 [dex 环境重建说明](docs/dex-native.md#环境重建)建立独立环境。
普通 `teleoperation` 环境不包含该后端的全部依赖。

在仓库根目录执行以下命令：

```powershell
. ./devtools/dex_native_env.ps1
if (-not (Test-Path $dexPython)) { throw '缺少 dex 独立环境，请先完成环境重建。' }
& $dexPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'dex 环境依赖检查失败。' }
$dexSession = Get-Date -Format 'yyyyMMdd_HHmmss'
```

后续 dex 命令均使用 `$dexPython`。
校准文件与结果文件不覆盖已有路径。重复运行时，请生成新的 `$dexSession`。

#### 使用 Vision Pro 录制导出与回放

先将 `$vrRaw` 设为已有的 Vision Pro 原始录制。
下方示例使用本地已有文件：

```powershell
$vrRaw = 'datasets/raw/visionpro_hand_arm_20261012_215137_live.h5'
$vrCal = "outputs/tmp_dex_native/vr_cal_$dexSession.json"
$vrNative = "outputs/tmp_dex_native/vr_native_$dexSession.h5"

& $dexPython -m teleoperation hand dex-calibrate --source visionpro `
  --input $vrRaw --start 3 --duration 2 --output $vrCal
if ($LASTEXITCODE -ne 0) { throw 'VR 校准失败。' }

& $dexPython -m teleoperation hand dex-export --source visionpro `
  --input $vrRaw --calibration $vrCal --output $vrNative
if ($LASTEXITCODE -ne 0) { throw '原装手关节角导出失败。' }

& $dexPython -m teleoperation replay dex-native --file $vrNative --render
```

`--start` 是相对于录制首帧的秒数。
示例中的 `--start 3` 只适用于这份录制。
使用其他录制时，请选择双手连续张开至少两秒的片段。

#### 使用摄像头实时驱动原装手

先准备 MediaPipe Hand 模型。
然后录制双手连续张开至少两秒的数据：

```powershell
$camera = 0
$handAsset = 'datasets/hand_landmarker.task'
$cameraRaw = "outputs/tmp_dex_native/camera_open_$dexSession.h5"
$cameraCal = "outputs/tmp_dex_native/camera_cal_$dexSession.json"

& $dexPython -m teleoperation hand record --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset --frames 0 --output $cameraRaw
if ($LASTEXITCODE -ne 0) { throw '摄像头校准录制失败。' }
```

在预览窗口中按 `q` 停止录制。
将 `$openStart` 设为双手开始连续张开的相对秒数，再执行校准：

```powershell
$openStart = 0
& $dexPython -m teleoperation hand dex-calibrate --source camera `
  --input $cameraRaw --start $openStart --duration 2 --output $cameraCal
if ($LASTEXITCODE -ne 0) { throw '摄像头校准失败，请检查张开手片段。' }

& $dexPython -m teleoperation.applications.realtime_hand_sim `
  --backend dex --dex-calibration $cameraCal --landmark-space world `
  --scene robots --hand both --camera-index $camera --model-asset-path $handAsset --frames 0
```

摄像头 dex 流程只接受 `world` 米制数据。
双臂双手入口也支持 `--backend dex --dex-calibration $cameraCal`。
使用该入口时，请替换第 4.3 节中的后端和 checkpoint 参数，并用 `$dexPython` 运行。
手臂参数保持相同设置。

#### 连接 Vision Pro 实时驱动原装手

先按 [Vision Pro SDK 安装说明](docs/dex-native.md#安装-vision-pro-实时-sdk)准备 `avp_stream`。
本地独立环境已安装依赖适配包 `avp_stream==2.51+wincontrib`。
该包保留 SDK 算法与网络协议，依赖变更记录在接入文档中。

1. 将电脑与头显连接到同一局域网。
2. 在头显中打开 Tracking Streamer。
3. 选择 Teleoperation。
4. 保持跟踪会话运行。
5. 将命令中的地址替换为应用显示的头显 IP。
6. 执行以下命令。

```powershell
& $dexPython -m teleoperation hand vr --backend dex `
  --ip '<头显地址>' --dex-calibration $vrCal `
  --scene robots --hand both --hand-mapping same `
  --report "outputs/tmp_dex_native/vr_live_$dexSession.json"
```

`$vrCal` 必须来自当前操作者的合格 VR 校准。
上方离线示例生成的校准仅在操作者与尺度适用时使用。
新操作者需先录制张开手片段，再按 VR 校准步骤生成自己的文件。
录制入口见[头显录制](docs/visionpro-offline-capture.md)。

Local Network 与 Remote 是并列连接信息区，无需寻找模式切换按钮。
在终端中按 `Ctrl+C`，或关闭 PyBullet 窗口，可停止运行。
VR 入口只控制手部，不生成肩、肘数据。

摄像头与 VR 使用不同的校准配置。
当前 VR 配置为 `dex_visionpro_palm_v2`，旧 v1 校准不能用于新路线。
需要上游 TransHandR 模型时，请使用[专用 VR 工作流](docs/transhandr-vr.md)。

#### 当前验证范围

截至 2026-10-28，记录中的验证状态如下：

| 项目 | 状态与范围 |
|---|---|
| 原装手可达姿态 | 600 个姿态通过验收。 |
| VR 离线数据 | 两份实录完成导出、回放和数值检查。 |
| 六手独立求解性能 | 预热后测试 1,000 帧，P95 为 22.1 ms；不含检测和渲染。 |
| 头显现场动作 | 已完成本人尺度校准；操作者确认动作清单。 |
| 头显连续运行 | 后续测试保存 2,409 次更新；接收新数据持续约 148 秒，未达到连续 5 分钟。 |
| 摄像头现场验证 | 新校准、方法比较、动作测试和连续 5 分钟测试仍待完成。 |
| 原装手碰撞检查 | 本轮没有新增原装手网格避碰。 |

人体目标拟合误差仍为厘米量级。
独立求解耗时不代表实时控制帧率。
数值检查结果不代表抓取精度或硬件控制精度。
完整指标、负载说明和待验收项目见 [dex 接入与验收记录](docs/dex-native.md)。

## 5. 查找其他功能与源文件

| 任务或主题 | 参考文档 |
|---|---|
| 灵巧手、机械臂与手臂协同重定向文献及项目定位 | [文献综述（2026-10-29）](docs/retargeting-literature-review-2026-10-29.md) |
| 手动对齐、训练参数、几何重定向和任务仿真 | [工作流参考](docs/workflow-reference.md) |
| 同步采集米制手部与肩、肘、腕关键点 | [世界坐标关键点采集](docs/world-landmark-capture.md) |
| TRON2A 标定、双臂双手控制和组合离线命令 | [身体与手臂工作流](docs/workflow-reference.md#4-扩展身体采集双臂和任务仿真) |
| Vision Pro 录制与离线数据 | [头显录制](docs/visionpro-offline-capture.md) |
| 上游 TransHandR VR 预处理、权重与 L21 推理 | [TransHandR VR 工作流](docs/transhandr-vr.md) |
| dex 环境、校准、性能与待验收项目 | [dex 接入与验收记录](docs/dex-native.md) |
| H5 格式、坐标与模型兼容性 | [数据契约](docs/contracts.md) |
| 输出滤波、限速与碰撞检查范围 | [手部输出说明](docs/hand-command-safety.md) |
| 模块职责与依赖关系 | [架构说明](docs/architecture.md) |
| 双臂离线验收与已测失败项 | [双臂诊断与验收](docs/arm-diagnostics.md) |
| 相机 / VR 腕朝向、初始对齐与人工验收清单 | [腕输入与验收](docs/wrist-input-and-acceptance.md) |
| hand_capture 与双臂分支迁移记录 | [分支整合记录](docs/branch-integration-2026-10-27.md) |
| 项目进展演示 | [演示材料](present/README.md) |

临时调整设置时，请使用命令行参数。
修改默认设置时，请查找以下文件：

| 默认设置 | 源文件 |
|---|---|
| 训练与模型 | [`config.py`](src/teleoperation/retargeting/hand/config.py) |
| 统一命令行接口 | [`schemas.py`](src/teleoperation/cli/schemas.py) |
| 资源路径 | [`paths.py`](src/teleoperation/paths.py) |

在命令后添加 `--help`，可查看全部参数。例如：

```powershell
& $python -m teleoperation hand record --help
& $python -m teleoperation hand export --help
& $python -m teleoperation replay l21-hand --help
& $dexPython -m teleoperation hand dex-calibrate --help
```

最后一条命令需要先加载第 4.4 节的独立环境。
