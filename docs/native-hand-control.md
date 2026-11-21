# 原生手部、Vision Pro 与统一控制

本分支接入 `teleoperation-system-retargeting` 的原生手部 NN/dex、三帧掌长输入、目标手型训练、Vision Pro 与接受命令录制回放。相机手臂使用本分支的 MediaPipe Pose、RealtimeArmBridge 方向映射和 paper IK。

来源固定在 `fe04415be19fcf1d0100bffc91bbe8cf01d8605c`。修改前提交为 `9b2b055ce075451fa3010ea996e35e002c219c8e`，检查点为 `checkpoint-20261121-1948-align-native-hands-vr`。代码来源文件与 SHA-256 见 [来源清单](native-hand-source-manifest.json)。

## 在这台电脑启动

已有统一环境包含 NN/dex 所需的依赖。以下入口显式选择当前克隆的源码，并将 DLL 和子进程路径指向所选解释器。每个新 PowerShell 终端先执行下面三行。

```powershell
Set-Location 'D:\2026\code\teleoperation-preprint'
$python = 'D:\2026\code\teleoperation-system-retargeting\.envs\teleoperation\python.exe'
$runtime = '.\src\teleoperation\runtime_environment.py'
```

```powershell
& $python -B -E -s $runtime -m teleoperation hand list-models
& $python -B -E -s $runtime -m teleoperation robot control --help
```

这里复用已有解释器。代码从 `teleoperation-preprint/src` 加载。移机时将 `$python` 改为已安装 PyTorch、MediaPipe、dex-retargeting、NLopt 及其 Pinocchio 依赖的 Python 路径。原来的 `teleoperation-repro` 环境可继续运行旧流程，该环境尚未安装 dex-retargeting。

## 摄像头双手和双臂

只驱动 G1 的左右手手指，dex 无需 NN 权重。

```powershell
& $python -B -E -s $runtime -m teleoperation robot control `
  --robot g1 --source camera --parts hands --side both --backend dex `
  --camera-index 0 --frames 0 --preview
```

同时驱动 H1-2 的双臂和双手。当前克隆有手部检测模型，身体检测模型可显式引用本机已有文件。

```powershell
& $python -B -E -s $runtime -m teleoperation robot control `
  --robot h1_2 --source camera --parts both --side both --backend dex `
  --pose-model-asset-path 'D:/2026/code/teleoperation-system-retargeting/datasets/pose_landmarker_lite.task' `
  --arm-ik-solver paper --arm-calibrate-frames 30 --frames 0 --preview
```

标定开始时让肩、肘、腕和手掌入镜，手掌张开朝向摄像头，保持中性姿势到标定完成。`--side left/right/both` 选择侧别，`--parts hands/arms/both` 选择部位，`--robots h1_2 gr1_t2 g1` 同时选三台机器人。`q`、`Esc` 或 `Ctrl+C` 结束。`--headless` 关闭仿真窗口，该模式省略 `--preview`。

NN 后端按每种手型显式指定一份 shared 权重。

```powershell
& $python -B -E -s $runtime -m teleoperation robot control `
  --robot g1 --source camera --parts hands --side both --backend nn `
  --hand-checkpoint g1 'outputs/your-model/model_best.pth' --frames 0 --preview
```

## Vision Pro

将地址替换为设备 Tracking Streamer 显示的地址。

```powershell
& $python -B -E -s $runtime -m teleoperation robot control `
  --robot g1 --source visionpro --parts both --side both --backend dex `
  --ip 192.168.1.100 --frames 0
```

用原始矩阵 H5 离线运行时，将 `--ip` 替换为 `--input-h5 <实际文件>`。独立手入口为 `hand control --source visionpro --hand-model g1 --backend dex --hand both`，也接受 `--ip` 或 `--input-h5`。`hand vr --backend transhandr --checkpoint <实际上游权重>` 使用显式的上游 L21 兼容流程。

## 采集、训练、导出和回放

新原生手训练需要三点掌坐标的 canonical-right 米制 H5。采集或对齐时显式选择新 profile，区分原分支历史 L21 文件。

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$raw = "datasets/raw/native_$stamp.h5"
$aligned = "datasets/raw/aligned_native_$stamp.h5"
& $python -B -E -s $runtime -m teleoperation hand record `
  --input-profile camera_palm_length_window3_v1 --landmark-space world --hand-side both `
  --frames 900 --output $raw --aligned-output $aligned

& $python -B -E -s $runtime -m teleoperation hand train `
  --input $aligned --hand-model g1 --run-name "native_$stamp" `
  --checkpoint-root outputs/hand_retargeting/models --device cpu --epochs 100
```

已有原始米制数据可以通过 `hand align --input <原始H5> --output <新H5> --landmark-space world --input-profile camera_palm_length_window3_v1` 生成新训练输入。已经按旧掌坐标对齐的数据需要从原始记录重新对齐。

`--hand-model l21 h1_2 gr1_t2 g1` 可依次训练四种手型。每种手型生成一份左右共享的 canonical-right 权重。`--resume <model_last.pth>` 恢复同一实验的优化状态，`--init-checkpoint` 初始化兼容权重。

```powershell
$checkpoint = "outputs/hand_retargeting/models/g1/shared/nn/camera_palm_length_window3_v1/native_$stamp/model_best.pth"
$joints = "outputs/hand_retargeting/native_joints_$stamp.h5"
& $python -B -E -s $runtime -m teleoperation hand export `
  --input $aligned --output $joints --hand-model g1 --backend nn `
  --checkpoint $checkpoint --hand both --device cpu
& $python -B -E -s $runtime -m teleoperation hand inspect --angle-h5 $joints
& $python -B -E -s $runtime -m teleoperation replay hand-joints `
  --input $joints --hand-model g1 --hand both
```

Dex 导出使用同一条 `hand export` 命令，选择 `--backend dex` 并省略 `--checkpoint`。新手部结果使用 `hand_joint_results_v1`，包含原生求解结果和实际接受的关节命令。

## 机器人接受命令录制

给 `robot control` 增加 `--record-results <新H5>` 与 `--report <新JSON>`。`robot replay --robot <目标> --input <命令H5>` 直接按录制的物理步数执行。多目标录制使用相同的 `--robots` 选择回放。写入路径须为新文件。

## 实现边界与数值来源

- `contracts/native_coordinates.py`、`retargeting/hand/native_coordinates.py` 和 `applications/native_hand_processing.py` 移植来源版本的三点掌坐标、左右规范化和三帧掌长中位数归一化。历史 L21 接口继续由现有模块读取自己的版本化契约。
- `learning/joint_dataset.py`、`joint_losses.py`、`joint_trainer.py` 是目标原生关节训练。显式 `--hand-model` 选择这条路线。旧训练调用继续读取原有配置和权重。
- `applications/mediapipe_robot_arms.py` 将本分支原有 MediaPipe/Paper 数值链路组合到新的 RobotControlSession。ArmSceneDriver 将求解与电机写入分开，统一输出负责每个共享物理世界的步进。
- 相机采集的 Unix 毫秒在 `_CameraSource` 转成秒。Windows 实时控制使用 `perf_counter()` 产生严格递增的高精度时间。L21 碰撞查询在边界包装为原有 `L21HandDOFs` 类型。
- Vision Pro 的实时输入、矩阵回放、设备选择和资源关闭使用来源实现。NN/dex 共用设备适配后的掌长窗口。显式 TransHandR 模式保留其轴向、腕原点、三帧堆叠与堆叠后缩放。
- 本地上游为 `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py` 和 `model/model_poseformer.py`，文件哈希写入来源清单。测试从实际上游 AST 提取左手数值语句，并比较实际上游模型的同权重输出。右手适配遵循来源实现及上游相邻注释，双手路由与丢帧行为另有测试。

## 验证边界

软件验证覆盖原生手规格、NN/dex、归一化、断点恢复、Vision Pro 离线输入、实际 CLI、接受命令录制回放，以及旧相机、手臂和任务回归。同输入四种手型的左右手 dex 关节结果与来源代码数值一致。真实摄像头动态效果、在线 Vision Pro 和桌面 GUI 仍需现场验收。

本轮完整测试运行 639 项，636 项通过、3 项因缺少本地历史权重跳过，耗时 139.988 秒。39 个注册 handler 与帮助入口全部验证。额外的三机器人 Vision Pro 原始记录验证运行 22 帧，12 个臂/手部位均产生跟踪命令，接受命令文件回放通过。来源与目标的左右手预处理、八个 Dex 目标输出最大绝对差均为 0。
