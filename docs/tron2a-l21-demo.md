# TRON2A + L21 仿真演示

本演示加载 TRON2A 双臂和左右 L21 手，周期性弯曲双肘及手指。它使用预设关节轨迹，不需要摄像头、Vision Pro、IK 或手部模型权重。

先按 [README 的资产下载步骤](../README.md#1-环境与路径)准备本地模型；模型文件不随代码上传。

从仓库根目录的 PowerShell 启动（Python 路径可按本机环境修改）：

```powershell
Set-Location D:\2026\code\teleoperation-preprint
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
& $python -B -m teleoperation.applications.tron2a_l21_demo --seconds 20
```

默认打开 PyBullet 窗口；按 Ctrl+C 可提前结束。无窗口检查：

```powershell
& $python -B -m teleoperation.applications.tron2a_l21_demo --headless --seconds 4
```

默认模型为 `assets/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf`，配套网格在 `assets/tron2-robot-description/tron2a/DACH_TRON2A/meshes/`；L21 模型位于 `assets/l21/left/` 和 `assets/l21/right/`。查看完整几何模型：

```powershell
& $python -B -m teleoperation.applications.tron2a_l21_demo --display full --seconds 20
```

本机模型仍位于 `assets/Tron2a/` 时，先用完整网格静止检查安装：

```powershell
& $python -B -m teleoperation.applications.tron2a_l21_demo `
  --urdf assets/Tron2a/urdf/robot.urdf --display full --static --seconds 300
```

`--static` 保持双肘和手指初始姿态；去掉该参数可检查运动时的安装稳定性。

演示默认仍使用 `--display simplified`。简化模式仅用于观察运动学和基本动作，TRON2A 碰撞几何被禁用，不能用于验证接触或抓取。`--display full` 加载本地下载的网格和碰撞几何，但本演示仍是关节轨迹演示，未验证抓取。其他兼容模型可通过 `--urdf <URDF路径>` 指定。

原夹爪在生成的临时 URDF 中移除可视几何、碰撞和质量；原 `grasper_L/R_Link` 固定坐标系保留，继续用于 IK、FK 和标定。上游原始资产不被改写。

左右 L21 的网格在临时 URDF 中解析为各侧的绝对路径，防止 PyBullet 按相同的相对 STL 文件名缓存而串用左右模型，尤其是轴向不同的拇指网格。

未指定标定文件时，L21 根坐标安装在原夹爪的法兰坐标系，位置偏移为零，绕 X 轴旋转 180 度（四元数 xyzw 为 `[1,0,0,0]`）。该法兰的向外方向为 -Z，L21 的手指沿 +Z 延伸，翻转后手指朝腕部外侧。两种 L21 URDF 的质心和虚拟根节点差异由共享加载代码处理。仅关闭相接的腕部/手根安装面的碰撞，其他手部碰撞保留。实时场景按整个底座最低点计算离地高度，回放使用记录的实际基座位置。

可使用 `--calibration <标定JSON>` 指定 `ArmCalibration` 文件中的双臂初始关节角和双手安装位姿；这不会启用摄像头遥操作。显式自定义安装位姿继续生效。旧标定中明确标记 `identity_placeholder_not_calibrated` 的单位占位值会在加载时迁移到上述模型安装变换，并记录迁移元数据；不带此标记的显式安装值不被替换。该变换来自 URDF 模型，不代表实机适配板外参已经测量。真实摄像头流程见 README 的 `dual calibrate-camera` 和 `robot control --robot tron2a_l21 --parts both`。

演示与历史 H5 回放现在也通过共享 RobotPartSceneBuilder / RobotOutput 执行，已移除独立 TRON2A world。

安装回归：`python -B -m unittest tests.test_tron2a_l21_mount -v`。覆盖实际演示加载、夹爪移除、默认朝向、地面间隙、安装约束、旧占位迁移及自定义安装保护。完整网格运动测试需安装 TRON2A 与 L21 本地资产。

2026-11-25 验证：5 项安装回归全部通过；本地完整网格运动 4 秒，左右手安装位置误差均小于 2 mm，无手与机器人之间的接触干涉，手指保持离地。完整 `unittest discover -s tests` 运行 653 项，652 项通过、1 项跳过，耗时 161.007 秒。修复前提交为 `2fdf78bd214e3917e1d8d749bfb7c8479479d96b`，检查点为 `checkpoint-20261125-2101-fix-tron2a-l21-mount`。源代码 GBK 严格检查及 Git diff 空白检查通过。
