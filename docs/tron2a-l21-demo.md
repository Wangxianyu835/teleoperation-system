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

演示默认仍使用 `--display simplified`。简化模式仅用于观察运动学和基本动作，TRON2A 碰撞几何被禁用，不能用于验证接触或抓取。`--display full` 加载本地下载的网格和碰撞几何，但本演示仍是关节轨迹演示，未验证抓取。其他兼容模型可通过 `--urdf <URDF路径>` 指定。

未指定标定文件时，双手以单位安装变换固定到末端，手的实际安装位置和朝向未经标定。可使用 `--calibration <标定JSON>` 指定 `ArmCalibration` 文件中的双臂初始关节角和双手安装位姿；这不会在演示中启用摄像头遥操作。真实摄像头流程见 README 的 `dual calibrate-camera` 和 `robot control --robot tron2a_l21 --parts both`。

演示与历史 H5 回放现在也通过共享 RobotPartSceneBuilder / RobotOutput 执行，已移除独立 TRON2A world。
