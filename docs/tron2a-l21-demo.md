# TRON2A + L21 仿真演示

本演示加载 TRON2A 双臂和左右 L21 手，周期性弯曲双肘及手指。它使用预设关节轨迹，不需要摄像头、Vision Pro、IK 或手部模型权重。

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

默认使用 `--display simplified`，因为本仓库缺少 TRON2A 的完整 STL 网格。简化模式仅用于观察运动学和基本动作，TRON2A 碰撞几何被禁用，不能用于验证接触或抓取。若已有完整模型，可通过 `--urdf <URDF路径> --display full` 使用它；URDF 引用的网格也必须存在。

未指定标定文件时，双手以单位安装变换固定到末端，手的实际安装位置和朝向未经标定。可使用 `--calibration <标定JSON>` 指定 `ArmCalibration` 文件中的双臂初始关节角和双手安装位姿；这不会在演示中启用摄像头遥操作。真实摄像头流程见 README 的 `dual calibrate-camera` 和 `dual camera --with-hands`。
