# 双臂跟随：抖动与延迟修复

> 2026-10-31：本文包含迁移前的算法实验和命令记录。当前相机输入、录制、回放与运行命令以 [SMPLer-X 链路](smplerx-arm.md) 为准；旧 Pose 输入和手部回退已删除。

适用入口：`python -m teleoperation.simulation.realtime_teleop`，或
`devtools/run_realtime_arm_windows.py`。输入是摄像头的 MediaPipe Pose world
肩、肘、腕；Vision Pro 的 TransHandR 数值预处理不经过本次修改。

## 定位与修改

1. **IK 热启动改变了实际仿真状态。** 原实现把第一遍位置解写入实际手臂，
   然后直接保留这个中间状态。只调用 `solve()`、不下发电机命令、不推进物理，
   G1 的单个关节就会改变约 0.35 rad。现在第二遍仍使用原来的热启动方法，
   完成后恢复求解前的位置和速度；异常退出也恢复。显示运动由电机和物理步进完成。
   `rest_poses="current"` 也在每次求解时读取当前关节位置，避免一直使用启动姿态。
2. **实时物理时间与采集时间不一致。** 原实现固定每帧推进 8 个 1/240 秒步长，
   只有采集为 30 FPS 时才对应实际时间。历史 G1 报告约为 12.77 FPS，
   物理时间因此只按实际时间的约 43% 推进。现在实时模式根据单调时钟计算步数，
   保留不足一个步长的小数；单次最多补 100 ms，避免长暂停后的集中追赶。
   离线 NPZ 回放仍固定步数。`--fixed-sim-steps` 可显式恢复实时固定步数。
3. **腕部姿态被错误地跨机器人共用。** 原入口拿机器人名称排序后的第一项姿态
   下发到所有机器人，默认三机器人时会把 G1 的 `None` 用给 H1。
   现在分机器人传入姿态，旧的共用四元数调用仍兼容。
   H1 的未标定/不可达腕姿态可能把末端位置拉远：若位姿解的位置误差超过
   profile 阈值且大于位置解误差，采用位置解。`arm.robots.<robot>.last_solve.<side>`
   记录 `position_fallback` 及两个候选解的误差；腕姿态误差仍如实保留。
4. **跟随比例选项不能解析机器人名称。** `--arm-gain g1 0.2` 原来会把 `g1`
   当浮点数而报错。现在支持按机器人覆盖，并在打开设备前拒绝未知机器人、
   零、负数和非有限比例。

没有添加额外的多帧平均滤波。原手指滤波、相对位移轴向、左右手标签、
机器人行程限制及摄像头坐标/模型契约继续使用既有实现。

## 运行与判断

在 PowerShell 中使用现有环境和已下载的手部/Pose `.task` 模型：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$session = Get-Date -Format 'yyyyMMdd_HHmmss'

& $python -m teleoperation.simulation.realtime_teleop `
  --robot g1 --side both --arm-trace `
  --no-arm-fallback `
  --report "outputs/tmp_realtime/arm_follow_$session.json" `
  --dump-npz "outputs/tmp_realtime/arm_follow_$session.npz"
```

先保持中性姿态约 30 个有效帧，再分别进行小幅左右、上下和前后移动。
`--no-arm-fallback` 使 Pose 丢失时保持上一目标，避免切换到另一套手部位置映射。
不加该参数时保留原来的兜底行为。

关注 `fps`、`bridge.clamped_by_robot`、`arm.residual_mm`、
`arm.robots.<robot>.last_solve` 和 `simulation_timing`。
如果频繁夹取行程，可先降低跟随比例，例如 `--arm-gain g1 0.2`，
再根据有效动作幅度调整；比例调整会减少动作幅度，不会扩大机器人可达空间。
不要只为消除夹取而放宽行程。

`simulation_timing.mode=elapsed` 表示实时物理随实际经过时间推进。
`capped_frames` 持续增加意味着循环经常超过 100 ms，需要继续测量采集、
检测、推理和渲染耗时。这次修改不承诺增加摄像头帧率，也不消除传感器检测噪声。
本次验收使用无相机测试与离线仿真；真人抖动和体感延迟仍需现场复测。

## 验证

```powershell
& $python -m unittest tests.test_arm_following tests.test_arm_ik_multi_robot tests.test_arm_realtime_bridge
& $python devtools/verify_realtime_arm_bridge.py --frames 60 --report outputs/tmp_arm_follow/bridge_verify.json
& $python -m unittest discover -s tests
```

修改前基线：318 项测试通过，1 项跳过。
新增回归覆盖双侧求解的位置/速度保持、求解异常恢复、13 FPS 下物理时间、
长暂停补偿上限、固定回放时钟、分机器人姿态与比例参数，以及实际回放循环。

修改后：完整套件 330 项测试通过，1 项跳过，退出码 0。
三机器人双侧 60 帧合成验收通过：稳态平均位置误差 G1 为 10.05/10.42 mm，
GR1 为 8.96/8.40 mm，H1 为 16.71/14.55 mm；81 次缺帧保持检查通过。
既有 36 帧 NPZ 仿真回放退出码 0，末帧双侧误差 H1 为 0.491/0.496 mm、
GR1 为 0.225/0.368 mm、G1 为 0.190/0.182 mm。后者是该测试序列的末帧数值，
不代表真人实时精度或整段平均误差。GBK 检查及 `git diff --check` 通过。

修改前提交：`07c738093b76ba94d7e5abfc0785f39b860adc3d`。
检查点：`checkpoint-20261022-1947-dual-arm-follow`。
