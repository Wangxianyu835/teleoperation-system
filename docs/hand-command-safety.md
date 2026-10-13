# 实时手部输出：滤波、角速度限制与 L21 指间碰撞检查

更新：2026-10-14。

原版 TransHandR 的核对结果见 [TransHandR 参考与差异](transhandr-safety-reference.md)。
当前模型已经沿用其三帧 Transformer、角度限位和 FK 点距离碰撞损失。
本页的在线滤波与网格路径检查是额外的执行层处理，不能称为原版 TransHandR 的已有实现。

## 输出链路

`模型/几何输出 -> 原生侧 L21 角度 -> 关节限位 -> 输出滤波 -> 角速度限制
-> L21 指间网格路径检查 -> 仿真命令`。

模型输入、权重与左右手角度契约没有改写。该层接在两个实时仿真入口的 `scene.apply`
之前，checkpoint 和 geometric 后端都经过它。离线导出与训练结果不经过该层。

默认采用与 dex-retargeting 的 `LPFilter` 相同的一阶低通递推：
`filtered += alpha * (target - filtered)`。为处理实际帧率变化，本项目将参数解释为
30 Hz 下的新样本权重，并按真实采样间隔换算。参考上游
[LPFilter](https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/optimizer_utils.py)
和 [SeqRetargeting](https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/seq_retarget.py)。

## 模型输出的 L21 双手仿真

在仓库根目录、已配置环境的 PowerShell 终端执行：

```powershell
$py = 'D:\Anaconda\envs\teleoperation\python.exe'
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_shared_20261014_000112/model_best.pth'
& $py -m teleoperation.applications.realtime_hand_sim `
    --scene hands --hand both --camera-index 2 `
    --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
    --joint-filter lowpass --filter-alpha 0.2 --joint-max-velocity 1.5 `
    --collision-margin-mm 1 `
    --report outputs/tmp_realtime/hand_commands.json
```

低通、角速度限制和 L21 指间检查默认开启；以上参数只是显式列出默认值。
`--filter-alpha` 越小越平滑，也会增加跟随延迟。`--joint-max-velocity` 单位是 rad/s，
1.5 rad/s 在 30 Hz 下每帧最多约 0.05 rad。速度限制也覆盖第一帧、丢手恢复和长时间间隔。
丢手时保持已接受的关节命令，不让卡尔曼滤波外推缺失的手。

选择已有常速度卡尔曼滤波器：

```powershell
--joint-filter kalman --kalman-q 1 --kalman-r 0.01
```

`--joint-filter none` 只关闭输出滤波，仍保留速度限制与碰撞检查。
原有 `--smoothing` 是几何后端内部平滑参数，checkpoint 后端不使用它；
模型输出现在使用 `--joint-filter`。不要把多个强滤波同时开启后产生的延迟当成检测失败。

## 碰撞检查的行为和边界

检查直接使用已加载 L21 URDF 的碰撞网格，检查不同手指的 link 对。
每次更新按不超过 0.02 rad 的关节步长采样从已接受命令到候选命令的路径。
检测到不满足距离约束时缩短该段运动，并逐关节保留可行的屈曲/侧摆；
最终同时运动的路径会再次检查。碰撞查询结束后恢复原仿真状态，查询本身不驱动机械手。

本仓库 URDF 的相邻掌骨在零姿态已有约 0.05 到 0.08 mm 的微小网格重叠。
只对初始相邻 `*_metacarpals` link 对记录基准，并允许再增加最多 0.1 mm 的数值容差，
报告中逐对声明。其他新接触和更深的穿插仍被拦截。若初始姿态有不属于这些情况的碰撞，
检查器会报错，不将其默认为可接受基准。

这是离散采样的仿真命令防护，不是连续碰撞证明，也不是硬件安全认证。
PyBullet 检查的是 URDF 加载后的碰撞形状（STL 通常被转换成凸碰撞体），
不等同于可见网格的逐三角形接触，结果可能偏保守。
它不覆盖同一根手指内部、手掌、两手互撞、外部物体或真实电机动态。
`--scene robots` 及 `simulation.realtime_teleop` 的三机器人原装手目前只使用
滤波和速度限制，不使用 L21 网格检查，因为它们的碰撞几何不同。
`--no-collision-guard` 可用于离线诊断对比；启动日志和报告明确标出是否有运行时网格检查。

## 报告与测试

`--report` 中的 `hand_commands` 给出滤波类型、角速度限制、限位次数、速度限幅次数、
原始和已接受命令的最大变化。`collision_guard` 分侧记录受阻次数、最后接触 link 对、
距离与初始掌骨容差。受阻动作可能无法完整跟随，需修正模型或输入，而不是将滤波当作避碰。

代码：`src/teleoperation/simulation/hand_commands.py`。
测试：`tests/test_hand_commands.py`，涵盖变帧率、尖峰、丢手恢复、关节限位、
真实网格碰撞投影、查询状态恢复和中间路径碰撞。
