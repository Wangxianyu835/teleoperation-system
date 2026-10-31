# 双臂跟随实现说明

当前相机身体输入、资产和运行命令见 [SMPLer-X 链路](smplerx-arm.md)，
映射与机器人配置见 [多机器人机械臂控制](multi-robot-arm.md)。

`PyBulletArmIK.solve()` 在热启动和候选求解后恢复关节位置与速度，异常时同样恢复。
`rest_poses=current` 每次读取实际关节角，`live` 是别名；`loaded` 使用加载姿态。
运动通过电机命令及物理步进执行。

`SimulationStepClock` 在实时模式按经过时间推进物理，保留不足一个步长的小数，
单次补偿最多 100 ms。固定步数可通过 `--fixed-sim-steps` 选择。
相机目标限速使用相邻采集时间差。

腕姿态按机器人、按侧传入。位姿候选的位置误差超过 profile 阈值且劣于位置候选时，
求解器可使用位置候选，并在 `last_solve` 中记录降级和候选误差。
`--arm-gain ROBOT GAIN` 支持分机器人覆盖，并在打开设备前验证参数。

验证命令：

```powershell
python -m unittest tests.test_arm_following tests.test_arm_ik_multi_robot tests.test_arm_realtime_bridge
```
