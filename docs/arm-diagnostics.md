# 机械臂诊断与离线验收

使用 [SMPLer-X 链路](smplerx-arm.md) 录制身体 H5，再通过同一控制流程回放和检查：

```powershell
python -m teleoperation.simulation.realtime_teleop --input-body outputs/session/body.h5 --backend geometric --headless --no-preview --report outputs/session/replay.json
python -m teleoperation.applications.arm_validation --input-body outputs/session/body.h5 --output-dir outputs/session/validation
```

录制使用 `smplerx_body_recording_h5_v1`，保留采集时间、模型身份、有效性、来源年龄与新结果标记。
离线回放不需要相机或模型推理。输入要求及环境配置见 SMPLer-X 文档。

## 检查内容

- 输入：不同的新身体结果、各侧有效率、过期与恢复状态、采集时间及结果年龄。
- 目标：中性标定进度、保持原因、目标行程与限速、肘腕目标几何。
- 求解：腕位置残差、腕姿态误差、上臂及前臂方向、肘支路恢复和位置降级。
- 执行：实际关节及末端运动、非驱动关节漂移、仿真步数与时间补偿。

报告中的 `observation` 或没有有效样本的指标不能作为通过证据。
录制、重复静态输入及合成测试只证明各自覆盖的行为。

## 真人检查

依次检查中性启动、静止、左右单臂、双臂、交叉、遮挡、离场和恢复。
确认左右方向正确，保持与恢复连续；观察腕姿态、肘弯方向、目标限速与实际末端误差。
退出后重新启动，确认相机、工作进程和仿真资源已释放。
