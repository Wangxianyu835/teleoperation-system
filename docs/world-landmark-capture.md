# 手部与身体采集

`hand record-world` 从同一次 RGB 采集中获取 MediaPipe Hands 手指数据和 SMPLer-X 身体、腕观测。
手部保存图像归一化坐标和米制 world 坐标；身体使用 SMPL-X 根相对米制坐标与图像投影。

```powershell
python -m teleoperation hand record-world --frames 300 --output outputs/session/capture.h5 --max-fps 5
```

使用 `--no-arms` 只录制手部，`--no-preview` 关闭预览。
组合录制采用 `smplerx_body_recording_h5_v1`，保留 RGB、模型结果、最终观测及独立时间。
根层手部数据仍供手部对齐工具使用；对齐时保留身体包，不重新解释身体坐标。

手部模型、SMPLer-X 源码、S32 权重、许可人体模型和独立工作进程需预先准备。
路径及可执行命令见 [SMPLer-X 链路](smplerx-arm.md)，手部训练与导出见 [统一手部流程](hand-retargeting-unified.md)。
