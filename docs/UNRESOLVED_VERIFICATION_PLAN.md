# 待验证清单

软件结果见 [P0/P1 验证报告](P0_VERIFICATION_REPORT.md)，坐标、单位和尺度路径见 [坐标说明](COORDINATE_SYSTEMS.md)。以下工作尚未完成；合成测试和 checkpoint 可加载均不能替代真实数据验证。

## P0：真实数据与物理一致性

| 工作 | 最小实验与验收证据 |
|---|---|
| 左右手标签和镜像 | 已知物理左右手在实际采集路径分别记录镜像/非镜像帧、MediaPipe category 和身份；明确 H5 标签是否需要交换 |
| URDF 基坐标与关节方向 | 渲染两侧 URDF/STL 及基坐标轴，逐个正向移动 17 个关节；保存图像/视频与轴方向、屈伸/侧摆和左右镜像对应表 |
| source-to-L21 物理对应 | 标记基向量通过实际对齐 CLI，对照已确认的 URDF 坐标；张掌/屈指姿态检验双侧掌法向、指向与拇指方向 |
| 人手与 FK 长度尺度 | 同一采集配置用尺或已知掌宽测量，至少两个相机距离；声明源与 URDF 单位，比较跨度，解释 source/robot scale 和 collision 阈值 |
| 独立录制质量 | 非训练录制评估有效输出率、限位、抖动和姿态误差；按修正目标重评旧 checkpoint，建立可比较基线 |
| 离线与实时预处理 | 相同 21 点观测贯穿两条路径，确认标签、坐标旋转、尺度和缓存重置是否与 aligned H5 一致 |

状态：`MANUAL VERIFICATION REQUIRED` / `PHYSICAL SCALE NOT VERIFIED`。保留当前变换和尺度，先取得证据再提出修改；本轮没有重训、摄像头实验或硬件操作。

## 硬件接入前

从当前官方 L21 SDK/手册确认命令长度、单位、左右侧限位和保留字段，在受控台架逐关节记录字节与方向、读回已知姿态。完成内部 17D → 硬件映射后再接入；历史 CAN/RS-485 代码不构成当前硬件协议证据。见 [关节契约](L21_JOINT_CONTRACT.md#hardware-sdk-evidence-and-limits)。

## 后续软件补测与维护

- mock `MediaPipeCameraAdapter`：时间戳、handedness 解析、缺失/非有限帧后的逐侧重置与恢复。
- 覆盖实时完整窗口后的异常输入和起始缺失侧行为；区分模型 `None` 与下游 hold 策略。
- 补测 `track_identity=False`；明确腕部相对输入在两侧候选都可信时的标签歧义。
- 明确 H5 metadata 策略：目前只强制 `coordinate_frame`，不校验 alignment；标记不能证明实际执行过变换。
- 锁定可复现依赖，处理可选 OpenCV 与 NumPy 版本冲突，见 [开发环境](DEVELOPMENT_ENVIRONMENT.md)。

对齐事务、root/group H5、尺度传播、T1/T2 和训练安全门已有验证。手部协议与双臂扩展仍耦合，此边界记入 [数据契约](HAND_CONTRACT.md)；本轮不重构。
