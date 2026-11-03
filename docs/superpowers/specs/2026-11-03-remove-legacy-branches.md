# 底层旧分支删除范围说明

基线：`86cabc4ea0aa8cbbadbf098223b1ff82bceb9515`。本文件将用户“计划一下怎么把部分底层旧分支删除”的要求落实为可执行范围；此次仅编写计划，不执行删除。

## 目标

相机手部只使用 world 米制人体数据和 `camera_palm_length_window3_v1` 控制输入；相机 NN 只加载每个目标手型一份 shared canonical right 权重；TRON2A 臂映射只接受 SMPLer-X 身体观测。旧数据、旧权重和旧分支直接拒绝，不提供迁移器、别名或隐式回退。

## 全局约束

- Python >=3.10，Windows PowerShell，使用现有 unittest；不新增运行依赖。
- 保留 CameraFrame、RawHandFrame、UpperBodyObservation、RawTeleopObservation、WristObservation、CanonicalHandFrame、HandWindow、HandJointResult、HandJointCommand 的当前结构。
- 保留手部原生命名关节、HandSpec 哈希、有效限位、速度约束、mimic 展开和接受命令回放语义。
- 保留当前相机 world H5、shared NN checkpoint、新版手结果 H5 和 SMPLer-X 身体 H5 的实际数值与有效性语义。
- 保留 MediaPipe normalized 图像点和 SMPL-X 图像投影作为预览/记录数据；禁止将其作为米制重定向输入。
- 保留上游 TransHandR VR 轴变换、三帧、固定缩放、原始18D输出及边界17D适配；禁止应用相机预处理或重写权重元数据。
- 删除范围不含现存数据文件、模型权重、机器人资产、依赖环境和上游 TransHandR 工程。
- 每个实现批次在改动前检查 Git 状态、保存 dirty intended state、创建 annotated checkpoint；完成验证后单独提交，不推送。

## 三个独立子项目

### A：TRON2A legacy 臂映射

删除绝对腕位置输入、旧工作空间静默 clip 和旧退化前臂处理分支。`input_mode` 参数只接受 `smplerx`，默认改为 `smplerx`；保留参数以免所有当前显式调用发生无意义签名变动。旧模式必须在控制器分配 IK 资源前拒绝。保留 fixed/forearm 朝向选择、近平直朝向保持、肩相对位置、工作空间显式拒绝、IK FK 核验和安全控制。

### B：旧相机 normalized 重定向

相机单帧 processor、pipeline、离线 alignment 和 raw writer 都只接受 world。保留 `landmark_space` 参数作为 world-only 边界，默认 world；显式 normalized 立即报错，不映射为 world。

手部已对齐 H5 reader 只接受 canonical world 米制布局；原始对齐入口要求真实 world 数据集和米制元数据，不再根据缺失属性推断 normalized。旧 `palm_local_to_l21_v2` 不再登记为受支持的相机对齐方式。

删除 `align_mediapipe_hand(..., robot_basis=...)` 旧分支，保留 `align_mediapipe_hand(points21, *, side)`。`build_l21_reference_basis` 仍被几何诊断和 VR 等价验证使用，保留。固定 source→L21 变换仍有上游/诊断用途，本轮不以历史命名为由删除。

保留 L21 几何诊断工具，但将它及其探针改为显式 canonical world 米制输入。不得为了保留探针而继续保留正式相机 normalized 分支。

### C：侧专用 NN 权重与失效辅助代码

删除 `joint_checkpoint_payload`、`load_joint_checkpoint` 和仅按侧模型组织的 `JointNNRetargeter`；保留 shared writer/loader、SharedJointNNRetargeter、HandRetargeter 和通用模型构建器。训练路径只接受 shared，新侧专用目录请求拒绝。

迁移现有测试到 shared 路线，保留动态输出维数、规格错配、限位篡改、无效模型输出、缺手不推理、左右轴适配与精确恢复训练的覆盖。

删除无消费者的旧18D文件审计模块 `retargeting/hand/diagnostics.py`，删除无消费者的身份跟踪模块 `retargeting/hand/tracking.py` 及悬空 import；不删除当前确实使用的身体 tracker、Kalman、VR18D网络与FK配置。

## 不并入此次实现

损坏的旧模块启动尾部、统一 CLI 收敛、sim demo/dummy 删除、任务体系改造、preflight 环境工具重建分别属于其他交付；本计划只记录这些现状，不搭便车重构。只修正本次改动涉及的 README/current-interfaces 与坐标诊断描述。

## 验收

每个子项目独立运行行为测试和完整 unittest；B/C 额外覆盖 dex 独立环境，B 核对真实本地 TransHandR 上游数值等价。禁止把删除旧行为测试等同于新行为通过；既有语义测试必须迁移或明确用“旧输入拒绝”测试取代。不能用测试数量不变作为验收标准，必须记录差异原因。真实设备与性能验收不由本次软件测试代替。

## 回滚

每个批次保留独立 annotated tag 和提交，可 `git diff <tag> -- <path>` 审查，或 `git switch -c rollback-<description> <tag>` 建恢复分支。旧数据要追溯时从原有 checkpoint 查看，不向当前代码加兼容 reader。
