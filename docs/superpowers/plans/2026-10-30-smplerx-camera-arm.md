# SMPLer-X 相机机械臂替换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划默认由当前会话顺序实施；不隐含授权创建子代理。

**Goal:** 用 SMPLer-X 替换全部活动的相机机械臂 MediaPipe Pose 输入，身体位置与腕朝向同源，并保留手指、VR 和旧录制的兼容接口。

**Architecture:** 单相机采集后分别分发 Hands 与独立 SMPLer-X 工作进程，使用版本化人体观测与时序追踪接入现有机械臂控制。实时、标定、记录与离线验收共用新输入语义，最终删除旧 Pose 控制代码。

**Tech Stack:** Python 主工程 >=3.10；官方 SMPLer-X S32、PyTorch/CUDA、SMPL-X；OpenCV、NumPy/SciPy；PyBullet / 现有 Robotics Toolbox IK；unittest。推理进程依赖单独锁定，以实际兼容性测试为准。

**Spec:** [2026-10-30-smplerx-camera-arm-design.md](../specs/2026-10-30-smplerx-camera-arm-design.md)

## Global Constraints

- 相机机械臂的位置、骨段方向和腕朝向统一来自 SMPLer-X / SMPL-X。禁止用 MediaPipe Hands 的腕点或掌朝向补驱机械臂。
- 保持当前 public input、window、result 接口兼容；新增字段提供默认值，旧文件不重写来源。
- Vision Pro / TransHandR 数值链路继续显式独立；不套用相机预处理或新相机标定。
- TRON2A 是可选机器人之一，新感知适配器不得硬编码机器人型号。
- 本次不更换重定向/IK 后端；机械臂向量优化独立实施和验收。
- 起点 S32 / 单人 / batch=1 / FP32；不假定 6 GB 显存或指定帧率已验证。
- 旧 Pose 仅用于既有文件回放和离线基线，不保留实时回退开关。
- 每个任务修改前执行 git status、提交已有明确改动、创建 annotated checkpoint；完成相关测试后提交，不 push。

## Review Focus

1. 工作进程迟到、冻结或退出：主控制继续保持，旧结果不能冒充新数据。任务 2、4、5 测试。
2. 镜像、裁剪或局部旋转组合错误：左右手和轴方向必须由合成变换证明。任务 3 测试。
3. Hands 丢失但身体可见 / 身体丢失但 Hands 可见：两条有效性独立。任务 5 测试。
4. 旧标定、旧 H5/NPZ：来源显式，拒绝跨来源误用，旧文件按原语义读取。任务 3、6 测试。
5. 新人进入、主体切换、重复帧：不自动改操作者身份，不累计错误标定。任务 2、4 测试。

## 文件职责

下列是拟新增或修改的路径，不表示已经存在的新实现。

| 路径 | 职责 |
|---|---|
| `contracts/body.py` | 新人体推理结果与帧身份协议 |
| `inputs/smplerx.py` | 主环境客户端、配置校验、结果生命周期 |
| `inputs/smplerx_geometry.py` | SMPL-X FK、坐标与投影适配，不开相机 |
| `inputs/body_tracking.py` | 身体/腕有效性、保持和渐进恢复 |
| `applications/camera_teleop_source.py` | 单次捕获，分发 Hands 与身体推理，组合现有公共观测 |
| `applications/camera_arm_pipeline.py` | 实时/回放共同的机械臂控制步骤 |
| `external/smplerx_worker/worker.py` | 独立环境运行的轻量协议入口，不导入主工程 |
| `external/smplerx_worker/adapter.py` | 固定上游版本的初始化、单帧推理与模型参数提取 |
| `data/body_recording.py` | 新版本人体记录读写 |
| `devtools/verify_smplerx_runtime.py` | 原始模型 smoke、显存与耗时测试 |
| `docs/smplerx-arm.md` | 部署、迁移、坐标证据、验收记录 |

以上 `contracts/inputs/applications/data` 路径均相对 `src/teleoperation/`。公共共享内存或网络服务不在第一版范围内；本地长度前缀子进程协议即可。

## Task 1: 推理环境与真实模型可运行性

**Files:** Create `external/smplerx_worker/README.md`, `external/smplerx_worker/environment.lock.yml`, `external/smplerx_worker/adapter.py`, `devtools/verify_smplerx_runtime.py`, `docs/smplerx-arm.md`; Test `tests/test_smplerx_runtime.py`。

**Interfaces:** 提供 `verify_smplerx_runtime.py --check` 和 `--input-video PATH --report PATH`。参数包括 `--worker-python`、`--smplerx-root`、`--checkpoint`、`--body-model-dir`、`--model s32`；输出机器可读依赖、资产哈希和性能报告。

- [ ] 先写配置 smoke 测试：缺路径给出准确错误；主工程 import 不加载 torch/OpenMMLab；报告不把缺资产标为通过。
- [ ] 使用下方公共测试命令运行 `tests.test_smplerx_runtime`，确认因功能缺失失败。
- [ ] 固定官方源码版本，核对 S32 配置、许可证与所需资产。使用兼容 RTX 4050 的独立环境，安装成功后锁定实际版本；若转 WSL2，记录路径映射与 CUDA 检查结果。
- [ ] 完成单帧和短视频 warmup 推理，记录检测、裁剪、前向、SMPL-X 派生耗时、峰值 allocated/reserved 显存及输入输出 shape。不使用训练数据集下载作为推理前置。
- [ ] 运行单测及真实 smoke；没有合法取得的模型资源则标记该项未完成，禁止用 mock 输出宣称模型可运行。
- [ ] 提交：`build: isolate and verify SMPLer-X S32 runtime`。

## Task 2: 带时间与身份的推理协议

**Files:** Create `src/teleoperation/contracts/body.py`, `src/teleoperation/inputs/smplerx.py`, `external/smplerx_worker/worker.py`; Modify `external/smplerx_worker/adapter.py`; Test `tests/test_smplerx_protocol.py`。

**Interfaces:**
- `BodyFrameId(session_id: str, sequence: int, capture_time_s: float)`，时间为同一主进程单调秒。
- `SmplerXResult`：包含 frame_id、推理完成时间、model_id/hash、root/body/hand 旋转参数、shape、根相对关节位置、全局关节旋转、检测框与逆裁剪所需信息、valid/reason；数组带显式 dtype/shape，不能使用不可信 pickle。
- `SmplerXClient.submit(frame: CameraFrame, *, frame_id: BodyFrameId) -> None`。
- `SmplerXClient.poll() -> SmplerXResult | None`，`close() -> None`；不得修改捕获时间。

- [ ] 写协议测试：分片读取、非法长度/shape/version、乱序、重复、迟到结果、stderr 日志和 EOF；例如 `assert accepted.sequence > previous.sequence`，畸形包拒绝且不下发机器人命令。
- [ ] 运行 `tests.test_smplerx_protocol`，确认新增测试先失败。
- [ ] 实现消息格式与最大长度检查、单活动/单等待帧调度；`submit` 不被重推理阻塞，等待帧被最新帧替换。工作进程只推理指定单人，身份歧义返回失效。
- [ ] 用 fake 子进程验证延迟、崩溃、关闭回收与重启会话；新会话旧包拒绝，worker stdout 不混入上游日志。
- [ ] 接真实 S32 做小批视频往返，比较直接调用与协议输出的数值一致性；记录本地传输开销。
- [ ] 提交：`feat: add bounded SMPLer-X worker protocol`。

## Task 3: SMPL-X 坐标、骨架与腕朝向适配

**Files:** Create `src/teleoperation/inputs/smplerx_geometry.py`; Modify `external/smplerx_worker/adapter.py`, `src/teleoperation/contracts/observations.py`, `src/teleoperation/retargeting/arm/wrist_orientation.py`, `src/teleoperation/retargeting/arm/calibration.py`, `src/teleoperation/retargeting/arm/retargeter.py`; Test `tests/test_smplerx_geometry.py`, `tests/test_wrist_observations.py`。

**Interfaces:** `smplerx_observations(result: SmplerXResult) -> tuple[UpperBodyObservation, dict[str, WristObservation]]`；`UpperBodyObservation` 新增可选 `metadata`，默认空映射，保留旧构造方式。

- [ ] 写合成骨架测试：零姿态、根旋转 90 度、肩/肘局部旋转组合、左右侧独立、q/-q 等价、逆裁剪投影、det(R)=+1、非法输入。断言 FK 腕位置与父子旋转链一致，不能直接取 body_pose 中某三个数当全局腕朝向。
- [ ] 运行 `tests.test_smplerx_geometry tests.test_wrist_observations`，确认新增断言先失败。
- [ ] 从固定版本 SMPL-X 模型关节名称/父节点关系建立映射，在工作进程 adapter 中用模型 FK 得到同一根参考的肩肘腕与全局旋转；主工程 geometry 只用 NumPy/SciPy 消费派生结果，不能引入第二套 SMPL-X/PyTorch 依赖。不要硬套 MediaPipe 点号或人体模型内部重排后的数组。
- [ ] 实现 body `arms` 为逆裁剪后的图像归一化投影、`world_arms` 为根相对米制坐标；元数据写入新来源。腕输出 `smplx_camera_axes/smplx_wrist_local`，显式提供经源码核验的轴变换，未知 frame 拒绝。
- [ ] 新增 source-aware 标定校验和来源分派，`camera_world` 旧规范继续用于旧回放；SMPL-X 使用独立来源配置。旧 neutral/标定直接传新输入时必须拒绝，不改旧 JSON 元数据。
- [ ] 固定体型后重算派生骨架，保留原始预测；用真实单帧 FK/投影 overlay 检查坐标，不把无视觉证据的数值测试称为现场验收。
- [ ] 相关测试通过后提交：`feat: adapt SMPL-X body and wrist observations`。

## Task 4: 追踪质量、遮挡与恢复

**Files:** Create `src/teleoperation/inputs/body_tracking.py`; Modify `src/teleoperation/retargeting/arm/wrist_orientation.py`; Test `tests/test_body_tracking.py`。

**Interfaces:** `BodyTracker.update(result: SmplerXResult | None, *, now_s: float) -> TrackedBody`；`TrackedBody` 包含 `body`、`wrists`、`state_by_side`、`reason_by_side`、`sample_id`、`age_s`、`fresh`。`reset(session_id: str) -> None`。

- [ ] 写时间驱动场景：静止噪声、单点跳变、长期错误、源冻结、无检测、单侧缺失、恢复、身份切换；断言 0.5 秒源超时生效，重复包不累计标定；恢复的 3 个样本必须新鲜且相互一致，确认后仍限速。
- [ ] 运行 `tests.test_body_tracking`，确认新增测试先失败。
- [ ] 实现校验与状态机。平移和旋转分别处理，旋转在 SO(3) 上插值；不逐元素平滑旋转矩阵或直接平均四元数。保持默认关闭长期运动预测。
- [ ] 与 `WristTracker` 分清职责：身体追踪管理源质量/新鲜度，腕层管理中性对齐；同一信号只有一个平滑与速率限制所有者，防止重复滤波累积延迟。源切换/worker 重启使相关中性失效。
- [ ] 通过相关测试并记录所有初始门槛来源；不得用较低更新率或永远保持冒充稳定性改善。
- [ ] 提交：`feat: track SMPL-X arm samples across occlusion and recovery`。

## Task 5: 相机组合输入与全部活动入口迁移

**Files:** Create `src/teleoperation/applications/camera_teleop_source.py`, `src/teleoperation/applications/camera_arm_pipeline.py`; Modify `src/teleoperation/inputs/mediapipe.py`, `src/teleoperation/simulation/realtime_teleop.py`, `src/teleoperation/applications/mediapipe_tron2a.py`, `src/teleoperation/applications/world_landmark_capture.py`, `src/teleoperation/applications/dual.py`, `src/teleoperation/cli.py`, `devtools/run_realtime_arm_windows.py`, `devtools/run_realtime_arm_elbow_follow_windows.py`; Test `tests/test_smplerx_camera_source.py`, `tests/test_arm_realtime_bridge.py`, `tests/test_mediapipe_tron2a.py`, `tests/test_cli_integration.py`。

**Interfaces:**
- MediaPipe Hands 适配器新增 `process_frame(frame: CameraFrame) -> RawHandFrame`，只检测调用方提供的图像，不自行读第二帧。
- `CameraTeleopSource.next_teleop_observation() -> RawTeleopObservation`；内部同一次采集送给 Hands 与 client，身体结果可较旧但保留自身 sample_id 和捕获时间。
- `CameraArmPipeline.update(body, wrists, *, sample_time_s: float, control_time_s: float)` / `step(dt_s: float)`：复用现有 bridge/driver/wrist 逻辑并供任务 7 回放调用。

- [ ] 写组合输入测试：同一捕获序号、Hands/身体异步时间不冒充同步；无手时臂仍动，无身体时手仍动；同一模型只推理一次供多机器人使用；`--no-arms` 不初始化 SMPLer-X。
- [ ] 运行本任务四个测试模块及新增 source 测试，先确认预期失败。
- [ ] 提取单相机所有权及 frame-processing 接口，保留手指旧 API 的结果；新 camera source 将腕来源送给臂消费者，禁止读取 Hands 腕用于机械臂兜底。
- [ ] 实时入口和 TRON2A 相机服务接入同一感知契约；TRON2A 仍按选定机器人资源校验，不让感知适配器依赖手指 checkpoint。公开旧服务命令可作为迁移兼容入口，但内部只走新身体输入。
- [ ] 增加统一 SMPLer-X 配置参数与 `--check`，启动错误给资源路径和原因；设备选择、模型缺失、CUDA OOM 都不启用旧 Pose。旧 Pose 专用 CLI 参数返回迁移错误，新帮助与 launcher 不再提示下载 pose.task。
- [ ] 增加新校准入口/来源支持，不能沿用原先硬编码 MediaPipe 的 TRON2A 摄像头标定检查。
- [ ] 运行相关回归，提交：`feat: drive camera arms from SMPLer-X observations`。

## Task 6: 记录与回放版本迁移

**Files:** Create `src/teleoperation/data/body_recording.py`; Modify `src/teleoperation/data/hand_body_recording.py`, `src/teleoperation/data/world_landmark_recording.py`, `src/teleoperation/data/arm_validation.py`, `src/teleoperation/simulation/realtime_teleop.py`; Test `tests/test_body_recording.py`, `tests/test_world_landmark_recording.py`, `tests/test_capture_branch_integration.py`。

**Interfaces:** `save_body_recording(path, samples, metadata) -> Path`、`load_body_recording(path) -> BodyRecording`；schema version 显式记录。旧 reader 继续沿旧来源分派，不覆盖输入文件。

- [ ] 写 round-trip 测试：原始参数、派生肩肘腕、腕四元数、源时间/完成时间、sample_id、来源、模型 hash、状态、固定体型和标定版本完整保留。丢失用 mask 与缺值，不伪造单位四元数。
- [ ] 写旧 NPZ/H5 和旧 neutral fixture 测试：仍能显式回放原协议；旧文件不能被导入为 SMPLer-X 成功证据；不明时间单位必须显式指定。
- [ ] 运行任务测试确认先失败，然后实现带来源的记录/读取。新 live dump 保留新来源，不再写死 MediaPipe metadata。
- [ ] 新回放分别支持已估计姿态与 RGB 重推理；明确只有 RGB 才能用于两个感知模型的对比。记录用于模型重推理的原始视频与逐帧时间映射，避免视频帧率替代实际时间。
- [ ] 通过相关回归，提交：`feat: version SMPL-X arm recordings and replay`。

## Task 7: 同路径验收与全链路资源测试

**Files:** Modify `src/teleoperation/applications/arm_validation.py`, `src/teleoperation/simulation/arm_validation_metrics.py`, `devtools/verify_smplerx_runtime.py`, `docs/smplerx-arm.md`; Test `tests/test_arm_validation.py`, `tests/test_arm_diagnostics.py`, `tests/test_smplerx_acceptance.py`。

**Interfaces:** 离线验收调用任务 5 的 `CameraArmPipeline`，显式写入 map mode、腕来源和控制配置。SMPLer-X performance report 包含可靠新样本 Hz、端到端 p50/p95、峰值显存、保持比例与源年龄。

- [ ] 写确定性回放测试：实时与回放收到相同姿态序列时，目标和命令在既有容差内一致；主机速度不改变结果；开启诊断不改变命令或物理步进。
- [ ] 运行任务相关测试确认先失败，修改验收工具，包含 posture + 新腕路线；旧 chain 指标独立命名，不冒充当前完整路线。
- [ ] 用同一 RGB 视频分别跑检查点基线和新模型；基线在独立进程或检查点导出目录运行，不能为比较将已删除的旧 Pose 运行路径重新并入主线。
- [ ] 视频覆盖静止 10 秒、左右臂单独动作、双臂、前伸、交叉、1–2 秒遮挡、出画、恢复、接近伸直；比较输入方向/腕旋转跳变、机器人抖动、关节变化、保持与覆盖率、恢复延迟。
- [ ] 在 RTX 4050 上先单模型，再叠加手指推理、单机器人、三机器人。warmup 后记录至少 300 个新输入；最后连续运行五分钟，检查资源释放、显存增长和积压。
- [ ] 以 >=10 Hz 可靠新身体结果、采集至结果 p95<=150 ms 为暂定工程目标；未达到就如实报告吞吐/精度取舍，不通过重复结果提升计数。不宣称 30 Hz 或真实空间精度。
- [ ] 共享行为变化完成后运行完整 unittest，提交：`test: validate SMPLer-X camera arm pipeline and runtime`。

## Task 8: 删除 MediaPipe Pose 活动代码并收尾

**Files:** Delete `src/teleoperation/inputs/pose_mediapipe.py`; Modify `src/teleoperation/inputs/mediapipe.py`, `tests/test_pose_mediapipe.py`（迁移/删除旧运行测试，保留独立旧数据 fixture）, `tests/test_mediapipe_input.py`, `tests/test_world_landmark_recording.py`, `tests/test_mediapipe_tron2a.py`, `src/teleoperation/simulation/arm_preview.py`, `README.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/smplerx-arm.md`。

**Interfaces:** 公共旧参数仅作显式迁移提示；旧数据读取不依赖被删除的 Pose parser；MediaPipe Hands 的采集、窗口和手指输出协议保留。

- [ ] 为活动 camera arm 路线添加约束测试：不创建/调用 PoseLandmarker，不在异常时调用旧驱臂代码，不消费 Hands 的腕观测。先运行确认删除前测试失败。
- [ ] 删除 Pose 初始化、detect、close 资源分支、手部位置兜底驱臂配置和无效文档提示；将仍需保留的旧数据 fixture 与 parser 解耦。
- [ ] 搜索 `PoseLandmarker|pose_mediapipe|pose_model_asset_path|pose_min_visibility|mediapipe_pose_world_meters`，逐处归类。生产实时调用必须清零；旧来源字符串只允许在显式兼容 reader、fixture 和历史文档中。
- [ ] 更新相机预览，注明“SMPL-X 重建/投影”，不再写 Pose 检测。报告包括模型版本、hash、推理后端、真实样本时间与拒绝原因。
- [ ] 执行完整 unittest、文档差异和 GBK 检查，更新真实验收结果与未完成项；提交：`refactor: remove MediaPipe Pose arm control path`。

## 公共验证命令（PowerShell）

以下设置只影响当前终端，不启动设备。按任务替换模块列表。

```powershell
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $python -m unittest tests.test_smplerx_geometry tests.test_body_tracking
& $python -m unittest discover -s tests
& $python devtools/check_gbk_safe.py --strict
git diff --check
```

预期：任务针对测试先因缺少对应功能失败，再通过；全套无新增失败。需要实际 GPU/资产的用例明确区分为集成验收，不能将 skip 等同于通过。

## 完成条件与回退

- [ ] 新相机手臂只消费 SMPLer-X 身体与腕观测；手指和 VR 回归通过。
- [ ] 活动 Pose 驱臂代码已删除；采集、标定、TRON2A 相机服务、实时、回放、诊断语义一致。
- [ ] 坐标测试、故障注入、完整单测及真实 GPU/现场验收各有独立结果，性能目标不足时明确列为未完成。
- [ ] 记录 pre/post commit 和各阶段 checkpoint。需要恢复时使用 `git restore --source=<tag> -- <paths>` 或从检查点建立回退分支，不 reset --hard，不 push。

本文件仅为实施计划。本次文档变更不安装依赖、不下载权重、不删除代码、不打开摄像头。
