# SMPLer-X 相机机械臂替换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划默认由当前会话顺序实施；不隐含授权创建子代理。

**Goal:** 用 SMPLer-X 提供相机机械臂的位置和腕朝向，删除旧 Pose 实时驱臂代码，保持手指控制和 VR 输入可用。

**Architecture:** 单相机采集后分别分发 Hands 与独立 SMPLer-X 工作进程，使用版本化人体观测与时序追踪接入现有机械臂控制。实时、标定、记录与离线验收共用新输入语义，最终删除旧 Pose 控制代码。

**Tech Stack:** Python 主工程 >=3.10；官方 SMPLer-X S32、PyTorch/CUDA、SMPL-X；OpenCV、NumPy/SciPy；PyBullet / 现有 Robotics Toolbox IK；unittest。推理进程依赖单独锁定，以实际兼容性测试为准。

**Spec:** [2026-10-30-smplerx-camera-arm-design.md](../specs/2026-10-30-smplerx-camera-arm-design.md)

本计划采用 ASD-STE100 的清晰技术写作原则。中文正文不声明符合其英文词典。每项任务给出文件、接口和通过条件。

本次文档修订基于 `5f7fc8481f81d04f9cb03a9982ed598462714bd2`。修改前检查点为 `checkpoint-20261030-1546-smplerx-plan-audit`。

实施状态（2026-10-30 更新）：任务 1 已接通真实 S32 后端、独立 Windows/CUDA 环境及单图 smoke；任务 2 已通过真实子进程图像往返；任务 3 已验证真实身体模型四种姿态的 FK。短视频、投影 overlay 和完整 GPU 验收仍未完成。任务 5 的相机源/入口已部分迁移，任务 6 的新身体记录尚未接通完整采集/回放，任务 8 的 parser 已删除但通用兜底实现和旧测试迁移尚未完成。全套测试尚未通过。准确状态、下载资产及未完成项见 [实施状态](../../smplerx-arm.md)。以下勾选框仍按完整验收条件记录，不把部分提交等同于任务完成。

## Global Constraints

- 相机机械臂的位置、骨段方向和腕朝向统一来自 SMPLer-X / SMPL-X。禁止用 MediaPipe Hands 的腕点或掌朝向补驱机械臂。
- 保持手指、VR 和新身体输入的 public input、window、result 接口；新增字段提供默认值。旧 MediaPipe 手臂录制接口不保留。
- Vision Pro / TransHandR 数值链路继续显式独立；不套用相机预处理或新相机标定。
- TRON2A 是可选机器人之一，新感知适配器不得硬编码机器人型号。
- 本次不更换重定向/IK 后端；机械臂向量优化独立实施和验收。
- 起点 S32 / 单人 / batch=1 / FP32；不假定 6 GB 显存或指定帧率已验证。
- 旧 Pose 不用于实时控制、文件回放或离线基线；旧 MediaPipe 手臂来源直接拒绝。
- 当前主环境是 `D:/Anaconda/envs/teleoperation/python.exe`。不依赖历史 launcher，不创建本机启动脚本。
- dex 已有独立第三方包环境和显式 Pinocchio bundle。SMPLer-X 使用第三个推理环境，不改写主环境或 dex 环境。
- 主环境 NumPy 配置偏差和 dex 重复 SciPy 元数据见 [环境核查](../../smplerx-environment-audit-2026-10-30.md)。不得把 pip check 通过写成环境已统一。
- 每个任务修改前执行 git status、提交已有明确改动、创建 annotated checkpoint；完成相关测试后提交，不 push。

## Review Focus

1. 工作进程迟到、冻结或退出：主控制继续保持，旧结果不能冒充新数据。任务 2、4、5 测试。
2. 镜像、裁剪或局部旋转组合错误：左右手和轴方向必须由合成变换证明。任务 3 测试。
3. Hands 丢失但身体可见 / 身体丢失但 Hands 可见：两条有效性独立。任务 5 测试。
4. 旧标定、旧 H5/NPZ：旧 MediaPipe 手臂来源直接拒绝，不进入新回放。任务 3、6、8 测试。
5. 新人进入、主体切换、重复帧：不自动改操作者身份，不累计错误标定。任务 2、4 测试。

## 保留和删除的边界

| 对象 | 实施结果 |
|---|---|
| 手指 | 保留 MediaPipe Hands、NN/dex 手指处理、手部窗口和关节结果接口 |
| 新相机手臂 | 肩肘腕和腕朝向全部来自 SMPLer-X / SMPL-X |
| VR | 保留 Vision Pro / TransHandR 的原数值路线，不输入 SMPLer-X |
| 旧 MediaPipe 手臂 H5/NPZ | 不保留 reader 或回放兼容；新程序拒绝旧来源 |
| 旧相机标定 | 不保留旧 MediaPipe 手臂标定分支；新人体输入重新标定 |
| 旧 Pose 运行路径 | 删除初始化、推理、模型检查和手部兜底驱臂代码 |
| dex 风格机械臂向量优化 | 另列后续计划；本次保留现有重定向与 IK，以便单独评估感知变化 |

“保留接口”不表示继续运行旧 Pose。新相机输入遇到错误时保持上一可靠目标，并报告错误。

## 文件职责

下列是拟新增或修改的路径，不表示已经存在的新实现。

| 路径 | 职责 |
|---|---|
| `contracts/body.py` | 新人体推理结果与帧身份协议 |
| `inputs/smplerx.py` | 主环境客户端、配置校验、结果生命周期 |
| `inputs/smplerx_geometry.py` | 消费工作进程 FK 结果，适配坐标与投影，不开相机、不加载人体模型 |
| `inputs/body_tracking.py` | 身体/腕有效性、保持和渐进恢复 |
| `applications/camera_teleop_source.py` | 单次捕获，分发 Hands 与身体推理，组合现有公共观测 |
| `applications/camera_arm_pipeline.py` | 实时/回放共同的机械臂控制步骤 |
| `external/smplerx_worker/worker.py` | 独立环境运行的轻量协议入口，不导入主工程 |
| `external/smplerx_worker/adapter.py` | 固定上游版本的初始化、单帧推理与模型参数提取 |
| `data/body_recording.py` | 新版本人体记录读写 |
| `devtools/verify_smplerx_runtime.py` | 原始模型 smoke、显存与耗时测试 |
| `docs/smplerx-arm.md` | 部署、迁移、坐标证据、验收记录 |
| `cli/smplerx_options.py` | 多个现有入口共享的 SMPLer-X 参数定义 |

以上 `contracts/inputs/applications/data` 路径均相对 `src/teleoperation/`。公共共享内存或网络服务不在第一版范围内；本地长度前缀子进程协议即可。

## 已有文件的修改和删除位置

行号对应本计划的代码基线。实施时按函数名定位，避免行号变化造成误删。文件路径相对仓库根目录。

| 文件 | 位置 | 操作 |
|---|---|---|
| `src/teleoperation/inputs/pose_mediapipe.py` | 全文件 | 最终删除 Pose 结果解析器 |
| `src/teleoperation/inputs/mediapipe.py` | 23；`MediaPipeCameraInput.__init__`，166-232 | 删除 Pose import、路径校验、options 和 landmarker 初始化。保留 Hands 初始化 |
| 同上 | `next_frame`，241-285 | 拆出单次捕获和 Hands 帧处理；删除 `include_pose` 检测分支。显式传入旧 Pose 参数时返回迁移错误 |
| 同上 | `next_teleop_observation`，315；`release`，365 | 移除旧 Pose 组合实现及其关闭分支。原组合方法保留迁移错误入口；保留 Hands 采集和释放 |
| `src/teleoperation/simulation/realtime_teleop.py` | `CameraFrameSource`，205；`run`，906 | 使用新组合源；移除 Pose 专用配置、缺文件警告和静默降级 |
| 同上 | `build_parser`，381；`run`，1086、1119、1393 | 旧 Pose 参数只报迁移错误；停止启用 Hands 驱臂兜底；结果报告记录新模型和新采样时间 |
| 同上 | `ReplayFrameSource`，279；`_record_frame`，789；`_save_dump`，813 | 删除旧 MediaPipe 手臂回放分支；新录制只写 SMPL-X 来源 |
| `src/teleoperation/simulation/arm_realtime.py` | `_operator_state`，1187；`_target_from_state`，1309；`_fallback_target`，1445 | 删除活动相机从 Hands 生成机械臂目标的分支。旧数据需要该基线时，从检查点运行；保持 body 几何映射与 IK 参数 |
| `src/teleoperation/applications/mediapipe_tron2a.py` | `_configuration`，38；`run`，100；`_runtime_configuration`，293；`run_camera`，371 | 改接新组合源。删除 Pose 文件检查、可见性阈值及 418 行附近的 MediaPipe 来源/手身同时戳假设。保留命令名称，但命令只启动新管线 |
| `src/teleoperation/applications/world_landmark_capture.py` | 18；`run`，141-234；`_draw_preview` | 删除 Pose parser import、自动下载和初始化；记录新人体样本及其独立时间，预览使用新投影 |
| `src/teleoperation/contracts/observations.py` | `UpperBodyObservation`；`RawTeleopObservation` | 分别追加可选 metadata 和 wrists 字段，默认值保留旧构造方式 |
| `src/teleoperation/retargeting/arm/calibration.py` | `NeutralPoseCollector.add`，201；`calibration`，263 | 只接受 SMPL-X 来源、时间和质量字段；删除旧 MediaPipe 标定分支，不伪造 pose_visibility |
| `src/teleoperation/retargeting/arm/retargeter.py` | `__init__`，13；`update`，38 | 只接受 SMPL-X 身体来源；旧 MediaPipe 手臂来源抛出迁移错误 |
| `src/teleoperation/retargeting/arm/wrist_orientation.py` | `WristTracker.update`，35；来源分派，67 | 加入显式 SMPL-X 轴变换；保留 VR 分支的现有数值与默认参数 |
| `src/teleoperation/applications/dual.py` | `Tron2ADualArmController`；`DualTeleoperator` | 新相机只消费组合观测的 SMPL-X wrists，旧手部 wrists 保留原来源 |
| `src/teleoperation/cli/schemas.py` | `hand_record_world`，152；`dual_calibrate_camera`，210；`dual_camera`，247 | 接入共享 SMPLer-X 参数；移除活动 Pose 默认路径和可见性参数 |
| `src/teleoperation/cli/registry.py` | camera 与 calibrate-camera 注册项，10-11 | 保持已有命令名和 handler 可解析；模型替换在 handler 内完成 |
| `src/teleoperation/data/hand_body_recording.py` | `write_hand_body_recording`，70；来源属性，130-131 | 删除旧 MediaPipe 手臂来源属性和旧手臂读取分支；手指记录字段单独保留 |
| `src/teleoperation/data/world_landmark_recording.py` | `write_world_landmark_recording`；`read_world_landmark_recording` | 删除旧身体回放 reader；新身体记录只使用 `body_recording.py` |
| `src/teleoperation/data/arm_validation.py` | `load_validation_recording`；`validate_recording` | 支持新版本身体记录和显式时间单位 |
| `src/teleoperation/applications/arm_validation.py` | `run_case`，27 | 与实时入口共用新控制管线，报告实际 map mode 和腕来源 |
| `src/teleoperation/simulation/arm_validation_metrics.py` | 验收指标计算 | 增加源年龄、有效更新率、保持比例和恢复延迟 |
| `src/teleoperation/simulation/arm_preview.py` | 身体点显示和来源文字 | 显示 SMPL-X 重建/投影，不再显示 Pose 检测说明 |
| `tests/test_pose_mediapipe.py` | 全文件 | 删除已移除解析器的专用测试；先把其他测试引用的 fixture 移入独立 fixture 模块 |
| `tests/test_mediapipe_input.py` | Pose mock，210-211；Pose 组合测试，277-385 | 删除旧 Pose 成功路径测试；替换为迁移错误与 Hands 回归测试 |
| `tests/test_mediapipe_tron2a.py`、`tests/test_world_landmark_recording.py` | Pose 资产参数、mock 和来源断言 | 改为新源；删除旧 MediaPipe 手臂文件读取断言 |
| `README.md`、`docs/architecture.md`、`docs/contracts.md` | 相机手臂的数据流、模型和记录说明 | 更新为 SMPLer-X，明确 VR 与手指边界 |

不删除 MediaPipe Hands 模块或 VR 模块。删除旧 MediaPipe 手臂来源字段、reader、回放分支、标定分支和测试。磁盘上的历史文件不由迁移脚本自动删除，但新程序不读取这些文件。Pose `.task` 文件不再被新手臂路径加载；本计划不自动清理用户下载的资产。

不修改历史 `devtools/run_realtime*_windows.py` 来推断或接管本机运行方式。实施后在文档中标明这些旧 launcher 不适用于新手臂入口。主环境需求偏差单独处理，不能借模型接入执行未经验证的批量升级。

TRON2A 的 `loadable=False` 属于通用机器人加载能力问题。本计划统一其感知接口，但不通过改布尔值宣称该问题已解决。后续将其作为可选机器人接入时，需另测实际 URDF、关节顺序、末端、碰撞和 FK。

## Task 1: 推理环境与真实模型可运行性

**Files:** Create `external/smplerx_worker/README.md`, `external/smplerx_worker/environment.lock.yml`, `external/smplerx_worker/adapter.py`, `devtools/verify_smplerx_runtime.py`, `docs/smplerx-arm.md`; Test `tests/test_smplerx_runtime.py`。

**Interfaces:** 提供 `verify_smplerx_runtime.py --check` 和 `--input-video PATH --report PATH`。参数包括 `--worker-python`、`--smplerx-root`、`--checkpoint`、`--body-model-dir`、`--model s32`；输出机器可读依赖、资产哈希和性能报告。

- [ ] 在 `test_missing_assets_fail_check` 中断言检查退出码非零且错误含缺失路径。在 `test_runtime_checker_import_is_lightweight` 中用干净子进程导入 `devtools.verify_smplerx_runtime`，断言不加载 torch/OpenMMLab。测试缺资产不能被标为通过。
- [ ] 使用下方公共测试命令运行 `tests.test_smplerx_runtime`，确认因功能缺失失败。
- [ ] 固定官方源码版本，核对 S32 配置、许可证与所需资产。使用兼容 RTX 4050 的独立环境，安装成功后锁定实际版本；若转 WSL2，记录路径映射与 CUDA 检查结果。
- [ ] 用 teleoperation 和 dex 各自的解释器复核包导入路径。记录 NumPy 配置偏差和两份 SciPy 元数据；保持这两个运行环境不变。
- [ ] 完成单帧和短视频 warmup 推理，记录检测、裁剪、前向、SMPL-X 派生耗时、峰值 allocated/reserved 显存及输入输出 shape。不使用训练数据集下载作为推理前置。
- [ ] 运行单测及真实 smoke；没有合法取得的模型资源则标记该项未完成，禁止用 mock 输出宣称模型可运行。
- [ ] 提交：`build: isolate and verify SMPLer-X S32 runtime`。

## Task 2: 带时间与身份的推理协议

**Files:** Create `src/teleoperation/contracts/body.py`, `src/teleoperation/inputs/smplerx.py`, `external/smplerx_worker/worker.py`; Modify `external/smplerx_worker/adapter.py`; Test `tests/test_smplerx_protocol.py`。

**Interfaces:**
- `BodyFrameId(session_id: str, sequence: int, capture_time_s: float)`，时间为同一主进程单调秒。
- `CameraSample(rgb: np.ndarray, frame_id: BodyFrameId, unix_time_ms: int)`；图像为原始 RGB，单调捕获时间在读到图像后立即记录。Unix 毫秒仅供既有 Hands VIDEO 时间接口使用。
- `SmplerXResult`：包含 frame_id、推理完成时间、model_id/hash、root/body/hand 旋转参数、shape、根相对关节位置、全局关节旋转、检测框与逆裁剪所需信息、valid/reason；数组带显式 dtype/shape，不能使用不可信 pickle。
- `SmplerXClient.submit(sample: CameraSample) -> None`。
- `SmplerXClient.poll() -> SmplerXResult | None`，`close() -> None`；不得修改捕获时间。

- [ ] 写协议测试：分片读取、非法长度/shape/version、乱序、重复、迟到结果、stderr 日志和 EOF；例如 `assert accepted.sequence > previous.sequence`，畸形包拒绝且不下发机器人命令。
- [ ] 在 `test_latest_pending_frame_replaces_old_one` 中阻塞首帧，提交序号 2、3，断言工作进程只接收 1、3。在 `test_old_session_result_is_rejected` 中断言重启前的结果不再被 poll 返回。
- [ ] 运行 `tests.test_smplerx_protocol`，确认新增测试先失败。
- [ ] 实现消息格式与最大长度检查、单活动/单等待帧调度；`submit` 不被重推理阻塞，等待帧被最新帧替换。工作进程只推理指定单人，身份歧义返回失效。
- [ ] 用 fake 子进程验证延迟、崩溃、关闭回收与重启会话；新会话旧包拒绝，worker stdout 不混入上游日志。
- [ ] 接真实 S32 做小批视频往返，比较直接调用与协议输出的数值一致性；记录本地传输开销。
- [ ] 提交：`feat: add bounded SMPLer-X worker protocol`。

## Task 3: SMPL-X 坐标、骨架与腕朝向适配

**Files:** Create `src/teleoperation/inputs/smplerx_geometry.py`; Modify `external/smplerx_worker/adapter.py`, `src/teleoperation/contracts/observations.py`, `src/teleoperation/retargeting/arm/wrist_orientation.py`, `src/teleoperation/retargeting/arm/calibration.py`, `src/teleoperation/retargeting/arm/retargeter.py`; Test `tests/test_smplerx_geometry.py`, `tests/test_wrist_observations.py`。

**Interfaces:** `smplerx_observations(result: SmplerXResult) -> tuple[UpperBodyObservation, dict[str, WristObservation]]`；`UpperBodyObservation` 新增可选 `metadata`，默认空映射。`RawTeleopObservation` 新增 `wrists: dict[str, WristObservation] | None = None`。保留旧构造方式。

- [ ] 写合成骨架测试：零姿态、根旋转 90 度、肩/肘局部旋转组合、左右侧独立、q/-q 等价、逆裁剪投影、det(R)=+1、非法输入。断言 FK 腕位置与父子旋转链一致，不能直接取 body_pose 中某三个数当全局腕朝向。
- [ ] 在 `test_neutral_fk_matches_worker` 中比较独立参考 FK 与 adapter 输出，位置容差为 `1e-5 m`，旋转矩阵容差为 `1e-5`。在 `test_legacy_calibration_rejects_smplx_source` 中断言旧标定与新来源组合抛出 ValueError。
- [ ] 运行 `tests.test_smplerx_geometry tests.test_wrist_observations`，确认新增断言先失败。
- [ ] 从固定版本 SMPL-X 模型关节名称/父节点关系建立映射，在工作进程 adapter 中用模型 FK 得到同一根参考的肩肘腕与全局旋转；主工程 geometry 只用 NumPy/SciPy 消费派生结果，不能引入第二套 SMPL-X/PyTorch 依赖。不要硬套 MediaPipe 点号或人体模型内部重排后的数组。
- [ ] 实现 body `arms` 为逆裁剪后的图像归一化投影、`world_arms` 为根相对米制坐标；元数据写入新来源。腕输出 `smplx_camera_axes/smplx_wrist_local`，显式提供经源码核验的轴变换，未知 frame 拒绝。
- [ ] 新增 source-aware 标定校验和来源分派；VR 使用原有来源配置，SMPL-X 使用独立来源配置。旧 MediaPipe neutral/标定直接拒绝，不改旧 JSON 元数据。
- [ ] 固定体型后重算派生骨架，保留原始预测；用真实单帧 FK/投影 overlay 检查坐标，不把无视觉证据的数值测试称为现场验收。
- [ ] 相关测试通过后提交：`feat: adapt SMPL-X body and wrist observations`。

## Task 4: 追踪质量、遮挡与恢复

**Files:** Create `src/teleoperation/inputs/body_tracking.py`; Modify `src/teleoperation/retargeting/arm/wrist_orientation.py`; Test `tests/test_body_tracking.py`。

**Interfaces:** `BodyTracker.update(result: SmplerXResult | None, *, now_s: float) -> TrackedBody`；`TrackedBody` 包含 `body`、`wrists`、`state_by_side`、`reason_by_side`、`sample_id`、`age_s`、`fresh`。`reset(session_id: str) -> None`。

- [ ] 写时间驱动场景：静止噪声、单点跳变、长期错误、源冻结、无检测、单侧缺失、恢复、身份切换；断言 0.5 秒源超时生效，重复包不累计标定；恢复的 3 个样本必须新鲜且相互一致，确认后仍限速。
- [ ] 在 `test_expired_source_stays_held` 中输入年龄 `0.501 s` 的结果，断言 `fresh` 为 False 且状态为 `lost_hold`。在 `test_recovery_requires_three_distinct_samples` 中重复第二个 sample_id，断言尚未恢复；收到第三个一致新样本后才取得恢复资格。
- [ ] 运行 `tests.test_body_tracking`，确认新增测试先失败。
- [ ] 实现校验与状态机。平移和旋转分别处理，旋转在 SO(3) 上插值；不逐元素平滑旋转矩阵或直接平均四元数。保持默认关闭长期运动预测。
- [ ] 与 `WristTracker` 分清职责：身体追踪管理源质量/新鲜度，腕层管理中性对齐；同一信号只有一个平滑与速率限制所有者，防止重复滤波累积延迟。源切换/worker 重启使相关中性失效。
- [ ] 通过相关测试并记录所有初始门槛来源；不得用较低更新率或永远保持冒充稳定性改善。
- [ ] 提交：`feat: track SMPL-X arm samples across occlusion and recovery`。

## Task 5: 相机组合输入与全部活动入口迁移

**Files:** Create `src/teleoperation/applications/camera_teleop_source.py`, `src/teleoperation/applications/camera_arm_pipeline.py`, `src/teleoperation/cli/smplerx_options.py`; Modify `src/teleoperation/inputs/mediapipe.py`, `src/teleoperation/simulation/realtime_teleop.py`, `src/teleoperation/applications/mediapipe_tron2a.py`, `src/teleoperation/applications/world_landmark_capture.py`, `src/teleoperation/applications/dual.py`, `src/teleoperation/cli/schemas.py`, `src/teleoperation/cli/registry.py`; Test `tests/test_smplerx_camera_source.py`, `tests/test_arm_realtime_bridge.py`, `tests/test_mediapipe_tron2a.py`, `tests/test_cli_integration.py`。

**Interfaces:**
- MediaPipe Hands 适配器新增 `capture_sample() -> CameraSample` 和 `process_sample(sample: CameraSample) -> RawHandFrame`。前者只捕获一次，后者只检测指定图像。旧 Hands API 委托这两步。
- `CameraTeleopSource.next_teleop_observation() -> RawTeleopObservation`；内部同一次采集送给 Hands 与 client，身体结果可较旧但保留自身 sample_id 和捕获时间。
- `CameraArmPipeline.update(tracked: TrackedBody, *, control_time_s: float) -> None` / `step(dt_s: float) -> None`：复用现有 bridge/driver/wrist 逻辑并供任务 7 回放调用。缺新样本时不重复累计标定。
- `add_smplerx_arguments(parser: argparse.ArgumentParser) -> None`：提供 `--smplerx-worker-python`、`--smplerx-root`、`--smplerx-checkpoint`、`--smplx-model-dir`。模型固定 S32。主环境直接调用现有模块入口，不依赖 launcher。

- [ ] 写组合输入测试：同一捕获序号、Hands/身体异步时间不冒充同步；无手时臂仍动，无身体时手仍动；同一模型只推理一次供多机器人使用；`--no-arms` 不初始化 SMPLer-X。
- [ ] 在 `test_body_wrists_do_not_overwrite_hand_wrists` 中断言组合 `wrists` 来源为 SMPL-X，`hands.wrists` 仍为原来源。在 `test_no_arms_does_not_start_worker` 中断言 worker 启动次数为 0。
- [ ] 运行本任务四个测试模块及新增 source 测试，先确认预期失败。
- [ ] 提取单相机所有权及 frame-processing 接口，保留手指旧 API 的结果。新 camera source 将 SMPL-X 腕写入 `RawTeleopObservation.wrists`。保持 `RawHandFrame.wrists` 的原来源，禁止机械臂读取该字段兜底。
- [ ] 实时入口和 TRON2A 相机服务接入同一感知契约；TRON2A 仍按选定机器人资源校验，不让感知适配器依赖手指 checkpoint。保留命令名称只为减少调用方改动，命令内部只走新身体输入，不提供旧数据兼容模式。
- [ ] 增加共享 SMPLer-X 配置参数和独立 runtime 检查命令。设备选择、模型缺失、CUDA OOM 均报告原因。旧 Pose 参数返回迁移错误，活动入口帮助不再提示下载 pose.task。
- [ ] 增加新校准入口/来源支持，不能沿用原先硬编码 MediaPipe 的 TRON2A 摄像头标定检查。
- [ ] 运行相关回归，提交：`feat: drive camera arms from SMPLer-X observations`。

## Task 6: 记录与回放版本迁移

**Files:** Create `src/teleoperation/data/body_recording.py`; Modify `src/teleoperation/data/hand_body_recording.py`, `src/teleoperation/data/world_landmark_recording.py`, `src/teleoperation/data/arm_validation.py`, `src/teleoperation/simulation/realtime_teleop.py`; Test `tests/test_body_recording.py`, `tests/test_world_landmark_recording.py`, `tests/test_capture_branch_integration.py`。

**Interfaces:** `save_body_recording(path, samples, metadata) -> Path`、`load_body_recording(path) -> BodyRecording`；schema version 显式记录。reader 只接受 SMPL-X 身体来源，遇到 `mediapipe_pose_*` 字段时抛出迁移错误。

- [ ] 写 round-trip 测试：原始参数、派生肩肘腕、腕四元数、源时间/完成时间、sample_id、来源、模型 hash、状态、固定体型和标定版本完整保留。丢失用 mask 与缺值，不伪造单位四元数。
- [ ] 写旧 NPZ/H5 拒绝测试：发现 `mediapipe_pose_*` 身体字段时返回迁移错误；不明时间单位和缺少 SMPL-X 模型 hash 时拒绝。
- [ ] 在 `test_legacy_arm_recording_is_rejected` 中断言旧 MediaPipe 手臂文件不会返回 `BodyRecording`。在 `test_roundtrip_preserves_capture_time` 中断言新记录的 sample_id、单调捕获时间和模型 hash 完全一致。
- [ ] 运行任务测试确认先失败，然后实现只接受 SMPL-X 来源的记录/读取。新 live dump 不写入 MediaPipe 来源字段。
- [ ] 新回放只支持 SMPL-X 已估计姿态或 RGB 重新推理。记录用于模型推理的原始视频与逐帧时间映射，避免视频帧率替代实际时间。
- [ ] 通过相关回归，提交：`feat: version SMPL-X arm recordings and replay`。

## Task 7: 同路径验收与全链路资源测试

**Files:** Modify `src/teleoperation/applications/arm_validation.py`, `src/teleoperation/simulation/arm_validation_metrics.py`, `devtools/verify_smplerx_runtime.py`, `docs/smplerx-arm.md`; Test `tests/test_arm_validation.py`, `tests/test_arm_diagnostics.py`, `tests/test_smplerx_acceptance.py`。

**Interfaces:** 离线验收调用任务 5 的 `CameraArmPipeline`，显式写入 map mode、腕来源和控制配置。SMPLer-X performance report 包含可靠新样本 Hz、端到端 p50/p95、峰值显存、保持比例与源年龄。

- [ ] 写确定性回放测试：实时与回放收到相同姿态序列时，目标和命令在既有容差内一致；主机速度不改变结果；开启诊断不改变命令或物理步进。
- [ ] 在 `test_live_and_replay_share_commands` 中向两条路径注入相同的姿态和时间，断言命令差值不超过 `1e-6 rad`。在 `test_repeated_results_do_not_inflate_hz` 中重复 poll 同一结果，断言有效新结果计数为 1。
- [ ] 运行任务相关测试确认先失败，修改验收工具，包含 posture + 新腕路线；旧 chain 指标独立命名，不冒充当前完整路线。
- [ ] 用同一 RGB 视频分别跑检查点基线和新模型；基线在独立进程或检查点导出目录运行，不能为比较将已删除的旧 Pose 运行路径重新并入主线。
- [ ] 视频覆盖静止 10 秒、左右臂单独动作、双臂、前伸、交叉、1–2 秒遮挡、出画、恢复、接近伸直；比较输入方向/腕旋转跳变、机器人抖动、关节变化、保持与覆盖率、恢复延迟。
- [ ] 在 RTX 4050 上先单模型，再叠加手指推理、单机器人、三机器人。warmup 后记录至少 300 个新输入；最后连续运行五分钟，检查资源释放、显存增长和积压。
- [ ] 以 >=10 Hz 可靠新身体结果、采集至结果 p95<=150 ms 为暂定工程目标；未达到就如实报告吞吐/精度取舍，不通过重复结果提升计数。不宣称 30 Hz 或真实空间精度。
- [ ] 共享行为变化完成后运行完整 unittest，提交：`test: validate SMPLer-X camera arm pipeline and runtime`。

## Task 8: 删除 MediaPipe Pose 活动代码并收尾

**Files:** Delete `src/teleoperation/inputs/pose_mediapipe.py`, `tests/test_pose_mediapipe.py`（先迁移仍被引用的 Hands fixture）；Modify `src/teleoperation/inputs/mediapipe.py`, `src/teleoperation/simulation/arm_realtime.py`, `src/teleoperation/data/hand_body_recording.py`, `src/teleoperation/data/world_landmark_recording.py`, `tests/test_mediapipe_input.py`, `tests/test_world_landmark_recording.py`, `tests/test_mediapipe_tron2a.py`, `src/teleoperation/simulation/arm_preview.py`, `README.md`, `docs/architecture.md`, `docs/contracts.md`, `docs/smplerx-arm.md`。

**Interfaces:** 公共旧参数仅作显式迁移提示；旧 MediaPipe 手臂数据读取接口删除；MediaPipe Hands 的采集、窗口和手指输出协议保留。

- [ ] 为活动 camera arm 路线添加约束测试：不创建/调用 PoseLandmarker，不在异常时调用旧驱臂代码，不消费 Hands 的腕观测。先运行确认删除前测试失败。
- [ ] 在 `test_camera_arm_never_uses_pose_landmarker` 中将 PoseLandmarker 的创建设为立即失败，断言新 source 完成一轮 fake SMPL-X 输入。在 `test_legacy_pose_option_returns_migration_error` 中断言旧参数返回清晰错误且不开相机。
- [ ] 按上方删除表移除 Pose 初始化、detect、close 资源分支、活动 Hands 驱臂分支、旧手臂录制字段和旧手臂 reader。仍被引用的 Hands fixture 移入 `tests/fixtures/hands.py`，不导入被删除的 parser。先用 `rg` 检查引用，再删除旧测试文件。
- [ ] 搜索 `PoseLandmarker|pose_mediapipe|pose_model_asset_path|pose_min_visibility|mediapipe_pose_world_meters|mediapipe_pose_normalized`，逐处归类。生产代码、录制 reader、回放、标定和测试中的旧手臂来源必须清零；只允许在拒绝旧数据的测试输入中临时出现。
- [ ] 删除旧 MediaPipe 手臂来源常量、schema 字段、标定字段和回放分支。保留的测试只构造一个临时旧字段，验证程序拒绝它；该字段不进入任何公共接口、记录格式或常量表。
- [ ] 删除旧手臂 reader 的测试和旧来源断言。运行 `rg` 确认生产代码中没有旧来源字符串、旧 reader 名称或旧标定分派。
- [ ] 更新相机预览，注明“SMPL-X 重建/投影”，不再写 Pose 检测。报告包括模型版本、hash、推理后端、真实样本时间与拒绝原因。
- [ ] 执行完整 unittest、文档差异和 GBK 检查，更新真实验收结果与未完成项；提交：`refactor: remove MediaPipe Pose arm control path`。

## 每项任务的 Git 和验证要求

实施前运行 `git status --short --branch`。脏工作区按 AGENTS.md 先保存明确的当前状态。创建该任务的 annotated checkpoint，然后编写测试和修改代码。

每项任务采用“新增测试失败、实现、测试通过、提交”的顺序。没有真实 GPU 或模型资源时，明确保留该项未完成。共享接口修改后运行完整 unittest。VR 回归使用 `tests.test_transhandr_vr`，手指回归使用 `tests.test_world_hand_pipeline` 和 `tests.test_dex_native`，分别使用所需解释器运行。

任务 7 比较相同视频的指标。只有可靠新结果计入 Hz。0.5 秒过期和 3 个一致样本恢复是初始工程设置，不是 SMPLer-X 提供的置信结论。保留 TRON2A 专用中性采集的 60 样本默认值；通用 bridge 的 30 样本和腕的 10 样本约束不得通过重复缓存满足。

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
