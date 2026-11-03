# 底层旧分支删除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除旧臂输入、旧相机 normalized 重定向和侧专用 NN 分支，使当前原生接口成为唯一相机运行路线。

**Architecture:** 按臂映射、相机坐标、模型权重三个独立子项目实施，各自可测试、提交和回滚，建议顺序执行。保留公共观测/窗口/结果结构，在边界拒绝退休参数和文件；VR 上游数值隔离保留。第四项只清理失去消费者的旧工具及相关说明，不改场景或推理架构。

**Tech Stack:** Python >=3.10、unittest、NumPy、PyTorch、h5py、PyBullet、Robotics Toolbox、既有独立 dex/SMPLer-X 环境、Windows PowerShell、Git。

**Spec:** `docs/superpowers/specs/2026-11-03-remove-legacy-branches.md`；执行者先读该范围说明及根目录 `AGENTS.md`。

## Global Constraints

- Python >=3.10，Windows PowerShell，使用现有 unittest；不新增运行依赖。
- 保留 CameraFrame、RawHandFrame、UpperBodyObservation、RawTeleopObservation、WristObservation、CanonicalHandFrame、HandWindow、HandJointResult、HandJointCommand 的当前结构。
- 保留手部原生命名关节、HandSpec 哈希、有效限位、速度约束、mimic 展开和接受命令回放语义。
- 保留当前相机 world H5、shared NN checkpoint、新版手结果 H5 和 SMPLer-X 身体 H5 的实际数值与有效性语义。
- 保留 MediaPipe normalized 图像点和 SMPL-X 图像投影作为预览/记录数据；禁止将其作为米制重定向输入。
- 保留上游 TransHandR VR 轴变换、三帧、固定缩放、原始18D输出及边界17D适配；禁止应用相机预处理或重写权重元数据。
- 删除范围不含现存数据文件、模型权重、机器人资产、依赖环境和上游 TransHandR 工程。
- 每个实现批次在改动前检查 Git 状态、保存 dirty intended state、创建 annotated checkpoint；完成验证后单独提交，不推送。

---

## 文件边界与基线证据

计划依据提交 `86cabc4ea0aa8cbbadbf098223b1ff82bceb9515`；文档修改前检查点 `checkpoint-20261103-2357-plan-legacy-branches`。实现开始时必须重新确认 HEAD，不能复用文档检查点代替后续代码检查点。

|批次|修改/删除对象|单一职责与依赖|
|---|---|---|
|A|`retargeting/arm/retargeter.py`、`applications/dual.py`|只支持SMPL-X肩相对映射，控制器无条件检查标定与工作空间|
|B|`contracts/coordinates.py`、`retargeting/hand/{coordinates,processing,alignment,geometric}.py`|相机world预处理与几何诊断边界；VR共享常量/参照FK不能误删|
|B|`applications/{hand_processing,hand_realtime,hand_align}.py`、`data/{hand_h5,hand_recording}.py`|组合流程、严格H5读取、world-only原始writer|
|B|`devtools/verify_geometric_hand.py`、`applications/diagnostics/diagnose_hand_coordinates.py`|诊断工具使用新坐标，不继续接受旧相机训练契约|
|C|`retargeting/hand/{checkpoint,interface}.py`、`paths.py`|shared权重身份、左右适配和训练目录|
|D|删除 `retargeting/hand/{diagnostics,tracking}.py`|无生产消费者的旧18D审计/身份工具，移除悬空import|
|A–D|对应tests、README.md、docs/current-interfaces.md|迁移行为证据与更新声明，不清空失败测试|

以上代码路径均以 `src/teleoperation/` 为前缀（明确写为 devtools/tests/docs 的除外）。下文源行号只定位当前基线，编辑后按符号定位。

保持不动：输入契约结构、手模型资产、VR preprocessing/model 数值、命令策略、任务场景、独立依赖版本。`normalized` 同时用于图像投影、掌长归一化和旧训练模式，不能全仓字符串删除。

## 实现通用命令与提交协议

每个新终端先执行，所有测试命令都要求检查 `$LASTEXITCODE`，失败即停止该批次的提交：

```powershell
Set-Location D:/2026/code/teleoperation-system-retargeting
$planPython = 'D:/Anaconda/envs/teleoperation/python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

每项任务开始前执行以下操作，把 `<description>` 替换为该任务给出的具体描述；禁止把尖括号文本当命令执行：

```powershell
git status --short --branch
git rev-parse HEAD
# 若dirty，先逐文件确认并提交 intended state；不能把不明改动打包。
$planStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
```

随后分别使用任务指定的 `git tag -a`，将 `$planTag` 与 pre-change hash 写入工作记录。checkpoint后的一项测试→实现→验证→提交是一个完整变更单元；下一项新变更重新创建checkpoint。若Git所有权检查阻止读取，只对该命令增加 `-c safe.directory=D:/2026/code/teleoperation-system-retargeting`；Git写入如被沙箱限制，按工具授权流程处理，不修改全局Git配置绕过。

### Task 1 / 子项目 A：移除 TRON2A legacy 臂输入

**Files:**
- Modify: `src/teleoperation/retargeting/arm/retargeter.py:13`，`ArmRetargeter.__init__/update`。
- Modify: `src/teleoperation/applications/dual.py:21`，`Tron2ADualArmController.__init__/update_side`。
- Test: `tests/test_smplerx_retargeter.py`、`tests/test_smplerx_calibration.py`、`tests/test_mediapipe_tron2a.py`。
- Modify: `docs/current-interfaces.md` 的TRON2A输入声明。

**Interfaces:**
- Consumes: `UpperBodyObservation` 的图像 `arms` 与米制 `world_arms`、`ArmCalibration.metadata`。
- Produces: `ArmRetargeter(calibration, *, input_mode='smplerx', orientation_mode=None)`；`update(observation, side) -> ArmTargetPose`。
- Produces: `Tron2ADualArmController(calibration_path, urdf_path, *, input_mode='smplerx', orientation_mode=None)`；只接受smplerx，48D输出结构不变。

- [ ] **Step 1：创建本批检查点。**

```powershell
$planTag = "checkpoint-$planStamp-remove-legacy-arm"
git tag -a $planTag -m 'Before removing legacy TRON2A arm mapping'
```

- [ ] **Step 2：在 SmplerXRetargeterTests 添加默认模式与非零肩位置的行为测试。** 文件已有calibration()、np、UpperBodyObservation、ArmRetargeter导入。

```python
def test_default_uses_metric_wrist_relative_to_own_shoulder(self):
    mapper = ArmRetargeter(calibration())
    points = np.array([[.3, .2, .1], [.3, .4, .1], [.4, .5, .15]])
    raw = UpperBodyObservation(
        {'left': points * 10, 'right': None},
        {'left': True, 'right': False}, 1.,
        'smplx_image_normalized', 'normalized',
        world_arms={'left': points, 'right': None})
    actual = mapper.update(raw, 'left')
    self.assertTrue(actual.valid)
    np.testing.assert_allclose(actual.pose[:3], points[2] - points[0], atol=1e-7)

def test_retired_arm_modes_are_rejected(self):
    for mode in ('legacy', 'camera_world'):
        with self.subTest(mode=mode), self.assertRaises(ValueError):
            ArmRetargeter(calibration(), input_mode=mode)
```

- [ ] **Step 3：运行RED，确认失败原因是旧默认映射/legacy仍接受，而不是依赖缺失。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_smplerx_retargeter -v
```

- [ ] **Step 4：简化构造和位置计算，删除全部legacy条件体。** 保留现有误差检查/四元数连续性，不重写数值算法。关键替换内容：

```python
# ArmRetargeter.__init__：默认input_mode='smplerx'
if input_mode != 'smplerx':
    raise ValueError('Arm input requires SMPLer-X root-relative metre observations')
self.input_mode = 'smplerx'
self.orientation_mode = orientation_mode or 'fixed'
# 原if input_mode == 'smplerx'里的metadata检查无条件执行。

# update：图像投影声明检查保留，位置只取world_arms。
if observation.coordinate_frame != 'smplx_image_normalized' or observation.units != 'normalized':
    raise ValueError('Expected camera body projection and root-relative world observation')
keypoints = None if observation.world_arms is None else observation.world_arms[side]
# 原valid/shape/finite检查之后：
wrist = points[2] - points[0]
orientation = (_forearm_rotation(points, min_bend_sine=0.05)
               if self.orientation_mode == 'forearm' else None)
# 删除legacy专用退化拒绝、workspace clip；保留SMPL-X近平直朝向保持。
```

`Tron2ADualArmController` 同样默认smplerx。保留先构造ArmRetargeter再分配solvers的顺序；标定安全参考FK核验无条件执行；`workspace_rejected` 改成 `target.valid and not in_workspace`。前臂平面符号连续性从if内提升为无条件的forearm有效朝向处理。

- [ ] **Step 5：将 test_legacy_absolute_wrist_behavior_is_unchanged 替换为旧输入拒绝测试，保留近平直、工作空间、真实FK、安全控制及48D测试。** 不改现有显式smplerx调用或.input_mode断言。更新文档为只接受smplerx。

- [ ] **Step 6：运行GREEN与共享回归。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_smplerx_retargeter tests.test_smplerx_calibration tests.test_mediapipe_tron2a -v
& $planPython -B -X utf8 -m unittest discover -s tests
git diff --check
git diff --stat
```

Expected：新测试通过，legacy显式请求失败，当前相机位置/朝向/控制测试通过，整套无新增失败；已有环境skip须逐项解释。

- [ ] **Step 7：提交本批并记录hash/tag。**

```powershell
git add src/teleoperation/retargeting/arm/retargeter.py src/teleoperation/applications/dual.py tests/test_smplerx_retargeter.py tests/test_smplerx_calibration.py tests/test_mediapipe_tron2a.py docs/current-interfaces.md
git commit -m "refactor!: remove legacy TRON2A arm input mapping"
git rev-parse HEAD
```

### Task 2 / 子项目 B：删除旧相机 normalized 处理，迁移诊断调用方

**Files:**
- Modify: `src/teleoperation/retargeting/hand/processing.py:16`、`alignment.py:6`、`coordinates.py:165`、`geometric.py:412`。
- Modify: `src/teleoperation/applications/hand_processing.py:21`、`hand_realtime.py`、`hand_align.py`。
- Modify: `src/teleoperation/data/hand_recording.py:13`、`hand_h5.py:19`。
- Modify: `src/teleoperation/contracts/coordinates.py`。
- Modify: `src/teleoperation/applications/diagnostics/diagnose_hand_coordinates.py`、`devtools/verify_geometric_hand.py`。
- Test: `tests/test_coordinate_contracts.py`、`test_coordinate_pipeline.py`、`test_mediapipe_realtime.py`、`test_world_hand_pipeline.py`、`test_world_landmark_recording.py`、`test_shared_human_dataset.py`。
- Modify: `README.md`、`docs/current-interfaces.md`。

**Interfaces:**
- Consumes: `RawHandFrame.world_hands`，`world_landmark_space='mediapipe_world_meters'`、`world_length_unit='m'`，无额外缩放。
- Produces: world-only `MediaPipePalmLocalProcessor(landmark_space='world')`、`MediaPipePalmLocalPipeline(landmark_space='world', *, input_profile=None)`。
- Produces: `align_recording(raw_hands, landmark_space='world')`；`align_mediapipe_hand(points21, *, side: str)`。
- Produces: `RawHandH5Writer(path, metadata=None, *, landmark_space='world')`；writer仍双份保存图像点与world点。
- Preserves: `build_l21_reference_basis(side)`、VR常量`COORDINATE_ALIGNMENT`、固定source→L21变换与VR实际数值。

- [ ] **Step 1：检查干净状态并创建检查点。**

```powershell
git status --short --branch
git rev-parse HEAD
$planStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$planTag = "checkpoint-$planStamp-remove-normalized-camera"
git tag -a $planTag -m 'Before retiring normalized camera retargeting'
```

- [ ] **Step 2：在 tests/test_world_hand_pipeline.py 的 WorldHandPipelineTests 添加边界测试。**

```python
def test_normalized_processing_and_writer_are_retired(self):
    from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
    from teleoperation.retargeting.hand.alignment import align_recording
    for factory in (MediaPipePalmLocalProcessor, MediaPipePalmLocalPipeline):
        with self.subTest(factory=factory), self.assertRaises(ValueError):
            factory('normalized')
    with self.assertRaises(ValueError):
        align_recording({'left': np.zeros((1, 21, 3)),
                         'right': np.zeros((1, 21, 3))}, 'normalized')
    output = self.root / 'retired.h5'
    with self.assertRaises(ValueError):
        RawHandH5Writer(output, landmark_space='normalized')
    self.assertFalse(output.exists())

def test_default_pipeline_uses_world_not_image_points(self):
    a, b = MediaPipePalmLocalPipeline(), MediaPipePalmLocalPipeline('world')
    for index in range(3):
        frame = world_frame(index)
        left = a.process_window(observation(frame))
        right = b.process_window(observation(frame))
    for side in SIDES:
        np.testing.assert_allclose(left.hands[side], right.hands[side], atol=0, rtol=0)
```

- [ ] **Step 3：在 CoordinatePipelineTests 添加旧H5拒绝测试；退休标识用字面量负例，不保留可运行旧契约常量。**

```python
def test_camera_reader_rejects_retired_aligned_files(self):
    for alignment in ('palm_local_to_l21_v2', 'source_to_l21_xyz'):
        self.write_h5(alignment, frame='l21')
        for reader in (read_coordinate_alignment, load_twohand_h5, TwoHandH5Dataset):
            with self.subTest(alignment=alignment, reader=reader), self.assertRaises(ValueError):
                reader(self.input)
```

- [ ] **Step 4：运行RED。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_world_hand_pipeline.WorldHandPipelineTests.test_normalized_processing_and_writer_are_retired tests.test_world_hand_pipeline.WorldHandPipelineTests.test_default_pipeline_uses_world_not_image_points tests.test_coordinate_pipeline.CoordinatePipelineTests.test_camera_reader_rejects_retired_aligned_files -v
```

Expected：当前normalized构造仍接受、默认pipeline使用图像点、旧H5 reader仍接受，测试因此失败。

- [ ] **Step 5：world-only默认与早期拒绝，删除分支体。** 下列检查放到各构造/函数第一段，writer必须在mkdir和h5py.File之前执行：

```python
if landmark_space != 'world':
    raise ValueError('Camera retargeting requires world metre landmarks')
```

`processing.py`删除 `_robot_bases` 及normalized取点分支，只从world_hands处理。`alignment.py`循环每侧只调用`align_mediapipe_hand(frame, side=side)`，统计output_matrix用`OPERATOR_TO_CANONICAL[side]`。`hand_align.py`只将world送给align_recording。`hand_realtime.py`无条件请求world副本，删除选择旧采集路线的三元表达式。processor/pipeline默认仍先输出米制canonical帧；正式control的profile归一化顺序不动。

`align_mediapipe_hand` 删除robot_basis参数和对应调用，保留现有side分支原样；删除不再被使用的`align_palm_local_coordinates`。保留构建L21参照基和固定轴变换，避免损坏几何/VR验证。

- [ ] **Step 6：收紧文件与契约范围，不动预览字段。**

在`hand_h5._read_coordinate_alignment`现有验证之后限制持久化相机输入：

```python
if alignment != WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT:
    raise ValueError('Camera H5 requires canonical world metre alignment')
```

原始`read_alignment_input`的mode选择改为：

```python
if landmark_space is None:
    declared = source.attrs.get('hand_processing_landmark_space')
    if isinstance(declared, bytes):
        declared = declared.decode('utf-8')
    if declared != WORLD_LANDMARK_SPACE:
        raise ValueError('Raw camera H5 must declare world processing space')
    landmark_space = 'world'
if landmark_space != 'world':
    raise ValueError('Camera alignment requires world metre landmarks')
```

后续已有world数据集、单位、形状、scale检查保留并无条件执行。`write_aligned_copy`删除normalized输出属性处理分支，保留复制RGB/身体/原始world字段行为。显式`landmark_space='world'`允许处理具有真实world副本但缺少processing-space属性的原始combined文件；仍须现有world单位证明，禁止自动猜测。

`contracts/coordinates.py`删除旧`PALM_LOCAL_COORDINATE_ALIGNMENT`及其metadata/分支，WORLD metadata改成独立字典，保持当前每个键值完全一致。保留`NORMALIZED_LANDMARK_SPACE`与`landmark_source('normalized')`给原始采集/预览，不把它视为重定向许可。`palm_local_metadata`仅接受world。全局SUPPORTED保留VR的source_to_l21_xyz、world和palm-length三种用途；相机H5以自身白名单禁止读取VR契约。

- [ ] **Step 7：迁移几何探针和坐标诊断，不保留normalized后门。**

`GeometricHandRetargeter.retarget`在循环前要求world metadata，native反变换无条件执行：

```python
from teleoperation.contracts.coordinates import (
    WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT, validate_landmark_contract)
alignment = hand_window.metadata.get('coordinate_alignment')
if alignment != WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT:
    raise ValueError('Geometric diagnostics require canonical world metre input')
validate_landmark_contract(alignment, hand_window.metadata, 'Geometric diagnostics')
```

`verify_geometric_hand.measure_side`使用`MediaPipePalmLocalPipeline('world')`；FK合成坐标作为world，而非图像推理输入：

```python
raw_observation = RawHandFrame(
    {side: raw, other: None}, 0.0, 'synthetic_l21_fk',
    {'world_landmark_space': 'mediapipe_world_meters',
     'world_length_unit': 'm', 'source_scale': 1.0},
    world_hands={side: raw, other: None})
window = pipeline.process_window(raw_observation)
```

对于此探针，不能把机器人原生左手FK坐标直接当相机world解剖坐标。先用`build_l21_reference_basis(side)`得到native掌局部坐标，再映射为操作者三点基`build_palm_basis`要求的解剖手性；用`align_mediapipe_hand(raw, side=side)`→`basis(side) @ OPERATOR_TO_CANONICAL[side]`的逆变换验证恢复原始native点。若左右手符号不一致，定位合成输入适配，不修改几何求解或VR数值来迎合探针。该探针不得删除已有零位/卷曲17维断言。

`diagnose_hand_coordinates.diagnose_h5`对已对齐相机文件用`read_coordinate_alignment`验证；`aligned_metadata_consistent`改为实际canonical world声明与验证结果，不再要求frame='l21'。旧aligned文件报错；无alignment的raw文件仍只读展示且明确未对齐。固定source→L21 transform如展示，只标为VR/reference，而非相机当前变换。

- [ ] **Step 8：迁移现有测试的夹具和断言。**

`test_coordinate_pipeline.write_h5`默认改为真实world canonical数据及`palm_local_metadata('world')`，保留旧文件为拒绝负例；将str/bytes正例测试改为world属性，并覆盖缺少时间、重复对齐和单位错误。`test_coordinate_contracts`旧native-root normalized正例改为canonical world保持距离/左右手一致性，固定source→L21的VR数学测试保留。`test_world_hand_pipeline`与`test_mediapipe_realtime`把对processing.build_l21_reference_basis的patch改为对coordinates.build_l21_reference_basis的patch，因为processor不再import它。删除只为旧坐标分支服务的测试，不删除raw normalized预览/采集数值测试。

- [ ] **Step 9：GREEN、dex与上游数值验证。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_coordinate_contracts tests.test_coordinate_pipeline tests.test_world_hand_pipeline tests.test_world_landmark_recording tests.test_mediapipe_realtime tests.test_shared_human_dataset tests.test_transhandr_vr -v
& $planPython -B -X utf8 devtools/verify_geometric_hand.py
& $planPython -B -X utf8 -m unittest discover -s tests
& '.venv/dex-native/Scripts/python.exe' -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters -v
& $planPython -B -X utf8 devtools/verify_vr_both.py --upstream-root D:/2026/code/TransHandR/TransHandR --input-h5 datasets/raw/visionpro_hand_arm_20261012_215137_live.h5 --checkpoint D:/2026/code/TransHandR/TransHandR/checkpoint/models/thumb3/linker/model_final.pth --report outputs/legacy_branch_cleanup/vr_equivalence.json
git diff --check
```

Expected：相机只接受world，预览点无变化，缺手恢复仍需三帧；两侧VR输入/原始输出最大差异0，报告保留实际源码SHA-256。真实上游资产若缺失，明确标记此验收未完成，不改metadata冒充通过。

- [ ] **Step 10：更新README/current-interfaces的实际支持范围并提交。**

```powershell
git diff --name-only
# 根据本任务Files清单逐一git add；不使用git add .混入未知文件。
git add src/teleoperation/contracts/coordinates.py src/teleoperation/retargeting/hand/coordinates.py src/teleoperation/retargeting/hand/processing.py src/teleoperation/retargeting/hand/alignment.py src/teleoperation/retargeting/hand/geometric.py src/teleoperation/applications/hand_processing.py src/teleoperation/applications/hand_realtime.py src/teleoperation/applications/hand_align.py src/teleoperation/data/hand_h5.py src/teleoperation/data/hand_recording.py src/teleoperation/applications/diagnostics/diagnose_hand_coordinates.py devtools/verify_geometric_hand.py tests/test_coordinate_contracts.py tests/test_coordinate_pipeline.py tests/test_mediapipe_realtime.py tests/test_world_hand_pipeline.py tests/test_world_landmark_recording.py tests/test_shared_human_dataset.py README.md docs/current-interfaces.md
git commit -m "refactor!: retire normalized camera retargeting branches"
git rev-parse HEAD
```

### Task 3 / 子项目 C：删除侧专用 NN 权重与推理适配

**Files:**
- Modify: `src/teleoperation/retargeting/hand/checkpoint.py:19`、`interface.py:30`、`src/teleoperation/paths.py:11`。
- Test: `tests/test_joint_checkpoints.py`、`test_joint_retarget_adapters.py`、`test_coordinate_pipeline.py`、`test_hand_control_cli.py`、`test_joint_training.py`。
- Modify: `docs/current-interfaces.md`。

**Interfaces:**
- Consumes: canonical right `HandSpec`、现有`hand_joint_nn_v1` shared payload和palm-length窗口。
- Produces: 保留`shared_joint_checkpoint_payload(model, spec)`、`load_shared_joint_checkpoint(path, spec, device='cpu')`、`save_joint_checkpoint(path, payload)`。
- Produces: 只保留`SharedJointNNRetargeter(specs, model, device='cpu', *, canonical_spec=None)`作为正式相机NN适配；`HandRetargeter`协议不变。
- Produces: `training_run_directory(root, hand_model, side, input_profile, run_name)`签名保留，side只能shared。

- [ ] **Step 1：新检查点。**

```powershell
git status --short --branch
git rev-parse HEAD
$planStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$planTag = "checkpoint-$planStamp-remove-side-nn"
git tag -a $planTag -m 'Before removing side-specific camera NN contracts'
```

- [ ] **Step 2：添加路径行为负例到JointCheckpointTests，运行RED。**

```python
def test_training_paths_only_allow_shared_weights(self):
    for side in ('left', 'right'):
        with self.subTest(side=side), self.assertRaises(ValueError):
            training_run_directory(Path('outputs/models'), 'l21', side,
                                   'camera_palm_length_window3_v1', 'run01')
```

```powershell
& $planPython -B -X utf8 -m unittest tests.test_joint_checkpoints.JointCheckpointTests.test_training_paths_only_allow_shared_weights -v
```

- [ ] **Step 3：迁移身份/限位测试，再删除旧API。**

`test_dynamic_heads_and_strict_identity_reload`及限位篡改测试改用shared payload/loader；保留每个错误元数据拒绝断言，增加侧专用旧文件负例：

```python
payload = shared_joint_checkpoint_payload(model, spec)
retired = {**payload, 'side': 'right', 'output_coordinates': 'independent_native'}
save_joint_checkpoint(path, retired)
with self.assertRaises(ValueError):
    load_shared_joint_checkpoint(path, spec, 'cpu')
```

`test_joint_retarget_adapters.py`改为用ArrayPredictor加一份shared模型，同时检查右侧输出首关节、左侧canonical_to_native_target_angles适配、缺手不调用模型、NaN/错维输出为invalid、米制未归一化窗口拒绝。保留实际数值断言：

```python
from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
from teleoperation.retargeting.hand.angles import canonical_to_native_target_angles
right = load_hand_spec('l21', 'right')
left = load_hand_spec('l21', 'left')
model = ArrayPredictor(right.bounds.mean(1))
adapter = SharedJointNNRetargeter({left.key: left, right.key: right}, model,
                                 canonical_spec=right)
window = HandWindow({'left': np.ones((3, 25, 3)), 'right': None},
                    1., 'test', palm_length_metadata())
results = adapter.retarget(window)
expected = canonical_to_native_target_angles(model.values, right, left).numpy()
np.testing.assert_allclose(results[left.key].positions, expected)
self.assertEqual(model.calls, 1)
self.assertFalse(results[right.key].valid)
```

`test_coordinate_pipeline`使用shared payload；`test_hand_control_cli`去除无用joint_checkpoint_payload导入；`test_paths_are_target_specific_and_reject_traversal`正例只保留shared，路径穿越负例仍测shared。

删除`joint_checkpoint_payload`、`load_joint_checkpoint`完整函数和`JointNNRetargeter`类及`__all__`条目。`paths.py`替换校验：

```python
if side != 'shared':
    raise ValueError('Camera NN training paths require shared target weights')
```

不改变shared payload任何既有键值，不重命名schema、不重存现有权重。

- [ ] **Step 4：GREEN与共享回归。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_joint_checkpoints tests.test_joint_retarget_adapters tests.test_joint_training tests.test_hand_control_cli tests.test_coordinate_pipeline tests.test_normalized_retargeters -v
& $planPython -B -X utf8 -m unittest discover -s tests
& '.venv/dex-native/Scripts/python.exe' -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters -v
rg -n '\b(joint_checkpoint_payload|load_joint_checkpoint|JointNNRetargeter)\b' src tests devtools
git diff --check
```

Expected：shared重载输出逐位一致、左右适配和缺手行为不变、resume通过；退休符号搜索无调用方（rg无匹配退出1是预期，不是测试失败）。

- [ ] **Step 5：更新文档并提交。**

```powershell
git add src/teleoperation/retargeting/hand/checkpoint.py src/teleoperation/retargeting/hand/interface.py src/teleoperation/paths.py tests/test_joint_checkpoints.py tests/test_joint_retarget_adapters.py tests/test_coordinate_pipeline.py tests/test_hand_control_cli.py tests/test_joint_training.py docs/current-interfaces.md
git commit -m "refactor!: remove side-specific camera NN interfaces"
git rev-parse HEAD
```

### Task 4：清理无消费者旧代码并完成范围审计

**Files:**
- Delete: `src/teleoperation/retargeting/hand/diagnostics.py`、`src/teleoperation/retargeting/hand/tracking.py`。
- Modify: `src/teleoperation/retargeting/hand/processing.py` 的未使用tracking import；`README.md`、`docs/current-interfaces.md`。
- Test: 现有 `tests/test_hand_core_regression.py`、`test_inspection_entrypoints.py`、`test_mediapipe_realtime.py`、`test_shared_human_dataset.py`；不为删除未用函数增加仅镜像实现的测试。

**Interfaces:**
- Consumes: A/B/C已完成的新边界；手部缺失/恢复仍由processor+TemporalBuffer负责。
- Produces: 无tracking/18D audit悬空导入，当前inspect仍只读hand_joint_results_v1；VR18D保持。

- [ ] **Step 1：检查消费者并创建检查点。**

```powershell
git status --short --branch
git rev-parse HEAD
rg -n 'HandIdentityTracker|_is_trackable_hand|_continuity_metrics|valid_legacy_angle_rows|summarize_angles|retargeting.hand.(tracking|diagnostics)' src tests devtools
$planStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$planTag = "checkpoint-$planStamp-remove-orphan-hand-tools"
git tag -a $planTag -m 'Before deleting orphan hand compatibility helpers'
```

Expected：tracking定义及processor未使用import，旧diagnostics仅自身定义。若此时新出现实际消费者，先在本批改成现有原生接口并增加相应行为断言，不能直接删其测试以消除ImportError。

- [ ] **Step 2：删除未使用import及两个完整模块，保留constants直接导入。** processor不再从tracking间接获取阈值；若B后没有任何阈值使用，连对应未用import一并删除。

```powershell
git rm -- src/teleoperation/retargeting/hand/diagnostics.py src/teleoperation/retargeting/hand/tracking.py
```

源代码编辑只删除`from .tracking ...`这一未使用依赖，不新增替代跟踪器。保留postprocessing.KalmanSmoother及inputs.body_tracking。

- [ ] **Step 3：用既有行为测试验证删除不改变当前链路。**

```powershell
& $planPython -B -X utf8 -m unittest tests.test_hand_core_regression tests.test_inspection_entrypoints tests.test_mediapipe_realtime tests.test_shared_human_dataset -v
& $planPython -B -X utf8 -m unittest discover -s tests
& '.venv/dex-native/Scripts/python.exe' -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters -v
& $planPython -B -X utf8 -m teleoperation hand list-models
& $planPython -B -X utf8 -m teleoperation hand control --help
& $planPython -B -X utf8 -m teleoperation dual camera --help
& $planPython -B -X utf8 -m teleoperation hand vr --help
git diff --check
```

- [ ] **Step 4：范围审计与文档收尾。**

```powershell
rg -n 'legacy|normalized|18D|18维|side.*checkpoint' src devtools README.md docs/current-interfaces.md
git diff --name-only $planTag
```

逐项分类剩余命中：图像预览、dimensionless掌长规范、VR原始18D/FK、旧输入拒绝错误、尚未并入本次的工具/入口说明。把每类保留理由写进current-interfaces；不以“搜索完全无legacy/normalized”为目标。明确写明旧数据直接作废、没有转换入口；记录测试pass/skip/fail实际数量、VR证据路径和各批次tag/hash。若触碰相机/VR共享代码，复核Task2实际等价报告而非引用历史0差异结论。

- [ ] **Step 5：提交文档与清理结果。**

```powershell
git add src/teleoperation/retargeting/hand/processing.py README.md docs/current-interfaces.md
git commit -m "refactor: remove orphan hand compatibility helpers"
git status --short --branch
git rev-parse HEAD
```

## 交付与回滚

每个子项目完成后报告：具体删除的参数值/函数/文件、保留的VR/预览边界、实际验证结果、pre-change hash/tag与post-change hash。不推送；不因主要测试通过而省略本任务已要求的dex或VR等价验证。

```powershell
git diff <实际检查点tag> -- src/teleoperation
git switch -c rollback-legacy-branch-cleanup <实际检查点tag>
```

回滚命令中的tag必须替换为工作记录中的实际值；优先创建恢复分支或恢复指定文件，不运行reset --hard，不删除标签。

## 计划自审记录

- A有默认映射/旧模式拒绝、空间拒绝、IK与安全回归；B有构造/H5拒绝、world一致性、几何调用迁移、dex与真实VR等价；C有shared重载、身份/限位篡改、左右/缺手、训练路径/resume；D有删除消费者审计与现有行为回归。
- 已区分图像normalized与旧NN输入；没有要求删除VR18D或build_l21_reference_basis。
- 所有新代码示例依赖现有源码/测试里已定义的类和夹具；不增加运行依赖，不把单侧权重改标签当shared。
- 任务可独立提交，建议A→B→C→D顺序，避免同时修改coordinates或同一组测试造成冲突。
- 损坏main尾部、sim demo/dummy、CLI统一与preflight重建明确不在本次范围，不会因“顺手清理”扩大删除。
