# Unused Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除源码审阅确认不再使用的旧模块、函数、测试依赖和文档说明；混合文件仅删除无用部分，剩余活跃分支交给用户处理。

**Architecture:** 依据入口与消费者逐项删除，不引入新架构。测试专用旧API退出时，将仍有价值的行为断言迁移到现有shared接口；保留所有仍有业务或验收工具消费者的代码。每批结束可独立验证与回退。

**Tech Stack:** Python、unittest、NumPy、PyTorch、HDF5、PyBullet、PowerShell、Git；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-04-unused-code-cleanup.md`。执行者先读本范围说明和AGENTS.md；此计划替代2026-11-03旧删除计划，不合并其活跃分支迁移任务。

## Global Constraints

- 不新增兼容层，不迁移活跃数据接口，不修改重定向数值算法，不增加依赖。
- 保留RGB/VR并列输入与相同NN/dex后端选择；不把VR来源标记为MediaPipe。
- 当前输入、窗口、结果、shared权重身份和录制接口保持原行为；删除范围内的未使用旧API按本授权退出。
- 每批修改前检查Git状态、保存dirty intended state、创建annotated checkpoint；验证后提交，报告hash/tag，不推送。
- 运行最小相关unittest；涉及共享行为时运行完整unittest discover -s tests；dex跳过项在独立环境补跑，逐项记录。
- 删除对象没有残余业务引用或文档使用说明；活跃入口、规格校验、缺手保持、左右适配及回放仍通过回归。
- 真实相机、头显和实体硬件没有执行时，明确记录未验收，不将离线/模拟测试写为硬件成功。

---

## 基线、执行顺序与Git约定

源码基线：`0b20331162fa04f785f45040e1056cf96e998cc5`。本计划文档修改前检查点：`checkpoint-20261104-0103-plan-unused-cleanup`。实施代码清理时重新建立各批检查点，不能复用计划检查点。

按Task 1→6顺序执行；本计划不需要并发智能体。每项中的复查、删除、测试、文档同步及提交是一个独立批次。用已有回归验证删除，不为“某符号不存在”增加永久测试；只迁移仍有行为价值的测试。

初始化执行环境：

```powershell
$cleanupRepo = 'D:/2026/code/teleoperation-system-retargeting'
$cleanupPython = 'D:/Anaconda/envs/teleoperation/python.exe'
Set-Location -LiteralPath $cleanupRepo
$env:PYTHONPATH = Join-Path $cleanupRepo 'src'
& $cleanupPython -B -X utf8 -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw '主环境解释器不可用，不能开始代码清理。' }
```

每批Step 1先执行以下公共步骤，再读取任务指定文件：

```powershell
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting status --short --branch
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting rev-parse HEAD
```

若有未提交变更，核对diff；属于当前预期状态的先按AGENTS.md提交，不覆盖不明改动。然后按任务给出的短名创建annotated tag，记录实际tag与hash：

```powershell
# 各Task明确给出cleanupBatch值；每个新批次重新生成时间。
$cleanupStamp = Get-Date -Format 'yyyyMMdd-HHmm'
$cleanupTag = "checkpoint-$cleanupStamp-$cleanupBatch"
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting tag -a $cleanupTag -m "Before $cleanupBatch"
if ($LASTEXITCODE -ne 0) { throw '检查点未创建，不开始编辑。' }
```

Git因所有权限制报错时只使用上述命令级safe.directory；不修改全局设置。写入Git被权限限制时使用现有授权流程，不跳过检查点。

每批测试成功后，git diff --check，显式git add本批文件并提交；删除文件用git add -u -- <本批路径>。记录提交hash和pre-change tag。测试不通过时先区分本批回归和既有问题，不降低断言掩盖失败。

## Task 1：删除无消费者的整模块

**Files:**
- Delete: `src/teleoperation/applications/visionpro.py`。
- Delete: `src/teleoperation/retargeting/hand/diagnostics.py`、`tracking.py`。
- Delete: `src/teleoperation/simulation/hand_mount.py`。
- Delete: `src/teleoperation/tools/hand_reports.py`、`training_hand_pose.py`。
- Delete: `src/teleoperation/learning/config.py`、`src/teleoperation/inputs/interface.py`。
- Modify: `src/teleoperation/retargeting/hand/processing.py:5`、`src/teleoperation/cli/schemas.py`的dual_realtime帮助。
- Test: `tests/test_world_hand_pipeline.py`、`tests/test_shared_human_dataset.py`、`tests/test_cli_integration.py`、`tests/test_unified_hand_inputs.py`。

**Interfaces:**
- Consumes: 当前MediaPipePalmLocalProcessor和dual realtime的鸭子类型输入协议。
- Produces: 原有process_frame()/reset()和CLI运行行为；不再导出InputSource及无消费者旧工具。

- [x] **Step 1：复查消费者并建立本批检查点。** 设置`$cleanupBatch = 'remove-orphan-modules'`，执行公共Git步骤。搜索模块名及导出符号；排除文档删除清单和只定义自身的命中。

```powershell
rg -n 'VisionProWorkflow|summarize_angles|valid_legacy_angle_rows|HandIdentityTracker|_is_trackable_hand|DEFAULT_MAX_CENTER_DISPLACEMENT|DEFAULT_MAX_SHAPE_RMSE|hand_mount|hand_reports|training_hand_pose|learning\.config|\bInputSource\b' src tests devtools README.md docs
```

- [x] **Step 2：删除八个文件和闲置导入。** processing.py删除整条`.tracking`导入，不更改identity_tracking=False或引入替代tracker。schemas.py把adapter帮助替换为以下准确描述，运行时代码不变：

```python
parser.add_argument(
    "--adapter", required=True,
    help="module:factory returning an object with next_observation() or an iterable of RawTeleopObservation",
)
```

- [x] **Step 3：同步本批文档命中。** 删除对应旧模块的用法、路径和功能介绍；仍使用的身体身份锁定不属于HandIdentityTracker，不删除其说明。保存本批实际删除文件清单。
- [x] **Step 4：运行已有行为回归和完整套件。**

```powershell
& $cleanupPython -B -X utf8 -m unittest tests.test_world_hand_pipeline tests.test_shared_human_dataset tests.test_cli_integration tests.test_unified_hand_inputs -v
& $cleanupPython -B -X utf8 -m unittest discover -s tests
```

预期：无ImportError，RGB/VR共同输入、相机窗口和CLI测试通过；仅已说明的可选依赖用例允许跳过。检索除当前计划/spec中的删除清单外，不再出现退役模块引用。

- [x] **Step 5：提交。** 提交消息`refactor: remove unused hand tools and input declarations`，记录hash/tag。

## Task 2：清理仍在使用文件中的无调用片段

**Files:**
- Delete: `src/teleoperation/applications/hand_export.py`。
- Modify: `src/teleoperation/applications/hand.py`、`realtime_hand_sim.py`。
- Modify: `src/teleoperation/learning/dataset.py`、`src/teleoperation/data/checkpoint.py`。
- Modify: `src/teleoperation/robots/action_mapping.py`、`partial.py`、`src/teleoperation/contracts/commands.py`。
- Modify: `src/teleoperation/retargeting/hand/postprocessing.py`。
- Test: `tests/test_hand_core_regression.py`、`test_joint_commands.py`、`test_shared_human_dataset.py`、`test_joint_training.py`、`test_native_hand_replay.py`、`test_cli_integration.py`、`test_application_entrypoints.py`。

**Interfaces:**
- Consumes: CLI直接使用hand_control:export，SimulationEnv使用validate_native_action，关节处理器使用KalmanSmoother。
- Produces: 上述接口、PartialLimbVector拒绝语义、Tron2AL21Command.flatten和数据集行为不变。

- [ ] **Step 1：复核并建立检查点。** `$cleanupBatch = 'trim-unused-members'`。核对以下对象只有定义或仅在待删除链内部互相调用。

```powershell
rg -n 'hand_export|TwoHandH5ChunkedGenerator|_is_valid_hand_frame|\b_stack\b|\b_mask\b|write_checkpoint|ActionAdapter|pack_tron2a_l21_action|H1JointTargets|GR1JointTargets|G1JointTargets|pack_partial_limb_vector|classify_jumps|detect_bad_frames|hold_last_valid|detect_identity_swaps|repair_stream|repair_bad_frames|moving_average|motion_metrics|compare_methods' src tests devtools README.md docs
```

- [ ] **Step 2：按spec精确删除。** 删除hand.py.export及hand_export.py；不改hand.py.align/inspect或CLI handler。删除dataset的旧生成器及_stack/_mask/_is_valid_hand_frame，不动_scalar_or_none。删除write_checkpoint，保留read_checkpoint。删除旧ActionAdapter链及三种专属JointTargets，清理dataclass/Mapping等随之闲置的导入；保留validate_native_action、Tron2AL21Command、PartialLimbVector。postprocessing只保留KalmanSmoother及其所需导入，不改变其实现和默认参数。
- [ ] **Step 3：移除失效模块启动尾部。** 删除realtime_hand_sim.py末尾`if __name__ == '__main__'`块，不新增main或转发入口；保留RealtimeL21Scene、CameraPreview、build_scene、build_preview及现有活跃辅助函数。删除相关docstring中依赖os._exit/main的过时说明，仅移除确认为尾部专用的导入。
- [ ] **Step 4：更新本批说明和运行回归。** 清除postprocessing模块说明里调用已删classify_jumps的建议，保留卡尔曼参数说明并与实际gate默认值一致；若文档列出realtime_hand_sim模块启动命令则删去，当前hand control命令保留。

```powershell
& $cleanupPython -B -X utf8 -m unittest tests.test_hand_core_regression tests.test_joint_commands tests.test_shared_human_dataset tests.test_joint_training tests.test_native_hand_replay tests.test_cli_integration tests.test_application_entrypoints -v
& $cleanupPython -B -X utf8 -m unittest discover -s tests
```

预期：训练、导出、原生动作验证、场景执行和回放保持原有行为；不以“旧模块帮助不报错”代替正式CLI验收。

- [ ] **Step 5：提交。** 消息`refactor: trim unused helpers from active modules`，记录hash/tag。

## Task 3：迁移有价值的测试，退出旧单侧NN与权重接口

**Files:**
- Modify: `src/teleoperation/retargeting/hand/interface.py`、`checkpoint.py`。
- Modify/Test: `tests/test_joint_retarget_adapters.py`、`test_joint_checkpoints.py`、`test_coordinate_pipeline.py`、`test_normalized_retargeters.py`、`test_normalized_dex.py`、`test_hand_control_cli.py`。
- Regression: `tests/test_joint_training.py`、`test_unified_hand_inputs.py`。

**Interfaces:**
- Consumes/Produces: `SharedJointNNRetargeter(specs, model, device='cpu', *, canonical_spec=None)`、`shared_joint_checkpoint_payload(model, spec)`、`load_shared_joint_checkpoint(path, spec, device='cpu')`、`save_joint_checkpoint(path, payload)`，签名及实现不变。
- Removed: JointNNRetargeter、joint_checkpoint_payload、load_joint_checkpoint。未使用旧接口不提供别名。

- [ ] **Step 1：建立检查点并核对测试覆盖。** `$cleanupBatch = 'remove-unused-side-nn'`。列出旧API所有调用；确认生产入口已使用shared版本。

```powershell
rg -n '\b(JointNNRetargeter|joint_checkpoint_payload|load_joint_checkpoint)\b' src tests devtools
```

- [ ] **Step 2：先迁移测试并在删除实现前运行。** `test_joint_checkpoints`的动态维数、身份篡改、限位篡改用例改调用shared payload/loader；保留synthetic21网络头覆盖、shared身份校验以及现有路径函数的原有测试，不改paths.py。坐标元数据测试改为right spec和shared payload，断言side='shared'、output_coordinates='canonical_right'、units='1'和7维输出。hand_control_cli移除闲置旧导入。

checkpoint测试中的构建与加载统一替换为：

```python
spec = load_hand_spec(name, "right")
model = build_joint_model(spec, SMALL).eval()
payload = shared_joint_checkpoint_payload(model, spec)
save_joint_checkpoint(path, payload)
loaded, metadata = load_shared_joint_checkpoint(path, spec, "cpu")
torch.testing.assert_close(model(values), loaded(values), atol=0, rtol=0)
self.assertEqual(metadata["side"], "shared")
```

保留原有hand_model/spec_hash/joint_names/input_profile/reference_geometry篡改循环，使用shared loader验证全部拒绝；angle_clamper.angle_min=-100的负例仍应报limits，不能删掉或改为只检测文件存在。

`test_joint_retarget_adapters`的左手有效/右手缺失测试，对l21/g1使用一份right模型，并以明确符号预期检查首关节和完整输出：

```python
specs = {(name, side): load_hand_spec(name, side) for side in ("left", "right")}
right = specs[(name, "right")]
canonical = right.bounds.mean(1)
model = ArrayPredictor(canonical)
adapter = SharedJointNNRetargeter(specs, model)
window = HandWindow(
    {"left": np.ones((3, 25, 3)), "right": None},
    1., "test", palm_length_metadata(),
)
results = adapter.retarget(window)
expected = canonical.copy()
expected[[0, 3, 6, 9] if name == "l21" else [1, 2, 4, 5, 6]] *= -1
np.testing.assert_allclose(results[(name, "left")].positions, expected)
self.assertEqual(results[(name, "left")].positions.shape, (right.dof,))
self.assertFalse(results[(name, "right")].valid)
self.assertEqual(model.calls, 1)
```

保留timestamp、joint_names、spec_hash断言。坏输出测试选择right侧，使用SharedJointNNRetargeter({spec.key: spec}, ArrayPredictor(values))验证错误维数/NaN为invalid；米制未归一化输入测试仍检查模型调用数为0。左右缺失互换的断言使用同一模型调用计数，不再假设每侧各有模型。

normalized_retargeters中的旧构造改为一份right模型。normalized_dex中对left-only目标显式提供canonical_spec，保持与dex结果的身份和维数比较：

```python
canonical_spec = load_hand_spec("l21", "right")
nn = SharedJointNNRetargeter(
    {spec.key: spec}, build_joint_model(canonical_spec, SMALL),
    canonical_spec=canonical_spec,
)
```

```powershell
& $cleanupPython -B -X utf8 -m unittest tests.test_joint_checkpoints tests.test_joint_retarget_adapters tests.test_coordinate_pipeline tests.test_normalized_retargeters tests.test_hand_control_cli -v
```

预期：新测试在仍保留旧实现时已经通过，证明覆盖迁移本身正确。若shared接口现有行为与旧接口不同，按照shared的公开语义调整测试，不趁清理修改算法。

- [ ] **Step 3：删除旧API。** 删除JointNNRetargeter类及__all__条目；删除两个旧checkpoint函数及其专属闲置导入，保留_reference_geometry和shared的全部严格校验。再次检索，src/tests/devtools不应残留旧API引用。
- [ ] **Step 4：验证并同步说明。** 运行Step 2测试加test_joint_training、test_unified_hand_inputs，随后完整unittest。独立dex环境补跑test_normalized_dex与test_normalized_retargeters（环境命令见Task 6）。文档仅保留shared权重用法；不删除活跃camera profile或声称所有历史路径已统一。
- [ ] **Step 5：提交。** 消息`refactor: retire unused side-specific neural interfaces`，记录hash/tag。

## Task 4：删除旧VR闲置控制循环及测试专用旧录制格式

**Files:**
- Modify: `src/teleoperation/applications/dex_native.py`、`visionpro_hand.py`。
- Delete: `src/teleoperation/data/world_landmark_recording.py`。
- Modify/Test: `tests/test_dex_native.py`、`tests/test_world_landmark_recording.py`。
- Regression: `tests/test_unified_hand_inputs.py`、`test_hand_control_cli.py`、`test_hand_control_h5.py`、`test_body_capture_replay.py`、`test_transhandr_vr.py`。

**Interfaces:**
- Preserved: hand dex-calibrate→calibrate；recording_frames(path, source)、file_hash(path)、canonical_window(frame)仍供CLI/验收工具使用。
- Preserved: DexInputAdapter、DexCalibration、DexNativeRetargeter、NormalizedDexRetargeter、NativeVectorSolver；不编辑它们的数值实现。
- Removed: LatestCameraInput、visionpro_run及其专属command_pipeline；旧world_landmarks_h5_v1独立reader/writer。

- [ ] **Step 1：建立检查点并复核工具依赖。** `$cleanupBatch = 'remove-idle-vr-and-world-format'`。读取dex_native_acceptance.py performance函数，确认recording_frames/canonical_window仍是消费者；读取CLI注册确认calibrate仍登记。

```powershell
rg -n 'visionpro_run|LatestCameraInput|applications\.dex_native|canonical_window|recording_frames|\bcommand_pipeline\b|world_landmark_recording|LegacyWorldRecordingTests' src tests devtools README.md docs
```

- [ ] **Step 2：清理闲置VR代码和专属测试。** 删除LatestCameraInput、visionpro_run、command_pipeline；删除只供它们使用的json/time/numpy/HandJointResultWriter/DexNativeRetargeter/VECTOR_PARAMETERS等导入，按剩余引用核对，不删仍供calibrate使用的hashlib/Path/iter_dex_observations/DexInputAdapter/DexCalibration。删除test_latest_source_drops_backlog_and_closes和旧visionpro_run录制用例；保留calibrate拒绝camera、原生模型/命令/场景、当前CLI校验测试。

已有test_unified_hand_inputs.test_vr_h5_uses_existing_shared_nn_and_generic_results覆盖当前VR控制的七帧、缺手、恢复及快捷入口一致性；test_hand_control_h5覆盖raw/accepted分离及严格回放；这些断言必须继续执行，不能只删除旧集成用例而不检查当前链。

- [ ] **Step 3：删除旧格式和混合测试中的专属部分。** 删除world_landmark_recording.py，删除同名测试文件的旧reader/writer导入、LegacyWorldRecordingTests和仅由它使用的frame/points fixture。保留CombinedCaptureTests、CLI命令注册、时间命名、防覆盖和组合H5消费者的所有用例，测试文件不整删。
- [ ] **Step 4：修正明确失效的提示片段。** visionpro_hand.py的缺权重错误改为`TransHandR VR requires --checkpoint`，删掉建议给当前dex传固定标定的尾句；保留NN/dex对错误传入dex_calibration的拒绝检查及当前CLI选项，不移除活跃标定命令。
- [ ] **Step 5：验证、同步文档、提交。**

```powershell
& $cleanupPython -B -X utf8 -m unittest tests.test_dex_native tests.test_world_landmark_recording tests.test_unified_hand_inputs tests.test_hand_control_cli tests.test_hand_control_h5 tests.test_body_capture_replay tests.test_transhandr_vr -v
& $cleanupPython -B -X utf8 -m teleoperation hand dex-calibrate --help
& $cleanupPython -B -X utf8 -m teleoperation hand vr --help
& $cleanupPython -B -X utf8 -m unittest discover -s tests
```

预期：旧格式引用归零，当前录制/H5/VR/TransHandR测试通过，dex-calibrate仍注册。不运行真实设备或重拟合标定作为本批前置条件。提交`refactor: remove unused VR control loop and recording format`，记录hash/tag。

## Task 5：清理过时文档并明确仍在使用的范围

**Files:**
- Modify: `README.md`、`docs/current-interfaces.md`、`docs/unified-hand-input.md`（只在有实际命中时修改）。
- Delete: `docs/superpowers/plans/2026-11-03-remove-legacy-branches.md`、`docs/superpowers/specs/2026-11-03-remove-legacy-branches.md`。
- Update status: 本计划及对应spec。
- Inspect: `docs/research/*.md`、`devtools/preflight.ps1`及所有修改文件docstring；不修改无关研究内容和环境引导算法。

**Interfaces:**
- Consumes: Tasks 1–4的真实删除清单和保留消费者。
- Produces: 与当前代码一致的使用说明；旧模块不再被介绍为可调用接口；历史设计不再指挥删除活跃分支。

- [ ] **Step 1：建立检查点。** `$cleanupBatch = 'prune-retired-code-docs'`。完整检索已删除文件名、类名、函数名、旧启动命令、旧格式名及旧计划链接；使用符号边界避免将JointNNRetargeter误匹配到SharedJointNNRetargeter。
- [ ] **Step 2：删除旧说明与旧计划/spec。** 清除README/current-interfaces中的旧删除流水账、过期接口介绍、旧方案执行说明。删除2026-11-03计划和spec，清理引用它们的链接。当前计划/spec的删除清单可保留作为本次工作记录，不保留旧API调用示例。
- [ ] **Step 3：修正文档的事实边界。** 明确新的共同手控制与仍在用的camera profile/训练/三机器人/TRON2A边界；删除“旧相机profile全部删除”这类与实际不符的泛称，保留准确的当前profile名称。保留dex-calibrate和验收工具仍用旧标定组件的事实，不宣称已全部移除。保留上游VR源码哈希表、数值证据及硬件验收限制；历史549/560测试数字不充作本次结果。
- [ ] **Step 4：检查链接和提交。** `git diff --check`；逐个检查受影响本地Markdown链接的目标文件是否存在，研究文档有链接时仅修链接。不按legacy/18D/normalized关键词整段删文档。提交`docs: remove retired interface documentation`，记录hash/tag。

## Task 6：最终回归与交接清单

**Files:**
- Inspect: 全部变更的src/tests/devtools/docs及CLI注册。
- Modify: 本计划的完成状态与`docs/current-interfaces.md`的实际验收结果。
- No changes: datasets、outputs、权重、上游TransHandR源码、仍在使用的旧分支。

**Interfaces:**
- Consumes: 前五批提交和检查点。
- Produces: 确认删除的对象、仍保留的消费者、实际测试pass/skip/fail、未执行硬件验收、可回退hash/tag。

- [ ] **Step 1：建立最终验收记录检查点。** `$cleanupBatch = 'verify-unused-cleanup'`。确认工作区无不明变更，核查diff没有修改retargeter数值方法、profile约定、控制默认值或路径API。
- [ ] **Step 2：运行完整主环境回归。**

```powershell
& $cleanupPython -B -X utf8 -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw '完整回归失败，不能宣告清理完成。' }
```

不写死通过数量；删除旧测试后数量会下降。分别记录测试总数、失败数、跳过数和具体原因。

- [ ] **Step 3：独立dex环境补验当前实际后端。**

```powershell
. ./devtools/dex_native_env.ps1
if (-not (Test-Path -LiteralPath $dexPython)) { throw 'dex独立环境缺失，记录未完成的后端验证。' }
& $dexPython -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters tests.test_unified_hand_inputs -v
if ($LASTEXITCODE -ne 0) { throw 'dex回归失败，不能将主环境skip写为通过。' }
```

如果本次确实未动算法/VR预处理和权重，不重新生成昂贵的上游数值报告，只运行test_transhandr_vr等现有回归；不能把旧报告写为本次新验证。如果diff意外触及数值路径，先撤回越界修改，数值迁移留给用户处理。

- [ ] **Step 4：确认删除完整且没有误删。** 将spec中的十个删除文件逐个用Test-Path确认不存在；用rg复查退役符号在src/tests/devtools中没有消费者。当前CLI帮助至少检查hand control/export/vr/dex-calibrate、hand record-world、dual realtime、replay hand-joints；这些仅执行--help，不打开设备。检查保留的Kalman、shared loader、NativeVectorSolver、TransHandR、MediaPipePalmLocalPipeline、camera profile仍有原有消费者。检查文档内无退役用法和断链，允许新计划/spec的删除清单定位名称。
- [ ] **Step 5：记录并提交验收。** 写明“已删清单”和“剩余交给用户清单”，逐项附具体消费者；勾选完成任务，记录真实验证结果和本批hash/tag。提交`docs: record scoped cleanup verification`。最终报告所有批次提交与检查点；不push。

## 最终应交还给用户的剩余代码

|剩余部分|保留原因与消费者|
|---|---|
|dex-calibrate、DexInputAdapter、DexCalibration、DexNativeRetargeter|CLI标定和dex_native_acceptance仍用；只删无调用旧控制循环|
|MediaPipePalmLocalPipeline与camera_palm_length_window3_v1|训练/三机器人/TRON2A仍用；不迁移活跃调用方|
|双臂legacy与normalized诊断分支|仍有旧行为测试或几何验收消费者；本次不重构|
|三处模型构造与两份GBK扫描|各有当前调用方；合并属于后续重构|
|活跃H5 schema与PartialLimbVector拒绝契约|不同录制用途/完整动作校验仍用|
|TransHandR、FK、Kalman、身体跟踪、原生执行|明确属于当前控制或上游验证能力|
|preflight.ps1及现有training_run_directory侧参数|工具/接口仍存在；本次不调整环境与路径政策|

## 回退

每个Task回退使用它自己的pre-change tag。优先只恢复该批文件，或从tag建立回退分支：

```powershell
# cleanupTag为该Task创建并记录的检查点；先核对它指向目标批次。
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting show --no-patch $cleanupTag
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting diff $cleanupTag -- src tests docs README.md
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting switch -c rollback-unused-cleanup $cleanupTag
```

不执行reset --hard，不删除tag，不重写历史。

## 执行记录

- Task 1：检查点 `checkpoint-20261104-0111-remove-orphan-modules`；修改前 `1bba6551b07c785ddef9d0b4eb86990623c70f7b`；相关回归35项（1项跳过），完整回归560项（557通过、3项dex依赖跳过）；无失败。
