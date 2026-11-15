# 固定掌根 Dex Position Implementation Plan

> **For agentic workers:** 在当前会话逐任务顺序执行。使用可用的执行计划工作流；未经用户要求不启用子智能体。

**Goal:** 将上游 PositionOptimizer 接入当前手部重定向接口，输出目标手独立关节。

**Architecture:** 新增独立 Position solver 和 HandRetargeter 适配器，通过现有工厂选用。
复用 HandSpec、build_hand_targets 和 JointCommandPipeline，在边界完成根坐标转换及具名关节排序。
保持现有 Vector 实现及显式 TransHandR VR 路由兼容。

**Tech Stack:** Python 3.10+、NumPy、PyTorch、dex_retargeting 0.5.0+winconda、Pinocchio、NLopt、unittest、HDF5；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-09-dex-position.md`，执行前完整阅读。

## Global Constraints

- 用户已确认：固定掌根，只输出手指关节。
- 新增 `--backend dex-position`；现有 `nn`、`dex`、`transhandr` 的含义和默认值保持兼容。
- 输入、窗口、结果和录制 schema 保持兼容；只在 metadata 增加真实算法信息。
- VR 不隐式套用相机预处理，显式 TransHandR 路由及其上游数值流程保持兼容。
- 不新增依赖、自由根关节、自动后端回退、求解重试或通用 solver 框架。
- 每个代码批次先运行 git status、保存已确认的当前修改、创建 annotated checkpoint；验证后提交，不 push。
- 主工作区存在其他任务修改；本计划位于 `outputs/tmp_dex_position_plan` 独立工作区，分支 `codex/dex-position-plan`。
- 规划前提交 `1c9a8f85420154af5e029565df99faa4e4a7486b`，标签 `checkpoint-20261109-0100-dex-position-plan`。

## 文件职责

|文件|职责|
|---|---|
|新增 src/teleoperation/retargeting/hand/dex_position.py|固定根 Position 求解、状态、结果转换及窗口适配|
|applications/hand_backend.py|工厂构造并保留精确选定的 model/side 集合|
|cli/hand_joint_options.py、robot_options.py、visionpro_options.py|新增后端参数选项|
|applications/hand_control.py、robot_control.py、visionpro_hand.py|参数前置检查、路由、metadata、可用后端展示|
|devtools/windows_runtime.py|Position 按 dex 依赖检查和选择解释器|
|tests/test_dex_position.py（新增）|真实优化器等价性、坐标、精度、状态和失败测试|
|tests/test_hand_backend_factory.py|工厂目标选择和 checkpoint 拒绝|
|tests/test_hand_control_cli.py、test_robot_control_cli.py、test_unified_hand_inputs.py、test_windows_launchers.py|入口、来源路由、依赖环境和录制兼容|
|README.md、docs/current-interfaces.md、docs/unified-hand-input.md|公开选项、输入输出、参数和限制|

表中省略的 applications/cli 路径前缀均为 `src/teleoperation/`。不改旧比较器、训练器、资产或 Vector 算法。

## Task 1：固定根 Position 求解和窗口适配

**Files:** 新增 `src/teleoperation/retargeting/hand/dex_position.py`、`tests/test_dex_position.py`。

**Interfaces:**

```python
class NativePositionSolver:
    def __init__(self, spec): ...  # HandSpec；直接加载 spec.urdf_path
    def reset(self) -> None: ...
    def solve(self, targets_native_m, timestamp): ...  # (N,3) -> HandJointResult

class NormalizedDexPositionRetargeter(HandRetargeter):
    def __init__(self, specs): ...  # dict[HandKey, HandSpec]
    def reset(self) -> None: ...
    def retarget(self, window): ...  # HandWindow -> dict[HandKey, HandJointResult]
```

两个类仅在实现后存在；后续任务按这些签名使用。复用 build_hand_targets，避免维护第二套归一化代码。

- [ ] 1. 写真实上游等价性测试：以 l21/right spec 建立两个独立 RobotWrapper，手动构造 PositionOptimizer 和 SeqRetargeting 作为 oracle，另一条调用 NativePositionSolver。同参数/限位/mimic/初值、连续三组可达 FK 目标，比较独立关节序列。oracle 不调用被测 solver 的构造助手。
- [ ] 2. 运行新增测试确认因导入新模块失败；使用主仓库 `.venv/dex-native/Scripts/python.exe`，PYTHONPATH 必须指向此工作区 src，不能误测 editable 安装的主工作区代码。
- [ ] 3. 实现初始化，配置如下（固定关节按 optimizer.fixed_joint_names 排序；mimic 配置沿用 spec 已解析的 driver 关系）：

```python
optimizer = PositionOptimizer(
    robot, list(spec.independent_joint_names), list(spec.tip_frames),
    np.asarray(spec.human_tip_indices),
    huber_delta=.02 * spec.reference_palm_length_m, norm_delta=0.,
)
# 若存在 dependent joints，在建立 SeqRetargeting 前设置 MimicJointKinematicAdaptor。
sequence = SeqRetargeting(optimizer, has_joint_limits=True, lp_filter=None)
optimizer.set_joint_limit(spec.bounds, epsilon=0.)
optimizer.opt.set_ftol_abs(1e-5 * spec.reference_palm_length_m)
optimizer.opt.set_xtol_abs(1e-6)
optimizer.opt.set_maxeval(100)
sequence.last_qpos = spec.q_reference.copy()
```

先将 spec.bounds 写入 robot 对应独立关节的 lower/upperPositionLimit，确保 SeqRetargeting 自己的裁剪也使用有效限位。
固定关节取 robot.q0 的对应值，并验证根的祖先没有优化关节；根的 FK 变换缓存只在满足固定根前提时成立。

- [ ] 4. 实现 solve：要求 finite `(len(spec.tip_frames),3)`；将局部米制目标转换到 FK 坐标系；保存 last_qpos，调用 sequence.retarget；核对 NLopt status > 0；用 sequence.joint_names 到 spec.independent_joint_names 的具名映射取输出；校验有限性/有效限位；构造原有 HandJointResult。数值失败恢复上一状态并返回 invalid，不复用 Vector 的 G1 recovery。
- [ ] 5. 实现窗口适配，每 key 调用 build_hand_targets；None 输入只 reset 对应 solver 并返回 invalid。reset 清理各 solver 并将初值恢复到 spec.q_reference。
- [ ] 6. 增加实际断言：完整目标集合；spec_hash/时间戳/独立关节顺序；单侧缺失不影响另一侧；reset 后重放一致；错误 profile 抛出；故障注入非正 NLopt status/NaN/越限输出时 invalid 且 last_qpos 恢复。用 translated/rotated fixed-root URDF 验证 p_fk 转换。
- [ ] 7. 扩展可达 FK 样本到四模型×双侧：按 bounds 内确定性采样生成目标，最大指尖误差 <= .005 m；失败显示 model/side/seed/残差，不能放松阈值掩盖问题。mimic 模型展开结果必须满足全部原生关节限位。

核心等价性断言使用：

```python
np.testing.assert_allclose(actual.positions, expected_named, atol=1e-6, rtol=0)
self.assertTrue(actual.valid, actual.error)
self.assertEqual(actual.joint_names, spec.independent_joint_names)
```

HandJointResult 的失败说明字段为 error，关节顺序字段为 joint_names，不引入同义属性兼容层。

- [ ] 8. 运行 `python -B -m unittest tests.test_dex_position tests.test_normalized_dex tests.test_normalized_retargeters -v`（dex-native 解释器）；确认 Position 测试实际执行，完成后 checkpoint 后续批次并提交 `feat: add fixed-root dex position hand retargeter`。

## Task 2：接通工厂、所有正式入口和记录

**Files:** 修改文件职责表中的工厂、CLI、applications、Windows runtime 及对应测试。

**Interfaces:** 保持 `build_hand_retargeter_from_specs(specs, backend, *, checkpoints=None, device='cpu')` 和各上层工厂签名。
新的分支返回 `NormalizedDexPositionRetargeter(specs)`，无需把非对称 spec 集合扩成笛卡尔积。

- [ ] 1. 写入口失败测试：hand control/export/vr 及 robot control 能解析 dex-position；与 dex 相同拒绝 NN checkpoint；空的 arm-only specs 不加载 dex；g1/right+h1_2/left 仅返回这两个 key。
- [ ] 2. 跑 `python -B -m unittest tests.test_hand_backend_factory tests.test_hand_control_cli tests.test_robot_control_cli tests.test_windows_launchers -v`，先确认新选项失败。
- [ ] 3. hand_backend 扩展允许集合，checkpoint 拒绝适用于 dex 与 dex-position，空集合检查维持原先顺序，新分支局部导入新类。保持现有 dex 分支不变。
- [ ] 4. 三个 CLI options 增加 dex-position。hand_control._selection、robot_control.preflight 将其按优化后端校验；visionpro_hand.run 将其纳入统一路由，继续拒绝固定 VR calibration/scale 和不合契约的侧映射。更新 list_models 的 backends 描述。
- [ ] 5. 新模块提供 POSITION_PARAMETERS 常量，hand_control._metadata 写入独立 Position 参数；robot_control 录制也写相应算法参数。保留现有 backend/target_vectors 等字段，绝不写入 VECTOR_PARAMETERS 或声称上游默认配置。
- [ ] 6. windows_runtime 中两处依赖检查和解释器选择将 dex-position 归入 dex 环境；测试 `--backend=dex-position` 和分开参数格式、最后选项生效、参数/返回码保持。未安装 dex 时显示安装诊断，不退到 NN。
- [ ] 7. 增加不启设备的集成测试：同一 HandWindow 经过工厂与直接适配器结果一致；camera、统一 VisionPro 已声明窗口进入同一 solver；hand vr 新分支进入 unified_run；transhandr 仍进入原分支。H5 roundtrip 通过现有读取器，metadata 真实 type=position，command 独立/展开关节维度正确。
- [ ] 8. 运行该任务测试加 `tests.test_unified_hand_inputs tests.test_hand_control_h5 tests.test_robot_command_recording tests.test_transhandr_vr tests.test_visionpro_trim`，再在正式环境跑全套；提交 `feat: expose dex position through hand and robot workflows`。

## Task 3：验收、文档与回归证据

**Files:** 更新 README.md、docs/current-interfaces.md、docs/unified-hand-input.md；在此计划记录实施结果。

**Interfaces:** 用户通过现有命令选算法；不创建新命令族，不改变既有比较器的 NN/Vector 约定。

- [ ] 1. 文档明确区分 position/vector 的损失、索引、坐标、尺度、固定根以及项目参数；记录安装包和参考库哈希、运行时数值等价结果。
- [ ] 2. 写入以下用法（实施后才可使用；输入必须是现有命令接受的真实文件，输出须为新文件）：

```powershell
python -B -m teleoperation hand control --source camera --hand-model l21 --hand right --backend dex-position
python -B -m teleoperation hand export --input aligned.h5 --output position_results.h5 --hand-model l21 --hand right --backend dex-position
python -B -m teleoperation hand vr --input-h5 visionpro_raw.h5 --hand-model l21 --hand right --backend dex-position --headless
python -B -m teleoperation robot control --robot g1 --parts hands --side right --source camera --backend dex-position
```

- [ ] 3. 在有相应录制数据时复用同一输入分别导出 NN、Vector、Position，核对 input_sha256、profile、有效时间交集、目标与命令策略一致；用既有 FK 指标分别计算 raw/command 的毫米误差及 solve_ms p50/p95、invalid 比例和限位违规数。Position 不传入仅支持 NN/Vector 的 compare_hand_methods。
- [ ] 4. 若无真实数据，仅报告 FK 合成验证；不宣称真实相机/VR 精度、帧率或手感通过。真实数据存在时记录文件路径和哈希，数值结果存 outputs，不提交原始私人录制。
- [ ] 5. 使用 verification-before-completion 技能核对实际命令输出，运行完整验证，确认现有相机和 TransHandR 回归没有失败后提交文档 `docs: document fixed-root dex position backend and validation`。

## 验证环境与命令

正式解释器为 `D:/Anaconda/envs/teleoperation/python.exe`（当前无 dex 包）；
优化器解释器为主仓库 `.venv/dex-native/Scripts/python.exe`（已确认 PositionOptimizer 可导入）。
独立工作区不复制/重装环境，设置 PYTHONPATH 使用当前工作区代码：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& D:/Anaconda/envs/teleoperation/python.exe -B -m unittest discover -s tests
& D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Scripts/python.exe -B -m unittest tests.test_dex_position tests.test_normalized_dex tests.test_hand_backend_factory tests.test_unified_hand_inputs -v
git diff --check
git status --short --branch
```

正式环境中 dex-dependent 的 skip 不能代替第二条的实际运行。共享代码改动必须执行完整 discover。
若有失败，保留日志、定位根因后修正；不能以修改期望值或删除断言代替修复。

## 本次规划验证

已核对基线接口、工厂/入口调用链、源文件 SHA256、Position/Optimizer AST 等价与运行时可导入性。
本次只新增规格和计划，不修改运行代码；不运行算法回归，不声称上述实施验收已通过。
规划文档提交前执行路径/属性核对、占位词检查和 git diff --check。
