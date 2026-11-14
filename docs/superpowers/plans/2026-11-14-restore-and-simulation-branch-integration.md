# 旧机械臂修正与仿真任务分支合并实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本次只编写计划；实施时默认在当前会话逐项执行，未经用户选择不分派子代理。

**Goal:** 将 `restore/arm-1aaabb2` 的有用机械臂修正与 `feat/arm-smplerx-body` 的任务评估改进整合进 `wrist-input-arm-follow`，保留当前正式控制契约，并使两条来源分支最终可按祖先关系安全清理。

**Architecture:** 先合并仿真任务，再合并机械臂与启动器，每条来源使用一个真正的双父合并提交。逐项移植独有行为，在现有模块中解决冲突；正式相机控制继续使用共同 SMPL/PINK 路线，诊断及 VR 的旧 IK 适配当前接口。所有测试通过后再快进交付目标分支。

**Tech Stack:** Python 3.10、NumPy、PyBullet、Pinocchio、SMPL-X、PINK、unittest、PowerShell、Git。

**Spec:** 用户于 2026-11-14 要求规划两条分支合入当前分支；本文“范围与兼容约束”为实施规格。另须阅读根目录 `AGENTS.md`、`docs/current-teleoperation-usage-report-2026-11-13.md` 和 `docs/unified-smpl-pink.md`。

## 0. 状态与固定基线

- 计划基线/目标：`wrist-input-arm-follow`，`9326ed006569c718b672cc3ff9874310a6553d4d`。
- 当前编写计划的工作树：`D:/2026/code/PyCharmMiscProject/teleoperation-smpl-pink`，分支 `codex/unified-smpl-pink`；编写前与目标同提交。
- 仿真来源：`41287ce789db37415917529df2b7a52717e5cfff`，7 个独有提交；共同祖先 `bd3600b84d50ca1edc1e8f6939fb97441976e6c8`。
- 机械臂来源：`d0be525ea8f9cdd110b700a88b0f91d9e54362ad`，9 个独有提交；共同祖先 `1aaabb2f3ea8a30070d6c563c3bcd0509358af96`。
- 当前正式版本之前完整验证：690 tests，OK，skipped=2；实施时必须重跑，不能将该结果作为合并后的结果。
- GitHub `dex-retargeting@05581fb` 已按用户指令删除。本地 `D:/2026/code/dex_retargeting` 的源码、分支和未跟踪文件保留。本计划不处理该目录。
- 本计划未开始两条来源分支的真实合并。旧分支文档中的验收数字属于历史记录。

## 1. 范围与兼容约束

1. 交付目标为 `wrist-input-arm-follow`；不要误将旧来源分支设为开发基线。
2. 保留当前 `robot control` / `robot replay`、统一手部 NN/dex、正式相机 SMPL/PINK、VR 腕部 IK 的分工。
3. 保留事务式接受状态、输出失败回滚、失跟踪保持、关节限位回收和隐藏临时求解姿态。
4. 不用旧版整个 `arm_ik.py`、`realtime_teleop.py`、启动器或测试覆盖当前文件；逐项保留来源行为和当前保护。
5. `SimulationEnv.step()` 仍返回 `(obs, success, done, elapsed)`；`elapsed` 和录制观察者传入的时间仍为墙钟秒，评测时间另取仿真秒。
6. `MetricsTracker.record(task_name, success, completion_time)` 的三参数调用继续可用；新增 `wall_time=None`，保留旧 summary 键。
7. `--robot` 必须显式选择、NN 权重、worker 路径、报告路径和错误退出码继续透传；正式 `robot control` 默认值不随旧诊断启动器改变。
8. 不修改 TransHandR 数值预处理和权重元数据，不修改人体/手部数据契约，不自动重标旧拟合 hash。
9. 任务名称、注册数量及难度表保持一致。来源对 `drawerbook`、`rotatehearth` 的简化属于任务定义变化，必须显式处理，见任务 2。
10. 本轮不混入此前审查的 VR 手指过期、旧回放计时、单有效帧统计三个独立缺陷；另行修复并验证。
11. 每个代码改动批次遵循 `AGENTS.md` 的干净工作树、检查点和提交约定。合并开始后，解决冲突属于同一个改动批次；在该批次结束前不开始下一条分支。
12. 实施需重新 fetch 并记录来源 SHA。来源变化时重新核对增量，不将更新后的分支名冒充本文固定版本。

## 2. 合并顺序与文件清单

先合仿真，因为其主要修改在任务和评估层，便于建立独立验收结果；之后合机械臂，最后统一回归。

| 批次 | 文件 | 处理 |
|---|---|---|
| 仿真场景 | `src/teleoperation/simulation/tasks/base.py`、`tools.py`、`bimanual.py`、`sequences.py` | 移植支撑面吸附、相对初始状态判据、容器、滑轨、铰链；保留当前动作映射 |
| 评估 | `simulation/environment.py`、`simulation/randomization.py`、`evaluation/metrics.py`、`applications/simulation.py` | 接入仿真时钟、连续窗口、失败统计；保持录制观察者接口 |
| 工具 | `devtools/audit_task_criteria.py`、`devtools/preflight.ps1`、`devtools/verify_realtime_arm_bridge.py` | 引入体检；A/B 开关适配当前诊断接口 |
| 仿真测试 | `tests/test_simulation_tasks.py`、`tests/test_application_entrypoints.py`、`tests/test_arm_realtime_bridge.py` | 保留现有测试并增量移植来源用例 |
| 中性姿态 | `robots/specification.py`、`robots/arm_spec.py`、`simulation/robot_loader.py`、`simulation/robot_part_scene.py` | 同一机器人参考值供 PyBullet、Pinocchio 和场景使用，避免只在 IK 构造器里偷偷改姿态 |
| 旧 IK | `simulation/arm_ik.py`、`simulation/realtime_teleop.py` | 保留当前求解保护；移植必要的姿态诊断，报告经核验的修正 |
| 启动器 | `devtools/windows_runtime.py`、`run_realtime_arm_windows.py`、`run_realtime_arm_elbow_follow_windows.py`、`run_realtime_windows.py` | 共享参数适配和窗口布局；保留旧脚本入口兼容 |
| 机械臂测试 | `tests/test_arm_ik_multi_robot.py`、`test_windows_launchers.py`、`test_gr1_arm_diagnosis.py`、`test_arm_display_stability.py`、`test_arm_fit.py` | 验证参考一致性、旧拟合拒绝、求解恢复和参数优先级 |
| 文档 | `docs/SIM_EVALUATION.md`、当前使用报告、`README.md`、新增 `docs/branch-integration-2026-11-14.md` | 当前入口为准；记录逐提交处置和已知限制 |

上述目录缩写均相对于 `src/teleoperation/`，测试相对于仓库根。

合并预演记录：仿真来源与当前基线只有 README 内容冲突，但自动合并仍会带入旧文档和需要审查的代码；机械臂来源会在启动器、`arm_ik.py`、`realtime_teleop.py`、测试及已删除文档处冲突。第一次真实合并之后，第二次冲突集合须重新计算。

## 3. 实施环境和命令

在隔离的集成工作树执行；用户现有工作树、解释器环境和未跟踪文件不移动、不删除。若本机策略限制共享 Git 元数据写入，使用正常的授权执行机制，不修改全局 `safe.directory`。

```powershell
$repo = 'D:/2026/code/PyCharmMiscProject/teleoperation-smpl-pink'
$integration = 'D:/2026/code/PyCharmMiscProject/teleoperation-branch-integration'
git -C $repo fetch origin
git -C $repo worktree add -b codex/integrate-restore-and-simulation $integration wrist-input-arm-follow
Set-Location $integration
$py = 'D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Scripts/python.exe'
$deps = 'D:/2026/code/teleoperation-system-retargeting/outputs/acceptance/teleopbench-gr1-20261111/dependencies'
$env:PYTHONPATH = "src;.;$deps"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
& $py -B -c "import pinocchio, pink, pybullet, torch; print(pinocchio.__version__)"
& $py -B -m unittest discover -s tests -q
```

缓存和日志写入集成工作树的 `outputs/branch-integration/`。需要真实 SMPL-X 运行时，按现有操作脚本添加已存在的 `outputs/unified-arm-runtime` 路径。不重装或升级依赖来掩盖代码回归。

## Task 1: 开始仿真合并并记录所有来源提交

**Files:** 新增 `docs/branch-integration-2026-11-14.md`；其余文件按任务 2、3 处理。

**Interfaces:** 消费固定来源 SHA 和当前基线；产出包含 7 个来源提交逐项处置的合并记录。

- [ ] 核对工作树干净，创建带实际时间的 annotated checkpoint。
- [ ] 运行 `git merge --no-ff --no-commit 41287ce789db37415917529df2b7a52717e5cfff`。冲突时继续解决当前合并，不能提交只标祖先而不移植行为的 `ours` 合并。
- [ ] README 保留当前操作路线，新增仿真评估说明链接。对自动恢复的 `docs/CONVENTIONS.md`、`docs/RETARGETING_PIPELINE.md`、`docs/multi-robot-arm.md` 保持当前删除状态，将有用说明迁入现有文档。不要让旧约定覆盖 `AGENTS.md`。
- [ ] 在记录中列出 `759d655`、`8ec3c21`、`4d78359`、`5c1682c`、`fca4d15`、`a7318cf`、`41287ce`：分别标明移植位置、替代实现或不采用的具体理由。任务 2、3 完成前不提交这个合并。

## Task 2: 场景和任务定义回归

**Files:** 上表中的 `tasks/base.py`、`tools.py`、`bimanual.py`、`sequences.py`、`simulation/randomization.py`、`tests/test_simulation_tasks.py`、`docs/SIM_EVALUATION.md`。

**Interfaces:** `BaseTask.reset()` 负责建物体、摆位、吸附和记录初始状态；`check_success()` 返回当前物理判据；`hold_steps(dt)` 将持续时间转换为步数。`DomainRandomizer.randomize(..., skip_names=())` 保留旧两参数调用。

- [ ] 先移植来源场景测试，在当前目标代码上单独运行，确认缺少相对判据、滑轨/铰链或支撑面处理导致的真实失败。
- [ ] 移植 `capture_initial_state`、`rise`、`displacement`、`tilt_deg`、`yaw_change_deg`、`is_inside`、`snap_to_supports` 及机构构件。固定当前关节名到动作索引的映射。
- [ ] 域随机化跳过关节体；随机化后重新吸附、记录初始状态；连续 reset 清理旧的几何和初始状态缓存。
- [ ] 默认采用来源分支的简化任务定义，但在 `docs/SIM_EVALUATION.md` 和集成记录逐项注明：`drawerbook` 不再要求后续关抽屉，`rotatehearth` 不再要求按下。新结果不得与原始完整目标直接混为同一评测结果。用户若要求保持原始目标，则先实现移动托盘内腔/按压关节并增加对应判据测试，此任务不得以删目标代替完成。
- [ ] 加入以下可观测断言（示意放入 `SimulationTaskSceneTests`，复用来源的 `_make_task`）：

```python
def test_hold_window_depends_on_physics_dt(self):
    task = self._make_task('pushcube')
    self.assertEqual(task.hold_steps(1 / 240), 24)
    self.assertEqual(task.hold_steps(1 / 30), 3)

def test_lift_threshold_is_relative_to_reset_pose(self):
    task = self._make_task('liftmug')
    pos, orn = task.get_object_pose('lid')
    self.assertFalse(task.check_success())
    pos[2] += .06
    p.resetBasePositionAndOrientation(task.objects['lid'], pos, orn,
                                    physicsClientId=self.client)
    self.assertTrue(task.check_success())
```

- [ ] 运行 `& $py -B -m unittest tests.test_simulation_tasks -q`；30 个注册任务均检查初始不成功、静置漂移不超过 20 mm、静置过程不自动成功和关节可动性。保留来源已知的臂展不足限制，本次不顺带整体搬动工作台。

## Task 3: 仿真时钟、报表和体检门禁

**Files:** `simulation/environment.py`、`evaluation/metrics.py`、`applications/simulation.py`、`devtools/audit_task_criteria.py`、`devtools/preflight.ps1`、`devtools/verify_realtime_arm_bridge.py`、相关三份测试。

**Interfaces:** 保留 `step()`、`run_episode()` 返回值；`MetricsTracker.record(..., wall_time=None)` 以 `completion_time` 接收仿真秒；`get_avg_wall_time()` 和 `get_avg_episode_time()` 为附加统计。

- [ ] 新增测试后运行红灯：单步物理后 `env.sim_time == env.dt`，成功需连续 `hold_steps`，超时按仿真秒，墙钟仍传入录制观察者。
- [ ] 移植来源统计逻辑；`__init__` 初始化 `dt`、`sim_time`、成功窗口；`reset` 清零；步数预算耗尽也 `_finalize_episode(False, elapsed)`，一次 episode 只能记一条试验。
- [ ] 测试明确模拟连续调用终止状态时不会重复记分，`max_steps=0` 也能有明确失败记录；保留 action_dim 按机器人变化和一次 reset 的现有测试。
- [ ] 添加统计兼容用例：

```python
def test_metrics_keep_legacy_record_call_and_separate_wall_time(self):
    tracker = MetricsTracker()
    tracker.record('pushcube', True, 12., wall_time=3.)
    tracker.record('pushcube', False, 40.)
    self.assertEqual(tracker.get_success_rate('pushcube'), .5)
    self.assertEqual(tracker.get_avg_time('pushcube'), 12.)
    self.assertEqual(tracker.get_avg_wall_time('pushcube'), 3.)
    self.assertEqual(tracker.get_summary()['pushcube']['num_trials'], 2)
```

- [ ] 体检脚本复用当前 registry 和类路径；preflight 仅增量接入可运行命令，不恢复旧目录/旧已移除 CLI。依赖缺失要明确报错或标明未执行，不能计作体检通过。
- [ ] 来源 `build_bridge`/`elbow_plan` 的 A/B 参数适配当前诊断工具，保持显式选择和报告字段；不得让正式 PINK 控制退回旧 chain/posture 算法。
- [ ] 执行 `& $py -B devtools/audit_task_criteria.py`，保留 JSON 与退出码；臂展不足明确记录为限制，不能把 30 项场景体检通过写成机器人可完成 30 项任务。
- [ ] 执行 `& $py -B -m unittest tests.test_simulation_tasks tests.test_application_entrypoints tests.test_arm_realtime_bridge -q`，然后运行全套 discover。
- [ ] `git diff --check`、`git diff --name-only --diff-filter=U` 均无问题后完成第一个 merge commit，记录测试数量、跳过原因及任务语义变更。

## Task 4: 开始 restore 合并，统一 GR1 参考姿态

**Files:** `robots/specification.py`、`robots/arm_spec.py`、`simulation/robot_loader.py`、`simulation/robot_part_scene.py`、`simulation/arm_ik.py`、相关机械臂测试。

**Interfaces:** 在规格模块增加 `arm_neutral_positions(robot_id) -> dict[str, float]`，返回新字典；机器人加载、部分场景与 `load_arm_spec` 从同一函数取得关节参考值。继续保留现有 `ARM_NEUTRAL` 导出，避免破坏已有导入。

- [ ] 第一批次提交且工作树干净后创建第二个 checkpoint，运行 `git merge --no-ff --no-commit d0be525ea8f9cdd110b700a88b0f91d9e54362ad`。
- [ ] 移植来源 `ElbowPoseSideTests` 的几何断言：官方 H1-2/GR1/G1 左右臂的前臂朝向、肩-腕连线相对肘侧、关节限位、弯角差 <= 5 度和幂等。先对当前加载器测量，再变更配置，不只看关节角正负。
- [ ] 记录官方 URDF 哈希与测量值。预期是 GR1 两侧 `elbow_pitch_joint` 从 +1.2 改为 -1.2；H1-2/G1 和 TRON2A 原有规则保持。自定义 URDF 不套用未经验证的翻转假设。
- [ ] 在同一参考值函数中实现经验证的 GR1 覆盖；三个加载位置统一消费，不能仅在 `PyBulletArmIK.__init__` 中重设关节而让 PINK 的 `ArmSpec.reference`、场景 hold 和拟合标识保持旧值。
- [ ] 测试参考函数不污染全局字典：

```python
def test_gr1_reference_is_independent_and_model_specific(self):
    from teleoperation.robots.specification import arm_neutral_positions
    gr1 = arm_neutral_positions('gr1_t2')
    self.assertAlmostEqual(gr1['left_elbow_pitch_joint'], -1.2)
    self.assertAlmostEqual(gr1['right_elbow_pitch_joint'], -1.2)
    gr1['left_elbow_pitch_joint'] = 0.
    self.assertAlmostEqual(arm_neutral_positions('gr1_t2')['left_elbow_pitch_joint'], -1.2)
    self.assertAlmostEqual(arm_neutral_positions('h1_2')['left_elbow_pitch_joint'], 1.2)
```

- [ ] 确认 `load_arm_spec` 的 Pinocchio reference 与场景实际关节值一致；在 `limits`、`hold`、`neutral_orientation` 采样前完成加载姿态初始化。PINK/旧 IK 都以一致参考运行，保留求解后位置和速度恢复。
- [ ] `ArmSpec` 身份包含 reference，GR1 拟合文件可能因此失效。测试旧 hash 被明确拒绝；用原人体模型/输入在新输出目录重做受影响拟合，比较残差、缩放与参考，不改写旧 JSON hash。H1-2/G1/TRON2A hash 未变化时不重做。
- [ ] 更新 `test_arm_fit.py`、`test_gr1_arm_diagnosis.py`、`test_robot_scene_selection.py` 的实际期望；保留既有失跟踪不求解、隐藏临时状态、失败输出不提交的断言。

## Task 5: restore 启动器与窗口行为

**Files:** `devtools/windows_runtime.py`、三个现有 launcher、`tests/test_windows_launchers.py`、`simulation/realtime_teleop.py`。

**Interfaces:** 将 `apply_elbow_preset(argv) -> list[str]` 放入共享 runtime，仅诊断手臂启动器调用；公开参数 `--arm-elbow-follow` 优先，支持旧快捷参数 `--elbow-follow` 和 `--off`。原 `run_realtime_arm_elbow_follow_windows.py` 路径仍可调用，委托同一个实现。

- [ ] 先扩展现有 launcher 测试，再移植参数优先级、帮助提示和 operator 默认；底层 CLI 默认及正式 `robot control` 行为保持当前约定。
- [ ] 用相同 `invoke()` 测试：主手臂 launcher 默认 operator、`--off`/`--elbow-follow off` 得到 off、显式 `--arm-elbow-follow` 取胜、NN checkpoint/worker/report/side 原样传递、缺 robot 仍先失败。
- [ ] 保留现有 `relaunch_with_canonical_python(..., script_path=...)`，不要恢复来源的独立解释器重试代码。
- [ ] 来源 Win32 布局逻辑移入共享 runtime；只有 GUI 且预览启用时才启动有限时长线程。headless、no-preview、布局关闭环境变量都不得枚举/移动窗口。用 mock 测布局和线程退出，不开启设备。
- [ ] 保留当前诊断统计，追加经实际验证的参考姿态说明；日志不得把源分支历史测量当成本次测量。
- [ ] 将来源 9 个提交 `b349acc`、`1c3ad33`、`d69b49d`、`69e4def`、`9bcc3fc`、`ce1947c`、`5810fce`、`e69950d`、`d0be525` 的处置写入集成记录；旧文档说明迁入现有使用报告，过期测试基线不恢复。
- [ ] 执行 `& $py -B -m unittest tests.test_windows_launchers tests.test_arm_ik_multi_robot tests.test_gr1_arm_diagnosis tests.test_arm_display_stability tests.test_arm_fit tests.test_unified_pink_ik tests.test_robot_control_session -q`，随后完整 discover。
- [ ] 在有桌面设备时另行运行三机器人 GUI/相机短会话，观察窗口和肘姿势。没有设备时明确保留人工验收项；不能用合成输入宣称真人手感改善。
- [ ] 冲突清零且所有代码测试通过后完成第二个 merge commit。

## Task 6: 交付、历史包含关系与恢复

**Files:** `docs/branch-integration-2026-11-14.md`、README 和当前使用报告；只记录实际执行结果。

- [ ] 在最终集成版本重新运行完整 suite，日志写入新目录；本轮新增测试必须实际执行，不能因为缺正确解释器而整批 skip。
- [ ] 核对两条来源确实被合并：

```powershell
git merge-base --is-ancestor 41287ce789db37415917529df2b7a52717e5cfff HEAD
git merge-base --is-ancestor d0be525ea8f9cdd110b700a88b0f91d9e54362ad HEAD
git diff --check
git status --short
```

- [ ] 从目标工作树执行 `git merge --ff-only codex/integrate-restore-and-simulation`。目标若已有新提交，重新整合和验证，不强推覆盖。
- [ ] 推送前核对用户本轮授权。只请求计划不构成这两条分支实际合并或发布的授权；获准实施后完成合并，发布按当时授权处理。
- [ ] 列出最终提交、两阶段 checkpoint、参考姿态/拟合迁移、任务简化定义、全部测试和未执行设备项。删除来源分支须在成功交付之后由用户决定。
- [ ] 回滚优先 `git revert -m 1 <merge_sha>`；用 checkpoint 建回滚分支或恢复选定文件。不要 reset --hard、批量清理 outputs 或删除用户已有 worktree。

## 完成标准

1. 两条来源完整进入提交历史，所有独有功能均有逐提交处置说明。
2. 30 项场景体检、统计/录制契约、跨机器人参考姿态和当前正式控制回归均通过。
3. 新旧拟合不冒充兼容；受影响资源可在新目录重建。
4. 来源任务简化、臂展不足和未进行真人设备验证均在文档中准确说明。
5. 没有恢复已淘汰的目录结构或用旧实现覆盖当前 SMPL/PINK 与接受命令流程。
