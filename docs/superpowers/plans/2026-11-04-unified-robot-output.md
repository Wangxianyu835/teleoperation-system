# Unified Robot Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将手与臂命令汇合到统一输出，默认选择单机器人，显式开启多机器人对比，覆盖相机四目标及现有VR腕控制能力。

**Architecture:** 目标规格→手/臂命令生产→按目标组装RobotCommandFrame→RobotOutput校验、路由、唯一物理步进。复用既有求解器和安全处理，在场景边界适配三机器人与TRON2A，并增加完整命令录制/回放。

**Tech Stack:** Python dataclasses/Protocol、NumPy、PyTorch、现有dex优化器、PyBullet、Robotics Toolbox、HDF5、unittest、PowerShell；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-04-unified-robot-output.md`及AGENTS.md。此计划与已完成的无调用代码删除计划独立。

## Global Constraints

- 默认单目标：新主入口必须显式给--robot；--robots只用于两个及以上不同目标，不指定目标时拒绝启动，不默认四台或暗选G1。
- 显示范围必须匹配目标和parts/side：单目标不出现其他机器人；both显示该目标整机，arms只显示对应机械臂，hands只显示对应手；left/right只显示所选侧的独立部位，both显示双侧；多目标只展示显式选中的目标。
- 目标ID固定为h1_2、gr1_t2、g1、tron2a_l21；目标自动绑定原装手或L21，常规机器人控制不再单独选择hand-model。
- 先保持现有重定向、归一化、权重校验、IK与安全数值；统一输出不再重复滤波或求解。
- 手、臂、左右侧独立状态；未更新部分保持上次接受命令或明确的初始化参考，不用零填充假装新观测。
- 控制时间使用统一会话相对秒与递增frame_id；来源时间保留原时基，异步身体结果不能冒充当前手帧。
- 同一控制周期所有目标命令先校验、再写入，每个物理连接只推进一次预定步数；不承诺运行时电机调用跨机器人事务回滚。
- 同一相机/SDK只采集一次，多目标共享观测；每目标独立权重、IK与命令状态。
- 四目标相机全臂全手均纳入交付；VR对H1-2/GR1-T2/G1沿用现有腕位姿控制，不伪造肩肘；TRON2A VR全臂当前无已验收映射，新入口启动前明确拒绝，新增算法不属于本次输出重构。
- 多机器人对比使用一个PyBullet连接、一个物理时钟和一个GUI；TRON2A与三机器人场景支持由外部注入连接和摆位，借用连接的对象不得disconnect公共连接。
- 保留现有手部输入/窗口/结果接口、shared权重和手H5；显式TransHandR路径与本地上游来源哈希保留，不改数值算法。
- 每批修改前检查Git状态并建立annotated checkpoint，测试后提交，不推送；共享行为改动执行完整unittest discover -s tests。

## 基线与每批工作方式

源码基线440f39e8322d28d94336b96609a7583ff2f47206。计划文档检查点checkpoint-20261104-1323-plan-unified-robot-output。实施时每批重新创建检查点，不复用文档检查点。

本计划按任务顺序执行，不需要并发智能体。任务1–5形成G1可执行主干；任务6接入TRON2A和共享场景；任务7完成完整主循环；任务8录制回放；任务9迁移入口并验收。每批保持已有入口可运行，避免一次性替换所有控制算法。

每批先检查status和diff；dirty intended state先核实并提交。Git使用命令级safe.directory，不改全局配置。公共执行命令：

```powershell
$robotPlanPython = 'D:/Anaconda/envs/teleoperation/python.exe'
$env:PYTHONPATH = 'src'
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting status --short --branch
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting rev-parse HEAD
# 每项任务给出robotPlanBatch值，生成该批检查点。
$robotPlanTag = 'checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmm') + '-' + $robotPlanBatch
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting tag -a $robotPlanTag -m "Before $robotPlanBatch"
if ($LASTEXITCODE -ne 0) { throw '检查点失败，不开始修改。' }
```

每项均先添加行为测试并确认预期失败，再实施最小修改。验证通过后git diff --check，显式stage本批文件并提交，记录pre/post hash及tag；失败先定位原因，不降低运动或限位断言。完整回归公共命令：

```powershell
& $robotPlanPython -B -X utf8 -m unittest discover -s tests
```

## 新接口与数据约定

新增模块职责在任务中实现，名称固定如下：

|文件|接口与职责|
|---|---|
|robots/control_spec.py|RobotControlSpec；resolve_robot_targets(robot=None, robots=None)；load_robot_control_spec(robot_id, *, resource_overrides=None)，显式注册手型、关节分组、限位、初始参考与规格哈希|
|contracts/robot_control.py|RobotPartCommand与RobotCommandFrame；只描述接受命令，不导入设备/仿真|
|applications/robot_commands.py|RobotCommandAssembler(specs, initial)；assemble(robot_id, frame_id, timestamp, updates)，完成独立部位保持和统一时间封装|
|simulation/robot_output.py|RobotBackend协议、RobotOutput；validate→write→每physics_group一次advance；read_state分离实际状态|
|simulation/robot_backends.py|NativeRobotBackend、Tron2AL21Backend、SimulationGroup；封装现有场景的装载和电机写入|
|simulation/robot_part_scene.py|RobotPartSceneBuilder；按目标、parts、side创建原装手/机械臂可见子场景，保留必要运动学支撑，独立取景|
|applications/robot_session.py|RobotControlSession：每帧一次观测、各目标求解、组装、提交、诊断与资源关闭|
|applications/robot_control.py|run(args)、replay(args)、preflight(args)，供新CLI和旧入口边界调用|
|cli/robot_options.py|目标、部位、来源、后端与权重参数；不加载重型依赖|
|data/robot_command_recording.py|RobotCommandWriter(path, specs, metadata)；append(batch, receipt)；iter_robot_commands(path, specs)，严格机器人命令H5|

命令字段：

```python
@dataclass(frozen=True)
class RobotPartCommand:
    joint_names: tuple[str, ...]
    positions: np.ndarray  # 接受的原生rad，固定一维，只读副本
    source_timestamp: float | None
    source_timebase: str  # 明确原始传感器时基；不能与control_seconds盲减
    observed: bool
    status: str  # tracking / holding / initialized

@dataclass(frozen=True)
class RobotCommandFrame:
    robot_id: str
    spec_hash: str
    frame_id: int
    timestamp: float  # 会话相对控制秒
    parts: Mapping[str, RobotPartCommand]  # 构造时复制并用只读mapping暴露
```

parts键为left_arm/right_arm/left_hand/right_hand中的显式启用集合。未启用部位不伪装有效，交给后端hold配置。所有启用部位都输出有限、规格匹配的命令；无新更新时保持旧值且observed=False，保留旧source_timestamp。frame及parts拷贝冻结，构造后不能通过原dict/数组篡改。spec_hash包含目标ID、受控关节顺序、限位、URDF/手规格身份，不包含GUI摆位。

禁止依赖四种机器人相同数组长度。按part名称映射到关节；旧数组布局只在边界转换。用于现有SimulationEnv的完整action，按其action_joint_names打包，不能用TRON2A的48D顺序。

## Task 1：明确机器人选择、规格和CLI

**Files:** Create robots/control_spec.py、contracts/robot_control.py、cli/robot_options.py、applications/robot_control.py的预检入口，以及tests/test_robot_control_contracts.py、tests/test_robot_control_cli.py；Modify cli/registry.py、cli/schemas.py。生产文件路径以src/teleoperation为前缀，测试路径以仓库根目录为基准。

**Interfaces:** 实现上述resolve_robot_targets/load_robot_control_spec与两个命令类型。新增robot control、robot replay路由到applications.robot_control；该应用先只提供参数预检与清晰的未连接运行错误，任务5/7完成实际循环，不静默返回成功。

- [x] 建立robot-output-contracts检查点，先写参数与规格测试：缺失目标、重复目标、单复数冲突、--robots只有1个、错误机器人、错误手权重映射全部在加载设备前拒绝。
- [x] 运行失败测试：`python -m unittest tests.test_robot_control_contracts tests.test_robot_control_cli -v`。
- [x] 实现目标注册与命令校验：名称、长度、有限值、rad单位、规格hash、frame_id/时间，独立部位数组不能混用；目标数和目标手型来源唯一。示例行为：

```python
self.assertEqual(resolve_robot_targets(robot='g1'), ('g1',))
self.assertEqual(resolve_robot_targets(robots=['g1', 'tron2a_l21']), ('g1', 'tron2a_l21'))
with self.assertRaises(ValueError):
    resolve_robot_targets()
with self.assertRaises(ValueError):
    resolve_robot_targets(robots=['g1', 'g1'])
```

- [x] 不硬编码所有机器人完整action数组；加载规格对照实际URDF/HandSpec。H1/GR1/G1通过现有arm_profiles和实际关节解析；TRON2A通过TRON2A_ARM_JOINTS和L21左右HandSpec。测试四目标的14臂＋24/22/14/34手执行关节集合。
- [x] 将parts/side同时解析为控制集合和默认显示集合：both→整机，arms→所选侧机械臂，hands→所选侧原装手。无额外显示参数也必须正确；未选目标不得进入场景加载列表。
- [x] 运行相关及完整回归；提交`feat: define explicit robot targets and command frames`。

## Task 2：拆开三机器人机械臂求解与执行

**Files:** Modify simulation/arm_scene_driver.py、arm_ik.py、wrist_control.py、applications/camera_scene_arm.py；Create tests/test_arm_command_production.py；Regression tests/test_arm_following.py、test_arm_ik_multi_robot.py、test_arm_realtime_bridge.py、test_transhandr_vr.py。

**Interfaces:** 新增ArmSceneDriver.solve_each，参数与现有apply_each的目标/腕/肘参数一致，返回每机器人report与命名关节值，不发电机或step。旧apply_each作为solve_each→adapter.apply包装，签名不变。新增solve_camera_arm_frame，复用当前apply_camera_arm_frame参数，返回targets及按机器人/侧的臂值；新增WristSceneController.solve_vr，旧apply_vr保留执行包装。

- [x] 建立split-arm-production检查点，先固定同初始q、姿态、中性标定和dt的对照测试。用真实DIRECT机器人求解，观察电机调用与physics步数，不用“源码里没有apply”代替行为验证。
- [x] 测试solve_each产生相同关节值且电机写入和物理步进均为0；旧apply_each仍实际下发。IK临时resetJointState是现有算法的一部分，成功及异常后都恢复原q/qdot，不能为了“纯函数”删掉暖启动。
- [x] 将apply_each的现有数值流程原样抽出；保留目标slew、肘分支恢复、肩跟随、坐标和日志。只有电机提交后才增加applied等执行计数；求解计数与执行计数分开。
- [x] 相机和VR新增求解路径不调用电机；旧包装的回归继续通过。新流程只消费solve路径，避免旧包装与输出层双写。
- [x] 运行`python -m unittest tests.test_arm_command_production tests.test_arm_following tests.test_arm_ik_multi_robot tests.test_arm_realtime_bridge tests.test_transhandr_vr -v`及完整套件；提交`refactor: separate arm command production from motor writes`。

## Task 3：汇合手臂命令，独立保持各部位

**Files:** Create applications/robot_commands.py、tests/test_robot_command_assembly.py；Modify applications/hand_backend.py（增加以目标集合构造的边界函数，原函数保留）；复用retargeting/hand/joint_commands.py与HandJointCommand，不修改滤波数学。

**Interfaces:** RobotCommandAssembler(specs, initial)；assemble(robot_id, frame_id, timestamp, updates: dict[str, RobotPartCommand | None])→RobotCommandFrame。initial来自场景有效参考姿态，updates只能包含启用部位。各帧完整覆盖启用部位。

- [x] 建立assemble-robot-commands检查点，先测“左手丢失但左臂继续变化、右侧不受影响”“初始未检测不从全零生成动作”“恢复不重放旧数据”“source时间不被control时间改写”。
- [x] 按机器人分开维护状态；手命令直接接收JointCommandPipeline已接受/已展开值，机械臂接收现有有界求解值，不进行第二次滤波、耦合或IK。
- [x] 校验同一个target/part的关节集合与顺序；拒绝G1帧混入L21的17维手，重复关节、未知part或倒退frame/time均拒绝且不污染上一帧。
- [x] 边界转换保留HandJointCommand接口。新函数`build_target_hand_retargeters(targets, backend, checkpoints, device)`从RobotControlSpec推导hand_model，复用当前NN/dex构造；每目标缺失权重均在相机打开前失败。
- [x] 运行`python -m unittest tests.test_robot_command_assembly tests.test_normalized_retargeters tests.test_joint_commands tests.test_unified_hand_inputs -v`及完整套件；提交`feat: assemble target-specific arm and hand command frames`。

## Task 4：统一输出与单次物理调度

**Files:** Create simulation/robot_output.py、simulation/robot_backends.py、tests/test_robot_output.py；Modify applications/three_robot_scene.py只增加命令校验与写入分离的边界，不改现有原生手检查。

**Interfaces:** RobotBackend提供robot_id/spec、physics_group、validate(frame)、write(frame)、read_state()、close()；SimulationGroup.advance(steps)唯一拥有步进。RobotOutput(backends).submit(batch: dict[str, RobotCommandFrame], *, steps: int)→OutputReceipt，receipt包含frame/time、各目标提交状态和步数；read_state()与命令分离。

- [x] 建立unified-robot-dispatch检查点。先测真实DIRECT G1：完整手臂帧写入后能动，非受控关节保持；双目标共享连接时物理只推进steps而非2×steps。
- [x] submit先核验整批目标、规格、部位、时间与关节限位，任一静态错误在电机调用前失败。合法批次先write所有目标，再按唯一physics_group advance一次；禁用后端在write内部step。
- [x] 不重复使用HandExecutor再下发同一只手；NativeRobotBackend复用scene.apply_native和已拆开的arm apply。scene的非手关节hold不能覆盖由统一输出拥有的手臂；未选部位仍保持。
- [x] 运行时后端写入异常不能伪称原子回滚：标记当前提交失败，停止该组继续步进，记录已提交目标并关闭资源。正常的IK无效在命令组装阶段独立hold，不作为后端致命异常。
- [x] 运行`python -m unittest tests.test_robot_output tests.test_hand_executor tests.test_arm_command_production tests.test_native_hand_replay -v`及完整套件；提交`feat: dispatch robot frames with one physics step owner`。

## Task 5：先打通G1单目标相机与VR闭环

**Files:** Create applications/robot_control.py、applications/robot_session.py、simulation/robot_part_scene.py、tests/test_robot_control_session.py、tests/test_robot_scene_selection.py；Modify applications/camera_teleop_source.py、hand_input.py（只提取原始观测访问边界）、cli/robot_options.py；复用hand_processing.py、camera_scene_arm.py、wrist_control.py。

**Interfaces:** preflight(args)返回已验证的目标、规格、手后端配置与来源能力；RobotControlSession.next_batch(raw, frame_id, control_seconds)返回完整批次，调用者submit；run(args)只拥有一个采集源、一个控制时钟和资源栈。

- [x] 建立g1-unified-control检查点，先用tests/smplerx_fixtures.py与tests/test_unified_hand_inputs.py的现有观测构造G1 DIRECT闭环测试。测试一帧只有一次采集，两侧手/臂提交前没有电机写入。
- [x] 相机both模式一次CameraTeleopSource采集；共同UnifiedHandPipeline处理raw.hands；臂通过现有body/wrist求解。hands-only不启动SMPLer-X；arms-only不加载NN/dex手模型。控制time由会话单调相对秒产生，原始观测time留在part来源字段。
- [x] 实现RobotPartSceneBuilder.build(spec, *, parts, side, group)：G1 both只显示一台居中的G1；arms只显示所选侧机械臂，hands只显示所选侧原装手。从原始URDF挑选可见link并保留正确关节坐标、mimic与网格路径；可保留不可见固定运动学祖先，不加载其他机器人。隔离未选部位碰撞，不用镜头裁切或只冻结全身代替。固定支撑场景与整机比较FK/命令，不要求接触动力学完全相同。
- [x] 显示测试检查body→目标映射、各可见link集合和至少一帧真实渲染：G1 hands/right只有右手，arms/both只有双臂；镜头根据可见几何自动取景，单目标不使用原先三机器人横向偏移。
- [x] VR只读一次VisionProInput/ReplayInput，手走共同pipeline，臂走solve_vr。保留25矩阵、轴和原始时间，不伪造upper_body，不用相机processor处理VR。原始VR H5 EOF与无新样本按既有规则处理，重放控制dt来自录制差分。
- [x] 先预检权重/标定/资源/输出覆盖，再分配设备。错误、中断、EOF都关闭源与场景；stdout与report标明source、parts、arm模式、目标和未实测硬件边界。
- [x] 运行`python -m unittest tests.test_robot_control_session tests.test_robot_scene_selection tests.test_robot_control_cli tests.test_unified_hand_inputs tests.test_body_capture_replay -v`及完整套件；提交`feat: run G1 through the unified robot output path`。

## Task 6：接入TRON2A及四目标共享场景

**Files:** Modify applications/dual.py、simulation/tron2a_world.py、applications/three_robot_scene.py、simulation/robot_backends.py、simulation/robot_part_scene.py、applications/robot_session.py；Create tests/test_robot_comparison.py；Extend tests/test_robot_scene_selection.py；Regression tests/test_mediapipe_tron2a.py、test_arm_ik_multi_robot.py。

**Interfaces:** DualTeleoperator增加`produce_parts(observation, *, control_seconds)`，复用当前手/臂控制后返回通用part命令；现有update保持返回Tron2AL21Command。Tron2AL21Backend完成通用part↔原专用命令映射。场景构造新增可选外部client_id、owns_client、base_pose；默认参数保持已有独立场景行为。

- [x] 建立four-target-comparison检查点，先比较固定观测/时钟下TRON2A通用命令与旧48D各分组数值相等；不通过改写专用顺序“统一”RobotLoader布局。
- [x] TRON2A从calibration获取safe_q、rotation、workspace和安装姿态，正式入口继续使用smplerx模式和原安全控制。初始化L21姿态必须以实际参考命令与spec校验，不将其他手型权重用于L21。
- [x] 统一场景只建一个PyBullet连接和地面，负责重力、dt、布局、GUI与最终disconnect。三机器人和TRON2A子场景使用注入连接，不重设全局物理、不再各自step/关闭共享连接；L21多body、约束及手部安装仍由TRON2A后端拥有。
- [x] 对比摆位通过机器人base_pose实现；验证IK在各机器人基座坐标下的目标不因横向摆位改变，TRON2A相机标定不被改成带GUI偏移的新标定。每个目标只允许写自己body的关节。
- [x] 将部位显示扩展到H1-2、GR1-T2、TRON2A＋L21。TRON2A arms模式不附加L21可见body，hands模式仅装载所选L21手且不要求身体标定；原full/simplified选项属于几何精度，不能冒充arms/hands显示范围。显式多目标parts=hands只显示选中目标的手，不生成整机对比墙。
- [x] 测试单目标仅加载1台；显式双目标及四目标共享同一输入frame_id/控制time；关闭一个借用场景不影响其他目标；共同advance次数不随目标数增加。保留每目标独立状态，单臂观测失效只影响对应part。
- [x] 运行`python -m unittest tests.test_robot_comparison tests.test_robot_scene_selection tests.test_mediapipe_tron2a tests.test_arm_ik_multi_robot tests.test_robot_output -v`及完整套件；提交`feat: support explicit four-target simulation comparison`。

## Task 7：完成目标选择与跨入口运行策略

**Files:** Modify cli/robot_options.py、applications/robot_control.py、robot_session.py、simulation/realtime_teleop.py、applications/mediapipe_tron2a.py、dual_realtime.py；Test tests/test_robot_control_cli.py、test_robot_control_session.py、test_cli_integration.py。

**Interfaces:** 新主入口robot control/replay正式可用；目标选项互斥，--robots至少两个。旧三机器人模块`--robot`重复形式在边界转换目标集合；不传目标时改为明确提示选择，取消“默认三台”的行为。现有dual camera显式对应tron2a_l21，保留已有参数及arm-only默认。

- [x] 建立explicit-target-routing检查点，测试以下实际参数行为（不是只比较源码字符串）：

```text
robot control --robot g1 --source camera --backend dex                => 仅G1
robot control --robots g1 h1_2 --source camera --backend dex           => 两个目标
robot control --source camera --backend dex                          => 参数错误
robot control --robot g1 --robots g1 h1_2 --backend dex                => 参数错误
robot control --robots g1 --backend dex                              => 参数错误
robot control --robot tron2a_l21 --source visionpro --ip unused --backend dex --parts both
                                                                      => 能力预检失败，不开设备
```

- [x] 旧相机入口把参数转换后调用共同session/输出，图像HUD/诊断作为观察器保留，不再私自电机下发或步进。原hands-only接口、独立L21、训练与显式TransHandR不改输入/结果契约；需写整机时在其执行边界接RobotOutput的hands/wrist模式。
- [x] NN每目标通过--hand-checkpoint MODEL PATH传一份shared；dex拒绝NN权重；无arms/hands相应部分不加载对应模型。TRON2A目标在camera arms/both时要求--tron2a-calibration；没选择TRON2A时拒绝其专属标定参数，避免静默忽略。
- [x] 预检和场景创建统一采用parts/side：hand-only的tron2a_l21不需要TRON2A URDF/身体标定/SMPLer-X；arm-only不要求手网格或手权重；启用的可见部分缺少真实资源时清晰报错，不悄悄显示其他机器人或简化代理手。
- [x] 启动摘要列出目标、source、parts、相机body/VR wrist、权重和物理周期；不提供实体机器人发送后端。
- [x] 运行CLI、统一会话、相机/VR旧接口回归与完整套件；提交`feat: make single robot selection the default workflow`。

## Task 8：统一完整命令录制与直接回放

**Files:** Create data/robot_command_recording.py、tests/test_robot_command_recording.py、tests/test_robot_command_replay.py；Modify applications/robot_control.py、robot_session.py、cli/robot_options.py；保留data/hand_joint_recording.py、data/command_h5.py现有格式。

**Interfaces:** 新schema=robot_command_results_v1。RobotCommandWriter.append(batch, receipt)写入已组装命令与提交结果；iter_robot_commands(path,specs)逐帧返回选定目标的RobotCommandFrame。新robot replay先完整校验再打开GUI；直接submit命令，不加载NN/IK/滤波。

- [x] 建立robot-record-replay检查点，先测四种维数/关节布局的真实命令往返、source时基/holding状态、不同步身体来源时间、只选择录制中一个目标回放。
- [x] H5根保存frame_ids/control_timestamps、schema、rad单位、target列表和录制状态；robots/<id>保存spec_hash、URDF身份、enabled_parts、每部位joint_names、positions、source_timestamp/timebase、observed/status。执行receipt单独保存实际提交状态、physics_dt和各组step数；可选actual状态不能覆盖命令值。
- [x] 保存目标、parts/side和整机/机械臂/手部场景模式；回放只加载选中目标和对应可见部位。关节规范身份与仅影响取景的摆位分开，不因隐藏身体修改原始HandSpec哈希。
- [x] writer拒绝覆盖、规格错配、关节缺失、NaN/Inf、时间/frame倒退和不完整批次；临时文件加完成标记后发布，异常保留明确failed/incomplete状态，不宣告完整成功。旧格式不改metadata冒充新schema，不增加泛化自动转换器。
- [x] 真实DIRECT回放时对比电机提交值/步进数；patched NN/IK/滤波在调用时抛异常，回放仍成功，证明没有重新求解。GUI按记录控制时间间隔等待；物理按已记录的physics_dt与step数推进，不用CPU时间重算，也不以时间差重新四舍五入导致累计漂移。
- [x] 运行`python -m unittest tests.test_robot_command_recording tests.test_robot_command_replay tests.test_hand_control_h5 tests.test_native_hand_replay -v`及完整套件；提交`feat: record and replay complete robot command frames`。

## Task 9：清除重复提交路径、更新文档并最终验收

**Files:** Modify README.md、docs/current-interfaces.md、docs/unified-hand-input.md及本计划/spec；核查applications/hand_control.py、simulation/realtime_teleop.py、applications/mediapipe_tron2a.py、camera_scene_arm.py、simulation/wrist_control.py的正式路径。

**Interfaces:** 所有已迁移整机工作流通过RobotOutput提交；旧单手H5、训练、TransHandR上游和任务action接口保持其原语义。SimulationEnv任务基准不在本次自动接通遥操作，也不宣称任务成绩。

- [x] 建立verify-robot-output检查点，删除迁移后确实无调用的下发分支/闲置import；保留旧公开包装只作为共同实现的转发，不复制求解循环。逐项列出现存未迁移诊断入口及保留原因。
- [x] 更新新主入口、显式目标选择、NN权重、TRON2A标定、VR能力矩阵、录制schema及旧入口行为改变；删除“默认同时三台”的现状说明，补单目标与显式四目标流程图。
- [x] 运行所有新测试、完整unittest；主环境dex跳过用例在独立环境补跑：

```powershell
& $robotPlanPython -B -X utf8 -m unittest discover -s tests
. ./devtools/dex_native_env.ps1
& $dexPython -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters tests.test_unified_hand_inputs tests.test_robot_control_session tests.test_robot_comparison -v
```

- [x] 用同一段合成SMPL-X/Hands观测验证G1单目标、H1/GR1/G1三目标、四目标含TRON2A的真实DIRECT运动与保持；VR原始矩阵H5验证三个支持的wrist目标及NN/dex。记录命令前后差异、实际关节运动、非控制漂移、step数和资源关闭；不只验help或mock调用。
- [x] 增加显示矩阵验收：四目标分别both/arms/hands、left/right/both，以及显式双目标hands对比；检查可见link身份、无未选机器人、单目标居中和真实渲染图。多body不能被误计为多目标，必要的不可见支撑不能露出身体几何或造成未选部位碰撞。
- [x] 若触碰上游VR数值/共享预处理，按AGENTS重新验证实际D:/2026/code/TransHandR/TransHandR来源、SHA-256及等价；未触碰则明确历史数值报告未重跑。真实相机/头显/实体硬件不在场时只报告软件验证。
- [x] 记录各批hash/tag、主/独立环境pass/skip/fail、能力限制和回退命令，提交`docs: verify unified robot output and workflow boundaries`，不push。

## 验收底线与回退

不得以“输出字典结构一致”代替运动验收；不得只读关节状态反推接受命令；不得将失效输入标为tracking；不得在提交和回放再次滤波。多目标共享source只采一次、共享client只step一次的行为必须有可观察证据。

每批独立tag。检查该批diff后可按路径restore，或建立恢复分支，保留其他已验证批次；不reset --hard、不删tag、不改共享历史。

```powershell
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting diff $robotPlanTag -- src tests docs README.md
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting switch -c rollback-robot-output $robotPlanTag
```

## 执行记录

- Task 1：pre abcf428a352c2ca605e8dc175b87563495d2c441；tag checkpoint-20261104-1341-robot-output-contracts；相关14项通过；完整564项，561通过、3项dex依赖跳过。已复验CLI轻量导入修复。

- Task 2：pre ee93a1dbec9005c76307c860a5fbecca5833c60a；tag checkpoint-20261104-1348-split-arm-production；相关172项通过；完整565项，562通过、3项dex依赖跳过。真实G1验证求解无电机/步进且q/qdot恢复，旧执行包装保留。

- Task 3：pre 4eddbf8c38163a0d037932762f88d0739603a09e；tag checkpoint-20261104-1352-assemble-robot-commands；相关21项（2项跳过）；完整567项，564通过、3项dex跳过；独立保持、来源时基、错误更新不污染状态已覆盖。

- Task 4：pre e58261d9c39812c6c9d0e3f64400e7e3f430dd1b；tag checkpoint-20261104-1357-unified-robot-dispatch；相关7项通过；完整568项，565通过、3项dex跳过；真实双目标组8步只推进8步，静态错误无电机写入，实际关节运动已验证。

- Task 5：pre 2ca3591b1524ae5437aad9218c7f5d5307f018de；tag checkpoint-20261104-1401-g1-unified-control；相关21项（1跳过），最终新增核心3项通过；完整571项，568通过、3项dex跳过。G1原装整机/右臂/右手真实渲染已查看；录制身体H5经注册CLI完成4帧，无相机/模型推理。TRON2A原装STL缺失已确认，后续显式简化几何用于其控制验证，不冒充原装网格。

- Task 6：pre c1385e6f24ab2cdc727b012134badf36a659c567；tag checkpoint-20261104-1411-four-target-comparison；相关113项通过；完整575项，572通过、3项dex跳过。四目标同连接/8步单次推进、原装三目标部位渲染、L21独立手、借用TRON2A连接关闭不影响公共场景已覆盖。新增显式--geometry simplified只用于TRON2A原网格缺失时的可见控制验证，full仍严格报缺资源。

- Task 7：pre 6fff4b294177bc9fa7b5c7f61505e39dd351f5aa；tag checkpoint-20261104-1417-explicit-target-routing；新增CLI预检、旧相机/VR手腕和TRON2A相机输出收口。相关100项中TRON65项及其余回归已通过，修正显式目标测试后完整577项，574通过、3项dex跳过；保留旧采集/诊断参数和单手H5边界。

- Task 8：pre 11a0461f329a7eb807b5aa4cc145ef9615a2feb7；tag checkpoint-20261104-1440-robot-record-replay；相关14项通过；完整582项，579通过、3项dex跳过。四布局H5往返、异步来源时基/holding、失败保留pending、拒绝覆盖/篡改、四目标回放与单目标抽取均覆盖。真实G1录制与回放逐帧实际q误差≤1e-12，步数完全一致；禁止构造IK/session/filter后仍能回放。TRON2A回放只用录制的safe_q/安装姿态，不启动人体映射。

- Task 9：pre 2d77fec09e39d6f0d82c52d7f4fe97504cce963a；tag checkpoint-20261104-1450-verify-robot-output；最终验收新增4项全部通过，主环境完整586项（583通过、3项dex依赖跳过），独立dex环境26项全部通过。四目标部位/侧面显示矩阵、VR手腕来源时间与stale holding、三目标VR NN/dex、统一录制回放、旧入口输出收口和文档流程图均已验证。真实相机、真实Vision Pro和实体机器人仍未在本机执行。

后续整理说明（2026-11-06）：新代码归并计划退出本历史计划曾描述、但最终没有运行时消费者的produce_parts和iter_robot_commands包装。实际统一控制仍走RobotControlSession，回放走保留receipt的RobotRecording.iter_frames；原实施记录保留。
