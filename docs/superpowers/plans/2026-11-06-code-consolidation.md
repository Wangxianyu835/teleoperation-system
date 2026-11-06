# 当前代码清理与合并 Implementation Plan

> **For agentic workers:** 逐任务使用可用的执行计划工作流；本项目由当前会话顺序执行，不启用子智能体。用户已批准执行，逐批记录结果。

**Goal:** 删除已确认无消费者的片段，将同职责重复实现归并到一个维护位置，保留现有功能与命令。

**Architecture:** 先做无行为变化的删除和薄文件归并，再收口后端构造、相机预览、初始化辅助和开发工具。保留各入口作为参数/来源适配器，输入处理、算法、命令和仿真职责不混合。

**Tech Stack:** Python 3.10+、unittest、NumPy、PyTorch、HDF5、PyBullet、MediaPipe、PowerShell；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-11-06-code-consolidation.md`。

## Global Constraints

- 先读spec和AGENTS；本计划基线为782712f，不复用2026-11-04旧清理清单。
- 28个注册CLI命令保持语义；公开输入/窗口/结果及录制格式保持兼容。
- 不改变手臂/手部数值算法或TransHandR上游流程，不增加泛化自动格式转换。
- 各目标选择、显示集合、失效holding、单次步进和无输入机械臂不做IK的修复保留。
- 每批先核对dirty状态并建立annotated checkpoint，相关回归通过后提交；共享行为运行全套；不push。
- 不清理用户数据、缓存、依赖、机器人资产；不通过删除有价值断言让测试通过。
- 不写“符号不存在”的永久测试；纯删除用现有行为测试和搜索证明，新公共实现测试实际行为。

## 1. 审查结果与处置清单

### 1.1 可直接删除的片段（15个）

以下对象在src/tests/devtools符号、字符串与动态调用复查中没有消费者。执行前再对当前HEAD复核。

|文件|删除对象|保留内容|
|---|---|---|
|applications/camera_record.py|default_model_asset()；另删MODEL_ASSET_CANDIDATES常量|独立录制入口、默认模型路径与run()|
|applications/realtime_hand_sim.py|preview_hud()|实际使用的CameraPreview、场景和运行辅助|
|applications/dual.py|DualTeleoperator.produce_parts()|update()和正式RobotControlSession；新入口未调用produce_parts|
|data/hand_body_recording.py|_stack_vector()|现用身体/手部记录格式|
|data/npy.py|read_frame()|read_recording()迁移到Task 2|
|data/npz.py|write_arrays()|read_arrays()迁移到Task 2|
|data/robot_command_recording.py|iter_robot_commands()|load_robot_recording()、RobotRecording.iter_frames()；回放需要receipt|
|retargeting/arm/rotations.py|matrix_from_quaternion()|被调用的rotation_matrix_to_quaternion()|
|retargeting/hand/geometric.py|_in_plane_angle()、GeometricHandRetargeter.expected_thumb_lean()|几何诊断和内部_robot_reference；不改数学|
|retargeting/hand/postprocessing.py|KalmanSmoother.filter_sequence()|在线step()/reset()|
|simulation/arm_realtime.py|AxisMap.with_defaults()|映射、parse()、桥和标定|
|simulation/arm_scene_driver.py|ArmSceneDriver.arm_values()|求解、输出和真实诊断函数|
|simulation/randomization.py|DomainRandomizer.get_randomized_params()|正在使用的randomize接口|
|simulation/robot_loader.py|RobotLoader.set_arm_pose()|任务环境加载和动作接口|

另清理contracts/robot_control.py中只构造却未读取的局部names。RobotBackend协议不按无调用删除：它是正式输出边界，应给RobotOutput的backends参数加Mapping[str, RobotBackend]注解，使定义真正被使用；不引入运行时继承或校验开销。

### 1.2 迁移后删除的薄文件（不是原本无用）

|原文件|合并目的地|调用方|
|---|---|---|
|applications/hand.py|CLI直接路由到hand_align.run、hand_inspect.run|cli/registry.py、schemas.py及CLI相关测试|
|applications/dual_export_cli.py|dual_export.py新增main(args)，保留打印和返回码|registry.py、schemas.py|
|contracts/hand_models.py|contracts/constants.py的HAND_MODELS|hand_spec、CLI、2个devtools；保持无重依赖|
|data/npy.py、data/npz.py|新data/array_recording.py|diagnostics/visualize_hand_keypoints.py|

结果为删除5个文件、新增1个文件，净减少4个源码文件。已有train/CLI适配器虽然短，仍负责参数组合、延迟依赖加载，不因行数少一并删除。

### 1.3 不应误删或强行合并

- `__main__.py`、simulation.__getattr__、tasks.__init__和任务装饰器注册：属于动态入口/导出。
- `applications/arm_validation.py`、`camera_record.py`、`learning/hand_evaluation.py`、`retargeting/hand/geometric.py`：业务src零导入，但有明确外部工具消费者。
- `MediaPipeCameraWorkflow.next_input()`：无仓库调用，但属于公开输入接口；保留，文档纠正其与next_window的关系。
- `retargeting/hand/kinematics.py`与`robots/hand_kinematics.py`：前者PyTorch可微FK，后者NumPy URDF子树/reference几何，不互相替代。
- `data/body_recording.py`、hand_body_recording.py、hand_joint_recording.py、robot_command_recording.py、command_h5.py：真实消费者和schema不同，不能合并为一个无版本检查的H5工具。
- `RobotPartWorld`、`ThreeRobotHandScene`、`RealtimeL21Scene`、`Tron2ADualArmWorld`：仍分别支持部位显示、旧诊断、独立L21和TRON安装；本次只共享确实相同的初始化辅助。
- accepted_robot_output.py和robot_session.py：分别适配已有接受命令与完整观测求解；不是两份相同控制循环。
- 新body/wrist求解和旧专用TRON2A安全控制：不以合并为名更改IK或标定。

## 2. 公共执行和验证步骤

当前阶段只保存计划/spec；源码没有修改，也没有重新运行测试。执行时每批重新检查状态、建tag，不用本计划文档状态冒充源码基线。

```powershell
$cleanupRepo = 'D:/2026/code/teleoperation-system-retargeting'
$cleanupPython = 'D:/Anaconda/envs/teleoperation/python.exe'
Set-Location -LiteralPath $cleanupRepo
$env:PYTHONPATH = Join-Path $cleanupRepo 'src'
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting status --short --branch
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting rev-parse HEAD
# 先解释、保存当前预期dirty变更，再建立本批检查点。
$cleanupTag = 'checkpoint-' + (Get-Date -Format 'yyyyMMdd-HHmm') + '-' + $cleanupBatch
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting tag -a $cleanupTag -m "Before $cleanupBatch"
```

各Task均以以下验证结束；源码与相关测试在执行测试时不能继续修改。

```powershell
& $cleanupPython -B -X utf8 -m unittest discover -s tests
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting diff --check
# 只stage本批明确路径，提交并记录post hash/pre tag，不push。
```

## Task 1：删除无调用片段并复核闲置导入

**Files:** 修改1.1表中14个文件及contracts/robot_control.py、simulation/robot_output.py；按AST候选清理以下混合文件中的纯闲置import：applications/benchmark.py、camera_teleop_source.py、dual_replay.py、hand_align.py、hand_control.py、hand_inspect.py、hand_processing.py、hand_realtime.py、realtime_hand_sim.py、robot_session.py、visionpro_results.py；learning/dataset.py、retargeting/hand/predictor.py、processing.py、normalization.py及simulation/arm_scene_driver.py等。

**Interfaces:** 所有仍被调用的函数签名和数值不变；输出backends使用现有RobotBackend Protocol进行类型说明。

- [x] 设置cleanupBatch=remove-unused-members并执行公共Git步骤；逐个符号rg复核src/tests/devtools和文档，排除只定义/历史计划命中。
- [x] 删除1.1中的15个函数/方法及常量；删除相关过时docstring（例如filter_sequence用法），保留next_input。
- [x] 闲置import必须先检索外部导入/属性访问。特别保留或迁移后再删除：hand_realtime.MediaPipePalmLocalPipeline、arm_diagnostics.sample_seconds、simulation.hand_commands.add_hand_command_arguments、diagnostics绘图导出、tasks和evaluation包导出。星号导入不能按“未使用”直接删除。
- [x] RobotOutput参数使用`backends: Mapping[str, RobotBackend]`，typing.Mapping和现有Protocol同文件维护。contracts/robot_control.py删除无用names局部变量，不改变L21按part命名空间行为。
- [x] 运行现有test_camera_record、test_mediapipe_realtime、test_joint_commands、test_robot_command_recording、test_robot_command_replay、test_arm_following、test_arm_display_stability及全套。
- [x] 更新current-interfaces对离线Kalman/produce_parts/iter_robot_commands的当前说明；历史实施计划加后续变更提示，不伪造原执行记录。提交`refactor: remove unused helpers and stale imports`。

## Task 2：合并薄转发与轻量数据文件

**Files:** 删除1.2列出的5个文件；新data/array_recording.py；修改cli/registry.py、schemas.py、contracts/constants.py、robots/hand_spec.py、cli/hand_joint_options.py、cli/visionpro_options.py和devtools/{compare_hand_methods,run_retargeting_pipeline}.py；核对test_hand_core_regression、test_coordinate_pipeline、test_hand_control_cli、test_hand_specs、test_cli_integration中的旧导入/handler断言。

**Interfaces:** 命令名/参数不变。新array_recording提供`read_recording(path)`、`read_arrays(path)`；分别保留原np.load行为和NPZ上下文关闭；不改其他录制格式。

- [x] 设置cleanupBatch=merge-thin-modules并建检查点。
- [x] registry与schemas的handler同步改为`teleoperation.applications.hand_align:run`、`teleoperation.applications.hand_inspect:run`、`teleoperation.applications.dual_export:main`，不保留旧内部转发文件。
- [x] 将dual_export_cli.main原8行移到dual_export.py；将HAND_MODELS原元组移到constants.py，保持元组内容和顺序；修改全部调用方的显式导入。
- [x] 合并npy/npz现用读取函数到array_recording.py，修改visualize_hand_keypoints.py的两个import，然后删除原文件。
- [x] 复用CLI轻量help回归，要求无NumPy/Torch时help仍成功。用临时NPY/NPZ做读取与关闭验证：

```python
np.save(root/'frames.npy', {'left': np.zeros((25,3))})
np.savez(root/'frames.npz', timestamps=np.array([0.,1.]))
assert read_recording(root/'frames.npy').item()['left'].shape == (25,3)
assert read_arrays(root/'frames.npz')['timestamps'].tolist() == [0.,1.]
```

- [x] 运行test_cli_integration、test_coordinate_pipeline、test_hand_core_regression、test_inspection_entrypoints、test_hand_specs、test_mediapipe_tron2a及全套；提交`refactor: consolidate dispatch and array recording helpers`。

## Task 3：统一手部NN/dex实例构造

**Files:** 修改applications/hand_backend.py、hand_control.py、dual.py、robot_session.py；扩展tests/test_normalized_retargeters.py、test_hand_control_cli.py、test_mediapipe_tron2a.py、test_unified_hand_inputs.py。

**Interfaces:** hand_backend新增`build_hand_retargeter_from_specs(specs, backend, *, checkpoints=None, device='cpu')`，消费HandKey→HandSpec及每模型一份shared权重，返回具备specs/reset/retarget的现有retargeter组合。保持build_camera_retargeter和build_target_hand_retargeters公开签名。

- [x] 设置cleanupBatch=share-hand-backend-construction并建检查点。
- [x] 先加行为对照：固定seed/同shared checkpoint、同specs和同HandWindow，通过旧公开入口与新工厂得到相同positions/valid/hand identity。每个模型权重只加载一次；左右侧共用canonical-right模型。dex同环境比对，arms-only零模型构造。
- [x] 先运行新测试确认因工厂尚未定义失败，再迁移共同的模型选择、device解析、shared加载和NN/dex实例化。
- [x] hand_control._selection保留source/profile及旧单权重选项校验；dual.load_camera_hand_retargeter保留L21两侧和调用签名；旧camera入口保留显式camera profile。它们只把完成预检的规格交给工厂，不删除边界限制。
- [x] 不合并手部命令后处理、碰撞guard或TransHandR专用模型加载；不把CameraRetargeters改名扩散到所有文件。本批先消除实现重复。
- [x] 运行test_normalized_retargeters、test_hand_control_cli、test_hand_control_h5、test_mediapipe_tron2a、test_unified_hand_inputs、test_robot_control_session和全套；独立dex环境补test_normalized_dex及test_unified_hand_inputs。提交`refactor: share native hand backend construction`。

## Task 4：预览与场景辅助各归其位

**Files:** 新applications/camera_preview.py、applications/realtime_support.py；修改applications/realtime_hand_sim.py、three_robot_scene.py、visionpro_hand.py、simulation/arm_preview.py、realtime_teleop.py、urdf_loader.py、camera_scene_arm.py；迁移tests/test_mediapipe_realtime.py、test_arm_preview.py、test_arm_ik_multi_robot.py和相关devtools导入。

**Interfaces:** camera_preview持有现有PREVIEW_WINDOW、CameraPreview、_PreviewCapture、attach_preview_capture、build_preview、_desktop_size/_window_pos。realtime_support持有现有_describe、_loop_fps、_median_interval、RELEASE_TIMEOUT_SECONDS、_drain_in_background。urdf_loader新增`prepare_physics(call, positional, *, client_id, timeout=15.0)`。

- [x] 设置cleanupBatch=share-preview-and-scene-support并建检查点。
- [x] 预览代码和运行辅助按原函数体移动；调用方改为从真实归属文件导入，删除旧文件的间接再导出，避免长出新兼容链。RealtimeL21Scene、resolve_model_asset和现被VisionPro使用的build_scene继续保留在原场景文件。
- [x] 将两个逐语句相同的_prepare_physics实现合成urdf_loader.prepare_physics；重试间隔、超时、报错内容不变，两个场景显式传client_id。
- [x] 将realtime_teleop.rigid_elbow_robots删除，消费者/测试改用applications.camera_scene_arm.rigid_elbow_robots，保留原集合行为。
- [x] 共享初始化新增行为测试：第一次调用抛Not connected、第二次成功；已超时则RuntimeError；不同client_id不得混用。预览沿用现有mock CV2/截图回归；不重新设计UI。
- [x] 运行test_mediapipe_realtime、test_arm_preview、test_native_hand_replay、test_transhandr_vr、test_arm_ik_multi_robot、test_arm_display_stability及全套；提交`refactor: centralize preview and simulation setup helpers`。

## Task 5：合并GBK扫描器和绘图重复函数

**Files:** 修改applications/diagnostics/check_gbk_safe.py、devtools/check_gbk_safe.py、devtools/preflight.ps1、tools/plotting.py、tools/hand_keypoints.py、tools/l21_fk.py；新tests/test_gbk_scan.py、test_plotting_helpers.py。

**Interfaces:** 保留CLI `tools check-gbk-safe`和脚本`python devtools/check_gbk_safe.py`。plotting新增`set_equal_axes(ax, points, *, padding, zero_radius)`，两个调用方显式传原参数。

- [x] 设置cleanupBatch=merge-duplicate-devtools并建检查点。
- [x] 先用临时目录测试扫描路径、中文/不可GBK字符、strict退出码、错误读取和跳过目录；两入口相同root得到相同结果。扫描核心保留在现有diagnostics文件，移除导入时修改sys.stdout的副作用。
- [x] 两份SKIP_DIRS有差异，按项目受维护源码范围统一为跳过assets、third_party、outputs、.venv、.git、__pycache__等生成/外部目录；写入行为测试，不假称原策略完全相同。脚本只解析参数、设置src路径后调用同一main。
- [x] preflight只运行一次正式GBK扫描，保留退出码和日志。删除“队友各维护一份必须重复跑”的过时说明。
- [x] 合并绘图坐标轴代码时，hand_keypoints保留padding=.55/zero_radius=1.0，l21_fk保留padding=.6/zero_radius=.1；测试零跨度和普通点云的轴范围与原值一致。
- [x] 运行新增两组测试、test_inspection_entrypoints、CLI静态help和全套；提交`refactor: reuse source scanning and plotting utilities`。

## Task 6：启动器共用一套准备逻辑，移除失效说明

**Files:** 新devtools/windows_runtime.py；修改devtools/run_realtime_windows.py、run_realtime_arm_windows.py、run_realtime_arm_elbow_follow_windows.py；新tests/test_windows_launchers.py。

**Interfaces:** 保留三个脚本入口供现有PyCharm启动配置使用；共同环境准备、解释器选择/重启、缓存、GUI/相机检查和报告读取放windows_runtime。该模块提供prepare_environment、redirect_caches、relaunch_with_canonical_python、run_check、summarise_camera_window，参数保持原调用语义。

- [ ] 设置cleanupBatch=consolidate-windows-launchers并建检查点。
- [ ] 为参数传递写测试：hands启动器添加--no-arms；普通arm启动器保留原参数；elbow启动器显式operator/off覆盖规则不变；调用方给的--robot、backend、worker路径、report不得丢失。
- [ ] 把run_realtime_windows中共用的准备/检查函数移动到windows_runtime；另外两个启动器不再经“手启动器→臂启动器→肘启动器”获取公共环境。解释器重启必须指向原调用脚本，避免切错工作流。
- [ ] 删除POSE_ASSET及“缺Pose模型只能动手”的错误检查，改为检查smplerx_defaults对应的worker/source/checkpoint/body-model路径；更新E盘/空壳venv/默认三台的错误提示。
- [ ] 不悄悄切换成不支持预览诊断参数的新CLI。仍保留旧诊断入口；没有--robot时在打开设备前清晰报错，不替用户默认选三台。
- [ ] 用mock subprocess/设备检查验证--check、重启退出码和参数传递；无需自动开启相机。运行test_cli_integration、test_arm_following、test_smplerx_assets、test_windows_launchers和全套；提交`refactor: consolidate Windows runtime launch support`。

## Task 7：最终核对与文档归档

**Files:** README.md、docs/current-interfaces.md、docs/unified-hand-input.md、本plan/spec；给2026-11-04两份相关历史计划加后续整理说明。

- [ ] 设置cleanupBatch=verify-code-consolidation并建检查点。
- [ ] 新增“命令→实际应用→算法/数据模块”表，注明正式机器人入口、独立手/训练、旧诊断和显式上游复现，不把旧诊断写成无用代码。
- [ ] 复做182文件级导入/符号/动态入口审计；确认删除对象没有当前代码、脚本、测试或现行用法文档残留；历史计划中的清单保留为历史记录。
- [ ] 对全部现存Python做AST解析；逐个注册命令定位handler、检查--help，并复查help未引入NumPy/Torch/相机重依赖。
- [ ] 主环境完整unittest；dex环境运行test_normalized_dex、test_normalized_retargeters、test_unified_hand_inputs、test_robot_control_session、test_robot_comparison、test_robot_final_acceptance、test_arm_display_stability；检查没有新增意外skip。
- [ ] 必须保留test_arm_display_stability的缺失输入零IK、单侧独立、异常渲染恢复，以及录制回放/parts显示矩阵检查。相机和真实头显未实测时单列“未实测”，不据离线测试宣称硬件通过。
- [ ] 记录实际删改文件、前后源码数量、各批hash/tag、测试pass/skip/fail和未合并部分；提交`docs: document consolidated code ownership and entry points`，不push。

## 预计结果与回退

不以总文件数作为唯一指标。薄模块预计净减少4个源码文件；预览、运行支持和Windows工具提取会新增职责明确的模块，整体维护点更少但文件总数未必下降。主要收益是15个无调用片段退出、同类工厂/扫描/初始化实现只保留一处，以及入口说明与当前代码一致。

每批通过前不得开始下一批。回退使用选定检查点恢复本批路径或建立rollback分支，不reset --hard、不重写历史：

```powershell
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting diff $cleanupTag -- src tests devtools docs README.md
git -c safe.directory=D:/2026/code/teleoperation-system-retargeting switch -c rollback-code-consolidation $cleanupTag
```

## 执行记录

- Task 1：pre 77ecb0bce0a1a5bbfb93d1a9fded43f7d572168f；tag checkpoint-20261106-2058-remove-unused-members；删除15个无调用函数/方法及闲置导入。相关40项、间接依赖补测25项通过；最终完整589项，586通过、3项可选dex依赖跳过。PALM_LENGTH_ALIGNMENT有效再导出保留；旧测试时钟补丁迁移到实际JointCommandPipeline归属模块。

- Task 2：pre 479226ee8401a8d206b8605d446bf280630a67db；tag checkpoint-20261106-2112-merge-thin-modules；5个薄文件迁移后删除、新增array_recording，源码净减少4个文件；相关101项通过；完整590项，587通过、3项dex依赖跳过。28命令保持，CLI静态help和数组读取/文件释放通过。

- Task 3：pre b350d15b7f5947be46fb632c68abdffb47782681；tag checkpoint-20261106-2120-share-hand-backend；新增tests/test_hand_backend_factory.py覆盖共享加载/跨入口数值/纯臂零模型/混合侧dex等价。相关93项（3跳过）；完整594项，590通过、4项dex依赖跳过（含新增dex工厂等价用例）；独立dex20项全部通过。算法、输入profile、shared校验和后处理未改。

- Task 4：pre b7db89e852f65d5947fe64736d8d08d70b7b33de；tag checkpoint-20261106-2129-preview-scene-support；新camera_preview/realtime_support归属明确，迁移类/函数AST与原实现一致；初始化重试集中到urdf_loader；重复rigid_elbow_robots删除并迁移测试。相关96项通过；完整596项，592通过、4项dex依赖跳过。

- Task 5：pre a372c915cfc6c121d6c9ac0bd69087eb63b31d25；tag checkpoint-20261106-2139-merge-devtools；GBK脚本/CLI共用扫描，preflight只执行一次；扫描覆盖自有robots源码、跳过外部/生成目录；两个绘图padding/零跨度默认值保持。相关15项、实际仓库GBK strict与PowerShell解析通过；完整600项，596通过、4项dex依赖跳过。
