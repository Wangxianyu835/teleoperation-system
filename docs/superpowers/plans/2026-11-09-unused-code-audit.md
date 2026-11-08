# 未调用代码清理与逻辑精简候选 Implementation Plan

> **For agentic workers:** 本次按用户授权在当前会话顺序执行，不启用子智能体。只删除已核实无消费者的实现；仍在运行的逻辑仅提出建议。

**Goal:** 删除没有消费者的代码，给用户提供其余逻辑精简的位置和前后对比。

**Architecture:** AST 扫描定义、名称/属性读取、导入、常量及局部赋值，再用全文检索核实字符串路由、装饰器注册、包导出和测试消费者。保留公开输入、窗口、结果接口以及各条算法路线。

**Tech Stack:** Python、AST、rg、unittest、PyBullet、HDF5；不新增依赖。

**Spec:** 本次用户请求：无调用代码直接删除；逻辑可能多余但被调用的代码保留，介绍前后对比，由用户决定。

## Global Constraints

- 修改前提交：`1c9a8f85420154af5e029565df99faa4e4a7486b`。
- annotated checkpoint：`checkpoint-20261109-0058-unused-code-audit`。
- 不改坐标处理、归一化、NN/dex/IK 数学、TransHandR 上游流程、权重或录制格式。
- 不删测试、机器人资产、数据、环境或生成文件。不推送。
- 公共输入接口 `MediaPipeCameraWorkflow.next_input` 和观测类型别名不因仓库无调用而删除。

## Task 1：删除已确认无消费者的片段

- [ ] 删除 `retargeting/arm/ik.py` 的 `ArmIK` 协议及仅服务该声明的 `Protocol` 导入。实际调用使用 `RoboticsToolboxArmIK`。
- [ ] 删除八项未读取的内部常量：`hand_body_recording.FORMAT_VERSION`、`config.EXCLUDED_COLLISION_PAIRS`、`geometric.ROLL_DIM`、`topology.VISIONPRO_SOURCE/MEDIAPIPE_APPROX_SOURCE`、`plotting.NAME_INDEX/ROBOT_EDGES/ROBOT_TIPS`。实际保存版本、碰撞实现和绘图拓扑有各自现用定义，保持原样。
- [ ] 删除 `hand_control._process` 未使用的私有 `live` 参数及其不可达的时钟分支。全部两个应用调用和四个测试调用均未传入该参数，继续使用 `window.timestamp`。
- [ ] 清理 `realtime_support`、相机/环境诊断、可视化、动作回放和外部 SMPLer-X 后端内已核实无使用/无消费者的导入。保留 `matplotlib.use('Agg')`、环境诊断的导入探针、任务注册导入及有测试消费者的再导出。
- [ ] 对照 checkpoint 检查 diff，确认保留函数除上述私有参数/分支与闲置导入外无行为变化。

## Task 2：验证和提交

- [ ] 主环境运行手部控制、统一输入、机械臂、绘图、应用入口、SMPLer-X 相关既有测试。
- [ ] 主环境运行 `python -B -X utf8 -m unittest discover -s tests`。
- [ ] 独立 dex 环境运行统一输入与 NN/dex 后端相关既有测试。
- [ ] 验证 28 个 CLI handler 可定位、无 site-packages 下的帮助入口可用、两种可视化实际出图及所有自有 Python 文件 AST 可解析。
- [ ] `git diff --check`，提交本次明确文件，报告提交哈希和 checkpoint。

## Task 3：仅报告的逻辑候选（不执行）

|位置|修改前|建议修改后|影响与边界|
|---|---|---|---|
|`applications/hand_control.py:run/_commands`；`applications/robot_control.py:preflight`|为预检创建一个空/临时 `JointCommandPipeline`，随后再创建正式实例|提取命令选项校验函数，预检只校验，正式实例只构造一次|减少重复构造；必须保留打开相机/GUI 前报错的时机。手部临时实例会创建每侧滤波状态，robot 预检实例的规格为空，开销较小。|
|`applications/simulation.py:create_environment`、`simulation/environment.py`|`record=False` 仍创建 `EpisodeRecording`、`SensorRecorder` 和数据目录，并在 reset/结束时更新记录状态|不录制时传 `recording=None`，只在开启录制时创建记录器|减少无用对象和目录副作用；改变调用者可观察的 `env.recording` 以及记录目录创建时机，须核对相关消费者。|
|`cli/registry.py`、`cli/schemas.py`、`cli/__init__.py:build_parser`|registry 存 handler，schema 又保存一遍 `_handler`；build_parser 解包 registry 的 handler 却未使用|schema 只描述参数，build_parser 从 registry 设置 `_handler`|保持命令行为，处理器路径只维护一处；先核对直接调用 schema 构建参数的测试。|
|`applications/benchmark.py`、`applications/diagnostics/check_environment.py`、`simulation/environment.py`|多个应用/导入阶段重复调用 `register_builtin_tasks()`；任务模块的 import 缓存使后续调用通常不再执行注册装饰器|保留明确初始化边界，其余应用复用已初始化注册表|减少重复职责，性能收益很小；`get_task()` 单独调用不会自动注册，不能一概删除注册入口。|
|`applications/benchmark.py:interactive_menu`；`simulation/tasks/sequences.py:DrawerBook.check_success`|调用 `list_tasks()` 和读取书位置后将结果赋值给不再读取的局部变量|删除未使用的枚举/查询及赋值，继续使用现有按级别列表和 `_is_above()` 判定|前者重复排序，后者重复物理状态查询；属于运行逻辑删除，按用户要求保留待决定。|
|`contracts/validation.py`|星号导入 constants 后又定义相同侧标识与维度别名|内部直接使用 constants 中的定义；保留确有消费者的公开导出|减少重复定义，基本无性能收益；星号导入形成的公开名字需保留兼容性。|

不建议把两种 FK、各版本 H5、独立手/整机输出适配器、无效输入保持、资源清理或输入边界校验视为同一套冗余实现。它们存在不同消费者或执行语义。

## 执行证据

待完成验证后填写实际结果；历史文档的测试数字不作为本次结果。
