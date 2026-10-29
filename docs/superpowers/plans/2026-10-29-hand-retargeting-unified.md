# 掌长归一化与重定向输出接口整合 Implementation Plan

> **For agentic workers:** 使用 superpowers:executing-plans 按任务执行；最终独立审查按该技能安排。用户要求覆盖技能默认的 worktree、逐任务提交和自行裁决规则。

**Goal:** 同一份人体 H5 可以训练多种目标手的 NN；运行时可以选择手型和 NN/dex，换操作者不需要人工掌长标定；所有方法输出同一目标手的独立关节结果。

**Architecture:** 原始人体数据保留米制，完成掌部朝向对齐后，在三帧窗口边界执行共享掌长归一化。目标手规格定义机器人关节、参考长度和运动学；NN 与 dex 共享目标构造和输出接口，统一处理命令与仿真执行。L21 是目标手之一，历史 18D 表示仅保留于权重或文件读取边界。

**Tech Stack:** Python、NumPy、PyTorch、URDF、dex-retargeting、PyBullet、H5、unittest。

**Spec:** 本聊天已确认的设计及下文具体协议。用户已确认三帧中位数、阈值、参数与兼容策略；实施证据见§15，真人跨操作者验收仍待完成。

**Status:** 代码实现、独立审查修复和最终双环境回归已完成；main 未提交、未推送。真人/跨操作者效果与提交等待用户验收。

## Global Constraints

- 在当前 main 分支实施；不创建新分支或 worktree。
- 用户验收前不提交；验收后提交说明为「重定向输出接口整合」；不推送。
- 修改前基线：11abbfa21e5e53675f179f5a9569aecaf612a9cc。
- 已建立注解标签：checkpoint-20261029-1156-hand-retargeting-unified。
- 当前未提交变更属于同一项已授权整合工作，不把中途状态提交为新基线。
- 计划未说明且影响行为/数值/数据语义的情况，暂停并询问用户；不自行增补行为。
- 原始 H5 不原地改写，不把机器人关节结果当公共人体训练集。
- 旧公开生产接口被新接口替代后必须删除，不能以别名或平行实现长期保留。
- 输入采集和 HandWindow 结构保持稳定；新版规范通过明确 metadata 区分。
- VR 继续使用明确的 VR 路线。本轮新的免标定规范先覆盖摄像头 world 训练、离线和实时。不得把相机预处理直接用于上游 VR 权重。
- 涉及 VR 接口迁移时，对照 D:/2026/code/TransHandR/TransHandR 实际源码及权重，记录路径与哈希并验证原模型数值不变。
- 公共命令速度取 CLI 与 URDF/耦合反推约束的较小值。用户已同意 L21 从旧默认 1.5 rad/s 收紧到 URDF 的 1.0 rad/s。
- 不新增外部依赖，不重写 dex 上游优化器，不改变 URDF 来适配算法。
- 合成 21 DOF 测试不冒充真实 21 DOF 手型接入；新真实手型需完整资产和任务参考点。

## 1. 已有改动与证据

已新增但尚未提交：
- src/teleoperation/contracts/hand_control.py：HandJointResult，独立坐标、目标标识、关节顺序和基本校验。
- src/teleoperation/robots/hand_spec.py：HandSpec、load_hand_spec；L21/H1-2/GR1-T2/G1 的独立坐标、耦合与有效约束。
- tests/test_hand_specs.py：8 项行为测试。
- 本计划文档。

未迁移旧消费者，未改变原有 NN/dex/VR 数值、训练或保存路径，也未实现掌长归一化。

测试证据：
- 原普通环境完整基线：465 项，1 项原有跳过，退出码 0，81.587 秒。
- 原 dex 环境完整基线：465 项，1 项原有跳过，退出码 0，85.334 秒。
- 新增基础实现后：普通环境 473 项，1 项原有跳过，退出码 0，82.565 秒。
- dex 相关定向检查 13 项通过。
- 最初的 14 errors/2 failures 已用相同代码对照确认由沙箱禁止 os.link 导致；沙箱外完整基线通过。不要为该权限问题改写校准保存算法。
- 日志位于 outputs/tmp_interface_integration/，不入 Git。
- 只完成基础子项，不将整体 task 1 或重构标为完成。

## 2. 掌长归一化的唯一公共规则

### 2.1 输入和时间规则

输入必须是来源明确、保留米制、已经腕点归零并完成规范朝向对齐的 hand25：
- 每帧形状 (25,3)，三帧窗口形状 (3,25,3)。
- 腕点索引 0，中指 MCP 索引 11；原 MediaPipe21 对应索引 0 和 9。
- 每侧独立处理，时间窗口只使用当前及过去帧。
- 训练、离线导出、实时 NN 和新的 dex 路线调用同一个函数，不复制公式。
- 不使用整份录制统计量、未来帧、首次张开手样本或操作者永久尺度。

### 2.2 公式与无效处理

对一侧窗口 X，先保证各帧腕点为原点：
- P[k] = X[k] - X[k,0]
- length[k] = norm(P[k,11])
- denominator = median(length[0], length[1], length[2])
- normalized[k] = P[k] / denominator

三帧共用同一个 denominator。数值退化下限定为 1e-6 m：
- 任一帧非有限、任一掌长 <= 1e-6 m、形状错误或掌部几何无效，均不输出该侧有效归一化窗口。
- 不用 epsilon 强行替代分母，不回退旧窗口，不借另一侧尺度，不使用旧标定。
- 实时应用清空无效侧缓冲，待连续三帧有效后恢复；离线样本有效性必须逐时间点相同。
- 不新增未经实测的绝对人体掌长范围和离群跳变阈值；若需要，另行询问用户。
- 不原地修改输入数组；三帧尚未就绪仍按原窗口约定处理。

### 2.3 新规范与单位

新增 input_profile：
camera_palm_length_window3_v1

新增 coordinate_alignment：
mediapipe_world_wrist_index_middle_canonical_right_palm_length_window3_v1

输出 metadata 明确声明：
- coordinate_frame = hand_canonical_right
- units = "1"，length_unit = "1"
- source_landmark_space = mediapipe_world_meters
- normalization = palm_length_window_median
- normalization_version = 1
- palm_reference_indices = [0,11]
- temporal_window = 3
- 每侧实际 denominator 保存在诊断字段 palm_length_m 中，值为米，不作为每人的永久校准。
- 原采集来源与时间戳保留；source 与变换后的单位不得混淆。

新规范不使用原 world-metre 校验逻辑冒充米制。已归一化输入再次进入归一化函数时拒绝处理。

### 2.4 核心接口

在 retargeting/hand/normalization.py 提供：
- normalize_hand_window(window: HandWindow) -> HandWindow

在 contracts/coordinates.py 定义并校验新规范；不在 data 层执行数学变换。

## 3. 公共人体 H5 与 Dataset

- 公共训练输入仍为人体米制 H5，原始录制或已有机器人无关 canonical world 对齐文件均通过明确的读取/对齐路径进入。
- 原始录制先执行 hand align，或由已有采集流程生成对齐文件；hand train 不隐式重写输入文件。
- 旧 normalized 图像坐标、旧机器人根坐标或机器人角度结果不作为新版公共训练输入，给出明确转换/重新采集提示。
- Dataset 输出人体窗口、当前帧人体目标、frame_id、timestamp、valid mask；不包含 robot model 或机器人关节标签。
- 公共归一化在窗口形成后执行；训练目标取 normalized[-1]，不得将归一化输入与米制人体目标混用。
- 丢帧、缺手、划分边界、无效恢复保留原时间线，不跨越 gap 或 train/validation 边界拼接窗口。
- 多个目标手复用同一份人体样本和同一份 split identity；模型权重和机器人目标分别构造。
- 训练前后 H5 SHA256 必须一致。所有衍生模型在 dataset.json 记录同一数据哈希、帧范围和划分规则。

## 4. 目标手规格与几何目标

HandSpec 增补：
- 机器人原生根坐标中的腕/掌参考点与食指、中指 MCP 对应 frame。
- 中性参考姿态 q_reference：有效界内的零位；超出有效界时显式投影到边界。
- reference_palm_length_m：该参考姿态下 root 到中指 MCP 对应 frame 的距离。
- canonical 与目标手原生根坐标的明确变换；左右手分别定义。
- 指尖/任务点对应、独立/从动关节关系和数据资产哈希。

不得仅因名字相似推断机器人腕点或 MCP；登记项逐个核对 URDF 和已有运动学。
参考掌长非正、参考点重复或构成退化掌部基时拒绝登记。

统一构造机器人米制目标：
target_native_m = transform_to_target_root(normalized_human_points) * reference_palm_length_m

共同计算归一化任务向量：
human_vectors = normalized[-1, selected_tips] - normalized[-1, wrist]
robot_vectors = (FK(q, selected_tips) - FK(q, root)) / reference_palm_length_m
比较时将二者置于同一明确的坐标系。

公共接口在 retargeting/hand/hand_model.py：
- build_hand_targets(window: HandWindow, spec: HandSpec) -> HandTargets
- DifferentiableHandModel.positions(q: Tensor, frame_names: tuple[str,...]) -> Tensor

数组位置依赖 joint_names/frame_names 映射，不能使用 L21 专属节点索引推断其他手型。

## 5. 唯一重定向结果与方法接口

公共生产基类保留在 retargeting/hand/interface.py：
- HandRetargeter.retarget(window: HandWindow) -> dict[HandKey, HandJointResult]
- HandRetargeter.reset() -> None
- HandKey = (hand_model, side)

公共结果由 contracts/hand_control.py 定义：
- hand_model、side、spec_hash
- joint_names：独立关节顺序
- positions：D 维独立关节坐标，rad
- timestamp、valid、error、solve_ms

新 NN、dex、几何方法均实现该接口。几何只声明已适配的 L21 能力，不强加掌长归一化到其内部旧公式。

旧 L21 NN：
- 输入保留其权重实际要求的米制或旧规范，不能强行归一化。
- 模型内部 18D 输出按原有手侧约定转换，只在结果边界去掉占位。
- 外部输出仍是 HandJointResult(hand_model="l21", positions[17])。
- 历史权重选择和预处理通过显式 legacy profile 处理；不能伪造新版 metadata。
- 旧网络和文件的转换是私有边界函数，不形成第二套公共生产结果类型。

## 6. NN 动态输出和训练

- PoseTransformer 主体复用，输出头 D = len(spec.independent_joint_names)。
- 新模型直接输出目标手原生独立坐标；角度范围来自有效关节限位，输出头不含占位。
- 左右手新模型分别训练；同一人体左右来源可以按已声明 canonical 规则池化，但人体来源侧和目标机器人侧必须分别记录。不得套用 L21 四个 roll 反号到别的手型。
- 保留可复现 seed、训练/验证帧间隔与原优化配置；新损失配置独立版本化，不声称等同旧六项 L21 loss。
- 新通用训练的主损失使用归一化任务向量 Huber 损失，delta=0.02（无量纲）。
- 增加弱中性姿态正则：各非零范围独立坐标的 ((q-q_reference)/range)^2 均值，默认权重 1e-4；零范围关节排除。该默认是本计划的提议，需随本计划确认，不能未经确认实施。
- 不把 dex 伪标签设为唯一监督，NN 使用可微 FK 对公共人体几何目标训练。
- FK 必须支持 batch、固定关节和可微耦合展开；不固定拼接五个 tip 占位，不把不可微第三方 FK 当成可微路径。
- CPU smoke 使用可覆盖的小网络、1 epoch 与少量样本；不把 smoke 权重称为可用跨操作者模型。
- 新增训练轮次不足、缺失验证集、无有效样本、错误目标规格等显式失败处理，不自动放松条件。

## 7. dex 新归一化路线

- 复用上游 VectorOptimizer/SeqRetargeting；目标由 HandSpec 与 build_hand_targets 提供。
- 新 camera_palm_length_window3_v1 路线不要求 --calibration 或先张开手。
- 向量优化使用机器人任务向量除以 reference_palm_length_m 后的无量纲表达；上游版本是否能以等价方式应用该比例，需要先核对安装源代码。不得在未核对情况下假设 scaling 参数会缩放机器人向量。
- 如现有优化器仅接受米制目标，可传 target_native_m，并以米制等价变换保持 Huber delta = 0.02 * reference_palm_length_m；记录实际目标单位和代价归一化。若正则/容差不能等价，暂停询问，不悄悄改变比较语义。
- 每个目标/手侧独立求解状态；丢手清理求解历史，执行命令保持并在恢复时限速。
- 现有 G1 多初值恢复和可达性检查原为米制，迁移时保留单位，不把 0.005 m 等阈值直接当作无量纲值。
- 新增 L21 17D dex 适配，首先完成同一真实手型 NN/dex 的共同结果验证。
- VR 历史 dex 仍使用其已有显式校准与预处理；不因新增相机免标定路线而删除有效 VR 校准能力，也不对 VR 自动套用相机新规范。

## 8. 权重目录、训练参数和加载规则

模型默认根：
outputs/hand_retargeting/models/

目录：
<hand_model>/<target_side>/nn/<input_profile>/<run_name>/

新规范示例：
outputs/hand_retargeting/models/l21/right/nn/camera_palm_length_window3_v1/experiment01/model_best.pth

每个目录：
- model_best.pth：最佳验证状态
- model_last.pth：最近完整 epoch，含 optimizer、scheduler、RNG 与数据顺序状态
- config.json：模型、预处理、手型、任务/损失配置
- dataset.json：H5 哈希、split identity、人体来源侧/目标侧与帧范围
- metrics.jsonl：按 epoch 记录指标
- training.log：运行日志

paths.py 提供：
training_run_directory(root, hand_model, target_side, input_profile, run_name) -> Path

新 checkpoint schema：hand_joint_nn_v1。
checkpoint 内必须包含模型配置、ordered independent joints、spec_hash、input_profile、
normalization 参数/版本、输出单位/约定、参考几何配置、训练状态及数据身份。
不能只依赖相邻 JSON 才能验证一个被复制的 .pth。

hand train：
- --input：一个公共人体米制对齐 H5
- --hand-model：必填，nargs+，注册 ID；逐手型依次训练
- --hand-side：left|right|both；新版默认 right，both 产生两个手侧模型
- --run-name：必填，各手型/手侧放各自目录
- --checkpoint-root：默认上述根目录
- --input-profile：新版默认 camera_palm_length_window3_v1
- --resume：显式 model_last.pth；必须单手型、单目标侧并校验 run/data/spec/profile
- --init-checkpoint：只允许同规范、同手型、同输出定义的新权重，不隐式部分加载旧权重
- 新版无必填 --calibration，无 warmup-open-hand 阶段
- 多手型在开始前完成全部目标/路径/数据验证；不因后面某个拼写错误而先启动第一个训练。
- 目录存在时拒绝覆盖，除非严格匹配的 --resume；原子发布 checkpoint 和配置。

旧 --hand-side shared 的新训练语义不再混用；现有脚本与文档更新为明确手型和目标侧。
已有 shared L21 权重仍可由边界加载器按其真实声明读取，不自动转换成新规范。

## 9. CLI、运行、导出和记录

新公共控制：
hand control --hand-model <id> --backend nn|dex --hand left|right|both

NN：
- 单目标侧用 --checkpoint；双侧用 --left-checkpoint 与 --right-checkpoint。
- 单个旧 shared L21 checkpoint 只在元数据明确支持双侧且输入规范一致时接受双侧路由。
- 先验证目标、checkpoint、输入规范、依赖和资源，再打开相机/GUI。
- 不自动搜索最新权重，不因权重不兼容回退几何算法。
- 免人工掌长标定；正常三帧窗口就绪后可输出。

dex：
- 新相机规范不要求校准文件；旧 VR profile 的校准要求仍在其明确入口检查。
- 不把给旧路线的 --calibration 忽略后假装生效；不适用参数给出明确错误。

hand list-models 输出：
hand_model、每侧独立自由度、展开关节数、NN/dex/geometric 适配能力、是否需要训练权重。
“适配能力”不等于本地已有可用权重或模型效果通过验收。

hand export：
- 与训练/实时共用人体窗口和归一化函数，使用明确目标手型及 checkpoint。
- 去除固定 frame_count x 18 结果假设；旧文件转换集中于读写边界。

新 H5 schema：hand_joint_results_v1。
保存：时间戳、手型/手侧、关节定义、规格哈希、input_profile、
raw independent positions、accepted expanded commands、valid、error、耗时、
归一化分母诊断及算法参数。两种方法使用同一读写器。
读取旧 L21 和 dex_native_results_v1 时显式转换；重放 accepted commands 不再次滤波。
历史 VR 诊断格式可保留，但生产结果类型只有一套。

## 10. 命令与执行

- 由目标手有效 bounds/velocity_limits 处理独立坐标。
- 限速为 min(CLI limit, URDF pulled-back limit)，L21 <= 1.0 rad/s。
- 滤波/限速后展开耦合；严格检查从动关节位置和速度。
- 保留独立侧状态、丢手保持、恢复首步限速与时间戳校验。
- 具名 HandJointCommand 与 HandJointResult 分开，前者表示可下发命令。
- HandExecutor.apply(command) 只解析机器人身份、侧、规格哈希和关节名。
- L21 原有指间碰撞检查保留；其他手不声称有等价检查。
- 同一目标手进行方法比较时统一场景、执行模式和输出处理配置。

## 11. 文件修改地图

已有基础文件：
- contracts/hand_control.py：继续完善统一结果/命令边界校验。
- robots/hand_spec.py：补充参考姿态、掌长、坐标及任务点语义。
- tests/test_hand_specs.py：补充真实参考几何及速度测试。

修改：
- contracts/coordinates.py：新无量纲规范，旧规范保持真实语义。
- retargeting/hand/interface.py：唯一 HandRetargeter。
- retargeting/hand/transformer.py、predictor.py、checkpoint.py：参数化输出和明确的权重分派；旧网络边界仍保持原数值。
- retargeting/hand/dex_native.py：新目标手配置、归一化路线和公共结果。
- retargeting/hand/geometric.py、transhandr_vr.py：迁入公共结果，保持各自数值算法。
- learning/dataset.py、trainer.py、losses.py：通用人体窗口、目标手训练和损失；删除失去调用的 L21 专用公共分支。
- applications/training.py、hand_export.py、hand_realtime.py、hand_visualization.py、camera_record.py、dual.py、visionpro_hand.py、dex_native.py、realtime_hand_sim.py：迁移所有消费者。
- applications/replay/ 下相关手部回放：旧格式边界读取，公共命令执行。
- simulation/hand_commands.py、realtime_teleop.py、three-robot/L21 场景调用：统一结果/命令；不修改手臂 IK 算法。
- robots/l21.py、native_hand.py：保留机器人规格和明确历史映射，移除被公共规格取代的重复定义/生产包装。
- data/hand_h5.py、hand_recording.py：通用 schema 与显式历史格式读取。
- cli/schemas.py、registry.py、checkpoints.py、paths.py：训练/控制/导出参数、路径和加载校验。
- 受影响 devtools、README、docs/architecture.md、docs/contracts.md、docs/dex-native.md、docs/hand-command-safety.md：更新实际入口与语义。

新增：
- retargeting/hand/normalization.py：唯一归一化数学实现。
- retargeting/hand/hand_model.py：公共任务目标、可微 FK/耦合适配。
- applications/hand_control.py、hand_control_factory.py：新公共循环与依赖组装。
- simulation/hand_executor.py：按身份和关节名执行。
- data/hand_control_h5.py：新公共结果格式读写（已有 writer 迁移后删除重复实现）。
- 目标手训练/模型若必须拆文件，则只拆一个明确职责，替换调用后删除原重复实现，不保留 joint_interface.py 等平行公共 API。

删除/替换：
- NativeHandRetargeter 基类、DexRetargetingRetargeter 占位类。
- HandRetargetResult、NativeHandResult 两套生产结果。
- L21HandAngles、L21HandDOFs 作为公共流程包装。
- 含上述类型的 re-export、旧调用方、过期注释和未使用工厂。
- 被新训练/导出替代且不再调用的固定 18D 生产分支。
保留：历史读写所需明确命名的 18D/17D 转换函数、真正使用中的机器人映射、上游原始网络内部输出。

## 12. 分任务执行与验证

每项先写行为测试并确认 RED，再实现最小代码并确认 GREEN；共享行为运行完整回归。
每项完成记录命令和结果。用户验收前不提交。

### Task 1：完善公共规格与结果
- [x] 核对已有基础实现，不重复建立已完成的 8 项测试。
- [x] 增加参考掌长/坐标/任务点与身份校验，未知或退化规格拒绝。
- [x] 测试耦合链、零倍率、负倍率、速度约束、真实四种手型与合成 21 DOF。
- [x] Run: python -B -m unittest tests.test_hand_specs -v
Expected: 全部通过；真实参考点与 URDF/FK 对照，不从被测函数自身计算 expected。

### Task 2：实现共享掌长归一化
- [x] Test: tests/test_palm_length_normalization.py；先运行失败。
- [x] 验证均匀缩放 0.7/1.0/1.3 的等价窗口归一化相同，容差 1e-6。
- [x] 验证三帧中位数、同分母、腕点归零、输入不修改、左右独立。
- [x] 验证非有限/零长度/1e-6 边界、重复归一化拒绝、无未来帧依赖。
- [x] 实现函数与新 metadata 校验。
- [x] Run: python -B -m unittest tests.test_palm_length_normalization -v
Expected: 全部通过，尺度诊断为米，输出 units=1。

### Task 3：Dataset 与流式等价
- [x] Test: tests/test_shared_human_dataset.py。
- [x] 同一 H5 离线与逐帧模拟得到相同窗口、目标、mask，容差 1e-6。
- [x] 覆盖缺手、恢复、gap、train/validation 边界和拒绝错误数据来源。
- [x] 两个目标手得到相同人体样本/split identity，H5 SHA256 不变。
- [x] Run: python -B -m unittest tests.test_shared_human_dataset tests.test_world_hand_pipeline -v
Expected: 新规范正确，旧 world 数据与旧权重不被隐式归一化。

### Task 4：公共目标与可微 FK
- [x] Test: tests/test_hand_model.py。
- [x] 验证固定参考掌长、左右目标坐标转换、root-to-tip 任务点。
- [x] 零位、边界、随机有效姿态对照独立 PyBullet/现有 FK；最大位置差 <=1e-5 m。
- [x] 代表性非退化姿态执行 float64 梯度检查；包含耦合关节。
- [x] 验证人体均匀缩放不改变新规范的机器人目标。
- [x] Run: python -B -m unittest tests.test_hand_model -v
Expected: FK、单位、目标和梯度通过。

### Task 5：公共方法接口及旧能力迁移
- [x] Test: tests/test_joint_retarget_adapters.py，迁移相关旧断言。
- [x] NN/dex/geometric/VR 返回同一 HandJointResult，不改变原算法结果。
- [x] 旧 L21 18D 除占位后逐项一致；旧 VR 上游原始结果一致。
- [x] 验证 invalid/reset/side/source 行为。
- [x] Run: python -B -m unittest tests.test_joint_retarget_adapters tests.test_transhandr_vr -v
Expected: 公共结果语义一致；VR 源码哈希与数值等价记录完整。

### Task 6：动态 NN、训练与权重管理
- [x] Test: tests/test_joint_training.py、tests/test_joint_checkpoints.py。
- [x] 7/17/21D 输出测试，错误手型/同维异义/输入规范/侧/关节顺序加载失败。
- [x] 实现规范化窗口训练、可微目标损失及按手型依次训练。
- [x] 同一 H5 为 L21 与 G1 做 CPU 1-epoch smoke，各自保存 17D/7D 模型；H5 哈希不变。
- [x] 目录创建前校验全部手型；目录已存在拒绝覆盖；原子保存。
- [x] 检查 best/last/config/dataset/metrics/log，恢复训练包括 RNG 和 epoch 次序，不混用旧模型。
- [x] Run: python -B -m unittest tests.test_joint_training tests.test_joint_checkpoints -v
Expected: 训练/重载/恢复闭环通过；smoke 不冒充动作精度验收。

### Task 7：新版 dex 与 L21 同手型求解
- [x] Test: tests/test_normalized_dex.py。
- [x] 先核对已安装上游优化器的 scaling、损失、容差、正则及单位；记录源路径/哈希。
- [x] 实现等价归一化目标，保留明确的旧 VR profile。
- [x] 用真实 L21 FK 的可达目标检查求解、限位和残差，不要求冗余角唯一反求。
- [x] NN/dex 同一手型的 key/names/D/units 一致，换人体整体尺度不需校准。
- [x] Run in dex environment: python -B -m unittest tests.test_normalized_dex -v
Expected: 无必填掌长校准，错误 profile 不静默回退，旧 VR 回归通过。

### Task 8：公共命令与场景执行
- [x] Test: tests/test_joint_commands.py、tests/test_hand_executor.py。
- [x] 检查 min(CLI,URDF) 速度、L21 1.0 rad/s、从动关节、丢手和恢复、长间隔。
- [x] 具名执行拒绝错误身份、hash、缺失/额外关节，未选择的手和手臂不受写入。
- [x] 保留 L21 碰撞检查的实际范围；算法比较使用相同执行模式。
- [x] Run: python -B -m unittest tests.test_joint_commands tests.test_hand_executor tests.test_hand_commands -v
Expected: 仅用户批准的速度收紧改变旧行为，其余相应行为断言保留。

### Task 9：CLI、导出、记录与回放
- [x] Test: tests/test_hand_control_cli.py、tests/test_hand_control_h5.py。
- [x] 接通 hand train/control/export/list-models，旧有效入口转调公共实现。
- [x] 错误参数/模型/规范在设备打开前失败，help 不加载模型或开设备。
- [x] 新相机路线无需 --calibration；错误给出的旧参数明确报错。
- [x] 检查双侧 checkpoint 路由，避免 shared 与两个独立模型混淆。
- [x] 新旧 H5 显式读写/转换，记录归一化诊断，回放不重复滤波。
- [x] Run: python -B -m unittest tests.test_hand_control_cli tests.test_hand_control_h5 tests.test_cli_integration -v
Expected: 训练到控制、导出与回放接通，格式和资源退出正确。

### Task 10：删除旧接口、全消费者迁移和文档
- [x] 迁移实时/dual/VR/capture/replay/devtools 所有调用点和测试。
- [x] 删除旧结果类型、旧基类、占位类和重复工厂；不留兼容别名。
- [x] AST/import 边界测试防止生产代码再次依赖已删除类型；仅历史格式边界允许显式18D转换。
- [x] 更新帮助、README、接口/训练/权重/输入规范和验收文档，移除误导性旧注释。
- [x] Run: python -B -m unittest discover -s tests
Expected: 普通环境与 dex 环境均通过，原 skip 明确列出，数值容差不放宽。

### Task 11：方法比较与跨操作者验收
- [x] Test: tests/test_hand_method_comparison.py：按时间戳/共同有效帧对齐，明确窗口预热和缺失行为。
- [x] 同一手型、同一人体数据、同一预处理/任务目标与输出处理，分别报告原始解和执行命令。
- [x] 指标：归一化及米制指尖误差、捏合距离、限位、速度、失败率、求解耗时。
- [ ] 跨操作者按人划分训练/验证/测试，禁止同一操作者相邻窗口充当跨人泛化证明。
- [ ] 至少不同操作者真人运行，无张开手标定，记录初始化、移出/恢复、人体大小和检测质量。
- [x] 若没有跨操作者数据/目标手权重，停在“框架与数值测试通过，真人泛化待验收”，不宣称目标完全达到。
- [x] 最终独立审查和完整回归完成，交给用户验收。
- [ ] 用户验收后才创建约定提交；当前未提交。

## 13. Review Focus

1. 单位：米制 H5/旧模型、无量纲窗口、机器人米制目标之间混用或重复缩放。
2. 因果一致性：离线用了未来尺度、整段统计或跨划分窗口，实时无法复现。
3. 身份：同维度不同手型、左右手、关节顺序和旧权重误加载。
4. 状态：换操作者或缺手恢复沿用旧尺度、solver/filter 状态导致跳变或串扰。
5. 单一接口：旧 L21/dex 类型未清理、遗留重导出或旧注释误导后续 agent。

对应任务分别为 2/4/7、3/11、1/5/6/9、2/5/7/8、10。
每类都有行为测试，不以单纯文本搜索替代验证。

## 14. 验证环境与交付规则

普通解释器：
D:/Anaconda/envs/teleoperation/python.exe

dex 解释器：
D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Scripts/python.exe

设置 PYTHONPATH=src、PYTHONDONTWRITEBYTECODE=1、UTF-8；
TEMP/TMP、绘图和模型缓存放 outputs 内。完整测试中硬链接需要足够执行权限。
日志写 outputs/tmp_interface_integration/，读结果尾部和完整错误，不掩盖基线失败。

每项执行记录更新本计划；用户未验收前全部变更未提交。
不修改尚未约定的算法行为；不把测试支持的结论扩大到真人泛化、实时帧率或真机安全。

## 15. 本轮执行记录

- 用户确认本修订计划及具体数值规则；无需人工掌长校准。
- Task 1 参考几何：先观察缺少参考属性/未拒绝错误 frame 的 RED；新增 NumPy-only URDF 子树参考计算和 HandSpec 参考掌长/坐标变换；10 项规格测试通过。
- Task 2：先观察缺少 normalization 模块的 RED；实现三帧中位数归一化、新坐标规范及单位校验；5 项归一化测试通过。
- Task 3：先观察 Dataset/pipeline 不接受 input_profile 的 RED；新增明确 profile，离线/实时调用同一函数，目标使用归一化最新帧，独立清理无效侧历史；与旧 world 回归合计 27 项通过，日志 outputs/tmp_interface_integration/palm_dataset.log。
- 正在运行上述共享修改后的完整普通环境回归：outputs/tmp_interface_integration/normalization_full.log。
- 归一化共享修改完整回归已通过：483 项，1 项原有跳过，82.652 秒。
- Task 4：先观察缺少 hand_model 模块的 RED；实现公共任务向量与可微 FK。四种手型双侧的零位/非零位均与 PyBullet frame 对照在 1e-5 m 内；L21 和 H1 耦合梯度检查通过；3 项测试通过，日志 hand_model.log。
- 用户额外确认：旧单手 VR 目标为左手时，也必须执行左手原生角度适配；上游原始 18D 值单独保存作诊断。不能保留错误的「右侧语义放左槽」行为。
- 上游源码 SHA256：main_visionpro_realtime.py = f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f；model/model_poseformer.py = 9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3；checkpoint/models/thumb3/linker/model_final.pth = ffc4aeb4fd3f8c5b6687e7d5378da43f459480eee355e59a46eac65a34c8c895。根目录 D:/2026/code/TransHandR/TransHandR。
- Task 5 NN/几何/VR 结果迁移中：新接口测试先 RED 后 GREEN；相关 111 项定向回归仅剩原 CLI 文本断言仍期待18D，已按新17D输出更新，正在复核。dex 与旧结果类型的最终删除尚未完成，不能标为整个接口迁移完成。
- 上述 CLI/NN 复核8项通过。dex改为公共独立坐标结果，命令层输出独立的 HandJointCommand；dex/旧结果回放及具名场景定向18项通过。
- 为完成任务5消费者迁移，提前实现任务8的公共命令基础类，3项测试通过；后续仍需把L21旧命令循环转入该实现并删除旧固定18D命令路径。
- 已删除旧公共结果包装和 NativeHandRetargeter/占位类；迁移所有生产导入及开发工具。历史有损L21映射的对照文件仍以专用诊断schema保存，不伪装满足耦合的原生命令；增加了不可回放为命令的行为测试。
- 正在进行上述接口清理的完整普通环境回归，日志 interfaces_full.log。新NN训练、免标定dex、公共控制CLI及新H5 schema尚未完成。

- Task 5 全部消费者已改为公共结果，VR 左侧始终原生适配且保存 raw18；已删除旧 Native/L21 结果包装和固定18D公共命令类。历史 L21有损显示明确放在 apply_legacy_l21_mapping 边界。
- Task 6：动态头与严格 checkpoint 测试先 RED 后 GREEN，7/17/合成21D 输出验证通过。新 learning/trainer.py 替换旧专用训练入口，公共H5顺序训练多手型；G1续训与连续两轮逐项相同；左右目标侧、数据/config/checkpoint identity 严格校验。目标限位 buffer 不能覆盖 spec 的回归先 RED 后 GREEN。日志 joint_training.log / identity_green.log。
- Task 7：核对安装 dex 0.5.0 VectorOptimizer 源码，其 scaling 仅乘人体目标。metric SmoothL1 = Lr * normalized SmoothL1，使用 beta=.02*Lr、ftol_abs=1e-6*Lr；xtol 保持 rad，norm_delta=0。记录于算法 metadata。实际 dex 环境四手型双侧归一化路线测试通过，不改变5mm G1阈值。
- Task 8：JointCommandPipeline 与 HandExecutor 已接入新控制和旧相机/VR消费者；L21 URDF 速度1rad/s，滤波后展开耦合。8项滤波/网格检查迁移通过。新控制用mock acquisition +真实PyBullet验证具名执行和H5记录，真人相机未验收。
- Task 9：hand train/control/export/list-models、hand inspect及replay hand-joints 已接通；统一 hand_joint_results_v1 writer/reader。导出保留预热和缺手完整时间线，L21导出在DIRECT场景中保留默认网格检查。新/历史输入规范显式区分，无权重回退。
- Task 10：删除无调用的旧训练/报表实现；历史六项loss/FK只在诊断模块保留。开发工具run_retargeting_pipeline改为新命令，不再伪造normalized数据给新训练。核心架构/contracts/VR文档已更新，新操作指南 docs/hand-retargeting-unified.md；旧历史工作流已明确标记适用范围。
- Task 11：比较测试先RED后GREEN；同输入哈希/profile/命令策略/目标向量、共同有效时间戳，分别评价raw和command误差、捏合、限位、速度、无效比例和耗时。没有跨人数据/真人验收，不宣称泛化。
- VR实际来源数值复核完成：devtools/verify_vr_both.py + 20261012_213812_live 实录 + thumb3实际权重，左右各10窗口、scale1/1.7，输入与raw输出最大误差0；右FK轴/offset一致。报告 outputs/tmp_interface_integration/vr_equivalence.json。
- 中间全套回归定位并修复了旧测试夹具/CLI参数、无第三方包帮助依赖、缺失文件原错误消息与测试缩进；所有具体失败已保留日志。最终普通/dex全套与独立审查尚在执行，暂不标为完成。

- 首个完整普通环境通过：504项、2跳过（原有skip与dex条件测试），89.474s；dex环境504项、1原有跳过，95.202s。之后仍有独立审查修复，不能把此前全套当作最终提交门禁。
- 独立审查第一代理因服务额度中断；备用新上下文审查完成。已确认并修复：单侧预热被反复清空；计划§8的checkpoint-root/input-profile/init-checkpoint遗漏。初始化同规范/目标/输出严格加载，所有目标预检在创建首个训练目录前完成；恢复参数互斥；训练架构从兼容新checkpoint重建。相关回归先RED后GREEN，training_options_green.log。
- 审查另确认2项Important：预热中的过小掌长旧帧延迟恢复；场景初始化失败连接泄漏。新增共享逐帧米制有效性检查，入buffer前排除坏侧，窗口仍使用原三帧中位数公式/阈值；两种场景构造器失败释放连接。15项相关测试通过，review_fixes_green.log；包括实际DIRECT连接失败探针。
- 审查Minor：相机依赖需在GUI前预检；已增加只读cv2/mediapipe导入检查，未提前打开设备，失败顺序测试先RED后GREEN。
- 新checkpoint按计划补齐reference_geometry配置（root/palm/tips/human indices/参考姿态规则），严格加载测试先RED后GREEN，geometry_metadata_green.log。
- 真实录制数值smoke：datasets/raw/aligned_camera_20261015_115030.h5前80帧另存到outputs，不修改原件；L21/G1右侧CPU小网络各1epoch，NN/dex同输入导出、指标比较、accepted具名回放全通。报告outputs/tmp_interface_integration/real_camera_smoke_v1/report.json，source哈希未变。仅为数值集成证据，不作为训练质量/不同操作者验收。
- 收尾：删除无生产调用的固定18D批量导出函数和重复命令构造函数，保留有真实调用的历史文件/FK/诊断边界。静态搜索无已删除的旧公共类型/命令管线导入，git diff --check通过。

- 最终审查修复后普通全套509项、2跳过，90.688s通过；真实录制指标随后发现首次恢复按名义1/30s而实际间隔26ms，造成1次超速。保持原恢复步长上限，并对上一条保持命令的更短实际间隔施加限制，落实既定URDF限速，不改指标/容差。新增回归先RED后GREEN，actual_recovery_speed_green.log（31项）。
- 修复后重跑真实录制smoke_v2：L21/G1各80帧、共同有效78帧；NN和dex accepted commands 的速度超限均为0；输入原件hash未变。最终报告 outputs/tmp_interface_integration/real_camera_smoke_v2/report.json。v1仅为定位过程记录，不能作修复后证据。
- 计划Task7补充真实L21双侧FK可达目标求解（不要求角度唯一），残差低于既有5mm米制检查阈值；同手型NN/dex key/hash/joint_names/D一致。测试 tests/test_normalized_dex.py 在dex环境通过。
- Task9 H5行为测试按计划文件名整理为 tests/test_hand_control_h5.py；未保留旧测试模块别名。
- 上游dex安装源码：D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Lib/site-packages/dex_retargeting/optimizer.py SHA256=6b816d65c81f2e959ea6b76876bb03c2b768be95064a1b4b9e1563d36262cdf8。只读核对，没有修改上游优化器。
- 上游dex安装源码：D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Lib/site-packages/dex_retargeting/seq_retarget.py SHA256=72cc74f1a0ff91fafc9421d3d6766a7dd2279b0964e0b376718ec385fbb94482。只读核对，没有修改上游优化器。

- 最新最终普通环境：511项，0失败/错误，3跳过（1原有 + 2仅dex环境），156.379s；日志 outputs/tmp_interface_integration/final_normal_verified.log。dex完整回归正在执行，所有生产代码保持该测试版本不再修改。

- 最终dex环境：511项，0失败/错误，1原有跳过，122.169s；日志 outputs/tmp_interface_integration/final_dex_verified.log。普通环境同版本511项/3跳过已通过；git diff --check通过，CLI无第三方包帮助通过。
- 最终审查结论With fixes：发现的Important与Minor均已复现、修复并经定向及完整回归验证。没有待修复的已确认代码问题；保留真人相机/VR连续运行、按操作者划分泛化、GPU逐位复现与真实新增21DOF资产等待验收，不扩大数值测试结论。
- 提交门禁：HEAD仍为11abbfa21e5e53675f179f5a9569aecaf612a9cc，注解checkpoint-20261029-1156-hand-retargeting-unified保留。全部改动未提交，未推送；按用户指示验收后提交“重定向输出接口整合”。
