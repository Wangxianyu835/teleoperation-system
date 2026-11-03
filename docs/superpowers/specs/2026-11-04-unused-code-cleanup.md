# 无调用旧代码分档清理范围

日期：2026-11-04。源码审阅基线：`0b20331162fa04f785f45040e1056cf96e998cc5`。

## 用户要求

将前次源码审阅发现的代码分为两档：

1. 已不再使用的旧代码全部删除，包括文档中对应的旧内容。
2. 仍有部分在使用的文件，只删除没有用上的部分；剩余内容交给用户处理。

当前交付仅为实施计划，不执行源代码、测试或现有使用文档的清理。

## 清理判断

- “使用”包括CLI注册、模块启动、静态/动态导入、应用调用和有意维护的诊断/验收脚本；不能只统计生产import。
- 仅由旧实现测试维持的接口可退出：迁移对当前系统仍有价值的测试，删除专门验证废弃实现的测试。
- 没有仓库内消费者的公开接口，按本次用户授权退出，不保留兼容别名或转换器。
- 测试混有当前业务时，只删除废弃用例及专属fixture，不删除整个测试文件。
- 有当前消费者的历史算法、CLI、profile或数据格式继续保留，不以名称中包含legacy/normalized/18D认定无用。
- 执行前重新核查下列清单。如新增当前消费者，将该项移入保留清单并说明原因，不扩张到迁移活跃流程。

## 第一档：整模块删除范围

路径以仓库根目录为基准：

1. `src/teleoperation/applications/visionpro.py`。
2. `src/teleoperation/retargeting/hand/diagnostics.py`。
3. `src/teleoperation/simulation/hand_mount.py`。
4. `src/teleoperation/tools/hand_reports.py`。
5. `src/teleoperation/tools/training_hand_pose.py`。
6. `src/teleoperation/learning/config.py`。
7. `src/teleoperation/retargeting/hand/tracking.py`；同步删除processing中的闲置导入。
8. `src/teleoperation/inputs/interface.py`；同步把CLI帮助中InputSource类名改为实际支持的next_observation()/迭代协议描述，不改变运行时协议。
9. `src/teleoperation/applications/hand_export.py`；同步删除hand.py中无调用的export包装，保留当前CLI直接handler。
10. `src/teleoperation/data/world_landmark_recording.py`；同步删除LegacyWorldRecordingTests及专属fixture，保留同测试文件的组合录制用例。

## 第二档：按符号删除范围

- `learning/dataset.py`：TwoHandH5ChunkedGenerator、_stack、_mask、_is_valid_hand_frame；保留TwoHandH5Dataset、iter_samples、_scalar_or_none。
- `data/checkpoint.py`：write_checkpoint；保留read_checkpoint。
- `robots/action_mapping.py`：_NativeActionAdapter、H1ActionAdapter、GR1ActionAdapter、G1ActionAdapter、pack_tron2a_l21_action、Tron2AL21ActionAdapter；保留validate_native_action。
- `contracts/commands.py`：只删除上述适配器专用的H1JointTargets、GR1JointTargets、G1JointTargets；保留Tron2AL21Command及其flatten。
- `robots/partial.py`：pack_partial_limb_vector；保留PartialLimbVector及验证器对部分动作的拒绝。
- `retargeting/hand/postprocessing.py`：classify_jumps、detect_bad_frames、hold_last_valid、detect_identity_swaps、repair_stream、repair_bad_frames、moving_average、motion_metrics、compare_methods；保留KalmanSmoother及其方法和默认参数。
- `retargeting/hand/interface.py`：JointNNRetargeter及__all__条目；保留HandRetargeter和SharedJointNNRetargeter。
- `retargeting/hand/checkpoint.py`：joint_checkpoint_payload、load_joint_checkpoint；保留shared payload/loader、save_joint_checkpoint和它们依赖的校验。
- `applications/dex_native.py`：LatestCameraInput、visionpro_run、仅供visionpro_run使用的command_pipeline，及随之变为闲置的导入；保留calibrate、recording_frames、file_hash、canonical_window。
- `applications/realtime_hand_sim.py`：删除调用不存在main的__main__尾部及其专属说明；保留场景/预览类和实际调用的辅助函数。

## 文档范围

- 修改README.md、docs/current-interfaces.md、docs/unified-hand-input.md中的退役模块、旧API、旧调用方式及错误的现状描述；没有命中时不为改文档而改文档。
- 清除被删函数的docstring示例和相关注释，例如postprocessing中classify_jumps使用建议。
- 删除已被本范围替代的2026-11-03旧删除计划及对应spec，避免其中迁移活跃legacy臂/normalized相机分支的任务被继续执行。
- 保留当前可用命令、当前camera profile、上游TransHandR源码路径/哈希、现有验收事实；历史测试数字只能标记为历史，不能声称是本次结果。
- 新计划和spec中的名称是删除对象定位清单，不是旧API使用说明；完成时勾选并记录真实清理结果。
- 不改写AGENTS.md、不改Git历史，不删除外部审阅报告、datasets、模型权重和outputs。

## 剩余内容交给用户处理

- hand dex-calibrate、DexInputAdapter、DexCalibration、DexNativeRetargeter及dex_native_acceptance.py所需依赖。
- MediaPipePalmLocalPipeline、camera_palm_length_window3_v1及训练/三机器人/TRON2A调用方。
- 双臂legacy映射、normalized诊断分支、training_run_directory当前接受的侧目录参数。
- 三处活跃模型构造、GBK双扫描工具、已有多种活跃录制schema；不在本次抽公共工厂或合并格式。
- 显式TransHandR流程、原始18D数值、FK、KalmanSmoother、身体跟踪、原生关节执行/录制/回放。
- 有现存用途但需维护的preflight.ps1环境工具。本次不重写环境引导逻辑。

## 全局约束与验收

- 不新增兼容层，不迁移活跃数据接口，不修改重定向数值算法，不增加依赖。
- 保留RGB/VR并列输入与相同NN/dex后端选择；不把VR来源标记为MediaPipe。
- 当前输入、窗口、结果、shared权重身份和录制接口保持原行为；删除范围内的未使用旧API按本授权退出。
- 每批修改前检查Git状态、保存dirty intended state、创建annotated checkpoint；验证后提交，报告hash/tag，不推送。
- 运行最小相关unittest；涉及共享行为时运行完整unittest discover -s tests；dex跳过项在独立环境补跑，逐项记录。
- 删除对象没有残余业务引用或文档使用说明；活跃入口、规格校验、缺手保持、左右适配及回放仍通过回归。
- 真实相机、头显和实体硬件没有执行时，明确记录未验收，不将离线/模拟测试写为硬件成功。
