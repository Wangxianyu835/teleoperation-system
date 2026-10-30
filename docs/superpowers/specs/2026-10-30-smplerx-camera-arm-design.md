# SMPLer-X 相机机械臂输入替换设计

日期：2026-10-30。状态：供用户评审，尚未实施。

## 文档规则

用户指定的规范是 ASD-STE100。该规范用于受控英语技术写作，不规定软件架构。本文使用中文，并采用其清晰写作原则：固定术语；使用主动语态；每条操作只描述一个动作；写明输入、输出和通过条件。本文不声明通过 ASD-STE100 英文词典或全文合规检查。

术语固定如下：SMPLer-X 是图像估计模型；SMPL-X 是参数化人体模型；手臂输入包括肩肘腕位置和腕朝向；手指输入包括现有手部关键点；旧录制是已有的 H5/NPZ 文件。

## 需求与边界

用户要求将 SMPLer-X 接入现有工程，删除 MediaPipe 驱动机械臂的代码，由 SMPLer-X 替代。背景问题是单相机自遮挡、肩肘腕估计误差和跳变造成机械臂抖动。

- 相机机械臂的位置、骨段方向和腕朝向统一来自 SMPLer-X / SMPL-X。禁止用 MediaPipe Hands 的腕点或掌朝向补驱机械臂。
- MediaPipe Hands 继续服务现有手指链路；NN/dex 手指输入、训练规范、窗口与结果接口兼容。
- Vision Pro / TransHandR 数值链路继续显式独立；不套用相机预处理或新相机标定。
- TRON2A 是可选机器人之一。新感知结果不以机器人型号分叉；其现有相机服务也迁移，不能留下一条活动的 MediaPipe Pose 控制路线。
- 本次替换感知输入，并为遮挡、过期结果、恢复加入时序处理；现有重定向/IK 后端先保留。dex 风格机械臂向量优化作为后续独立改动，便于归因性能差异。
- 旧 MediaPipe 手臂录制、来源标签、旧手臂 reader 和旧手臂标定不属于新版本兼容范围。新相机机械臂不再加载或调用 PoseLandmarker。

本次计划不创建或依赖新的本机启动脚本。用户当前从 `teleoperation` 环境运行工程。仓库内的历史 launcher 不是本机启动方式的证据。修改现有模块入口和 CLI 即可接入新模型。

## 现状与风险

基线：`84af227fa881d2fb15bfdc5f8e94615e3619e39e`；检查点：`checkpoint-20261030-1336-smplerx-arm-plan`。

`inputs/mediapipe.py` 同时持有相机、Hands 和可选 Pose。`simulation/realtime_teleop.py` 使用该组合观测，`RealtimeArmBridge` 生成位置/肘目标，`WristSceneController` 从 Hands 腕观测生成朝向。TRON2A 的 `applications/mediapipe_tron2a.py`、`retargeting/arm/calibration.py`、`retargeting/arm/retargeter.py` 硬编码了 MediaPipe 坐标来源。采集与 H5 写入也硬编码该来源。

`applications/arm_validation.py` 默认使用旧 chain 路线，没有复用实时入口的独立腕控制；必须改为与当前控制路径一致，不能用旧验收替代新路线验收。

实机预检记录：RTX 4050 Laptop，6 GB 显存；内存约 31.6 GiB；主环境 PyTorch 2.14.1+cu130，CUDA 可用。未加载 SMPLer-X，无显存峰值和实时帧率证据。

2026-10-30 的本机环境复核见 [环境核查](../../smplerx-environment-audit-2026-10-30.md)。主环境解释器是 `D:/Anaconda/envs/teleoperation/python.exe`。主环境和 dex 环境的 `pip check` 均通过，已检查的主要模块均可导入。主环境 NumPy 2.2.6 与 `environment.yml` 的 1.26.4、`requirements-retargeting.txt` 的 `<2` 不一致。dex 使用 `.venv/dex-native`，并通过 `.pth` 加载 `.venv/dex-pin-bundle`。两个路径同时暴露不同版本的 SciPy 元数据，当前实际导入 1.13.1。这些是重建风险，不能由 `pip check` 排除。

## 方案选择

| 方案 | 取舍 | 决策 |
|---|---|---|
| 在主环境直接引入官方依赖 | 部署路径短，但旧 OpenMMLab/PyTorch 依赖可能与当前工程冲突 | 不作为首选 |
| 独立 SMPLer-X 推理进程，主工程通过版本化本地协议接入 | 需管理协议、进程和过期结果，但隔离依赖、便于 GPU 故障处理 | 推荐，作为本计划默认方案 |
| 保留 MediaPipe Pose 并融合两者 | 需要冲突处理，且不满足用户替换要求 | 不采用 |

默认使用官方 S32 预训练模型、单人、batch size 1、完整官方前向；第一版不裁剪网络、不重新训练、不实时渲染网格。先验证 FP32，混合精度仅在数值和性能对比通过后启用。

独立进程可位于兼容的本机环境。若 Windows 原生依赖无法稳定运行，再使用支持 CUDA 的 WSL2 工作进程；先验证环境能力，不预先声称某套旧版本组合适用于 RTX 4050。环境锁文件记录实际通过的版本。

## 数据与运行结构

一次捕获 RGB 帧，分发给 Hands 和 SMPLer-X；相机只有一个所有者。Hands 保持既有处理顺序；SMPLer-X 每个控制进程只有一个实例，估计一次人体后供所有机器人消费。

工作进程通过本地子进程 stdin/stdout 的长度前缀二进制协议交换帧与结果；stdout 只承载协议，日志写 stderr。协议带版本、序号、源采样时间和会话标识。活动帧最多一个，待处理帧最多一个，新帧替换等待帧，不阻塞手指/控制循环。

源图像时间、主机单调时间、推理完成时间、控制时间分别记录。同一个结果重复读取不能增加有效帧数或标定进度；超时依据源采样时间，不能给缓存重新打时间戳。单人首次锁定后不因面积变化自动切人，丢失或身份歧义时保持并重新确认。

SMPL-X 局部旋转沿骨架计算全局朝向；关节取同一次模型计算、同一坐标系中的肩肘腕。图像预览点通过相机投影及逆裁剪得到，禁止把模型内部混合的 root-relative / hand-relative 点直接拼接。

源坐标标记为 `smplx_camera_root_relative_m`，图像投影标记为 `smplx_image_normalized`；单位、原点、轴方向和矩阵乘法规则通过上游源码与合成旋转实验确认。新来源不能重标为 `mediapipe_pose_world_meters`。平移只使用明确的相对参考，不将单目 cam_trans 宣称为真实机器人外参。

腕观测标记为 `smplx_camera_axes` / `smplx_wrist_local`。保留通用 WristObservation 字段，局部人体腕轴到机器人工具轴通过中性对齐处理。相机手臂消费者只读取新腕来源；手指原始腕诊断可保留并独立标记。

`RawTeleopObservation` 增加可选的 `wrists` 字段，默认 `None`。新相机入口把 SMPL-X 腕写入此字段。`RawHandFrame.wrists` 仍表示原手部来源，不覆盖为 SMPL-X 数据。机器人相机腕控制只读取组合观测的新字段。VR 保留自己的来源分派，旧 MediaPipe 手臂回放分派删除。

## 连续性与遮挡

统一输出 `calibrating / tracking / hold / lost_hold / reacquiring / fault` 状态。检查有限值、人体检测状态、骨段退化、样本时间、姿态/骨段变化与追踪状态差异。SMPLer-X 没有对应可靠可见性时不伪造 visibility=1 或置信概率。

默认控制策略：样本年龄上限 0.5 秒；瞬态异常保持上一可靠目标；连续 3 个相互一致的新样本才确认恢复，恢复受位置和旋转速率限制。3 帧一致只表示恢复资格，不能绕过速率约束。初始参考采用现有 30 个有效身体样本和 10 个稳定腕样本的约束，按同一会话管理；不得用重复缓存累计样本。

位置、方向、关节速度/加速度的数值复用各机器人现有已验证设置；新增设置需在验收前写入配置及报告，不能在失败后放宽。预测第一版默认关闭，长期遮挡保持而非猜测运动。人体体型在完成会话初始稳定估计后固定，并用固定体型重算派生关节，原始参数仍完整记录。

## 兼容性与删除

“保留手指和 VR 接口，并清理旧手臂兼容代码”的具体含义如下。

| 对象 | 保留的能力 | 与新手臂输入的关系 |
|---|---|---|
| 手指 | MediaPipe Hands 检测、现有 NN/dex 手指处理、HandWindow 和关节结果接口 | Hands 只驱动手指；它的腕点和掌朝向不再驱动新相机手臂 |
| VR | Vision Pro / TransHandR 输入格式、数值处理和设备入口 | VR 继续直接提供原有观测，不经过 SMPLer-X |
| 旧手臂录制 | 不提供旧 MediaPipe 手臂回放兼容 | 新版本拒绝旧手臂来源字段 |
| 旧手臂标定 | 不提供旧来源标定兼容 | 新 SMPL-X 输入必须生成新标定 |

接口兼容不表示保存旧实时算法或旧手臂录制。旧 Pose 初始化、检测、驱臂、reader 和标定分支会被删除。错误使用旧 Pose 参数或旧手臂数据时，入口返回迁移说明并停止处理。

- 保持 RawHandFrame、UpperBodyObservation、RawTeleopObservation、HandWindow 既有调用可用；需要附加字段时使用可选默认值，不变更旧字段含义。
- 新录制和标定只接受 SMPLer-X / SMPL-X 来源；旧 MediaPipe 手臂来源直接拒绝，不通过改 metadata 转换。
- 删除 PoseLandmarker 运行路径、专用实时参数与模型检查、手部位置兜底驱臂路径；废弃公共调用参数保留显式迁移错误信息，不能继续启动旧后端。
- 删除旧手臂录制 reader、旧手臂来源字段和旧手臂回放测试。磁盘上的历史文件不由安装或迁移脚本自动删除，但新程序不读取这些文件。

TRON2A 当前在 `simulation/arm_profiles.py` 中仍为 `loadable=False`。本计划不把注册状态解释为已经可以从通用入口选择。新相机感知不依赖机器人型号，现有 TRON2A 专用服务使用同一新感知输入。将 TRON2A 完整接入通用机器人加载器需要单独完成资源、关节、末端和 FK 验证；不能仅改一个布尔值。

## 验收

依次执行：模拟协议与坐标单元测试、原始推理 smoke、确定性姿态回放、GPU 全链路视频比较、真人五分钟验收。缺权重、模型文件或兼容环境时，可完成无权重测试，但整体状态只能记为未完成。

S32 工程可用性的暂定目标：完整仿真与手指链路并行时，可靠身体结果更新至少 10 Hz，采集到新人体结果可用的 p95 延迟不超过 150 ms；这是设计目标，非已达指标。控制周期单独记录，不能用重复目标冒充人体更新频率。

同一 RGB 视频比较静止抖动、跳变频次、遮挡错误动作、保持比例、有效跟随比例与恢复延迟。无三维真值时只报告稳定性与代理指标，不宣称真实三维精度提升；改善至少要求主要稳定性指标下降且覆盖率/延迟无不可接受退化，由验收报告呈现取舍。

## 参考

- https://www.asd-ste100.org/about.html — 受控技术英语的适用范围。

- https://github.com/MotrixLab/SMPLer-X — 模型、部署及 S32 配置。
- https://github.com/MotrixLab/SMPLer-X/blob/main/main/SMPLer_X.py — 参数输出、局部/相对坐标处理。
- https://github.com/MotrixLab/SMPLer-X/blob/main/main/inference.py — 图像预处理、人体裁剪与推理流程。
- https://arxiv.org/abs/2309.17448 — 方法与评估。

实施时记录上游 commit、关键文件 SHA256、权重 SHA256、SMPL-X 文件 SHA256；模型资源按官方授权取得，不提交到 Git。
