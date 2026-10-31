# 灵巧手、机械臂与手臂协同动作重定向：文献综述及本项目定位

检索截止：**2026-10-29（北京时间）**。项目核对基线：`5ee295db24fdfe4f992a5cef49d23888f73b0c05`。
修改前检查点：`checkpoint-20261029-1045-retargeting-literature`。

本文围绕“人类动作如何转换为异构机器人可执行动作”，按用户提供的流程图定位论文。附件仅作为流程参考。论文事实与作者报告的结果尽量引用论文原文、作者页面和官方代码；关于本项目的判断依据本地源码。文末的路线建议是基于这些证据作出的工程判断，尚未在本项目执行新的算法对比实验。

## 1. 核心结论与选文范围

**工程上成熟的主线仍然是：明确的坐标/尺度处理，机器人模型驱动的几何优化或学习映射，机械臂 IK，以及独立的命令与控制层。** AnyTeleop、dex-retargeting、Open-TeleVision、Bunny-VisionPro、ACE 是可参考的公开系统。学习方法并非天然比优化方法先进；目标函数、可观测信息、部署分布和任务定义往往更关键。[AnyTeleop](https://yzqin.github.io/anyteleop/)、[dex-retargeting](https://github.com/dexsuite/dex-retargeting)、[Open-TeleVision](https://arxiv.org/html/2407.01512v1)、[Bunny-VisionPro](https://arxiv.org/html/2407.03162v1)、[ACE](https://arxiv.org/html/2408.11805v1)。

**2025—2026 年值得重点关注的进展分成两条：** 一条继续改善实时动作映射，包括 GeoRT 的学习式几何对应、SBR 的采样稳定性以及少量指导下的跨手型映射；另一条将目标从姿态相似扩展为手物接触、物体运动与物理可执行性，例如 TopoRetarget、DexOAK、ReForce、SPIDER、OATH。这两条路线的输入条件和计算预算不同，不能混成一个排行榜。

**本项目已经覆盖第一条路线中的三类后端，但第二条路线的观测与闭环仍不完整。** 更合适的定位是“多输入、多机器人、模块化的运动学重定向与执行验证平台，包含学习、几何和 Vector 优化后端”。当前源码不足以支持“统一手臂接触优化”“通用物体操作”或“已达到跨任务 SOTA”的表述。

选文标准与证据口径：

- “成熟”看公开实现、可复现的配置与接口、实机/多平台证据和适用边界；不等同于论文发表早、引用多或作者声称最优。
- “近期”主要覆盖 2025—2026 年，最新纳入至 2026-10-19 首发的 DexOAK。本文是有目的的代表性综述，不保证穷尽截止日前所有论文。
- **正式发表、作者宣称接收、预印本**分别标注。作者仓库的会议标注不额外推断为已核实 DOI 和页码。
- **公开代码、仅项目页、Coming Soon、未核实代码**分别处理。“未核实”不表示断言作者没有代码；本次没有下载和运行所有外部仓库。
- MediaPipe、FrankMocap 等属于感知；URDF/FK/Jacobian 属于机器人建模；ACT、Diffusion Policy、VLA 属于策略学习。它们可以组成完整系统，但不能全部算作重定向求解器。

## 2. 按原流程图建立位置坐标

```mermaid
flowchart LR
    A[A 输入：摄像头 / Vision Pro / 录制文件] --> B[B 原始观测]
    B --> C[C 坐标与拓扑处理]
    C --> E[E 重定向：几何 / 优化 / 学习]
    B --> R[录制文件]
    R --> D[D 离线处理 / 训练 / 导出]
    D --> E
    E --> F[F 机器人关节结果]
    F --> G[G 命令滤波 / 限速 / 映射]
    G --> H[H PyBullet 或机器人执行]
    H -. 机器人状态 / 物体 / 触觉反馈 .-> E
```

实线对应原图；虚线是理解接触与物理方法时需要补上的反馈通路，**不表示本项目已具备完整反馈实现**。离线物理重定向还会在 D 内反复调用仿真，不只是完成一次 E→H。

| 位置 | 应研究的问题 | 本项目对应位置 | 文献如何归位 |
|---|---|---|---|
| A 输入 | 遮挡、漂移、帧率、设备、同步 | `inputs/camera.py`、`mediapipe.py`、`smplerx.py`、`body_tracking.py`、`visionpro.py`、`replay.py` | ACE、DexCap 的采集贡献主要在此 |
| B 原始观测 | 关键点/矩阵、左右手、单位、时间戳、缺失；是否有物体与力 | `contracts/observations.py`、`data/*recording.py` | 接触方法往往要求额外物体 pose、mesh、接触或力目标 |
| C 坐标与拓扑 | 世界/掌局部、相对/绝对、尺度、21/25 点、左右手、机器人对应 | `retargeting/hand/{coordinates,topology,processing}.py`、`applications/visionpro_processing.py`、`retargeting/arm/calibration.py` | 输入规范不一致会使 E 的比较失去意义 |
| D 离线 | 数据集、训练损失、示范清理、轨迹精化、模型导出 | `learning/{dataset,losses,trainer}.py`、`applications/training.py` | GeoRT/TransRetarget 训练在此；SPIDER 等的主工作流也在此 |
| E 求解 | 把人体目标变成机器人关节/参考动作 | `retargeting/hand/{predictor,geometric,dex_native,transhandr_vr}.py`；`retargeting/arm/{retargeter,ik}.py`；`simulation/arm_ik.py` | 真正对比几何、优化、学习及接触目标的核心位置 |
| F 结果 | 角度含义、关节顺序、主动/被动与 mimic 约束 | `contracts/{hand,arm}.py` | 同为“角度”也可能是 L21 中间表示或不同机器人的原生关节 |
| G 命令 | 映射、平滑、限速、失效保持、必要的反馈修正 | `retargeting/hand/postprocessing.py`、`simulation/hand_commands.py`、`retargeting/arm/safety.py`、`robots/*` | 滤波不能代替正确求解；ReForce 等跨越 E/G 并需要 H 的反馈 |
| H 执行 | 实际状态、碰撞、跟踪误差、力、任务成功 | `simulation/*` | PyBullet 能显示运动，不自动证明接触保持或真机可行 |

以上源码路径均相对 `src/teleoperation/`。文件读写和在线/离线组装还可参阅 [架构说明](architecture.md) 与 [数据契约](contracts.md)。

## 3. 当前项目在文献中的实际位置

| 已有路线 | 核对到的算法事实 | 文献类别与位置 | 不能据此推断的能力 |
|---|---|---|---|
| PoseTransformer / TransHandR VR | 多帧手关键点窗口→L21 18 维表示，其中包含占位维；随后得到 17 个活动 DOF。训练损失通过 FK 使用方向、指尖、拇指、距离及可配置碰撞项；VR 维持上游专用三帧数值路线 | 学习式运动学重定向，C/D/E/F；公开论文名与 TransRetarget 有明确关联 | 网络有时序建模不意味着接触动力学正确；训练碰撞损失不等于完整网格碰撞安全保证 |
| `GeometricHandRetargeter` | 使用人体骨段、原生 URDF 轴和标定求角；拇指链有局部数值反解；不依赖训练权重 | 几何为主的运动学映射，C/E/F | 不是 dex-retargeting Vector 或 DexPilot 的论文复现；“geometric”也不是绝对没有局部优化 |
| `DexNativeRetargeter` | 实际调用 `VectorOptimizer` 与 `SeqRetargeting`；按目标机器人原生 URDF 求解；支持 mimic；使用腕/掌根至指尖向量；相机和 VR 有显式独立 profile | 与 dex-retargeting 成熟优化路线直接对应，C/E/F | **当前是 Vector，不是 DexPilot。** 未包含完整手物接触、场景碰撞或力闭合目标；对外部求解器有边界适配与失败恢复 |
| TRON2A 机械臂 | 标定人体肩肘腕后生成末端目标；`RoboticsToolboxArmIK` 调用 `ikine_LM`，多初值并按相邻姿态选解 | 末端任务驱动的数值 IK，C/E/F | 单个末端任务不唯一确定冗余臂的肘部姿态 |
| 多机器人 PyBullet 臂 | `PyBulletArmIK` 使用本机模型限位、末端目标、rest pose 与可选肘参考；有机器人特定 profile | IK 与姿态偏好结合，C/E/F/G | `restPoses` 近似表达姿态偏好，不等价于已经实现严格 HQP 或带独立肘位置约束的求解器 |
| 双臂双手 | `DualTeleoperator.update()` 分别调用手部和手臂路线，再组装命令 | 同帧观测下的模块化组合，C/E/F/G | 不是把全部臂手关节、物体状态和接触力放入同一个优化问题 |

核查要点：`geometric.py` 顶部仍有“未接入任何 CLI”的历史注释，但 `applications/realtime_hand_sim.py::build_retargeter()` 已有 `geometric` 分支。因此这里按实际调用关系定位，不把历史注释当作当前功能清单。

### 3.1 TransHandR 与公开 TransRetarget 的关系

本地上游 `D:/2026/code/TransHandR/TransHandR/paper/ieeeconf_latex/root_EN.tex` 使用题名 **TransRetarget: A Human-Robot Hand Motion Retargeting Framework Based on Temporal-Spatial Transformer**，作者字段为 Anonymous。[公开作者代码](https://github.com/ZaynLeung/TransRetarget) 使用同一题名、标注 IROS 2026，README 同时保留“TransHandR 安装与环境说明”。这足以说明名称关联，**不足以认定公开分支、用户本地源码与任意 checkpoint 数值相同**。

本次重新计算的本地 SHA256：

| 来源路径 | SHA256 |
|---|---|
| `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `D:/2026/code/TransHandR/TransHandR/model/model_poseformer.py` | `9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3` |
| 本仓库 `src/teleoperation/retargeting/hand/dex_native.py` | `ee7bbaed4381b70ba02f36c9bf19c89e2e81b718ff5715f9575f83463dd71473` |
| 本仓库 `src/teleoperation/simulation/arm_ik.py` | `854e955f17e47ab29cd671e3a2cc25057d4cbb313e6995b0ca1517bd9c8c57f8` |

本次只做综述和源码定位，没有重跑上游数值等价实验。原有等价验证、来源快照及“输入侧/目标侧、资源管理、有效性处理、输出防护”等边界适配单独记录于 [TransHandR VR 文档](transhandr-vr.md)。不能将相机掌局部预处理直接套到上游 VR 路线上，也不能通过改 checkpoint 元数据宣称兼容。

## 4. 灵巧手：成熟路线与近期在线方法

### 4.1 如何理解几何、优化和学习

几何映射直接从骨段方向、夹角或关节轴关系计算机器人目标，计算小、容易解释，但形态差异与捏合误差可能明显。优化方法通过机器人 FK 比较指尖位置/向量/方向，主动协调可用 DOF；学习方法则把这种对应或优化过程离线学习，再在线前向推理。**“学习”描述求解方式，“接触感知”描述目标与信息，两者不是互斥类别。**

用于理解的统一示意式是：

\[
q_t^*=\arg\min_{q\in\mathcal Q}
\sum_k w_k\rho(\|v_k^r(q)-s v_k^h\|)
+\lambda\|q-q_{t-1}\|^2
+w_p L_{pinch}+w_o L_{orientation}+w_c L_{collision}.
\]

这里不是复写某一篇论文的完整公式。不同方法只启用其中一些项；当前项目 Vector 的时间正则参数为 `norm_delta=0`、内部低通关闭，不能因为统一式中出现平滑项就声称当前配置启用了它。

### 4.2 重点方法对照

| 文献/状态 | 核心贡献与位置 | 公开实现与成熟度 | 对本项目的价值与局限 |
|---|---|---|---|
| **DexPilot**；2019 首发，ICRA 2020 | 指间关键向量和捏合启发式驱动的优化；原系统还含多视角视觉与手臂遥操作。C/E/F，系统覆盖 A–H | 成熟方法基线；[NVIDIA 出版页](https://research.nvidia.com/publication/2020-05_dexpilot-vision-based-teleoperation-dexterous-robotic-hand-arm-system)、[原文](https://arxiv.org/abs/1910.03135)。后续库的实现应与原系统区分 | 最适合与现有 Vector 比较捏合效果；不是现有项目已经接入的算法 |
| **AnyTeleop / dex-retargeting**；RSS 2023 / 持续维护的库 | 将通用手观测与机器人 FK 优化连接，库提供 position、vector、DexPilot 等不同目标。C/E/F | [论文](https://arxiv.org/abs/2307.04577)、[作者项目](https://yzqin.github.io/anyteleop/)、[官方库](https://github.com/dexsuite/dex-retargeting)。工程复现优先级高 | 本项目已采用 Vector；下一步应比较目标，而不是重复接入同一个库。手部库不自动继承 AnyTeleop 整套臂部避障 |
| **TransRetarget**；作者页/仓库标 IROS 2026 | 多帧时空 Transformer 建模帧内关节和帧间运动，通过运动学损失训练；C/D/E/F | [作者出版页](https://zaynleung.github.io/publications)、[代码](https://github.com/ZaynLeung/TransRetarget)、[作者论文](https://zaynleung.github.io/TransRetargetv1/static/pdf/iros2026.pdf)。本次未核实 DOI/正式页码 | 解释本项目已有学习路线；应检查训练域、输入坐标、手型与目标损失，不能仅凭 Transformer 名称判断先进性 |
| **GeoRT**；IROS 2025 | 无监督学习人体与机器人可达空间对应，关注方向、覆盖、均匀响应、捏合与自碰撞；训练后直接输出 q。C/D/E/F | [论文](https://arxiv.org/abs/2503.07541)、[IEEE](https://ieeexplore.ieee.org/document/11247700/)、[官方代码](https://github.com/facebookresearch/GeoRT)。有训练与部署工具 | 很有价值的实时学习基线。新手型仍需模型、配置与训练；作者提醒视觉与手套分布差异。约 1 kHz 是算法推理口径，非端到端频率 |
| **Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation**；作者页标 IEEE Robotics and Automation Practice 2026，前身 ICRA 2025 Workshop | 分析整体手姿、腕相对指尖、捏合、指尖方向和时间正则的作用；C/E/F | [项目](https://mingrui-yu.github.io/retargeting/)、[官方代码](https://github.com/Mingrui-Yu/retargeting)。有离线、配置和实机适配 | 优先用来设计本项目的目标函数消融。指尖法向/方向可能比单纯位置更能解释杯把、工具等任务失败 |
| **Smooth Operator / SBR**；2026-07-08 首发，预印本 | 批量采样候选关节，按精英样本与代价加权更新控制分布，减少局部解跳变；C/E/F，稳定性影响 G | [论文](https://arxiv.org/abs/2607.07491)、[项目](https://mimicrobotics.github.io/smooth-operator/)、[官方代码](https://github.com/mimicrobotics/mimic_retargeter_lab)。有 18 人、3 任务研究；采样版本需 GPU | 是改善抖动的新求解器候选。仍是运动学目标，不能当成 SPIDER 那种接触动力学采样 |
| **AnyDexRT**；2026-07-09 首发，预印本 | 自监督对应加少量人类指导，接触分类器修正 pinch；C/D/E/F | [论文](https://arxiv.org/abs/2607.08341)、[项目](https://chenxi-wang.github.io/projects/anydexrt/)。作者页 Code Coming Soon | 适合跟踪跨手型适配；calibration-free 不等于不需要坐标规范、模型或训练，也不代表具备物体力控制 |

学习方法的较早脉络还包括 RSS 2022 的 [Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on Youtube](https://robotic-telekinesis.github.io/)。它说明从人类视频学习在线手部映射早已有研究基础，因此“用了神经网络”本身不能作为新颖性结论。

## 5. 机械臂：成熟数值基础与形态保持的新进展

手臂问题的关键是：**同一个腕位姿可能对应多个肩肘构型。** 只保证手腕误差小，不能保证肘跟随、上臂方向、避障和姿态自然。应将目标表示与 IK 求解器分开讨论。

### 5.1 成熟基础

| 方法 | 原理与适用场景 | 局限与流程位置 |
|---|---|---|
| **Damped Least Squares / LM 数值 IK** | 通过阻尼或正则抑制奇异附近的关节变化；已知机器人模型和腕目标即可构建。Wampler 1986，[原始论文 DOI](https://doi.org/10.1109/TSMC.1986.289285) | 成熟求解基础，E/F；不自动解决人机尺度、肘姿态、障碍或接触语义 |
| **任务优先级 / 零空间控制** | 高优先级保持末端任务，剩余自由度用于姿态、肘或限位偏好。Nakamura、Hanafusa、Yoshikawa 1987，[出版社页](https://journals.sagepub.com/doi/10.1177/027836498700600201) | E/F；零空间局部线性方法仍需考虑奇异与任务冲突 |
| **Hierarchical QP** | 将任务与不等式按优先级处理，适合腕、肘、躯干和约束联合。Escande、Mansard、Wieber 2014，[原文页](https://journals.sagepub.com/doi/10.1177/0278364914521306) | E/F；严格层级不是给某任务一个大权重；全身运动学可行不等于动态平衡 |
| **Pink / Mink / Pinocchio** | [Pink](https://github.com/pink-kinematics/pink) 在 Pinocchio 模型上做局部 differential IK；[Mink](https://github.com/kevinzakka/mink) 是 MuJoCo 体系中的相关实现，支持多任务、限位、碰撞对等 | 工程工具，不是三个可直接与论文结果排名的完整 retargeter；仍需定义人机对应、目标、权重与执行控制 |

微分 IK 的理解式为：

\[
\min_{\dot q}\sum_i w_i\|J_i(q)\dot q-v_i^{des}\|^2+\lambda\|\dot q\|^2,
\quad q_{min}\le q+\Delta t\dot q\le q_{max},\quad |\dot q|\le\dot q_{max}.
\]

碰撞距离和更严格的任务层级可继续加入，但**这个加权示意式本身并非严格 HQP**。用于本项目时，还需区分求解残差、最终命令误差和物理执行误差。

### 5.2 2025—2026 年代表工作

| 文献/状态 | 真正新增的内容与位置 | 本项目如何看待 |
|---|---|---|
| **GMR — Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking**；2025-10 首发，官方代码标 ICRA 2026 | 身体配对、静态朝向偏置、局部非均匀缩放与两阶段 IK；B/C/E/F，结果可供 D 中的跟踪策略训练。[原文](https://arxiv.org/html/2510.02252v1)、[代码](https://github.com/YanjieZe/GMR) | 适合学习跨机器人模型和全身目标处理；原文实验不涉及物体或复杂场景交互，不是灵巧手接触方案 |
| **WARP — Whole-Body Retargeting for Learning from Offline Human Demonstrations**；2026-06 首发，作者页标 CoRL 2026 接收 | 闭式 c-SEW 肩肘腕几何、掌位姿约束、肘平面、躯干偏移和底座分配；C/E/F，主要用于 D 中离线示范。[论文](https://arxiv.org/abs/2606.29940)、[项目](https://warp-retargeting.github.io/) | 与本项目“腕到了、肘没跟上”的问题直接相关。依赖目标形态和任务假设，不能假定任意 URDF 即插即用；本次未核实公开代码入口 |
| **Vision-Guided Arm–Hand Decoupled Motion Retargeting Using Graph Networks for Cross-Platform Robots**；IEEE TASE 2026-08-31 | 官方摘要描述臂手分流的图表示与跨平台映射；B/C/D/E/F。[IEEE 条目](https://ieeexplore.ieee.org/abstract/document/11673165)、[DOI](https://doi.org/10.1109/TASE.2026.3729358) | 是近期学习式候选。本次仅核实元数据与官方摘要，正文访问受限，故不对精确损失、速度、避碰或开源程度作肯定判断；不与 DexTele 自动视作同一论文 |

## 6. 手臂协同：同时运行、共同目标和物理闭环要分清

可用下面四种含义辨别“协同”：

1. **时间与坐标协同**：腕、手指分别求解，同步下发。
2. **任务关系协同**：额外保留左右手相对位姿、肘平面或共同操作物体的几何关系。
3. **关节与接触共同求解**：在共享问题中联合考虑手臂变量、碰撞/接触或物体目标。
4. **物理闭环协同**：利用机器人、物体、触觉/力反馈修正动作，或在仿真内生成动态可行轨迹。

一个系统可以跨越其中若干层；不能仅凭演示同时动了两只手和两条臂，就认定属于第 3 或第 4 种。

| 系统/论文 | 方法与流程位置 | 协同实质、证据与局限 |
|---|---|---|
| **AnyTeleop**，RSS 2023 | 手指运动学优化与腕目标臂部控制；A/B/C/E/F/G/H。[项目](https://yzqin.github.io/anyteleop/) | 成熟跨平台参考；手和臂主要分别映射，避障不等于共同手物接触优化 |
| **Open-TeleVision**，CoRL 2024 | 沉浸式主动视觉；臂用 Pinocchio CLIK/零空间，手用向量优化；A–H 中的在线路径。[原文](https://arxiv.org/html/2407.01512v1)、[代码](https://github.com/OpenTeleVision/TeleVision) | 公开实机系统，适合参考视觉反馈与采集；其 ACT 训练是 D 的下游分支，不能当作重定向本体 |
| **ACE**，CoRL 2024 | 外骨骼编码器得到腕位姿、手部相机得到关键点；分别映射至机器人；A/B/C/E/G/H。[原文](https://arxiv.org/html/2408.11805v1)、[代码](https://github.com/ACETeleop/ACETeleop) | 成熟输入/跨平台方案；硬件可靠性提升与映射算法贡献应分开。各机器人仍需过滤、约束和控制适配 |
| **Bunny-VisionPro**，2024 首发；本次按可核实预印本版本引用 | 臂部考虑碰撞与奇异性，手部处理主动/被动耦合；A/B/C/E/G/H。[原文](https://arxiv.org/html/2407.03162v1)、[代码](https://github.com/Dingry/BunnyVisionPro) | 原文明确解耦组件；公开真机与用户研究使其适合 Vision Pro 工程参考，但不是统一手臂接触优化 |
| **DexCap**，RSS 2024 | SLAM 腕定位、电磁手套、环境点云；重定向后进入 DexIL 策略学习；A/B/C→E/F→D→H。[原文](https://arxiv.org/abs/2403.07788)、[代码](https://github.com/j96w/DexCap) | 主贡献在可移植示范采集和学习；并非只替换在线 E 的算法。指长差异仍可能破坏工具握持 |
| **ObjRetarget**，2026-07-04，预印本 | 臂部人体轨迹初始化与拟人约束，接触阶段以手物多面体簇保几何，最后统一时间调度；B/C/E/F/G/H，主要处理示范。[原文](https://arxiv.org/html/2607.03828v1) | 与项目上游作者方向相关，值得重点阅读；原文明确 hand–arm decoupled formulation，不能改称统一全关节动力学优化。需物体识别/跟踪、接触阶段检测；代码本次未核实 |
| **DexTeleop-0**，2026-06-22，预印本 | 分开的腕 IK 与向量手映射给名义命令，再用触觉和 box-QP 求接触残差；E/F→反馈修正→G/H。[论文](https://arxiv.org/abs/2606.23431)、[项目](https://henryhcliu.github.io/dexteleop-0/) | 代表从几何同步走向力感知共享自主。真机证据存在，代码 Coming Soon；原文设置下并非所有任务都优于每个基线 |
| **DexTele**，2026-07-07，预印本 | 图网络分流处理臂/手并融合特征、隐空间优化；另用 VLM 初估抓力和 MPC 反馈调整；B/C/D/E/G/H。[原文](https://arxiv.org/html/2607.05883v1) | 不能简单称完全独立，也不能称完整统一接触动力学求解；本次未核实官方代码，工程成熟度证据弱于已开源系统 |

## 7. 更前沿的变化：从动作相似到交互可执行

### 7.1 物体与接触感知的运动学参考

| 文献/状态 | 相比纯手姿映射多了什么 | 位置、条件与证据边界 |
|---|---|---|
| **TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation**；2026-06-15，预印本 | 构建手物稀疏交互图，以距离加权 Laplacian、方向一致性、运动学约束及穿透处理保留交互结构。[论文](https://arxiv.org/abs/2606.16272)、[作者项目](https://toporetarget2026.github.io/TopoRetarget/) | B/C/E/F，输出还用于 D 中 RL 跟踪；需要手物几何/轨迹。手物接触拓扑与项目的 21→25 点拓扑转换是不同概念；不能用独立复现仓库冒充官方代码 |
| **DexOAK: Object-Aware Kinematic Retargeting**；2026-10-19，预印本 | 先交互图优化产生种子，再用人手点与物体点云学习映射，以接触/非穿透/自碰撞损失微调。[论文](https://arxiv.org/abs/2610.26762)、[原文](https://arxiv.org/html/2610.26762v1) | B/C/D/E/F，再接动态精化。是本次纳入的最新首发工作；网络推理免逐帧优化，不代表无需离线优化、训练或物体观测；本次未核实代码 |

两者的重要变化在于：误差基准从“人的手指在哪”转向“手指相对物体在哪里、是否保持需要的交互”。但**几何上接触成立仍不保证具有正确摩擦、支撑力和动态行为**。

### 7.2 在物理仿真或反馈中修正参考

| 文献/状态 | 主要贡献 | 应落在哪条流程，而非直接替换什么 |
|---|---|---|
| **SPIDER: Scalable Physics-Informed Dexterous Retargeting**；2025-11 首发，公开框架 | 人体示范给任务结构，大规模仿真采样与逐步撤去的虚拟接触引导修正轨迹；跨多机器人/数据源。[论文](https://arxiv.org/abs/2511.09484)、[官方代码](https://github.com/facebookresearch/spider)、[文档](https://facebookresearch.github.io/spider/) | D 内的物理重定向循环，产出 F/G 后再训练或执行。属于公开程度较高的离线前沿框架；有独立模拟器与 GPU 依赖，不是 PyBullet 实时后端的零成本替换 |
| **DexH2R: Task-oriented Dexterous Manipulation from Human to Robots**；2024 首发，2026-01 v2 | 运动学 primitive 上叠加任务驱动残差 RL，关注物体与接触状态。[原文](https://arxiv.org/html/2411.04428v2) | D/E/H；未来参考窗口等条件需检查因果性。本次未核实官方代码；与 ICCV 2025 同名 handover 数据集无关 |
| **Towards Human-level Dexterous Teleoperation / TeleDexter**；2026-07，预印本 | 手与物体子目标共同跟踪，RL 学习接触序列并迁移到硬件。[论文](https://arxiv.org/abs/2607.11481)、[作者项目](https://bigai-dex.github.io/blog/teledexter/) | D 训练，在线 E/H。作者说明控制器物体专用，实时手物 MoCap 要求较高；不能视作通用无训练替换件 |
| **ReForce: Learning Force-aware Retargeting for Dexterous Manipulation**；2026-08-16，预印本 | 使用仿真训练的通用力跟踪器，在运动学参考上预测残差，跟踪目标力。[原文](https://arxiv.org/html/2608.15560v1) | 主要跨 E/G/H 的反馈层；既支持在线也支持离线翻译。需要目标力和机器人指尖力反馈；当前使用法向力，剪切、滑移与接触分布仍是局限 |
| **Morphometric Imitation: From Morphology and Contact Aware Hand Retargeting to Sim-to-Real Visuomotor Policy**；2026-09-23，预印本 | 形态/接触匹配 MMO→残差 RL 动态精化→视觉策略蒸馏。[论文](https://arxiv.org/abs/2609.28660)、[作者项目](https://morphometricimitation.github.io/) | D/E/F/H 多阶段系统。真实策略成功率属于整条流程，不能全部归因于第一阶段 MMO；作者项目有代码入口，本次未运行验证 |
| **FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting**；2026-10-01，预印本 | 跨多段参考共享策略，以手物几何、未来目标和 off-policy RL 摊薄逐轨迹训练成本。[论文 v2](https://arxiv.org/abs/2610.01849)、[原文](https://arxiv.org/html/2610.01849v2) | D 中批量物理数据生成。多参考训练不等于对任意未知在线动作零样本控制；未来参考也是离线/在线边界的重要条件 |
| **OATH: Object-Agnostic Tracking for Dexterous Hands, Enabling Rapid Physics-based Retargeting**；2026-10-09，预印本 | 以物体表面局部接触误差信号训练共享残差策略，跨物体修正运动学参考。[论文](https://arxiv.org/abs/2610.14379)、[原文](https://arxiv.org/html/2610.14379v1) | D 中仿真 rollout 生成物理轨迹；需要物体模型/接触目标及训练。快于实时的 rollout 吞吐不自动表示端到端在线遥操作可用；本次未核实完整开源实现 |

### 7.3 从视频学习的早期脉络

[DexMV: Imitation Learning for Dexterous Manipulation from Human Videos（ECCV 2022）](https://yzqin.github.io/dexmv/) 将恢复的手物运动重定向为机器人示范，再做模仿学习；[DexVIP: Learning Dexterous Grasping with Human Hand Pose Priors from Video（CoRL 2021 / PMLR 2022）](https://proceedings.mlr.press/v164/mandikal22a.html) 将视频抓姿先验用于 RL。它们说明 D 与 E 长期紧密相连，但 DexVIP 不能列作逐帧实时 retargeter。

## 8. 如何读论文数字，避免得出错误排名

| 例子 | 原文数字的口径 | 对本项目能得出的结论 |
|---|---|---|
| GeoRT 的约 1 kHz | 作者报告的映射网络推理速度 | 可以考察 E 的低时延潜力，不能据此承诺 A→H 整条链 1 kHz |
| SBR 的用户实验 | 固定手架、mimic P0.50、18 人、3 任务；各方法硬件运行频率统一 30 Hz；成功率 54.1%，Hybrid 52.1%；SBR 完成时间较长 | 低抖动与任务成功有关，且存在速度/稳定性权衡；不能称所有指标、手型都显著胜出。[原文](https://arxiv.org/html/2607.07491v2) |
| DexOAK 的 46/56 真机回放 | 从仿真成功结果选取轨迹，按要求初始化物体后开环回放 | 是离线生成与执行证据，不能写成任意实时遥操作成功率。[原文](https://arxiv.org/html/2610.26762v1) |
| FlashDexRetarget 的 90% / 约 30 GPU 小时 | 指定 50 段示范、指定手型、规定参考和成功判据；对比 CHORD 是同论文预算设置 | 有批量生成效率潜力，不能外推为本项目硬件上的训练成本或未知任务成功率。[原文](https://arxiv.org/html/2610.01849v2) |
| OATH 的 58% 测试 / 54% 未见物体 | DexYCB 分割与其仿真判据；0.4 计算秒/参考秒是 rollout 口径，尚有前处理 | 有跨物体共享控制器证据；不能直接与 DexOAK 或 SBR 比成功率。[原文](https://arxiv.org/html/2610.14379v1) |

尤其要区分：姿态误差、接触匹配、物体轨迹误差、任务成功率、训练成本、算法耗时和端到端延迟。它们回答不同问题。现有项目文档的历史测试数量与性能也属于历史记录，本次没有重新运行算法基准，不把它们当作本次测得结果。

## 9. 对本项目最有价值的推进顺序

以下为建议，不代表此次已经实施代码改动。

### 9.1 近期：把成熟路线比较清楚

1. **固定基线与公平边界。** 在同一份录制数据、同一机器人/手侧、明确相机/VR profile 下比较已有模型、几何、Vector。输出统一为正确机器人语义，再统计原始求解和最终命令；不要让各方法额外滤波程度不同而无法解释差异。
2. **优先加入 DexPilot 对照及目标函数消融。** 重点检查腕至指尖位置、指间捏合、DIP→TIP 方向、mimic 与关节限制。AKO 论文比盲目更换网络更能指导这些诊断。改任何求解器前保存 checkpoint。
3. **手臂目标从“仅腕”扩展到明确的肘/姿态要求。** 参考任务优先级、QP 和 WARP 的 SEW 思路；先测清本项目现有 rest pose、肘参考和机器人可达性，再决定是否需要新求解器。
4. **保留三层日志。** 目标与求解结果、G 后命令、H 中实际关节/FK 各自记录。物理没跟上时，不能把问题全部归给重定向；关节夹限后也不能仍用夹限前误差代替最终误差。

### 9.2 中期：选择一个有代码的新在线基线

- **GeoRT**：检验离线训练能否换来更稳定、低时延的直接映射，尤其看相机/VR 与训练分布差异。
- **SBR**：检验采样式求解是否能降低当前优化解跳变；考虑 GPU/JAX/MJX 环境和机器人模型接入成本。
- **AnyDexRT**：在代码与依赖更清楚后，再评估少样本指导是否值得替代当前显式标定。

这里不建议一次更换输入、坐标处理、网络、手臂 IK 和滤波器，否则无法判断收益来源。

### 9.3 中长期：真正研究操作，而不只是跟随手势

先明确选题是“离线人类示范转机器人数据”还是“实时人机遥操作”。前者可优先参考 SPIDER、TopoRetarget、DexOAK、OATH；后者若目标是力敏感操作，可重点研究 ReForce 与 DexTeleop-0。它们都需要比当前手关键点更多的信息。

需要逐项补齐的能力包括：物体 pose/mesh 或点云、手物接触标记、机器人实际状态、必要的力/触觉、统一时间戳与任务判据。之后才谈接触保持、动态精化和任务成功。保持现有公共输入、窗口与结果接口兼容，通过显式扩展的新契约承载额外信息。

若复现工作依赖 MuJoCo Warp、Isaac 或其他并行仿真，先作为隔离的离线研究依赖评估；本项目当前已冻结的 PyBullet 执行层不在此次综述中更换。

## 10. 建议的评价协议与阅读顺序

### 10.1 评价协议

| 维度 | 指标与测试内容 | 避免的误判 |
|---|---|---|
| 输入与契约 | 单位、坐标、左右侧、时间戳、丢帧/缺手恢复、相机与 VR 分开 | 把感知误差、镜像和尺度错误当求解器问题 |
| 手部运动学 | 各机器人 FK 下指尖向量误差、捏合距离/时刻、指尖方向、关节越限、mimic 残差 | 用人体关节角一一对应异构机器人，或只看总体均值 |
| 手臂运动学 | 腕位置/方向、肘位置/平面、上臂/前臂方向、失败率、关节限位与奇异附近跳变 | 腕对齐便认定全臂跟随正确 |
| 时序 | P50/P95/P99 求解耗时、采集至执行延迟、角速度/加速度、抖动、恢复时间 | 低通降低抖动但引入延迟，或只报告最快单帧 |
| 安全与执行 | 自碰撞、臂/环境碰撞、G 后命令误差、H 实际跟踪误差、失效状态 | 训练 loss 或仿真开关被当作充分安全保证 |
| 操作 | 按物体/任务分层的成功率、滑移、接触 F1、穿透、物体轨迹误差、力误差（若有） | 用自由空间手势相似度证明抓取或工具操作有效 |
| 泛化与成本 | 分开操作者/物体/手型/设备的训练测试，标注标定、调参、训练与硬件成本 | 将“跨物体”与“跨机器人”“零样本”混用 |

建议先录制/选择张手、握拳、拇食/拇中捏合、多指汇合、侧摆、遮挡恢复；手臂包含屈肘、张臂、跨中线、工作空间边界和双手相对移动。接触评估另设抓杯、抓薄物、工具、双手共同物体等任务，不能与自由空间跟随混算一个成功率。

### 10.2 阅读顺序

1. **先确定项目基线**：AnyTeleop / dex-retargeting → TransRetarget → AKO 目标分析。
2. **再比较新在线求解**：GeoRT → Smooth Operator → AnyDexRT。
3. **补足机械臂理解**：DLS/任务优先级 → Pink/Mink → GMR/WARP。
4. **理解完整遥操作**：Open-TeleVision → Bunny-VisionPro → ACE/DexCap，分别看感知、标定、求解和执行贡献。
5. **研究接触与物理可行**：TopoRetarget → DexOAK → SPIDER/OATH；力敏感实时任务另读 ReForce/DexTeleop-0；批量数据生成再读 FlashDexRetarget/Morphometric Imitation。

## 11. 核心文献书目与核实说明

下列题名供检索、引用管理和组会阅读。正文各表已附论文/项目/代码链接。年份以正式发表或注明的首发为准。

1. Handa et al. **DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System**. ICRA, 2020; arXiv:1910.03135.
2. Qin et al. **AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System**. RSS, 2023; arXiv:2307.04577.
3. Liang et al. **TransRetarget: A Human-Robot Hand Motion Retargeting Framework Based on Temporal-Spatial Transformer**. 作者页面及官方代码标 IROS 2026；未补写未经核实的 DOI/页码。
4. Yin et al. **Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm**. IROS, 2025; arXiv:2503.07541.
5. Xin et al. **Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation**. 作者页标 IEEE Robotics and Automation Practice, 2026；前身 ICRA 2025 Workshop。
6. Malate et al. **Smooth Operator: A Real-Time Sampling-Based Algorithm for Kinematic Hand Retargeting**. arXiv:2607.07491, 2026.
7. **AnyDexRT: Calibration-Free Dexterous Hand Retargeting with Few-Shot Human Guidance**. arXiv:2607.08341, 2026.
8. Wampler. **Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods**. IEEE TSMC, 16(1):93–101, 1986.
9. Nakamura, Hanafusa, Yoshikawa. **Task-Priority Based Redundancy Control of Robot Manipulators**. IJRR, 6(2):3–15, 1987.
10. Escande, Mansard, Wieber. **Hierarchical quadratic programming: Fast online humanoid-robot motion generation**. IJRR, 33(7):1006–1028, 2014.
11. Cheng et al. **Open-TeleVision: Teleoperation with Immersive Active Visual Feedback**. CoRL, 2024; arXiv:2407.01512.
12. Yang et al. **ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation**. CoRL, 2024; arXiv:2408.11805.
13. Ding et al. **Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning**. arXiv:2407.03162, 2024；本文未独立核实其后正式会议归属。
14. Wang et al. **DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation**. RSS, 2024; arXiv:2403.07788.
15. Araujo et al. **Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking**. arXiv:2510.02252, 2025；官方代码标 ICRA 2026。
16. Chen et al. **WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations**. arXiv:2606.29940, 2026；作者页标 CoRL 2026 接收。
17. **Vision-Guided Arm–Hand Decoupled Motion Retargeting Using Graph Networks for Cross-Platform Robots**. IEEE TASE, 23:16191–16205, 2026; DOI:10.1109/TASE.2026.3729358.
18. Lai et al. **ObjRetarget: An Object-Aware Motion Retargeting Framework with Anthropomorphic Arm Constraints and Polyhedral Hand Modeling**. arXiv:2607.03828, 2026.
19. Liu et al. **DexTeleop-0: Force-Aware Bimanual Dexterous Teleoperation with Ego-Centric Perception towards Shared Autonomy**. arXiv:2606.23431, 2026.
20. Lai et al. **DexTele: A Dual-Arm Dexterous Teleoperation System Based on Motion Retargeting and Adaptive Force Control**. arXiv:2607.05883, 2026.
21. Wu et al. **TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation**. arXiv:2606.16272, 2026.
22. Spinola, Katzschmann, Schmid. **DexOAK: Object-Aware Kinematic Retargeting**. arXiv:2610.26762, 2026.
23. Pan et al. **SPIDER: Scalable Physics-Informed Dexterous Retargeting**. arXiv:2511.09484, 2025.
24. Zhao et al. **DexH2R: Task-oriented Dexterous Manipulation from Human to Robots**. arXiv:2411.04428, 2024，2026-01 v2。
25. Li et al. **Towards Human-level Dexterous Teleoperation**. arXiv:2607.11481, 2026.
26. Wu et al. **ReForce: Learning Force-aware Retargeting for Dexterous Manipulation**. arXiv:2608.15560, 2026.
27. Sadjadpour et al. **Morphometric Imitation: From Morphology and Contact Aware Hand Retargeting to Sim-to-Real Visuomotor Policy**. arXiv:2609.28660, 2026.
28. Lee et al. **FlashDexRetarget: Accelerating Dexterous Manipulation Data Generation through Multi-Motion Retargeting**. arXiv:2610.01849, 2026.
29. Sigurdson et al. **OATH: Object-Agnostic Tracking for Dexterous Hands, Enabling Rapid Physics-based Retargeting**. arXiv:2610.14379, 2026.
30. Qin et al. **DexMV: Imitation Learning for Dexterous Manipulation from Human Videos**. ECCV, 2022.
31. Mandikal, Grauman. **DexVIP: Learning Dexterous Grasping with Human Hand Pose Priors from Video**. CoRL 2021, PMLR 164, 2022.

同名排除：**DexH2R: A Benchmark for Dynamic Dexterous Grasping in Human-to-Robot Handover（ICCV 2025）** 是另一项 handover 工作，不能用其数据集/代码证明第 24 项动作重定向方法可复现。

## 12. 此次交付与验证范围

- 新增本文，并在 README 文档索引登记；未修改算法、模型、数据格式或依赖。
- 变更前 `git status --short --branch` 显示工作区干净；`git diff --check` 通过；已创建上列 annotated checkpoint。
- 变更前运行 `powershell -NoProfile -ExecutionPolicy Bypass -File devtools/preflight.ps1`，退出码 1：两类既有失败为 C 盘已有缓存、相对历史基线 `10ebd9f` 的既有文件改动；没有清理缓存或回退历史修改。这些失败不构成此次文献事实或算法测试结果。
- 文档交付检查退出码 0：31 条书目编号连续、本地文档链接有效、README 索引存在、代码围栏配对、4 项来源 SHA256 一致；`git diff --check` 通过。
- 变更后重跑同一 preflight，退出码仍为 1，仍为上述两类失败，历史基线差异仍为 60 个文件；本次不清理缓存、不回退或改动既有算法。
- 本次未重跑数值等价、实时设备、物理基准或完整 unittest。纯文档变更不影响共享运行行为，不能将本文的检查表述为算法测试通过。
