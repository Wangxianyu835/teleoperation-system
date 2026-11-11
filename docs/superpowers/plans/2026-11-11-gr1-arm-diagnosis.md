# GR1-T2 手臂位置与朝向冲突诊断 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在相同机器人模型、末端和目标下，确定当前 GR1-T2 手臂大量保持旧命令的原因，给出下一次业务修改所需的证据。

**Architecture:** 先建立运动学一致的离线诊断入口，依次隔离模型、位置任务、朝向任务、命令接受策略与实际执行。官方原样结果单独保存；控制变量实验明确标为诊断配置，不冒称官方原样或论文实验。

**Tech Stack:** Python、unittest、NumPy/SciPy、Pinocchio、PINK、PyBullet、h5py、Matplotlib。沿用首轮产物目录中的隔离依赖，不升级现有环境。

**Spec:** [首轮源码和数据对照报告](../../research/2026-11-11-teleopbench-gr1-arm-baseline.md)。本计划落实用户要求：聚焦实际手臂重定向准确性，复现单目视觉与 Vision Pro；不改公共接口，不改测试控制器。

## Global Constraints

- 当前已知结果是诊断线索，不等于已证明根因；不能直接删除拒绝逻辑或提高全部阈值。
- 第一台目标为 GR1-T2；首轮诊断为 VR。H1-2、G1 与单目视觉在它之后分阶段推进。
- 官方源码只读：`D:/2026/code/TeleOpBench-reference`，提交 `4cc410d149ce63c86f485b257cba3b3e0ec031d6`。
- 首轮当前代码基线：`f116e1c22f37c931b8ba8ae597be9a95b738255c`；业务被测版本与前一提交相同。
- 原始 H5 只读；不改坐标元数据，不伪造 SDK 版本，不猜测 ARKit→WebXR 腕局部轴。
- 公共 input、window、result、CLI 和命令契约兼容；相机和 VR 算法配置分别注明。
- 每个代码变更先检查 Git、创建 annotated checkpoint；验证后独立提交，不推送。
- 共享行为发生修改时执行完整 `unittest discover -s tests`；本轮写计划不执行方案。

## 1. 实验对象与固定条件

原样回放继续保留两份输入：

```text
datasets/raw/visionpro_hand_arm_20261012_215137_live.h5  # 103.458 秒，主要诊断
datasets/raw/visionpro_hand_arm_20261012_213812_live.h5  # 164.984 秒，独立验证
```

两段都来自既有采集，并不代表独立操作者或独立坐标版本。第二段不能用来边调参数边选最优结果：先锁定配置再验证它。

控制变量实验统一使用官方 `GR1T2_inspire_hand.urdf`、原始关节轴、相同活动关节、同一基座参考、`L_hand_base_link` / `R_hand_base_link` 控制点。头、腰、腿和手指在两边固定为同一状态，避免一个求解器能动头、另一个不能。

初始姿态、参考姿态、有效限位逐关节保存到 manifest。诊断用共同有效限位取原始限位中央 90%，测试关节状态在该区间进一步留出余量。此项是共同实验配置，不改当前运行 profile 或官方源码。

## 2. 文件职责

以下为执行本计划时拟新增的文件，当前只新增本计划：

| 路径 | 职责 |
|---|---|
| `devtools/gr1_arm_diagnosis.py` | 固定模型、生成 FK 可达目标、运行求解矩阵、采样拒绝前后结果、输出数据 |
| `tests/test_gr1_arm_diagnosis.py` | 验证模型/参考点一致、目标生成、分组隔离和指标计算 |
| `outputs/acceptance/gr1-arm-diagnosis-<session>/` | 每次独立保存配置、逐帧 CSV/NPZ、日志、图和预览 |
| `docs/research/2026-11-11-gr1-arm-diagnosis-results.md` | 记录实验结果、限制、根因判断和业务修改建议 |

第一阶段至第四阶段不修改 `src/teleoperation/`。若现有记录不足，可在独立诊断实例中包装 `_ik_call` / `_ik_call2` 收集返回值，原调用仍执行一次；保存并恢复临时状态，不通过 monkeypatch 跳过生产保护或修改官方常量。

## Task 1: 校验同模型、同控制点和度量工具

**Files:** 新增 `devtools/gr1_arm_diagnosis.py`、`tests/test_gr1_arm_diagnosis.py`。

**Interfaces:** 诊断脚本接受 `--stage fk|matrix|replay --upstream PATH --input PATH --output PATH`。`--output` 已存在则报错。FK 阶段不需要 input；matrix 使用生成的目标；replay 必须提供 H5。

- [ ] 核验官方提交、源文件和模型哈希；输出当前源码提交、模型、末端、14 个活动关节、有效限位和包版本。
- [ ] 用官方 URDF 建立 Pinocchio 与 PyBullet 运动学；PyBullet 派生副本只移除 visual/collision，不改 origin、axis、limit、inertial 或父子关系。
- [ ] 生成一个中间关节姿态：其余关节零值裁到有效区间，双肘请求 -0.8 rad 后再裁到有效区间内部。将最终值完整记录。
- [ ] 单个关节依次增加/减小 0.1 rad，裁到有效区间内部。用这一组已知关节状态的 FK 生成目标，同时保存原始 q、位置和旋转；不将未知可达性的笛卡尔点当成正例。
- [ ] 验证两套 FK 在同状态、同链接原点的误差：位置不超过 `1e-5 m`、旋转不超过 `1e-4 rad`。先校验旋转正交性再计算角度。
- [ ] 原样 VR 源码产生的非正交矩阵只进入原样基线，不进入有效旋转诊断正例；不得用静默 SVD 修复掩盖输入问题。

测试的数值断言使用以下定义，避免只检查返回类型：

```python
np.testing.assert_allclose(T_pin[:3, 3], T_bullet[:3, 3], atol=1e-5)
angle = Rotation.from_matrix(T_pin[:3, :3] @ T_bullet[:3, :3].T).magnitude()
self.assertLessEqual(angle, 1e-4)
np.testing.assert_allclose(T_target[:3, :3].T @ T_target[:3, :3], np.eye(3), atol=1e-6)
self.assertGreater(np.linalg.det(T_target[:3, :3]), 0.999999)
```

执行：

```powershell
$env:PYTHONPATH = "$PWD/src"
$python = '.venv/dex-native/Scripts/python.exe'
& $python -m unittest tests.test_gr1_arm_diagnosis -v
& $python devtools/gr1_arm_diagnosis.py --stage fk --upstream D:/2026/code/TeleOpBench-reference --output outputs/acceptance/gr1-arm-diagnosis-fk
```

**完成条件：** FK 一致性通过；否则先修正诊断器中的原点、惯性坐标或链接选择，不能进入求解器比较。验证后提交独立诊断工具，提交信息 `test: add controlled GR1 arm kinematics fixtures`。

## Task 2: 用四组实验分开位置、朝向与求解器影响

**Files:** 扩展同一诊断脚本及其测试，不改生产算法。

**Interfaces:** 读取 Task 1 的模型、状态与 FK 目标定义，输出 `matrix.csv`、逐次候选关节角、FK 误差、耗时和配置。每个 case、group、sample 有唯一编号。

| 组 | 求解方法 | 相同目标中的约束 |
|---|---|---|
| A | 当前 PyBullet 算法 | 只用位置 |
| B | 当前 PyBullet 算法 | 同一位置 + 合法目标朝向 |
| C | PINK | 只用位置 |
| D | PINK | 同一位置 + 合法目标朝向 |

- [ ] 每组独立实例，从同一 q 开始；相同关节约束、参考姿态。初始状态不得从上一组继承。
- [ ] 对位置模式，当前 adapter 明确 `ik_mode='position'` 且 `orientation=None`；PINK 腕朝向代价置零。只传 None 不足以保证所有 profile 关闭朝向约束。
- [ ] 对位姿模式，使用同一个 FK 生成的合法位置和旋转。PINK 按记录的任务代价求解；当前算法使用其原有候选与接受过程。
- [ ] 分开记录单次调用和静态目标保持 2 秒的结果。PINK 是速度积分算法，不能只比较一次调用与多次内部迭代算法。
- [ ] 共同诊断时钟为 30 Hz，每次积分使用 `dt=1/30`；官方原样 `dt=0.005` 另外作为时钟对照，标签分别保存，不能混在同一结果行。
- [ ] 两组求解器先保持纯运动学评估，排除电机误差；连续序列用上一条接受命令推进状态，静态 case 之间重新初始化。
- [ ] 两侧分别报告位置/朝向误差、限位、单步变化、成功/保持原因。姿态任务结果不以“没抛异常”作为通过条件。

位置及旋转误差统一计算：

```python
position_error_mm = 1000 * np.linalg.norm(p_fk - p_target)
orientation_error_deg = np.rad2deg(
    Rotation.from_matrix(R_fk @ R_target.T).magnitude()
)
```

执行：

```powershell
& $python -m unittest tests.test_gr1_arm_diagnosis -v
& $python devtools/gr1_arm_diagnosis.py --stage matrix --upstream D:/2026/code/TeleOpBench-reference --output outputs/acceptance/gr1-arm-diagnosis-matrix
```

**完成条件：** 每组都有完整相同 case 覆盖，没有跨组状态串扰；能区分“只用位置也差”与“加入朝向才变差”。初步诊断参考值为静态保持末段位置误差不超过 10 mm、朝向不超过 5°，执行前固定到 manifest；它们是工程筛查值，不是论文结果或精细操作验收保证。未达到则保留失败，不临时放宽阈值。

验证后提交 `test: isolate GR1 position and pose IK behavior`。

## Task 3: 直接追踪“算出的动作为什么没被接受”

**Files:** 扩展诊断采样及结果报告；生产源代码仍只读。

**Interfaces:** 每个更新记录位置候选、位姿候选、最终命令三份 q，以及它们各自的 FK、限位状态、最大关节变化和拒绝原因。记录四个阶段的误差，不能复用临时状态恢复后的错误测量。

- [ ] 针对 B 组，采样 `_ik_sequence` 中的位置候选与加入朝向候选。
- [ ] 记录 `position_candidate_residual_mm`、`pose_candidate_residual_mm`、`position_fallback`、`wrist_rejection`、`wrist_joint_step_scaled` 与候选关节是否有限/越限。
- [ ] 对采样前后完整关节位置和速度作比较，确保诊断没有改变下一帧初始状态。
- [ ] 为几何可达但被保持的 case 保存可独立复现的模型哈希、q_start、p_target、R_target、限位和原始候选。
- [ ] 统计候选比旧姿态改善的位置误差、损失的朝向误差以及实际被拒绝比例。不能把“拒绝少”单独作为更好。
- [ ] 加入不可达目标、非法旋转、非有限解、明显越限解作为负例，验证诊断能正确分类；此阶段不使生产代码接受它们。

**完成条件：** 至少为每类实际出现的保持原因提供可复现 case；若没有合成可达 case 被误拒绝，就不能直接得出拒绝逻辑有 bug。检查真实输入目标可达性、朝向参考和初始化差异。

验证后提交 `test: capture GR1 candidate and accepted arm commands`。

## Task 4: 接回真实录制，验证诊断能解释现场现象

**Files:** 诊断脚本、报告与本轮独立产物目录。

- [ ] 先用 103 秒录制复现首轮原配置；报告是否重现左右约 2,867/2,865 次冲突，若有版本/初始化差异先解释差异。
- [ ] 在当前同一模型、同一目标位置上，只关闭朝向约束，比较目标到接受命令的误差、实际运动幅度和保持原因。它是原因隔离实验，不直接作为上线方案。
- [ ] 对实际目标逐帧检查是否在有效限位下有可行解；未求到解标为“本求解未找到”，不能当作数学不可达证明。
- [ ] 头部参考和初始腕参考分别产生目标，单独画出目标差；本步骤不同时更换模型和求解器。
- [ ] 不对 SDK/WebXR 未验证腕局部轴作完整位姿公平排名。需要原生 WebXR 同步录制或有依据的固定局部轴适配后，另做位姿复现。
- [ ] 配置锁定后，用 165 秒录制独立运行；不再依据第二段结果调参。
- [ ] 输出三份便于人看的结果：相同坐标视角的动作视频、误差/拒绝统计表、典型失败帧及解释。单独标出预热、缺失、恢复、限位和静态保持。

```powershell
& $python devtools/gr1_arm_diagnosis.py --stage replay --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_215137_live.h5 --output outputs/acceptance/gr1-arm-diagnosis-replay103
& $python devtools/gr1_arm_diagnosis.py --stage replay --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 --output outputs/acceptance/gr1-arm-diagnosis-replay165
```

**完成条件：** 结果能够指出主要问题发生在目标定义、模型、位姿求解、接受策略或执行中的哪些层；证据不足的部分明确留下，而不是挑一条较好曲线宣布修好。

验证后提交 `docs: report controlled GR1 arm diagnosis`。全部 Task 1–4 完成后本轮诊断结束。

## 3. 诊断之后的业务修改决策

诊断结果出来后另写最小修复计划，不在当前计划中预先指定源码替换：

| 证据 | 对应的下一次修改方向 |
|---|---|
| 相同 q 的两套 FK 不一致 | 先修正模型、参考系或末端定义；不改 IK |
| 已知可达目标的位置模式也失败 | 检查 IK 关节锁定、限位、初始状态和收敛配置 |
| 位置模式正常，合法位姿模式失败 | 检查朝向转换、任务权重、位姿求解和参考姿态 |
| 候选可用但最终大量保持 | 为候选选择建立明确的位置优先规则，保留越限/无效解保护 |
| 下发命令正确，实际末端跟不上 | 检查电机参数、物理步数、执行频率；不改目标映射 |

任何位置优先方案都需要同时记录牺牲了多少朝向，不能以朝向完全失效换取“位置误差好看”。只有证据指向对应模块，才考虑修改 `arm_ik.py`、`arm_profiles.py`、`wrist_control.py` 或 `wrist_orientation.py`；每次只处理一个已证实问题，测试先证明问题，再修复。

共享修改后至少运行：

```powershell
$env:PYTHONPATH = "$PWD/src"
& $python -m unittest tests.test_arm_following tests.test_arm_ik_multi_robot tests.test_wrist_observations tests.test_robot_control_session -v
& $python -m unittest discover -s tests
```

## 4. 后续真实操作与单目视觉

离线诊断和对应修改验证后，现场采集四类动作：静止保持、各方向小幅移动、尽量定点转腕、接近并操作物体。记录操作者输入、目标、接受命令、实际末端及对象状态，不仅保存动画。

操作顺序为接近指定点、推方块、抓起并放下、双手配合；每项固定起始位置、操作说明、成功条件与重复次数。先记录现配置作为基线，再用同条件比较改动。任务成功率和时间单独报告，离线厘米误差不能代替抓取成功。

视觉链路先新增可用输入：RGB 或 SMPL root/body pose，并记录单位、坐标、时间戳和形状参数。既有手部关键点 H5 不能替代身体输入。先固定姿态参数比较模型重建、缩放和末端目标，再加入相机推理。遵循官方参数和本文首轮报告，不把现有两段方向映射称为论文方法。

上述现场、视觉和其他机器人扩展是后续独立阶段，不在本轮诊断工具里实现物体识别、自动抓取规划、全仓库工作流重构或测试控制器接入。

## 5. 本次计划文档检查点

- 文档修改前提交：`f116e1c22f37c931b8ba8ae597be9a95b738255c`。
- 文档修改前标签：`checkpoint-20261111-2009-gr1-diagnosis-plan`。
- 当前仅写计划；未创建诊断工具，未执行新实验，未修改业务代码。
- 执行时在本 chat 逐项进行，沿用用户已指定的目录和范围；无需重新询问使用何种执行方式。
