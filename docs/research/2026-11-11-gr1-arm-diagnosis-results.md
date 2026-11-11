# GR1-T2 手臂准确性诊断结果

日期：2026-11-11。对应[执行计划](../superpowers/plans/2026-11-11-gr1-arm-diagnosis.md) Tasks 1–4。只新增诊断工具、测试和文档；`src/teleoperation/`、公共接口、测试控制器、官方源码与原始 H5 均未修改。没有实际物体操作或实体机器人试验。

## 1. 结论

本轮确定了两个不同的问题：

1. 当前位姿求解在部分可达目标上产生较差的位置候选；接受规则随后保持旧状态，使机器人无法逐步接近更好的位置。不能仅凭此认定目标不可达，也不能把关闭朝向当成修复。
2. 拒绝新解时，代码复制的是物理仿真后的实际关节状态。如果该状态已轻微越限，就会把越限数值作为新命令返回，随后被严格命令校验拦停。第二段录制在 45.435 秒处明确复现了这一问题。

关闭朝向显著改善了部分位置误差，但朝向明显变差。PINK 在小幅已知可达目标上表现更精确，在一个大幅目标上仍有明显位置误差；本轮不支持“直接换 PINK 即完成操作目标”的结论。

## 2. 固定条件与产物

官方目录 `D:/2026/code/TeleOpBench-reference` 保持提交 `4cc410d149ce63c86f485b257cba3b3e0ec031d6`，只读。沿用[首轮基线报告](2026-11-11-teleopbench-gr1-arm-baseline.md)的模型和输入来源。

| 产物 | 路径 |
|---|---|
| 统一模型 FK 核对 | `outputs/acceptance/gr1-arm-diagnosis-fk/` |
| 四组实验与 5 ms 时钟对照 | `outputs/acceptance/gr1-arm-diagnosis-matrix/` |
| 96 个宽范围目标及缩小后的反例 | `outputs/acceptance/gr1-arm-diagnosis-probe-complete/` |
| 103 秒原链路/关闭朝向对照 | `outputs/acceptance/gr1-arm-diagnosis-replay103/` |
| 165 秒验证，含停止帧 | `outputs/acceptance/gr1-arm-diagnosis-replay165-complete/` |
| 汇总、图与动作预览 | [gr1-arm-diagnosis-final](../../outputs/acceptance/gr1-arm-diagnosis-final/) |

最初的 `gr1-arm-diagnosis-replay165/` 是未保留错误前数组的失败尝试；`-complete` 版本在诊断入口捕获并保存原始异常、候选、实际状态和前缀数据，未修复或绕过生产校验。原样重跑在同一帧停止。以 `-complete` 为报告依据。

工具：[gr1_arm_diagnosis.py](../../devtools/gr1_arm_diagnosis.py)；图表：[gr1_arm_diagnosis_plots.py](../../devtools/gr1_arm_diagnosis_plots.py)；测试：[test_gr1_arm_diagnosis.py](../../tests/test_gr1_arm_diagnosis.py)。

## 3. 同模型、同参考点核对

控制实验采用官方 `GR1T2_inspire_hand.urdf`，Pinocchio 和 PyBullet 均只保留十四个手臂自由度，其他关节在零位固定。相同末端为 `L_hand_base_link`、`R_hand_base_link`，有效限位均为原始范围中央 90%。派生 URDF 去掉视觉/碰撞几何，保留运动学和惯性参数；不进行接触仿真。

生成一个参考状态与十四个关节各 ±0.1 rad 的状态，共 29 个关节状态、左右各测一次，共 58 次 FK 核对。所有目标都由已知合法关节状态生成，因而存在精确运动学解。

- 最大位置差：`1.7829614e-5 mm`。
- 最大朝向差：`6.0535083e-6 deg`。
- 通过预先固定的 FK 一致性门槛：`1e-5 m`、`1e-4 rad`。
- 读取候选前后完整关节位置与速度一致性通过测试。

统一模型实验使用世界链接原点，不依赖当前应用将 PyBullet 基座惯性坐标称为 base 的约定。原生录制对照则完整保留当前应用的参考系，不能把这两个实验的坐标数组直接拼在一起。

## 4. 位置与朝向的四组实验

每组独立初始化；每个目标调用 60 次。30 Hz 组的积分时间为 2 秒。5 ms 组也是 60 次更新，积分仅 0.30 秒，作为时钟差异对照，不能称为相同积分时长。

原始 58 个侧/状态组合中，30 个没有改变被测侧的目标（参考状态与另一侧关节变化）。下表排除这些组合，仅统计 28 个真正改变被测侧目标的例子，避免 PINK 大量零值使中位数误导。

| 方法 | 任务 | 位置误差中位数 / mm | 位置 P95 / mm | 朝向误差中位数 / deg |
|---|---|---:|---:|---:|
| 当前 PyBullet | 位置 | 2.708 | 2.844 | 7.705，未约束 |
| 当前 PyBullet | 位置与朝向 | 2.812 | 3.013 | 0.00179 |
| PINK，30 Hz | 位置 | 0.000019 | 0.000100 | 5.186，未约束 |
| PINK，30 Hz | 位置与朝向 | 0.000026 | 0.000104 | 0.00000026 |

全部 58 个组合的位置模式均通过预先固定的 10 mm 筛查；位姿模式全部同时通过 10 mm / 5° 筛查。没有触发拒绝。精度结果仅适用于这一组小幅目标、相同初始姿势和模型，不代表整个工作空间或物体操作效果。

当前方法即使收到本来已经达到的完整位姿，位置/姿态两次求解后也会出现约 2.63 mm 位置偏移，同时朝向误差变小。这保留在实验结果中，未为追求零误差修改算法。

## 5. 可达但被保持的具体例子

对有效关节区间内部使用固定随机种子 `20261111` 生成 48 个更大幅姿态，左右各一次，共 96 个有已知 FK 解的目标。87 个未标记拒绝，9 个触发 `pose_position_conflict`；“未拒绝”不等于达到误差要求。

对拒绝样本逐步减小关节变化，保存最近的仍会触发拒绝的样本：[witness.json](../../outputs/acceptance/gr1-arm-diagnosis-probe-complete/witness.json)。这是一种离散缩小，不声称获得全局最小反例。

该样本中：

- 原姿态距离目标约 164.331 mm、朝向误差约 62.420°。
- 位置候选误差约 3.308 mm。
- 加入朝向后的候选位置误差约 60.370 mm，触发拒绝，返回原姿态。
- 原配置连续更新 60 次仍保持同一姿态，位置误差没有改善。

同一目标的诊断对照：

| 60 次更新后的结果 | 位置误差 / mm | 朝向误差 / deg |
|---|---:|---:|
| 当前原规则 | 164.331 | 62.420 |
| PINK 位姿任务 | 63.165 | 0.854 |
| 诊断性允许向合法位置候选接近 | 3.294 | 79.395 |

第三行只在离线诊断状态中使用了三次位置候选，并限制每步最大关节变化 0.35 rad；未改生产接受规则。它说明原规则确实能阻断有用的位置进展，也说明直接接收位置候选并没有解决完整位姿，不能作为最终修复。

## 6. 103 秒真实录制：只改变朝向输入

使用当前原生机器人模型、当前输入和同一控制会话。诊断分支只将 `driver.solve_each` 的 orientation 参数改为 None；GR1 原 profile 已是 position，因此实际关闭朝向。WristTracker、平移目标、滤波、缺失处理、物理步数和命令校验保持原样。

两种模式均完成 2,886 帧。逐帧位置目标差为严格零。位置候选、位姿候选与最终命令分别采样，采样恢复位置和速度；原模式复现首轮左右 2,867 / 2,865 次冲突。

排除前 2 秒：

| 指标（中位数） | 原模式左 | 关闭朝向左 | 原模式右 | 关闭朝向右 |
|---|---:|---:|---:|---:|
| 接受命令 FK 到目标的位置误差 / mm | 187.526 | 25.834 | 84.053 | 51.933 |
| 相对原朝向目标的误差 / deg | 38.803 | 135.854 | 33.003 | 77.350 |

原模式在左 2,847 帧、右 2,039 帧拒绝时，存在关节限位内、位置误差比保持姿态更小的位置候选。这不是完整位姿可达性的证明，但确认保持规则对位置跟随有显著影响。

[误差曲线](../../outputs/acceptance/gr1-arm-diagnosis-final/gr1-arm-diagnosis-replay103.png)；[实际仿真姿态预览](../../outputs/acceptance/gr1-arm-diagnosis-final/native-preview.gif)。动画使用物理步进后的肩、肘、末端位置；没有物体和接触任务。

## 7. 165 秒验证：保持状态引起越限停止

配置未根据第二段调参。原模式完成 1,274 帧，在第 1,274 帧（从零编号）、45.435 秒处抛出 `Part exceeds joint limits`。关闭朝向模式完成全部 4,631 帧。

停止帧左腕 roll 的证据：

| 项目 | rad |
|---|---:|
| 规格上限 | 0.610000 |
| 前一帧已接受命令 | 0.5850182571 |
| 物理步进后的实际关节值 | 0.6240279180 |
| 拒绝后返回的关节值 | 0.6240279180 |
| 这一帧位置候选的关节值 | 0.5173389939 |
| 这一帧位姿候选的关节值 | 0.3162669825 |

候选均在限位内。超限来自复制实际状态：实际值超过上限 0.0140279 rad，约 0.8037°。`solve()` 在姿态冲突时复制保存的实际状态；`_arm_values()` 只对 0.01 rad 以内的越限进行截断，因此该值原样进入严格的命令校验并被拒绝。原始运行和加入故障保存的重跑均在同一帧停止。

源码证据：[arm_ik.py 的拒绝分支](../../src/teleoperation/simulation/arm_ik.py:978)、[小量限位截断](../../src/teleoperation/simulation/arm_ik.py:743)、[命令限位校验](../../src/teleoperation/robots/control_spec.py:61)。

仅比较两种模式都执行过的前 1,274 帧，仍排除最初 2 秒：

| 指标（中位数） | 原模式左 | 关闭朝向左 | 原模式右 | 关闭朝向右 |
|---|---:|---:|---:|---:|
| 位置误差 / mm | 121.246 | 91.057 | 125.809 | 78.633 |
| 朝向误差 / deg | 50.784 | 94.618 | 54.888 | 137.402 |

右侧在两种模式各自建立参考点时出现恒定约 0.0506 mm 的最大分量差（左侧为零），因此第二段不能称为位置目标逐位完全相同。已验证差值是常量、保留原值且未做抵消；图中明确标注。这个小偏移不能解释数厘米到十厘米的误差变化，但应作为比较限制报告。

停止后的原模式没有执行数据，不与关闭朝向的完整 165 秒统计混作公平对比。[共同前缀图](../../outputs/acceptance/gr1-arm-diagnosis-final/gr1-arm-diagnosis-replay165-complete.png)。

## 8. 参考选择与可达性边界

为单独展示参考选择的影响，额外比较同一腕输入的相对位移 `Q*(w(t)-w(0))` 与 `Q*((w(t)-h(t))-(w(0)-h(0)))`。采用同一轴映射、单位增益、无机器人位置偏移；两者差只由头部平移变化决定。它不冒充完整官方末端目标，也不混入机器人惯性基座或不同模型。

对应图保存在最终图目录的 `*-reference-motion.png`。原样复现中的 ARKit/WebXR 腕局部轴不确定性仍未闭合，本轮没有用它对完整位姿作公平排名。

真实录制逐帧保存了候选的有限性、关节限位、误差和是否接受。未找到符合条件的解标为本求解未找到，不将其称为数学不可达。对这些真实位姿的全局可达性、接触可行性和操作成功率尚无证明。

## 9. 验证与复现

- 诊断工具测试 10 项通过。
- 诊断及手臂、腕、控制会话相关测试共 79 项通过。
- 完整 `unittest discover -s tests`：622 项通过。
- 103 秒两模式运行退出码 0；165 秒诊断退出码 1，明确表示记录到原生产命令失败；两模式的前缀/完整结果均保存。
- 官方检出保持干净；当前生产文件哈希与修改前相同。
- 源码、模型、输入、包版本和运行脚本快照见最终目录 `provenance.json`；脚本输出时的边界保存版本差异只涉及诊断故障记录，不是控制算法修改。

```powershell
$env:PYTHONPATH = "$PWD/src"
$python = '.venv/dex-native/Scripts/python.exe'
$upstream = 'D:/2026/code/TeleOpBench-reference'
& $python -m unittest tests.test_gr1_arm_diagnosis -v
& $python devtools/gr1_arm_diagnosis.py --stage fk --upstream $upstream --output outputs/acceptance/gr1-fk-new
& $python devtools/gr1_arm_diagnosis.py --stage matrix --upstream $upstream --output outputs/acceptance/gr1-matrix-new
& $python devtools/gr1_arm_diagnosis.py --stage probe --upstream $upstream --output outputs/acceptance/gr1-probe-new
& $python devtools/gr1_arm_diagnosis.py --stage replay --upstream $upstream --input datasets/raw/visionpro_hand_arm_20261012_215137_live.h5 --output outputs/acceptance/gr1-replay103-new
& $python devtools/gr1_arm_diagnosis.py --stage replay --upstream $upstream --input datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 --output outputs/acceptance/gr1-replay165-new
& $python -m unittest discover -s tests
```

输出目录要求不存在。PINK 使用上一轮产物目录中的隔离依赖；Pinocchio/PyBullet 使用 dex-native 环境。停止帧数值对本机该依赖和物理版本可重复，不承诺跨版本逐位一致。

## 10. 后续最小修改建议

第一项修复应独立处理保持语义：新解被拒绝时保持上一条合法接受命令，不能把可能已越限的实际反馈直接转为新命令；同时保留实际状态用于诊断，不伪造观测或放宽限位。

第二项需要针对完整位姿设计位置/朝向取舍：不能仅关闭朝向，也不能无条件接受位置候选。后续应比较经过验证的朝向目标、候选选择和运动连续性，分别报告位置进展与朝向代价，再做操作验收。

这两项是下一次业务变更范围，不在本轮诊断中实施。当前不应以修改模型标签、增加滤波、放大限位或直接替换求解器来掩盖已发现的问题。单目视觉、原生 WebXR 验收、其他机器人和实际物体操作仍属于后续阶段。
