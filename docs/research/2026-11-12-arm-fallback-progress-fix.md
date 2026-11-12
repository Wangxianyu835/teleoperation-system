# 第二批修复：让有位置进展的退回候选通过

日期：2026-11-12。基线提交 `486b1defc6ba5273116901e936fb7aadb52cf8b1`，修改前检查点 `checkpoint-20261112-1356-arm-fallback-progress`。本批落实用户要求的最小卡住机制修复；上一批“拒绝后保持上一条合法命令”继续生效。

## 原因与行为

`PyBulletArmIK._ik_sequence` 已经在位姿解的位置残差过大且不如位置解时，选择位置解并标记 `position_fallback=True`。原先 `solve` 看到这个标记就设置 `pose_position_conflict`，于是刚选出的更好位置解又被拒绝，上层保持旧命令。即使位置目标可达，也可能每帧重复该过程。

本批只修改生产文件 `src/teleoperation/simulation/arm_ik.py` 的候选接受阶段：

1. `position_fallback` 不再直接触发拒绝。每次仍执行既有的位置、朝向求解和候选选择。
2. 保留有限性和原有 URDF 限位检查：大于 `0.01 rad` 的越限仍拒绝；容差内的数值越限先夹到精确边界。
3. 保留原有单次关节变化上限 `0.35 rad`，从本帧实际关节状态向候选缩短步长。这是实际状态到本次命令的差值约束，不是时间归一化的速度约束，也不等同于相邻命令差值约束。
4. 再检查缩步后的输出符合精确限位，避免实际起点已越限时，缩步结果仍越限。
5. 对位置退回候选，临时设置这次真正要输出的关节角，通过 FK 重新测量。只有相对本帧实际姿态的位置误差改善超过 `1e-6 m`（1 微米）才接受。该阈值只排除 FK 舍入差异，不是跟随精度达标阈值。
6. 没有进展则设置 `position_fallback_no_progress`，上层沿用上一条合法命令。FK 检查无论成功或异常，都恢复原有全部活动关节位置、速度及 GUI 渲染状态。

诊断增加 `fallback_position_before_mm`、`fallback_position_after_mm` 和 `fallback_orientation_after_deg`，分别记录限幅后候选的位置进展和朝向代价。没有新增命令缓存、公共接口、重试、求解器或坐标转换。

这条规则只保证被接受的位置退回候选相对同一帧实际起点更接近同一位置目标；不保证其朝向改善，不保证动态系统下一时刻一定改善，也不保证移动目标的跨帧误差单调下降。

## 已知可达卡住反例

复用前次定位到的右臂目标，由合法目标关节角通过官方固定版本 GR1 模型 FK 生成。调用实际生产 `solve`，不开启诊断代码中的假想接受覆盖。

| 阶段 | 位置误差 | 朝向误差 |
|---|---:|---:|
| 初始 / 旧规则保持处 | 164.331 mm | 62.420° |
| 本批第 1 次更新 | 77.538 mm | 71.323° |
| 本批第 10 次更新 | 3.294 mm | 79.427° |
| 本批第 60 次更新 | 3.294 mm | 79.427° |

原先位置候选约 3.308 mm、位姿候选约 60.370 mm，却最终维持 164.331 mm；本批位置候选能实际输出。接近后再无超过 1 微米的位置进展时保持，所以后续出现 `position_fallback_no_progress` 是真实局部停滞，不再是“一出现退回标记就冻结”。

上述是关节命令逐次到位的受控运动学实验，不是物理驱动或实体机器人结果。每次输出均验证精确限位及 `0.35 rad` 步长，并验证临时 FK 不改变仿真状态。

## 原生 GR1 录制对照

比较对象是紧邻本批的 `486b1de` 保持来源修复版本，不混用更早会中断的版本。两边均走 `--mode baseline`，保留朝向输入，使用相同录制、模型、目标处理及输出路径。`baseline` 是原诊断工具中“保留朝向”的模式名，本次含义是前后代码版本对照，不是关闭朝向的消融。

详细逐帧验证、位置和朝向统计记录在 [verified.json](../../outputs/acceptance/gr1-fallback-progress-verification/verified.json)。统计与前次一致，排除开头 2 秒。`baseline-summary.json` 中的位置和朝向误差使用接受命令的 FK；验证文件另列仿真物理实际位置误差，二者分别记录。

两段分别完整完成 2,886 帧和 4,631 帧，无命令校验或运行中断。仿真物理实际位置相对同帧目标的误差中位数：

| 录制 | 手臂 | 修改前 | 修改后 |
|---|---|---:|---:|
| 103 秒 | 左 | 210.3 mm | 23.6 mm |
| 103 秒 | 右 | 137.6 mm | 33.2 mm |
| 165 秒 | 左 | 73.0 mm | 56.9 mm |
| 165 秒 | 右 | 75.6 mm | 45.3 mm |

接受命令 FK 的位置误差中位数分别为：103 秒左 `210.9 → 23.8 mm`、右 `137.7 → 33.5 mm`；165 秒左 `72.4 → 57.3 mm`、右 `75.4 → 45.9 mm`。它们与上述物理实际结果接近，但不混为同一指标。

位置改善伴随明显朝向代价。接受命令 FK 的朝向误差中位数：

| 录制 | 手臂 | 修改前 | 修改后 |
|---|---|---:|---:|
| 103 秒 | 左 | 23.4° | 130.3° |
| 103 秒 | 右 | 39.9° | 67.9° |
| 165 秒 | 左 | 38.4° | 57.7° |
| 165 秒 | 右 | 40.4° | 78.6° |

这说明最小修复解除了一类位置解误拒绝，但没有解决完整位姿跟随。尤其 103 秒左侧的朝向误差足以影响实际操作，不能把“位置更近”视为“操作姿态正确”。

逐帧验证结果：

- 103 秒接受位置退回更新：左 265、右 121；165 秒：左 215、右 340。这些输出的 FK 位置进展均超过 1 微米，且 FK 与命令输出一致。
- 两段全部记录命令有限且符合原有限位。拒绝时 `observed=False`，与上一合法命令严格数组相等：103 秒左 2,606、右 2,744 次；165 秒左 4,201、右 4,187 次。
- 仍有大量真实无进展拒绝：103 秒左 2,606、右 2,729 次；165 秒左 3,534、右 4,079 次。另有合法性拒绝：103 秒右 15 次；165 秒左 667、右 108 次。不能据此宣称所有卡住都消失。
- 165 秒仍有左 49、右 16 次原始位置候选比当前状态好却保持的诊断标记。这些原始候选尚未等同于经过精确限位和缩步的最终输出；本批保留最终输出检查，不为了消除统计标记绕过检查。

当前规则没有朝向退化上限。若下一步要求位置和朝向同时可用，应另做候选约束或完整位姿求解实验；本批不继续叠加未经验证的策略。

## 测试与复现

新增 7 项受控接受规则测试先在旧代码上运行，5 项失败；实现后全部通过。覆盖：合法退回前进、最终候选很好但缩步后变差、无进展、NaN/Inf/显著越限、实际起点越限导致缩步结果仍越限、FK 异常恢复位置速度、普通位姿候选维持既有接受流程。

另增加上述真实 GR1 可达反例回归，并调整两项旧测试，使其分别验证有界进展和远目标仍未到达，避免把“必须冻结”当成正确行为。

专项测试 39 项通过；完整 `unittest discover -s tests` 共 632 项通过，耗时 178.187 秒，包括现有相机、H1-2、G1、GR1 和手部路径。

```powershell
$env:PYTHONPATH = "$PWD/src"
$python = '.venv/dex-native/Scripts/python.exe'
& $python -m unittest tests.test_arm_fallback_progress tests.test_gr1_arm_diagnosis tests.test_arm_following tests.test_arm_command_production tests.test_robot_control_session -v
& $python -m unittest discover -s tests
& $python devtools/gr1_arm_diagnosis.py --stage replay --mode baseline --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_215137_live.h5 --output outputs/acceptance/gr1-fallback-progress103
& $python devtools/gr1_arm_diagnosis.py --stage replay --mode baseline --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 --output outputs/acceptance/gr1-fallback-progress165
& $python outputs/acceptance/gr1-fallback-progress-verification/witness.py
& $python outputs/acceptance/gr1-fallback-progress-verification/verify.py
```

重放输出目录要求不存在。产物位于本地、按仓库规则忽略；本报告及测试随源码提交。

## 来源与适用范围

[provenance.json](../../outputs/acceptance/gr1-fallback-progress-verification/provenance.json) 保存本次生产文件、诊断工具、测试和原始 H5 的 SHA-256；同目录保存运行时 `arm_ik.py` 快照。逐帧验证还核对当前文件与该来源清单一致。

TransHandR 的手部数值管线没有变动。实际参考源 `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py`（SHA-256 `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f`）和 `model/angle2real.py`（SHA-256 `bad2c807125a1e2c3e26b51a38af35be2336c6d3aa35b5e0e3e7a8bbc9af5d74`）在验证时保持一致。手部输入、窗口、结果接口和相机预处理均未改动。

可达反例使用 `D:/2026/code/TeleOpBench-reference` 的固定参考提交 `4cc410d149ce63c86f485b257cba3b3e0ec031d6` 及原有受控模型。两段原生重放使用项目内 GR1 模型；不能把这两种模型的绝对误差混作一个实验。

本批验证范围是离线录制和仿真。没有连接实时头显或实体机器人，没有验证接触、抓取和双臂物体操作。朝向代价需要后续单独处理，当前不能据此宣称“操作目标一致”已实现。
