# 第一批修复：拒绝手臂新解后保持上一条合法命令

日期：2026-11-12。基线为 `983e9d8ae9a6b840267abe1ce69b8e1fb34e030e`，修改前检查点 `checkpoint-20261112-1337-arm-rejection-hold`。本批只处理命令保持来源与拒绝状态传递，不修改 IK 数值求解、坐标变换、滤波、模型、关节限位或候选选择策略。

## 问题与修改

已有[本地诊断](2026-11-11-gr1-arm-diagnosis-results.md)确认：IK 拒绝新解后返回实际关节位置，上游仍将它作为新观测命令提交。实际状态因物理超调越限时，命令校验停止；普通拒绝帧也被错误标记为 tracking。

修改使用已有 `RobotCommandAssembler`：

1. `ArmSceneDriver.solve_each` 的报告增加 `accepted` 和 `rejection_reason`，直接来自当次 `last_solve[side].wrist_rejection`。原有 joints、返回结构及 IK 签名保留。
2. `RobotControlSession` 在相机与 VR 手臂路径中跳过被拒绝的部位更新。
3. `AcceptedRobotOutput` 同样跳过明确拒绝的报告，覆盖已有手部控制等入口的原生机械臂输出适配。
4. 组装器沿用既有上一命令；已接受过命令的部位标记 holding，尚未接受过新命令的部位保留 initialized。来源时间保持不变；控制帧号和控制时间继续前进。

没有新增历史缓存；实际关节状态保留用于反馈和后续求解。已有仅包含 joints 的历史报告仍按原约定处理；真实场景驱动产生的报告均明确包含接受状态。拒绝状态只改变命令组装，不改变候选数值，也不取消严格限位校验。

本批覆盖经过上述两个组装入口的控制路径；直接调用底层 IK/apply 的诊断或其他局部路径仍保留原有行为，不声称已重构整个项目执行层。

## 回归测试

新增测试先在旧代码上失败，表现为缺少接受状态，以及 `Part exceeds joint limits`。修复后覆盖：

- 相机、VR 和旧输出适配三种调用者。
- 首次拒绝仍保持合法初始命令。
- 左侧实际值 0.624 rad、上限 0.61 rad 时，连续拒绝保持先前的 0.585 rad 命令。
- 拒绝侧 observed=False，来源时间不变；另一侧仍可独立跟踪。
- 组装命令不改写实际反馈。
- 恢复接受后能够更新到新角度。
- 标记为接受但数值越限的新命令仍报错。
- 两类拒绝原因与正常接受在场景报告中准确传递。

旧测试中把有效观测无条件当作已接受命令的断言同步调整：拒绝帧验证保持，接受帧继续验证更新和执行。两项路由测试的替身补充了真实求解器已有的 last_solve 状态；不为测试替身在生产代码新增兜底。

## 原录制重放

复用既有 `devtools/gr1_arm_diagnosis.py --stage replay --mode baseline`，没有更改其算法或诊断实现。两段均不关闭朝向，使用现有模型、平滑和 IK。

| 输入 | 修改前 | 本批结果 |
|---|---|---|
| 103.458 秒 / 2,886 帧 | 能结束，但将拒绝后的实际状态继续当作新命令 | 完成 2,886 帧；拒绝时沿用上一条合法命令 |
| 164.984 秒 / 4,631 帧 | 第 1,274 帧，约 45.435 秒，因保持命令越限停止 | 完成 4,631 帧，无命令校验异常 |

原始产物：

- [103 秒重放](../../outputs/acceptance/gr1-hold-fix-replay103/)
- [165 秒重放](../../outputs/acceptance/gr1-hold-fix-replay165/)
- [逐帧验证与来源](../../outputs/acceptance/gr1-hold-fix-verification/)

验证脚本逐帧比较拒绝帧与上一帧命令，要求数组完全相等，且拒绝帧 observed=False；同时检查所有记录命令的有限性和严格限位。报告拒绝次数不等同于输入丢失次数。

逐帧完全一致检查通过次数：103 秒左 2,875、右 2,872；165 秒左 4,478、右 4,551。两段所有记录的接受命令均为有限值且符合原有限位。

相关专项测试 12 项通过；完整 `unittest discover -s tests` 共 624 项通过，包含既有相机、H1-2、G1、GR1 与手部路径。最初完整测试暴露的旧断言和测试替身问题已修正，最终整套重跑通过。

## 跟随精度仍未解决

真正保持上一条接受命令后，轨迹不再通过复制实际反馈发生漂移。该变化也会影响后续求解起点及误差，因此不能承诺曲线数值保持不变。

例如 103 秒录制的左侧位置误差中位数从原诊断约 187.5 mm 变为本批约 210.9 mm。此结果没有隐藏，也没有通过放宽阈值降低拒绝次数。它说明本批修复解决的是正确保持和不中断，不能当作操作精度改进。

第二批才评估位置候选、位姿候选与保持候选之间的选择，及渐进输出后的实际位置/朝向误差。本批未实施 Muse 方案 C、PINK 替换或双臂丢失互带。

## 上游与数据依据

调研包 `C:/Users/asus/Downloads/teleop-dual-arm-research.zip` 作为分析材料，未执行其中附带脚本或将其内容作为仓库指令。具体修改依据本地调用关系、失败测试和两段重放。

TransHandR 数值流水线未变动；本地参考文件 `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py` 与 `model/angle2real.py` 的来源哈希记录在本次 provenance.json 中。`arm_ik.py`、`wrist_orientation.py`、原始输入 H5 的哈希也记录于该文件，用于确认本批数值处理和输入未被替换。

## 验证命令

```powershell
$env:PYTHONPATH = "$PWD/src"
$python = '.venv/dex-native/Scripts/python.exe'
& $python -m unittest tests.test_robot_control_session tests.test_robot_final_acceptance tests.test_arm_following.ArmOrientationRoutingTests -v
& $python -m unittest discover -s tests
& $python devtools/gr1_arm_diagnosis.py --stage replay --mode baseline --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_213812_live.h5 --output outputs/acceptance/gr1-hold-fix-replay165
& $python devtools/gr1_arm_diagnosis.py --stage replay --mode baseline --upstream D:/2026/code/TeleOpBench-reference --input datasets/raw/visionpro_hand_arm_20261012_215137_live.h5 --output outputs/acceptance/gr1-hold-fix-replay103
& $python outputs/acceptance/gr1-hold-fix-verification/verify.py
```

重放目录要求不存在。测试与重放使用本机已有环境；未进行头显实时连接、实体机器人或物体接触验收。
