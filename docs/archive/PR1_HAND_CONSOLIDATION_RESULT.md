# PR1 Hand Retargeting Consolidation Result

日期：2026-10-06。已实际修改、测试并创建本地 commits；未 push。
teleoperation-system 为唯一 production hand-retargeting source of truth。

## 1. Source SHA

`138fc2d9cc11580421c7094c5f4b8c703a690b82`，branch `refactor/mytrans-cleanup`。
parent `4a8837f243bc8969ec5773d7605d2702bdc3582e`；original baseline
`1267171b2bd6ae4dffb896c26329813fed00996d`。迁移前后 source HEAD 与 clean 工作树均验证。

## 2. Destination baseline

`2df04ce431dc08b3186a422a3c842ed27550875c`。
开始修改前验证目标 branch=`retargeting`、HEAD=该 SHA、working tree clean。
目标 repository：`https://github.com/Wangxianyu835/teleoperation-system.git`。

## 3. Working branch

`refactor/consolidate-hand-retargeting`，从固定目标 baseline 创建。

## 4. Files copied from mytrans

按文件 ownership 使用固定 SHA 的 Git blob 同步，没有复制整个 repo/package。
以下 19 个实现文件与固定 source 内容相同（仅忽略换行与 EOF 空行）：

```text
model/losses.py
retargeting/coordinates.py
retargeting/data.py
retargeting/training.py
retargeting/model.py
retargeting/hand_core.py
retargeting/retargeter.py
retargeting/inputs/__init__.py
retargeting/inputs/mediapipe.py
retargeting/mediapipe.py
retargeting/preprocessing.py
retargeting/runtime.py
retargeting/plotting.py
retargeting/visualization.py
retargeting/inference.py
retargeting/simulation.py
scripts/compare_training_hand_pose.py
scripts/diagnose_hand_coordinates.py
scripts/verify_p0.py
```

`inference.py` 与 hand-only `retargeting/simulation.py` 逐项比较后同步，继续使用手部 H5
消费协议；没有覆盖目标的 `simulation/` 包。模型 architecture / FK 两个文件两边相同。

## 5. Files manually merged

| 文件 | 合并决定 |
|---|---|
| `retargeting/config.py` | hand 常量/defaults 合入；保留目标历史 offline checkpoint 默认、L21 URDF，新增输出归 ignored outputs |
| `retargeting/contracts.py` | hand shape 常量统一；保留目标 arm 校验、VisionPro compatibility、固定 48D action helpers |
| `retargeting/cli.py`, `__main__.py` | 统一五个 hand commands，lazy import；不迁 source dual commands |
| `config/retarget_io.py` | hand 校验/构建委托 canonical；保留原 arm/action 行为 |
| `input_adapters/npy_replay_adapter.py` | 修复不存在模块的 import，改用 CanonicalHandProcessor 与独立侧重置 |
| `main_train_twohand.py`, `main_offline_twohand.py` | 仅作为 hand package CLI compatibility wrappers |
| `scripts/align_h5_coordinates.py` | 兼容 wrapper，唯一算法在 preprocessing；现在生成 palm-local |
| requirements 文件 | 按实际 imports 分类，root application requirements 保持原样 |
| `tests/test_cli_integration.py` | 仅保留 hand CLI，移除 source dual command 断言 |
| `tests/test_dual_arm.py` | 只增加外部 URDF env override，原默认及测试断言保留 |
| README、两份历史上下文 docs、新增 hand/PR2/result docs | 保留系统身份，记录当前 hand 维护入口与协议 |

## 6. Files intentionally kept from teleoperation-system

保留 `teleop/`、`envs/`、`tasks/`、`simulation/`、robot adapters/模型资产、benchmark、
recorder、`main.py`、`demo_teleop.py`、三个 dual roots、目标 arm/command/dual modules。
`model/pose_transformer.py`、`model/kinematics.py` 与 source 相同，无 diff。
`retargeting/tracking.py` 保留目标版本（两边数值实现相同，避免无意义差异）。
对 590 个预先登记的目标 owned 文件逐个 SHA256 核验：588 个未变；其余两个仅为
第 5 节中明确手部边界合并的 `config/retarget_io.py` 和 NPY adapter。
混合模块的 arm/action/VisionPro 函数 AST 与 baseline 完全一致。

## 7. mytrans files intentionally NOT migrated

- `dual_realtime.py`、`dual_replay.py`：未迁；目标 root dual application/teleop 架构保留。
- `dual_teleop.py`：未用 source 覆盖，保留 destination。
- `command.py`：未用 source 覆盖，保留目标已存在的 RobotCommand；PR1 不重新设计整机协议。
- `config.py`：只合并 hand-specific 部分；未迁 source 全局 robot/TRON2A 路径。
- `runtime.py`：只迁共享设备选择；目标系统配置继续独立负责应用职责。
- requirements：未整体替换；保留 application deps，分 runtime/training/camera/tools/dev。
- source `test_runtime_defaults.py`：未迁依赖历史 mytrans run 路径的断言；新增目标 default 回归。
- source checkpoints/weights、recording/H5、outputs/runs/logs/PNG/tensors/cache/IDE、third_party：均未迁。

所有 smoke 数据和权重是目标本地新生成，位于 Git 忽略的 `outputs/pr1_checks/`。
测试只读使用已有外部 TRON2A URDF，不复制第三方 description。

## 8. Contract verification

Hand25 topology、MediaPipe21 转换、`[B,3,25,3]`、18D=root+17DOF、18→17 conversion 均通过。
legacy 标识/矩阵/function 保留。palm-local MCP=6/11/16/21，row-vector
`points @ B_source @ B_robot.T`，左右独立零位 FK basis，orthogonal、det +1、
translation/rigid-rotation invariance、distance preservation、source 退化 zeros、robot
退化 fatal 均通过。无新增 scale normalization、smoothing、previous-frame fallback 或 legacy stacking。

| H5 | Checkpoint | 结果 |
|---|---|---|
| legacy | legacy | PASS，实际训练/导出 |
| palm-local | palm-local | PASS，实际训练/导出/warm start |
| legacy | palm-local | FAIL as required，已有输出 bytes 保留 |
| palm-local | legacy | FAIL as required，导出及 warm start 拒绝 |

H5 missing/empty/unknown alignment、checkpoint missing/mismatch、训练 Dataset equality、
事务式 preprocessing 全由迁移回归保护；没有 force、ignore 或 filename inference。

## 9. Test migration

迁入/更新 14 个 source hand test 文件，包括 transaction、coordinate geometry/pipeline、
diagnostics、losses、HandRetargeter、MediaPipe capture/realtime、training safety、Dataset、CLI。
另新增 `test_hand_application_integration.py`，保留现有目标测试；只改 dual-arm 测试资产路径配置。
新增集成回归覆盖 NPY 缺帧重置、VisionPro 三帧、H1/GR1/G1 native 手映射、目标默认路径
与保留的 application packing 行为。

## 10. Test commands

使用现有 `D:/Anaconda/envs/TransHandR/python.exe`（Python 3.10.20），未安装/改写环境。
将 TEMP/TMP/MPLCONFIGDIR 指向 ignored outputs，禁用 bytecode 写入。

```powershell
$env:TRON2A_TEST_URDF='D:/2026/code/mytrans/third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf'
& 'D:/Anaconda/envs/TransHandR/python.exe' -B -m unittest discover -s tests -v
& 'D:/Anaconda/envs/TransHandR/python.exe' -B -u outputs/pr1_checks/cli_smoke.py
& 'D:/Anaconda/envs/TransHandR/python.exe' -B outputs/pr1_checks/fresh_checkpoint_equivalence.py
& 'D:/Anaconda/envs/TransHandR/python.exe' -B outputs/pr1_checks/verify_ownership.py
git diff --check
```

`outputs/pr1_checks/*` 为本机保留的验证脚本/日志，不是迁入或提交的生产源码。
clone 后的完整测试资源配置方式见 [使用说明](HAND_RETARGETING.md)。

## 11. Test results

目标 baseline：43 项，40 passed、3 errors（缺 TRON2A URDF）。
合并后完整 tests：**166 项，165 passed、1 skipped、0 failed/errors**。
唯一 skip 为历史 `palm_local_v2` 权重缺失的离线/实时角度等价性用例；未迁移该权重。

另外以本次实际训练的新 checkpoint 执行该等价性测试的完整 scenario assertions：
**61 次角度比较，最大误差 `2.98023224e-07` radians**。真实模型 realtime CLI 的
synthetic capture 检查也通过：左 2 次输出、右 5 次，左侧缺帧后重新三帧恢复，资源释放一次。

验证环境已有 NumPy 2.2.5 / SciPy 1.15.3 / Torch 2.10.0+cu130 / PyBullet 3.2.7 等；
未将该已有环境冒充 requirements 全新安装验收。最初 root Python 3.12 的缺 PyBullet
错误通过使用现有完整 TransHandR 环境解决。干净环境安装未另做。

## 12. CLI smoke results

**31 项命令检查全部通过**，覆盖 package root/五个 subcommands help、三个 hand wrappers、
三个目标 dual roots help，以及如下真实链路：

| 检查 | 结果 |
|---|---|
| align | raw 40 帧 → palm Hand25，原输入 SHA256 不变 |
| legacy / palm train | 各 CPU 1 epoch，30 train windows / 4 validation windows；保存有限 best_val 与模式 metadata |
| legacy / palm export + inspect | 各侧 `(40,18)`、38 valid、root=0、finite，输出 alignment 相等 |
| 交叉 checkpoint export | 两种 mismatch 均非零退出；输出 sentinel bytes 不变 |
| warm start | 同模式继续训练 1 epoch；异模式拒绝 |
| `verify_p0.py` | 原有 FK / losses / model gradients 有限，input/checkpoint hash 不变 |
| FK comparison | 实际载入 checkpoint 并生成本地诊断 PNG |
| NPY validator | 缺帧序列输出 2 个完整窗口 |
| native hand replay | H1-2 / GR1-T2 / G1 各双手 40 帧，PyBullet DIRECT 完成 |
| system replay | H1/pushcube 38D dummy actions 40 步完成播放 |
| `test_import.py` | 六项检查含 H1/pushcube 300 步端到端，无 FAIL/traceback |

synthetic replay 只验工程链路：pushcube dummy 动作未完成任务；H1 native 手限位
clamp 16.67%，其余两模型此数据 clamp 为 0。未将一轮 synthetic 训练当作模型质量提升。
真实摄像头/VR/手套硬件未运行；realtime 输入及资源生命周期由回归和 synthetic capture 覆盖。

## 13. Integration regressions found

NPY replay 原 import 指向已不存在的 `input_adapters.hand_keypoints`，已修复并纳入回归。
迁移后 strict hand fixtures 需要真实 alignment 声明，已更新测试。
baseline 缺 TRON2A description 导致的三项错误用显式 test-only URDF override 解决，
没有复制资源、skip arm 测试或改变 arm IK。没有未解决的本次集成回归。
benchmark/task/robot architecture、VisionPro legacy 与所有受保护应用文件保持原状。

## 14. PR2 follow-up findings

见 [PR2_FOLLOW_UP.md](PR2_FOLLOW_UP.md)：ACTION_ORDER 存在应用间顺序差异；固定
48D RobotCommand 与 benchmark 38/36/28D 需显式转换；partial command helper
会生成可变长度；validity 控制策略、RobotJointCommand 重复边界、PyBullet 占位 indices、
arm IK/calibration 接口、TRON2A joint-name mapping 和外部资产路径需统一。未在 PR1 改设计。

## 15. git diff --stat

代码/CLI commit：33 files，1759 insertions / 446 deletions。
测试 commit：16 files，2159 insertions / 1 deletion。
另有六个文档文件：README、OFFLINE_PIPELINE、PROJECT_CONTEXT、HAND_RETARGETING、
PR2_FOLLOW_UP 和本报告。完整最终统计以实际提交为准：

```powershell
git diff --stat 2df04ce431dc08b3186a422a3c842ed27550875c..HEAD
```

## 16. git status

交付目标：destination branch 为 `refactor/consolidate-hand-retargeting`，全部授权修改
提交为本地 commits，`git status --short` 无输出；source 同样 clean 且 HEAD 未变。
ignored outputs 保留本机验证证据，不进入版本控制。

## 17. commits

- `faa3740cac2a9fcffa3de8ec2da237b7e6ff7fb1` — `refactor(retargeting): consolidate canonical hand core and CLI`
- `9549c723539a5d66000ebbc1adcf7e081769c2b2` — `test(retargeting): migrate hand contracts and application regressions`
- 包含本报告的文档 commit — `docs(retargeting): document canonical hand pipeline and PR1 results`

包含本报告的文档 commit 的实际 SHA 可通过
`git log --oneline 2df04ce431dc08b3186a422a3c842ed27550875c..HEAD` 查询；不在文档内写自引用 SHA。
未 push。

## 18. Remaining blockers

PR1 无阻塞。外部 TRON2A description、真实历史 checkpoint 和相机 Tasks asset 不由
本次源码迁移携带；对应环境需自行配置。真实硬件/物理尺度与模型质量未在 synthetic
工程 smoke 中验收；整机 command 与 arm integration 留待 PR2。

## 19. Ready for PR2?

**YES**。canonical hand 已实际合入，完整测试与实际链路通过；PR2 应继续统一目标应用
command 与 robot/arm 消费边界，production 手部实现继续只在 teleoperation-system 修改。
