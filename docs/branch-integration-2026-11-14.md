# 2026-11-14 分支集成记录

目标为 `wrist-input-arm-follow`。在现有独立工作树
`D:/2026/code/PyCharmMiscProject/teleoperation-smpl-pink` 上使用
`codex/integrate-restore-and-simulation` 实施，保留主项目工作树。

## 基线

- 正式代码基线：`9326ed0`；计划文档提交：`aa121c8`。
- 第一阶段检查点：`checkpoint-20261114-2141-before-simulation-merge`，指向 `aa121c8`。
- 实施前完整回归：690 tests，OK，skipped=2。
- 解释器：`D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Scripts/python.exe`。
- PINK/QP 搜索路径：主项目 `outputs/acceptance/teleopbench-gr1-20261111/dependencies`。
- 原有用户工作树、环境和未跟踪文件均保留。当前阶段未推送、未删除来源分支。

## 仿真来源逐提交处置

来源固定为 `41287ce789db37415917529df2b7a52717e5cfff`。

| 提交 | 内容 | 集成处理 |
|---|---|---|
| `759d655` | 恢复旧约定文档 | 不恢复已淘汰的约定；根 AGENTS.md 继续生效 |
| `8ec3c21` | posture/肩锚点/肘跟随离线 A/B | 增量合入 verify_realtime_arm_bridge 及测试，正式相机 PINK 路线不变 |
| `4d78359` | 步数耗尽记录失败 | 合入 applications/simulation.py，墙钟返回契约保留 |
| `5c1682c` | 30 项场景、机构和判据修正 | 合入任务模块、域随机化和场景回归测试 |
| `fca4d15` | 清理任务模块未使用导入 | 随任务模块合入 |
| `a7318cf` | 持续成功窗口、仿真时间与门禁 | 合入环境/metrics；preflight 改为当前命令和严格退出码 |
| `41287ce` | reset 前仿真时钟初始化 | 合入环境构造器 |

README 保留当前入口说明；旧 `CONVENTIONS.md`、`RETARGETING_PIPELINE.md`、
`multi-robot-arm.md` 不恢复，实际有用的解释进入当前文档。

### 第一阶段证据

- 合入前来源场景用例实际失败：liftmug/opendrawer/openmicrowave 初始即成功；杯盖测试初始判据错误。
- 合入后场景、入口、桥接测试：127 tests，OK。
- 本轮新增 episode 重复完成测试先失败（一次完成被记为 4 条），再修为只记 1 条；reset 后新 episode 可正常记录。
- 同时覆盖零步预算、仿真超时和录制观察者继续接收墙钟秒。
- 场景体检：PASS=30，FAIL=0；臂展不足保留为报告限制。
- 任务语义：drawerbook 不要求放入后关闭抽屉；rotatehearth 不要求按下。未声称等同原始完整任务。
- 日志位于 `outputs/tmp_branch_integration/` 和 `outputs/preflight/`。

- 第一阶段完整门禁：710 tests，OK，skipped=2；依赖、CLI help、完整 suite 和再次场景体检四项均成功。
- 门禁日志：`outputs/preflight/20261114_215034_846/`。

第二阶段处置将在该批次验证后补充。
