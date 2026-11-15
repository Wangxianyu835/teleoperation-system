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
- 原有用户工作树、环境和未跟踪文件均保留。来源分支保留供用户决定清理。

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

## restore 来源逐提交处置

来源固定为 `d0be525ea8f9cdd110b700a88b0f91d9e54362ad`。
第二阶段检查点：`checkpoint-20261114-2156-before-restore-merge`，指向 `bccc095`。

| 提交 | 内容 | 集成处理 |
|---|---|---|
| `b349acc` | 相机与仿真窗口并排 | Win32 查询、布局和有限等待线程迁入 windows_runtime；headless/no-preview/off 不触碰窗口 |
| `1c3ad33` | PyCharm 路径说明去硬编码 | 保留当前薄启动器和共享 setup，当前文档不引用旧工作树 |
| `d69b49d` | 主启动器接入肘跟随快捷参数 | 共享 apply_elbow_preset；旧肘跟随文件仍作为兼容入口，转调同一实现 |
| `69e4def` | 统一入口与旧测试基线 | 保留当前 AGENTS；增量测试行为，不恢复旧失败容忍和重复实现 |
| `9bcc3fc` | launcher 帮助提示 | 当前 runtime 在 --help 时解释 --elbow-follow/--off/--check |
| `ce1947c` | 旧流水线文档更新 | 实际入口和限制迁入当前使用报告及本文 |
| `5810fce` | operator 诊断启动默认值 | 仅手臂诊断 launcher 默认 operator，显式公开参数优先；正式 robot control 默认不变 |
| `e69950d` | GR1 肘反向加载修正 | 在共享机器人参考声明中修正，加载器/部分场景/PINK 共用；没有在 IK 构造器中另改场景 |
| `d0be525` | 肘修正的现场读法 | 记录本次 FK 测量、模型身份和拟合迁移，不冒充旧现场测量 |

没有恢复旧 arm_ik、realtime_teleop、verify_camera_record 或测试文件；当前事务提交、
求解失败保持、临时关节状态恢复、输入契约和相机 SMPL/PINK 路线继续有效。

### 官方 GR1 几何证据与拟合迁移

本次先在未修改的场景测量，再对肘角取反测量，最后在共享参考声明中实施。
GR1 官方 URDF SHA256：`5686e0189a9277c5fa01e67e5b0048581e1c7719fb37063f13e53e5a1d123ae0`。
修改过的 GR1 URDF 必须先验证其参考姿态，代码明确拒绝把官方模型的修正套给未知模型。

| 侧 | 原肘角 rad | 原前臂方向 x | 新肘角 rad | 新前臂方向 x | 原/新弯角 deg |
|---|---:|---:|---:|---:|---|
| 左 | +1.2 | -0.795097 | -1.2 | +0.985492 | 111.204741 / 111.204696 |
| 右 | +1.2 | -0.795460 | -1.2 | +0.985128 | 111.204594 / 111.204612 |

H1-2、G1、TRON2A 规格 hash 经与既有拟合逐一比较均未改变。只有 GR1 从
`fd6f470b904cd8b14248073e2955c340b3e702e09422d777cda2a2db43f3859f`
变为 `216e8837ab7da104b4c9111ceeed29cf5c2612d04666f5b9603eb599208b6dbf`。

旧 GR1 拟合被明确拒绝。本次使用同一许可 SMPL-X neutral 模型和 20 betas，
重新执行 300 次拟合，文件保存到工作树 `outputs/tmp_branch_integration/fittings-reference-v3/gr1_t2.json`。
人体模型 hash 不变，betas 与原拟合最大差为 0，RMS 仍为 0.016272710636258125 m。
这是重新运行拟合的产物，没有改写旧文件 hash。

其他机器重建示例（路径按实际位置替换）：

```powershell
python -m teleoperation arm fit --robots gr1_t2 --body-model-path datasets/smpl_models `
  --smplx-module-dir outputs/unified-arm-runtime --body-model-type smplx `
  --gender neutral --num-betas 20 --fit-steps 300 --device cpu `
  --output-dir outputs/fittings-reference-v3 --report outputs/fit-reference-v3.json
```

使用新 GR1 时通过 `--arm-fitting-dir` 或操作脚本 `-FittingDirectory` 显式选择新目录。
多机器人需要该目录同时包含全部所选机器人拟合；未受影响的旧拟合可原样复制后复用。
仓库不分发许可人体模型，也不提交本机生成的拟合文件。

### 第二阶段验证

- 机械臂/启动器/PINK/场景针对性回归：92 tests，OK，skipped=1。
- 新增共享参考、旧拟合拒绝和窗口/参数测试先失败后通过；IK 重新构造保留所有关节位置和速度。
- 真实 SMPL-X 重建 + 合成身体录像 + 正式 robot control：12 个观测、38 帧接受命令，last_errors 为空。
- `robot replay` 回放同一录制：38 帧、91 个物理步，退出码 0。
- 对应报告在 `outputs/tmp_branch_integration/gr1-control-reference-v3.json`、`gr1-replay-reference-v3.json`。
- 没有开启相机、头显或 GUI；真人手感和实际窗口摆放仍需设备验收。
- 最终完整回归：718 tests，OK，skipped=2。依赖、CLI help、完整 unittest 和 30 项场景体检四项门禁全部通过。
- 最终门禁日志：`outputs/preflight/20261114_220450_381/`。
- 已在新拟合目录补齐 H1-2、G1、TRON2A 的原拟合文件；复制前逐一校验当前机器人规格 hash，未修改其内容。

## 交付核验

两条来源均使用独立双父 merge commit，交付时核对固定来源 SHA 是目标分支的祖先。
此前审查的三个独立缺陷仍不属于本次集成修复。复现命令和实际产物保留在上述日志目录，
身体录像验证为合成输入，不能作为设备验收。
