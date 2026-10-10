# current-state 验收与实时手部坐标修复（2026-10-09）

## A. 分支、环境与验收结论

- 分支 `current-state`，实际 HEAD `1df25ca522fb07f47cc1420bf53f8aa3794f0a0d`，与指定基准相同。
- 开始时工作区干净；本次改动尚未提交。没有 merge/rebase，也没有修改历史 checkpoint。
- 正式环境：`D:\Anaconda\envs\teleoperation\python.exe`，Python 3.10.20、Torch 2.14.1+cu130、NumPy 1.26.4。测试、推理与短训练使用 CPU。
- 沙箱内 `conda.exe` 启动出现系统 DLL 重定位错误，直接调用正式环境 Python 正常。
- `pip install -e . --no-deps` 初次遇到沙箱临时文件权限限制；改用仓库临时目录，在正式环境成功执行 `pip install -e . --no-deps --no-build-isolation --disable-pip-version-check`，未升级依赖。
- 完整 unittest 没有新增失败。仍有 3 个 ERROR 来自缺失外部 TRON2A URDF，不能将完整验收写成全绿。

| 检查 | 修改前 | 最终结果 |
|---|---|---|
| 完整 unittest | 197 项；189 PASS、0 FAIL、7 个测试方法 ERROR、1 SKIP；ERROR 记录共 17 条（含 subtests） | 212 项；208 PASS、0 FAIL、3 ERROR、1 SKIP |
| 架构边界 | 7/7 PASS | 7/7 PASS |
| 新增坐标测试 | 尚未存在 | 15/15 PASS |
| Package / 五类 CLI help | PASS | PASS |
| 三机器人 120-step smoke | 全部 PASS | 全部 PASS |
| 两种坐标 H5、CPU 短训练、checkpoint reload、导出 | 与重构前历史精确一致 | 与同一历史精确一致 |
| 实际 TRON2A 双臂 FK / IK / dual export | URDF 缺失 | 仍受阻，未改成 skip |
| 正式 palm_local_v2 模型、真实摄像头 | 默认资源缺失 | 未验收；原有模型测试仍 SKIP |

初次沙箱测试还有临时目录与硬链接权限错误。使用仓库内 TEMP/TMP，并在批准的非沙箱执行中重跑完整测试后排除了这些错误。上表采用相同条件下的修改前与最终结果，未把权限错误算作代码缺陷。

保留的三个 ERROR，均为 `FileNotFoundError: TRON2A URDF was not found`：

1. `test_dual_arm.DualArmTests.test_dach_urdf_has_two_complete_seven_dof_chains`
2. `test_dual_arm.DualArmTests.test_ik_accepts_previous_solution_for_its_current_pose`
3. `test_dual_arm.DualArmTests.test_offline_export_has_fixed_command_and_holds_invalid_arm`

原有 SKIP：`test_mediapipe_realtime.PalmLocalV2AngleEquivalenceTests.test_same_checkpoint_batched_offline_and_streaming_realtime_angles`，默认 palm_local_v2 checkpoint 不存在。

默认 legacy checkpoint、默认 palm_local_v2 checkpoint、`third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf` 和仓库根 `hand_landmarker.task` 均不存在。另一个工作区有 MediaPipe task 文件，但本轮 synthetic 验证不打开摄像头，也不复制资源。

## B. 发现的问题与修复

| 文件 / 函数 | 原因与影响 | 状态 |
|---|---|---|
| `apps/dual.py` / `DualTeleoperator.__init__` | 固定 canonical pipeline，无法为 MediaPipe normalized 数据执行正式 palm-local preprocessing | 已修复：显式模式并复用共享 pipeline |
| `apps/dual.py` / `DualTeleoperator.update` | 双侧 strict finite 检查提前中止，使单侧 NaN/Inf 无法按正式 palm-local 策略清除该侧、继续另一侧 | 已修复：手部由声明和所选 processor 验证；上身仍 strict |
| `apps/dual.py` / `_raw_observation` | 原字典边界接受一般三维数组，可能把模型窗口当作 raw；边界拒绝也没有清空旧历史 | 已修复：仅 transforms25 可为三维 raw，结构拒绝清空历史 |
| `apps/dual_realtime.py` / `main` | 加载 checkpoint 没有传 expected alignment，默认 legacy | 已修复：模式决定 expected alignment，并显式传入 loader 与 DualTeleoperator |
| `retargeting/hand/predictor.py` / `TwoHandRetargeter.predict` | checkpoint 加载时虽严格，但推理没有把 HandWindow alignment 与已加载模型比较 | 已修复：模型 forward 前精确比较；缺失、未知和 mismatch 均失败 |
| `retargeting/hand/checkpoint.py` / `load_hand_checkpoint` | 原 loader 的精确比较本身正确，但验证结果未保留给后续推理 | 已修复：成功严格加载后附加运行期字符串属性，不进入 state_dict |
| `tests/test_mediapipe_realtime.py` / 两个 camera workflow 测试 | mock 仍设置旧 next_frame/release；迁移后使用 next_observation/close | 已修复测试接口，保留时间戳、重置及释放断言 |
| `tests/test_hand_core_regression.py` / identity recovery 测试 | 错把 fixture 当作 TemporalBuffer 访问私有窗口方法 | 已修复为访问 fixture.buffer；保留恢复断言 |
| `tests/test_native_hand_replay.py` / invalid controls 测试 | 引用未定义 argv，11 个 subtests 未实际走到 CLI | 已修复为 sys.argv；继续断言设备连接前退出 |

三个测试文件的问题均在本轮改动前复现，属于重构后的测试接口迁移遗漏，没有改动被测数值算法或容差。

## C. 坐标契约与调用链

修复前：

```text
hand: raw MediaPipe21 → palm-local processor → buffer → HandWindow → palm checkpoint
dual: raw combined observation → canonical processor → buffer → HandWindow
      checkpoint loader 未传 expected alignment，默认 legacy
```

修复后：

```text
显式 hand_preprocessing
  ├─ palm-local + source=mediapipe_approx + mediapipe_normalized
  │    → 共享 MediaPipePalmLocalPipeline
  │    → 原 MediaPipePalmLocalProcessor → 原 TemporalBuffer → HandWindow
  ├─ palm-local + 已声明 l21 / palm_local_to_l21_v1 的 25 点
  │    → 验证并直接缓冲，不再次对齐
  └─ legacy + 已声明 l21 / source_to_l21_xyz 的 landmarks
       或 source=visionpro 的 transforms25
       → 原 CanonicalHandPipeline → HandWindow

预处理 alignment == loader 严格核验的 checkpoint alignment
                         == 每次 predict 的 HandWindow alignment
```

`--hand-preprocessing legacy|palm-local` 默认 legacy，保留旧 checkpoint 默认加载语义。没有根据文件名、shape 或 supported-values 集合猜测坐标模式。未知输入必须补充声明；仅存在 l21 而缺少 alignment 也拒绝。已对齐数据的 source_landmark_space 仅表示来源，不触发再次对齐。

Legacy 仍使用原 canonical、topology、wrist-relative、identity tracking 和三帧缓冲；没有隐式增加固定旋转。Vision Pro 仍调用原 transforms25 提取函数。普通未绑定 checkpoint 的 TwoHandRetargeter 构造兼容旧工具/测试；正式 dual 拒绝没有明确 alignment 的模型。离线 H5 的读取和训练/导出函数没有改变。

| 预处理 | checkpoint | 结果 |
|---|---|---|
| Legacy | Legacy | PASS |
| Palm-local | Palm-local | PASS |
| Legacy | Palm-local | 加载前契约检查失败 |
| Palm-local | Legacy | 加载前契约检查失败 |
| Unknown / 缺少声明 | 任意 | 边界或推理前失败 |

手部 timestamp 原样传递到 canonical frame 和 HandWindow，timestamp_unit 原样保留。hand 摄像头 workflow 的 relative seconds 与 raw_timestamp_ms 保留原行为；dual command timestamp 仍来自上身观测，不把 Unix 毫秒擅自转换为秒。

## D. 修改清单与数值边界

| 修改文件 | 理由 |
|---|---|
| `src/teleoperation/apps/hand_processing.py` | 共享既有 palm-local pipeline，增加 dual 输入契约选择与已对齐输入分支 |
| `src/teleoperation/apps/hand_realtime.py` | 改为导入共享 pipeline，保留原可导入名称与 camera workflow |
| `src/teleoperation/apps/dual.py` | 显式预处理与模型检查，严格 raw 边界及拒绝时重置 |
| `src/teleoperation/apps/dual_realtime.py` | 显式传递 expected alignment 与模式 |
| `src/teleoperation/cli/schemas.py` | 仅增加 dual realtime 的一个模式选项 |
| `src/teleoperation/retargeting/hand/checkpoint.py` | 记录已经严格验证的运行期 alignment |
| `src/teleoperation/retargeting/hand/predictor.py` | 在 predict 前比较 HandWindow 与模型 alignment |
| `tests/test_dual_hand_coordinates.py` | 新增 15 项坐标契约、等价、失效恢复、入口组装和清理测试 |
| `tests/test_mediapipe_realtime.py` | 修复旧 camera mock，使既有断言实际执行 |
| `tests/test_hand_core_regression.py` | 修正测试夹具中的 buffer 访问 |
| `tests/test_native_hand_replay.py` | 修正未定义 argv |
| `README.md` | 补充一段模式选择说明和验收报告链接 |
| `docs/contracts.md` | 记录实际输入声明、alignment 与时间戳契约 |
| `docs/architecture.md` | 更新共享 pipeline 与 dual 模式说明，区分原目录迁移验收 |
| 本报告 | 记录结果、证据和环境限制 |

PoseTransformer、Hand FK、palm-local/21→25 数学、tracking、arm IK/calibration/safety、18→17 转换、机器人原生映射、URDF、loss 与优化器实现均未修改。Dual.update 的手臂求解与命令拼接仍使用原计算。18D、17DOF、三帧、左右独立 validity 和 `[LA7|LH17|RA7|RH17]` 48D 顺序保持。

## E. 数值和运行证据

新增 synthetic 测试使用确定性双手 raw21 序列，涵盖连续单/双手、单侧缺失、恢复、NaN/Inf、退化 palm basis 和重复 handedness detection。

- hand 与 dual 对 canonical frame `(25,3)`、HandWindow `(3,25,3)`、float32、数值、alignment、timestamp/unit、有效侧和三帧顺序逐帧比较，全部精确一致。
- 使用临时真实 PoseTransformer checkpoint（固定随机种子初始化，严格重载），CPU 比较 61 个单侧 18D 输出，最大角度差 0。该测试证明处理路径等价，不证明正式训练模型的效果。
- 任一侧失效立即清空该侧窗口，另一侧持续输出；恢复第 1、2 帧仍无输出，第 3 帧只包含新帧。结构或坐标声明拒绝后也无法使用旧窗口。
- 已对齐 palm 输入在禁用 alignment 函数的 guard 下直接缓冲，值不改变；legacy landmarks 与 Vision Pro transforms25 与原 CanonicalHandPipeline 相同。
- 有效 DualTeleoperator.update synthetic 测试保留真实 hand model、arm calibration/retargeter/safety 和 48D 拼接，仅替代 URDF/IK 构造依赖；这不算真实双臂 IK 验收。

历史参考是目录重构前 `c57c165518d0b51dc66dd23fa5eac1469b80956b` 的原代码，通过 git show 提取到输出目录独立进程执行。仅将历史 L21 配置的资源路径指向迁移后相同 URDF/mesh；Git 确认左右 URDF 是 0/0 变更的路径迁移。

20 帧固定双手数据，legacy 使用原固定轴对齐函数，palm 使用历史/当前各自的 align_h5。两种模式分别验证：raw → alignment → aligned H5 → Dataset → 已有 PR1 smoke checkpoint inference → angles H5；然后执行完整真实模型与 FK/loss 的 1 epoch CPU 训练（batch=8、val_ratio=0.3、固定 seed），保存新 checkpoint、严格 reload 并导出。

修改前和修改后均与同一历史代码精确比较：aligned H5、Dataset input/target/mask、已有 checkpoint 导出角度、新训练 state_dict 的所有参数名称和值、best_val、重载后导出角度全部相同，最大差值 0。临时输出均位于 `outputs/current_state_validation`，未覆盖正式 checkpoint。

三机器人 smoke 使用真实公开 CLI `replay actions --dummy --steps 120 --robot ROBOT --no-render`，审计实际 p.stepSimulation 调用：

| robot | 维度 | 实际物理步数 | joint name/index 检查 | 状态有限与释放 |
|---|---:|---:|---|---|
| h1_2 | 38 | 120 | 名称唯一；逐项匹配 Bullet jointInfo | PASS |
| gr1_t2 | 36 | 120 | 同上 | PASS |
| g1 | 28 | 120 | 同上 | PASS |

每步检查动作、关节位置/速度/反力/力矩、底座位置/四元数/线速度/角速度均有限。每个 CLI 的 probe 与执行环境各 reset/close 一次，断言连接关闭。任务是否完成不作为此 smoke 通过条件。

现有 benchmark 入口测试也通过：实际子进程选择 level-1 tasks，每任务真实执行 5 个物理步；这仅为有界基本执行，不是完整性能 benchmark。

日志和可重跑的本地审计脚本：

- `outputs/current_state_validation/unittest_before_unrestricted.log` / `unittest_final.log`
- `outputs/current_state_validation/architecture_final.log` / `dual_coordinates.log`
- `outputs/current_state_validation/audit_runtime.py` / `runtime_final.log` / `smoke_*.json`
- `outputs/current_state_validation/audit_data_history.py`
- `outputs/current_state_validation/data_comparison_before.json` / `data_comparison_after.json`
- `outputs/current_state_validation/data_history.log` / `data_before.log` / `data_after.log`

这些输出被仓库既有 ignore 规则忽略，仅留本地供检查。没有完整冻结的历史摄像头录制或 TRON2A 实际 IK/物理输出可比较；这些部分仍是“缺少可验证基线”或缺资源，不能由本轮新输出自证正确。

## F. 后续任务

当前保存的 `origin/main` 为 `a8092615fee256b9a5beef4c6c7c4aa2d149f584`，相对 current-state 是 ahead 12 / behind 13。**本地 main 实际为 `2927f2d62aaa62d6629e749893893581f2ffc153`，是 current-state 的祖先**；本轮未 fetch，不声称远端实时状态。

值得单独审查迁回的新架构修改：Apache-2.0 LICENSE；GBK-safe 文本清理；原生手 AABB 对角线距离与自适应 yaw（当前已有手部中心取景和显示颜色，仍固定 yaw=135）；三机器人演示/快照工具；环境配置、缓存路径、preflight 与运动量化工具。仅列清单，未迁移旧目录、未 merge/rebase。

下一步先补齐正式 checkpoint 和 TRON2A URDF/mesh，重跑保留的 ERROR/SKIP，并完成真实 FK/IK 与模型验收。之后单独评估 `DualTeleoperator → Tron2AL21Command[48] → Tron2AL21ActionAdapter → TRON2A + L21 PyBullet`。本轮未实现该闭环、硬件控制或新的机器人双臂 IK。
