# PR1.5 — Application Entry Stabilization

三项入口 bug 已修复，native-hand replay 改进已独立提交，完整回归通过。
**Ready for PR2: NO**：原始默认记录模式的长程 main smoke 未通过，需要独立确认
记录资源使用与 native crash。添加 `--no-record` 后完整 demo/task 入口正常退出。
本轮没有开始 RobotCommand PR2，也没有修改手部模型质量。

## 1. 正式环境、基线与原始修改

| 项目 | 实测结果 |
|---|---|
| 环境 | `conda activate teleoperation`；本轮正式测试未使用 `.venv` 或 TransHandR |
| Python executable | `D:\Anaconda\envs\teleoperation\python.exe` |
| Python version | 3.10.20 |
| Torch / CUDA | 2.14.1+cu130 / 13.0，CUDA available=True，RTX 4050 Laptop，实际 CUDA tensor 通过 |
| pip check | `No broken requirements found.`；没有安装、卸载或下载依赖 |
| 初始 branch | `refactor/consolidate-hand-retargeting` |
| 初始 baseline SHA | `4ab8386c04dc3d728a1cc3798083ce35c30ac224`，包含 PR1、ENV-1、CUDA 声明 |
| replay commit / PR1.5 branch baseline | `03254478e58c4edfad26672b6a47a6a5456a453c` |
| PR1.5 branch | `fix/application-entrypoints`；PR1 尚未合并到本地 main，从正确 HEAD 建分支，未 merge/pull |

开始时没有 staged 修改，8 个 tracked 文件修改、2 个 untracked 文档：

```text
 M README.md
 M docs/HAND_RETARGETING.md
 M docs/INTERFACE_CONTRACT.md
 M docs/OFFLINE_PIPELINE.md
 M docs/PR2_FOLLOW_UP.md
 M docs/PROJECT_CONTEXT.md
 M docs/TEAM_ONBOARDING.md
 M scripts/replay_hand_native.py
?? docs/README.md
?? docs/SYSTEM_ARCHITECTURE.md
```

初始 tracked diff：8 files，151 insertions、53 deletions。初始文件与 diff 已保存到
本机 ignored `outputs/pr15_checks/before/`、`before_diff.patch`、`snapshot.json`。
这些此前已授权的架构/文档入口修改均保留并纳入文档提交，没有丢弃用户修改。

原始 replay diff：**+68 / -13（81 行）**，完整 patch 在本机 ignored
`outputs/pr15_checks/replay_original_diff.patch`。包括左右手颜色、真实 finger AABB
特写、view、start/end frame、速度节奏、原始帧 HUD 和循环。审查后仅补一处 clip
边界修正：完整记录执行原有身份检测/修复后，再截取播放区间，避免裁剪影响检测上下文。
检测、修复函数本身及 validity mask 均未修改；`--no-repair` 直接使用原始输入值。
原始 H5、模型预测、坐标模式、checkpoint、native mapping/sign/limits 没有变化。

## 2. 三项入口修复与生命周期

**action_dim 根因**：创建环境后 `action_dim=None`，旧 main 在 reset 前将其缓存到
全局，`np.zeros(None)` 产生标量，随后索引导致 IndexError。
现移除全局维度缓存；`partial(demo_controller, env=env)` 只绑定环境，不提前读取维度：

```text
SimulationEnv(...)
  → 绑定 controller（不 reset、不生成 action）
  → run_episode() 自行 reset 一次
  → load robot，建立 action_joint_names / action_joint_indices / action_dim
  → controller callback 读取当前 env.action_dim，产生有限 ndarray
  → step(action)
```

reset 前显式调用控制器会给出明确 RuntimeError。生产维度来自环境，没有硬编码
38/36/28；保留已有正弦动作及 benchmark 顺序/转换。真实 H1/GR1/G1 回归验证 shape、
finite、joint names/indices、一遍 episode 一次 reset 和 recorder.reset，避免 double reset。

**demo 根因**：单独 `--demo` 没有 task，因此既不进入 task 也不进入 benchmark 分支。
现复用既有默认 pushcube 和 `evaluate_task(num_trials=3)`。`--demo --task NAME`
选择指定任务演示；task 单独使用运行一次；benchmark 运行既有 Level 1 与 trials；
benchmark 与 demo/task 明确互斥，冲突在环境创建前报 argparse 错误。交互 demo/quit 保留。

真实多 episode 回归另发现：RobotLoader 修改 PyBullet 搜索路径，第二次 reset 查找
相对 `plane.urdf` 失败。完成三次 demo 必须修复此阻断，因此仅在
`SimulationEnv._load_ground()` 使用同一 pybullet_data 地面的绝对路径。
未改 reset 次数、随机化、recorder、任务判定、动作 converter 或仿真步进。

**inspect 根因**：wrapper 导入不存在的 `retargeting.inspect.main`。
现仅转发 `retargeting.cli.main(["inspect", *sys.argv[1:]])`，没有复制 inspect 实现。
help、合法 H5 与 canonical 输出一致；缺文件/invalid schema 保留异常及非零退出码。

## 3. 修改文件、提交与测试

| 修改文件 | 作用 |
|---|---|
| `scripts/replay_hand_native.py` | 已有 UI 改进和保持完整检测/修复上下文的 clip 修正 |
| `main.py` | action_dim 生命周期、demo/task/benchmark 路由和帮助 |
| `envs/simulation_env.py` | 重复 reset 使用同一地面资产绝对路径 |
| `inspect_angle_h5.py` | canonical CLI forwarding |
| `tests/test_native_hand_replay.py` | 4 项真实机器人数值/控制回归 |
| `tests/test_application_entrypoints.py` | 7 项实际应用生命周期、CLI 路由/step/reset、默认记录短程测试 |
| `tests/test_inspect_wrapper.py` | 4 项 wrapper/canonical 一致性回归 |
| `README.md`、`docs/README.md`、`docs/SYSTEM_ARCHITECTURE.md` | 当前入口、完整架构、索引、resolved bugs 与验收边界 |
| `docs/HAND_RETARGETING.md` | replay 用法、clip 上下文、正式环境示例 |
| `docs/INTERFACE_CONTRACT.md`、`docs/OFFLINE_PIPELINE.md`、`docs/PR2_FOLLOW_UP.md`、`docs/PROJECT_CONTEXT.md`、`docs/TEAM_ONBOARDING.md` | 保留此前 pending 的文档索引/范围说明，没有改契约条款 |
| 本报告 | PR1.5 事实和未通过项 |

本地提交（未 push）：

| SHA | Commit |
|---|---|
| `0325447` | `feat(replay): improve native hand validation workflow` |
| `bdb68d0` | `fix(app): initialize actions after reset and execute demo episodes` |
| `069238f` | `fix(cli): restore angle h5 inspect compatibility wrapper` |
| `4ccbac8` | `test(app): add entrypoint regression coverage` |
| `608a8aa` | `test(app): cover bounded demo with default recording` |

另有最后的文档提交 `docs(app): record entrypoint validation and recording limits`；SHA 以最终
`git log` 为准，避免在提交内容中自引用。working tree clean、最终 diff/stat 与 SHA
在文档提交后再核验，保存在本机 ignored `outputs/pr15_checks/final_git.log`。

| 验证 | total / passed / skipped / failures / errors |
|---|---|
| 修改前 baseline full suite | 166 / 165 / 1 / 0 / 0 |
| replay targeted | 4 / 4 / 0 / 0 / 0 |
| application + inspect targeted | 10 / 10 / 0 / 0 / 0 |
| 后补默认记录短程 targeted | 1 / 1 / 0 / 0 / 0 |
| relevant hand application / CLI / CLI integration / simulation | 18 / 18 / 0 / 0 / 0 |
| 最终 full suite | **181 / 180 / 1 / 0 / 0** |

完整 suite 使用正式激活环境、普通 `python -B -m unittest discover -s tests -v`，
设置 `TRON2A_TEST_URDF=D:/2026/code/mytrans/third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf`。
唯一 skip 是项目本地历史 palm_local_v2 checkpoint 不存在的既有等价性测试；没有新增 skip。
日志均位于 ignored `outputs/pr15_checks/`。最初入口 targeted 揭示重复 reset 的地面路径
错误，修正后重新验证通过；最终 suite 无新增失败。

CLI 回归在子进程使用真实 main/environment/reset/randomization/controller/step，
只将既有 run_episode.max_steps 限为 5；审计 demo 三次 reset、15 次 step 与实际有限
ndarray，不允许 silent no-op。包含开启默认记录的短程模式。没有新增生产测试参数或演示框架。

replay sentinel 回归对实际 H1/GR1/G1 mapping 检查默认与六种 view、半速、裁剪、
循环/无限循环下同一原始帧的目标字典和 clipping 数完全相同，输入 H5 哈希不变；
invalid 帧跳过、motor target、相机特写、速度等待和错误参数均覆盖。
默认修复的孤立塌零帧在完整序列与单帧 clip 中目标一致，证明边界上下文未丢失。

## 4. 实际 smoke：区分通过与未通过

| 实际命令 | 结果 |
|---|---|
| `python main.py --demo --no-render` | **未通过**：开启默认记录，运行约 260.53 s 后退出码 3221225477（0xC0000005），没有正常结束；native crash 根因未确认 |
| `python main.py --task pushcube --robot h1_2 --no-render` | **未完成**：默认记录模式进程工作集约 11.3 GB；本轮主动停止，退出码 4294967295，约 280.73 s |
| `python main.py --demo --no-render --no-record` | PASS：约 28.28 s，三次真实 episode，退出码 0，无 traceback / `[FAIL]` |
| `python main.py --task pushcube --robot h1_2 --no-render --no-record` | PASS：约 15.61 s，退出码 0，无 traceback / `[FAIL]` |
| `python inspect_angle_h5.py --help` | PASS，退出码 0 |
| `python inspect_angle_h5.py --angle-h5 outputs/env_checks/palm_angles.h5` | PASS，退出码 0 |
| `python -m retargeting inspect --angle-h5 outputs/env_checks/palm_angles.h5` | PASS，退出码 0，输出与 wrapper 逐字节一致 |

inspect 的真实文件共 6515 帧，left valid=3566、right valid=3197，nonfinite=0、
out-of-limits=0，`coordinate_alignment=palm_local_to_l21_v1`。

默认 record 每一步缓存双相机 RGB/depth/seg，实测并发两个进程合计约 20 GB 内存。
记录路径的内存使用是直接观察；native crash 的具体原因不能仅凭该观察确定。
demo 在停止操作前已异常退出；仅仍存活的本轮 task 测试进程被主动停止，身份核对后终止。
没有关闭生产默认记录，没有更改 recorder，也没有把失败日志覆盖成通过。

两条首次长程 smoke 是本轮并发启动的，这会放大资源压力，不能将该结果当成单独命令
必定失败的证明。随后用正式环境单独探测五步默认记录：本机 PyBullet 返回 Python tuple，
双相机五步缓存对象合计 55,302,270 bytes，新增一步约 11,059,968 bytes。
按该短程增量外推 10000 步，仅相机缓存就约 **103.0 GiB**（是估算，不是实际跑满的测量）。
机器可见物理内存约 31.6 GiB，当时可用虚拟内存约 11.9 GiB，因此没有继续强行重跑
完整记录 smoke。探测成功日志及对象类型/大小见本机 ignored
`outputs/pr15_checks/recording_footprint.log/json`；其中 json 为测量明细。

no-record smoke 达到现有 step 上限可以正常退出，但 task 成功没有得到证明；
既有 max_steps 耗尽路径不会 finalize/记录 metrics，因此 demo summary 可为空。
没有将“退出正常”或“无 FAIL 字样”写成任务完成或成功率通过。

## 5. 未变化的边界与最终结论

| Diff 检查 | YES / NO |
|---|---|
| hand contract diff | **NO** |
| model / kinematics / losses diff | **NO** |
| training diff | **NO** |
| RobotCommand diff | **NO** |
| arm IK diff | **NO** |
| benchmark action converter diff | **NO** |
| checkpoint / 原始输入 artifact 改动 | **NO**，SHA-256 与 ENV-1 快照一致 |
| `docs/PR1_HAND_CONSOLIDATION_RESULT.md` 历史内容改动 | **NO** |

679 个原始 tracked 文件完成哈希核对，其中 603 个受保护文件不变，包括
model、retargeting、teleop mapping/filters、输入 adapters、config、机器人资产与依赖声明。
mytrans HEAD 仍为 `138fc2d`，working tree clean。
Hand25 / `[B,3,25,3]`、18D root placeholder + 17 L21 DOF、18→17 adapter 不变。
legacy `source_to_l21_xyz`、palm `palm_local_to_l21_v1` 的
H5 == Dataset == training == checkpoint == inference alignment 契约回归通过，cross-mode 继续失败。

`git diff --check` 通过；最终 committed diff/stat 与 `git status --short` 在
`outputs/pr15_checks/final_git.log`。此前 dirty 文档已保留并提交，未 reset/clean/discard，未 push。

仍存在的问题：

- 默认记录的长程 main smoke 未通过；短程实测相机 tuple 缓存每步约 11 MB，
  外推 10000 步约 103 GiB。须独立诊断资源使用/native crash，并串行复跑原始两个命令。
- hand engineering chain 可运行；**hand model/FK visual quality = NOT VALIDATED**。
  predicted FK 与人手输入仍有差异，thumb motion 可能不足；本轮不修改，后续独立做质量诊断。
- 实时整机/command/order/validity/per-robot conversion、真实 arm calibration/IK/mount、
  相机资产及所有任务/硬件验收仍待后续阶段。
- step 上限耗尽的 metrics/recording 缺口和 test_import 打印 FAIL 仍退出 0 等已有边界仍保留。

ENV-1 正常、三项入口 bug 修复、full suite 无新增失败、受保护核心未变化；
**Ready for PR2: NO**，因为默认记录真实 smoke 仍未达到本轮完整入口验收要求。
下一步先做独立 recording/native-crash 诊断，复验原始命令，再开始 PR2。
