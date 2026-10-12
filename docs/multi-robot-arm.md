# 多机器人手臂驱动：按 robot_type 选臂并驱动机器人关节

最后更新：2026-10-12（新增 §8 实时链路）| 分支：cs/multi-robot-arm（基线 10ebd9f，队友线 HEAD d8bca93）
相关代码：`src/teleoperation/simulation/arm_ik.py`、`arm_profiles.py`、`arm_scene_driver.py`、
`arm_realtime.py`、`realtime_teleop.py`
相关测试：`tests/test_arm_ik_multi_robot.py`（16 个用例）、`tests/test_arm_realtime_bridge.py`（30 个用例）；
离线验收 `devtools/verify_arm_multi_robot.py`、`devtools/verify_realtime_arm_bridge.py`

## 1. 为什么需要这一层

仿真侧原来只会驱动 **TRON2A** 一只手臂：手臂关节顺序、URDF、标定都写死在
`retargeting/arm/ik.py` 与 `robots/specification.py` 里（共 4 处）。论文三台机器人
（H1-2 / GR1-T2 / G1）的 URDF 在 `assets/robots/from_teleopbench/` 里齐全，但仿真入口
`applications/three_robot_scene.py` 每帧把**除手指以外**的所有关节（含 7+7 手臂关节）
锁在加载角度上，所以三台机器人的手臂一直是**静止**的。

这两件事都不能靠改队友文件解决（CONVENTIONS 约定 2/3）：`three_robot_scene.py`、
`retargeting/arm/{ik,calibration}.py`、`robots/specification.py`、`contracts/arm.py`、
`simulation/robot_loader.py` 都是队友基线内的文件，改动会在
`devtools/preflight.ps1` 里被记成「修改了队友文件」。

因此本层全部是**新增文件**，选择机器人的动作放在「求解层 / 场景层」，**不动任何既有接口**：

```
ArmProfile         simulation/arm_profiles.py    按 robot_type 的每机器人参数（新增）
  -> PyBulletArmIK simulation/arm_ik.py         每台已加载机器人的 IK 适配器（新增）
  -> ArmSceneDriver simulation/arm_scene_driver.py  摘掉场景对手臂的 hold，写电机（新增）
```

## 2. 接口不变（为什么能共存）

- 动作向量约定不变：仍是 `[左臂 7] + [右臂 7] + [左手 N] + [右手 N]`，本层只按**关节名**
  取 `loader.arm_joints[side]`，不新增也不减少维度（48 / 38 / 36 / 28 与手 17 / 12 / 11 / 7
  都不变）。
- 任务空间契约不变：`ArmTargetPose.robot` 已经是字段，本层读 `robot_type` 决定用哪台手臂。
- 手指链路不变：`scene.apply(side, angles_18)` 一行未动；手指关节本来就不在 hold 名单里。

## 3. 用法

```python
from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

scene = ThreeRobotHandScene(sides=("right",), gui=False)   # 队友的类，原样
driver = ArmSceneDriver(scene, sides=("right",))           # 摘掉手臂 hold + 每机器人 IK

driver.apply("right", (0.22, 0.0, 0.18), source="base")    # 机器人自身基座坐标（米）
driver.apply_operator("right", (0.05, 0.0, 0.02))          # 操作者坐标：按 robots profile 缩放+偏移
driver.apply_each("right", {"h1_2": (...), "g1": (...)})   # 每台机器人各自的目标

driver.step()                                              # = scene.step()（锁其余关节 + 8 步物理）
driver.residuals_mm()      # {robot: {side: mm}}  目标与手腕实际位置之差
driver.hold_drift()        # 未被驱动的关节（腿/腰/头）最大漂移，手指单列 hand_drift_rad
driver.summary()           # 报告负载（JSON 可序列化）
scene.close()
```

命令行演示（无 GUI，可 `--report` 落 JSON）：

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe -m teleoperation.simulation.arm_scene_driver `
    --headless --frames 30 --settle 15 --report outputs\tmp_arm_demo.json
E:\python3.11.7\python.exe -m teleoperation.simulation.arm_scene_driver --list-profiles
```

## 4. 每机器人参数与实测

`ARM_PROFILES` 是唯一的「哪台机器人用哪条手臂、怎么映射」的地方：

| robot_type | ik_mode | metric_scale | metric_offset (基座系, 米) | ee link | 阈值 |
|---|---|---|---|---|---|
| h1_2 | pose | 0.7 | (0.24, 0.0, 0.19) | right_wrist_yaw_link | 30 mm |
| gr1_t2 | position | 0.4 | (0.16, 0.0, 0.12) | right_hand_pitch_link | 25 mm |
| g1 | position | 0.4 | (0.16, 0.0, 0.10) | right_wrist_yaw_link | 25 mm |

实测（`devtools/verify_arm_multi_robot.py`，2026-10-12；`three_robot_scene` + 本层，重力
-9.81、hold 生效、按各自 profile 幅度整圈扫掠 40 帧）：

| robot_type | ik_mode | 稳态残差 mean / max | 手腕行程（目标） | 非驱动关节漂移 |
|---|---|---|---|---|
| h1_2 | pose | 17.0 / 23.8 mm | 99.5 (100.0) mm | 0.0003 rad |
| gr1_t2 | position | 8.9 / 13.3 mm | 60.5 (60.0) mm | 0.0013 rad |
| g1 | position | 7.7 / 14.9 mm | 59.6 (60.0) mm | 0.0017 rad |

补充事实（同样实测）：

- 手臂电机力矩 200 / 500 / 1000 结果完全一致，所以默认沿用 200（与
  `devtools/verify_arm_from_capture.py` 相同）；稳态残差不是力矩问题。
- 冷启动前几帧是 32-55 mm，手臂还在往里走；阈值只描述稳态。
- PyBullet 的 `calculateInverseKinematics` **从当前关节角起步**，单次冷解会卡在局部极小
  （同一目标：h1_2 56 mm、g1 59 mm、gr1_t2 46 mm），因此 `solve()` 采用**两遍热启动**：
  先解位置，把手臂 snap 到该解，再解一次（pose 模式用调用方给的腕姿态，没给就用第一遍
  自然落到的姿态）。同一目标立刻降到 18 / 43 / 7 mm，扫掠稳态如表。
- gr1_t2 没有 wrist 链（手装在 `right_hand_pitch_link` 上），g1 的手 link 在 wrist 之后；
  两台的姿态约束都跟不上未标定的姿态约定，所以 `ik_mode="position"`。
- 目标必须给**机器人自身基座系**坐标（`source="base"`）：少这一步会低 866 mm（h1_2 的
  base_z 1.00 m 加抬离地面量）；`source="operator"` 会先按 profile 缩放 + 偏移再换到世界系。

## 5. 本层不改的东西

`retargeting/arm/{ik,retargeter,calibration,safety}.py`、`robots/specification.py`、
`contracts/arm.py`、`simulation/robot_loader.py`、`applications/three_robot_scene.py`、
`applications/realtime_hand_sim.py` 一律未改；`TRON2A` 仍登记在 profile 表里但标
`loadable=False`（本仓库没有它的 URDF / 标定 JSON / 手部检查点），调用 `profile_for("tron2a")`
会明确报错而不是偷偷用论文机器人的约定。

## 6. 交给队友的同步事项（本层刻意没做，避免改基线文件）

1. `retargeting/arm/ik.py:30-42,79`：TRON2A 关节顺序 / URDF 路径 / 标定可参数化，接受
   `robot_type` 或 profile；`retargeting/arm/calibration.py:149-175`、`robots/specification.py:4-14`
   同理。契约不变（仍返回按关节名的目标角）。
2. `applications/three_robot_scene.py:287-291`：把 14 个手臂关节从 `hold_ids` 里去掉，
   两行即可（等价于 `ArmSceneDriver._release_holds` 做的事）。本层用包装类实现，所以场景
   文件保持零改动；等队友同意后再合进他们那侧更干净。
3. `retargeting/arm/wrist_targets.py`：如果要真用 pose 模式，需要一份手眼标定把操作者腕
   姿态映射到机器人基座系；目前只有约定，h1_2 之外的两台因此退化成 position。
4. 文档同步：按 PROJECT_CONTEXT 11.1 把本文件挂进文档索引（本层不改队友的
   `docs/PROJECT_CONTEXT.md` / `docs/CONVENTIONS.md`，避免出现「修改队友文件」）。

## 7. 限制与下一步

- 没有接摄像头：本层只吃「手腕目标位姿」（任务空间），21 点 -> 手腕目标仍由
  `retargeting/arm/wrist_targets.py` 提供；`ArmSceneDriver.apply_operator` 就是给那条链路留的入口。
- profile 里的 scale / offset 是**调参**不是标定：换机器人、换摆位或换操作者都需要重测。
- 三台机器人共用一套「基座系目标」时，幅度取各 profile 的最小值（CLI 默认行为）；
  要每台不同的幅度就用 `apply_each`。
- 未做：双臂同时驱动（`sides=("left","right")` 已支持，但没做双臂协调测试）、
  与人手 21 点的端到端联调、把 `ArmSceneDriver` 接进 `realtime_hand_sim` 的 `--scene` 开关。

## 8. 实时链路（摄像头 -> 手 18 维 + 手臂相对重定向）

本节讲把本层接进**实时摄像头入口**的那一层，同样全是新增文件（队友文件只读）。

基线入口 `applications/realtime_hand_sim.py` 只驱动手指：手臂被 `three_robot_scene.py` 每帧
按在加载角度上（`hold_ids` + `HOLD_FORCE=500`），而且它用的
`MediaPipeCameraWorkflow.next_window()` 只回手部窗口，**同一帧 pose 的肩/肘/腕在那层就被丢了**。

```
MediaPipeCameraInput.next_teleop_observation()   # 同一帧：手 21 点 + pose 肩/肘/腕
  |- 手：MediaPipePalmLocalPipeline（三帧窗口 + 掌面局部对齐）
  |      -> GeometricHandRetargeter / 训练检查点 -> 18 维 -> scene.apply(side)（基线那套，一行未改）
  `- 臂：RealtimeArmBridge（相对重定向：以中性姿势为原点，每侧独立）
         -> ArmSceneDriver.apply_each(source="base") -> PyBulletArmIK 两遍热启动 -> 每台 7 关节
```

| 文件 | 干什么 |
|---|---|
| `src/teleoperation/simulation/arm_realtime.py` | `AxisMap`、`RealtimeArmBridge`（中性标定 / 相对映射 / 每侧行程夹取 / 丢帧保持 / 21 点兜底）、`load_neutral` / `save_neutral` |
| `src/teleoperation/simulation/realtime_teleop.py` | 实时入口 CLI：`build_parser` / `run` / `main`、`CameraFrameSource`、`ReplayFrameSource`（`--input-npz` 回放）、手臂 HUD、`--report`、`--dump-npz` |
| `devtools/verify_realtime_arm_bridge.py` | 离线验收（无相机、无人可跑），含 `--probe-travel` / `--probe-lateral` 两个"只量参数、不改文件"的模式 |
| `devtools/run_realtime_arm_windows.py` | PyCharm 一键入口（`--check` 只体检） |
| `tests/test_arm_realtime_bridge.py` | 30 个用例（轴向 / 几何 / 标定 / 相对映射 / 夹取 / 保持兜底 / 报告 IO / 入口默认值 / 回放元数据） |
| `arm_profiles.py`（改） | `realtime_travel_m` / `realtime_offset` + `realtime_neutral(side)` / `travel_limit(side)` |
| `arm_scene_driver.py`（改） | `__init__` 透传 `ik_iterations` / `residual_threshold` |

`CameraFrameSource` 是**复刻**基线那 20 行胶水（只 import 队友公开接口），因为 `next_window()`
会丢掉同帧 pose；队友文件本身零改动。

### 8.1 怎么跑

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"

# 真机：体检 -> 跑（先站定中性姿势约 30 帧）
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --check
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-trace     # 现场核轴向
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop `
    --report outputs\tmp_realtime\run.json --save-neutral outputs\tmp_realtime\neutral.json

# 复用上次的中性姿势 / 回基线（只驱动手）
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --neutral-json outputs\tmp_realtime\neutral.json
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --no-arms

# 无相机复验：录一段 -> 回放
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --headless --frames 300 `
    --dump-npz outputs\tmp_realtime\frames.npz
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --headless `
    --input-npz outputs\tmp_realtime\frames.npz --frames 40 --no-preview --report outputs\tmp_realtime\replay.json
```

`--dump-npz` 的键名：`timestamps` / `{left,right}_hand`（归一化 21x3）/ `world_{left,right}_hand`
（米制 21x3）/ `{left,right}_arm`（pose 肩/肘/腕 3x3）；NaN = 那一帧没检到（回放时触发"保持上一帧"）。

### 8.2 参数（默认值与实测）

轴向：操作者系（pose world）x 向右 / y 向下 / z 远离相机为正；机器人基座系 X 前 / Y 左 / Z 上。
默认 `AxisMap` = 顺序 `(Z, X, Y)` / 符号 `(-1, +1, -1)`。**第一次上真机必须先加 `--arm-trace` 核**：
只往一个方向动手时应当只有一个 `delta_op_cm` 分量明显变化、符号与画面一致；对不上就用
`--arm-axis-order` / `--arm-axis-signs` 覆盖。没核过之前，"跟手"只是相对自洽，不能叫标定。

| robot_type | ik_mode | 中性工作点（基座系，米） | 每侧行程上限（米） | 残差阈值 |
|---|---|---|---|---|
| h1_2 | pose | (0.24, 0, 0.19) | left 0.10 / right 0.15 | 30 mm |
| gr1_t2 | position | (0.16, 0, 0.12) | left 0.10 / right 0.12 | 25 mm |
| g1 | position | left (0.16, +0.10, 0.10) / right (0.16, 0, 0.10) | 0.04 / 0.04 | 25 mm |

- 行程上限由 `--probe-travel` 实测（**两侧同时驱动**才有意义：只驱动一侧时另一侧解冻下垂、会把整机带歪）；超限按每轴夹取，夹取帧数写进 `--report`。
- g1 左臂在 Y=0 时残差均值 31.2 mm，Y=+0.10 降到 19.8 mm（`--probe-lateral` 复核）；h1_2 / gr1_t2 两侧都是 Y=0 最好 —— 左右不对称是实测结论，不是调参洁癖。
- 入口默认：`--side both`、`--landmark-space world`、`--arm-orientation auto`、`--arm-ik-iterations 60`、`--arm-ik-residual-threshold 1e-4`、`--arm-calibrate-frames 30`、`--calibration-frames 15`、`--sim-steps 8`、`--print-every 15`、`--frames 0`。
- **60 / 1e-4 是特意选的**：实测比 `PyBulletArmIK` 自带的 200 / 1e-5 又快又准（h1_2 稳态残差 10.3 / 11.7 vs 18.4 / 20.4 mm）；三台同时跑约 11.2 fps、单侧约 17.7 fps（本机 CPU）。
- 两个"标定帧"不同义：`--calibration-frames`（开手标定，基线口径）vs `--arm-calibrate-frames`（手臂中性帧），混用会互相污染。
- 常量：`REFERENCE_REACH_M=0.60`、`ARM_LENGTH_BAND=(0.5, 2.0)`、`MIN_OPERATOR_REACH_M=0.15`、`DEFAULT_MAX_TRAVEL_M=0.10`、`FALLBACK_IMAGE_SCALE=0.7`；gain 按标定那几帧的臂长 / 0.60 归一化。
- pose 丢帧（None / NaN）-> 保持上一帧目标；默认允许"21 点 -> 手腕"兜底（乘 0.7），`--no-arm-fallback` 可关。
- 退出码：0 = 至少一侧有输出，1 = 全程没数据，3 = 解释器 / 依赖 / 模型不对；`__main__` 用 `os._exit` 绕开 MediaPipe `close()` 约 42 s 的阻塞（别"顺手改成正常 return"）。

### 8.3 验收证据（2026-10-12，本机实测，命令 + 退出码 + 不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `E:\python3.11.7\python.exe devtools\verify_realtime_arm_bridge.py` | 0 | 中性标定 60 帧完成（有效帧 left 45 / right 46）；标定后第一帧目标 = 中性工作点（最差轴 0.0000 mm）；稳态残差 mean 9.5~21.4 mm（阈值 25 / 30 mm，判据只看 mean）；丢帧保持 81 次全中；未驱动关节漂移 <= 0.0005 rad |
| 同上加 `--arm-ik-iterations 60 --arm-ik-residual-threshold 1e-4` | 0 | 同上，稳态残差 mean 5.2~11.7 mm（h1_2 10.3 / 11.7 mm）—— 这就是入口默认口径，数字与 profile 阈值一致 |
| `E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --headless --input-npz outputs\tmp_realtime\replay_frames.npz --frames 40 --no-preview --report ...json` | 0 | 40 帧回放完、不读相机；手 24 / 18 帧有效（其余是 3 帧滑窗预热与刻意丢帧）；手臂每侧 40 / 40 帧有目标；三台残差 h1_2 11.3 / 20.6、gr1_t2 3.7 / 5.1、g1 24.8 / 14.8 mm；hold drift <= 0.0005 rad |
| `E:\python3.11.7\python.exe -m pytest tests -q` | 0 | 259 passed, 3 skipped（含本层 30 条） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | 0 | `RESULT: [OK]`（8 项全过；unittest discover `Ran 262 tests`、skipped=3；4 条 CLI 冒烟 exit 0） |

> **`-Base` 必须显式给 `d8bca93`**：`preflight.ps1` 的默认基线是 `10ebd9f`，而本支在那之后已合进队友
> 45 个提交，于是第 4 项会把**队友自己的改动**算成"改了队友文件"（本轮实测 49 个 `[FAIL]`，全在队友线）。
> 相对本层起点 `d8bca93`，本支只有 6 个"新增文件"、第 4 项 `[OK]`。
>
> 离线验收脚本的 IK 参数默认是"**不下发**"，即跟随 `PyBulletArmIK` 的 200 / 1e-5；要核入口默认
> （60 / 1e-4）就显式带 `--arm-ik-iterations 60 --arm-ik-residual-threshold 1e-4`。两条都跑通，数字差异见上表。

### 8.4 实时链路特有的坑

1. **报告段不能在 `finally` 里读 PyBullet**：断连后再 `getLinkState` 会抛 "Not connected to physics server"，必须在断连前取值（本入口已按此顺序写）。
2. `np.load` 的句柄会锁文件（Windows 上连临时目录都删不掉）：`ReplayFrameSource` 一次性读进内存。
3. **`--input-npz` 回放必须给每一帧带 world 元数据声明**（`source_landmark_space` / `world_landmark_space=mediapipe_world_meters` / `world_length_unit=m` / `source_scale`）：队友的 `MediaPipePalmLocalProcessor` 在 world 口径下会校验这几项，缺一项整场第一帧就 `ValueError`，退出码 1。`--dump-npz` 只存数组，所以由 `ReplayFrameSource` 自己补声明 —— 回归用例 `test_world_replay_declares_the_metadata_the_pipeline_demands`。
4. `reset()` 要连帧计数一起清，否则 `frames` 把上一轮累计带进新一轮。
5. 只在**入口**里 `print` 数字，别在桥里 print：桥是纯计算，报告才是证据（约定 5）。

### 8.5 没做的（交接）

- **真机未做**：还没站在摄像头前跑过一次、`--arm-trace` 的轴向（§8.2 第一条）**没核**，所以不要说"已标定 / 已跟手"。
- 手臂目标是**相对重定向**（以中性姿势为原点），不是手眼标定；换操作者、换站位、换相机摆放都要重做中性标定。
- 未做：双臂协调动作测试、与 `retargeting/arm/wrist_targets.py` 那条 21 点 -> 手腕链路的联合调试（`--arm-orientation auto` 目前只有 h1_2 带姿态约束）。
