# 多机器人手臂驱动：按 robot_type 选臂并驱动机器人关节

最后更新：2026-10-15（§8.18 「跑一会 GR1-T2 的大臂与小臂夹角反过来」的根因是**斜坡只削了腕那一点**：`--arm-map posture` 的命令点是刚性的一对（腕 = 肘 + L_f*v），肘目标不限速、腕每帧最多 40 mm，于是这一对被拉近（命令的前臂段 250.6 -> 51.1 mm）、IK 把前臂折到反方向；修法是标记 `elbow_rigid` 后在会压扁的帧上保持上一对，四个候选修法的 A/B 见 §8.18.4）| 前一轮 §8.17 「肘跟手」交付（`--arm-elbow-follow`）与现场固化档入口 §8.17.10 | 更早 §8.16 **「肘/小臂不跟手」的机制是 PyBullet `restPoses` 把解往加载姿势拽**：探针 A/B 把「精确可达目标」的残差从 6.6-140.1 mm 打到 0.0 mm / 拽动 35.0 度打到 0.00 度，证明主因是零空间代价项而不是链路帧口径；但同一条现场录制上换 `--arm-rest-poses live` **不是一边倒的赢**（肘任务只 3/6 侧变好、腕精度中位 17.2 -> 40.5 mm 变差），所以 CLI 默认仍留在 `current`、`live` 只作实验档）| 前一轮 §8.15「大臂抬不起来 / 张不开怀抱 / GR1-T2 前臂还是反的」的根因是那场跑在默认 `chain` 口径上（`--arm-map` 默认改 `posture`、`report_table.py` 加行程与夹断列）| 前一轮 §8.14 中性标定永远等不到数据（`--arm-calibrate-mode track`、`record_posture` 静默丢腕命令、`--help` 崩溃）| 分支：cs/multi-robot-arm（基线 10ebd9f，队友线 HEAD d8bca93）
相关代码：`src/teleoperation/simulation/arm_ik.py`、`arm_profiles.py`、`arm_scene_driver.py`、
`arm_realtime.py`、`realtime_teleop.py`、`arm_preview.py`
相关测试：`tests/test_arm_ik_multi_robot.py`（31 个用例）、`tests/test_arm_realtime_bridge.py`（84 个用例）、
`tests/test_arm_preview.py`（25 个用例）；
离线验收 `devtools/verify_arm_multi_robot.py`、`devtools/verify_realtime_arm_bridge.py`、
`devtools/verify_arm_wing.py`、`devtools/probe_elbow_task.py`、`devtools/probe_posture_task.py`、
`devtools/probe_arm_segment_rigidity.py`（2026-10-14 新增：锚点刚性与求解器自检；
同日给 `ik_frame_test` 加了 `restPoses` A/B = `identity_current_rest`，见 §8.16.2）

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
| `src/teleoperation/simulation/realtime_teleop.py` | 实时入口 CLI：`build_parser` / `run` / `main`、`CameraFrameSource`、`ReplayFrameSource`（`--input-npz` 回放）、手臂 HUD（`arm_hud` / `arm_lines`）、`--report`、`--dump-npz` |
| `src/teleoperation/simulation/arm_preview.py` | **采集窗口的手臂层**：`ArmCameraPreview`（子类化基线预览，基线文件零改动）、`finite_arm_points`、`outside_frame`、`build_arm_preview` |
| `devtools/verify_realtime_arm_bridge.py` | 离线验收（无相机、无人可跑），含 `--probe-travel` / `--probe-lateral` 两个"只量参数、不改文件"的模式 |
| `devtools/run_realtime_arm_windows.py` | PyCharm 一键入口（`--check` 只体检） |
| `devtools/run_realtime_arm_elbow_follow_windows.py` | **2026-10-15 现场版固化档**：上一层入口 + 钉住 `--arm-elbow-follow operator`（`--off` = 历史行为对照），零复制（见 §8.17.10） |
| `tests/test_arm_realtime_bridge.py` | 83 个用例（轴向 / 几何 / 标定 / 相对映射 / 夹取 / 保持兜底 / 报告 IO / 入口默认值 / 回放元数据 / 标定口径 hold-vs-track / 帮助文案可渲染） |
| `tests/test_arm_preview.py` | 21 个用例（合法点才画 / NaN 与形状不对当"没有" / `--no-arm-overlay` 逐像素等于基线 / 镜像翻面 / 画面外提示 / `build_arm_preview` 开关语义） |
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
| h1_2 | pose | left (0.24, +0.05, 0.19) / right (0.24, -0.05, 0.19) | left 0.10 / right 0.15 | 30 mm |
| gr1_t2 | position | left (0.16, +0.10, 0.12) / right (0.16, -0.10, 0.12) | left 0.10 / right 0.12 | 25 mm |
| g1 | position | left (0.16, +0.10, 0.10) / right (0.16, -0.10, 0.10) | 0.04 / 0.04 | 25 mm |

- 行程上限由 `--probe-travel` 实测（**两侧同时驱动**才有意义：只驱动一侧时另一侧解冻下垂、会把整机带歪）；超限默认**逐轴剪断**（`--arm-travel-mode clip`，2026-10-13 晚起，见 §8.9），等比缩是 `--arm-travel-mode scale`（10-13 白天到晚上的默认，会把没超限的轴一起压小）；两种口径与横向护栏的对照见 §8.8 / §8.9。夹取 / 护栏帧数都写进 `--report`。
- 中性工作点的**横向 y 必须分左右**（`+` = 机器人左侧，REP-103）：两侧都写 0 时两个腕子被命令到
  **同一个点**，松手就是\"前臂交叉在胸前\"——根因、量法与取舍见 §8.6。y 是\"看着对\"与\"IK 解得准\"的折中。
- g1 左臂在 Y=0 时残差均值 31.2 mm，Y=+0.10 降到 19.8 mm；右臂 Y=-0.10 只比 Y=0 差 1.3 mm
  （18.3 -> 19.6 mm），于是两侧对称在 +-0.10。h1_2 左臂最多 +0.05（+0.10 会到 30.05 mm，越 30 mm 上限）。
- 入口默认：`--side both`、`--landmark-space world`、`--arm-orientation off`（2026-10-13 起，见 §8.8）、`--arm-ik-iterations 60`、`--arm-ik-residual-threshold 1e-4`、`--arm-calibrate-frames 30`、`--calibration-frames 15`、`--sim-steps 8`、`--print-every 15`、`--frames 0`、`--arm-travel-mode clip`、`--arm-lateral-guard-mm 0`、`--arm-slew-mm-s 600`（= 30 fps 下 20 mm/帧，见 §8.9.4；`--arm-slew-mm` 可改回固定 mm/帧，`--arm-lateral-guard-mm 20` 可恢复横向护栏，`--no-arm-gain-quality-guard` 关标定护栏）；另有三项 2026-10-14 的现场默认：`--arm-anchor upper`（肩端取肘的父连杆，§8.13.2）、`--arm-ik-scope arm`（IK 只动本侧手臂，§8.13.3）、`--arm-elbow-recovery-margin-deg 8` + `--arm-elbow-recovery-frames 12`（支路恢复的滞回与冷却，§8.13.4）。
- **60 / 1e-4 是特意选的**：实测比 `PyBulletArmIK` 自带的 200 / 1e-5 又快又准（h1_2 稳态残差 10.3 / 11.7 vs 18.4 / 20.4 mm）；三台同时跑约 11.2 fps、单侧约 17.7 fps（本机 CPU）。
- 两个"标定帧"不同义：`--calibration-frames`（开手标定，基线口径）vs `--arm-calibrate-frames`（手臂中性帧），混用会互相污染。
- 常量：`REFERENCE_REACH_M=0.60`、`ARM_LENGTH_BAND=(0.5, 2.0)`、`MIN_OPERATOR_REACH_M=0.15`、`DEFAULT_MAX_TRAVEL_M=0.10`、`FALLBACK_IMAGE_SCALE=0.7`、`TRAVEL_MODES=("clip","scale")`、`DEFAULT_LATERAL_GUARD_M=0.02`、`MIN_PLAUSIBLE_ARM_REACH_M=0.30`、`ARM_REACH_SIDE_TOLERANCE=0.25`（后四个是 2026-10-13 的护栏，见 §8.8）；gain 按标定那几帧的臂长 / 0.60 归一化，标定臂长不可信时退回 `metric_scale`。
- pose 丢帧（None / NaN）-> 保持上一帧目标；默认允许"21 点 -> 手腕"兜底（乘 0.7），`--no-arm-fallback` 可关。
- 退出码：0 = 至少一侧有输出，1 = 全程没数据，3 = 解释器 / 依赖 / 模型不对；`__main__` 用 `os._exit` 绕开 MediaPipe `close()` 约 42 s 的阻塞（别"顺手改成正常 return"）。

### 8.3 验收证据（2026-10-12，本机实测，命令 + 退出码 + 不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `E:\python3.11.7\python.exe devtools\verify_realtime_arm_bridge.py --side both --report outputs\tmp_realtime\verify_default_after.json` | 0 | 中性标定 60 帧完成（有效帧 left 45 / right 46）；标定后第一帧目标 = 中性工作点（最差轴 0.0000 mm）；稳态残差 mean 18.7~23.1 mm（h1_2 18.8 / 23.1、gr1_t2 19.1 / 18.7；阈值 25 / 30 mm，判据只看 mean）；丢帧保持 81 次全中；未驱动关节漂移 <= 0.0005 rad（§8.6 改静息点后重跑） |
| 同上加 `--arm-ik-iterations 60 --arm-ik-residual-threshold 1e-4 --report outputs\tmp_realtime\verify_ik60_after.json` | 0 | 同上，稳态残差 mean 7.4~10.8 mm（h1_2 7.5 / 10.8、gr1_t2 7.9 / 7.7 mm）—— 这就是入口默认口径，数字与 profile 阈值一致（§8.6 改静息点后重跑） |
| `E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --headless --input-npz outputs\tmp_realtime\replay_frames.npz --frames 40 --no-preview --report outputs\tmp_realtime\replay_report4.json` | 0 | 40 帧回放完、不读相机（3.42 s / 12.1 fps）；手 24 / 18 帧有效（其余是 3 帧滑窗预热与刻意丢帧）；`applied={'left': 24, 'right': 18}`；三台残差 h1_2 14.9 / 7.0、gr1_t2 6.4 / 4.3、g1 24.8 / 3.2 mm；hold drift <= 0.0003 rad（§8.6 改静息点后重跑：g1 右 14.8 -> 3.2 mm 变好，g1 左 24.8 与改前一字不差 = 它本来就用 +0.10） |
| `E:\python3.11.7\python.exe -m pytest tests -q` | 0 | 280 passed, 3 skipped（289 subtests；含本层 30 + 21 条）—— 改前基线 259 passed / 3 skipped |
| 真机无头：`E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --headless --frames 120 --preview-dump-dir outputs\tmp_realtime\preview_arm --preview-dump-frames 6 --dump-npz outputs\tmp_realtime\live_frames.npz --report outputs\tmp_realtime\live_arm.json` | 0 | 真相机 120 帧 / 14.2 fps；`camera preview: window=off arm_overlay=on dumped_png=6`（存下带手臂层的 PNG）；右臂中性标定 30/30 -> `delta_op_cm=[-22.38, +30.08, -7.95]`、肘角 102.8°；三台稳态残差 3.9~14.7 mm；hold drift <= 0.0002 rad；左臂本帧不入镜 -> 走 21 点兜底（§8.6 改静息点**之前**的真机记录，残差要重跑才算最新） |
| 真机两窗口：`E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --frames 150 --preview-dump-dir outputs\tmp_realtime\preview_gui --preview-dump-frames 3 --report outputs\tmp_realtime\gui_arm.json` | 0 | 150 帧 / 10.1 fps；`camera preview: window=on mirror=on arm_overlay=on rect=(28,91,640,480)` + PyBullet 三机器人窗口同时开；双侧 `L=P R=P`、`cal=ok`、手 15 帧有效；三台残差 2.6~27.1 mm；`sim_frames=150`（§8.6 改静息点**之前**的真机记录） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | 0 | `RESULT: [OK]`（8 项全过，2 warnings；unittest discover `Ran 283 tests`、`OK (skipped=3)`；4 条 CLI 冒烟 exit 0；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`）——§8.6 改静息点后重跑同结果 |

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

### 8.5 采集窗口：手和手臂在同一个窗口

要求是"一个窗口看采集、另一个窗口看仿真"，而基线的 `CameraPreview`
（`applications/realtime_hand_sim.py`）只画 **21 点手部骨架**（那个入口不驱动手臂）。
既要叠加手臂、又**不能改基线文件**，于是**子类化**它的预览：

```
CameraPreview（基线：取画面 / 镜像 / 挪窗口 / 置顶 / 存 PNG / q|Esc 退出）   <- 一行未改
  `- ArmCameraPreview.annotate()      simulation/arm_preview.py（新增）
       画面 + 21 点手骨架（基线画的）
       + 肩->肘->腕 折线（每侧同色系浅一档，腕点放大成橙点）
       + 每侧一行数据：来源字母（P/H/-）+ 相对中性姿势的三轴位移厘米 + 臂长 + 肘角
       + 画面外提示：`right 3/3 arm point(s) outside the frame (step back or tilt the camera)`
       + 底部图例：`arm: P=pose arm H=hand fallback -=held`
```

- 窗口标题与基线区分开：`camera: hands + arms (q/Esc quits)`。
- 数据由入口准备（`camera.arms()` 的归一化肩/肘/腕 + `bridge.trace()` 的位移），预览**只画**：
  像素不回灌重定向；开不出窗口（没桌面 / 没 HighGUI）只打印一行 note，主链路与 `--report` 不受影响。
- `--no-arm-overlay` 回到基线那种"只有手"的窗口；`--report` 的 `preview.arm_overlay` 记录本场开没开。
- 非有限（NaN / Inf）或形状不对的手臂点统一当"没有这一侧"（`finite_arm_points`），与桥的
  "丢帧保持"同口径：看不见就不画，也不猜。
- 这层折线是 **pose 归一化坐标的显示投影**，不是三维重建（本层没有相机内参、没有手眼标定）。

**实测现象（第一次上真机就会遇到）**：人离相机太近时，MediaPipe 的肩/肘/腕会落到画面外 ——
本机实测（640x480）肩 `y=0.96`、肘 `y=1.11~1.44`、腕 `y=0.90~1.57`，换算成像素就是
459（画面内）/ 533（画面外）/ 433（画面内）。所以"窗口里看不见手臂线"通常是
**手臂没进画面**而不是没数据；往后退到肩肘腕都进画面，折线立刻出现，
`outside_frame()` 也会把"有几个点在画面外"直接写在窗口上。

### 8.6 静息工作点：\"两手交叉\"的根因与修复（2026-10-12 实测）

现象：实时遥操作里**松手（回到中性姿势）时两只手叠在一起、前臂交叉在胸前**，
而加 `--arm-travel-mm 150` 几乎看不出区别。根因在中性工作点（`realtime_offset`）上，
跟轴顺序、跟行程都无关：

| robot | 肩关节在基座系的 y（URDF `*_shoulder_pitch_joint` 的 origin） | 改前中性 y | 后果 |
|---|---|---|---|
| h1_2 | left +0.14806 / right -0.14806 | 两侧都是 0 | 两个腕子被命令到**同一个点**，前臂只能收敛、互相穿插 = \"交叉\" |
| gr1_t2 | left +0.10128（roll +0.09515）/ right -0.10094 | 两侧都是 0 | 同上 |
| g1 | left +0.10022 / right -0.10021 | left +0.10 / right 0 | 左臂正常；**右臂中性点就在中线**、行程只有 4 cm，永远够不到自己那侧 |

（REP-103：机器人基座系 x 前 / y 左 / z 上，所以左肩 y 为正。手部重定向本来就是**相对中性姿势**的，
所以\"操作者两手相距 40 cm\"在中性姿势下被压成\"机器人两手相距 0\"。）

- **`--arm-travel-mm` 治不了这个**：它改的是\"围绕中性点还能走多远\"，中性点本身不动 ——
  松手看到的就是中性点，行程只在大幅动作时才可见（§8.2 已注：行程要两侧同时驱动才有意义）。
- 改法：把每侧中性 y 挪到自己**肩膀下方**（让机器人像人一样\"手在身侧\"），h1_2 因残差只挪一半。

**改前 / 改后（同一条命令 + 同一份录制数据 `outputs\tmp_realtime\axis2_frames.npz`，约 2280 帧）：**

| 机器人/侧 | 改前中性 y | 改后中性 y | 稳态残差 mean（mm）改前 -> 改后 | 上限 |
|---|---|---|---|---|
| h1_2 left | 0.00 | +0.05 | 16.8 -> 22.7 | 30 |
| h1_2 right | 0.00 | -0.05 | 27.9 -> **22.7** | 30 |
| gr1_t2 left | 0.00 | +0.10 | 7.8 -> 18.4 | 25 |
| gr1_t2 right | 0.00 | -0.10 | 13.0 -> 13.4 | 25 |
| g1 left | +0.10 | +0.10（未变） | 22.5 -> 22.5 | 25 |
| g1 right | 0.00 | -0.10 | 11.8 -> 18.9 | 25 |

```
E:\python3.11.7\python.exe devtools\verify_realtime_arm_bridge.py --input-npz outputs\tmp_realtime\axis2_frames.npz --side both --report outputs\tmp_realtime\npz_axis2_half.json
-> 退出码 0；稳态残差 6 条全部 [OK]，相比改前基线没有新增失败
```

> `*_neutral_start` 那 3 条 `[FAIL]` 是喂**录制**数据必然踩到的：录下来的开头就是当时那下的
> \"伺服进入段\"。改前基线同样 3 条、逐条相同，因此判据是稳态残差（约定 5）。
> g1 左臂改前改后都是 22.52 / 22.5 mm —— 它本来就用的 +0.10，这对数字同时是\"A/B 管线确实生效\"的对照。

- 取舍是量出来的：`--probe-lateral --side both` 扫 y = 0 / +-0.10 / +-0.15 / +-0.20。
  注意它**一次把两侧移到同一个 y**，所以判\"左右对称张开 a\"要读**左臂在 +a 行、右臂在 -a 行**
  （数字是稳态残差 mean / max，毫米）：

| 扫描 y | h1_2 左 / 右 | gr1_t2 左 / 右 | g1 左 / 右 |
|---|---|---|---|
| +0.00 | 13.5 / 14.2 | 9.6 / 9.1 | 31.2 / 18.3 |
| -0.10 | 29.8 / 28.0 | 22.1 / 17.3 | 89.5 / 19.6 |
| -0.15 | 74.6 / 35.6 | 43.6 / 28.0 | 177.9 / 28.3 |
| -0.20 | 146.0 / 44.2 | 102.3 / 39.8 | 290.4 / 37.4 |
| +0.10 | 27.7 / 27.2 | 16.0 / 21.5 | 21.1 / 81.9 |
| +0.15 | 35.4 / 68.8 | 25.2 / 45.8 | 25.1 / 149.4 |
| +0.20 | 44.2 / 137.9 | 35.6 / 93.5 | 31.1 / 260.1 |

  读法：三台**两侧都最想要中线**（13.5 / 14.2 / 9.6 / 9.1 / 18.3 mm），越往自己肩膀下挪残差越大；
  也就是\"手在身侧（好看）\"与\"IK 解得准（数字好看）\"天生冲突，取值是这条曲线上选的折中点。
  h1_2 左臂 +0.10 会到 30.05 mm（越 30 mm 上限），所以它取 +0.05（22.7 mm）；gr1_t2 / g1 能接受满量 +-0.10。
- 左右能力不同是**手臂自己的关节限位**，不是映射错：h1_2 左臂 +0.10 越线、右臂 -0.10 反而更好
  （27.9 -> 22.4）；g1 左臂本来就需要 +0.10。
- 真机观感仍要**你本人**看一次：本层只有残差数字，没有自动化的渲染判据。

#### 8.6.1 2026-10-13 复核更正：窗口里那一下"交叉"是谁造成的 + 目标速率限制

用**同一份录制数据**（`outputs\tmp_realtime\live_demo_frames.npz`，2026-10-13 19:43 那场
300 帧真机演示）无相机重放，按 A/B 只换一个开关：

```powershell
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py ^
  --input-npz outputs\tmp_realtime\live_demo_frames.npz --frames 300 --headless ^
  --arm-slew-mm 0  --report outputs\tmp_realtime\replay_slew0.json
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py ^
  --input-npz outputs\tmp_realtime\live_demo_frames.npz --frames 300 --headless ^
  --arm-slew-mm 20 --report outputs\tmp_realtime\replay_slew20.json
```

1. **不是左右接反**：两张报告 `arm_last_targets` 里三台的左目标都在基座系 +y、右目标都在 -y
   （h1_2 `neutral_m=[0.24,+-0.05,0.19]` + `delta_base_m` 逐轴可复算）。
2. **窗口里那一下交叉是操作者自己的手**：最后一帧 `h1_2/right`
   `delta_op_m=[+0.081, -0.079, +0.164]` -> `delta_base_m=[-0.15, +0.091, +0.089]`，
   即 `op_x=+8.1 cm`（画面右 = 你自己的左侧）把右腕从中性 -0.05 推到 **+0.041（越过机器人中线
   4.1 cm）**，实测腕位 +0.043。左右两只手的 x 位移都在 +0.08~0.11 m，说明那一帧你是**整体
   往画面右移**；相对重定向传的是"手相对中性姿势的位移"，横向整体位移没有单独扣掉。
   -> 想"窗口里永不出现交叉"要做的是**横向护栏**（左 y >= +0.01 / 右 y <= -0.01），
   代价是故意把一只手跨到另一侧也会被挡住；这是产品取舍，还没做（第 5 条）。
3. **更正 8.6 的一处口径**：§8.6 / `diag_neutral.py` / 离屏渲染量到的
   "h1_2 实际 y = ±0.003、前臂间距 6.0 mm、双手并到中线"是**把中性目标只下发一次**的合成结果。
   真实链路每帧都下发，标定期 h1_2 残差就是 9 mm（`E:\cache\tmp\live_demo.out` 的 frame 15/30 行，
   重放同步复现）。所以"真机上双手并到中线"这句**不成立**，别再引用它当现场现象。
4. **量到并修掉的真实缺陷：两步 IK 的分支锁定**。目标一大步到位时，第二遍解会停在一个坏分支
   上、并且每帧都解回同一个分支。同机同口径（gravity -9.81、60 次迭代 / 1e-4、入口同一套设置）：

   | 场景 | h1_2 | gr1_t2 | g1 |
   |---|---|---|---|
   | 只下发一次然后保持（离屏渲染 / 探针的算法） | 47.8 mm，前臂间距 6.0 mm | 31.6 mm | 68.7 mm |
   | 同一工作点以 20 mm/帧走进去 | **8.9 mm / 83.5 mm** | **6.7 / 200.0** | **8.9 / 186.8** |
   | 重放整场 帧 105-165（稳态段） | 45 mm -> **9 mm** | - | - |

   于是新增入口参数 `--arm-slew-mm`（默认 **20 mm/帧**，`0` = 旧行为）：
   每帧最多让**下发目标**朝目标点移动 20 mm，一侧的第一次下发从腕子**当前**位置起步（不跳），
   之后的帧从上一次下发位置起步。报告里多了 `target_slew_mm` 与 `target_slew_limited_frames`。
   实现：`arm_scene_driver.ArmSceneDriver._slew_target`；逐帧曲线
   `E:\cache\tmp\probe_slew_curve.py`（tmp，不入库）。
   回归：`tests\test_arm_ik_multi_robot.py::ArmTargetSlewTests`（2 例）；
   `pytest tests -q` -> 298 passed / 3 skipped（改前 296/3，退出码都是 0）。

5. **仍未解决，下一轮**：
   - 重放里帧 45-90 与 180-300 两段 h1_2 残差 100-200 mm，**开不开速率限制都一样**，
     所以不是目标速率问题；这两段正好是操作者 pose 数据自己跳变/缺失的区间
     （`--arm-trace --print-every 1` 可见）。要先分清是 pose 抖动被 gain 放大，还是求解器
     在那几帧又落回坏分支。
   - 是否加横向护栏（第 2 条），以及 h1_2 的中性横向取值要不要跟着动，都等确认再改。

### 8.7 没做的（交接）

- **轴向已用真实会话离线复算核对**（2026-10-12）：从 `--dump-npz` 的 world 手部轨迹逐帧复算
  "操作者轴位移 -> 机器人基座系目标"，与 §8.2 的 `AxisMap` 一致。例：某一帧 `op=[-1.2, +31.7, +13.8] cm`
  按 `AxisMap=(Z,X,Y)/(-1,+1,-1)`、gain 0.58 应得 `(-7, -80, -184) mm`，实测 g1 左臂目标
  `[-84, -8, -150] mm` —— 逐轴吻合（x 差 4 mm 是报告里那个伺服抖动；`-184 -> -150` 正是
  `--arm-travel-mm 150` 的每轴夹取）。所以"跟手"不再只是相对自洽；还差的只是 gain 的归一化口径
  （按标定那几帧臂长 / 0.60）与 §8.6 那种左右能力差异。
- 真机已跑通：**无头**与**两窗口**两种模式（证据见 §8.3 最后两行）；多轮真实会话里两侧手臂
  都有目标与残差记录（§8.6 的 A/B 就是 2280 帧实录）。
- 仍没做的：中性工作点的真机**观感**要你本人再确认一次（§8.6 结尾：本层只有残差数字，没有渲染判据）。
- 手臂目标是**相对重定向**（以中性姿势为原点），不是手眼标定；换操作者、换站位、换相机摆放都要重做中性标定。
- 未做：双臂协调动作测试、与 `retargeting/arm/wrist_targets.py` 那条 21 点 -> 手腕链路的联合调试。

### 8.8 三个护栏 + 姿态约束：现场那场「H1-2 抽动 / 幅度不对 / 前臂差点贴上」的逐项复算（2026-10-13）

现场那场录了 1560 帧（约 8 fps，`outputs\tmp_realtime\live_long_frames.npz`），观感问题
报了三件：H1-2 抽动、手臂幅度跟手对不上、前臂差点贴到一起。用**同一份录制**逐项复算
（每次只改一个开关，退出码都是 0）：

| # | 配置 | h1_2 前臂间距 min | h1_2 残差 L / R | 姿态误差 L / R | 肩 yaw 行程 L / R | 夹取帧 |
|---|---|---|---|---|---|---|
| 1 | 现场原样（姿态 on + clip + 无护栏） | 2.21 mm | 95.8 / 133.3 mm | 46.0 / 46.4 度 | 322.4 / 235.5 度 | 6599 |
| 2 | `--arm-orientation off` | 1.47 mm | 17.1 / 3.9 mm | 1.5 / 0.4 度 | 209.6 / 68.7 度 | 6599 |
| 3 | 2 + `--arm-travel-mode scale --arm-lateral-guard-mm 20`（**推荐组合**） | 27.9 mm | 24.5 / 4.9 mm | 2.3 / 0.7 度 | 194.7 / 66.8 度 | 6221 |
| 4 | 2 + `--arm-lateral-guard-mm 20`（仍 clip） | 33.0 mm | 13.8 / 4.2 mm | 1.2 / 0.4 度 | 200.1 / 69.4 度 | 6599 |

配置 1 的数字与现场实录逐位一致（2.21 / 95.8·133.3 / 6599），所以这张表是在同一口径下比。
g1 与 gr1_t2 在 1-4 里始终是 position-only，残差 9.3 / 6.0 与 9.2 / 8.3 mm 基本不动。

复算命令（无相机，1560 帧约 2 分钟，本机约 13 fps）：

```powershell
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop `
  --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 1560 --headless `
  --arm-slew-mm 20 --arm-calibrate-frames 60 `
  --arm-orientation off --arm-travel-mode scale --arm-lateral-guard-mm 20 `
  --report outputs\tmp_guard\e3_rec.json
```

**1. 抽动 = 姿态约束这条链路（h1_2 是唯一的 `pose` 机器人）。** 表 1 vs 2 只改了
`--arm-orientation`：残差降 5 倍以上，姿态误差 46 度 -> 1.5 度，肩 yaw 从「绕一圈」
（322.4 / 235.5 度）回到 200 度以内的正常量级。为什么会这样：桥里的腕姿态来自三点
造的腕坐标系（x = 前臂方向，z = 上臂 x 前臂），平面法向在手臂接近伸直时几乎不可
观测，而判据 `MIN_ELBOW_PLANE_SINE = 0.05` 只挡住 2.9 度以内的弯曲。拿同一份录制
离线复算（`E:\cache\tmp\probe_orientation_noise.py`，不需要 PyBullet）：相邻两帧
「相对中性旋转」的夹角中位数 8.4 / 9.0 度（真实手腕在 8~13 fps 下的量级），
但 p90 45.4 / 40.7 度、最大 178.2 / 174.4 度 —— **4~7% 的帧命令了一个 60 度以上
（最多近 180 度）的姿态跳变**（左 89 帧、右 55 帧），IK 只能牺牲位置去追它，
于是位置残差 100 mm 级 + 肩关节持续回绕，看起来就是「抽动」。

处置：入口 `--arm-orientation` 默认改 `off`（不删任何东西，`--arm-orientation on`
或 `auto` 随时回到旧行为，`arm_profiles.h1_2.ik_mode` 仍是 `pose`，把「机器人能力」
和「这次用不用」分开）。要真正把姿态用起来，得先做三件事中的至少前两件：
提高 `MIN_ELBOW_PLANE_SINE`（只信明确弯曲的肘）/ 对姿态目标做离群帧剔除（单帧
>60 度就保持上一帧）/ 用 21 点定出手轴补上缺的那一个自由度。

顺带发现一个**潜在雷**（本场景没触发，换摆放要小心）：`arm_ik.to_world()` 只把
*位置*从机体系转到世界系，而 `calculateInverseKinematics` 的 `targetOrientation`
参数是按**世界系**解释的 —— 基座带旋转的机器人，姿态目标会差一个 `R_base`。
实测本场景三台基座都是单位旋转（`E:\cache\tmp\probe_base_frame.py`：h1_2 / gr1_t2 / g1
base_angle 都是 0.00 度），所以解释不了这次的现象，但那条注释「GR1-T2 does not load
with an identity base rotation」提示别的入口确实会带旋转。

**2. 交叉 = 目标本身可以越中线（不是"左右接反"）。** h1_2 的中性工作点只有 y=+-0.05 m，
而每轴行程上限是左 0.10 / 右 0.15 m，于是**命令侧**就允许左腕 y 走到 -0.05（身体另一侧）。
新增 `--arm-lateral-guard-mm`（2026-10-13 白天默认 20 mm，**当晚默认改回 0**，见 §8.9.4）：
左腕 y >= +20 mm、右腕 y <= -20 mm，
只动 y、x/z 与 y 的其他范围照跟，越线时把 y 停住。表 2 vs 3 / 4：前臂间距
1.47 mm -> 27.9 / 33.0 mm。护栏挡回的帧数进报告（`arm_guarded_frames`：3 配置 924 帧、
4 配置 1375 帧），HUD 每帧显示 `guard=`，`0` 表示关闭。

**3. 幅度/跟手 = 标定臂长 + 行程饱和。** 同一份录制、`--arm-calibrate-frames 60`
标出来的中性臂长是**左 0.2742 / 右 0.3976 m**（同一个人两侧差 31%，都不是一条手臂该有的
长度），`0.60 / 0.2742` 把 gain 顶到 2 倍上限（h1_2 左 1.4）；再叠上永远被夹断的行程
（`arm_clamped_frames` 6599 次 / 1560 帧 = 每帧每轴都夹），就是「手一动、机器人就钉死」。
新增两条：

- `--arm-travel-mode scale`（2026-10-13 白天到晚上的默认，**10-13 晚已改回 `clip`**，见 §8.9.4）：
  超限时**整条位移等比缩** —— 方向不变，只让最远的那一轴正好落在上限；`clip` 是逐轴剪。
  实测结论（§8.9）：等比缩会被最抖的纵深轴拖累，把横向/竖向摆幅一起压小，所以默认取逐轴剪。
- **标定可信度护栏**（`--no-arm-gain-quality-guard` 可关）：标定臂长 < 0.30 m，或两侧
  相差 > 25% 时不再做臂长归一化，退回 `profile.metric_scale`，并把原因写进报告的
  `arm_length_trust` / `arm_gain_guarded_frames`，启动时也打印一行。

  > 注意：`--arm-calibrate-frames 30` 与 `60` 会标出**不同**的中性 —— 同一份录制，
  > 30 帧得到左 0.334 / 右 0.338 m（两侧一致、可信，gain 1.2575 / 1.2419），
  > 60 帧得到上面那组（左不可信 -> 退回 0.7）。所以同一套代码两次跑可以给出不同 gain。
  > 现场应当固定帧数，或者用 `--save-neutral` / `--neutral-json` 复用同一份中性。

**4. 仍未解决（与本轮无关，配置 1-4 都复现）：** 帧 810-870 与 1455-1515 三台机器人的
**左臂**同时出现 0.47~0.57 m 的残差尖峰（HUD 上 `hand valid=left=0`、`arm L=H`），
持续约 1 秒后自己恢复。目标被 hold 住而实测腕位跑掉了，所以不是目标速率问题，
更像场景里的碰撞/大幅瞬移。下一步：在那几帧打 `getContactPoints()` + 关节角曲线，
分清是「撞到了」还是「求解器自己跑飞」。

验收（改动后，本机实测）：

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `python -m unittest discover -s tests -v`（`PYTHONUTF8=1`） | 0 | `Ran 311 tests / OK (skipped=3)`（改前 301 / 3，新增 10 例护栏用例） |
| `python devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符 |
| `python -m teleoperation.simulation.realtime_teleop --help` | 0 | 三个新开关可解析、`%` 转义正确 |
| 上面那条 1560 帧复算（配置 3） | 0 | h1_2 残差 24.5 / 4.9 mm、前臂间距 27.9 mm、姿态误差 2.3 度 |

### 8.9 「横向摆动丢失」的根因与修复（2026-10-13 现场：大臂只上下摆、肘几乎不动）

现场（两个窗口都开着，跑 1560 帧那份录制）：h1_2 / gr1_t2 的大臂**只能上下摆、不能左右摆**，
肘夹角几乎不变；g1 **大臂不动、小臂能动**，也不能左右摆。

**结论**：三条独立机制叠加，按对"摆动"的贡献排序（数字都在下面两张表里）：

| # | 机制 | 实测代价 | 处置 |
|---|---|---|---|
| 1 | 斜坡是**固定 20 mm/帧**，而现场摄像头循环只有 8-12 fps -> 目标速度被钉在 0.20-0.24 m/s。录制里操作者手的每帧位移是**中位 37 mm / p90 147 mm**（10.05 fps -> p90 1.48 m/s），于是斜坡**帧帧生效** | 正弦左右摆（±9 cm、16 帧/周期）机器人腕摆幅只剩操作者的 **0.43**、相位滞后 2.1 帧 | 斜坡改**速率口径** `--arm-slew-mm-s`（默认 600 mm/s，另有 40 mm/帧天花板）-> **0.80**、滞后 0.6 帧 |
| 2 | `--arm-travel-mode scale`：**一个轴超限就把整条位移等比压小**，而现场 op_z（纵深，MediaPipe 最抖的轴）经常是最大轴 | 现场那帧 (-9.3, +6.4, +24.5) cm：g1 左腕横向只剩 **1.31 cm**、肩 roll 2.0 度、肘 2.8 度 | 默认改**逐轴剪断**（`clip`）-> 横向 **3.44 cm**、roll 5.5 度、肘 7.1 度 |
| 3 | 横向护栏 20 mm（左腕 y >= +20 mm / 右腕 y <= -20 mm）把"往中线方向摆"的一半行程直接挡掉 | h1_2 左腕 ±9 cm 摆动的传递比 0.49 -> 0.43；右腕 9.0 cm 的向内请求只落 2.9 cm | **默认改回 0（关掉）**（2026-10-13 晚现场决定）：跨中线是操作者自己把右手横过身体中线的忠实映射。要"窗口里永不跨中线"给 `--arm-lateral-guard-mm 20`（折中 10），代价是两前臂间距会掉到 2.21 mm（§8.8 配置 3 实测） |

**肘为什么几乎不动**：桥每帧算 `elbow_bend_deg`，但**没有任何一项在跟踪肘角** ——
目标只有"腕位置 + （h1_2）腕姿态"，驱动层 `track_posture=False`。肘只能靠 IK 的零空间跟着走，
于是肘角变化基本是位移量的导数：横向目标被压到 1-3 cm 时肘只转 2-3 度；横向一旦活下来
（`clip` + 关护栏）肘就转到 **7.1 度（g1 左）/ 8.3 度（g1 右）/ 14.96 度（h1_2 右）**。
**所以"肘不动"是位移压缩的症状，修压缩就修了它**；真要论文那种肘姿态跟踪得单独做（见 §8.7）。

#### 8.9.1 量法与命令（探针都不入库，产物在 `E:\cache\tmp`）

| 探针 | 量什么 | 命令（在仓库根目录跑） |
|---|---|---|
| 单轴隔离 | 每次只沿**一个轴**给固定位移，读回三台每侧**每个关节角**、腕实际位移、残差、被夹/被护栏标记 | `E:\python3.11.7\python.exe E:\cache\tmp\probe_axis_response.py --modes live,clip,noguard,noslew,clip_noguard,clip_noguard_noslew --report E:\cache\tmp\probe_axis_response.json`（退出码 0，111 s） |
| 正弦跟随 | 操作者按 ±9 cm 横摆 + ±24 cm 纵深摆，量机器人腕的**摆幅传递比与相位滞后** | `E:\python3.11.7\python.exe E:\cache\tmp\probe_sine.py --modes live,clip_noguard,clip_noguard_slew40,clip_noguard_slew60,clip_noguard_slew80 --kind with_z --depth-cm 24 --report E:\cache\tmp\probe_sine60.json`（退出码 0，39 s） |
| 现场手速 | 从录制 npz 量操作者每帧位移分位与主频（给正弦挑参数，不靠猜） | `E:\python3.11.7\python.exe E:\cache\tmp\probe_speed.py`（退出码 0，只读） |
| 关节/步长汇总 | 把上面的 json 压成"按关节"的表（含第一帧实际步长） | `E:\python3.11.7\python.exe E:\cache\tmp\probe_joints.py E:\cache\tmp\probe_axis_response.json live_worst elbow` |

三个探针复现的是**同一条链路**：`RealtimeArmBridge.update()` -> `ArmSceneDriver.apply_each()`
-> PyBullet 两遍热启动 IK（60 迭代 / 1e-4），操作者几何用现场那份中性标定
（`outputs\tmp_realtime\live_long_neutral.json`，左 reach 0.2742 / 右 0.3976 m），
所以 gain（h1_2 左 0.70 / 右 1.056，gr1_t2 与 g1 左 0.40 / 右 0.604）与现场一致。

#### 8.9.2 稳态数字（`probe_axis_response.json`）

case `live_worst` = 现场一帧复合位移 (-9.3, +6.4, +24.5) cm，**只换 travel_mode**：

| 机器/侧 | 模式 | 腕横向位移 | 肩 pitch | 肩 roll | 肘 | 残差 |
|---|---|---|---|---|---|---|
| g1/left | live（scale） | **1.31 cm** | +9.83 | -2.01 | -2.76 | 7.1 mm |
| g1/left | clip | **3.44 cm** | +6.18 | -5.54 | +7.11 | 5.9 mm |
| g1/right | live | 1.68 | +10.17 | -1.90 | -4.92 | 8.5 mm |
| g1/right | clip | 4.21 | +4.72 | -5.11 | +8.32 | 8.3 mm |
| h1_2/right | live | 6.24 | +29.02 | -9.28 | -15.70 | 5.5 mm |
| h1_2/right | clip | **10.35** | +28.47 | -13.06 | -10.58 | 6.6 mm |

这正是现场那句"大臂只上下摆"：`scale` 下 g1 的位移几乎全在 pitch（上下）上，
roll（左右）只转 2 度；`clip` 把 roll 换回 5.5 度、肘换回 7.1 度，且**残差更小**。
`scale` 当初的理由（"盒角比任何单轴都远、IK 到不了、手再动也不跟"）在这批数字里没有出现：
g1 三轴都剪断（盒角）时残差 5.9 mm，判据 25 mm。

#### 8.9.3 动态数字（`probe_sine*.json`）

±9 cm 横向正弦 + ±24 cm 纵深正弦，16 帧/周期（现场 8-12 fps 下一次左右摆约 1-2 s）：

| 模式 | 横摆传递（腕/操作者） | 相位滞后 | 纵深传递 | 残差 均值/最大 |
|---|---|---|---|---|
| live（scale, 护栏 20, 20 mm/帧） | **0.43** | 2.1 帧 | 0.25 | 8.2 / 15.1 mm |
| clip + 20 mm/帧 | 0.49 | 2.2 | 0.24 | 7.9 / 13.5 |
| clip + 40 mm/帧 | **0.80** | 0.6 | 0.43 | 11.1 / 19.8 |
| clip + 60 mm/帧 | 0.80 | 0.4 | 0.48 | 10.9 / 26.1 |
| clip + 80 mm/帧 | 0.80 | 0.4 | 0.48 | 10.8 / 23.5 |

40/60/80 三档横摆传递都是 0.80（斜坡不再是瓶颈），但最大残差从 19.8 涨到 26.1 mm
（判据 30 mm），所以天花板取 **40 mm/帧**。同一条链路拆开看（`live` / `clip+40`）：
操作者 9.00 cm -> 桥目标 6.30 cm（gain 0.70）-> 驱动层实际下发 3.38 / 6.51 cm
-> 腕实际 3.86 / 7.23 cm。护栏那一档的差价：0.43 vs 0.49。

**在真实录制上复核斜坡步长**（同一份 `live_long_frames.npz` 前 120 帧，`--arm-travel-mode
clip`，逐帧 HUD 的瞬时残差）：

| `--arm-slew-mm` | 残差均值 g1 / gr1_t2 / h1_2 | 残差最大 g1 / gr1_t2 / h1_2 | 斜坡生效的调用数 |
|---|---|---|---|
| 20（旧默认） | 12.0 / 9.3 / 10.6 mm | 48 / 28 / 40 mm | 359 |
| 30 | 12.6 / 9.7 / 12.1 | 57 / 36 / 50 | 359 |
| 40（= 新默认在 12 fps 下的换算值） | 12.9 / 10.0 / 13.1 | 61 / 36 / 58 | 359 |

均值只涨 8-24%，而正弦那边摆幅传递 0.49 -> 0.80 —— 所以 40 是"摆动回来了、跟手误差没坏"
的折中点。注意三档的 `sat` 都是 359（120 帧 × 3 台 —— 这份录制的手速中位 37 mm/帧本身就
超过任何一档），也就是说现场那份输入下斜坡**始终**在生效，误差主要是输入抖动而不是步长。

#### 8.9.4 默认值（2026-10-13 起，改前 -> 改后）

| 开关 | 旧默认 | 新默认 | 为什么 |
|---|---|---|---|
| `--arm-travel-mode` | `scale` | **`clip`** | 只压超限轴（§8.9.2）；`scale` 会被最抖的纵深轴拖垮横向 |
| `--arm-slew-mm-s` | 无（等价固定 20 mm/帧） | **600 mm/s** | 速率口径：30 fps 下 20 mm/帧（与旧口径同速），10 fps 下 40 mm/帧（天花板） |
| `--arm-slew-mm` | 20.0 | **None** | 留给"逐字复现 2026-10-13 旧数字"；给了它就按固定 mm/帧 走 |
| `--arm-lateral-guard-mm` | 20 | **0（关掉）** | 护栏挡掉的正是"往中线方向摆"那半程（§8.9 表 #3）；现场 2026-10-13 晚决定：跟手优先，跨中线是操作者自己的动作 |
| `--arm-orientation` | `off` | `off`（不变） | 见 §8.8 |

`describe()` / `summary()` 与 `--report` 新增 `target_slew_mm_per_s`、
`target_slew_ceiling_mm`、`arm_slew_mm_last_frame`（本场实际用到的每帧上限）。

#### 8.9.5 现场怎么跑（改完的两条 A/B 命令）

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"

# 新默认（clip + 600 mm/s + 护栏关）：左右摆应当明显回来
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-trace --print-every 5

# 旧行为（逐字复现改前那天）：--arm-travel-mode scale --arm-slew-mm 20 --arm-lateral-guard-mm 20
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-trace --print-every 5 `
    --arm-travel-mode scale --arm-slew-mm 20 --arm-lateral-guard-mm 20
```

窗口里看三件事：`sat=`（斜坡生效的调用数）在旧口径下会一直涨、新口径涨得慢；
`delta_op_cm=[..]` 与 `P d(cm)[..]` 的横向分量应当同向同幅度；`guard=` 现在应当一直是 0。
护栏关掉后**要留意前臂会不会真的撞上**：操作者把右手横过身体中线时两前臂间距会掉到
2.21 mm（§8.8 配置 3 实测），看到了就加 `--arm-lateral-guard-mm 10`（只挡最后 1 cm）。

#### 8.9.6 验收（改动后，本机实测）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `$env:PYTHONPATH="$pwd\src"; python -m unittest tests.test_arm_realtime_bridge -q` | 0 | `Ran 53 tests / OK`（新增"只有一个轴超限时别的轴原样保留"与"斜坡默认是速率口径"两条） |
| `$env:PYTHONPATH="$pwd\src"; python -m unittest tests.test_arm_ik_multi_robot -q` | 0 | `Ran 24 tests / OK`（新增"速率口径在慢循环里换算"一条） |
| `$env:PYTHONPATH="$pwd\src"; python -m pytest tests -q` | 0 | `311 passed, 3 skipped, 300 subtests passed`（改前 308 passed + 3 skipped） |
| `python devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符 |
| `python devtools\verify_realtime_arm_bridge.py --report E:\cache\tmp\verify_after_clip.json` | 0 | 六个"机器 x 侧"全 `[OK]`（残差均值 18.7-25.6 mm / 天花板 25-30 mm、行程与命令一致）、`realtime arm bridge verified`；夹取计数与改前同一份报告一致（left 61 / right 48） |
| `python -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --neutral-json outputs\tmp_realtime\live_long_neutral.json --frames 120 --headless --no-preview` | 0 | 同一份录制的 120 帧：报告里 `arm_travel_mode=clip` / `arm_lateral_guard_mm=0.0` / `arm_slew_mm_per_s=600` / `arm_slew_mm_last_frame=40.0`；夹取 359、**护栏 0**、斜坡生效调用 144（旧口径 263） |
| `python -m teleoperation.simulation.realtime_teleop --help` | 0 | 新开关 `--arm-slew-mm-s` 可解析、`%` 转义正确 |

**还没做 / 仍开着的问题**：帧 810-870 与 1455-1515 左臂残差尖峰（§8.8 第 4 条，与本轮无关）；
H1-2 中性工作点 ±0.05 m 撑不住（§8.6.1）；护栏关掉后前臂间距最小 2.21 mm 是下一个要盯的
（真撞上就给 `--arm-lateral-guard-mm 10`）；"论文那种肘姿态跟踪"仍未做（§8.7，见 §8.10）。

---

### 8.10 「只动大臂没反应 / 张不开怀抱」的根因与两条出口（2026-10-13 晚，现场反馈后）

现场三句话（原话要点）：**(a)** 三个机器人只能微微张开，不可能像人一样张开怀抱；
**(b)** 大臂基本不动，G1 与 GR1-T2 尤甚（G1"是不是胳膊坏了"）；**(c)** 双手握拳保持在
胸前不动、只动大臂和小臂，三个机器人都没反应 —— 怀疑是不是摄像头只有三个手臂采集点。

**先回答 (c) 的猜测**：三点数据**够用**（论文的肘任务也只需要方向），而且肘那一点本来就在
我们手上 —— 问题是历史口径 `chain = 腕 - 肩` **把肘那一点扔掉了**。下面几组测量把"输入侧"
与"映射侧"分开。

#### 8.10.1 测量一：目标层（不含 IK），`devtools/verify_arm_wing.py`

合成手臂流：`swing` = 握拳贴胸、只把肘绕「肩-腕」轴摆开 90 度（手不动）；`hug` = 张开怀抱
（横向 223 mm）。每台每侧量"机器人目标动了多少毫米"。

| 动作 | 操作者 | 机器人 | `chain` 目标 | `segments` 目标 | 行程上限 |
|---|---|---|---|---|---|
| swing | 肘动 326-358 mm、腕 0 mm | h1_2 | **0.0 mm** | 67.8 / 74.4 mm | 100 / 150 mm |
| swing | 同上 | gr1_t2 | **0.0 mm** | 8.0 / 10.3 mm | 100 / 120 mm |
| swing | 同上 | g1 | **0.0 mm** | 2.0 / 2.2 mm | 40 mm |
| hug | 腕动 223 mm | h1_2 | 100 / 150 mm | 100 / 150 mm | 贴上限 |
| hug | 同上 | gr1_t2 | 92.1 mm | 77.0 mm | 100 mm |
| hug | 同上 | g1 | 40.0 mm | 40.0 mm | **40 mm（正好贴死）** |

**这就是 (c) 的根因**：`chain` 口径下"只摆大臂"这一帧的目标位移**恒为 0**（六个组合全是 0.0 mm），
机器人当然一动不动。同时 `segments` 也**不是万能的**：它等于 `(R-I)(L_u*u + L_f*v)`，机器人两段
比例接近操作者时两项互相抵消 —— g1 实测上臂/前臂 0.193/0.184（人 0.30/0.28），所以只剩 2 mm；
h1_2 的 0.331/0.229 差得多，所以有 68-74 mm。

另一个硬数字：**g1 的每轴行程上限就是 40 mm**（`realtime_travel_m`），张开怀抱时目标正好贴在
它上面 —— 即使映射 100% 传递，g1 的两手也只能各走 4 cm。

#### 8.10.2 测量二：IK 层，`devtools/probe_elbow_task.py`（论文的"肘位置任务"）

PyBullet 有 `calculateInverseKinematics2`：一次交**两个**位置目标（腕 + 肘），对应论文的
`FrameTask(腕) + 肘位置任务`。四个变体（腕目标一样，只换肘目标）：

| 变体 | h1_2 上臂转角 | gr1_t2 | g1 | 说明 |
|---|---|---|---|---|
| `control`（腕固定、不给肘目标） | **0.01 度** | 0.04 | 0.01 | 复现现场 (b)(c)：7 自由度冗余链把解留在原地 |
| `elbow_task_first`（肘横移 100 mm） | 33.2 度 | 10.0 | 7.1 | 肩真的转起来了（肩关节 +0.12…+0.82 rad） |
| `consistent_first`（腕+肘同链，肘任务第一遍） | 29.0 度 | 16.2 | 11.9 | 两个目标同时可达 |
| `consistent_last`（同上，肘任务**最后一遍**） | 25.5 度 | 14.3 | 11.2 | 收尾不再把肘拉回去 |

两条结论：**只喂腕位置时整条臂纹丝不动（0.01 度）**，这是 (b) 的根因；把肘当第二个位置目标
交进去，肩就会转 —— 但两个目标必须**出自同一条按连杆长缩放的链**（否则一个摆开、一个不动，
前臂长度就得变，IK 只能两边各让一步：实测腕残差 107 mm、肘离目标 200+ mm）。

#### 8.10.3 测量三：真实录制端到端 A/B

用 `live_long_frames.npz` 的帧 400-520（右臂，120 帧 = 91 valid + 29 held；这一段三点全有限）：

| 口径 | h1_2 肩行程 | h1_2 上臂转角 | h1_2 肘跨距 | h1_2 腕残差 | g1 肩行程 | g1 腕残差 |
|---|---|---|---|---|---|---|
| `chain`（现值，只喂腕） | 49.93 度 | 34.09 度 | 189.2 mm | **6.38 mm** | 34.93 度 | 11.87 mm |
| `--arm-map segments --arm-elbow-task` | **83.98 度** | **57.18 度** | **299.4 mm** | 65.75 mm（超 30 天花板） | 31.72 度 | 27.61 mm |
| 同上 + `--arm-elbow-gain 0.25` | 84.55 度 | 67.89 度 | 348.9 mm | 52.05 mm | 31.72 度 | 27.60 mm |

**这一段录制里 h1_2 的手臂张开了约 68%**（肩行程 49.9 → 84.0 度、肘跨距 189 → 299 mm），代价是
腕残差 6.4 → 65.8 mm（超天花板）。`--arm-elbow-gain` 能把残差买回一些（0.25 → 52.05 mm），但对
**g1 无效**：它的肘目标先被自己的 ±40 mm 行程夹住，力度旋钮根本不起作用。**如实说**：PyBullet 的
两个位置目标等权、没有优先级，所以这是"手腕精度换肩部幅度"的连续折中，不是白拿 ——
现场用 `--arm-trace` + 报告里的 `residual_mm` 看着调。

#### 8.10.4 测量四：g1 的"张不开"不是行程夹的

同一段录制，`--arm-travel-mm 100`（每轴上限 40 → 100 mm）：

| g1 右臂 | 夹取帧数 | 肩行程 | 肘跨距 | 腕残差 |
|---|---|---|---|---|
| 上限 40 mm（现值） | 102 / 120 | 34.93 度 | 118.3 mm | 11.87 mm |
| 上限 100 mm | 25 / 120 | 35.82 度 | 123.0 mm | 12.29 mm |

上限确实夹得很狠（夹取帧 102 → 25），但**手臂动作几乎没变**（+0.9 度 / +4.7 mm）：这一段录制里
g1 其实跟着转了约 35 度，看不出"完全不动"。所以 (a)(b) 在 g1 上主要不是行程问题，而是
（i）那一段操作者动作幅度本身不大、（ii）现场那场的中性工作点/标定不同（§8.6.1）。

#### 8.10.5 输入侧：三点里有多少帧是"没有手臂"的（回答"是不是摄像头采集点的问题"）

对 `live_long_frames.npz` 直接数 NaN（这正是**上一场现场**的录制）：

| 范围 | 左臂三点全 NaN | 右臂三点全 NaN | 有效帧（三点全有限） | \|腕-肩\| < 150 mm（被 `MIN_OPERATOR_REACH_M` 拒） |
|---|---|---|---|---|
| 全段 1560 帧 | 193 帧（12.4%） | 279 帧（17.9%） | 1367 / 1281 帧 | 119 / 55 帧 |
| 前 120 帧 | — | **79 帧（66%）** | 41 帧 | 4 帧 |

三点只要有一个 NaN，`_frame_inputs` 就整帧拒收 → 桥保持上一帧（`held`）→ 机器人不动。
前 120 帧只有 10 帧真的产生了新目标（其余是 30 帧标定 + 79 帧无手臂 + 4 帧太近）。
**所以 (a)(b) 里确实有一部分是输入侧**：姿态模型在那些帧没检到手臂（人没进画面 / 手臂出框 /
遮挡 / 光照），以及"握拳贴胸"时 |腕-肩| 掉到 150 mm 以下被下限拒收。这一条用户猜对了一半 ——
但"只摆大臂零反应"是映射侧，已复现并给了出口。

#### 8.10.6 新开关（默认全关，行为与改前逐字一致）

| 开关 | 默认 | 作用 |
|---|---|---|
| `--arm-map {chain,segments}` | `chain`（**2026-10-14 起 CLI 默认改为 `posture`**，见 §8.15；库/桥的默认仍是 `chain`） | 位移口径：`segments` = 论文 `scale_joints`（上臂段 + 前臂段，各按「机器人连杆长 / 你的连杆长」缩放） |
| `--arm-elbow-task` | 关 | 把肘也当位置目标交给 IK（论文的肘任务），需要 `--arm-map segments` |
| `--arm-elbow-gain 0-1` | 1.0 | 肘目标的力度（残差与幅度的折中旋钮） |
| `--arm-elbow-pass {first,last}` | `last` | 肘任务放第几遍；`first` 会被腕姿态收尾拉回（肘误差 141-256 mm） |

启动时会打印实测连杆长与肘任务状态，例如
`arm map: segments | measured link lengths (m) g1:r 0.193+0.184 ... | elbow task on (pass last)`。

现场怎么试（先站定中性姿势，再只摆大臂）：

```powershell
# A: 现状（对照组）
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py
# B: 两段式 + 肘任务（--arm-elbow-gain 先试 1.0 / 0.5 / 0.25，盯报告里的 residual_mm）
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-map segments --arm-elbow-task
```

#### 8.10.7 验收（改动后，本机实测）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `$env:PYTHONPATH="$pwd\src"; python -m pytest tests -q` | 0 | `318 passed, 3 skipped, 300 subtests passed`（改前 311；新增 7 条两段式/肘任务用例） |
| `python devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符 |
| `python devtools\verify_arm_wing.py --report outputs\tmp_realtime\wing.json` | 0 | h1_2 两侧 `[OK]`（swing 目标 67.8 / 74.4 mm，chain 全 0.0）；g1 / gr1_t2 `[FAIL]`（两段比例太接近而互相抵消）—— 这条 FAIL 是**要留下的结论**，不是脚本坏了 |
| `python devtools\probe_elbow_task.py --report outputs\tmp_realtime\probe_elbow_task.json` | 0 | 6/6 `[OK]`：control 上臂 0.01-0.06 度 -> 肘任务 7.1-33.3 度 |
| `python devtools\run_realtime_arm_windows.py --input-npz outputs\tmp_realtime\live_long_frames.npz --headless --frames 120 --robot g1 --side right --arm-map segments --arm-elbow-task` | 0 | 端到端跑通，启动行 `arm map: segments ... g1:r 0.193+0.184 | elbow task on (pass last)`（新路径在真实录制上无异常） |
| 同上，输入换成 `outputs\tmp_realtime\window400_frames.npz`（帧 400-520 切片） | 0 | §8.10.3 那张表（h1_2 肩 49.93 -> 83.98 度 / 残差 6.38 -> 65.75 mm） |

**还没做的**（下一轮）：① 肘目标与腕目标在 PyBullet 里等权，想要"腕优先、肘兜底"得自己写加权
或两层 IK；② g1 / gr1_t2 的两段比例与操作者太接近，纯摆动仍会抵消，是否给它们单独一套权重需要
现场再量；③ `MIN_OPERATOR_REACH_M=150 mm` 的下限对"握拳贴胸"太严（那一场 119 / 55 帧被拒），
要不要改成按可见度判而不是按距离判。

### 8.11 方向口径 `--arm-map posture`（2026-10-13 深夜）

#### 8.11.1 为什么又换了一层

现场反馈（用户原话）：H1-2 抽抽、G1 大臂抬起来的幅度很微小、GR1-T2「基本上就很不一样」。把这三句
翻成数字，其实是同一件事：

- 位移口径的幅度天生挂在「中性姿势下 |腕 - 肩|」这根**短基线**上：双手握拳贴胸时它只有约 0.27 m，
  一次旋转作用在这么短的向量上，目标位移就小到看不见；
- 更硬的一层是 profile 的**逐轴行程上限**（g1 每轴 40 mm）。两段式算出的原始位移有 190 mm
  （60 度摆臂 x 0.193 m 上臂），被夹掉 4.7 倍 —— 「g1 幅度很微小」就是这么来的；
- 肘任务的两个位置目标在位移口径下会**各自**被夹断，夹断后互相不一致，IK2 只能一边满足一个、
  来回跳（H1-2「抽抽」的候选解释；另见 8.10 里 725 mm 那一行）。

`posture` 换一层：不搬位移，搬**方向**，长度用机器人自己量出来的连杆（每台每侧 FK 量一次，
见启动行 `measured links (m)`）：

    u = 你的上臂单位方向（过 axis_map，与位移口径同一条轴向约定）
    v = 你的前臂单位方向
    elbow_target = 肩锚点 + L_u(robot) * u
    wrist_target = elbow_target + L_f(robot) * v

三个直接后果（都可证伪）：① 你的上臂转 40 度，三台机器人都转 40 度，与臂长、站位、中性姿势
都无关；② 两个目标同时可达（`wrist - elbow` 正好等于机器人的前臂向量），IK2 不再互相拉扯；
③ **逐轴行程上限不参与**（目标本来就落在机器人自己那根「上臂 + 前臂」球面上），所以 g1 也能
给出全幅度动作。`--arm-posture-gain`（0-1，默认 1.0）是**夹角插值**：0.5 = 停在机器人自己的
中性方向与你的方向正中，0 = 回中性；它不是位移缩放。

#### 8.11.2 实测对比（同一段录像、同样 200 帧）

```powershell
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map segments --arm-elbow-task --report outputs\tmp_realtime\ab_segments.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --report outputs\tmp_realtime\ab_posture.json
```

| 机器人/侧 | 肩行程（度） | 肘点行程（mm） | 行程夹断帧数 | 末帧腕残差（mm） |
|---|---|---|---|---|
| h1_2 左 | 84.05 -> **124.95** | 444.8 -> **598.3** | 854 -> **0** | 73.1 -> 110.4 |
| h1_2 右 | 67.80 -> **118.51** | 387.3 -> **585.0** | 同上 | 54.5 -> 150.3 |
| gr1_t2 左 | 74.66 -> **111.11** | 297.9 -> **424.2** | 同上 | 73.5 -> 223.4 |
| gr1_t2 右 | 67.33 -> **91.18** | 279.3 -> **374.6** | 同上 | 61.2 -> 242.4 |
| g1 左 | 35.77 -> **158.53** | 112.2 -> **387.7** | 同上 | 36.3 -> 239.8 |
| g1 右 | 23.44 -> **93.59** | 76.7 -> **297.4** | 同上 | 6.8 -> 77.7 |

左起第二列就是这一轮的目的：g1 左肩 35.8 -> 158.5 度、肘点 112 -> 388 mm，正是「大臂抬不起来」
的直接修复；`arm_clamped_frames` 854 -> 0 说明那个夹子不再是主角。

**如实说没解决的**：末帧腕残差是**报告写出那一刻的单帧快照**（`arm_scene_driver.residuals_mm`
= 最后交给 IK 的那个点与实测 EE 的距离），不是整段分布，两个口径各只有一个数，所以最后一列只能
当「这一帧 IK 解到位没有」看。方向口径下它变大（77-242 mm）有两种可能：① 你的方向把目标指到了
机器人关节限位到不了的锥面外；② 末帧恰好是快速运动里的一帧（目标每帧上限 40 mm，IK 追不上）。
**下一轮要么记逐帧残差分布，要么按 ① 的假设给目标加可达性投影。** 另外 g1 两前臂最小间距
2.98 mm（幅度变大后会在中线附近靠近），而横向护栏默认是 0（关）：要挡就加
`--arm-lateral-guard-mm 30`，代价是「往中线那半程」被削掉（见 8.9）。

#### 8.11.3 现场怎么看

```powershell
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-map posture
```

启动行应当出现（2026-10-13 深夜本机实测）：

```
arm map: posture | measured links (m) h1_2:l 0.331+0.229 h1_2:r 0.331+0.229 gr1_t2:l 0.251+0.251 gr1_t2:r 0.247+0.250 g1:l 0.193+0.184 g1:r 0.193+0.184 | elbow task: built by the direction recipe (pass last)
arms: h1_2, gr1_t2, g1 | ik_mode {...} | orientation off | travel limit {'g1': {'left': 0.04, ...}} m (clip) | lateral guard 0.0 mm | ...
```

`travel limit` 那段在方向口径下**不再生效**（g1 的 0.04 只是打印出来对照），每帧的 `info` 里
`travel_limit_applied` 为 `false`、`elbow_target` 为 `"posture"`，并带 `operator_upper_deg` /
`robot_upper_deg` 一对数：gain=1 时两者接近，差的是一份**固定偏置**（「机器人加载姿势的方向」
与「你中性方向」之间，本机实测约 8 度）。要看这一帧的具体数字：加 `--arm-trace` 或直接看
`--report` 的 `arm_last_targets`。

标定仍然是必需的：站定中性姿势约 `--arm-calibrate-frames`（默认 30）帧，两侧都 `cal=ok` 之后
手臂才开始跟手；在那之前 `cal=left=n/30 right=n/30` 且手臂停在中性工作点。

#### 8.11.4 验收（改动后，本机实测）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `python -m pytest tests -q` | 0 | `327 passed, 3 skipped, 306 subtests passed`（改前 318；新增 9 条方向口径用例） |
| `python devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符 |
| 8.11.2 那两条 `--headless` 命令 | 0 / 0 | 启动行分别为 `arm map: segments ...` 与 `arm map: posture ...`；夹断帧数 854 -> 0 |
| `python devtools\run_realtime_arm_windows.py --arm-map posture` | 运行中 | 启动行 `arm map: posture ... | elbow task: built by the direction recipe (pass last)`；窗口 fps 7.5 |

**下一轮待办**：① 逐帧残差分布（判断 8.11.2 最后一列到底是不是「追不上」）；② 要不要用
「相对你中性方向的旋转」消掉那份固定偏置（现在绝对方向口径下它是个常数，约 8 度）；
③ h1_2 的 `ik_mode="pose"` 与方向口径同时用时姿态收尾会不会把腕拉走（本轮用的是默认
`orientation off`）；④ `devtools/verify_arm_wing.py` 还没加 `posture` 模式（它只比 chain / segments）。


### 8.12 「前臂弯曲方向和我的相反 / 手交界处抽搐」：支路陷阱与关键点跳变（2026-10-13 深夜第二轮）

#### 8.12.1 现场第二轮反馈（原话）

> 现在三个机器人的大臂抬起和落下还行，比之前好很多。但是 GR1-T2 这个机器人的手臂弯曲的方向和我的
> 手臂弯曲方向是相反的。H1-2 手臂和手掌交界的地方有时候也会抽搐。

两句都要变成**数字**才有得改。本轮先补了「要的方向 vs 走到的方向」这个逐帧仪表（§8.12.2），
再看它指向哪。上一轮 §8.11.2 的最后一列（末帧腕残差）当时只有一个快照，正是这次要补的那一块。

#### 8.12.2 新增的逐帧仪表（报告里直接能读）

`arm_ik.record_posture` 现在把**这一帧姿势对应的命令**一起记下来（`command_elbow` /
`command_wrist`，基座系），`posture_summary` 于是能给出几个新数（每个机器人每侧）：

| 字段 | 含义 |
|---|---|
| `upper_dir_error_deg` (median/p90) | 这一帧**上臂方向**与要求方向的夹角（肘点任务的达成度） |
| `forearm_dir_error_deg` (median/p90) | 这一帧**前臂方向**与要求方向的夹角 —— 现场那句「弯曲方向相反」就是它 |
| `forearm_dir_dot` {X,Y,Z} | dot(走到的, 要的) 在**基座系**三个轴上的中位数：哪一轴为负就是哪一轴反了 |
| `forearm_opposite_fraction` | 夹角 > 90 度的帧占比（1.0 = 整场都在反向） |
| `upper_struggle_fraction` | 肘方向误差 > 30 度的帧占比 |
| `worst_frame` | 前臂误差最大的那一帧的**原样向量**（要的 / 走到的，上下臂各一对） |
| `elbow_recoveries` / `branch_rejections` | 8.12.4 那条支路恢复救回 / 拒绝的次数 |

同一段录像 `outputs/tmp_realtime/live_long_frames.npz`、200 帧，改前（= §8.11 那一版）的原样数字：

| 机器人/侧 | 上臂误差 中位（度） | 前臂误差 中位（度） | 前臂反向帧占比 | dot X / Y / Z |
|---|---|---|---|---|
| h1_2 右 | 5.07 | 6.77 | 0.000 | +0.23 / +0.01 / +0.38 |
| g1 右 | 8.11 | 5.36 | 0.000 | +0.25 / -0.00 / +0.63 |
| h1_2 左 | 21.07 | 30.34 | 0.017 | +0.42 / +0.13 / +0.08 |
| g1 左 | 56.98 | 39.93 | 0.000 | +0.34 / +0.08 / +0.05 |
| **gr1_t2 左** | **50.77** | **72.78** | **0.339** | +0.04 / +0.08 / +0.09 |
| **gr1_t2 右** | **37.66** | **50.11** | **0.108** | **-0.065** / -0.00 / +0.73 |

两件事一眼可读：

1. h1_2 右臂与 g1 右臂是**好的**（5-8 度）——所以整条链路本身没问题，误差不是「映射错了」；
2. gr1_t2 两侧都是坏的，而且 34% / 11% 的帧前臂与要求方向**夹角超过 90 度**（右臂 X 轴上
   `dot` 中位数是负数 = 反了）——现场那句「弯曲方向相反」在这张表里就是第 4 列。

`worst_frame`（gr1_t2 左，第 184 个采样帧）的原样向量说明了它是**怎么**反的：

```
要的上臂方向 (-0.14, +0.91, -0.39)   走到的是 (+0.88, +0.31, +0.37)
要的前臂方向 (+0.79, -0.34, +0.52)   走到的是 (-0.06, +0.37, -0.93)
肩 (0.075, 0.101, 0.387)  肘目标 (0.040, 0.330, 0.289)  腕目标 (0.220, 0.252, 0.407)
```

肘目标离**当前**肩正好 0.251 m（= 它的上臂长，可达），可走到的肘却甩到了腕目标的**前面**
（`achieved_upper` 指向 +X 前方）——这是「肘落在肩-腕连线的另一侧」的**反向支路**，不是角度差一点。

#### 8.12.3 根因：IK 会**卡在**反向支路里（warm start + restPoses 正反馈）

PyBullet 的 IK 从**当前**关节角出发，代价里除了位置还有「离 `restPoses` 多远」。某一帧把肘送进
反向支路之后：那个解离当前姿势**更近**，于是下一帧又从它出发、再解回它；位置那一项差出来的距离
补不上姿势代价，于是**每一帧都从坏支路再解一次，再也出不来**。gr1_t2 左臂 91% 的帧肘方向误差
> 30 度（`upper_struggle_fraction` = 0.909）、34% 的帧前臂反向，就是这么来的。

h1_2 右臂只有 5 度是因为它从没进那条支路。所以这不是「目标错了」，是**求解器的路由问题**：
同一份目标、不同的起始姿势，落在两个不同的解上。

第二个症状（H1-2 手臂与手掌交界处抽搐）来自**输入侧**，同一份录音直接量得出来
（`outputs/tmp_realtime/check_arm_jitter.py`，逐帧方向变化，度/帧）：

| 侧 | 中位 上臂/前臂 | p90 | 最大 |
|---|---|---|---|
| 左 | 4.69 / 5.90 | 21.4 / 34.4 | **153.2 / 153.8** |
| 右 | 4.89 / 6.92 | 21.0 / 35.4 | **116.3 / 161.8** |

中位 5 度、最大 150+ 度：**最后那一档是 MediaPipe 的关键点跳变，不是你的手**（10 fps 下 150 度/帧
= 1500 度/秒，是真人肩关节峰值的 3 倍）。方向口径 1:1 搬方向，跳变就直接变成「一帧里把肘转
150 度」的请求 —— IK 只能翻支路或者抽一下，位置就在腕/手交界处。

#### 8.12.4 三条改动（每条都能单独关掉做 A/B）

| # | 改动 | 默认 | 关掉它 |
|---|---|---|---|
| ① | **关键点跳变护栏**：单帧方向变化超过 600 度/秒（按这一帧**实际时长**换算）就保持上一帧，拐角度数写进那一帧 `info.operator_direction_glitch_deg`；同一个「太快」的方向**连续** 3 帧还是它，就放过去（`GLITCH_HOLD_FRAMES=3`，防止手臂卡死在旧方向） | 开 | `--arm-direction-rate-deg-s 0` |
| ② | **肘支路恢复**：当前解的「前臂方向误差 + 0.5x上臂方向误差」超过 25 度时，把手臂送回**加载姿势**（两个支路在这里几乎重合）再解一遍同一条序列，谁好要谁；同时设腕残差护栏（不许拿「方向好一点、手却跑掉」的解换） | 开 | `--arm-elbow-recovery-deg 0` |
| ③ | **目标跟随当前肩**：肘/腕目标按「当前肩 FK - 加载时肩锚点」整体平移，目标于是永远落在「离当前肩正好一根上臂」的球面上（躯干被手臂反作用力推动时这一条才起效，本轮量到过 72 mm） | 开 | `--arm-no-shoulder-follow` |

①的 600 度/秒不是拍的：真人肩峰值约 500 度/秒，录音最大 150 度/帧（约 1500 度/秒）超了 3 倍。
按帧长换算，所以 8-12 fps 的现场与 30 fps 的名义帧率不会得到两种行为（测试里给 0.25 s 的帧长，
10 fps 才不会被误判成跳变）。

②的判据第一版是「肘毫米 + 0.25x腕毫米」，实测**它救对了 g1 左臂（56.98 -> 16.21 度）却把
gr1_t2 弄得更差（前臂 72.8 -> 85.5 度、反向帧占比 0.339 -> 0.488）**：毫米可以被「肘近 30 mm、
腕近 30 mm」同时满足而方向仍然不对。改成**方向**判据后两个数同一条，现场反馈的那一项就是判据本身。
第二版（只有方向、没有腕残差护栏）又把 h1_2 左臂的腕残差从 139 mm 抬到 457 mm、两前臂最小间隙
掉到 5.18 mm —— 那是操作者能看见的坏账，所以补了腕残差护栏（报告字段 `branch_rejections`）。

#### 8.12.5 实测对比（同一条命令、同样 200 帧；三条全开 vs 全关）

```powershell
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --report outputs\tmp_realtime\ab_all_on.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-no-shoulder-follow --arm-elbow-recovery-deg 0 --report outputs\tmp_realtime\ab_baseline.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-direction-rate-deg-s 0 --report outputs\tmp_realtime\ab_noglitch.json
```

| 机器人/侧 | 上臂误差（度）改前 -> 改后 | 前臂误差（度）改前 -> 改后 | 前臂反向帧占比 改前 -> 改后 | 腕残差（mm）改前 -> 改后 |
|---|---|---|---|---|
| h1_2 左 | 21.07 -> 23.31 | 30.34 -> 30.31 | 0.017 -> 0.033 | 138.9 -> 148.3 |
| h1_2 右 | 5.07 -> 5.06 | 6.77 -> 6.77 | 0.000 -> 0.000 | 172.1 -> 171.0 |
| **gr1_t2 左** | 50.77 -> 51.14 | 72.78 -> **69.93** | 0.339 -> 0.364 | 248.7 -> 247.7 |
| gr1_t2 右 | 37.66 -> 37.66 | 50.11 -> 50.12 | 0.108 -> 0.108 | 274.7 -> 273.9 |
| **g1 左** | 56.98 -> **23.44** | 39.93 -> **22.50** | 0.000 -> 0.025 | **245.6 -> 97.1** |
| g1 右 | 8.11 -> 8.12 | 5.36 -> 5.08 | 0.000 -> 0.000 | 86.9 -> 124.4 |

两前臂最小间距（`arm_forearm_gap_mm.min_mm`）：h1_2 69.38 -> 68.89、gr1_t2 **35.41 -> 42.83**、
g1 **7.20 -> 19.79**（§8.11.2 里 g1 那 2.98-7.2 mm 的「两前臂贴中线」是这条改动的直接受益者）。

改后的启动行（现场照着核对）：

```
arm map: posture | measured links (m) h1_2:l 0.331+0.229 ... g1:r 0.193+0.184 | elbow task: built by the direction recipe (pass last) | elbow branch recovery: >25 deg
```

**如实说没解决的**：gr1_t2 两侧仍是 38-51 度（表里唯一没修好的一行），前臂反向帧占比仍有
0.36 / 0.11。三条改动里对它有效的是①：把①的「连续拦 3 帧就放开」去掉（= 硬拦，实测 123 个有效帧
里拦下 37 帧）时，它的前臂误差掉到 44.4 度、反向帧占比 0.083 —— 但同一改动把 g1 左臂从 17.6 度
抬到 43.3 度。一份参数同时服务两台机器人的需求，说明 **gr1_t2 的问题不只是输入抖动**：它自己的
IK 收敛/限位还有一层没查（下一轮：单独扫 `--arm-ik-iterations` / `--arm-ik-residual-threshold`，
并给目标加「可达性投影」）。h1_2 左臂（21 -> 23 度）与 g1 右臂腕残差（86.9 -> 124.4 mm）也只是
没变坏，不算白拿的改善。

#### 8.12.6 现场怎么看

```powershell
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-map posture
```

- 启动行多一段 `elbow branch recovery: >25 deg`；若显示 `off` 说明有人用 `--arm-elbow-recovery-deg 0`
  关掉了它；
- 报告里看 `arm.posture.<robot>.<side>`：`forearm_opposite_fraction` 应当**接近 0**，
  `upper_dir_error_deg.median` 越小越像；`worst_frame` 给出最差那一帧的两个方向向量，可以直接读；
- 抽搐还在的话看 `bridge.counts.direction_glitch`（被护栏拦下的帧数）与每帧
  `info.operator_direction_glitch_deg`：如果拦下的帧里拐角常年 100 度以上，那是摄像头/光照下的
  关键点质量问题 —— 护栏只能不让它搬给机器人，修不了输入本身。

#### 8.12.7 验收（改动后，本机实测）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `python -m pytest tests -q` | 0 | `333 passed, 3 skipped, 306 subtests passed`（改前 327；新增 6 条跳变护栏用例） |
| `python devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符（**坑**：PowerShell 的 `Set-Content -Encoding UTF8` 会写 BOM，本轮踩到过，改完要复查） |
| 8.12.5 那三条 `--headless` 命令 | 0 / 0 / 0 | 启动行含 `elbow branch recovery: >25 deg`；g1 左臂前臂误差 39.9 -> 22.5 度、腕残差 245.6 -> 97.1 mm；g1 两前臂最小间距 7.2 -> 19.8 mm |
| `python devtools\run_realtime_arm_windows.py --arm-map posture` | 运行中 | 启动行多出 `elbow branch recovery: >25 deg` |

**下一轮待办**：① gr1_t2 单独的 IK 收敛/限位排查（扫 iterations 与 residual；给目标加可达性投影）；
② 现场判定本轮三条改动（幅度 / 像不像 / 抽不抽）；③ 左手普遍比右手差（h1_2 左 21 度 vs 右 5 度）
是输入质量还是映射的左右不对称 —— 需要一份**左右对称的**动作录制来分离；④
`devtools/verify_arm_wing.py` 补 `posture` 模式（仍未做）；⑤ §8.11.4 的②（用「相对你中性方向的
旋转」消掉那约 8 度的固定偏置）。

### 8.13 「前臂还是反的」的几何根因：肩锚点不是肘的父连杆 + IK 只该动这条手臂（2026-10-14）

#### 8.13.1 现场第三轮反馈（原话）

> GR1-T2 前臂弯曲的方向还是和我相反；H1-2 和 GR1-T2 的整条手臂有时候会抽。

第一句就是 §8.12 末尾自己留下的那一行没修好（「gr1_t2 两侧仍是 38-51 度、反向帧占比 0.36 / 0.11」），
第二句是本轮的新症状。两条都在**求解层**（输入没动、映射没动）：同一段录像、同一条命令
（`outputs/tmp_realtime/live_long_frames.npz`，200 帧，`--arm-map posture`）改前改后即可复现，
不需要重新采集。

#### 8.13.2 根因一：`arm_points()["shoulder"]` 是按**名字**匹配的，它会随 shoulder_roll 摆动

`arm_points(side)` 里的 `"shoulder"` 取的是「连杆名里第一个含 shoulder 的连杆原点」：
h1_2 / g1 上它是 `shoulder_pitch_link`，gr1_t2 上它是 `upper_arm_pitch_link` ——
**都不是肘的父连杆**。`shoulder_roll` 一转（现场那 40-90 度），这个点就跟着摆动；而
`arm_points()["upper"]`（= h1_2/g1 的 `shoulder_yaw_link`、gr1_t2 的 `upper_arm_yaw_link`，
肘关节的父连杆）与肘**刚性相连**，距离与姿势无关。

`devtools/probe_arm_segment_rigidity.py`（本轮新增的探针，直接读 PyBullet 的 link state）:

| 机器人/侧 | 名字匹配的"肩"连杆 | shoulder_roll 转 100 度后它动了 | 肘父连杆原点动了 | 名字口径算出的上臂长 | 刚体口径 = profile 内置值 |
|---|---|---|---|---|---|
| h1_2 左/右 | `shoulder_pitch_link` | **-61.92 mm** | 0.00 mm | 331.1 mm | 185.05 mm |
| gr1_t2 左 | `upper_arm_pitch_link` | **+101.58 mm** | 0.00 mm | 250.8 mm | 178.67 mm |
| gr1_t2 右 | `upper_arm_pitch_link` | **+101.69 mm** | 0.00 mm | 246.7 mm | 178.67 mm |
| g1 左/右 | `shoulder_pitch_link` | **-48.53 mm** | 0.00 mm | 192.9 mm | 82.05 mm |

两列最右值得盯一下：**刚体口径量出来的上臂长与 profile 里写死的内置长度一模一样（差 0.00 mm）**，
名字口径则多出 72-146 mm（h1_2 331.1 vs 185.05 差 +146.1；gr1_t2 250.8 vs 178.67 差 +72.1；
g1 192.9 vs 82.05 差 +110.9）。也就是说历史上被当成「上臂」的那根骨头，
其实是从「上臂 pitch 连杆原点」量到肘，而不是从**肩关节中心**量到肘。

同一个探针的 `recipe_roll_test`：把 shoulder_roll 拧 +60 度，按某个锚点口径造出配方的那对目标
（`anchor + L_u * u`、`+ L_f * v`，`L` 用加载时量的常数），再看这对目标与真实几何差多少
（`unreachable_by_mm` = 「目标用的连杆长」与「当前锚点到肘的真实距离」之差；+ = 把手臂当成比实际长，
- = 当成比实际短）：

| 机器人/侧 | 老锚点（"shoulder"）差多少 | 新锚点（"upper"）差多少 |
|---|---|---|
| h1_2 左 | **+51.5 mm** | -0.0 mm |
| h1_2 右 | **-56.3 mm** | -0.0 mm |
| gr1_t2 左 | **+81.1 mm** | +0.0 mm |
| gr1_t2 右 | **-83.2 mm** | +0.0 mm |
| g1 左 | **+29.0 mm** | -0.0 mm |
| g1 右 | **-39.2 mm** | +0.0 mm |

±29-83 mm 的偏差**两个方向都有**（有的侧把手臂当成太长、有的侧当成太短），这解释了现场为什么
两个机器人不一样：gr1_t2 的锚点摆动最大（102 mm），它就是最坏的那一行。选一个新锚点不需要
调参 —— 只有「与肘刚性相连」的那个点能让「肩 + L_u*u」同时落在可达壳面上。

一个反直觉、也是 §8.12 的支路恢复治不了它的原因：**老口径下这对目标的残差反而很小**
（`recipe_roll_test` 同一变体的肘/腕兑现 5-10 mm）。目标本身被搬到了几何上不一致的地方，
求解器把一个「错的目标」解得很准，于是肘和腕都到了位、方向却是错的 —— 这正是现场看到的
「前臂弯曲方向相反」。看残差找不出这个问题，只有量锚点刚性才能。

#### 8.13.3 根因二：PyBullet 的 IK 把**整机**所有可动关节当变量

`arm_ik.solve` 走的是 `p.calculateInverseKinematics(body, end_effector, ...)`，它把
**所有可动关节**（髋、腿、手指、另一条手臂）都当自由度，`restPoses` 再把它们往加载姿势拉。
于是给手臂的那两个位置任务被别的关节分掉一部分，手臂上剩下的解自然差一截。探针的
`pose_tracking_test`（目标 = 机器人自己一条合法姿势的 FK，**精确可达**，只差 35 度摆放）:

| 机器人/侧 | 整机 / 60 迭代 肘,腕(mm) | 只动本侧手臂 / 60 迭代 | 只动本侧手臂 / 800 迭代 |
|---|---|---|---|
| h1_2 左 | 79.27, **237.88** | 77.42, **27.70** | 134.20, 248.93 |
| h1_2 右 | 126.39, 61.98 | 111.72, 41.67 | 105.58, 86.86 |
| gr1_t2 左 | 158.93, 139.05 | 127.76, 102.52 | 365.80, 197.24 |
| gr1_t2 右 | 145.50, 89.25 | 128.99, **11.64** | 140.54, 46.59 |
| g1 左 | 49.50, **187.26** | 34.11, **23.13** | 81.88, 187.62 |
| g1 右 | 20.92, 75.34 | 21.82, **19.50** | 90.03, 169.59 |

三列读出来的三件事（本轮实测，同一条命令的 `--arm-ik-scope arm/body`、`--arm-ik-iterations`）：

1. 锁住非本侧手臂之后**腕残差全线下降**（237.88 -> 27.70、187.26 -> 23.13、89.25 -> 11.64）；
2. **肘残差只降一点点**（79 -> 77、159 -> 128、50 -> 34、21 -> 22）—— 说明肘那一项还有一层
   （目标帧口径 / 关节限位）没查，见 §8.13.8；
3. **迭代数从 60 加到 800 一律更差（6/6 侧）**：PyBullet 的阻尼最小二乘在这里过冲，
   「多迭代一会儿就好了」这个直觉不成立。

第二个自检 `reachable_pair_test`（从拧开 50 度的姿势解一对精确可达的目标）复核同一件事：
名字口径 28.30 / 23.81 mm（h1_2 左）-> 刚体口径 1.97 / 0.96 mm；gr1_t2 左 146.14 / 71.58
-> 71.52 / 39.31；g1 左 61.32 / 58.40 -> 7.52 / 6.11（两面都变的只有 h1_2 右与 g1 右的腕，
分别 5.75 -> 8.48、11.90 -> 15.48 mm，如实记下）。

#### 8.13.4 三条改动（每条都能单独关掉做 A/B）

| # | 改动 | 默认 | 关掉它 |
|---|---|---|---|
| ① | **肩锚点换成肘的父连杆**：`arm_points()["upper"]`（刚体）同时用于配方的肘/腕目标、`_shoulder_shift` 的平移量、`verify_arm_wing.py` 的臂长口径 | `upper` | `--arm-anchor shoulder` |
| ② | **IK 只动本侧手臂**：求解前把其余所有关节（髋、腿、手指、另一条手臂）锁在当前角度（±1e-6 rad，`arm_ik._locked_arguments`），求解完只写本侧手臂 | `arm` | `--arm-ik-scope body` |
| ③ | **支路恢复加滞回与冷却**：新支路要比当前支路好 `ELBOW_RECOVERY_MARGIN_DEG=8` 度才换，换过之后 `ELBOW_RECOVERY_COOLDOWN_FRAMES=12` 帧内不再试 | 开 | `--arm-elbow-recovery-margin-deg 0 --arm-elbow-recovery-frames 0` |

①②是几何/求解口径的**修正**（不是调参）：锚点只有「与肘刚性相连」的那一个点才让配方目标落在可达
壳面上；IK 的自由度只有收在本侧手臂上，两个位置任务才不被身体分掉。③治的是本轮的第二个症状
（「整条手臂会抽」）：1 度门槛时两支几乎一样，每帧来回换支路（下面 A/B 里 200 帧试了 68 次），
肉眼就是整条手臂抽一下。

三个开关的默认值在 `realtime_teleop.build_parser()`，现场启动器
`devtools/run_realtime_arm_windows.py` 把多余参数原样转给同一个 parser，所以它自动继承。
`PyBulletArmIK` / `ArmSceneDriver` 的**类**默认值仍是历史行为（`lock_others=False`、`shoulder`
可选），理由：`devtools/verify_*.py`、`tests/` 是回归基准，数字要能对得上老记录；只有实时入口
按现场结论改默认。想让别的调用方也用修好的口径，显式传 `lock_others=True` / `shoulder_anchor_key="upper"`。

#### 8.13.5 实测对比（同一条命令、同样 200 帧）

```powershell
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --report outputs\tmp_realtime\ab5_lock_upper.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-ik-scope body --arm-anchor shoulder --report outputs\tmp_realtime\ab3_fix.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-ik-scope body --report outputs\tmp_realtime\ab4_anchor_upper.json
E:\python3.11.7\python.exe src\teleoperation\simulation\realtime_teleop.py --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-anchor shoulder --report outputs\tmp_realtime\ab7_arm_shoulder.json
```

四格（锚点 x IK 范围，其余全默认；读表脚本 `devtools/report_table.py`）：

| IK 范围 / 锚点 | 前臂反向帧占比（6 侧里 >0 的侧数 / 最差） | 前臂方向误差（6 侧均值 / gr1_t2 左） | 腕残差（6 侧均值 / gr1_t2 左） |
|---|---|---|---|
| 整机 / `shoulder`（历史口径） | 4 侧 / **0.339** | 31.14 / 72.79 | 52.80 / 98.11 |
| 整机 / `upper`（只改锚点） | 3 侧 / 0.099（重跑 2 侧 / 0.083） | 27.82 / 65.96 | 57.50 / 134.56 |
| 本侧手臂 / `shoulder`（只锁关节） | 3 侧 / 0.231 | 27.68 / 64.55 | 18.95 / 20.49 |
| **本侧手臂 / `upper`（新默认）** | **0 侧 / 0.000** | **21.29 / 44.74** | **17.91 / 21.17** |

新默认这一行跑了 **3 次**（`ab5_lock_upper` / `ab8_default_rep1` / `ab9_default_rep2`），
反向帧占比 6/6 侧全 **0.000**、腕残差 21.17 / 21.17 / 21.18 mm、gr1_t2 左前臂误差
44.74 / 44.74 / 44.73 度 —— 也就是「不再反着弯」和「腕到位」这两件事逐次可复现。
另外三个口径也各重跑过一次（`ab12` / `ab13` / `ab14`，与 `ab3_fix` / `ab4` / `ab7` 对得上：
反向帧占比 0.339 / 0.099 / 0.231 **逐字相同**），核实它们并没有偶然变好。
需要说明的是**前臂方向误差中位与 jerk 峰值逐次有 ±20 度量级的抖动**（warm start 路径相关，
同一配置两次跑可以差一倍：h1_2 左 29.87 -> 49.39 度），所以上面那张表里这两列是其中一次的样本，
只有「反向帧占比、腕残差」两列可以当硬指标看。整机 / `upper` 也重跑过一次（`ab6_body_upper`）：
反向帧占比 0.099 -> 0.083，仍在 gr1_t2 左 / 右存在，说明**只改锚点不够**。

两个症状分开看：

- 前臂方向（现场第一句）：6 侧里 5 侧变好（h1_2 左 29.87 -> 16.44 度、gr1_t2 左 72.79 -> 44.74、
  gr1_t2 右 50.12 -> 39.01、g1 左 21.95 -> 13.23、h1_2 右 6.76 -> 6.71），
  只有 g1 右从 5.36 微涨到 7.61 度。**反向帧占比 6/6 侧归零**是这一轮的核心结论。
- 手臂抽搐（现场第二句，③的 A/B，其余全默认）：

| 指标 | 1 度门槛 / 无冷却 | 8 度滞回 / 冷却 12 帧（默认） |
|---|---|---|
| 支路恢复次数（6 侧合计） | 68 | **3** |
| 被护栏拒掉的恢复（6 侧合计） | 436 | **47** |
| 上臂 jerk 最大（g1 左） | 62.51 度 | **39.09** |
| 肘 jerk 最大（g1 左） | 192.61 度 | **125.94** |
| 肘 jerk 最大（gr1_t2 右） | 52.33 度 | 84.39（**变差**） |
| 肘方向误差中位（h1_2 左） | 18.22 度 | 21.89（**变差**） |
| 肘任务中位（gr1_t2 左） | 161.77 mm | 174.11（**变差**） |

  即：滞回 + 冷却把「每帧换支路」压到 3 次、抖动峰值明显下降，代价是两支接近时不再抢
  「更优的那一支」，h1_2 左与 gr1_t2 左的肘误差各涨 3.7 度 / 12 mm。这是一个**取舍**，
  不是纯赚：现场如果觉得「抽」比「差几度」难受，就用默认；反之
  `--arm-elbow-recovery-margin-deg 1 --arm-elbow-recovery-frames 0` 回到灵敏档。
  两个方向都留了开关，且都在报告里可读。

#### 8.13.6 现场怎么看

```powershell
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-map posture
```

启动行现在多出两段（一行行核对，对不上就是默认没生效）：

```
arm map: posture | shoulder end = upper (rigid: the elbow parent link) | ik scope = arm |
measured links (m) h1_2:l 0.185+0.229 h1_2:r 0.185+0.229 gr1_t2:l 0.179+0.240 ... |
elbow task: built by the direction recipe (pass last) |
elbow branch recovery: >25 deg (margin 8 deg, retry every 12 frames)
```

- `shoulder end = upper` + `rigid: the elbow parent link` = ①生效。量出来的上臂长应当是
  **0.185 / 0.179 / 0.082** 这一组（= profile 内置值）；出现 `shoulder` / `historical name-matched link`
  就是回到了老口径。
- `ik scope = arm` = ②生效；`body` = 历史行为。
- `elbow branch recovery: >25 deg (margin 8 deg, retry every 12 frames)` = ③生效。

现场判据（沿用 §8.12.6 的动作）：

1. **前臂方向**：抬到肩高、屈肘再伸肘各两次。GR1-T2 与 H1-2 的前臂应当和你的同向
   （本轮之前 GR1-T2 是反的）。
2. **不再抽**：同一动作重复 10 次，肩/肘不应出现 1-2 帧的抽动；还在抽就先看报告里
   `elbow_branch_recoveries` 与 jitter 列，再决定加 margin 还是 `--arm-elbow-recovery-frames 0` 回到灵敏档。
3. 两类反馈都不要用「残差小」当通过标准：§8.13.2 已经证明**残差小也可能方向反**
   （老口径下求解器把错的目标解得很准）。判据只有「方向对不对、抽不抽」。

一句口径提醒：`--arm-map posture` 的肘/腕目标直接建在锚点上，是本轮的主战场；
`--arm-map segments`（位移口径）不建目标，但也吃同一个锚点（肩跟随的平移量，
见 `ArmSceneDriver._shoulder_shift`），所以两个 map 都会受 `--arm-anchor` 影响。

#### 8.13.7 验收（改动后，本机实测）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | **0** | `RESULT: [OK] all checks passed (2 warning(s))`；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；第 5 项 `nothing forbidden and no big asset staged`；第 8 项 `Ran 340 tests in 80.9s [OK (skipped=3)]`（改前 336，§8.12.7 的 `333 passed + 3 skipped` 同源；本轮新增 4 条）+ 4 条 CLI 冒烟 exit 0 |
| `python devtools\check_gbk_safe.py --strict` | **0** | 未发现 GBK 无法编码的字符 |
| `python devtools\check_no_c_drive.py --strict` | **0** | 结论 `[OK]`，无 C 盘写入隐患（preflight 第 3 项） |
| 8.13.5 的四条 `--headless` 命令（默认 + 三个回退口径） | **0 / 0 / 0 / 0** | 默认那一行跑三次：6/6 侧前臂反向帧占比 0.000；启动行含 `shoulder end = upper (rigid: the elbow parent link)` 与 `ik scope = arm` |
| `python -m unittest discover -s tests -p test_arm_ik_multi_robot.py` | **0** | `Ran 28 tests ... OK`（改前 24：新增 `FieldArmDefaultsTests` 2 条 + `RigidShoulderAnchorTests` 2 条） |
| `python devtools\probe_arm_segment_rigidity.py` | **0** | 锚点刚性：刚体口径「肘到锚点」185.1 mm（sweep 0.0 mm）且与 profile 内置值一致，名字口径 331.1 mm（sweep 61.9 mm）；`lock=on` 把 h1_2 左腕残差 237.9 -> 27.7 mm；锚点自检 28.3 / 23.8 -> 2.0 / 1.0 mm |

**跑给队友看的一条**（本分支的基线是 `d8bca93`，不是 `preflight.ps1` 的默认基线 `10ebd9f`；
后者会把队友线 45 个提交算成「改了队友文件」，第 4 项出现 49 个假 `[FAIL]`，见 §8.3 表后的说明）：

```powershell
cd F:\simulation_platform_cs
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93; $LASTEXITCODE   # 期望 0
```

#### 8.13.8 如实说：这一轮没解决什么

- **肘残差仍有几十毫米**：`pose_tracking_test` 里那是**精确可达**的目标（就是机器人自己 FK 出来的
  一条合法姿势），只动本侧手臂时肘仍差 34-128 mm（腕能到 12-27 mm）。下一轮要查两件事：
  肘任务的世界点与腕任务的 link 帧点是不是同一口径，以及关节限位是否把那条合法姿势排除了。
  注意「加迭代」不是办法 —— 60 -> 800 迭代 6/6 侧全部更差。
- **前臂方向误差没有归零**（gr1_t2 左 44.7 度中位）。反向帧占比 6/6 侧归零 = 「不再反着弯」，
  但离「跟你一样」还差 20-40 度；§8.11.4 的②（相对中性方向的旋转口径）是同一件事的另一条线索。
- **前臂 jerk 峰值变大**（h1_2 左 13.07 -> 42.71 度；同一侧上臂 16.42 -> 12.17、肘 95.86 -> 77.42
  反而更小）。身体不再替手臂吸收操作者录进来的 153 度/帧方向跳变，全落在前臂上。
  方向跳变护栏（§8.12 ①）挡掉了一部分，p99 仍高：下一轮把门槛按**机器人可达角速度**定，
  而不是按操作者的 600 度/秒。
- **`wrist` 点也不是刚体**：探针顺手量到 `left_wrist_pitch_joint` 一转，`arm_points()["wrist"]`
  到肘的距离变 23.28 mm。它目前只用于报告/度量、没进目标，但同类「名字匹配当刚体」的用法要再扫
  一遍 —— `arm_points()` 四个点里只有 `upper` / `elbow` 是父连杆口径。

**下一轮待办**：① 上面第一条（肘任务的帧口径 / 限位）—— **§8.16 已量到主因是 PyBullet `restPoses` 那个
零空间代价项把解往加载姿势拽（不是帧口径；限位只占 h1_2 右 0.06 度）**，`--arm-rest-poses live` 能关掉它，
但现场那条录制上不划算（§8.16.4）；② 现场判定本轮三条改动；③ §8.12.7 留下的
四条仍未做（gr1_t2 收敛、左右不对称的录制对照、`verify_arm_wing.py` 的 `posture` 模式、§8.11.4 的②）。

### 8.14 「三台机器人一动不动地卡在胸前」：中性标定永远等不到数据（2026-10-14，`--arm-calibrate-mode`）

#### 8.14.1 现场症状与四步定位

2026-10-14 `live_round2` 现场（原话「手臂卡在胸前举着，怎么动都不跟」）：

1. HUD 的标定行**四分钟没变过**：`cal=left=15/30 right=16/30`。
2. 三台机器人的腕目标恒等于各自 profile 的中性工作点（HUD `arm h1_2: left=[ +0, +0, +0]mm` 那一行）。
3. 那份录像里 **85% 的帧没有任何有效手臂数据**（`counts["held_reason"]`：`no_pose_landmarks` 为主，
   另有一批 `reach_below_min` —— 关键点抖动导致「肩-腕」短于 `MIN_OPERATOR_REACH_M`）。
4. 于是 `hold` 口径下永远攒不够 `--arm-calibrate-frames` 帧 -> `_has_neutral()` 恒为 False
   -> 三台全部停在 `valid=False, held=True`（`_target_from_state` 末尾那条退路）。

**这不是臂映射的错，是「没有数据就永远不开始」**：映射 / 增益 / IK 那一路在本场里一次都没被执行到。
离线复算把这一点钉死了 —— 同一份稀疏输入，`hold` 的横轴指令行程只有 **8.2 mm**，`track` 是 **200.0 mm**。

#### 8.14.2 两个口径

| 口径 | 行为 | HUD | 谁在用 |
|---|---|---|---|
| `hold` | 攒够 `N` 帧之前目标钉在 profile 的中性工作点（`valid=False, held=True`），一帧都不跟 | `cal=left=15/30 right=16/30` | `RealtimeArmBridge` 的**构造默认值**（历史行为逐字保留） |
| `track` | 这一侧**第 1 帧有效手臂**就把「已有样本的中位数」当临时中性开始跟手；攒够帧时收敛到与 `hold` **逐位相同**的终值 | `cal=track left=15/30 right=16/30` | `realtime_teleop` 的 **CLI 默认**（2026-10-14 起） |

```powershell
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py                          # track（新默认）
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-calibrate-mode hold  # 回到历史行为
```

刻意留的分工：**库（`RealtimeArmBridge(calibrate_mode=...)`）默认仍是 `hold`，只有 CLI 默认 `track`**。
这样队友的离线探针与既有调用点行为一个字都不变，本节这套 hold/track 对照也才可复现 ——
两条都由 `tests/test_arm_realtime_bridge.py::CalibrateModeTests` 钉住。

「为什么这一帧没跟手」逐帧计入 `counts["held_reason"]`（`calibrating` / `no_pose_landmarks` /
`reach_below_min` / `other`，见 `arm_realtime.HOLD_REASONS`），退出摘要里成行打印。
现场先看这四个数，不要先怀疑臂映射。

#### 8.14.3 离线 A/B（可复现，本机实测）

夹具 `outputs/tmp_realtime/ab_sparse_arms.npz`（300 帧，故意稀疏：手臂左 15 / 右 16 帧有效，手部密集），
四条命令只有「标定口径 + 门槛帧数」两个开关不同：

```powershell
cd F:\simulation_platform_cs
$py = "E:\python3.11.7\python.exe"
foreach ($tag in @("hold30", "track30", "hold8", "track15")) {
    $mode   = if ($tag -like "track*") { "track" } else { "hold" }
    $frames = [int]($tag -replace "\D", "")
    & $py devtools\run_realtime_arm_windows.py --input-npz outputs\tmp_realtime\ab_sparse_arms.npz `
        --headless --frames 300 --arm-calibrate-frames $frames --arm-calibrate-mode $mode `
        --report "outputs\tmp_realtime\ab_calmode_$tag.json" `
        --arm-frame-dump "outputs\tmp_realtime\ab_calmode_${tag}_frames.npz"
    "== $tag exit=$LASTEXITCODE"      # 期望四个 0
}
& $py outputs\tmp_realtime\ab_calmode_compare.py     # 汇总 -> outputs\tmp_realtime\ab_calmode_table.txt
```

四条退出码全 **0**，比较脚本给出的表（节选；`cmd ptp xyz` = 命令点在机器人基座系逐轴的峰峰值 mm）：

| 配置 | 模式 | 标定进度 | complete | arm valid | cmd frames | cmd ptp xyz 左 | cmd ptp xyz 右 | clamped |
|---|---|---|---|---|---|---|---|---|
| `hold`  cal-frames=30（现场设置） | hold | 15/30 | False | 177 | 299 | **8.2** / 397.2 / 416.7 | **8.2** / 630.3 / 320.1 | 0 |
| `track` cal-frames=30（新默认） | track | 15/30 | False | 207 | 299 | **200.0** / 405.2 / 440.3 | **208.6** / 467.3 / 329.3 | 535 |
| `hold`  cal-frames=8（对照：门槛够得着） | hold | 8/8 | True | 193 | 299 | 119.9 / 406.1 / 416.7 | 122.6 / 630.3 / 320.1 | 150 |
| `track` cal-frames=15（对照：门槛够得着） | track | 15/15 | True | 207 | 299 | 200.0 / 405.2 / 440.3 | 208.6 / 467.3 / 329.3 | 535 |

怎么读这张表（四个坑，本轮全踩过）：

1. **`arm applied` 不能用来判「跟没跟手」**：`hold` 也在「施加命令」（施加的是预设工作点），
   两种模式都是 600 帧（`arm.applied_frames`，两侧之和）。判据只有 `cmd ptp xyz` / `arm valid` / `clamped`。
2. **只看最大轴会把真症状藏起来**：四个配置的 y/z 两轴都是几百毫米（操作者本来就在上下前后动），
   差异全在**横轴**：`hold` 8.2 mm vs `track` 200.0 mm —— 这正是现场那句「大臂不左右摆」。
3. **`hold8` 是「不是映射的错」的对照组**：门槛降到 8 帧（这份录像够得着）后 `complete=True`，
   横轴立刻有 119.9 mm。它没到 200.0 是因为前 8 帧还在等 + `gain_trusted` 在 `complete` 之前
   退回 `metric_scale`（见比较脚本末段「why」块）。
4. **`clamped` 是「计数器之和」，不是帧数**：逐机器人逐侧计数（3 台 x 2 侧 = 6 个），记的是
   「这一帧的请求超过了该侧 `realtime_travel_m`」（`arm_realtime.py`：`request = max|delta|`，
   `clamped = request > limit`），**与 `--arm-travel-mode` 无关**（clip / scale 都记）。
   `track` 的 535 表示上限经常在工作，不是「堵死了」；横轴 ptp 恰好 200.0 mm = 2 x h1_2 左限 0.10 m。

#### 8.14.4 同轮修掉的两个「证据看不见」的缺陷（不修这两个，这张表根本读不出来）

1. `src/teleoperation/simulation/arm_ik.py::record_posture`：`command_wrist` 与 `command_elbow`
   被**同一个 `and`** 串起来，于是在默认 `--arm-elbow-task off` 下**两个都没记** ->
   `posture_summary()` 的「asked vs achieved」整族变成 `None`、逐帧 dump 里 `cmd_wrist` 全 NaN。
   离线 A/B 看起来就像「手臂根本没收到命令」，真症状（横轴 8 mm）被埋进「没有数据」。
   修法：两者**各自独立**记录（腕从第一次 `solve` 起每帧都记；肘只在肘任务开且有锚点测量时记），
   `posture_summary()` 新增 `wrist_command_frames` 与 `command_note` 两个字段（肘命令缺失时也照算
   `wrist_task_mm`），不再静默返回 `None`。验收指标：`cmd frames` 从 0（全 NaN）变成 299。
2. `outputs/tmp_realtime/ab_calmode_compare.py` 的列口径：原来的 `applied` 是**手**的计数
   （`scene_summary.applied_frames`），拿它判手臂必然看错（两种模式都是 300）。现在分成
   `arm cmd`（命令点被记下的帧数；修 `record_posture` 前是 0）/ `arm applied`（IK 真正施加的帧数）/
   `hand applied`，并补 `cmd ptp xyz` 与 `wrist ptp xyz`（逐轴峰峰值）以及末段「why」块
   （中性锚点是「操作者自己的中性记录」还是「profile 预设工作点」、`gain_trusted`、
   逐侧 `realtime_travel_m`）。

#### 8.14.5 顺手修的 `--help` 崩溃（不修的话这个新开关根本查不到）

`argparse` 渲染帮助时会对每条 `help` **再做一次 `%` 格式化**
（`argparse.py::_expand_help` -> `self._get_help_string(action) % params`）。所以 help 字符串里
任何剩下的裸 `%` 都会抛异常，整条命令的帮助都打不开 —— 这是**改前就存在**的缺陷：

```powershell
# 取改前的文件（checkpoint 803a275）直接跑：
E:\python3.11.7\python.exe outputs\tmp_realtime\old_realtime_teleop.py --help; $LASTEXITCODE
# 改前：退出码 1，TypeError: %o format: an integer is required, not dict（argparse.py 第 640 行）
# 改后：退出码 0，帮助全文正常打印，含 --arm-calibrate-mode {hold,track}
```

凶手是 `--arm-elbow-recovery-deg` 那段文案里的 `91%%`：它自己先用 `"% CONST"` 格式化过一道，
`%%` 被吃成一个 `%`，argparse 再看到就崩。修法：那一块改成 f-string，**字面百分号只留 `%%`
给 argparse**；顺手把它错写的 `metavar="MM"` 改成 `DEG`。守卫是新增的
`tests/test_arm_realtime_bridge.py::HelpTextTests`（3 条）：`build_parser().format_help()`
不抛异常、渲染结果里含 `91% of frames`、且**不含** `%%`。

#### 8.14.6 现场怎么看

```powershell
cd F:\simulation_platform_cs
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --arm-map posture
```

启动行 + 提示语里这四处就是本节（一行行核对，对不上就是默认没生效）：

```
arms: h1_2, gr1_t2, g1 | ik_mode {...} | orientation off | travel limit {...} m (clip) |
      ... | neutral mode track | neutral 30 frames | target slew 600 mm/s (ceiling 40 mm/frame)
[INFO ] arms follow all three robots; keep a neutral arm pose for about 30 frames per side:
         the arms already follow you from the first frame that sees your arm
         (--arm-calibrate-mode track) while that neutral builds up
keep a neutral arm pose (elbows slightly bent, hands in front of the chest) for about 30
frames per side; the arms already follow from the first frame that sees your arm, and the
neutral keeps converging while you hold still.
```

- `neutral mode track` = 本节生效；`hold` 时这一行是 `neutral mode hold`，提示语换成
  `until then the arms hold their working point.`
- 判据①**第一帧就该动**：抬手 / 摆臂，三台都要有反应，不用等 30 帧。
- 判据②**中性仍在收敛**：继续站定 2-3 秒，`cal=track n/30` 走满后变成 `cal=ok`；
  之后的目标轨迹与 `hold` 攒够帧后的终值一致（同一批样本、同一个中位数）。
- 退出摘要先看 `arm held frames:` 那一行：`calibrating` 很大 = 还在等标定（`hold` 才常见）；
  `no_pose_landmarks` 很大 = 摄像头压根没看到手臂（调站位 / 光照，别调参数）；
  `reach_below_min` 大 = 关键点抖动（见 §8.9）。
- 退出摘要的 WARN 也分口径：`track` 下没攒够不再是「calibration never completed」，而是
  `the neutral is still a temporary median (cal=track ...)`，并明说「手臂照常跟手」。

#### 8.14.7 验收（改动后本机实测：命令 + 退出码 + 关键不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| 8.14.3 的四条 A/B 命令（wrapper 直接调用，无需 PYTHONPATH） | **0 / 0 / 0 / 0** | `hold30` 横轴 8.2 mm vs `track30` 200.0 mm；每侧 `cmd frames=299`（修 `record_posture` 前 0 / 全 NaN） |
| `E:\python3.11.7\python.exe outputs\tmp_realtime\ab_calmode_compare.py` | **0** | 生成 `ab_calmode_table.txt`（30 行），含逐轴 `cmd ptp xyz` 与「why」块 |
| `E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --help`（`PYTHONPATH=src`） | **0** | 帮助全文打印，含 `--arm-calibrate-mode {hold,track}` 与 `91% of frames`；**改前退出码 1 + `TypeError`**（见 8.14.5） |
| `E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py --help` | **0** | 现场入口自己也能打开帮助（改前同样崩） |
| `E:\python3.11.7\python.exe outputs\tmp_realtime\diag_help_crash.py` | **0** | `[CRASH]` **0** 条（改前 1 条，指向 `--arm-elbow-recovery-deg`）；末尾 `parse_args(['--help']) -> SystemExit(0)` |
| `E:\python3.11.7\python.exe -m pytest tests -q` | **0** | `345 passed, 3 skipped, 309 subtests passed in 84.85s` |
| `E:\python3.11.7\python.exe -m unittest discover -s tests` | **0** | `Ran 348 tests in 76.296s` + `OK (skipped=3)`（比改前多 3 条 = `HelpTextTests`） |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | **0** | `[OK]` 未发现 GBK 无法编码的字符 |
| `E:\python3.11.7\python.exe devtools\check_no_c_drive.py --strict` | **0** | `[OK]` 未发现 C 盘写入隐患 |
| `run_realtime_arm_windows.py ... --arm-calibrate-frames 30`（**不带**模式开关，检验 CLI 默认） | **0** | 报告 `bridge.calibrate_mode=track`、HUD `cal=track left=0/30`；横轴 ptp 200.0 / 208.6 mm，与显式 `track30` **逐位相同** |
| 同上再加 `--arm-travel-mode scale` | **0** | `clamped` 仍 535（该计数器与模式无关）；最大轴照旧钉在 2 x 上限，非最大轴按比例变小（左 y 405.2 -> 352.3、z 440.3 -> 407.4 mm） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | **0** | `RESULT: [OK] all checks passed (2 warning(s))`（8 项全过）；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；第 8 项 `Ran 348 tests in 78.3s [OK (skipped=3)]` + `python -m teleoperation --help -> exit 0`（`--help` 崩溃修复在**文档化的 CLI 冒烟路径**上复验）；**不要**用默认基线 `10ebd9f` 跑，否则队友线 49 个文件会被算成「改了队友文件」（§8.3 已说明，属已知假 FAIL） |

#### 8.14.8 改动清单（谁的文件 / 谁的默认值）

| 文件 | 改动 |
|---|---|
| `src/teleoperation/simulation/arm_realtime.py` | `RealtimeArmBridge(calibrate_mode=...)`（库默认仍是 `hold`）、临时中性、`HOLD_REASONS` 分类计数、`status()` 带模式前缀 |
| `src/teleoperation/simulation/realtime_teleop.py` | `--arm-calibrate-mode {hold,track}`（CLI 默认 `track`）、HUD 与退出摘要、按模式的 WARN 文案、`--help` 崩溃修复 + `metavar` 修正 |
| `src/teleoperation/simulation/arm_ik.py` | `record_posture` 腕 / 肘独立记录；`posture_summary()` 新增 `wrist_command_frames` / `command_note` |
| `devtools/run_realtime_arm_windows.py` | 启动提示语按模式分支；`--arm-calibrate-mode` 透传 |
| `tests/test_arm_realtime_bridge.py` | `CalibrateModeTests`（标定口径）、`HelpTextTests`（帮助可渲染 / 无 `%%` 泄漏） |
| `outputs/tmp_realtime/**` | 稀疏夹具、A/B 脚本、对照表、诊断脚本（已 gitignore，不入库） |

**这些文件都在我们这一侧**，没有踩到队友已入库的 `retargeting/**`、`config/**`、`input_adapters/**`。

**落地提交**：`87d3b53 fix: 实时手臂中性标定加 --arm-calibrate-mode（CLI 默认 track），并修 --help 崩溃`
（6 文件 / +601 -39，本机 `cs/multi-robot-arm` 分支；未 push）。

#### 8.14.9 如实说（本轮没解决的）

- A/B 用的是**合成稀疏夹具**（300 帧，手臂左 15 / 右 16 帧有效），复现的是「帧稀疏 -> 标定门永远
  够不着」这一条**机制**，不是现场那场录像的全部干扰（光照 / 站位 / 摄像头帧率）。
- `track` 只解决「不开始」：`track30` 里 `no_pose_landmarks` 仍有 170 / 182 帧，那些帧**任何口径都
  不会让机器人动** —— 现场的第一优先级仍是「MediaPipe 到底有没有看到手臂」（§8.9 / §8.12.6 的
  站位与光照清单）。
- `track` 开头几帧的中性基准是**临时中位数**（样本少时更抖），所以「看见就动」那一瞬间的幅度 /
  方向可能偏；站定几秒后与 `hold` 一致。要严格可复现的离线对照仍用 `hold`。
- **行程上限确实在压幅度**（本轮查过，不再是猜测）：`track30` 的 `clamped=535`（6 个计数器之和）
  与横轴 ptp 恰好 2 x h1_2 左限 0.10 m，都说明「请求超过该侧上限」经常发生。同一条命令再加
  `--arm-travel-mode scale`：`clamped` 与**最大轴**的 ptp 都不变（`request = max|delta|`，两种模式
  都把最大轴压到正好等于上限），只有**非最大轴**按比例变小（左 y 405.2 -> 352.3、z 440.3 -> 407.4 mm）。
  也就是说 `travel_mode` 改变不了「最大轴被压到上限」这件事，要真正放开横向幅度只能抬
  `--arm-travel-mm`（现场判定，改完重跑本节这套 A/B）。
- §8.13.8 那三条（肘残差、前臂方向 20-40 度、前臂 jerk）本轮**没有动**，阈值与结论都照旧。
- 本轮全部是离线夹具 + 单元测试；**没有真机验证**，也没有现场摄像头实拍验证。


### 8.15 「大臂抬不起来 / 张不开怀抱 / GR1-T2 前臂还是反的」：那一场跑在默认 `chain` 口径上（2026-10-14 现场第三轮）

#### 8.15.1 现场原话与真正的变量

> 改代码之前，我的手臂向上平举时三个机器人的大臂都能抬起来，现在抬不起来了；我张开手臂时
> 机器人也张不开怀抱（和之前一样）；GR1-T2 的大臂与小臂夹角是反的，和之前一样。

这一场的命令是 `E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py`（**不带任何开关**），
日志 `outputs/tmp_realtime/live_now.out.log` / `short_run.log` 里**没有** `arm map:` 那一行 ——
因为那一行历史上只在 `--arm-map segments|posture` 时打印。也就是说这一场跑的是**入口的历史默认
`chain`**；而 §8.10-§8.13 修好的东西（肘位置目标、方向口径、刚性肩锚点、IK 只动本侧手臂）全在
`posture` 那一支里。之前几场现场是**显式带 `--arm-map posture`** 跑的
（`outputs/tmp_realtime/` 里 68 行启动日志都是 posture，其余 8 行是 segments，**一行 chain 都没有**）。

于是三句反馈有同一个根因：

| 现场症状 | `chain` 口径下的机制（本次逐项复现） |
|---|---|
| 大臂抬不起来 | 目标是「中性工作点 + 缩放的（腕 - 肩）位移」，先被 profile 的**逐轴行程上限**剪断（g1 每轴 40 mm、h1_2 150 mm）；§8.10.1 已量过「只动大臂」时目标位移恒为 **0.0 mm** |
| 张不开怀抱 | 同一根夹子：目标位移被压在 40-150 mm，横向那半程也一起被削（§8.9 / §8.10.4） |
| GR1-T2 前臂夹角反 | `chain` **不向 IK 交任何肘目标**，7 自由度冗余链的弯肘方向是求解器自己的选择 —— 现场看到的「反」不是映射算错，而是**没有要求** |

#### 8.15.2 A/B：同一份录制、同样代码与默认，只差 `--arm-map`

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
# A: 历史默认（现场这一场就是它）
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --report outputs\tmp_realtime\ab15_chain.json
# B: 方向口径（现场之前几场一直在显式用的那一支）
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --report outputs\tmp_realtime\ab16_posture.json
E:\python3.11.7\python.exe devtools\report_table.py outputs\tmp_realtime\ab15_chain.json outputs\tmp_realtime\ab16_posture.json
```

| 机器人/侧 | 肩行程（度） | 上臂方向行程（度） | 肘跨距（mm） | 前臂反向帧占比 | 前臂方向误差中位（度） | 夹断帧数 | 腕任务残差中位（mm） |
|---|---|---|---|---|---|---|---|
| h1_2 左 | 50.11 -> **119.1** | 48.09 -> 110.08 | 271.4 -> **561.1** | - -> 0.011 | - -> 18.79 | 370 -> **0** | 3.06 -> 23.64 |
| h1_2 右 | 44.48 -> **186.83** | 44.27 -> 164.87 | 250.4 -> **663.7** | - -> 0.000 | - -> 10.54 | 460 -> **0** | 2.55 -> 21.06 |
| gr1_t2 左 | 78.65 -> **168.28** | 79.93 -> 154.84 | 333.6 -> **494.1** | - -> 0.028 | - -> 43.00 | 370 -> **0** | 2.64 -> 14.48 |
| gr1_t2 右 | 88.51 -> **149.97** | 87.70 -> 135.03 | 359.4 -> **479.8** | - -> 0.000 | - -> 35.61 | 460 -> **0** | 2.27 -> 14.50 |
| g1 左 | 37.24 -> **243.6** | 35.73 -> 177.68 | 123.0 -> **395.8** | - -> 0.017 | - -> 10.17 | 370 -> **0** | 3.72 -> 16.46 |
| g1 右 | 25.56 -> **164.38** | 24.45 -> 174.84 | 85.1 -> **392.2** | - -> 0.000 | - -> 8.36 | 460 -> **0** | 3.64 -> 12.78 |

读三条：

1. **「大臂抬不起来」看第 1、2 列**：六侧全部变好，g1 左 37.2 -> 243.6 度（6.5 倍）、上臂方向行程
   35.7 -> 177.7 度（5 倍）；
2. **「张不开怀抱」看第 3 列**：六侧全部变好，g1 两臂 123 / 85 -> 396 / 392 mm，h1_2 右 250 -> 664 mm，
   而 `bridge.counts.clamped` 从 370 / 460 帧掉到 **0 / 0** —— 那个夹子不再是主角；
3. **「前臂反」看第 4、5 列**：`chain` 那一列是 `-`，因为 chain **不要求肘**，报告里没有「要的方向」
   可比 —— 这正是「夹角反了」在 chain 下**量不出来**的原因（只能看到机器人自己弯到哪）。
   `posture` 下六侧反向帧占比 0.000-0.028，前臂方向误差中位 8.4-43.0 度。

#### 8.15.3 改动（按约定 14，全部落在既有文件里，没有新文件）

| 文件 | 改动 |
|---|---|
| `src/teleoperation/simulation/realtime_teleop.py` | `--arm-map` 的 **CLI 默认 `chain` -> `posture`**（库 / 桥默认仍是 `chain`，`devtools/verify_*.py` 与老 A/B 记录逐字不变）；help 文案改写；**chain 时也打印一行启动提示**（说明它不给肘目标、肩被夹在 40-150 mm，并指向 `--arm-map posture`）——这一次现场就是「日志里一行口径提示都没有」才被读成「代码被改坏了」 |
| `devtools/report_table.py` | 读表脚本加「肩行程 / 上臂方向行程 / 肘跨距 / 夹断帧数」四列、缺字段打 `-`（改前读 chain 报告会 `TypeError: unsupported format string passed to NoneType.__format__` 直接崩 —— 工具崩了就没人看得到这一层） |
| `tests/test_arm_realtime_bridge.py` | `PostureMappingTests` 新增 1 条：入口默认是 `posture`，显式 `--arm-map chain` / `segments` 仍能进 |
| `docs/multi-robot-arm.md` | 本节 + §8.10.6 的默认值行加注 |
| `docs/RETARGETING_PIPELINE.md` | 实时入口小节记录新默认（现场不再需要加开关） |

#### 8.15.4 验收（改动后，本机实测：命令 + 退出码 + 关键不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `E:\python3.11.7\python.exe -m pytest tests\test_arm_realtime_bridge.py tests\test_arm_preview.py tests\test_arm_ik_multi_robot.py -q` | 0 | `137 passed, 49 subtests passed`（含新增的那 1 条） |
| 8.15.2 的两条 `--headless` 命令 | 0 / 0 | 报告 `bridge.map_mode` 分别是 `chain` / `posture`；六侧的肩行程 / 肘跨距 / 夹断帧数见上表 |
| `E:\python3.11.7\python.exe devtools\report_table.py`（两份报告一起读） | 0 | 不再抛 `TypeError`，缺字段打 `-` |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | 0 | 8 项全过；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；第 8 项 `Ran 353 tests ... OK (skipped=3)` |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | 0 | 未发现 GBK 无法编码的字符 |

#### 8.15.5 现场怎么跑（这一次不用再加开关）

```powershell
cd F:\simulation_platform_cs
E:\python3.11.7\python.exe devtools\run_realtime_arm_windows.py
```

启动行现在（默认）应当是：

```
arm map: posture | shoulder end = upper (rigid: the elbow parent link) | ik scope = arm |
measured links (m) ... | elbow task: built by the direction recipe (pass last) |
elbow branch recovery: >25 deg (margin 8 deg, retry every 12 frames)
```

看到 `arm map: chain ...` 就说明显式退回历史口径了（它会自己把代价写在那一行里）。
动作按 §8.13.6 的三条看：抬到肩高屈肘伸肘两次（看前臂方向）、重复 10 次（看抽不抽）、
张开怀抱（看幅度）。

#### 8.15.6 如实说（本轮没解决什么）

- **GR1-T2 仍是六侧里最差的一侧**：`posture` 下它前臂方向误差中位 43.0 / 35.6 度、反向帧占比
  0.028 / 0.000，而**肘任务残差中位 175.8 mm（max 271.0）**，h1_2 / g1 是 27.7-72.1 mm
  （§8.13.8 记的是同一件事：肘任务的世界点与「腕任务用的那个 link 帧点」是不是同一口径，
  以及关节限位有没有把合法姿势排除掉 —— 这两条本轮仍未查，加迭代数不是办法；**同一天深夜已结清，
  见 §8.16**：主因是 PyBullet `restPoses` 这个零空间代价项把解往加载姿势拽，不是帧口径，限位只占
  h1_2 右 0.06 度）。
- 幅度与精度仍是同一根跷跷板：`posture` 下腕任务残差中位 12.8-23.6 mm（chain 是 2.3-3.7 mm，
  但那是「目标本来就被夹小」换来的），末帧快照还会出现 g1 左 102.2 mm 这种数（§8.11.4 已说明）。
- **没有真机验证**：本轮的机制复现在 2026-10-13 那份真实录制（200 帧）上，现场摄像头帧率 /
  站位 / 光照仍是另一层（§8.14.9 的 `no_pose_landmarks` 那些帧任何口径都不会让机器人动）。

### 8.16 「肘/小臂不跟手」的机制：`restPoses` 是**代价项**，但换成「当前姿势」不是解药（2026-10-14 深夜，`--arm-rest-poses`）

#### 8.16.1 症状与要证的假设

§8.15 之后六侧的读数仍是一对互相矛盾的数：**腕任务残差中位 12.8-23.6 mm（位置基本兑现），肘任务残差
中位 27.7-175.8 mm（命令的肘位几乎没兑现）**。§8.13.8 与 §8.15.6 把「肘任务的世界点跟腕任务那个 link
帧点是不是同一口径」和「关节限位有没有把合法姿势排除」记成待查项 ①。

本轮要证的假设：PyBullet 的 `calculateInverseKinematics` 里 `restPoses` **不只是起点** —— 代价里还有
一项「离 `restPoses` 多远」，于是每一帧都在把解往**加载时**的直臂下垂姿势拽。若它成立，给一对**精确可达**
的目标（机器人自己 FK 出来的腕 + 肘）也不该解到目标上。

#### 8.16.2 判别式：探针里的 A/B（同一对精确可达的目标）

`devtools/probe_arm_segment_rigidity.py` 的 `ik_frame_test` 现在把恒等测试做两遍：一遍走 `solve()` 的默认
（`restPoses` = 加载姿势），一遍把 `posture=` 交成**当前**关节角（`_rest_poses(posture)` 本来就是这条接口）。

| 机位（腕 / 肘连杆） | 恒等残差 腕（mm）加载姿势 -> 当前姿势 | 恒等残差 肘（mm）加载姿势 -> 当前姿势 | 解完被拽动（度）加载姿势 -> 当前姿势 | 最紧臂关节限位余量（度） |
|---|---|---|---|---|
| h1_2 左（left_wrist_yaw / left_elbow_pitch） | 24.27 -> **0.00** | 15.28 -> **0.00** | 2.642 -> **0.000** | 13.25 |
| h1_2 右（right_wrist_yaw / right_elbow_pitch） | 41.01 -> **0.47** | 23.36 -> **0.38** | 4.252 -> **0.136** | **-0.06** |
| gr1_t2 左（left_hand_pitch / left_lower_arm_pitch） | 73.20 -> **0.00** | **140.06** -> **0.00** | **35.032** -> **0.000** | 24.05 |
| gr1_t2 右（right_hand_pitch / right_lower_arm_pitch） | 14.22 -> **0.00** | 7.23 -> **0.00** | 1.631 -> **0.000** | 24.25 |
| g1 左（left_wrist_yaw / left_elbow） | 17.47 -> **0.00** | 6.63 -> **0.00** | 1.959 -> **0.000** | 58.47 |
| g1 右（right_wrist_yaw / right_elbow） | 31.02 -> **0.00** | 33.25 -> **0.00** | 27.733 -> **0.000** | 33.5 |

读两条：

1. **零空间项就是那根绳子**：同一对目标、同一份代码，只把 `restPoses` 从加载姿势换成当前姿势，六侧残差
   6.6-140.1 mm -> 0.0 mm、拽动 1.63-35.03 度 -> 0.00 度。所以 §8.13.8 的 ① 前半（「肘任务的世界点跟腕
   任务的帧点是不是同一口径」）**不是**主因 —— 求解器能解到目标，是它**被代价项拉着不想去**。
2. **唯一没塌到 0 的是 h1_2 右**（0.47 / 0.38 mm），它的 `min_limit_headroom_deg = -0.06`：这条由机器人
   自己 FK 出来的姿势**本身就站在一条臂关节限位外 0.06 度** —— §8.13.8 的 ②（限位排除合法姿势）在这一侧
   确实成立，但量级只有 0.06 度，解释不了几十毫米的残差。

复现（退出码 0，`wrote tmp_ik_frame\probe_ik_frame2.json`）：

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe devtools\probe_arm_segment_rigidity.py --json tmp_ik_frame\probe_ik_frame2.json
```

#### 8.16.3 改动（`--arm-rest-poses`，opt-in；按约定 14 全落在既有文件里）

| 文件 | 改动 |
|---|---|
| `src/teleoperation/simulation/arm_ik.py` | ① `rest_poses` 多认一档 `"live"`（校验与报错文案同步）；② 新增 `_live_rest_poses()`（每次 `solve` 重读本侧臂关节角）；③ `_rest_poses(posture)` 改成三档分派：`posture` 优先 -> `"live"` 走 `_live_rest_poses()` -> 否则 `limits["rest"]`；④ `_teleport()` **固定**用原始 `limits["rest"]`（支路恢复要的是「跟当前姿势无关的那一支重解」，`live` 会把那行变成原地重解）；⑤ `describe()` 的 `rest_poses` 进报告 JSON |
| `src/teleoperation/simulation/realtime_teleop.py` | `--arm-rest-poses {current,zero,live}`（**默认 `current` = 历史行为逐字不变**）；启动行加 `rest poses = {当前档}`，现场不看命令行也知道跑的是哪一档 |
| `src/teleoperation/simulation/arm_scene_driver.py` | 构造参数 `rest_poses` 透传 + 场景 CLI 的 `--rest-poses` 同档位 |
| `tests/test_arm_ik_multi_robot.py` | 新增 `LiveRestPosesTests` 2 条（`test_live_keeps_an_exactly_reachable_target_and_the_load_pose_does_not`：恒等目标下 `live` 残差 <= 1.0 mm 且严格优于加载姿势；`test_live_is_accepted_and_a_typo_still_fails`）+ 入口 1 条（`test_rest_poses_live_is_reachable_from_the_field_entry`：默认仍是 `current`、`--arm-rest-poses live` 能进）。同一文件的 `seed_pose` 顺手把肘种子夹进关节限位（±5 度），否则 g1 那种肘行程最小的侧会让恒等测试变成「限位饱和」而不是「求解器」 |
| `docs/multi-robot-arm.md` + `docs/RETARGETING_PIPELINE.md` | 本节（§8.16）+ §8.13.8「下一轮待办 ①」与 §8.15.6 首条加指向；`RETARGETING_PIPELINE.md` 的「最后更新」行与 §4.2 手臂入口末条（`--arm-rest-poses` 与 §8.16 的结论）同步 |

#### 8.16.4 现场录制 A/B：`current` vs `live`（同一份输入、同样 200 帧、只差一个开关）

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-rest-poses current --report outputs\tmp_realtime\ab_rp_current2.json
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-rest-poses live --report outputs\tmp_realtime\ab_rp_live2.json
E:\python3.11.7\python.exe devtools\report_table.py outputs\tmp_realtime\ab_rp_current2.json outputs\tmp_realtime\ab_rp_live2.json
```

两份报告的 `bridge.counts` **逐位相同**（`valid left 152 / right 119`、`held 48 / 81`、`posture 546 / 600`、
`clamped 0 / 0`），所以差别只来自零空间参考这一项：

| 机位 | 肘任务残差 中位/最大（mm） | 腕任务残差 中位（mm） | 前臂方向误差 中位（度） | 上臂方向误差 中位（度） | 上臂方向行程（度） | 前臂 jerk 最大（度） | 支路拒绝帧数 |
|---|---|---|---|---|---|---|---|
| h1_2 左 | 72.05 / 288.75 -> **46.05** / 253.67 | 23.64 -> **12.14** | 18.79 -> **14.01** | 22.43 -> **14.24** | 110.08 -> **143.36** | 22.16 -> 87.76 | 12 -> **6** |
| h1_2 右 | 37.61 / 317.02 -> 72.79 / 314.77 | 21.06 -> **4.52** | 10.54 -> 18.37 | 11.72 -> 22.74 | 164.87 -> 167.37 | 9.80 -> 26.82 | 9 -> **7** |
| gr1_t2 左 | 175.79 / 271.01 -> **80.96** / 319.08 | 14.48 -> 70.87 | **43.00 -> 17.02** | 58.99 -> **26.01** | 154.84 -> 162.10 | 20.21 -> 32.51 | 14 -> **11** |
| gr1_t2 右 | 137.92 / 333.43 -> **69.21** / 331.84 | 14.50 -> 99.15 | **35.61 -> 13.42** | 45.33 -> **22.44** | 135.03 -> **83.51** | 12.86 -> 23.55 | 14 -> **10** |
| g1 左 | 27.68 / 180.90 -> 103.61 / 188.74 | 16.46 -> 29.59 | 10.17 -> 27.15 | 19.46 -> **78.20** | 177.68 -> 155.36 | 48.84 -> 37.45 | 9 -> **14** |
| g1 右 | 27.00 / 162.71 -> 93.96 / 179.10 | 12.78 -> 26.75 | 8.36 -> 26.31 | 18.95 -> **72.58** | 174.84 -> 175.60 | 17.30 -> 50.73 | 8 -> **15** |

**这不是一边倒的赢，所以默认不换**（六侧中位数的平均：肘任务 79.7 -> 77.8 mm 几乎不动；腕任务 17.2 -> 40.5 mm
变差 2.4 倍）：

1. **肘任务（本轮的目标）只有 3/6 侧变好**：gr1_t2 左 175.8 -> 81.0、右 137.9 -> 69.2、h1_2 左 72.1 -> 46.1；
   另外 3/6 侧**变差**（g1 左 27.7 -> 103.6、g1 右 27.0 -> 94.0、h1_2 右 37.6 -> 72.8）。
2. **前臂方向是这轮最实在的收获**：gr1_t2 两臂 43.0 -> 17.0、35.6 -> 13.4 度（§8.15.6 记的「GR1-T2 仍是
   六侧最差」在 `live` 下被抹平），h1_2 左 18.8 -> 14.0 也小优。
3. **代价在位置精度与上臂方向**：g1 两臂上臂方向误差 19.5 -> 78.2 / 19.0 -> 72.6 度、腕任务 16.5 -> 29.6 /
   12.8 -> 26.8 mm，gr1_t2 右的腕任务 14.5 -> 99.2 mm；前臂 jerk 最大值普遍上升（h1_2 左 22.2 -> 87.8 度）、
   支路拒绝帧数 g1 两臂 9 / 8 -> 14 / 15。

**为什么会这样**：把参考换成「当前姿势」等于把那项代价变成**纯阻尼** —— 它只抵抗关节变化，不再指定肘/上臂
该在哪。目标一跳，它就在 60 次迭代里拖后腿；而这份录制里操作者输入本身很脏（200 帧里 `no_pose_landmarks`
左 31 / 右 79 帧、`reach_below_min` 1 / 2 帧；标定臂长左 0.2742 / 右 0.3976 m，同一个人两侧差 31%，见
§8.8 / §8.9.1；单帧方向跳变可达 153 度，见 §8.13.8）。原来的加载姿势参考虽然**拽**，同时也是一根**稳定的锚**（把解固定在
一个一致的支路上）。在「位置目标 + 零空间项」这一个代价里，「拽」与「锚」是同一项的两面，换个参考只是换了
偏置方向。要真正「肘按你的姿势摆」，需要的是把**操作者的关节角**当参考（论文的 `PostureTask`），而不是
「让零空间别拽我」。

#### 8.16.5 顺带结清的两个数

1. **同侧两机型 gain 差一倍（h1_2 1.2575 vs gr1_t2 0.7186）不是 bug**：`gain = profile.metric_scale *
   clamp(参考臂长 0.6 m / 本侧标定臂长, band)`（`arm_realtime.py::_gain`；`metric_scale` = h1_2 0.7 /
   gr1_t2 0.4 / g1 0.4，`arm_profiles.py` 第 174 / 197 / 217 行）。本轮两份报告的 `bridge.gain_by_side`
   逐字是 `{left: {h1_2 1.2575, gr1_t2 0.7186, g1 0.7186}, right: {h1_2 1.2419, gr1_t2 0.7096, g1 0.7096}}`，
   标定臂长 left 0.334 / right 0.3382 m：`1.2575 = 0.7 * 1.7964`、`0.7186 = 0.4 * 1.7965`（右侧
   `0.6 / 0.3382 = 1.7741`）—— **同一侧两机型的比恒等于 0.7 : 0.4 = 1.75**，跟标定好坏无关。上一轮记录里
   那对 0.9965 / 0.569 也是同一个比（`0.569 / 0.9965 = 0.571 = 0.4 / 0.7`），只是那一场的标定臂长约 0.42 m。
   臂长归一化被护栏关掉时（`_arm_length_trust` 判不可信：本侧低于 `min_plausible_reach_m`，或左右差超过
   容差）`_gain` 直接返回 `metric_scale`，逐帧信息里的 `gain_source` 会写 `metric_scale`。-> **不要去"修"
   这个不对称**，它是三台机器人灵敏度的手柄级调参（真要动就整条命令一起动，见 `--arm-gain`）。
2. **肘残差的第二个来源（限位）量到了，但很小**：h1_2 右那条由它自己 FK 出来的"精确可达"姿势，最紧的臂
   关节余量是 **-0.06 度**（§8.16.2 表最后两列）—— 即 §8.13.8 的 ② 在那一侧**成立**，但 0.06 度解释不了
   30-140 mm 的残差（主因已由 §8.16.2 第 1 条定为零空间项）。

#### 8.16.6 现场怎么看 / 下一轮做什么

- **现场不用加开关**：默认仍是 `current`（= 现场那条命令的历史行为，逐字不变），启动行在 `arm map =
  posture` 后面多一段 `rest poses = current`，不看命令行也知道跑的是哪一档。
- 想判断"肘跟不跟手"，看报告 `arm.posture[robot][side]`：`elbow_task_mm.median > 50` 就是**命令的肘位没
  兑现**（拿它跟同一行的 `wrist_task_mm.median` 对比 —— 后者 12-24 mm 说明腕任务基本兑现）。跑的是哪一档
  看 `/arm/robots/<robot>/rest_poses`（本轮两份分别是 `current` / `live`）。
- 要复现本节的 A/B：加 `--arm-rest-poses live`，其余参数一个字不改。
- **下一轮真正的方向**：要让「肘按操作者的姿势摆」，缺的**不是** `_rest_poses(posture=...)` 这个接口（它
  已经在，§8.16.2 的探针 A/B 就是用它做的），而是把**操作者的肩 / 肘 / 腕三点变成这台机器人手臂的关节角**
  —— 现场入口 `apply_each` 至今一个 `posture` 都不传（§8.15.6 同一结论）。`live` 只是「零空间别再拽我」，
  不等于「肘按我的姿势摆」；从本节数字看，它单独用时收益（前臂方向）与代价（腕精度、上臂方向、抖动）基本
  抵消，所以留着当实验档、不当默认。

#### 8.16.7 验收（命令 + 退出码 + 关键不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `$env:PYTHONPATH="$pwd\src"; E:\python3.11.7\python.exe devtools\probe_arm_segment_rigidity.py --json tmp_ik_frame\probe_ik_frame2.json` | 0 | 六侧「精确可达」目标的恒等残差（加载姿势 -> 当前姿势）：腕 14.22-73.20 -> 0.00-0.47 mm、肘 6.63-140.06 -> 0.00-0.38 mm；被拽动 1.631-35.032 度 -> 0.000-0.136 度；`min_limit_headroom_deg` 六侧 13.25 / **-0.06** / 24.05 / 24.25 / 58.47 / 33.5 |
| §8.16.4 两条 `--headless`（`--arm-rest-poses current` / `live`） | 0 / 0 | 两份报告 `bridge.map_mode = posture`、`bridge.counts` 逐位相同（`valid 152 / 119` 等）、`/arm/robots/*/rest_poses` 分别是 `current` / `live`；只有参考这一项不同 |
| `E:\python3.11.7\python.exe devtools\report_table.py outputs\tmp_realtime\ab_rp_current2.json outputs\tmp_realtime\ab_rp_live2.json` | 0 | 上表六行都打得出来（缺字段才打 `-`，这两份没有） |
| 重跑 §8.16.4 两条命令（本轮写到新文件 `ab_rp_current3.json` / `ab_rp_live3.json`） | 0 / 0 | **数字逐位相同**：`ab_rp_current3` 与 `ab_rp_current2`（以及更早的 `ab_rp_current` / `ab16_posture`）、`ab_rp_live3` 与 `ab_rp_live2`（以及 `ab_rp_live`）的 `arm.posture` + `bridge.counts` + `bridge.gain_by_side` + `rest_poses` 全等（求解器在这份输入上确定性）；`current3` 与 `live3` 两份则确实不等 |
| `E:\python3.11.7\python.exe -m pytest tests -q` | **0** | `353 passed, 3 skipped, 312 subtests passed in 88.82s`（本轮新增 3 条：parser 1 + `LiveRestPosesTests` 2；末尾的 `nanobind: leaked ...` 是队友 `roboticstoolbox` 既有的引用计数提示，不是失败） |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | **0** | `[OK]` 未发现 GBK 无法编码的字符（`arm_ik.py` / `realtime_teleop.py` / `arm_scene_driver.py` / 测试里都用 `->`，不写全角箭头） |
| `E:\python3.11.7\python.exe devtools\check_no_c_drive.py --strict` | **0** | `[OK]` 未发现 C 盘写入隐患（`PIP_CACHE_DIR` / `MPLCONFIGDIR` / `HF_HOME` / `TORCH_HOME` / `XDG_CACHE_HOME` / `TEMP` / `TMP` 七项全在 `E:\cache`） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | **0** | `RESULT: [OK] all checks passed (2 warning(s))`（**第一遍因第 5 项 `tmp_*` 报 FAIL，按本节末尾那条加完本机 exclude 后，连跑两遍都是 `[OK]`**）；第 2 项 `no non-GBK characters in .py` + `tools check-gbk-safe --strict -> exit 0`；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；第 5 项 `nothing forbidden and no big asset staged`；第 8 项 `Ran 356 tests [OK (skipped=3)]`（80.8 / 81.2 s 两次实测）+ CLI 冒烟 exit 0；**`-Base` 要显式给 `d8bca93`**（默认基线 `10ebd9f` 会把队友线 49 个文件算成「改了队友文件」，§8.3 已说明） |

> **预检第 5 项与 `tmp_*`（本轮踩到并解决）**：本工作区检出的 `.gitignore` 是**较老的一版**，没有主线
> `F:\simulation_platform\.gitignore` 第 59 行的 `tmp_*/`；于是工作区里只要有 `tmp_ik_frame/`，第 5 项就报
> `[FAIL] 1 item(s) should not be committed: tmp_ik_frame/ (inside tmp_ -- must stay out of git)`。**不能改
> `.gitignore`**：它在基线 `d8bca93` 里，而第 4 项的 `$teamPaths = @('.')` 把整个基线树都算队友的（只放行
> `.clinerules` / `docs/CONVENTIONS.md` / `docs/RETARGETING_PIPELINE.md` 与 `devtools/` 前缀），改它就是规则 3
> 违规。本轮处理：把 `tmp_*/` 加进**本机** `F:\simulation_platform\.git\info\exclude`（该 clone 的 worktree 共享、
> 不进版本库、碰不到任何队友文件）—— 加完 `git check-ignore -v tmp_ik_frame` 输出
> `info/exclude:8:tmp_*/` 指向 `tmp_ik_frame`、`git status --porcelain` 不再列它、第 5 项回到 `[OK]`。**要根治**
> 只能让队友线把那条 `tmp_*/` 补进 `.gitignore`（或用户明确同意后单独提一个 `chore:` 提交）。

### 8.17 「肘 / 小臂不跟手」的交付：把操作者的肘弯当零空间姿势参考（2026-10-15，`--arm-elbow-follow`）

#### 8.17.1 症状与上一轮留下的那一半

§8.15.6 与 §8.16.1 记的是同一对矛盾数：**腕任务基本兑现（中位 12.8-23.6 mm）而肘任务中位
27.7-175.8 mm（最大 271-336 mm）** —— 命令的肘位几乎没落地，现场看到的就是「肘不正、小臂方向
不对」。§8.16 把机制结清了（`restPoses` 是**代价项**不是起点，能把一对**精确可达**的目标拉走
6.6-140.1 mm），也把「参考换成手臂当前位置」（`--arm-rest-poses live`）试过、判定不划算
（§8.16.4）。**剩下的那一半是接线**：`solve(posture=...)` 这个接口早在（§8.16.2 的探针 A/B 就是
用它做的），但现场入口 `apply_each` 至今一个 `posture` 都不传（§8.15.6 / §8.16.6 两处都写着）。

#### 8.17.2 改动（`--arm-elbow-follow {off,operator}`，默认 `off`；按约定 14 全落在既有文件里）

| 文件 | 改动 |
|---|---|
| `src/teleoperation/simulation/arm_ik.py` | ① `_bend_deg()`：本机肘处夹角（口径与 `arm_realtime.elbow_bend_degrees` 一致，180 = 伸直）；② `_measure_elbow_posture()`：**加载时实测**肘关节名与斜率 d(夹角)/d(关节角)（两侧各拧 ±5 度取较长的那段弦，测完原样恢复）；③ `posture_from_bend(side, bend_deg)`：操作者这一帧的**绝对**肘弯 -> 本侧肘关节的零空间参考角（越界夹关节限位、伸直方向封顶 180 度）；④ `describe()` 多一段 `elbow_posture`（关节名 / 斜率 / 加载姿势下的夹角） |
| `src/teleoperation/simulation/arm_scene_driver.py` | `apply_each(..., elbow_bend_deg=None)`：给这一项时逐台换算成参考角交给 `solve(posture=...)`；返回值里多一段 `posture`（这一帧真正交给零空间项的参考角） |
| `src/teleoperation/simulation/arm_realtime.py` | 新增 `elbow_bend_target()`（`{侧: 这一帧的肘弯 or None}`，机器人无关的那一份）+ 静态 `_bend_of()`；`summary()` 多一段 `elbow_bend_deg`（你的中性肘弯 / 这一帧的肘弯） |
| `src/teleoperation/simulation/realtime_teleop.py` | 新旗子 `--arm-elbow-follow {off,operator}`（**默认 `off` = 历史行为逐字不变**）+ 启动行加 `elbow follow = {当前档}` + 每帧把 `bridge.elbow_bend_target()` 转给 `apply_each` |
| `devtools/run_realtime_arm_elbow_follow_windows.py`（新，§8.17.10） | **现场版固化档入口**：薄包装，把 `--arm-elbow-follow operator` 钉死（`--off` 走历史档），其余全部委托 `devtools/run_realtime_arm_windows.py`（环境 / 体检 / 两个窗口 / 报告 / 退出码零复制） |
| `tests/test_arm_ik_multi_robot.py` | 新增 `ElbowPostureTests` 4 条（实测的关节/斜率；加载姿势夹角就是历史参考；折叠方向 1:1；伸直方向封顶；`apply_each` 真的把参考交给 `solve` 并留在报告里）+ 入口 1 条（默认 `off`、`operator` 能进） |
| `tests/test_arm_realtime_bridge.py` | 新增 `ElbowBendTargetTests` 5 条（标定前给 None；直臂读 180；弯臂读实际角；断流那一帧沿用上一帧；`summary()` 带中性/当前两个数）；§8.17.10 又加 `ElbowFollowLauncherTests` 6 条（固化档入口钉的是哪一档：默认档 / `--off` / 透传 / 显式档不被覆盖 / 打印最后生效项 / 钉子能被入口 parser 接受） |
| `docs/multi-robot-arm.md` + `docs/RETARGETING_PIPELINE.md` | 本节（§8.17）+ 入口末条与「最后更新」行同步 |

#### 8.17.3 关节名与斜率是**实测**的（写死会在某一台做反）

`_measure_elbow_posture` 只在加载时各侧跑一次：把 `elbow_link`（名字里含 elbow 的那个关节；
PyBullet 里子连杆索引就是关节索引）拧 ±5 度，看本机肘处夹角变多少，再原样恢复。三台实测
（`tmp_ik_frame/probe_elbow_posture.py`，退出码 0）：

| 机器人 / 侧 | 肘关节名 | 斜率 d(夹角)/d(关节角) | 加载姿势下的夹角（度） | 关节限位（弧度） |
|---|---|---|---|---|
| g1 左 / 右 | `left`/`right_elbow_joint` | **+0.998** | 172.93 / 173.04 | -1.0472 .. 2.0944 |
| gr1_t2 左 / 右 | `left`/`right_elbow_pitch_joint` | **-0.999** | 111.20 / 111.21 | -2.27 .. 2.27 |
| h1_2 左 / 右 | `left`/`right_elbow_pitch_joint` | **+0.711** | 155.20 / 155.27 | -0.95 .. 3.18 |

两件事一眼看得出来：**gr1_t2 的肘关节正方向与另两台相反**（-1.00 vs +1.00）—— 写一张表就会在
它身上把「跟你一样弯」做成「反着弯」，正是现场 2026-10-13「前臂弯曲方向相反」那种错法的长相；
而 **h1_2 的加载姿势离伸直只有 25 度**，贴着伸直那一侧的弦只有 0.711。

为什么还要「两侧各拧一次、取较长的弦」：夹角用 `acos` 量（180 = 伸直），离伸直不到一个探针角时
读数会**折返**。实测曲线（`tmp_ik_frame/elbow_curve.py`，把肘关节从加载姿势起逐点拧过去）：

| 关节角偏移（度） | -40 | -20 | 0 | +10 | +40 |
|---|---|---|---|---|---|
| g1 左 夹角（度） | 133.05 | 153.04 | 173.03 | 176.89 | **146.95** |
| gr1_t2 左 夹角（度） | 151.13 | 131.18 | 111.21 | 101.21 | 71.23 |
| h1_2 左 夹角（度） | 120.72 | 139.16 | 155.20 | 160.36 | **150.22** |

读两条：**往折叠方向是干净的 1:1**（gr1_t2 全程 1:1；g1 的 -40 -> -40.06 度；h1_2 的 -20 ->
-23.39 度，因为它那段弦只有 0.711），**往伸直方向会越过 180 折返**（g1 在 +7 度左右到达 178 的
峰值，再往后掉回 146.95）。所以 `posture_from_bend` 往折叠方向不封顶、往伸直方向封顶在 180：
不然参考会变成「让肘去姿势任务的反面」，正是 §8.16 那根绳子的老毛病。

#### 8.17.4 基准必须是**本机自己加载姿势下的夹角**（第一版取错了，A/B 喜忧参半）

第一版把参考写成「加载姿势的肘角 + 你的肘弯相对**你中性**的变化量」。它错在把两条不同手臂的
夹角当成了同一个基准：位置任务命令的那个夹角就是**你的绝对肘弯**，而本机在加载姿势下自己的夹
角是它自己的数（g1 172.9 / gr1_t2 111.2 / h1_2 155.2 度），现场那位操作者两侧的中性肘弯则是
**138.9 / 72.5 度**（报告 `/bridge/elbow_bend_deg/*/neutral`）。于是中性那一帧参考仍在跟命令对
着干：g1 的参考停在 172.9（命令要 138.9），**gr1_t2 甚至一开始就被要求往伸直拉 27.5 度**。

第一版 A/B 的读数就是喜忧参半（现场口径、同一条命令只差这一档、200 帧）：腕任务中位 3/6 侧改
善、前臂方向 2/6 侧改善，而 h1_2 右明显变差（肘任务 31.85 -> 56.03 mm、前臂方向 8.14 -> 14.74
度）。改成「绝对角 + 本机自己的基准」后见 §8.17.5。

> **顺带一条口径教训（影响本文件里所有 A/B 的读法）**：现场口径里有两个机制按**墙钟时间**换算
> —— 关键点跳变护栏（`--arm-direction-rate-deg-s`，每帧上限 = 速率 x `frame_dt`）与速率斜坡。
> 于是同一条命令两次跑，只要机器负载不同，就可能有一帧的护栏判定不同、整场轨迹随之分叉：本轮
> 实测 105 ms/帧 与 117 ms/帧 两次跑在 `arm.posture` 上有 271 个叶子不同（`direction_glitch`
> 15/6 -> 14/5）。§8.16.4 里两次相同是因为那两次的墙钟接近（107.07 -> 104.98 ms），不是
> 「这段代码绝对确定」。**本节所有 A/B 因此都加 `--arm-direction-rate-deg-s 0 --arm-slew-mm 40`
> 把这两个机制钉死**（护栏关；斜坡改成固定 40 mm/帧 = 现场 8-12 fps 下本来就生效的那个天花板），
> 关掉之后同一档连跑两遍逐位相同（见 §8.17.5 的不变量一栏）。

#### 8.17.5 确定性 A/B：`--arm-elbow-follow off` vs `operator`

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-direction-rate-deg-s 0 --arm-slew-mm 40 --report outputs\tmp_realtime\ab_ef_off_ng2.json
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-elbow-follow operator --arm-direction-rate-deg-s 0 --arm-slew-mm 40 --report outputs\tmp_realtime\ab_ef_on_ng2.json
E:\python3.11.7\python.exe devtools\report_table.py outputs\tmp_realtime\ab_ef_off_ng2.json outputs\tmp_realtime\ab_ef_on_ng2.json
```

两份报告的 `bridge.counts` **逐位相同**（`valid left 152 / right 119`、`held 48 / 81`、
`posture 546 / 600`、`clamped 0 / 0`、`direction_glitch 0 / 0`），所以差别只来自零空间参考这一项：

| 机位 | 肘任务残差 中位/最大（mm） | 腕任务残差 中位（mm） | 前臂方向误差 中位（度） | 上臂方向误差 中位（度） | 上臂方向行程（度） | 支路恢复/拒绝 |
|---|---|---|---|---|---|---|
| h1_2 左 | 60.58 / 265.14 -> **49.92** / 322.18 | 19.71 -> **15.78** | 15.73 -> **12.49** | 18.85 -> **15.47** | 127.26 -> 113.40 | 1/11 -> 2/8 |
| h1_2 右 | 31.85 / 336.23 -> 47.41 / **304.75** | 15.74 -> **11.56** | 8.14 -> 12.75 | 9.86 -> 14.80 | 187.24 -> 160.12 | 0/9 -> 0/9 |
| gr1_t2 左 | 197.68 / 336.17 -> 196.56 / 339.37 | 13.27 -> **11.97** | 51.01 -> **50.11** | 66.75 -> **66.09** | 155.40 -> 160.04 | 3/11 -> 3/11 |
| gr1_t2 右 | 137.61 / 330.41 -> **129.61** / 330.18 | 13.61 -> 13.65 | 34.48 -> **33.74** | 45.29 -> **42.92** | 143.74 -> 145.31 | 1/15 -> 2/14 |
| g1 左 | 27.79 / 167.69 -> **25.23** / 162.69 | 14.72 -> **9.28** | 8.42 -> 8.51 | 19.43 -> **17.72** | 154.67 -> 150.87 | 1/7 -> 0/8 |
| g1 右 | 26.97 / 135.34 -> **25.71** / 145.50 | 11.11 -> **7.44** | 8.06 -> 8.07 | 18.92 -> **18.00** | 151.16 -> 171.67 | 1/7 -> 3/5 |

读三条：

1. **肘任务中位 5/6 侧改善**：h1_2 左 60.58 -> 49.92、gr1_t2 右 137.61 -> **129.61**、g1 左
   27.79 -> **25.23**、g1 右 26.97 -> **25.71**，gr1_t2 左基本持平（197.68 -> 196.56）；只有
   h1_2 右变差（31.85 -> 47.41）。
2. **腕任务 6/6 侧不恶化**：19.71 / 15.74 / 13.27 / 13.61 / 14.72 / 11.11 ->
   15.78 / 11.56 / 11.97 / 13.65 / 9.28 / 7.44（g1 两侧与 h1_2 两侧还明显改善）。也就是说这一
   项**没有把位置任务挤掉** —— 这是它跟「加一个硬姿态任务」的关键差别（见 §8.17.9）。
3. 代价如实列：**h1_2 右**（前臂方向 8.14 -> 12.75 度、上臂方向 9.86 -> 14.80 度）与个别抖动
   峰值（h1_2 左前臂 jerk 10.24 -> 28.81 度；g1 右肘 jerk 56.83 -> 92.67 mm、上臂 14.72 ->
   25.87 度）。其余四侧的 jerk 基本持平或略降。

不变量（同一档连跑两遍逐位相同）：`ab_ef_off_ng2` 与 `ab_ef_off_ng`（改前代码 + 改名前的报告
键）**1996 个叶子全等**、只差 3 个时间叶子与 2 个被改名的键（`delta` -> `current`）；
`ab_ef_on_ng3` 与 `ab_ef_on_ng2` **1976 个叶子全等**、只差 3 个时间叶子。

**默认档与改前逐字同路（实测，不是读代码推断）**：把 4 个源文件 `git stash push` 回到 `2170cfb`
（真正的改前代码）跑同一条确定性命令（`ab_baseline_ng.json`），与现在的 `off` 档对比 —— **1976
个共同叶子全等**，只差 22 个「本节新增的字段」（基线报告里没有 `elbow_posture` /
`elbow_bend_deg`）与 3 个时间叶子。

#### 8.17.6 现场怎么看 / 下一轮做什么

- **现场不用加开关**：默认仍是 `off`（= 现场那条命令的历史行为，实测逐字同路，见 §8.17.5）；启动行
  现在多一段 `elbow follow = off` / `operator`，不看命令行也知道跑的是哪一档。
- 想判断这一档值不值：跑一次 `--arm-elbow-follow operator`，看报告里这三处 ——
  `/arm/robots/<robot>/elbow_posture`（量到的肘关节、斜率、加载姿势下的夹角）、
  `/bridge/elbow_bend_deg`（你的中性肘弯 / 这一帧的肘弯）、以及 `arm.posture[robot][side]` 的
  `elbow_task_mm` 与 `wrist_task_mm` 这一对数（前者是要修的那一项，后者是「有没有把位置任务挤掉」
  的判据 —— §8.17.5 里 6/6 侧不恶化）。
- **下一轮三件事**：① 现场判定（本节证据全部来自本机回放，现场摄像头那道坎仍在：
  `no_pose_landmarks` 左 31 / 右 79 帧）；② h1_2 右为什么变差 —— 现场那位操作者**两侧的中性肘弯
  不对称**（左 138.9 / 右 72.5 度，§8.17.4），而 h1_2 的斜率只有 0.711，右侧的参考摆幅被放大；
  ③ 抖动（h1_2 左前臂 jerk 10.24 -> 28.81 度）要不要给这一档加增益上限或按关节速度限幅。
- **要不要转成默认**：先不。§8.16 的 `live` 与本节这一档都是「机制成立、收益与代价混在一起」，
  默认值留给现场那条录制上的读数说话（与 `--arm-map posture` 2026-10-14 转正走同一条路径）。

#### 8.17.7 验收（命令 + 退出码 + 关键不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `$env:PYTHONPATH="$pwd\src"; E:\python3.11.7\python.exe tmp_ik_frame\probe_elbow_posture.py` | 0 | 三台两侧的肘关节名 / 斜率 / 加载夹角（§8.17.3 那张表）；`posture_from_bend(bend_at_hold + step)` 的实测夹角：g1 -40.06 / -20.02、gr1_t2 **-40.01 / -20.00**、h1_2 -50.01 / -23.39；`+1000` 一档封顶（g1 +6.37） |
| `E:\python3.11.7\python.exe tmp_ik_frame\micro_elbow.py <robot> <side>` | 0 | 写进去的关节角**读回来逐位相同**（`want` 与 `read` 一致），夹角变化与上一行一致（第一版就是靠这个探针抓出「度当弧度用」的量纲 bug） |
| §8.17.5 那两条 `--headless`（`off` / `operator`） | 0 / 0 | `bridge.counts` 逐位相同（`valid 152 / 119` 等）；两档只差 `elbow follow` 这一项；同档连跑两遍逐位相同（只差 3 个时间叶子） |
| `E:\python3.11.7\python.exe devtools\report_table.py outputs\tmp_realtime\ab_ef_off_ng2.json outputs\tmp_realtime\ab_ef_on_ng2.json` | 0 | §8.17.5 的六行都打得出来 |
| `git stash push`（把 4 个源文件退回 `2170cfb`）后跑同一条确定性命令（`ab_baseline_ng.json`） | 0 | 与 `off` 档 **1976 个共同叶子全等**（只差 22 个新增字段 + 3 个时间叶子）；随后 `git stash pop` 成功、9 个文件改动原样找回 |
| `E:\python3.11.7\python.exe -m pytest tests -q` | **0** | `369 passed, 3 skipped, 342 subtests passed in 105.83s`（基线 353 passed / 3 skipped / 312 subtests；§8.17 本轮新增 10 条 + 30 subtests，§8.17.10 的固化档入口又加 6 条；末尾 `nanobind: leaked ...` 是队友 `roboticstoolbox` 既有的引用计数提示，不是失败） |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | **0** | `[OK]`。第一遍**报 FAIL**（4 处）：测试文档串里用了排版减号 `U+2212`（`−20` 那种写法的确比 `-` 好看），改成 ASCII `-` 后通过 —— 约定 3 拦下的正是这个 |
| `E:\python3.11.7\python.exe devtools\check_no_c_drive.py --strict` | **0** | `[OK]` 未发现 C 盘写入隐患（七项缓存全在 `E:\cache`） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | **0** | `RESULT: [OK] all checks passed (2 warning(s))`；第 2 项 `no non-GBK characters in .py`；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；第 5 项 `nothing forbidden and no big asset staged`；第 8 项 `Ran 372 tests [OK (skipped=3)]` + CLI 冒烟全 0 |

#### 8.17.8 如实说（这一轮没解决什么）

- **只有本机回放证据，没有现场验证**。现场摄像头那道坎仍在（`no_pose_landmarks` 左 31 / 右 79
  帧），而这一档吃的正是「你那三个点」的质量。
- **h1_2 右变差，没查到根因**：只给观察 —— 现场那位操作者左右中性肘弯不对称（138.9 / 72.5 度），
  而 h1_2 的斜率只有 0.711，右侧的参考摆幅被放大。下一轮该拿一条**单侧**录制对照。
- **参考的权限只有 7-22%**（§8.16.2 引的 `probe_posture_task`）：它不是「命令肘」，而是「别再把我
  往加载姿势拽」。所以肘任务残差六侧最大仍有 304-342 mm —— 那些是目标本来就远、或者被关节限位
  挡住的帧（§8.16.4 的 `solution_elbow_mm` 与 `elbow_task_mm` 那对数就是用来分开这两种的）。
- **幅度是加载姿势处的一阶近似**：h1_2 那段弦 0.711，实测「-20 度」会走出 -23.4 度（17% 过冲）。
  折叠方向不封顶；本轮没加增益，先拿干净数据 —— 若现场出现超调，再按 §8.17.6 的③加限幅。
- **「手掌相对手臂的朝向」仍未交付**：§8.17.9 只说明 `--arm-orientation` 那条通道不是路。

#### 8.17.9 顺带结清：`--arm-orientation` 不是「手掌相对手臂的朝向」那条路

上一轮留下的另一半是「掌相对手臂的朝向」（§8.11.4 的②）。这一轮把操作者的腕部三点坐标系当**姿态
任务**交给 IK 试了一遍（`--arm-orientation off` / `auto` / `on`，同一份输入、同样 200 帧，
`tmp_ik_frame/orient_ab.py`，退出码 0）：

| 机位 / 指标 | off | auto | on |
|---|---|---|---|
| h1_2 左 腕任务残差（mm） | **23.64** | 38.93 | 38.93 |
| h1_2 左 前臂方向误差（度） | **19.60** | 32.92 | 32.92 |
| h1_2 左 前臂 jerk 峰值（度） | **13.05** | 95.06 | 95.06 |
| h1_2 右 腕任务残差（mm） | 20.41 | **14.55** | 14.55 |
| h1_2 右 前臂方向误差（度） | **9.46** | 13.39 | 13.39 |
| h1_2 右 前臂 jerk 峰值（度） | **9.80** | 55.48 | 55.48 |
| gr1_t2 / g1 的六个数 | 三档**完全相同** | = | = |

读三条：

1. `auto` 与 `on` 在这一份输入上逐位相同，而 `frames oriented` 是 182 / 200（左）与 200 / 200
   （右）—— **只有 h1_2 拿到了姿态约束**，g1 / gr1_t2 是 position-only，那两台三档没差别。
2. 拿到约束的那一台反而更差：腕残差 23.64 -> **38.93** mm、前臂方向 19.60 -> **32.92** 度、
   前臂 jerk 峰值 13.05 -> **95.06** 度（右同理，9.80 -> 55.48 度）。三点造的腕坐标系在手臂接近
   伸直时**平面法向会翻**（§8.11.4 记的 4-7% 帧命令 >60 度跳变就是它）。
3. `off` 那一档打印出来的 `orientation error 0.65-1.74 度`**不能**拿去跟 auto/on 的 3.10-5.39 度
   对比：没有命令姿态时 IK 用第一遍自己落下的姿态收尾（§arm_ik.solve 的说明），那个数衡量的是
   「自洽」而不是「跟你一样」。

结论：`--arm-orientation` 保持默认 `off`，姿态那一条继续按 §8.11.4 的「相对中性旋转」口径走，
**不要靠它去修手掌朝向**。「肘跟手」这一步走零空间参考（本节 §8.17.5），两者不是同一条通道。


















#### 8.17.10 现场版固化档入口：`devtools/run_realtime_arm_elbow_follow_windows.py`（2026-10-15）

现场看过这一档之后，第一件想做的事是「下次一键跑出同样的东西」。所以新增一个**薄包装**
（约定 14 的例外③：用户要求单独成文件）：它不复制任何逻辑，只把参数交给
`devtools/run_realtime_arm_windows.py` 之前钉上 `--arm-elbow-follow operator`。

| 它负责 | 怎么做的 |
|---|---|
| 钉住这一档 | `apply_pin(argv)` 追加 `--arm-elbow-follow operator`；你已显式写过这一项就不插手 |
| 对照档 | 本文件自己的糖 `--off` = `--arm-elbow-follow off`；其余开关原样透传（同一个 parser） |
| 环境 / 体检 / 两个窗口 / 报告 / 退出码 | 一律调 `run_realtime_arm_windows.main(argv)`，零复制 |
| 解释器不对 | `hand_over()` 用 E: 的解释器重跑**本文件**。上一层那个同名工具重跑的是 `run_realtime_windows.py`（只驱动手部），会把 `--arm-*` 喂给手部 parser 报 `unrecognized arguments` —— 本文件把这条岔路堵掉 |

第一行会自报家门：`this build pins --arm-elbow-follow operator (add --off ...)`。

**报告落盘有个坑（2026-10-15 实测）**：那次会话本应写
`outputs\tmp_realtime\arm_run_20261015_134436.json`，日志里 `report -> ...` 也打了，但**文件没生成** ——
因为退出方式是**直接关 PyBullet 窗口**，进程被就地结束（日志尾是
`btShutDownExampleBrowser stopping threads`），报告写不到。**按摄像头窗口的 q / Esc 退出才会落盘。**

验收（都是本机实跑）：

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| `E:\python3.11.7\python.exe devtools\run_realtime_arm_elbow_follow_windows.py --check` | **0** | 第一行 `this build pins --arm-elbow-follow operator`；`interpreter : E:\python3.11.7\python.exe`、`modules present: cv2, mediapipe, pybullet`、`cameras readable: [0]` |
| 同上 + `--input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --report tmp_ik_frame\ef_launcher_probe.json` | **0** | 启动行 `... \| elbow follow = operator \| ...`（钉子确实到了 parser 与场景驱动），报告里有 `elbow_posture` |
| 同上 + `--off` | **0** | 启动行 `elbow follow = off`（对照档有效） |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | **0** | `未发现 GBK 无法编码的字符` |


### 8.18 「跑一会 GR1-T2 的大臂与小臂夹角反过来」：斜坡只削了腕那一点（2026-10-15）

现场原话：**「刚开始还行，跑一会大臂小臂夹角反过来」**，跑的是 §8.17.10 那个固化档。结论先给：

1. **不是「肘跟手」那一档带来的**（8.18.1）：同一份录制上 `--arm-elbow-follow off` / `operator`
   两档的命令点序列几乎逐位相同，这一档既不是病根也不是解药。
2. **病根在它下面两层的斜坡上**（8.18.2、8.18.3）：`--arm-map posture` 的两个命令点是**刚性**的
   一对（`wrist = elbow + L_f*v`），而 `ArmSceneDriver._slew_target` 只限速**腕**那一点。快速动作时
   肘目标一步跳到新位置、腕每帧最多走 40 mm，两点被拉近到「命令自己的前臂段从 250.6 mm 缩到
   51.1 mm」，那条方向就成了噪声，IK 只能把前臂折到反方向 —— 现场看到的「夹角反过来」。
3. **修法**（8.18.4、8.18.5）：`ArmSceneDriver.apply_each(..., elbow_rigid=...)`，由入口把「方向口径」
   那几台标成刚性；斜坡在**会压扁这一对的帧上不放行**，改为**保持上一帧下发的那一对**。四个候选
   修法的 A/B 表在 8.18.4：这一版把 GR1-T2 左臂的**前臂反向帧 7 -> 3**（剩下的 3 帧在未修复的
   版本里逐帧相同）、**压扁帧 11 -> 0**，而前臂抽动 / 腕残差 / 肘任务都不动，g1 的前臂间距还从
   0.55 mm 改善到 23.8 mm。

#### 8.18.1 先排除：这一档不是病根

同一条确定性命令（同一份 `live_long_frames.npz`、200 帧、`--arm-direction-rate-deg-s 0
--arm-slew-mm 40`），只差 `--arm-elbow-follow`，`--arm-frame-dump` 逐帧转储对比：

| 指标（gr1_t2 左） | `off` | `operator` |
|---|---|---|
| 命令自己的前臂段 `|腕-肘|` 最小值（mm） | 51.1 | 51.1 |
| 压扁帧（< 50% 名义 250.6 mm） | 11 | 11 |
| `forearm_dir_error` 中位（度） | 51.01 | 50.11 |
| 前臂反向帧（> 90 度） | 7 | 8 |
| `elbow_recoveries` / `bridge.counts` | 3 / 逐位相同 | 3 / 逐位相同 |

两档的**逐帧命令点差 0.2-1.2 度**（那是 §8.17 那一档的零空间参考在起作用），病根症状一模一样 ——
也就是说这一场「反」与 §8.17 的新功能无关，它只是把老问题一起带进了现场。

#### 8.18.2 抓到的帧：一对刚性目标被斜坡压扁

`posture_summary` 的 `worst_frame`（第 19 帧）原始数字：`wanted_fore [0.911, -0.387, 0.144]`、
`achieved_fore [-0.732, -0.291, -0.616]`、`forearm_error_deg 128.74`；命令点是
`elbow [0.108, 0.115, 0.292]` / `wrist [0.155, 0.095, 0.300]` —— **两点只差 51.1 mm**（该台自己的
前臂长是 250.6 mm）。逐帧位移（`--arm-frame-dump`，基座系）：

| 帧 | 肘命令这一步走了 | 腕命令这一步走了 | 命令的 `|腕-肘|` | IK 走到的前臂方向 vs 命令 |
|---|---|---|---|---|
| 17 | -（还没数据） | - | 250.6 mm | 一致 |
| 18 | **145.9 mm** | 42.4 mm | **79.3 mm** | 100.3 度 |
| 19 | 4.8 mm | 44.4 mm | **51.1 mm** | **128.7 度** |
| 20 | 26.3 mm | 44.1 mm | 61.4 mm | 125.2 度 |
| 21 | 32.9 mm | 40.8 mm | 93.8 mm | 91.5 度 |

读法：肘目标**不限速**、腕目标每帧最多 40 mm（记录里含逐帧肩平移，所以是 42-44 mm），于是这一对
被拉近；`L_f*v` 只剩 51 mm 时它的**方向**已经由噪声说了算，而 IK 要同时满足两个几乎重合的位置
目标 —— 走出来的手臂就是折回去的那一条。这正是「跑一会（动作快起来）才出现」的原因：站着不动时
腕的斜坡不会生效。

#### 8.18.3 复现与对照（命令 + 逐帧转储）

```
E:\python3.11.7\python.exe -m teleoperation.simulation.realtime_teleop --input-npz outputs\tmp_realtime\live_long_frames.npz --frames 200 --headless --arm-map posture --arm-direction-rate-deg-s 0 --arm-slew-mm 40 --report tmp_ik_frame\g_off.json --arm-frame-dump tmp_ik_frame\g_off_arm.npz
```

对照组只要把 `--arm-slew-mm 40` 换成 `--arm-slew-mm 0`（= 这一对不会被削）：命令的 `|腕-肘|`
**逐帧恒为 250.6 mm**、压扁帧 **0**、前臂反向帧 **7 -> 2** —— 这一步就把「压扁」与「IK 本身走
反」（§8.13/§8.14 那一类）分开了。逐帧分析工具：`tmp_ik_frame/probe_frames.py`（临时工具，不入库；
读 `--arm-frame-dump` 的 npz + `--arm-trace` 的日志）。

#### 8.18.4 四个候选修法的 A/B（同一份 200 帧、同一条命令，只差修法）

| 修法 | 命令 `[腕-肘]` 最小（mm） | 压扁帧 | 前臂反向帧 | 方向误差中位（度） | 上臂方向误差中位（度） | 前臂抽动 max（度） | 前臂间距 min（mm，g1） | 腕残差 h1_2 左右（mm） |
|---|---|---|---|---|---|---|---|---|
| V0 不改（肘新鲜、腕限速） | 51.1 | 11 | 7 | 51.01 | 66.75 | 26.41 | 0.55 | 33.36 / 31.31 |
| V1 两点**整体平移**（否） | 250.6 | 0 | **23** | 59.40 | 70.92 | 27.08 | 17.29 | 33.12 / 31.34 |
| V2 全程关斜坡（否） | 250.6 | 0 | 2 | 41.88 | 57.78 | **130.11** | 2.94 | **45.34 / 51.64** |
| V3 会压扁就按请求值一步到位（否） | 166.7 | 0 | **1** | 45.11 | 61.74 | **69.54** | 0.44 | 33.52 / 31.41 |
| V4 会压扁就**保持上一对**（采用） | **151.1** | **0** | **3** | 50.27 | 65.77 | **26.32** | **23.76** | 33.21 / 31.27 |

（前臂反向帧 / 方向误差 / 抽动都是 gr1_t2 左臂；腕残差两档判据 30 mm。V4 的
`target_slew_pair_skips` = 37 帧，V3 = 9 帧。）

读三条：

1. **V1（先试的「整体平移两点」）被数据否掉**：形状保住了，但肘目标被拖离配方点 —— 那条上臂方向
   就被污染了（`upper_err` 66.8 -> 70.9 度、前臂反向帧 7 -> **23**、g1 两侧与 h1_2 两侧全线变差）。
   「方向口径」的肘目标不只是「跟腕保持 L_f 的那一点」，它同时是**上臂方向**的载体（§8.11.1）。
2. **V2（全程关斜坡）方向指标最好，但代价不能收**：腕残差 h1_2 左 33.4 -> **45.3** mm 越过 30 mm
   判据、前臂抽动 max 26 -> **130** 度 —— 「目标到达前先慢慢走过去」这件事正是斜坡存在的理由
   （§8.9），不能为了这一项把它关掉。
3. **V3（会压扁的那帧按请求值一步到位）症状最好（7 -> 1 帧）但要跳一步**：前臂抽动 max
   26 -> 69.5 度，而「手臂会抽」正是 §8.13.4 刚修掉的东西。**V4 取折中**：不跳、也不压扁，
   残差与抽动按兵不动，间距还顺带改善（g1 0.55 -> 23.76 mm；那一项本来是「两前臂贴中线」的老问题）。
   代价是那 37 帧（200 帧里 18%）这一对**保持不动** —— 手上感觉是「动作快的时候机器人等一下再跟」。

#### 8.18.5 改动清单（都在既有文件里，约定 14）

| 文件 | 改了什么 |
|---|---|
| `src/teleoperation/simulation/arm_scene_driver.py` | `apply_each(..., elbow_rigid=None)`（新参数，默认 `None` = 老调用方逐字不变）；`SLEW_PAIR_FLOOR_RATIO = 0.6`；会压扁的帧改为「保持上一帧下发的那一对」（新状态 `commanded_pair`）；报告新增 `target_slew_pair_skips`（`describe()` 与 `summary()` 各一处） |
| `src/teleoperation/simulation/realtime_teleop.py` | 新增模块级 `rigid_elbow_robots(per_side)`：只把 `info["elbow_target"] == "posture"` 那几台标成刚性，并在 `apply_each` 时传 `elbow_rigid`（`--arm-elbow-task` 那条独立肘目标的路**不标记**，§8.15 的 A/B 继续逐字有效） |
| `tests/test_arm_ik_multi_robot.py` | `ArmTargetSlewTests` 新增 `test_the_cap_makes_way_when_it_would_squash_a_rigid_elbow_pair`（4 个 subTest：照旧限速 / 护栏保持上一对 / 未标记不护栏 / 第一帧还没有可保持的一对）；`FieldArmDefaultsTests` 新增 `test_only_the_posture_recipe_is_marked_as_a_rigid_elbow_pair` |

护栏的判据：限速后的 `[腕命令 - 肘命令] < 0.6 * [请求的 腕-肘]`（方向口径里右边恒等于该台自己的
前臂长，所以这实际上是一条**绝对**地板：GR1-T2 151 mm、H1-2 137 mm、G1 111 mm）。

#### 8.18.6 验收（命令 + 退出码 + 关键不变量）

| 命令 | 退出码 | 关键不变量 |
|---|---|---|
| 8.18.3 那两条 `--headless`（`--arm-slew-mm 40` / `0`，各带 `--arm-frame-dump`） | 0 / 0 | 40 档：命令 `[腕-肘]` 最小 151.1 mm（= 地板值）、压扁帧 0、前臂反向帧 3、`target_slew_pair_skips` 37；0 档：恒 250.6 mm、压扁帧 0、反向帧 2 |
| 同上 + `--arm-elbow-follow operator`（`tmp_ik_frame/g_on_g4.json`） | 0 | 前臂反向帧 4（`off` 档 3）、前臂方向误差中位 49.41（`off` 档 50.27）—— 这一档在修好之后仍然不恶化 |
| `E:\python3.11.7\python.exe -m pytest tests -q` | **0** | `371 passed, 3 skipped, 346 subtests passed`（上一轮基线 369 / 3 / 342；本轮新增 2 条用例 + 4 个 subTest；末尾 `nanobind: leaked ...` 是队友 `roboticstoolbox` 既有的引用计数提示，不是失败） |
| `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | **0** | `[OK]` |
| `E:\python3.11.7\python.exe devtools\check_no_c_drive.py --strict` | **0** | `[OK]`（七项缓存都在 `E:\cache`） |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -Base d8bca93` | **0** | `RESULT: [OK] all checks passed (2 warning(s))`；第 4 项 `809 teammate-owned files under baseline: 0 modified/deleted`；`unittest discover -> Ran 374 tests [OK (skipped=3)]` |

#### 8.18.7 如实说（这一轮没解决什么）

- **只有本机回放证据**：现场那一次（324 帧、`arm_run_20261015_141233.json`）报告里
  `target_slew_limited_frames=986`、GR1-T2 左臂前臂方向误差中位 46.9 度、反向帧 0.4%，与本节机制
  一致；但**没有现场对照**（现场跑一次要人站镜头前，而且必须按 q/Esc 退出报告才会落盘，见 §8.17.10）。
- **剩下那 3 帧「反」不是本节修的**：GR1-T2 左臂第 184-186 帧（104.7 / 105.1 / 92.5 度）在**未修复**
  的版本里逐帧数字相同，属于 §8.13/§8.14 那一类（IK 支路 / 目标本身），要单独查。
- **系统性的 45-50 度中位前臂方向误差没动**：GR1-T2 左臂在**每一版**（V0-V4）里都是 42-59 度，
  现场旧长跑里也有 34-51 度 —— 这是另一件事。
- **护栏的语义是「等一等」，不是「跳过」**：37/200 帧这一对保持不动，现场如果觉得「快动作时有点
  跟不上」，可以把 `SLEW_PAIR_FLOOR_RATIO` 调小（0.4 会让护栏更少介入）或换 V3 那条路（症状更好
  但要跳一步）—— 两条路的数据都在 8.18.4 的表里，换之前请重跑那一张表。
- **`--arm-lateral-guard-mm` 仍只动腕那一点**（默认 0，今天是关的）：一旦打开，这一对同样会被那个
  护栏压偏（`arm_realtime._apply_lateral_guard` 原地改的是腕目标），需要和本节同一套处理。留作下一轮。

