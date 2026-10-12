# 多机器人手臂驱动：按 robot_type 选臂并驱动机器人关节

最后更新：2026-10-12 | 分支：cs/multi-robot-arm（基线 10ebd9f，队友线 HEAD d8bca93）
相关代码：`src/teleoperation/simulation/arm_profiles.py`、`arm_ik.py`、`arm_scene_driver.py`
相关测试：`tests/test_arm_ik_multi_robot.py`（16 个用例）

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
