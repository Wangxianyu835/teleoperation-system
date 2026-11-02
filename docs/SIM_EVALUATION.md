# 30 个仿真任务的判据与场景体检（2026-11-02）

本文记录 `src/teleoperation/simulation/tasks/**` 里 30 个内置任务的**可评测性**修复：
判据不再白送、场景不再自己动、门和抽屉真的能推。不谈策略强弱，只谈
「任务本身能不能被公平地打 0 分或 1 分」。

## 0. 结论与证据

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 任务体检 | `E:\python3.11.7\python.exe devtools\audit_task_criteria.py` | `PASS=30 FAIL=0`，退出码 0 |
| 任务回归 | `E:\python3.11.7\python.exe -m pytest tests\test_simulation_tasks.py -q` | `7 passed` |
| 端到端报表 | `E:\python3.11.7\python.exe -m teleoperation sim run --benchmark --trials 1 --robot g1 --no-render --no-record` | 5 行 0.0%（演示控制器）、`EpTime(s)` 7.6~8.1 s、退出码 0 |
| 全量回归 | `E:\python3.11.7\python.exe -m pytest tests -q` | `580 passed, 4 skipped`，退出码 0 |
| GBK 安全 | `E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict` | `[OK]`，退出码 0 |

报表里 0.0% 是**演示控制器（正弦摆动）本来就做不成任务**，不是任务坏了 ——
0 / 30 的白送率才是检验任务的标准，见下面的体检。`AvgTime(s)` 印 `-` 表示
没有成功 trial（旧版印 `inf`）；失败 trial 的耗时看 `EpTime(s)`。

体检脚本的三种判定（默认静置 240 步 = 1 s，阈值 20 mm）：

- `t0_success`：t=0 时 `check_success()` 就是 True -> 一次 episode 白送成功。
- `settle_drift`：不给任何动作，物体自己动了 > 20 mm（穿模被弹飞、自由刚体掉落）。
- `settled_success`：静置后判据自己成立 -> 重力替操作者完成了任务。
- `joint_stuck`：门/抽屉挂在关节上却驱动不动 -> 任务永远做不成。

修复前（同一脚本，`robot=g1`）：**t=0 成功 3/30**（`liftmug`、`opendrawer`、
`openmicrowave`），**静置 1 s 自移动 > 20 mm 的 21/30**（`closedrawer` / `drawerbook`
0.280 m、`plateoven` 0.149 m、`openmicrowave` 0.140 m）。修复后 30/30 全过。

## 1. 三类根因

### 1.1 物体摆进桌面（21/30）

`load_fixed_table()` 的桌面高是 `0.40 + 0.02 = 0.42` m，而所有任务都把物体中心写成
`z = 0.42` 或 `0.43`，**忽略了自己的半高**：圆柱 h=0.12 的瓶子里嵌进桌面 6 cm，
烤面包机（半高 0.1）嵌进 10 cm。PyBullet 会先把它弹起来再落地，这一跳被测成
「物体自己动了」——和策略一点关系都没有。

修法（`BaseTask`）：

- 建体时登记几何（`_geometry`：box/cylinder/sphere + 尺寸）；
- `_z_half_extent()` / `_xy_half_extents()` 按**当前姿态**算世界系半高（圆柱侧倒时
  半高从 h/2 变成 r）；
- `snap_to_supports()` 在 `reset()` 末尾把「陷进支撑面」的物体**只抬不压**地贴到
  支撑面上（桌面 / 内腔底面 / 其他物体顶面），迭代两轮处理「a 落 b、b 被抬起」的链式情况；
- 域随机化之后再吸附一次并重记初始位姿（`SimulationEnv.reset`），否则 ±1 cm 的 z 抖动
  又会把物体压进桌面。

### 1.2 判据是绝对阈值（3/30 白送）

旧判据形如「门离机身 > 0.12 m」「杯盖比杯身高 0.05 m」，而初始摆位恰好满足。
修法：`reset()` 记录 `initial_poses`，判据改用相对量——

| 助手 | 含义 |
| --- | --- |
| `displacement(name)` | 相对初始位置的位移（米） |
| `rise(name)` | 相对初始位置的抬升（米，可为负） |
| `tilt_deg(name)` / `yaw_change_deg(name)` | 相对初始姿态的转角（度，后者带符号） |
| `is_over(obj, anchor, tol)` | 水平投影是否落在 anchor 上 |
| `is_inside(name, cavity)` | 是否落在 `load_open_box()` 的内腔里 |

### 1.3 门和抽屉是自由刚体（0.28 m 掉落 + 判据不可达）

`door` / `drawer` 原本是 `load_box()` 出来的自由刚体，摆在柜体内部高度（和实心柜体
体积重叠），静置就被挤飞。修法：真关节。

| 构件 | 用什么 | 为什么 |
| --- | --- | --- |
| `load_hinged_panel()` | `createMultiBody`：质量 0 基座（放在 pivot）+ 1 个 `JOINT_REVOLUTE` link | 轴默认竖直 `(0,0,1)`，重力对门不产生力矩 -> 不推就不动；面板几何用 `collisionFramePosition/visualFramePosition` 偏到 pivot 之外，转起来才是「绕合页转」而不是「绕自己中心转」 |
| `load_sliding_panel()` | 同样的多体 + `JOINT_PRISMATIC` link，行程用 `jointLowerLimit/jointUpperLimit` 限定 | 轨道水平，重力不产生沿轨道的力 -> 抽屉不会自己滑出/滑回 |
| `load_open_box()` | 5 块薄板拼上方开口的壳 | PyBullet 的 box 没有内腔，「把盘子放进烤箱」「把笔插进笔筒」物理上不可达；没有壳时物体只能停在实心块顶上，判据只能退化成「在它上方」 |

踩过的坑（写在这里省下一个人一天）：

1. `p.JOINT_HINGE` 不存在，PyBullet 里叫 `p.JOINT_REVOLUTE`；
2. 本机 pybullet (202010061) 的 `p.createConstraint` **只接受 `JOINT_PRISMATIC` /
   `JOINT_FIXED`**，`JOINT_REVOLUTE` 直接 `createConstraint failed`（探针见
   `tmp_arm_main/probe_constraint4.py`）；
3. 对「静态底座多体」的 link 调用 `p.applyExternalForce` **完全没有反应**
   （200 N 推不动 0.15 kg 抽屉），但关节电机和 `p.applyExternalTorque` 正常；
   所以体检脚本用电机驱动代替「机器人推」来验可动性；
4. 门/屏幕与家具本体之间必须留出 >= 2 cm 的真实缝隙（Bullet 接触余量会把 5 mm
   当成接触），或者用 `ignore_collision_with` 关掉这对碰撞，否则门一推就顶死。
5. 竖直轴铰链自带「不白送」性质，水平轴（笔记本屏）靠 `hold_force` 当合页摩擦
   顶住重力，判据只在撑住的那一刻成立。

## 2. 逐任务改动

除了下面这些，其余任务的场景与判据**只**受益于 1.1 的贴面吸附，代码未动。

| 任务 | 旧症状 | 现在的判据 / 摆位 |
| --- | --- | --- |
| `liftmug` | t=0 成功（0.055 > 0.05） | 杯盖 `rise > 5 cm`；杯身贴桌面、杯盖贴杯顶 |
| `opendrawer` | t=0 成功、抽屉被挤飞 0.28 m | 抽屉挂滑轨（0~0.20 m），`拉动 > 12 cm` |
| `closedrawer` | 抽屉被挤飞 0.28 m、判据随时蒙对 | 抽屉初始拉出 15 cm，`推回 > 10 cm` |
| `openmicrowave` | t=0 成功、门掉 0.14 m | 门是竖直轴铰链（初始关闭），`往外开 > 60 度` |
| `closemicrowave` | 门被重力摆进 2 cm 判据 | 门初始开 95 度，`推回 > 60 度` |
| `drawerbook` | 抽屉埋在柜体里、书停在抽屉顶上也算成功 | 抽屉改成开口托盘，判据 `书在内腔里`；**去掉「再把抽屉关上」子目标**（要能关就得是滑轨刚体，就不可能有内腔） |
| `plateoven` | 烤箱实心、门自由刚体掉 0.15 m | 烤箱改开口壳（盘子从上面放进去）+ 门挂铰链，判据 `盘子在内腔 且 门关上` |
| `breadtoaster` | 按钮埋在箱体内、判据要求 z<0.47 不可达 | 机身改开口壳 + 侧面压杆挂滑轨，判据 `面包在内腔 且 压杆推 > 2 cm` |
| `rotatefaucet` | 手柄悬空掉落、判据看绝对欧拉角 | 手柄挂竖直轴铰链（轴过基座中心），`转到 > 80 度` |
| `rotatehearth` | 「按下到 z<0.43」= 往桌面里按 3 cm，永远不可达 | 旋钮挂竖直轴铰链，`转过 > 25 度`；**去掉「按下」子目标** |
| `openlaptop` | 屏幕自由刚体、判据看绝对欧拉角 | 屏幕绕后缘水平轴合上，判据 `转 > 15 度 且 抬起 > 3 cm` |
| `pourwater` | 判据不提杯子在哪、壶嵌进桌面 6 cm | `壶倾斜 > 60 度 且 水平位置对上杯子` |
| `penbrushpot` | 笔筒实心，笔可停在筒顶 | 笔筒改开口壳，判据 `笔在筒内腔里` |
| `pottomatoout` / `pottomatoplate` | 锅实心、番茄摆在锅心（不可能） | 锅改开口壳（番茄真在锅内、盖压锅沿），判据 `不在锅内` / `在盘子上且不在锅内` |
| `potbimanual` | 两个手柄与锅体重叠被弹开 | 手柄移到锅两侧，留 1 cm 间隙 |

## 3. 以后改任务请守的规矩

1. **不要在 `_set_initial_positions` 里手写「桌面高度 + 常量」**：直接给一个大致位置，
   让 `snap_to_supports()` 贴面（它只抬不压，叠放关系保持不变）。
   要摆到「家具内部」就用 `load_open_box()` 的返回值定位。
2. **判据必须相对初始位姿**：用 `rise/displacement/tilt_deg/yaw_change_deg`，
   不要写 `z > 0.66` 这种绝对阈值。
3. **改完任务必须跑体检**：`devtools/audit_task_criteria.py`（退出码 0）+
   `pytest tests/test_simulation_tasks.py`。新任务要同时进
   `tasks/registry.py` 的难度表。
4. 关节体（门/抽屉/压杆）会被自动排除在域随机化之外（`jointed_objects`），
   不要手工去搬它们。

## 4. 已知限制（别把体检通过当成「任务已验收」）

1. **判据可达性 != 策略成功率**。体检只证明「驱动关节/物体能满足判据」，
   还没有一个能从观测走到判据的控制器。`sim run --benchmark` 里内置的
   sinusoid 演示控制器在所有任务上都是 0.0%（2026-11-02 实测 5 个任务），
   这属于策略问题，不是任务问题。
2. `openlaptop` 的水平铰链在本机实测最多转到 24~33 度（再大就顶住，原因未查清），
   判据是按这个可达范围定的（15 度 / 3 cm）。换 pybullet 版本要重测。
3. `rotatehearth` 去掉了「按下」、`drawerbook` 去掉了「关抽屉」，
   这两个子目标在「实心块 + 刚体」的建模下不可达，需要真正的柔性/内腔模型才能加回来。
4. 域随机化仍会给物体 ±2 cm 的水平抖动；关节体的「关节角」不随机（只随机摩擦/颜色）。
5. `task.max_time` 是**墙钟** 300 s，和 `max_steps`（10000 步 @240 Hz 约 42 s 仿真时间）
   不同量纲；耗尽的 episode 现在也会记进 `MetricsTracker`（见提交 4d78359），
   但 `elapsed` 仍是墙钟，跨机器不可比 —— 建议后续改成固定步数。
6. 体检脚本只跑 1 s。长时间漂移（例如接触力累积导致的缓慢滑移）还没覆盖。

## 5. 下一步

1. 一个能真正走通任务的参考控制器（或回放一段真人遥操作），把「接口 -> 任务」
   的成功率打出来，作为 30 个任务的上限基准；
2. 判据加「持续窗口」（连续 N 步为真才算成功），避免一次碰撞的瞬时状态被算作成功；
3. `elapsed` 换成仿真时间，`--benchmark` 表里同时打印仿真秒与墙钟秒；
4. 把体检接进 CI（退出码 0/1），任务改动必须过门。
