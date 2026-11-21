# 30 个仿真任务的「测试与评估」怎么做（批次 1、2 已落地）

> **2026-11-21 最新状态**：三机器人场景验收已达到 `90/90`（30 任务 x H1-2 /
> GR1-T2 / G1），且排除名单为空。本文下方“19/30”“3 个假成功”等段落保留为
> 问题定位和修复过程记录，不再是当前状态。最新证据与修复清单见
> `docs/session-2026-11-19-task-batch.md` 的“2026-11-21 30 任务可评测性收口”一节。

> 目标：复现 TeleOpBench 论文 §3.1 / §4 的**仿真任务测试与评估**，先做**视觉（MediaPipe）**这条接口。
> 本文是这件事的**唯一入口文档**：口径、批次划分、已测数字、偏差声明、下一步。
> 最后更新：2026-11-09（批次 1：30 任务可评测性审计完成，工具 `devtools/audit_tasks.py`）。

## 1. 论文要的是什么（照着抄口径，不抄数字）

| 项 | 论文（TeleOpBench） | 我们对应物 |
|---|---|---|
| 任务 | 30 个分层任务，每个有**显式完成判据**（Table 1） | `src/teleoperation/simulation/tasks/` 注册的 30 个类，判据在各自 `check_success()` |
| 指标 | 每个 trial 记 **success + completion time** | `evaluation/metrics.py` 的 `MetricsTracker`（成功率 / 平均完成时间） |
| 量化范围 | 从 30 个里挑 **10 个代表任务**（push cube / pick cube / pick place cube / uprear cup / ball trashcan / ball mug / ball bimanual / pot bimanual / pot tomato plate / pen brushpot） | 同一批 10 个先做，其余 20 个按同一流程补 |
| 被试 | **4 名操作者** | 先 1 人跑通，再补人（论文口径才可比） |
| 视觉链路 | SMPLer-X 估上半身 + 缩放标定 -> PINK IK；手用 MediaPipe + Dex-Retargeting；Kalman 平滑 | 我们的 `realtime_teleop`：MediaPipe pose -> 手臂 IK（`arm_ik` + `arm_realtime`）；MediaPipe hand -> 18 维 -> 原装手映射 |
| 观测记录 | (a) 机器人状态向量 (b) 头戴第一视角 + 固定第三视角 (c) 物体位姿 | `simulation/sensors.py` + `data/recorder.py` |

论文结论里跟"视觉接口"直接相关的一条：**视觉在 Task 4（uprear cup）与 Task 7（ball bimanual）上明显差**，
完成时间最长。我们这条链路如果在这两个任务上也是短板，说明复现方向是对的。

## 2. 分四个批次做（每批一个 checkpoint + 一个 commit）

| 批次 | 内容 | 需要人在摄像头前吗 | 状态 |
|---|---|---|---|
| 1 | 30 任务可评测性审计（稳定 / 假成功 / 判据可触发 / 手臂可达） | 否 | **已完成**（本文 §3） |
| 2 | 任务评估入口：单机器人 + 任务物体 + 工作台标定 + 逐帧判成功 + 逐 trial 落 JSON | 否 | **已完成**（本文 §4） |
| 3 | 单任务端到端试点（pushcube，真摄像头），确认画面 / HUD / 判据 / 记录 | **是**（约 30 分钟） | 待做 |
| 4 | 10（再 30）个任务 x N trial，统计成功率与完成时间，出论文 Table 2 样式的表 | **是**（1 人约 35 分钟/10 任务，30 任务分 2 次现场） | 待做 |

## 3. 批次 1：30 任务审计（已完成）

命令与退出码（本机回落解释器，2026-11-09）：

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = "$pwd\src"; $env:PYTHONUTF8 = "1"
E:\python3.11.7\python.exe devtools\audit_tasks.py                     # 退出码 1 = 有任务不可评测
E:\python3.11.7\python.exe -m unittest tests.test_task_audit -v        # 退出码 0（7 tests OK）
```

报告：`outputs/tasks_audit/audit.json`（每任务的四项检查、oracle 摆法、漂移明细、三台机器人的可达性）。

### 3.1 五项检查怎么判

| 检查 | 判据 |
|---|---|
| load | `get_task(name)(client).reset()` 不抛异常 |
| stable | 空转 600 步（2.5 s）后：任一物体位移 <= 0.10 m、不掉到台面以下、没有 NaN |
| no_fp | reset 时与空转后 `check_success()` 都必须是 `False` |
| trig | 按 `ORACLE` 表把物体摆成满足判据的姿态后 `check_success()` 必须是 `True`（验证**判据逻辑**，不代表物理能做到） |
| reach | 肩 -> 肘 -> 腕连杆长之和当半径，量每个可搬物体（mass > 0，不含 table）到肩的距离；再网格搜索整体平移，看"抬台 / 前移"能不能救 |

### 3.2 结果：19/30 通过

| 任务 | 层 | 稳定 | 无假成功 | 判据可触发 | 结论 |
|---|:--:|:--:|:--:|:--:|---|
| pushcube | 1 | OK | OK | OK | OK |
| pickcube | 1 | OK | OK | OK | OK |
| pickplacecube | 1 | OK | OK | OK | OK |
| uprearcup | 1 | OK | OK | OK | OK |
| balltrashcan | 1 | OK | OK | OK | OK |
| rotatefaucet | 2 | OK | OK | OK | OK |
| rotatehearth | 2 | OK | OK | OK | OK |
| closemicrowave | 2 | OK | OK | OK | OK |
| openlaptop | 2 | OK | OK | OK | OK |
| ballmug | 2 | OK | OK | OK | OK |
| pourwater | 2 | OK | OK | OK | OK |
| liftmug | 2 | OK | **FAIL** | OK | FALSE_POSITIVE |
| openmicrowave | 2 | FAIL | **FAIL** | OK | FALSE_POSITIVE |
| opendrawer | 2 | FAIL | **FAIL** | OK | FALSE_POSITIVE |
| closedrawer | 2 | FAIL | OK | OK | UNSTABLE |
| breadtoaster | 2 | FAIL | OK | OK | UNSTABLE |
| ballbimanual | 3 | OK | OK | OK | OK |
| potbimanual | 3 | OK | OK | OK | OK |
| pottomato | 3 | OK | OK | OK | OK |
| pottray | 3 | OK | OK | OK | OK |
| stackboxes | 3 | OK | OK | OK | OK |
| panhearth | 3 | OK | OK | OK | OK |
| tidyuptable | 3 | OK | OK | OK | OK |
| pottomatoout | 3 | FAIL | OK | OK | UNSTABLE |
| plateoven | 3 | FAIL | OK | OK | UNSTABLE |
| pottomatoplate | 4 | FAIL | OK | OK | UNSTABLE |
| penbrushpot | 4 | FAIL | OK | OK | UNSTABLE |
| twistbottlecaps | 4 | FAIL | OK | OK | UNSTABLE |
| drawerbook | 4 | FAIL | OK | **FAIL** | NOT_TRIGGERABLE |
| stacktoyblocks | 4 | OK | OK | OK | OK |

### 3.3 四类问题（都是**场景 / 判据**的问题，不是链路的问题）

1. **假成功 3 个**（`liftmug` / `opendrawer` / `openmicrowave`）：reset 时判据已经成立，跑出来只能
   是 100%。例如 `liftmug` 判 `|盖高 - 杯高| > 0.05`，初始就差 0.055；`opendrawer` 判
   `|抽屉 x - 柜体 x| > 0.12`，初始 0.14；`openmicrowave` 初始门就是开的（差 0.15 m）。
   -> 这三个必须在评估层标 `criteria_invalid`（不计入成功率），或由批次 2 的场景覆盖修初始位姿。
2. **判据不可触发 1 个**（`drawerbook`）：它要求抽屉"关上"（`|抽屉 x - 柜体 x| < 0.17`），
   而抽屉被柜体挤飞（见第 3 类），永远不满足。
3. **场景不稳 10 个**：抽屉 / 门 / 盖子 / 笔 / 瓶与相邻物体**初始位姿重叠**，2.5 s 内被物理弹开
   0.10-0.60 m，`closedrawer` / `drawerbook` / `opendrawer` 的抽屉、`plateoven` 的烤箱门甚至掉到台面以下。
   实测漂移：`oven_door` 0.602 m、`drawer` 0.578 m、`lid` 0.218-0.219 m、`pen` 0.212 m、
   `door` 0.221 m、`button` 0.144 m、`bottle` 0.106 m。
4. **可达性 0/57**：工作台面在 z=0.42 m，机器人站着（URDF 底座固定、肩不能平移）：

   | 机器人 | 肩关节 z | 上臂 + 前臂 | 可达半径 | 57 个可搬物体在圈内 | 最近物体距离 | 最优统一平移后 |
   |---|---:|---:|---:|---:|---:|---|
   | h1_2 | 1.434 m | 0.403 + 0.177 | 0.580 m | **0 / 57** | 1.014 m | 55 / 57（dx=-0.66, dz=+1.00）|
   | gr1_t2 | 1.248 m | 0.261 + 0.108 | 0.369 m | **0 / 57** | 0.847 m | 27 / 57（dx=-0.70, dz=+0.86）|
   | g1 | 1.091 m | 0.252 + 0.141 | 0.393 m | **0 / 57** | 0.709 m | 27 / 57（dx=-0.72, dz=+0.72）|

   （可达半径只算连杆原点链长、不含手掌，是**保守下界**；最近物体都是 `penbrushpot/pot_holder`。）

### 3.4 批次 1 给批次 2 的输入（已定，不需再讨论）

1. **工作台必须标定**：台面从 z=0.42 抬到约 z=1.0，物体同时往机器人跟前收。
   在**我们的场景层**统一平移（`TaskScene` 里做，不改 baseline 的 task 代码），
   并按 §3.3 第 4 类那张表按机器人给一组（dx, dz）。
2. **假成功与不可触发的任务单独标**：`liftmug` / `opendrawer` / `openmicrowave` 记
   `criteria_invalid`；`drawerbook` 在抽屉修好之前不计入。
3. **不稳任务**：batch 2 的场景在 `task.reset()` 之后做"退让"（把重叠物体按不重叠的位姿放回去），
   或先空转 600 步再记初始状态；两条路都要在报告里写清楚改了什么。

## 4. 批次 2：场景层 + 任务评估入口（已完成）

工具：`devtools/task_eval.py`（场景层 + trial runner + `--verify`），用例扩到
`tests/test_task_audit.py` 的 17 个（批次 1 的 7 个 + 场景层 10 个）。

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = "$pwd\src"; $env:PYTHONUTF8 = "1"

# ① 验收：30 个任务过场景层，四列检查（稳定 / 无假成功 / 判据可触发 / 在可达圈内）
E:\python3.11.7\python.exe devtools\task_eval.py --verify --robot all      # 退出码 0
# ② 空转：不下发动作，判据不许自己变真
E:\python3.11.7\python.exe devtools\task_eval.py --task pushcube --trials 2 --policy idle
# ③ 假动作：证明动作真的送到关节上（joint_travel_rad = 2.07 rad）
E:\python3.11.7\python.exe devtools\task_eval.py --task pushcube --policy dummy --max-sim-s 3
# ④ 契约H 动作文件回放
E:\python3.11.7\python.exe devtools\task_eval.py --policy replay --actions datasets\actions\x.h5
```

报告：`outputs/task_eval/verify_all.json`（每任务 shift / clamped / compressed / separated /
settle / in_reach 全在里面），trial 报告默认 `outputs/task_eval/<robot>_<policy>_<N>.json`。

### 4.1 根因：30 个任务**全部**把物体中心摆在台面高度

`load_cylinder(name, r, height, ...)` 的 `height` 在 PyBullet 里是**全高**，而所有任务都把
物体的**中心**放在 `z=0.42`（台面顶面）。实测（`outputs/tmp_tasks/probe_sink.log`）：

| 现象 | 数字 |
|---|---|
| 陷进台面的任务 | **30 / 30**（漏出来的只有 `rotatehearth` 的旋钮） |
| 单个物体下陷 | 0.005 m（plate）~ 0.37 m（cabinet） |
| 只做"抬到台面上"就完全不漂的任务 | **18 / 30** |
| 还要再做"物体间退让"的任务 | 12 个（盖子陷在锅里、门陷在本体里、抽屉陷在柜体里、手柄陷在基座里） |
| 抽屉被弹飞 | 1.20 m（box 半宽 0.14 比柜体半宽 0.15 还大，实心体必然互穿） |

这就是批次 1 那 10 个"不稳"的物理来源 —— 不是链路的错，是**摆法**的错。

### 4.2 场景层怎么做（6 步，全部只改运行时位姿，不碰 baseline）

1. `SCENE_OPS`（只有 3 个任务需要）：抽屉换成 0.015 m 厚的**薄面板**（质量 / 摩擦沿用原体），
   `closedrawer` 摆在拉开位置、`drawerbook` / `opendrawer` 摆在关上位置。
2. **平移**（`solve_shift`，纯 numpy）：物体 + 家具一起搬到肩前 `0.66 x 臂长`、抬到离肩
   `0.8 x 臂长` 的球面上（**只抬不降** —— `pickcube` / `potbimanual` 的判据里有绝对 z）；
   桌子单独平移，近边留在机器人本体前 0.24 m，并保证**盖住**所有物体（`pottomatoplate` 的
   盘子离锅 0.2 m，桌子不跟着走就会从桌沿掉下去，实测掉 1.18 m）。
3. **落地支撑**：台面下面补一根从地面到台面的静态 box（`x >= 0.18`，不穿腿），
   抬到 1.2 m 的板子看起来才像桌子。
4. **抬到台面上**：AABB 底面低于台面的（含 static 的地标 / 柜体）逐个抬上来。
5. **退让**（`_separate`）：AABB 互相嵌入的按质量取"轻的那个"——浅嵌入 / 在上面的叠到顶上，
   其余沿最小嵌入轴推开（`lid` 叠到锅顶、`tomato` 摆到锅边、`oven_door` 挪到烤箱外）。
6. **收进可达圈**：按 AABB 连通分组（柜体+抽屉、烤箱+门算一套）平移；还放不下就
   以肩为圆心**等比收缩**（`compressed` 实测 0.9955-0.999，只有几毫米）；最后空转 600 步验稳定性。

**`sink` 模式**（只给 `pickcube` / `potbimanual`）：这两条的判据是绝对 z（`cube z > 0.66`、
`pot z > 0.58`），台面一抬就恒真。改成**桌面保持作者写的 0.42 m、把机器人底座降下去**
（fixed base，实测降 0.61-0.66 m），判据原意不变、物体照样进圈。

### 4.3 结果：72/90 格可评测（3 台机器人 x 30 任务）

| 机器人 | OK | CRITERIA_INVALID | CRITERIA_ABSOLUTE | SCENE_UNSUPPORTED | 最大漂移 | 台面高度 |
|---|---:|---:|---:|---:|---:|---|
| h1_2 | **24** | 3 | 2 | 1 | 0.051 m | 1.22-1.44 m |
| gr1_t2 | **24** | 3 | 2 | 1 | 0.051 m | 1.05-1.27 m |
| g1 | **24** | 3 | 2 | 1 | 0.094 m | 0.85-1.08 m |

* 30 个任务**全部** `stable`（< 0.10 m）+ 判据可触发；10 个**代表任务全部 OK**（批次 3 有任务可做）。
* 漂移剩下的最大 0.094 m（g1 的锅被两个手柄推）也在阈值内，且已写进 report。
* 18 格排除项不是"没测"，是**测出来是坏任务**，见 4.5。
### 4.4 trial runner：一次 trial 记什么

`evaluate_task()` 用 `SimulationEnv(record=False)`；**不用** baseline 的 `env.step()`
超时（它按 `time.time()` 算，空转一个 trial 也要等 300 s），改成自己数物理步（1/240 s），
每步 `task.check_success()`，成功即记 `completion_time_s`。每个 trial 落一条：

```json
{"trial": 0, "policy": "idle", "success": false, "completion_time_s": null,
 "steps": 720, "sim_time_s": 3.0, "joint_travel_rad": 0.0165, "wall_s": 1.05,
 "scene": {"shift": {}, "clamped": {}, "compressed": {}, "separated": [],
           "settle": {"stable": true, "max_drift_m": 0.0}, "in_reach": {},
           "criteria_at_reset": false},
 "final_objects": {"cube": [], "target": []}}
```

`joint_travel_rad` 是"动作到底有没有送到关节上"的证据：idle = 0.0165 rad（只在重力下微沉），
`--policy dummy` = 2.07 rad。

### 4.5 交给任务作者的问题清单（我们不改 baseline，只报）

| 任务 | 性质 | 证据 | 建议 |
|---|---|---|---|
| `liftmug` / `opendrawer` / `openmicrowave` | 判据在 reset 即为真 | 阈值 <= 自身半宽：盖子与杯高差 0.055 > 0.05、抽屉与柜体 x 差 0.14 > 0.12（柜体半宽 0.15）、门与本体差 0.15 > 0.12（本体半宽 0.15） | 阈值改成"相对初始位姿的位移" |
| `breadtoaster` | 判据永不成立 | 按钮初始 `z=0.48`、半高 0.01，正好躺在烤面包机上；判据要 `< 0.47` | 同上 |
| `rotatehearth` | 判据永不成立 | 旋钮躺在台面 `z=0.44`，判据要 `< 0.43` = 按进台面 | 同上 |
| `pottomatoout` | 场景摆不出来 | 判据要番茄"离开锅"，初始必须"在锅里"（`z=0.44 < 锅顶 0.48`），实心 cylinder 装不住球 | 换空心容器，或改成相对位移 |
| 全部 30 个 | **摆法** | `cylinder height` 是全高，物体中心却摆在台面 `z=0.42` -> 30/30 陷进台面 | 初始位姿按 `table_top + 半高` 写 |

`drawer` / `cabinet` 那组还有一条几何硬伤：抽屉的 box 半宽 0.14 比柜体半宽 0.15 大，
实心体必然互穿（我们只能换成薄面板绕开）。

### 4.6 批次 3 第一步：摄像头现场档（`--policy camera`）

`task_eval.py --policy camera` 把**正在用的那条链路**原样接到任务上，一行都没重写：

```text
CameraFrameSource.next_frame()   # MediaPipe 手 21 点 + pose 肩/肘/腕（米制）
  -> RealtimeArmBridge.update()  # 相对重定向，默认 --arm-map posture（方向口径）
  -> PyBulletArmIK.solve()       # 每侧 7 个臂关节角（腕 + 肘两个位置任务）
  -> 按 env.action_joint_names 落位 -> env.task.apply_action()
```

命令行（默认 DIRECT，加 `--render` 开 PyBullet 窗口，加 `--preview` 同时开摄像头预览窗）：

```powershell
E:\python3.11.7\python.exe devtools\task_eval.py --task pushcube --policy camera `
    --side right --render --preview --max-wall-s 60 `
    --json outputs\task_eval\camera_pushcube.json --snapshot-dir outputs\task_eval\snapshots
```

要点（都是代码里写死的原因，改之前先读这一段）：

* **一个摄像头帧 = 12 个物理步**（`--substeps`，默认 12 = 0.05 s 仿真）。MediaPipe 一帧
  30-50 ms，物理步是 0.1 ms 级；每步读一次摄像头会把"你手的 1 秒"压成机器人的 0.02 秒。
  12 步正好和实测 19-21 fps 的相机同节奏。
* **动作不是数组而是 callable**：`run_trial` 支持 `callable(step) -> action`，动作边走边算
  （`CameraPolicy`）。数组策略的老路径行为不变（用例
  `tests/test_task_audit.py::TrialRunnerTests` 用"两者同轨迹"钉住）。
* **手关节不动**：只驱动 `--side` 那一条 7 DoF 臂，18 维手角度保持 reset 值。pushcube 只需要
  手臂推；混进手部链路只会让失败原因难分辨。
* **现场先标定**：起手 1.5 s（30 帧）记中性姿势，控制台打 `cal=right=12/30` -> `cal=ok`，
  这段期间机器人不动。`cal=ok` 之后才开始跟手。
* **两个窗口各管一件事**：摄像头预览窗看骨架/手点有没有检到（左上角 `L=1 R=1`），
  PyBullet 窗口看机器人跟得对不对（方向反了调 `--arm-axis-order/signs`，不用改代码）。
* **证据字段**：trial 里多一段 `camera`：`frames` / `hand_frames{L,R}`（检到手的帧数）/
  `held_frames`（这一帧没跟手，含 `held_reason` 分类）/ `bridge_counts`（valid / posture /
  direction_glitch...）/ `last_target`（最后一帧目标 + gain 与姿态度量）。`--snapshot-dir`
  另外每 trial 落一张离屏 PNG（`task_eval.snapshot()`，`ER_TINY_RENDERER`，DIRECT 也出图）。

2026-11-09 实测（同一台机，摄像头 19.2 fps）：

| 项 | 值 | 说明 |
|---|---|---|
| 管线 | 通 | 238 帧、`joint_travel_rad = 1.16`（`--policy idle` 空跑只有 0.0165）|
| `pushcube` 成功 | 否 | 台面被抬到 1.34 m，方块离肩 0.51 m / 臂展 0.553 m（92%），要"先够到再推 0.54 m"，第一次试跑没做到 |
| 手部检到率 | 0/238 | 试跑时没人站到镜头前，pose 仍给出（多半是幻检）；现场必须先看 `L=1 R=1` 与 `cal=ok` |

结论口径：**链路已通、任务未解**。下一档要解决的是"够得着"（换更近的台面摆放或更简单的
任务），不是链路本身。

## 5. 与论文的偏差（写进任何对外材料，不要含糊）




| 项 | 论文 | 我们 |
|---|---|---|
| 仿真器 | NVIDIA Isaac Sim | PyBullet（约定 6 冻结） |
| 手腕 / 上体估计 | SMPLer-X + 身体参数标定 | MediaPipe pose 关键点 + 我们自己的 7 DoF IK |
| IK | PINK | PyBullet `calculateInverseKinematics`（+ 肘位置任务 / 零空间参考）|
| 任务资产 | 论文场景（含真实物体模型与质量 / 摩擦参数） | 简化的 box / cylinder 代理物体，判据也是代理判据 |
| 场景几何 | 按机器人身高布置 | 台面按机器人抬到 0.85-1.44 m、物体用同一套代理几何（批次 2 已标定）；`pickcube` / `potbimanual` 改用"降机器人"保绝对 z 判据 |
| 被试 | 4 人 | 先 1 人 |

## 6. 相关

- 工具：`devtools/audit_tasks.py`（批次 1 审计）、`devtools/task_eval.py`（批次 2 场景层 + trial runner）
- 用例：`tests/test_task_audit.py`（17 个：审计表 / 场景层 / trial runner）
- 手臂链路与实测数字：[`multi-robot-arm.md`](multi-robot-arm.md)、[`RETARGETING_PIPELINE.md`](RETARGETING_PIPELINE.md)
- 任务代码（只读，队友维护）：`src/teleoperation/simulation/tasks/`、`environment.py`、`evaluation/metrics.py`

## 7. 批次 5：无人脚本档（`--policy scripted`，2026-11-16）

> 批次 3 / 4 都标着「需要人在摄像头前」。本轮**没有人**、也不能等人 —— 所以把「人的动作」换成
> 一串确定性航点，走**同一条 IK + 同一条动作下发**，任务判据仍然由任务自己说。
> 工具：`devtools/task_scripted.py`（策略 + 任务脚本）。`--policy idle` 的旧行为一个字节没变。

### 7.1 先把四个「让够得着不成立」的根因修掉

| # | 现象（实测） | 根因 | 修法 |
|---|---|---|---|
| 1 | `in_reach` 说物体在圈内，IK 残差 **58-81 mm** | `measure_arm` 把「肩 link 原点→肘」+「肘→腕」当半径（h1_2 **0.5527 m**），可那两段在静止位姿并不共线：实测伸直只有 **0.4067 m** | `task_eval.sample_reach()`：按 URDF 限位随机采样 7 个臂关节、用 FK 直接量「腕离肩多远」，h1_2 = **0.4084 m**；中心也改成**肩关节轴**（不再是名字匹配到的肩 link） |
| 2 | 每条航点都差 ~0.12 m，手臂「顶住」不动 | `TaskScene._settle` 空转 600 步**没给电机**：手臂靠重力垂到腿里，手与大腿互穿后**自己卡自己**（`shoulder_pitch` 指令 −1.03 rad，实测停在 **+0.44 rad**；力 200/500/1200 一样，关掉重力也一样） | `_hold_motors()`：settle 前把全身可动关节按当前角锁住（与现场 `arm_scene_driver` 的 hold motors 同做法） |
| 3 | 第一次试跑就把方块**扫飞 0.5 m** | `PyBulletArmIK.solve()` 为跑第二遍会 `resetJointState`（机械臂**瞬移穿过**桌面上的物体） | 策略里每次求解前后**存/还原整机角度**，只把解写进动作向量，交给位置电机平滑走过去 |
| 4 | `lift`/`over` 都能到，`down` 永远差 0.09-0.19 m | 桌面顶面比肩关节**还高 5 cm**，桌子 + 落地支撑在 x≥0.18 是一堵墙 —— 垂着的手往前摆，在 z≈0.9 就怼上墙 | 航点固定四段：**收肘到胸前（x<0.18）→ 抬到桌面上方 → 平移到物体正上方 → 下探**；航点还夹进「肩关节 0.36 m」球内 |

另有一条纯口径问题：单侧 `ik_mode="pose"` 对同一目标残差 **80.8 mm**，`"position"` 只有 **1.1 mm**
（`outputs/tmp_probe/ik_variants.txt`）。脚本档不需要腕朝向（抓取靠约束），所以用 `position` + `lock_others`。

### 7.2 场景层新增两步（只在 `--policy scripted` 下生效）

1. **`_layout_for_arm()`**：按这台臂的真实几何重摆整幕 —— 物体质心放到「肩前 0.26 m、横向 ≤0.12 m」；
   相对布置**等比压缩**进 0.12 m（pushcube 的方块离目标 0.54 m，0.41 m 的手臂一次搬运走不完）；
   桌面顶面放到**比肩低 0.22 m**；桌子近沿留在 x≥0.24 m；落地支撑按新桌面高度**重建**（旧支撑会变成胸口的墙）。
2. **`_pull_into_ik_workspace()`**：用**真 IK 残差**（≤15 mm）复核每个物体，并把物体局部（≤0.30 m）
   挪到「残差合格且离肩 ≤0.80×reach」的最近点。半径模型只能保证「勉强够到」，从上方接近需要余量。

### 7.3 抓取与搬运（诚实声明）

* **抓取用约束辅助**：手（掌 + 五指任一 link）进到物体 **10 cm** 以内时，用
  `p.createConstraint(JOINT_FIXED)` 把物体刚性挂在手部 base link 上，松手时删掉。够不着就**不挂**，
  并记 `grasp MISSED (hand gap=…)` —— 隔空抓物会让成功率变成假的，所以留这道闸。
  纯接触抓 3-6 cm 的代理方块在 PyBullet 里成功率极低；本仓库没有触觉 / 摩擦标定。
* **搬运对物体闭环**：物体与手是刚性约束，腕一转「物体−腕点」的世界系偏移就变，所以每段都用
  **物体实测位置**修腕点目标（静态偏移补偿实测差 **12 cm**）。
* **判据在 reset 即为真 -> 不计成功**：场景摆完后 `task.check_success()` 若已经为真，这条 trial 记成
  失败并注明 `criteria_at_reset`（`balltrashcan` / `ballbimanual` 就栽在这里，如实报）。

### 7.4 怎么跑

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = "$pwd\src"; $env:PYTHONUTF8 = "1"

# 单个任务（带每段到位日志）
E:\python3.11.7\python.exe devtools\task_eval.py --task pushcube --policy scripted --scripted-trace

# 对照：关掉抓取约束（纯接触）—— 证明成功不是「隔空抓」来的
E:\python3.11.7\python.exe devtools\task_eval.py --task pushcube --policy scripted --no-grasp-assist

# 30 个任务全跑（无人，DIRECT，不开窗口）
E:\python3.11.7\python.exe devtools\task_eval.py --policy scripted --max-sim-s 30 --task <30 个名字> ^
    --json outputs\task_eval\scripted_all_h1_2.json
```

### 7.5 结果（2026-11-16 实测，h1_2，每任务 1 trial）

报告：`outputs/task_eval/scripted_all_h1_2.json`（每次 trial 的 `scripted_report.log` 里逐段写着
`wrist_err` 与 `grasp` 的实测间隙）。

| 任务 | 结果 | 完成时间 | 备注 |
|---|---|--:|---|
| pushcube | ✅ | 4.71 s | 抓→搬到 target 上方→松手，判据在搬运途中成立 |
| pickcube | ✅ | 2.78 s | 举高 0.30 m（判据是绝对 z） |
| pickplacecube | ✅ | 2.89 s | 放到盘子上方 |
| ballmug | ✅ | 3.07 s | 球放进杯子 |
| pourwater | ✅ | 3.96 s | 提起 + 前臂滚 80°（判据 roll > 60° 且 z > 0.5） |
| potbimanual | ✅ | 3.77 s | 举起锅 0.30 m |
| pottomato | ✅ | 2.87 s | 番茄放进锅 |
| panhearth | ✅ | 4.18 s | 锅放到灶上 |
| twistbottlecaps | ✅ | 3.55 s | 瓶盖放到瓶口 |
| uprearcup | ❌ | — | 抓到了、也举起了，但「拧正」那一步用的是 `elbow_roll`，杯子没转正 |
| balltrashcan | ❌ | — | 抓放路径撞到垃圾桶本体（2 cm 球滚走 2.3 m） |
| rotatefaucet | ❌ | — | 手柄抓到了，`elbow_roll` 转 100° 后手柄朝向判据未达成 |
| rotatehearth / breadtoaster | ❌ | — | 判据含**绝对 z**（`knob z < 0.43` / `button z < 0.47`），摆法一改就永远不成立（批次 1 已列 `CRITERIA_ABSOLUTE_Z`） |
| openmicrowave / opendrawer / liftmug | ❌ | — | reset 时判据即为真（批次 1 的 `CRITERIA_INVALID`）；本轮**如实记成失败**（`criteria_at_reset`），不刷成功率 |
| ballbimanual / pottomatoout / plateoven | ❌ | — | 场景摆完判据即为真（`z > 0.6` / 番茄已在锅外 / `door_closed` 本来就满足）-> 同样不计成功 |
| closemicrowave / closedrawer / openlaptop / drawerbook | ❌ | — | 需要「推/拉门、拉抽屉、翻屏幕」这类**保持接触的持续推压**，脚本档只有「抓起来搬」，没有推压控制 |
| pottray / stackboxes / tidyuptable / pottomatoplate / penbrushpot / stacktoyblocks | ❌ | — | 抓取/搬运途中撞到旁边的物体（这些任务的目标物彼此贴着），或落点差几厘米没进 `_is_above` 阈值 |

**结论口径（不许含糊）**：批次 5 做到的是「**在无人条件下，9/30 的任务能被脚本真正做完**，并且
失败原因逐条可查」。剩下的 21 条不是链路问题，分三类：①判据本身失效（7 条，批次 1 已标）；
②需要接触式推压（4 条）；③抓取路径碰撞/落点精度（10 条）。这三类分别对应下一批的三件事：
改判据为「相对初始位姿」、加「推压原语」、给抓取加局部避障与落点二次校正。

---

## 8. 摄像头任务档接上真实原装手（2026-11-18）

原来的 `--policy camera` 只驱动手臂，手部关节保持 reset 值；调用 `pickcube` 等抓取任务时，
摄像头手部链路根本没有进入 `env.action_joint_indices`。现在 `CameraPolicy` 在同一帧里做：

    MediaPipe 三帧手部窗口
      -> Geometric / checkpoint 手部重定向（18D L21）
      -> NativeHandAdapter（17D 语义自由度 -> 该机器人原装手）
      -> action[hand_joint_name]（按关节名，不按位置猜）

同时保留原来的 pose 手臂 -> RealtimeArmBridge -> PyBulletArmIK。手和臂分别写同一份完整动作向量，
然后 `TaskScene` / `Environment` 用任务自己的成功判据检查。

### 8.1 命令

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = "$pwd\src"; $env:PYTHONUTF8 = "1"

# 先做捏取验收：右手拿起 6 cm 方块
E:\python3.11.7\python.exe devtools\task_eval.py `
    --task pickcube --robot h1_2 --policy camera --side right `
    --preview --render --max-wall-s 60 `
    --json D:\teleop_reports\pickcube_camera_h1.json

# A/B：完全恢复“只驱动手臂”的旧行为
E:\python3.11.7\python.exe devtools\task_eval.py `
    --task pickcube --robot h1_2 --policy camera --side right `
    --camera-hand off --preview --render --max-wall-s 60
```

默认参数：

- `--camera-hand on`
- `--hand-backend auto`（有 checkpoint 用 checkpoint，没有回退 geometric）
- `--hand-limit-mode clamp`（忠实语义；`rescale` 只作视觉辅助）
- 手部和手臂共用前 30 帧站立标定；手臂要求中性姿势，手部要求**该侧手掌张开**

手部报告新增：

- `camera.hand_valid_frames` / `hand_invalid_frames`
- `camera.last_hand.mapped_joints`（这帧实际写进原生手的关节）
- `camera.hand_clipped_joints`（原生行程截断次数）
- `camera.hand_calibration_status`

摄像头档与脚本档现在共用 `_layout_for_arm()` + `_pull_into_ik_workspace()`：
`pickcube` 的旧摆法实测让手掌/掌根接触桌面（腕点残差约 **95 mm**、接触法向力最高
3914 N），新的真实 IK 工作区摆法把桌面放在肩下 0.22 m、物体放到真 IK 可达点后，残差约
**11 mm**，抓取前接触力为 0。任务报告新增 `robot_object_contacts`，下一条 trial 可以直接
核对手臂是否仍撞桌。

### 8.2 验收顺序

相机任务档**不使用** `task_scripted.py` 的 `grasp_assist`；物体只能靠真实接触抬起。因此先做：

1. `pickcube`：确认拇指 + 食指/中指捏取能抬起 6 cm 方块；
2. `balltrashcan`：确认 3 cm 球形物体的稳定性；
3. `uprearcup`：确认侧放杯子的包络抓取与转正；
4. `pourwater`：确认水壶 + 腕部滚转；
5. `potbimanual`：最后验证双手大物体。

如果第 1、2 项成功而第 3 项失败，问题优先归到包络抓取而不是基础手部映射；如果第 1 项就失败，
先看 `unmapped_joints`、`hand_clipped_joints` 和该机器人原生手映射，而不是先改 30 个任务。

验收（2026-11-18）：`pytest tests -q` -> **517 passed, 3 skipped, 464 subtests passed**；
摄像头短冒烟报告 `camera_hand_smoke.json` 退出码 0，H1-2 的 12 个原生手关节、geometric
后端与任务动作向量均已加载。短冒烟时画面无人，所以 `hand_valid_frames=0` 是输入侧事实，
不是接线失败。

2026-11-18 现场复测（`pickcube_camera_h1_live_v4.json`）：摄像头 992 帧里右臂有效
822 帧、右手最终有效并写入 12 个原装关节，`joint_travel_rad=1.5274`，末端
`robot_object_contacts={}`（没有机器人 link 与桌/方块接触）。方块从
`(0.303, 0, 0.520)` 被轻微推到 `(0.346, -0.022, 0.520)`，但没有抬起，因此失败点已从
“卡桌/手臂不动”转到抓取接触与闭合时机。下一轮只调局部抓取，不再改任务布局。

另外，现场如果把右手继续往下压，摄像头目标仍可能指定到桌面实体内部，造成“手又钻到
桌下并卡住”。摄像头档新增桌面目标护栏：目标位于桌面 x/y 投影内时，腕点至少保持在
桌面上方 `--camera-table-clearance-mm`（默认 60 mm）；桌前沿以外仍允许向下。报告记录
`camera.table_guard_count` 与 `camera.last_table_guard`，可直接区分“输入想穿桌”和
“机械臂自己穿桌”。

### 8.3 摄像头任务“手在动、机械臂完全不动”（2026-11-18）

根因不是摄像头，也不是 hold 电机：`CameraPolicy` 创建 `PyBulletArmIK` 时没有覆盖
`ik_mode`，因此 H1-2 沿用了 profile 的 `pose` 模式。摄像头任务只提供位置目标；第二遍
姿态 IK 会把位置解重新拽回近似加载姿态，结果恰好是：

- 手指走另一条直接关节链，能抽搐/开合；
- 七个臂关节看起来完全不动。

修法：

- 摄像头任务默认显式 `--camera-ik-mode position`；
- 手臂和手指分离施力：`--camera-arm-force 1000`、`--camera-hand-force 300`；
- 每 20 帧打印目标腕点、实际腕点与滞后毫米数；
- trial 报告记录 `last_arm_target_world` / `last_arm_world` / `last_arm_error_mm`。

v11 实测中，连续两帧从目标 `[0.061,-0.063,0.346]`、实际
`[0.063,-0.061,0.343]`（4.2 mm 滞后）到快速横移后的 199.6 mm 暂态，随后收敛到
31.8 mm；机械臂已实际跟随。该次后段长时间无人，最终没有完成抓取，因此不能把这次
`success=false` 当作握力失败。

进一步现场复测暴露了大块代理桌的碰撞问题：桌面虽然被腕点护栏挡住，但上臂/前臂仍会与整块
桌体碰撞，侧抬只能到一半；手指靠近桌面时又被接触力反复弹开。摄像头任务现在额外关闭
**机器人—桌子**碰撞（`disable_robot_table_collision()`），保留机器人—方块、机器人与地面的
碰撞以及桌面的承物作用。报告场景段写
`scene.robot_table_collision_links_disabled`。

最终摄像头档默认值：

- layout：肩前 `0.36 m`，桌面高度 `shoulder_z - 0.18 m`；
- `--camera-ik-mode position`；
- `--camera-arm-force 3000` / `--camera-hand-force 300`；
- `--camera-target-speed-mm-s 300`；
- `--camera-table-clearance-mm 60`；
- `--hand-smoothing 0.25`；
- `CameraPolicy` 桥开启实时档同款 `jitter_filter=True`，并对 3D 手腕目标做动态限速。

v15 现场报告：右臂有效 851/950 帧、右手最终有效并写入 12 个原装关节，
`table_guard_count=298`、`target_slew_limited_frames=638`，最终
`robot_object_contacts={}`。此时机械臂/手的抖动与卡桌已分离；方块仍在桌面上，
下一阶段只处理“抓取接触深度与闭合时机”。


