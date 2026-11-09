# 30 个仿真任务的「测试与评估」怎么做（批次 1、2 已落地）

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

