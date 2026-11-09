# 30 个仿真任务的「测试与评估」怎么做（批次 1 已落地）

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
| 2 | 任务评估入口：单机器人 + 任务物体 + 工作台标定 + 逐帧判成功 + 逐 trial 落 JSON | 否 | 待做 |
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

## 4. 与论文的偏差（写进任何对外材料，不要含糊）

| 项 | 论文 | 我们 |
|---|---|---|
| 仿真器 | NVIDIA Isaac Sim | PyBullet（约定 6 冻结） |
| 手腕 / 上体估计 | SMPLer-X + 身体参数标定 | MediaPipe pose 关键点 + 我们自己的 7 DoF IK |
| IK | PINK | PyBullet `calculateInverseKinematics`（+ 肘位置任务 / 零空间参考）|
| 任务资产 | 论文场景（含真实物体模型与质量 / 摩擦参数） | 简化的 box / cylinder 代理物体，判据也是代理判据 |
| 场景几何 | 按机器人身高布置 | **台面高度与物体位置尚未标定**（批次 2 修） |
| 被试 | 4 人 | 先 1 人 |

## 5. 相关

- 工具：`devtools/audit_tasks.py`（批次 1），用例：`tests/test_task_audit.py`
- 手臂链路与实测数字：[`multi-robot-arm.md`](multi-robot-arm.md)、[`RETARGETING_PIPELINE.md`](RETARGETING_PIPELINE.md)
- 任务代码（只读，队友维护）：`src/teleoperation/simulation/tasks/`、`environment.py`、`evaluation/metrics.py`

