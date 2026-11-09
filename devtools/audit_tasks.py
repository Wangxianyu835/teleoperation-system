"""30 个操作任务的「可评测性」审计 —— 任务评估的第 1 步，只读、不改队友代码。

为什么要这个工具
----------------
论文（TeleOpBench）的任务级评测口径是「每个 trial 记 success + completion time」，
所以能不能测，先要确认三件事在**我们的 PyBullet 场景里**成立：

1. 场景摆出来是稳的（不会一放手物体就飞 / 掉出桌面）；
2. 判据在 reset 时是 False（否则不管人做什么都是 100%）；
3. 判据是能被满足的（否则那个任务永远 0%）；
4. 任务物体在机器人**手臂可达范围**内（否则 0% 不是"人做得差"，是场景不可达）。

实测（2026-11-09 第一次跑，``--steps 600``）只有 19/30 通过，四类问题全都出现了：

* **假成功 3 个**：``liftmug`` / ``opendrawer`` / ``openmicrowave`` 在 reset 时判据
  就已经为真 —— 不管人做什么都是 100%；
* **判据不可触发 1 个**：``drawerbook``（要求抽屉「关上」，而抽屉被柜体挤飞）；
* **场景不稳 10 个**：抽屉 / 柜体 / 门 / 盖子初始位姿互相重叠，2.5 s 内自己动了
  0.10-0.60 m，``closedrawer`` / ``drawerbook`` / ``opendrawer`` 的抽屉甚至掉到台面下；
* **可达性 0/57**：工作台面在 z=0.42 m，而 H1-2 肩关节在 z=1.434 m、肩到腕只有
  0.579 m（GR1-T2 0.369 / G1 0.393）—— 57 个可搬物体没有一个在可达半径内，
  最近的是 ``penbrushpot/pot_holder``，距离 0.71-1.01 m。把整个场景统一平移
  （机器人往桌前挪 + 抬台）能救回 h1_2 的 55/57、GR1-T2 与 G1 各 27/57。

先有这张表，才谈得上"成功率"；否则测出来的是场景的毛病，不是人（或链路）的。

批次 2 的场景层（把台面按机器人抬到作业高度、物体抬到台面上、互相嵌入的退让）在
``devtools/task_eval.py``；本工具是它的**输入**（先量 baseline 有多坏，再量场景层修好了多少）。


五项检查（每个任务）
--------------------
load              ``get_task(name)(client).reset()`` 不抛异常
stable            空转 ``--steps`` 步（默认 600 = 2.5 s）后：没有物体位移 > 0.10 m，
                  没有物体落到台面以下，没有 NaN
false_positive    reset 时与空转后 ``check_success()`` 都必须是 False
triggerable       把物体摆成满足判据的姿态（下面 ORACLE 表）后 ``check_success()``
                  必须 True。**它验证的是判据逻辑，不是物理可行性** —— 物理可不可行
                  由 reach 那一段回答，两者分开看
reach             每台机器人「肩 -> 肘 -> 腕」的连杆长之和当可达半径，量每个可搬物体
                  （mass > 0，排除 table）到肩关节的距离；再用 (dx, dz) 网格搜索整体平移
                  场景，看"抬台 / 把物体往机器人跟前挪"能不能让全部物体进圈

用法::

    E:\\python3.11.7\\python.exe devtools\\audit_tasks.py
    E:\\python3.11.7\\python.exe devtools\\audit_tasks.py --task pushcube liftmug
    E:\\python3.11.7\\python.exe devtools\\audit_tasks.py --no-reach --steps 240
    E:\\python3.11.7\\python.exe devtools\\audit_tasks.py --json outputs\\tasks_audit\\audit.json

退出码：0 = 30 个任务四项检查全通过；1 = 有任务失败（假成功 / 判据不可触发 / 不稳定）；
2 = 工具自身异常（连不上 PyBullet、任务名写错等）。

范围声明
--------
* 只读：只 ``reset()`` / 查位姿 / 查判据，不写任何文件到 ``src/teleoperation/**``；
* ORACLE 里的摆法只改物体的 base pose，**不推进物理**，所以不改变任何任务代码；
* reach 用的是连杆原点链长（不含手掌），是**保守下界**：真解出来的可达范围只会更大一点点。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: 三台论文机器人（与 applications/three_robot_scene.ROBOT_ORDER 一致）
ROBOTS = ("h1_2", "gr1_t2", "g1")
TABLE_KEY = "table"
#: 「稳定」的阈值：空转后任一物体位移超过它就报 UNSTABLE
DRIFT_M = 0.10
#: 低于「台面 z - 这个值」算掉出台面（台面 z 取 table 物体的 base 位置）
BELOW_TABLE_M = 0.05
DEFAULT_STEPS = 600
DEFAULT_JSON = ROOT / "outputs" / "tasks_audit" / "audit.json"

#: 把物体摆成「满足判据」的姿势。键 = 任务名，值 = 一串 op，按顺序执行：
#:
#:   ("copy_pose", obj, ref)        位置 + 姿态都抄 ref（推方块 / 放到盘子上）
#:   ("align", obj, ref)            只把 xy 抄 ref（关门 / 关抽屉 / 盘子进烤箱）
#:   ("above", obj, ref, h)         xy = ref 的 xy，z = ref 的 z + h（_is_above 口径）
#:   ("set_pos", obj, (x, y, z))    直接给绝对位置
#:   ("set_z", obj, z)              只改高度（抬起 / 按下）
#:   ("raise", obj, dz)             在当前高度上抬 dz
#:   ("move_by", obj, (dx, dy, dz)) 相对平移
#:   ("rotate", obj, (rx, ry, rz))  欧拉角（弧度）
#:   ("upright", obj, z)            摆正（单位四元数）并给高度
#:
#: 每条都对着 ``simulation/tasks/*.py`` 里那个任务的 ``check_success`` 写，
#: 所以任务判据一改，这里必须同步改（tests/test_task_audit.py 会盯着键集合 = 30 个）。
ORACLE = {
    # ---- Level 1 基础拾取放置 ----
    "pushcube": [("copy_pose", "cube", "target")],
    "pickcube": [("set_z", "cube", 0.72)],
    "pickplacecube": [("above", "cube", "plate", 0.06)],
    "uprearcup": [("upright", "cup", 0.46)],
    "balltrashcan": [("above", "ball", "trashcan", 0.02)],
    # ---- Level 2 工具操作 ----
    "rotatefaucet": [("rotate", "handle", (0.0, 0.0, 1.6))],
    "rotatehearth": [("set_z", "knob", 0.40), ("rotate", "knob", (0.0, 0.0, 1.0))],
    "openmicrowave": [("move_by", "door", (0.20, 0.0, 0.0))],
    "closemicrowave": [("align", "door", "microwave_body")],
    "opendrawer": [("move_by", "drawer", (0.10, 0.0, 0.0))],
    "closedrawer": [("align", "drawer", "cabinet")],
    "liftmug": [("raise", "lid", 0.10)],
    "openlaptop": [("rotate", "laptop_screen", (1.2, 0.0, 0.0))],
    "ballmug": [("above", "ball", "mug", 0.06)],
    "pourwater": [("set_z", "kettle", 0.55), ("rotate", "kettle", (1.2, 0.0, 0.0))],
    "breadtoaster": [("above", "bread", "toaster", 0.06),
                     ("set_z", "button", 0.44)],
    # ---- Level 3 双手协作 ----
    "ballbimanual": [("set_pos", "ball", (0.5, 0.0, 0.65))],
    "potbimanual": [("set_z", "pot", 0.62)],
    "pottomato": [("above", "tomato", "pot", 0.06)],
    "pottray": [("above", "pot", "tray", 0.08)],
    "stackboxes": [("above", "box_a", "box_b", 0.055)],
    "panhearth": [("above", "pan", "hearth", 0.07),
                  ("move_by", "lid", (0.15, 0.0, 0.0))],
    "tidyuptable": [("above", "item1", "basket", 0.07),
                    ("above", "item2", "basket", 0.07)],
    "pottomatoout": [("set_pos", "tomato", (0.5, 0.3, 0.45))],
    "plateoven": [("align", "plate", "oven"), ("align", "oven_door", "oven")],
    # ---- Level 4 长时域序列 ----
    "pottomatoplate": [("above", "tomato", "plate", 0.06)],
    "penbrushpot": [("above", "pen", "pot_holder", 0.05)],
    "drawerbook": [("above", "book", "drawer", 0.08)],
    "twistbottlecaps": [("above", "cap", "bottle", 0.04)],
    "stacktoyblocks": [("above", "block_b", "block_a", 0.085),
                       ("above", "block_c", "block_b", 0.17)],
}



def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def fresh(client: int, ground: bool = True) -> None:
    """把一个 client 重置成干净场景（每个任务 / 每台机器人都从零开始，互不串味）。"""
    import pybullet as p
    import pybullet_data

    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    if ground:
        p.loadURDF("plane.urdf", physicsClientId=client)


def poses(client: int, objects: dict) -> dict:
    """{物体名: (位置 3, 四元数 4)}，都是 float ndarray。"""
    import pybullet as p

    out = {}
    for key, body in objects.items():
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
        out[key] = (np.asarray(pos, dtype=float), np.asarray(orn, dtype=float))
    return out


def set_pose(client: int, body: int, pos=None, orn=None) -> None:
    import pybullet as p

    cur_pos, cur_orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
    p.resetBasePositionAndOrientation(
        body,
        [float(v) for v in (cur_pos if pos is None else pos)],
        [float(v) for v in (cur_orn if orn is None else orn)],
        physicsClientId=client)


def apply_oracle(client: int, task, ops) -> list:
    """按 ORACLE 的 op 串改物体 base pose（不推进物理），返回执行过的 op 文本。"""
    import pybullet as p

    done = []
    for op in ops:
        kind, name = op[0], op[1]
        if name not in task.objects:
            raise KeyError(name)
        body = task.objects[name]
        pos, _orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
        pos = np.asarray(pos, dtype=float)
        if kind == "copy_pose":
            ref_pos, ref_orn = p.getBasePositionAndOrientation(
                task.objects[op[2]], physicsClientId=client)
            set_pose(client, body, ref_pos, ref_orn)
        elif kind == "align":
            ref_pos, _ = p.getBasePositionAndOrientation(
                task.objects[op[2]], physicsClientId=client)
            set_pose(client, body, (ref_pos[0], ref_pos[1], pos[2]))
        elif kind == "above":
            ref_pos, _ = p.getBasePositionAndOrientation(
                task.objects[op[2]], physicsClientId=client)
            set_pose(client, body,
                     (ref_pos[0], ref_pos[1], ref_pos[2] + float(op[3])))
        elif kind == "set_pos":
            set_pose(client, body, op[2])
        elif kind == "set_z":
            set_pose(client, body, (pos[0], pos[1], float(op[2])))
        elif kind == "raise":
            set_pose(client, body, (pos[0], pos[1], pos[2] + float(op[2])))
        elif kind == "move_by":
            set_pose(client, body, pos + np.asarray(op[2], dtype=float))
        elif kind == "rotate":
            set_pose(client, body, orn=p.getQuaternionFromEuler(
                [float(v) for v in op[2]]))
        elif kind == "upright":
            set_pose(client, body, (pos[0], pos[1], float(op[2])),
                     (0.0, 0.0, 0.0, 1.0))
        else:
            raise ValueError(f"unknown oracle op: {kind}")
        done.append(" ".join(str(part) for part in op))
    return done


def movable_objects(client: int, task) -> dict:
    """{物体名: 质量}，排除 table；质量 0 = 固定家具（不是可搬物体）。"""
    import pybullet as p

    out = {}
    for key, body in task.objects.items():
        if key == TABLE_KEY:
            continue
        out[key] = float(p.getDynamicsInfo(body, -1, physicsClientId=client)[0])
    return out


def link_points(client: int, body: int) -> dict:
    """{连杆名: 世界坐标原点}（当前姿势下的 FK）。"""
    import pybullet as p

    out = {}
    for index in range(p.getNumJoints(body, physicsClientId=client)):
        info = p.getJointInfo(body, index, physicsClientId=client)
        point = p.getLinkState(body, index, computeForwardKinematics=1,
                               physicsClientId=client)[0]
        out[info[1].decode("utf-8", "replace")] = np.asarray(point, dtype=float)
    return out


def pick_link(points: dict, *needles: str):
    """名字里同时含所有 needles 的第一个连杆；找不到返回 (None, None)。"""
    for name, point in points.items():
        low = name.lower()
        if all(needle in low for needle in needles):
            return name, point
    return None, None


def measure_reach(client: int, robot: str) -> dict:
    """一台机器人的肩/肘/腕连杆原点 + 上臂 / 前臂 / 可达半径（不含手掌）。"""
    from teleoperation.simulation import RobotLoader

    fresh(client, ground=False)
    body = RobotLoader(client).load_robot(robot)["robot"]
    points = link_points(client, body)

    shoulder_name, shoulder = pick_link(points, "shoulder", "pitch")
    if shoulder is None:
        shoulder_name, shoulder = pick_link(points, "shoulder")
    elbow_name, elbow = pick_link(points, "elbow")
    wrist_name, wrist = pick_link(points, "wrist", "yaw")
    if wrist is None:
        wrist_name, wrist = pick_link(points, "wrist")
    out = {"robot": robot,
           "links": {"shoulder": shoulder_name, "elbow": elbow_name,
                     "wrist": wrist_name}}
    if shoulder is None or elbow is None or wrist is None:
        out.update({"upper_m": None, "fore_m": None, "reach_m": None,
                    "shoulder_m": None})
        return out
    upper = float(np.linalg.norm(elbow - shoulder))
    fore = float(np.linalg.norm(wrist - elbow))
    out.update({"upper_m": round(upper, 4), "fore_m": round(fore, 4),
                "reach_m": round(upper + fore, 4),
                "shoulder_m": [round(float(v), 4) for v in shoulder]})
    return out


def best_scene_shift(shoulder, cloud, reach, dx_range=(-1.20, 0.11, 0.02),
                     dz_range=(-0.20, 1.21, 0.02)) -> dict:
    """(dx, dz) 网格搜索：整体平移场景后，最远物体到肩的距离能压到多小。

    平移作用在**场景物体**上（等价于把机器人往桌前挪 / 把台面抬高），只回答
    「几何上还有没有救」，不代表任何任务判据会被破坏（判据只看物体之间的
    相对关系，统一平移不影响）。

    同时给出平移后**进圈物体数**：单个最远物体可能是被物理甩飞的异常值，
    百分比比「最远值」更能说明问题。
    """
    if shoulder is None or not cloud:
        return {"farthest_m": None, "inside": None, "dx_m": None, "dz_m": None,
                "at_boundary": None}
    points = np.asarray(cloud, dtype=float)
    shoulder = np.asarray(shoulder, dtype=float)
    dxs = np.arange(*dx_range)
    dzs = np.arange(*dz_range)
    best = None
    for dx in dxs:
        shifted = points.copy()
        shifted[:, 0] += dx
        for dz in dzs:
            moved = shifted.copy()
            moved[:, 2] += dz
            dists = np.linalg.norm(moved - shoulder, axis=1)
            far = float(dists.max())
            if best is None or far < best[0]:
                best = (far, float(dx), float(dz),
                        int((dists <= float(reach)).sum()))
    far, dx, dz, inside = best
    eps = 1e-9
    return {"farthest_m": round(far, 4), "inside": inside,
            "dx_m": round(dx, 3), "dz_m": round(dz, 3),
            "at_boundary": bool(abs(dx - dxs[0]) < eps or abs(dx - dxs[-1]) < eps
                                or abs(dz - dzs[0]) < eps or abs(dz - dzs[-1]) < eps)}


def check_task(client: int, name: str, steps: int) -> dict:
    """一个任务的四项检查：load / stable / false_positive / triggerable。"""
    import pybullet as p
    from teleoperation.simulation.tasks import get_task

    row = {"task": name, "load_ok": False, "stable": None,
           "max_drift_m": None, "below_table": [], "succ_at_reset": None,
           "succ_after_idle": None, "succ_oracle": None, "oracle_ops": [],
           "oracle_error": None, "drift": {}, "flags": [],
           "objects": {}, "verdict": "LOAD_ERROR", "error": None}
    try:
        fresh(client)
        task = get_task(name)(client)
        task.reset()
        row["load_ok"] = True
        start = poses(client, task.objects)
        row["succ_at_reset"] = bool(task.check_success())
        for _ in range(int(steps)):
            p.stepSimulation(physicsClientId=client)
        end = poses(client, task.objects)
        row["succ_after_idle"] = bool(task.check_success())
        table_z = float(end[TABLE_KEY][0][2]) if TABLE_KEY in end else 0.4
        drifts = {key: float(np.linalg.norm(end[key][0] - start[key][0]))
                  for key in end}
        row["max_drift_m"] = round(max(drifts.values()), 4) if drifts else 0.0
        row["drift"] = {key: round(value, 3) for key, value in
                        sorted(drifts.items(), key=lambda pair: -pair[1])
                        if value > 0.01}
        row["below_table"] = sorted(
            key for key, value in end.items()
            if key != TABLE_KEY and value[0][2] < table_z - BELOW_TABLE_M)
        finite = all(np.all(np.isfinite(value[0])) and np.all(np.isfinite(value[1]))
                     for value in end.values())
        row["stable"] = bool(row["max_drift_m"] <= DRIFT_M
                             and not row["below_table"] and finite)
        masses = movable_objects(client, task)
        row["objects"] = {key: [round(float(v), 4) for v in end[key][0]]
                          for key in end if masses.get(key, 0.0) > 0.0}
        try:
            row["oracle_ops"] = apply_oracle(client, task, ORACLE[name])
            row["succ_oracle"] = bool(task.check_success())
        except (KeyError, ValueError, IndexError) as error:
            row["oracle_error"] = f"{type(error).__name__}: {error}"
        # 三项检查互相独立，flags 全记下来；verdict 只挑最严重的一个当标题：
        # 判据假成功 > 判据不可触发 > 场景不稳（顺序 = 对"成功率可信度"的伤害）。
        if not row["stable"]:
            row["flags"].append("UNSTABLE")
        if row["succ_at_reset"] or row["succ_after_idle"]:
            row["flags"].append("FALSE_POSITIVE")
        if row["succ_oracle"] is not True:
            row["flags"].append("NOT_TRIGGERABLE")
        for verdict in ("FALSE_POSITIVE", "NOT_TRIGGERABLE", "UNSTABLE"):
            if verdict in row["flags"]:
                row["verdict"] = verdict
                break
        else:
            row["verdict"] = "OK"
        task._clear_objects()
    except Exception as error:  # noqa: BLE001 - 一个任务炸掉不能拖垮整张表
        row["error"] = f"{type(error).__name__}: {error}"
        row["flags"] = ["LOAD_ERROR"]
        row["verdict"] = "LOAD_ERROR"
    return row


def reach_report(client: int, rows, robots) -> dict:
    """每台机器人的可达半径 vs 全部可搬物体；再网格搜索统一平移还有没有救。"""
    cloud = [position for row in rows for position in row["objects"].values()]
    labels = [(row["task"], key) for row in rows for key in row["objects"]]
    out = {}
    for robot in robots:
        item = measure_reach(client, robot)
        if item.get("reach_m") and cloud:
            shoulder = np.asarray(item["shoulder_m"], dtype=float)
            dists = np.linalg.norm(np.asarray(cloud, dtype=float) - shoulder,
                                   axis=1)
            nearest = int(np.argmin(dists))
            item["objects"] = {
                "total": int(dists.size),
                "inside": int((dists <= item["reach_m"]).sum()),
                "nearest_m": round(float(dists.min()), 4),
                "farthest_m": round(float(dists.max()), 4),
                "nearest": "/".join(labels[nearest]),
            }
            item["shift"] = best_scene_shift(shoulder, cloud, item["reach_m"])
            item["shift_covered"] = bool(
                item["shift"]["farthest_m"] is not None
                and item["shift"]["farthest_m"] <= item["reach_m"])
        out[robot] = item
    return out


def print_task_table(rows) -> None:
    print(f"{'task':<16} {'load':<5} {'stable':<6} {'no_fp':<6} {'trig':<5} verdict")
    print("-" * 58)
    for row in rows:
        load = "OK" if row["load_ok"] else "FAIL"
        stable = "-" if row["stable"] is None else (
            "OK" if row["stable"] else "FAIL")
        no_fp = "-" if row["succ_at_reset"] is None else (
            "OK" if not (row["succ_at_reset"] or row["succ_after_idle"])
            else "FAIL")
        trig = "-" if row["succ_oracle"] is None else (
            "OK" if row["succ_oracle"] else "FAIL")
        print(f"{row['task']:<16} {load:<5} {stable:<6} {no_fp:<6} {trig:<5} "
              f"{row['verdict']}")
    print("-" * 58)


def print_reach(reach) -> None:
    for robot, item in reach.items():
        if not item.get("reach_m"):
            print(f"{robot:<8} reach unknown (links: {item.get('links')})")
            continue
        print(f"{robot:<8} shoulder={item['shoulder_m']} upper={item['upper_m']} "
              f"fore={item['fore_m']} -> reach={item['reach_m']} m")
        stats = item.get("objects")
        if stats:
            print(f"         inside reach {stats['inside']}/{stats['total']}, "
                  f"nearest {stats['nearest_m']} m ({stats['nearest']}), "
                  f"farthest {stats['farthest_m']} m")
        shift = item.get("shift")
        if shift and shift.get("farthest_m") is not None:
            print(f"         best uniform shift dx={shift['dx_m']} dz={shift['dz_m']} "
                  f"-> farthest {shift['farthest_m']} m, inside {shift['inside']}"
                  f"/{stats['total'] if stats else '?'} "
                  f"(at_boundary={shift['at_boundary']})")


def print_diagnostics(rows) -> None:
    """非 OK 的任务逐个说明：哪些检查挂了、哪个物体动了多少、oracle 报了什么。"""
    bad = [row for row in rows if row["verdict"] != "OK"]
    if not bad:
        return
    print()
    print("diagnostics (only the tasks that are not OK):")
    for row in bad:
        print(f"- {row['task']:<16} flags={','.join(row['flags']) or '-'} "
              f"verdict={row['verdict']}")
        if row["error"]:
            print(f"    error: {row['error']}")
        if row["drift"]:
            moved = ", ".join(f"{key} {value} m" for key, value
                              in list(row["drift"].items())[:3])
            print(f"    drift after idle: {moved}")
        if row["below_table"]:
            print(f"    below table: {row['below_table']}")
        if row["succ_at_reset"] or row["succ_after_idle"]:
            print(f"    success already true at reset: {row['succ_at_reset']}, "
                  f"after idle: {row['succ_after_idle']}")
        if row["oracle_error"]:
            print(f"    oracle error: {row['oracle_error']}")
        elif row["succ_oracle"] is not True:
            print(f"    oracle ops did not satisfy the criterion: "
                  f"{row['oracle_ops']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/audit_tasks.py",
        description="30 个任务的可评测性审计：稳定性 / 假成功 / 判据可触发 / 手臂可达")
    parser.add_argument("--task", nargs="+", default=None,
                        help="只审这些任务（空格分隔；默认 30 个全审）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"空转步数，240 步 = 1 s（默认 {DEFAULT_STEPS}）")
    parser.add_argument("--robot", action="append", choices=ROBOTS, default=None,
                        help="只量这些机器人的可达性（可重复；默认三台）")
    parser.add_argument("--no-reach", action="store_true",
                        help="跳过可达性那一段（只查判据，快 3 倍）")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON,
                        help=f"报告路径（默认 {DEFAULT_JSON}）")
    parser.add_argument("--verbose", action="store_true",
                        help="每个任务跑完就打一行（默认只打最后的表）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import pybullet as p

    from teleoperation.simulation.tasks import list_tasks, register_builtin_tasks

    register_builtin_tasks()
    names = list_tasks()
    if args.task:
        unknown = [name for name in args.task if name not in names]
        if unknown:
            log("FAIL", f"unknown task(s): {unknown}")
            return 2
        names = list(args.task)
    missing = [name for name in names if name not in ORACLE]
    if missing:
        log("FAIL", f"no oracle entry for {missing} -- task criteria changed? "
                    f"update ORACLE in this file")
        return 2

    robots = tuple(args.robot or ROBOTS)
    client = p.connect(p.DIRECT)
    if client < 0:
        log("FAIL", "could not create a DIRECT physics client")
        return 2
    try:
        rows = []
        for name in names:
            row = check_task(client, name, args.steps)
            rows.append(row)
            if args.verbose:
                log("INFO", f"{name}: {row['verdict']}")
        reach = {} if args.no_reach else reach_report(client, rows, robots)
    finally:
        p.disconnect(client)

    print()
    print_task_table(rows)
    if reach:
        print()
        print_reach(reach)
    print_diagnostics(rows)

    summary = {
        "tasks": len(rows),
        "ok": sum(1 for row in rows if row["verdict"] == "OK"),
        "false_positive": [r["task"] for r in rows
                           if "FALSE_POSITIVE" in r["flags"]],
        "not_triggerable": [r["task"] for r in rows
                            if "NOT_TRIGGERABLE" in r["flags"]],
        "unstable": [r["task"] for r in rows if "UNSTABLE" in r["flags"]],
        "load_error": [r["task"] for r in rows if r["verdict"] == "LOAD_ERROR"],
    }
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "steps": int(args.steps), "robots": list(robots),
               "oracle_ops": {row["task"]: row["oracle_ops"] for row in rows},
               "tasks": rows, "reach": reach, "summary": summary}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nreport = {args.json}")

    print()
    log("INFO", f"{summary['ok']}/{summary['tasks']} tasks pass all four checks")
    for key in ("false_positive", "not_triggerable", "unstable"):
        if summary[key]:
            log("FAIL", f"{key}: {summary[key]}")
    if summary["load_error"]:
        log("ERROR", f"load_error: {summary['load_error']}")
    for robot, item in reach.items():
        stats = item.get("objects")
        if item.get("reach_m") and stats:
            log("INFO", f"{robot}: reach={item['reach_m']} m, objects inside "
                        f"reach {stats['inside']}/{stats['total']}, nearest "
                        f"{stats['nearest_m']} m ({stats['nearest']})")

    if summary["ok"] == summary["tasks"]:
        log("OK", "all tasks are evaluable: stable, no false positive, "
                  "criteria triggerable")
        return 0
    log("WARN", "some tasks are not evaluable yet -- see verdict column "
                "(FALSE_POSITIVE / NOT_TRIGGERABLE / UNSTABLE)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
