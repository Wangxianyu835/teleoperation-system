"""任务级评估（批次 2）：把 30 个任务的场景摆成「手臂够得着、物体不乱飞」，再逐 trial 记成功与耗时。

为什么需要这个文件（批次 1 的结论）
------------------------------------
``devtools/audit_tasks.py`` 已经证明：baseline 的场景**不能直接测**。除了判据本身的问题，
还有一个几何根因 —— 30 个任务全都把物体**中心**摆在台面高度 z=0.42：

* ``load_cylinder(..., height=h)`` 的 ``h`` 在 PyBullet 里是**全高**，物体中心在台面上
  -> 底部陷进台面 h/2（如 ``pen`` 陷 0.06 m、``bottle`` 0.06 m、``drawer`` 0.14 m、``cabinet`` 0.37 m）；
* 物理一放手就把它们弹出来，2.5 s 内自漂 0.10-1.20 m（实测 ``drawer`` 被弹飞 1.20 m）；
* 且台面在 z=0.42，而 H1-2 肩在 z=1.43、臂长只有 0.58 m -> 57 个可搬物体 **0 个够得着**。

所以本文件做三件事（都不改 ``src/teleoperation/**``）：

1. **场景层**：把整幕（含桌子）平移到手臂够得着的地方（按机器人 FK 算，见 ``solve_shift``），
   给台面加一根落地支撑（看起来像高台，不再是悬空板），把每个物体**抬到台面上**，
   再按 AABB 把互相嵌入的物体分开（"上面那个放到下面那个顶上"，其余沿最小嵌入轴推开），
   最后空转 ``SETTLE_STEPS`` 步验一次稳定性。逐条写进 report。
2. **trial runner**：``env.reset()`` -> 场景层 -> 逐帧推进（物理步 = 1/240 s，**自己算仿真时间**，
   不用 baseline 的 ``time.time()`` 超时）-> 每帧 ``task.check_success()``，
   成功即记 completion time；每个 trial 落一条 JSON。
3. **``--verify``**：把 30 个任务都过一遍场景层，打印 稳定/无假成功/判据可触发/在可达圈内 四列，
   这是批次 2 的验收证据（批次 1 的同一把尺子，只是先摆场景再量）。

已知不可修的判据（不计入成功率）
--------------------------------------
``liftmug`` / ``opendrawer`` / ``openmicrowave`` 的阈值比自己物体的半宽还小，**贴着就已经成立**，
所以 reset 时判据为真；``drawerbook`` 用 ``SCENE_OPS`` 把抽屉换成薄面板后可以触发。

实测（2026-11-09，``--verify --robot all``）：**72/90 格可评测**（3 台机器人 x 30 任务，
每台 24 个 OK），30 个任务全部稳定（最大漂移 0.094 m）、10 个代表任务全部 OK；
其余 18 格 = 3 个 ``criteria_invalid`` + 2 个 ``criteria_absolute`` + 1 个 ``scene_unsupported``
（每个机器人各 6 个，见 ``docs/task-evaluation.md`` §4.3 / §4.5）。

用法::

    # 批次 2 验收：30 个任务过场景层，看四列
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --verify

    # 空转 3 个 trial（验证判据不会自己变真），落 JSON
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --task pushcube --trials 3 --policy idle

    # 用假动作驱动（验证动作能真的送进机器人、判据不误触）
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --task pushcube --trials 1 --policy dummy

    # 回放现成的原生动作文件（契约 H）
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --policy replay --actions datasets\\actions\\x.h5

退出码：0 = 全部 trial 按预期（有成功或空转时 0 成功都可以，看 ``--expect``）；1 = 有一项检查失败
（假成功 / 场景不 statble / 判据被场景层破坏）；2 = 工具自身异常（任务名写错、连不上 PyBullet）。

范围声明
--------
* 只读 baseline：场景改动全部发生在**运行时**的 base pose / 新建的支撑体上，不写任何任务代码；
* 判据没有改：``CRITERIA_INVALID`` 只是**标记**，工具不会去"修"判据；
* 场景层会改变任务的初始摆法（这是它存在的理由），改动逐条写在 report 的 ``scene`` 段里，
  对外报数时必须一并给出（见 ``docs/task-evaluation.md`` §4 的偏差表）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
DEVTOOLS = Path(__file__).resolve().parent
for extra in (ROOT / "src", DEVTOOLS):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import audit_tasks as audit  # noqa: E402  量具复用：fresh / poses / set_pose / link_points / ORACLE

ROBOTS = audit.ROBOTS
TABLE_KEY = audit.TABLE_KEY
DT = 1.0 / 240.0
#: 落地支撑（bench）在 x 上的起点（米）：桌子可能伸到机器人胸前，支撑要留在腿前
BENCH_MIN_X = 0.18
#: 空转多少步算"稳"（600 步 = 2.5 s，与 audit 默认一致）
SETTLE_STEPS = 600
#: 一次 trial 最多跑多少仿真秒（论文没人给它无限时间）
DEFAULT_MAX_SIM_S = 60.0
#: 论文 Table 1 里被拿去做量化对比的 10 个代表任务
DEFAULT_TASKS = ("pushcube", "pickcube", "pickplacecube", "uprearcup", "balltrashcan",
                 "ballmug", "ballbimanual", "potbimanual", "pottomatoplate",
                 "penbrushpot")
#: 判据在 reset 时就成立（阈值 <= 物体半宽），成功率只能是 100%，不能计入
CRITERIA_INVALID = {
    "liftmug": "reset 时 |lid z - mug z| = 0.055 > 0.05；杯子半高 0.05，盖子在杯顶就已经超阈值",
    "opendrawer": "reset 时 |drawer x - cabinet x| = 0.14 > 0.12；柜体 x 半宽 0.15，抽屉贴着柜面即算拉开",
    "openmicrowave": "reset 时门心离本体心 0.15 > 0.12；本体 x 半宽 0.15，门贴着本体即算拉开",
}
#: 场景层里对**单个任务**的额外摆法（op 语义见 apply_scene_ops）。
#: 只在"通用阶梯救不了"的地方写，写一条就要在 docs/task-evaluation.md 里留一句为什么。
SCENE_OPS = {
    # 抽屉的 box 半宽 0.14 比柜体半宽 0.15 还大，物理上必然和柜体互相嵌 -> 弹飞 1.2 m。
    # 改成"贴在柜面上的薄抽屉面板"：关上时 |dx| = 0.155 < 0.17（closedrawer / drawerbook 的阈值），
    # 拉开 3 cm 以上就 > 0.17，判据仍然有意义；opendrawer 的阈值 0.12 小于柜体半宽，单独标 invalid。
    "closedrawer": [("resize", "drawer", (0.015, 0.18, 0.1)),
                    ("place", "drawer", (0.62, 0.3, None))],
    "drawerbook": [("resize", "drawer", (0.015, 0.18, 0.1)),
                   ("place", "drawer", (0.555, 0.3, None))],
    # opendrawer 的判据 |dx| > 0.12 比柜体 x 半宽 0.15 还小 -> 修不了，标 CRITERIA_INVALID；
    # 但场景仍然换成薄抽屉面板，免得抽屉被弹飞 1.2 m（测出来至少是"判据无效"，不是"物理爆炸"）。
    "opendrawer": [("resize", "drawer", (0.015, 0.18, 0.1)),
                   ("place", "drawer", (0.555, 0.3, None))],
}
#: 判据里写死了**世界坐标 z**、且**改场景也救不回来**的任务。场景层抬台面后这些判据要么恒真
#: （pickcube / potbimanual 一开始就"成功" ---- 这两个已用 ``SCENE_MODE="sink"`` 救回），
#: 要么永远不成立（breadtoaster / rotatehearth 要求按到 0.43 以下 = 按进台面里）。
#: **不是场景层能修的** —— 判据得改成"相对初始位姿"，这是交给任务作者的问题清单
#: （docs/task-evaluation.md §4.5）。
CRITERIA_ABSOLUTE_Z = {
    "breadtoaster": "`button_pos[2] < 0.47`：按钮初始 z=0.48、半高 0.01 = 正好躺在烤面包机上 -> 永不成立",
    "rotatehearth": "`pos[2] < 0.43`：旋钮躺在台面上 z=0.44，要求按到 0.43 = 按进台面里 -> 永不成立",
}
#: 判据本身要求"物体在容器里"，而容器是**实心** primitive —— 场景层摆不出来。
SCENE_UNSUPPORTED = {
    "pottomatoout": "判据要求番茄离开锅（xy>0.06）；初始必须在锅里（z=0.44 < 锅顶 0.48），"
                    "实心 cylinder 装不住球，只能摆锅边 -> reset 即判成功。需要空心锅或相对位移判据。",
}
#: 判据里写死绝对 z、但任务本身还能救的：改用"把机器人降到作业高度"（桌面保持作者写的
#: 0.42 m），这样判据的原意不变。``rotatehearth`` / ``breadtoaster`` 不在这里 ——
#: 它们要求把旋钮/按钮按到桌面以下（按进台面里），降机器人也救不了。
SCENE_MODE = {"pickcube": "sink", "potbimanual": "sink"}
#: 这些 verdict 是"已定性、已交回任务作者"的排除项，不算"不可评测"（退出码不受影响）
EXCLUDED_VERDICTS = ("CRITERIA_INVALID", "CRITERIA_ABSOLUTE", "SCENE_UNSUPPORTED")
# ---------------------------------------------------------------------------
# 基础操作：位姿 / AABB / 质量（全部走 base pose，不推进物理）
# ---------------------------------------------------------------------------
def aabb(client: int, body: int):
    lo, hi = p.getAABB(body, physicsClientId=client)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def body_pose(client: int, body: int):
    pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=client)
    return np.asarray(pos, dtype=float), np.asarray(orn, dtype=float)


def move_body(client: int, body: int, delta) -> np.ndarray:
    """按 delta 平移 base pose（不推进物理），返回新位置。"""
    pos, orn = body_pose(client, body)
    new = pos + np.asarray(delta, dtype=float)
    p.resetBasePositionAndOrientation(
        body, [float(v) for v in new], [float(v) for v in orn],
        physicsClientId=client)
    return new


def body_mass(client: int, body: int) -> float:
    return float(p.getDynamicsInfo(body, -1, physicsClientId=client)[0])


def solve_shift(shoulder, points, reach, table_min_x=None, forward=0.66,
                target=0.80, clearance=0.24, margin=0.02, step=0.02) -> dict:
    """纯 numpy：算出"把物体挪到手臂够得着的地方"要平移多少。

    约束（都是实测踩出来的）
    ------------------------
    * 物体整体横移到肩前 ``forward * reach``（h1_2: 0.38 m），**不横移 y**（判据里有 |y| 绝对量）；
    * 抬到离肩 ``target * reach`` 的球面上（h1_2: 0.46 m），只抬不降 ——
      ``pickcube`` / ``potbimanual`` / ``uprearcup`` 的判据含 **绝对 z**（``z > 0.66`` 等），降了就永远不成立；
    * 桌子**单独**平移：近边留在肩前 ``clearance``，免得台面切进机器人本体
      （物体离桌沿还有 ~0.14 m，照样在桌面上）。

    返回 dict：``dx_obj`` / ``dz_obj`` / ``dx_table`` / ``farthest_m`` / ``inside`` / ``feasible``。
    """
    info = {"dx_obj": 0.0, "dy_obj": 0.0, "dz_obj": 0.0, "dx_table": 0.0,
            "farthest_m": None, "inside": None, "total": None,
            "reach_m": None if reach is None else round(float(reach), 4),
            "feasible": False}
    if table_min_x is not None:
        info["dx_table"] = round(float(clearance) - float(table_min_x), 4)
    if shoulder is None or reach is None or float(reach) <= 0.0 or not points:
        return info
    s = np.asarray(shoulder, dtype=float)
    pts = np.asarray(points, dtype=float)
    radius = float(reach)
    total = int(pts.shape[0])
    info["total"] = total
    base = np.linalg.norm(pts - s, axis=1)
    if float(base.max()) <= radius - margin:
        info.update({"farthest_m": round(float(base.max()), 4),
                     "inside": int((base <= radius).sum()), "feasible": True})
        return info
    centre = pts.mean(axis=0)
    limit = radius - margin
    preferred = float(s[0] + min(float(forward) * radius, 0.9 * limit) - centre[0])
    # 先按"质心放肩前 forward*reach"，不行就左右挪 dx：plateoven 的烤箱+门+盘子横向
    # 铺开 0.4 m，质心对准肩时门还在圈外（实测 0.597 > 0.533），整体再往怀里拉 0.1 m 就够。
    candidates = [preferred]
    for order in range(1, 46):
        candidates.append(preferred - order * step)
        candidates.append(preferred + order * step)
    chosen = None
    for dx in candidates:
        horizontal = np.hypot(pts[:, 0] + dx - s[0], pts[:, 1] - s[1])
        if float(horizontal.max()) >= limit:
            continue
        allowed = np.sqrt(limit * limit - horizontal * horizontal)
        low = float((s[2] - allowed - pts[:, 2]).max())
        high = float((s[2] + allowed - pts[:, 2]).min())
        if low > high:
            continue
        dz = max(0.0, float(low) + 0.25 * (float(high) - float(low)))
        chosen = (float(dx), dz)
        break
    if chosen is None:
        # 横向铺得比臂展还宽（g1 臂长 0.30 m，ballmug 的球和杯相距 0.63 m）：
        # 先把质心放到肩下 0.55*limit 的高度，横向交给 _compress_to_reach 收缩
        best_dx = min(candidates,
                      key=lambda value: float(np.hypot(
                          pts[:, 0] + value - s[0], pts[:, 1] - s[1]).max()))
        horizontal = float(np.hypot(centre[0] + best_dx - s[0], centre[1] - s[1]))
        allowance = float(np.sqrt(max(1e-6, limit * limit - (0.55 * limit) ** 2)))
        dz = max(0.0, float(s[2] - centre[2] - 0.55 * limit))
        chosen = (float(best_dx), dz)
        info["needs_compression"] = True
        info["centre_horizontal_m"] = round(horizontal, 4)
        info["horizontal_allowance_m"] = round(allowance, 4)
    dx, dz = chosen
    dists = np.linalg.norm(pts + np.array([dx, 0.0, dz]) - s, axis=1)
    info.update({"dx_obj": round(dx, 4), "dz_obj": round(dz, 4),
                 "farthest_m": round(float(dists.max()), 4),
                 "inside": int((dists <= radius).sum()),
                 "feasible": bool(float(dists.max()) <= radius - margin)})
    return info




# ---------------------------------------------------------------------------
# 场景层
# ---------------------------------------------------------------------------
class TaskScene:
    """把一个任务场景摆成"可评测"：够得着 + 物体躺在台面上 + 互不嵌入。

    只在运行时改 base pose / 新建一根落地支撑，不碰任务代码；每做一步都记进 ``self.report``，
    对外报成功率时必须把这份 report 一起给出（改了初始摆法 = 改了任务难度）。
    """

    def __init__(self, client: int, task, robot: str = "", shoulder=None,
                 reach=None, name: str = "", robot_id=None):
        self.client = client
        self.task = task
        self.robot = robot
        self.shoulder = None if shoulder is None else np.asarray(shoulder, dtype=float)
        self.reach = None if reach is None else float(reach)
        self.name = name or type(task).__name__
        self.robot_id = robot_id
        self.mode = SCENE_MODE.get(self.name, "raise")
        self.extra_bodies = []
        self.report = {"task": self.name, "robot": robot, "shift": {},
                       "bench_top_m": None, "rest_on_table": {}, "ops": [],
                       "clamped": {}, "compressed": {}, "separated": [],
                       "settle": {}, "criteria_at_reset": None, "in_reach": {},
                       "notes": []}

    # -- 查询 ---------------------------------------------------------------
    def _movable_keys(self) -> list:
        out = []
        for key, body in self.task.objects.items():
            if key == TABLE_KEY:
                continue
            if body_mass(self.client, body) > 0.0:
                out.append(key)
        return out

    def _table_top(self) -> float:
        table = self.task.objects.get(TABLE_KEY)
        if table is None:
            return 0.4
        return float(aabb(self.client, table)[1][2])

    # -- 五个步骤 -----------------------------------------------------------
    def _fit_scale(self, points, dx: float, margin: float = 0.02,
                   fill: float = 0.95) -> float:
        """横向铺得比臂展还宽时要缩多少（1.0 = 不用缩）。"""
        if self.shoulder is None or self.reach is None:
            return 1.0
        limit = (float(self.reach) - margin) * fill
        pts = np.asarray(points, dtype=float)
        horizontal = np.hypot(pts[:, 0] + dx - self.shoulder[0],
                              pts[:, 1] - self.shoulder[1])
        worst = float(horizontal.max())
        return 1.0 if worst <= limit or worst <= 0.0 else limit / worst

    def _robot_sink(self, points, dx: float, margin: float = 0.02):
        """``sink`` 模式：算机器人 base 要下降多少，才能让作者摆的物体进可达圈（z 一点不动）。"""
        if self.robot_id is None or self.shoulder is None or self.reach is None:
            return None
        limit = float(self.reach) - margin
        pts = np.asarray(points, dtype=float)
        horizontal = np.hypot(pts[:, 0] + dx - self.shoulder[0],
                              pts[:, 1] - self.shoulder[1])
        if float(horizontal.max()) >= limit:
            return None
        allowed = np.sqrt(limit * limit - horizontal * horizontal)
        return float((allowed + pts[:, 2] - self.shoulder[2]).min())

    def _shift(self) -> None:
        # 物体 + 家具一起平移（柜体/微波炉本体是 static，但它们和抽屉/门是一套，
        # 只挪 movable 会把抽屉从柜体上撕下来，判据 |drawer x - cabinet x| 直接失真）
        keys = [key for key in self.task.objects if key != TABLE_KEY]
        table = self.task.objects.get(TABLE_KEY)
        points = [body_pose(self.client, self.task.objects[key])[0] for key in keys]
        table_min_x = None
        table_max_x = None
        if table is not None:
            lo, hi = aabb(self.client, table)
            table_min_x, table_max_x = float(lo[0]), float(hi[0])
        info = solve_shift(self.shoulder, points, self.reach, table_min_x=table_min_x)
        if self.mode == "sink":
            sink = self._robot_sink(points, info["dx_obj"])
            if sink is None:
                # g1 臂长 0.30 m，锅+两个手柄横着铺 0.32 m：先按比例收一点再降
                scale = self._fit_scale(points, info["dx_obj"])
                if scale < 0.999:
                    for key in keys:
                        pos, orn = body_pose(self.client, self.task.objects[key])
                        new = [float(self.shoulder[0] + (pos[0] - self.shoulder[0]) * scale),
                               float(self.shoulder[1] + (pos[1] - self.shoulder[1]) * scale),
                               float(pos[2])]
                        p.resetBasePositionAndOrientation(
                            self.task.objects[key], new,
                            [float(v) for v in orn], physicsClientId=self.client)
                    points = [body_pose(self.client, self.task.objects[key])[0]
                              for key in keys]
                    info["scaled_xy"] = round(scale, 4)
                    sink = self._robot_sink(points, info["dx_obj"])
            if sink is not None:
                pose, orn = body_pose(self.client, self.robot_id)
                p.resetBasePositionAndOrientation(
                    self.robot_id, [float(pose[0]), float(pose[1]),
                                    float(pose[2] + sink)],
                    [float(v) for v in orn], physicsClientId=self.client)
                self.shoulder = self.shoulder + np.array([0.0, 0.0, sink])
                info.update({"dz_obj": 0.0, "robot_dz_m": round(sink, 4),
                             "feasible": True})
                dists = np.linalg.norm(np.asarray(points) + np.array(
                    [info["dx_obj"], 0.0, 0.0]) - self.shoulder, axis=1)
                info.update({"farthest_m": round(float(dists.max()), 4),
                             "inside": int((dists <= float(self.reach)).sum())})
            else:
                self.report["notes"].append("sink mode failed, falling back to raise")
        if table is not None and keys:
            # 桌子必须**盖住**平移后的物体，否则贴着远端的物体（``pottomatoplate`` 的 plate
            # 在 x=0.30，锅在 0.50）会从桌沿掉下去（实测掉 1.18 m）。
            boxes = [aabb(self.client, self.task.objects[key])[0][0] for key in keys]
            obj_min_x = min(boxes) + info["dx_obj"] - 0.04
            info["dx_table"] = round(min(info["dx_table"], obj_min_x - table_min_x), 4)
            info["table_span_m"] = [round(table_min_x + info["dx_table"], 4),
                                    round(table_max_x + info["dx_table"], 4)]
        self.report["shift"] = dict(info, objects=keys)
        offset = np.array([info["dx_obj"], 0.0, info["dz_obj"]], dtype=float)
        if float(np.abs(offset).max()) > 1e-9:
            for key in keys:
                move_body(self.client, self.task.objects[key], offset)
        if table is not None and (abs(info["dx_table"]) > 1e-9
                                  or abs(info["dz_obj"]) > 1e-9):
            move_body(self.client, table, (info["dx_table"], 0.0, info["dz_obj"]))
        if not info["feasible"] and info["total"]:
            self.report["notes"].append(
                "reach: %s/%s inside, farthest %s m vs reach %s m"
                % (info["inside"], info["total"], info["farthest_m"], info["reach_m"]))

    def _apply_ops(self) -> None:
        for op in SCENE_OPS.get(self.name, []):
            kind, key = op[0], op[1]
            if key not in self.task.objects:
                self.report["notes"].append("scene op skipped, no object %r" % key)
                continue
            if kind == "resize":
                self._resize(key, op[2])
            # resize 会换 body id，所以每个分支都要重新取 body / pose（踩过：拿着旧 id 写回去）
            body = self.task.objects[key]
            pos, orn = body_pose(self.client, body)
            if kind == "place":
                target = op[2]
                new = [pos[0] if target[0] is None else float(target[0]),
                       pos[1] if target[1] is None else float(target[1]),
                       pos[2] if target[2] is None else float(target[2])]
                p.resetBasePositionAndOrientation(
                    body, new, [float(v) for v in orn], physicsClientId=self.client)
            elif kind == "above":
                ref, _ = body_pose(self.client, self.task.objects[op[2]])
                p.resetBasePositionAndOrientation(
                    body, [float(ref[0]), float(ref[1]), float(ref[2] + op[3])],
                    [float(v) for v in orn], physicsClientId=self.client)
            elif kind == "align":
                ref, _ = body_pose(self.client, self.task.objects[op[2]])
                p.resetBasePositionAndOrientation(
                    body, [float(ref[0]), float(ref[1]), float(pos[2])],
                    [float(v) for v in orn], physicsClientId=self.client)
            elif kind != "resize":
                raise ValueError("unknown scene op: %r" % (op,))
            self.report["ops"].append(" ".join(str(part) for part in op))

    def _resize(self, key: str, half_extents) -> None:
        """把实心 box 换成另一个尺寸（抽屉 / 门这类"面板"用），质量与摩擦沿用原体。"""
        body = self.task.objects[key]
        mass = body_mass(self.client, body)
        friction = float(p.getDynamicsInfo(body, -1, physicsClientId=self.client)[1])
        pos, _orn = body_pose(self.client, body)
        p.removeBody(body, physicsClientId=self.client)
        self.task.load_box(key, [float(v) for v in half_extents],
                           [float(v) for v in pos], color=[0.7, 0.4, 0.2, 1.0],
                           mass=mass, friction=friction)


    def _compress_to_reach(self, margin: float = 0.02) -> None:
        """整组按比例**收缩**到水平可及圈内（最后手段）。

        g1 臂长 0.30 m，而 ``ballmug`` 的球和杯相距 0.63 m —— 怎么平移都放不进去。
        以肩为圆心把 x/y 等比缩，物体之间的**相对**布局（哪个在哪个上面/旁边）不变，
        只把间距变小；收缩比例写进 report（``compressed``），对外必须一起报。
        """
        if self.shoulder is None or self.reach is None:
            return
        limit = float(self.reach) - margin
        compressed = {}
        for group in self._touch_groups():
            factor = 1.0
            for key in group:
                pos, _orn = body_pose(self.client, self.task.objects[key])
                vertical = float(pos[2] - self.shoulder[2])
                allowed = float(np.sqrt(max(1e-6, limit * limit - vertical * vertical)))
                horizontal = float(np.hypot(pos[0] - self.shoulder[0],
                                            pos[1] - self.shoulder[1]))
                if horizontal > allowed:
                    factor = min(factor, allowed / horizontal)
            if factor >= 0.999:
                continue
            for key in group:
                pos, orn = body_pose(self.client, self.task.objects[key])
                new = [float(self.shoulder[0] + (pos[0] - self.shoulder[0]) * factor),
                       float(self.shoulder[1] + (pos[1] - self.shoulder[1]) * factor),
                       float(pos[2])]
                p.resetBasePositionAndOrientation(
                    self.task.objects[key], new,
                    [float(v) for v in orn], physicsClientId=self.client)
            compressed["+".join(group)] = round(factor, 4)
        self.report["compressed"].update(compressed)

    def _bench(self) -> None:
        """台面以下补一根落地支撑 —— 抬到 1.2 m 的板子悬空看着不像"桌子"。

        支撑在 x 上从 ``BENCH_MIN_X`` 起：桌子可能为了让开物体而伸到机器人胸前，
        支撑要留在腿前面，不然会穿模（视觉问题，不是物理问题 —— 两个都是 static）。
        """
        table = self.task.objects.get(TABLE_KEY)
        if table is None:
            return
        lo, hi = aabb(self.client, table)
        top = float(hi[2])
        left = max(float(lo[0]), BENCH_MIN_X)
        if float(hi[0]) - left < 0.05:
            left = float(lo[0])
        half = [max(0.05, (float(hi[0]) - left) / 2.0),
                max(0.05, float(hi[1] - lo[1]) / 2.0),
                max(0.02, top / 2.0)]
        centre = [(left + float(hi[0])) / 2.0, float((hi[1] + lo[1]) / 2.0), top / 2.0]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.35, 0.35, 0.38, 1.0],
                                  physicsClientId=self.client)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=centre,
                                 physicsClientId=self.client)
        self.extra_bodies.append(body)
        self.report["bench_top_m"] = round(top, 4)

    def _rest_on_table(self) -> None:
        """把陷进台面的物体抬上来（baseline 的 cylinder ``height`` 是全高，中心却摆在台面高度）。

        静态物体（``pushcube`` 的 target 地标、柜体、微波炉本体）也要抬：它们同样是"中心摆在
        台面高度"，不抬就埋在台面里，``below_table`` 会把它判成不稳定。
        """
        top = self._table_top()
        lifted = {}
        for key, body in self.task.objects.items():
            if key == TABLE_KEY:
                continue
            lo, _hi = aabb(self.client, body)
            gap = top - float(lo[2])
            if gap > 1e-4:
                move_body(self.client, body, (0.0, 0.0, gap + 1e-4))
                lifted[key] = round(float(gap), 4)
        self.report["rest_on_table"].update(lifted)

    def _conflicts(self) -> list:
        items = [(key, body) for key, body in self.task.objects.items()
                 if key != TABLE_KEY]
        boxes = {key: aabb(self.client, body) for key, body in items}
        out = []
        for i, (key_a, _body_a) in enumerate(items):
            for key_b, _body_b in items[i + 1:]:
                lo = np.maximum(boxes[key_a][0], boxes[key_b][0])
                hi = np.minimum(boxes[key_a][1], boxes[key_b][1])
                pen = hi - lo
                if float(pen.min()) > 1e-4:
                    out.append((key_a, key_b, pen))
        return out

    def _touch_groups(self, touch: float = 0.08) -> list:
        """按 AABB 间隙把物体分组（<= touch 视为一套家具，例如柜体+抽屉、烤箱+门）。"""
        keys = [key for key in self.task.objects if key != TABLE_KEY]
        boxes = {key: aabb(self.client, self.task.objects[key]) for key in keys}
        parent = {key: key for key in keys}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for index, key_a in enumerate(keys):
            lo_a, hi_a = boxes[key_a]
            for key_b in keys[index + 1:]:
                lo_b, hi_b = boxes[key_b]
                gap = np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))
                if float(np.linalg.norm(gap)) <= touch:
                    parent[find(key_a)] = find(key_b)
        groups = {}
        for key in keys:
            groups.setdefault(find(key), []).append(key)
        return list(groups.values())

    def _clamp_to_reach(self, margin: float = 0.02) -> None:
        """把还落在可达球外的**整套**物体沿水平方向收拢到球面内。

        按 AABB 连通的组一起平移：``drawerbook`` 的抽屉被单独拉 0.10 m 会离开柜体
        （|dx| 从 0.155 变 0.255），``drawer_closed`` 判据立刻失效。只动 x/y、不动 z：
        不动 z 物体才不离台面浮起，也不破坏判据里的绝对高度。
        """
        if self.shoulder is None or self.reach is None:
            return
        limit = float(self.reach) - margin
        clamped = {}
        for group in self._touch_groups():
            need = 0.0
            positions = {}
            for key in group:
                pos, _orn = body_pose(self.client, self.task.objects[key])
                positions[key] = pos
                vertical = float(pos[2] - self.shoulder[2])
                allowed = float(np.sqrt(max(1e-6, limit * limit - vertical * vertical)))
                horizontal = float(np.hypot(pos[0] - self.shoulder[0],
                                            pos[1] - self.shoulder[1]))
                need = max(need, horizontal - allowed)
            if need <= 1e-6:
                continue
            centre = np.mean(np.stack(list(positions.values())), axis=0)
            direction = np.array([self.shoulder[0] - centre[0],
                                  self.shoulder[1] - centre[1], 0.0])
            length = float(np.linalg.norm(direction))
            if length < 1e-9:
                continue
            delta = direction / length * need
            for key in group:
                move_body(self.client, self.task.objects[key], delta)
            clamped["+".join(group)] = round(need, 4)
        self.report["clamped"].update(clamped)

    def _separate(self, rounds: int = 8, gap: float = 0.003) -> None:
        """把互相嵌入的物体分开：浅嵌入 / 上面那个 -> 叠到顶上；其余沿最小嵌入轴推开。"""
        for _ in range(rounds):
            conflicts = self._conflicts()
            if not conflicts:
                break
            for key_a, key_b, pen in conflicts:
                mass_a = body_mass(self.client, self.task.objects[key_a])
                mass_b = body_mass(self.client, self.task.objects[key_b])
                if mass_a <= 0.0 and mass_b <= 0.0:
                    continue
                if mass_a <= 0.0 or (mass_b > 0.0 and mass_b < mass_a):
                    lighter, heavier = key_b, key_a
                else:
                    lighter, heavier = key_a, key_b
                lo_l, hi_l = aabb(self.client, self.task.objects[lighter])
                lo_h, hi_h = aabb(self.client, self.task.objects[heavier])
                centre_l = (lo_l + hi_l) / 2.0
                centre_h = (lo_h + hi_h) / 2.0
                height_l = float(hi_l[2] - lo_l[2])
                height_h = float(hi_h[2] - lo_h[2])
                stacked = (float(lo_l[2]) >= centre_h[2] - 0.01
                           or float(pen[2]) <= 0.5 * min(height_l, height_h))
                if stacked:
                    dz = float(hi_h[2] - lo_l[2]) + gap
                    if dz > 0.0:
                        move_body(self.client, self.task.objects[lighter], (0.0, 0.0, dz))
                        self.report["separated"].append(
                            {"move": lighter, "onto": heavier, "axis": "z+",
                             "amount_m": round(dz, 4)})
                else:
                    axis = 0 if float(pen[0]) <= float(pen[1]) else 1
                    sign = 1.0 if centre_l[axis] >= centre_h[axis] else -1.0
                    amount = float(pen[axis]) + gap
                    delta = np.zeros(3)
                    delta[axis] = sign * amount
                    move_body(self.client, self.task.objects[lighter], delta)
                    self.report["separated"].append(
                        {"move": lighter, "from": heavier,
                         "axis": "xy"[axis] + ("+" if sign > 0 else "-"),
                         "amount_m": round(amount, 4)})

    def _settle(self, steps: int) -> None:
        start = audit.poses(self.client, self.task.objects)
        for _ in range(int(steps)):
            p.stepSimulation(physicsClientId=self.client)
        end = audit.poses(self.client, self.task.objects)
        drifts = {key: float(np.linalg.norm(end[key][0] - start[key][0])) for key in end}
        top = self._table_top()
        below = sorted(key for key, value in end.items()
                       if key != TABLE_KEY and float(value[0][2]) < top - 0.05)
        worst = sorted(drifts.items(), key=lambda pair: -pair[1])[:3]
        self.report["settle"] = {
            "steps": int(steps),
            "max_drift_m": round(max(drifts.values()), 4) if drifts else 0.0,
            "drift": {key: round(value, 3) for key, value in worst if value > 0.01},
            "below_table": below,
            "stable": bool(drifts and max(drifts.values()) <= audit.DRIFT_M and not below),
        }

    def _in_reach(self) -> None:
        keys = self._movable_keys()
        if self.shoulder is None or self.reach is None or not keys:
            return
        dists = {key: float(np.linalg.norm(
            body_pose(self.client, self.task.objects[key])[0] - self.shoulder))
            for key in keys}
        self.report["in_reach"] = {
            "total": len(dists),
            "inside": int(sum(1 for value in dists.values() if value <= self.reach)),
            "nearest_m": round(min(dists.values()), 4),
            "farthest_m": round(max(dists.values()), 4),
            "reach_m": round(self.reach, 4),
        }

    # -- 对外 ---------------------------------------------------------------
    def prepare(self, settle_steps: int = SETTLE_STEPS, shift: bool = True,
                bench: bool = True) -> dict:
        # SCENE_OPS 用的是**作者写的绝对坐标**（0.555 = 柜面 0.55 + 半厚 0.015），
        # 所以必须排在平移之前；平移一次动全部，相对关系不会被拆散。
        self._apply_ops()
        if shift:
            self._shift()
        if bench:
            self._bench()
        self._rest_on_table()
        self._clamp_to_reach()
        self._rest_on_table()
        self._separate()
        self._clamp_to_reach()
        self._compress_to_reach()
        self._settle(settle_steps)
        # 静置后物体可能被挤出去几厘米（锅被两个手柄推），最后再收一次；
        # 只动 x/y、不再跑物理，drift 记的是"收之前"的漂移，不掩饰。
        self._clamp_to_reach()
        self._in_reach()
        self.report["criteria_at_reset"] = bool(self.task.check_success())
        return self.report

    def cleanup(self) -> None:
        for body in self.extra_bodies:
            try:
                p.removeBody(body, physicsClientId=self.client)
            except p.error:
                pass
        self.extra_bodies = []


# ---------------------------------------------------------------------------
# trial runner
# ---------------------------------------------------------------------------
def shift_oracle(ops, dz: float) -> list:
    """把 ORACLE 里**绝对 z** 的 op 抬 dz —— 场景层抬了多少，判据的绝对量就跟着抬多少。

    不这么做的话，``pickcube``（``cube z > 0.66``）这类判据会被场景层"抬坏"，
    ``--verify`` 会误报成 NOT_TRIGGERABLE。
    """
    out = []
    for op in ops:
        kind = op[0]
        if kind == "set_z":
            out.append(("set_z", op[1], float(op[2]) + dz))
        elif kind == "set_pos":
            out.append(("set_pos", op[1], (op[2][0], op[2][1], float(op[2][2]) + dz)))
        elif kind == "upright":
            out.append(("upright", op[1], float(op[2]) + dz))
        else:
            out.append(op)
    return out


def measure_arm(robot: str) -> dict:
    """两肩中点 + 单臂臂长（上臂 + 前臂）。

    用**两肩中点**而不是某一侧：``pushcube`` 的方块在 y=-0.2，离左肩 0.4 m、离右肩 0.0 m，
    只取"第一个 shoulder_pitch"会把可达判成不可达（实测 dz 被逼到 +1.00 m，台面抬到肩高）。
    """
    from teleoperation.simulation import RobotLoader

    client = p.connect(p.DIRECT)
    try:
        audit.fresh(client, ground=False)
        body = RobotLoader(client).load_robot(robot)["robot"]
        points = audit.link_points(client, body)
    finally:
        p.disconnect(client)
    low = {name.lower(): point for name, point in points.items()}
    shoulders = {name: point for name, point in low.items()
                 if "shoulder" in name and "pitch" in name}
    if not shoulders:
        shoulders = {name: point for name, point in low.items() if "shoulder" in name}
    elbows = {name: point for name, point in low.items() if "elbow" in name}
    wrists = {name: point for name, point in low.items() if "wrist" in name}
    if not shoulders or not elbows or not wrists:
        return {"robot": robot, "reach_m": None, "shoulder_m": None}
    upper = fore = 0.0
    for side in ("left", "right"):
        near_shoulder = [point for name, point in shoulders.items() if side in name]
        near_elbow = [point for name, point in elbows.items() if side in name]
        near_wrist = [point for name, point in wrists.items() if side in name]
        if near_shoulder and near_elbow and near_wrist:
            length = (float(np.linalg.norm(near_elbow[0] - near_shoulder[0]))
                      + float(np.linalg.norm(near_wrist[0] - near_elbow[0])))
            if length > upper + fore:
                upper = float(np.linalg.norm(near_elbow[0] - near_shoulder[0]))
                fore = float(np.linalg.norm(near_wrist[0] - near_elbow[0]))
    if upper + fore <= 0.0:
        first_shoulder = next(iter(shoulders.values()))
        first_elbow = next(iter(elbows.values()))
        first_wrist = next(iter(wrists.values()))
        upper = float(np.linalg.norm(first_elbow - first_shoulder))
        fore = float(np.linalg.norm(first_wrist - first_elbow))
    centre = np.mean(np.stack(list(shoulders.values())), axis=0)
    return {"robot": robot,
            "reach_m": round(upper + fore, 4),
            "upper_m": round(upper, 4),
            "fore_m": round(fore, 4),
            "shoulder_m": [round(float(v), 4) for v in centre],
            "shoulders": {name: [round(float(v), 4) for v in point]
                          for name, point in shoulders.items()}}


def build_actions(policy: str, env, path: str = None, steps: int = None):
    """按 policy 造动作流；``None`` = 每步不下发动作（idle）。"""
    if policy == "idle":
        return None, {"policy": "idle"}
    from teleoperation.applications.replay.replay_actions import (
        make_dummy_actions, validate_dim)
    if policy == "dummy":
        actions = make_dummy_actions(env.action_dim, n_steps=int(steps or 480))
        validate_dim(actions, env.action_dim, env.robot_type)
        return actions, {"policy": "dummy", "source": "sine-wave"}
    if policy == "replay":
        if not path:
            raise ValueError("--policy replay 需要 --actions <契约H 动作文件>")
        from teleoperation.applications.replay.actions import load_action_file
        actions, meta = load_action_file(path)
        validate_dim(actions, env.action_dim, env.robot_type)
        return actions, dict(meta, policy="replay", file=str(path))
    raise ValueError("unknown policy: %r" % (policy,))


def joint_positions(env) -> np.ndarray:
    """按 action 顺序取机器人关节角（用来证明"动作真的送到关节上了"）。"""
    indices = getattr(env, "action_joint_indices", None) or []
    if not indices:
        return np.zeros(0)
    return np.array([float(p.getJointState(env.robot_id, int(index),
                                           physicsClientId=env.client)[0])
                     for index in indices])


def run_trial(env, actions, max_steps: int) -> dict:
    """逐步推进并每帧查判据；**自己算仿真时间**（baseline 用 time.time()，空转也要等 300 s）。"""
    progress = max(1, int(max_steps) // 10)
    success = False
    completion = None
    step = 0
    before = joint_positions(env)
    t0 = time.time()
    for step in range(int(max_steps)):
        if actions is not None:
            env.task.apply_action(actions[min(step, len(actions) - 1)],
                                  joint_indices=env.action_joint_indices)
        p.stepSimulation(physicsClientId=env.client)
        if env.task.check_success():
            success = True
            completion = (step + 1) * DT
            break
        if env._render and step % progress == 0:
            time.sleep(DT)
    after = joint_positions(env)
    travel = (float(np.abs(after - before).max())
              if before.size and after.size == before.size else 0.0)
    return {"success": bool(success),
            "completion_time_s": None if completion is None else round(completion, 3),
            "steps": int(step + 1),
            "sim_time_s": round((step + 1) * DT, 3),
            "joint_travel_rad": round(travel, 4),
            "wall_s": round(time.time() - t0, 2)}


def object_positions(env) -> dict:
    return {key: [round(float(v), 4) for v in body_pose(env.client, body)[0]]
            for key, body in env.task.objects.items()}


def evaluate_task(robot: str, task_name: str, trials: int, policy: str,
                  actions_path: str = None, max_steps: int = 14400,
                  settle_steps: int = SETTLE_STEPS, scene: bool = True,
                  render: bool = False, verbose: bool = True) -> dict:
    """一个任务跑 N 个 trial，返回 report（含每个 trial 的 JSON 段）。"""
    from teleoperation.simulation.environment import SimulationEnv

    arm = measure_arm(robot) if scene else {}
    report = {"meta": {"robot": robot, "task": task_name, "trials": int(trials),
                       "policy": policy, "scene": bool(scene),
                       "settle_steps": int(settle_steps), "max_steps": int(max_steps),
                       "criteria_invalid": task_name in CRITERIA_INVALID,
                       "criteria_note": CRITERIA_INVALID.get(task_name),
                       "arm": {key: arm.get(key) for key in
                               ("reach_m", "upper_m", "fore_m", "shoulder_m")}},
              "trials": []}
    # record=False：baseline 的 record=True 每步抄 11 MB 传感器数据，评估用不上
    for index in range(int(trials)):
        env = SimulationEnv(robot_type=robot, task_name=task_name, render=render,
                            record=False,
                            data_dir=str(ROOT / "outputs" / "tmp_task_eval"))
        try:
            env.reset(randomize=False)
            scene_report = None
            if scene:
                scene_report = TaskScene(env.client, env.task, robot=robot,
                                         shoulder=arm.get("shoulder_m"),
                                         reach=arm.get("reach_m"),
                                         name=task_name,
                                         robot_id=env.robot_id).prepare(settle_steps)
            actions, policy_meta = build_actions(policy, env, actions_path, max_steps)
            result = run_trial(env, actions, max_steps)
            trial = dict(result, trial=index, **policy_meta,
                         scene=scene_report, final_objects=object_positions(env))
            report["trials"].append(trial)
            if verbose:
                print("[%s] %s/%s trial %d: %s" % (
                    "OK  " if trial["success"] else "FAIL", task_name, robot, index,
                    ("success at %.2f s" % trial["completion_time_s"])
                    if trial["success"] else
                    "no success in %.1f s" % trial["sim_time_s"]), flush=True)
        finally:
            env.close()
    successes = [item for item in report["trials"] if item["success"]]
    times = [item["completion_time_s"] for item in successes]
    report["summary"] = {
        "successes": len(successes),
        "trials": len(report["trials"]),
        "success_rate": (round(len(successes) / len(report["trials"]), 4)
                         if report["trials"] else 0.0),
        "mean_completion_s": round(float(np.mean(times)), 3) if times else None,
    }
    return report


# ---------------------------------------------------------------------------
# --verify：30 个任务过场景层后的四项检查（批次 2 的验收证据）
# ---------------------------------------------------------------------------
def verify_task(client: int, name: str, robot: str, shoulder, reach,
                settle_steps: int = SETTLE_STEPS) -> dict:
    from teleoperation.simulation.tasks import get_task

    row = {"task": name, "robot": robot, "scene": None, "stable": None,
           "max_drift_m": None, "no_fp": None, "trig": None, "in_reach": None,
           "criteria_invalid": name in CRITERIA_INVALID,
           "criteria_absolute": name in CRITERIA_ABSOLUTE_Z,
           "scene_unsupported": name in SCENE_UNSUPPORTED, "error": None,
           "verdict": "LOAD_ERROR"}
    task = None
    robot_id = None
    try:
        audit.fresh(client)
        if SCENE_MODE.get(name) == "sink":
            from teleoperation.simulation import RobotLoader
            robot_id = RobotLoader(client).load_robot(robot)["robot"]
        task = get_task(name)(client)
        task.reset()
        scene = TaskScene(client, task, robot=robot, shoulder=shoulder,
                          reach=reach, name=name, robot_id=robot_id)
        report = scene.prepare(settle_steps)
        row["scene"] = report
        row["stable"] = bool(report["settle"].get("stable"))
        row["max_drift_m"] = report["settle"].get("max_drift_m")
        row["no_fp"] = not bool(report["criteria_at_reset"])
        info = report.get("in_reach") or {}
        row["in_reach"] = {"inside": info.get("inside"), "total": info.get("total")}
        dz = float(report["shift"].get("dz_obj") or 0.0)
        audit.apply_oracle(client, task, shift_oracle(audit.ORACLE[name], dz))
        row["trig"] = bool(task.check_success())
        if row["criteria_invalid"]:
            row["verdict"] = "CRITERIA_INVALID"
        elif row["criteria_absolute"]:
            row["verdict"] = "CRITERIA_ABSOLUTE"
        elif row["scene_unsupported"]:
            row["verdict"] = "SCENE_UNSUPPORTED"
        elif not row["stable"]:
            row["verdict"] = "UNSTABLE"
        elif not row["no_fp"]:
            row["verdict"] = "FALSE_POSITIVE"
        elif not row["trig"]:
            row["verdict"] = "NOT_TRIGGERABLE"
        elif info and info.get("inside") != info.get("total"):
            row["verdict"] = "OUT_OF_REACH"
        else:
            row["verdict"] = "OK"
    except Exception as error:  # noqa: BLE001 - 一个任务炸掉不能拖垮整张表
        row["error"] = "%s: %s" % (type(error).__name__, error)
        row["verdict"] = "LOAD_ERROR"
    finally:
        if task is not None:
            try:
                task._clear_objects()
            except Exception:  # noqa: BLE001 - 清理失败不影响结论
                pass
    return row


def print_verify_table(rows) -> None:
    print("%-16s %-6s %-6s %-6s %-6s %-6s %s" % (
        "task", "stable", "no_fp", "trig", "reach", "drift", "verdict"))
    print("-" * 66)

    def mark(value):
        return "-" if value is None else ("OK" if value else "FAIL")

    for row in rows:
        reach = row["in_reach"] or {}
        reach_text = "%s/%s" % (reach.get("inside"), reach.get("total"))
        drift = "-" if row["max_drift_m"] is None else "%.3f" % row["max_drift_m"]
        print("%-16s %-6s %-6s %-6s %-6s %-6s %s" % (
            row["task"], mark(row["stable"]), mark(row["no_fp"]), mark(row["trig"]),
            reach_text, drift, row["verdict"]))
    print("-" * 66)


def verify(tasks, robots, settle_steps: int = SETTLE_STEPS,
           verbose: bool = True) -> list:
    """每个任务一个干净 client + 一台机器人的 FK：过完场景层再量四项检查。"""
    from teleoperation.simulation.tasks import register_builtin_tasks
    register_builtin_tasks()
    rows = []
    for robot in robots:
        arm = measure_arm(robot)
        shoulder = arm.get("shoulder_m")
        reach = arm.get("reach_m")
        if verbose:
            print("\n[%s] reach=%s m  shoulder=%s" % (robot, reach, shoulder),
                  flush=True)
        client = p.connect(p.DIRECT)
        try:
            for name in tasks:
                row = verify_task(client, name, robot, shoulder, reach, settle_steps)
                rows.append(row)
                if verbose:
                    shift = (row["scene"] or {}).get("shift") or {}
                    print("  %-16s %-17s dz=%+.3f far=%s note=%s" % (
                        name, row["verdict"], float(shift.get("dz_obj") or 0.0),
                        shift.get("farthest_m"),
                        "; ".join((row["scene"] or {}).get("notes") or []) or "-"),
                        flush=True)
        finally:
            p.disconnect(client)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="30 个任务的任务级评估（场景层 + trial runner）")
    parser.add_argument("--task", nargs="+", default=None,
                        help="任务名（默认：--verify 时 30 个全跑，否则 10 个代表任务）")
    parser.add_argument("--robot", default="h1_2", choices=list(ROBOTS) + ["all"],
                        help="机器人（--verify 时 all = 三台全跑）")
    parser.add_argument("--trials", type=int, default=1, help="每个任务跑几个 trial")
    parser.add_argument("--policy", default="idle", choices=["idle", "dummy", "replay"],
                        help="idle=不下发动作（查假成功）；dummy=正弦假动作；replay=契约H 动作文件")
    parser.add_argument("--actions", default=None, help="--policy replay 的动作文件")
    parser.add_argument("--max-sim-s", type=float, default=DEFAULT_MAX_SIM_S,
                        help="一个 trial 最多多少**仿真**秒（默认 60）")
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS,
                        help="场景摆好后空转多少步（默认 600 = 2.5 s）")
    parser.add_argument("--no-scene", action="store_true", help="不套场景层（baseline 原样跑）")
    parser.add_argument("--verify", action="store_true",
                        help="跑四项检查（30 个任务过场景层），不跑 trial")
    parser.add_argument("--render", action="store_true", help="GUI 渲染（默认 DIRECT）")
    parser.add_argument("--json", default=None, help="报告落盘路径")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from teleoperation.simulation.tasks import list_tasks, register_builtin_tasks

    register_builtin_tasks()
    names = list_tasks()
    tasks = list(args.task) if args.task else None
    if args.verify:
        tasks = tasks or names
        unknown = [name for name in tasks if name not in names]
        if unknown:
            print("[FAIL] unknown task(s): %s" % unknown)
            return 2
        robots = list(ROBOTS) if args.robot == "all" else [args.robot]
        rows = verify(tasks, robots, args.settle_steps)
        print()
        print_verify_table(rows)
        failed = [row for row in rows
                  if row["verdict"] not in EXCLUDED_VERDICTS + ("OK",)]
        payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "criteria_invalid": CRITERIA_INVALID,
                   "criteria_absolute_z": CRITERIA_ABSOLUTE_Z,
                   "scene_unsupported": SCENE_UNSUPPORTED,
                   "scene_ops": {key: [list(op) for op in ops]
                                 for key, ops in SCENE_OPS.items()},
                   "rows": rows}
        target = Path(args.json) if args.json else (
            ROOT / "outputs" / "task_eval" / "verify.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("report -> %s" % target)
        if failed:
            print("[FAIL] %d task(s) not evaluable: %s" % (
                len(failed), ", ".join(sorted({row["task"] for row in failed}))))
            return 1
        scored = len([row for row in rows if row["verdict"] == "OK"])
        print("[OK  ] %d/%d tasks evaluable, %d excluded: %s" % (
            scored, len(rows), len(rows) - scored,
            ", ".join(sorted({row["verdict"] for row in rows
                              if row["verdict"] in EXCLUDED_VERDICTS}))))
        return 0

    tasks = tasks or list(DEFAULT_TASKS)
    unknown = [name for name in tasks if name not in names]
    if unknown:
        print("[FAIL] unknown task(s): %s" % unknown)
        return 2
    max_steps = max(1, int(round(args.max_sim_s / DT)))
    reports = []
    for name in tasks:
        report = evaluate_task(args.robot, name, args.trials, args.policy,
                               actions_path=args.actions, max_steps=max_steps,
                               settle_steps=args.settle_steps,
                               scene=not args.no_scene, render=args.render)
        reports.append(report)
        summary = report["summary"]
        print("[%s] %s/%s: %d/%d success, mean %s s" % (
            "OK  " if summary["successes"] or args.policy == "idle" else "FAIL",
            name, args.robot, summary["successes"], summary["trials"],
            summary["mean_completion_s"]), flush=True)
    target = Path(args.json) if args.json else (
        ROOT / "outputs" / "task_eval"
        / ("%s_%s_%d.json" % (args.robot, args.policy, len(tasks))))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "criteria_invalid": CRITERIA_INVALID, "reports": reports}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print("report -> %s" % target)
    false_success = [report["meta"]["task"] for report in reports
                     if report["summary"]["successes"] and args.policy == "idle"]
    if false_success:
        print("[FAIL] idle policy already satisfies: %s" % false_success)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
