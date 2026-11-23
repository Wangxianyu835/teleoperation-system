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

    # 摄像头实时驱动（现场用：右手举到胸前 -> 机器人右臂跟；先站好 1.5 s 做中性标定）
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --task pushcube --policy camera ^
        --side right --max-wall-s 40 --render

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
SUCCESS_HOLD_STEPS = 120
SUCCESS_TABLE_MARGIN_M = 0.08
#: 论文 Table 1 里被拿去做量化对比的 10 个代表任务
DEFAULT_TASKS = ("pushcube", "pickcube", "pickplacecube", "uprearcup", "balltrashcan",
                 "ballmug", "ballbimanual", "potbimanual", "pottomatoplate",
                 "penbrushpot")
#: 判据在 reset 时就成立（阈值 <= 物体半宽），成功率只能是 100%，不能计入。
#: The originally listed liftmug / opendrawer / openmicrowave cases were fixed
#: by using seated-pose thresholds and non-overlapping closed geometry.
CRITERIA_INVALID = {}
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
    # The task itself now uses the same thin panel and a 0.19 m open threshold.
    "opendrawer": [("resize", "drawer", (0.015, 0.18, 0.1)),
                   ("place", "drawer", (0.565, 0.3, None))],
}
#: 判据里写死了**世界坐标 z**、且**改场景也救不回来**的任务。场景层抬台面后这些判据要么恒真，
#: 要么永远不成立（breadtoaster / rotatehearth 要求按到 0.43 以下 = 按进台面里）。
#: **不是场景层能修的** —— 判据得改成"相对初始位姿"，这是交给任务作者的问题清单
#: （docs/task-evaluation.md §4.5）。
CRITERIA_ABSOLUTE_Z = {}
#: 判据本身要求"物体在容器里"，而容器是**实心** primitive —— 场景层摆不出来。
SCENE_UNSUPPORTED = {}
#: Pairs whose AABB overlap is intentional because one object is inside a
#: hollow container.  The generic separation pass must not eject the content.
SCENE_CONTAINERS = {
    "pottomatoout": {("pot", "tomato")},
}
#: 判据里写死绝对 z、但任务本身还能救的：改用"把机器人降到作业高度"（桌面保持作者写的
#: 0.42 m），这样判据的原意不变。``rotatehearth`` / ``breadtoaster`` 不在这里 ——
#: 它们要求把旋钮/按钮按到桌面以下（按进台面里），降机器人也救不了。
SCENE_MODE = {}
#: Per-task camera workspaces that need a lower table plus a matching robot drop.
CAMERA_WORKSPACE_PROFILES = {
    "uprearcup": {
        "robot_drop_m": 0.03,
        "table_top_m": 0.89,
        "layout_forward_m": 0.20,
        "layout_table_min_x_m": 0.08,
    },
}
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


def guard_table_target(point, bounds, clearance_m: float = 0.06,
                       margin_m: float = 0.05):
    """Keep a wrist target from being commanded into the table collider.

    The guard applies only when the target projects inside the table's x/y footprint.
    In front of the table it remains legal to move below the tabletop. Returns
    ``(guarded_point, was_guarded)``.
    """
    point = np.asarray(point, dtype=float)
    low, high = (np.asarray(value, dtype=float) for value in bounds)
    inside_xy = (low[0] - margin_m <= point[0] <= high[0] + margin_m
                 and low[1] - margin_m <= point[1] <= high[1] + margin_m)
    floor = float(high[2]) + max(0.0, float(clearance_m))
    if inside_xy and point[2] < floor:
        guarded = point.copy()
        guarded[2] = floor
        return guarded, True
    return point, False


def grasp_assist_excluded(task_name: str) -> frozenset:
    """Objects that must stay as targets instead of being latched as grasped."""
    if task_name == "pickplacecube":
        return frozenset(("plate",))
    return frozenset()


def camera_workspace_profile(task_name: str) -> dict:
    return dict(CAMERA_WORKSPACE_PROFILES.get(task_name, {}))


def camera_objects_frozen(freeze_mode: str,
                          allow_gui_mouse_picking: bool) -> bool:
    """GUI dragging requires dynamic bodies; formal runs may freeze them."""
    return str(freeze_mode) == "on" and not bool(allow_gui_mouse_picking)


def clamp_horizontal_velocity(client: int, body: int,
                              max_linear_m_s: float = 0.30,
                              max_angular_rad_s: float = 1.5) -> bool:
    """Limit only explosive horizontal/angular contact velocity of a task object."""
    linear, angular = p.getBaseVelocity(body, physicsClientId=client)
    linear = np.asarray(linear, dtype=np.float64)
    angular = np.asarray(angular, dtype=np.float64)
    horizontal = float(np.linalg.norm(linear[:2]))
    spin = float(np.linalg.norm(angular))
    changed = False
    if horizontal > max_linear_m_s > 0.0:
        linear[:2] *= float(max_linear_m_s) / horizontal
        changed = True
    if spin > max_angular_rad_s > 0.0:
        angular *= float(max_angular_rad_s) / spin
        changed = True
    if changed:
        p.resetBaseVelocity(
            body, [float(value) for value in linear],
            [float(value) for value in angular],
            physicsClientId=client)
    return changed


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
                       "ik_reach": {}, "notes": []}


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
                    pairs = SCENE_CONTAINERS.get(self.name, set())
                    if ((key_a, key_b) in pairs
                            or (key_b, key_a) in pairs):
                        continue
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

    def _clamp_to_reach(self, margin: float = 0.02,
                        exclude_keys=None) -> None:
        """把还落在可达球外的**整套**物体沿水平方向收拢到球面内。

        按 AABB 连通的组一起平移：``drawerbook`` 的抽屉被单独拉 0.10 m 会离开柜体
        （|dx| 从 0.155 变 0.255），``drawer_closed`` 判据立刻失效。只动 x/y、不动 z：
        不动 z 物体才不离台面浮起，也不破坏判据里的绝对高度。
        """
        if self.shoulder is None or self.reach is None:
            return
        limit = float(self.reach) - margin
        excluded = set(exclude_keys or ())
        clamped = {}
        for raw_group in self._touch_groups():
            group = [key for key in raw_group if key not in excluded]
            if not group:
                continue
            need = 0.0
            positions = {}
            for key in group:
                pos, _orn = body_pose(self.client, self.task.objects[key])
                positions[key] = pos
                vertical = float(pos[2] - self.shoulder[2])
                if abs(vertical) >= limit:
                    # No horizontal point can enter the sphere at this height.
                    # Return rather than turning the numerical guard into a
                    # multi-metre object displacement.
                    self.report["notes"].append(
                        "clamp skipped for %s: vertical %.3f m outside reach %.3f m"
                        % (key, vertical, limit))
                    need = 0.0
                    break
                allowed = float(np.sqrt(max(0.0, limit * limit
                                            - vertical * vertical)))
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

    def _hold_motors(self, force: float = 2000.0) -> None:
        """把所有可动关节按**当前角**锁住（现场档 ``arm_scene_driver`` 的 hold motors 同做法）。

        不锁的后果实测（``outputs/tmp_probe/track.txt``）：settle 的 600 步里手臂靠重力垂到
        自己腿里、手与大腿互穿，之后 shoulder_pitch 指令 -1.03 rad 却在 +0.44 rad 卡住不动
        （电机 200/500/1200 都一样；把重力关掉也一样）——不是电机不够力，是**自己卡自己**。
        """
        if self.robot_id is None:
            return
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client)
            if info[2] == p.JOINT_FIXED:
                continue
            angle = float(p.getJointState(self.robot_id, index,
                                          physicsClientId=self.client)[0])
            p.setJointMotorControl2(self.robot_id, index, p.POSITION_CONTROL,
                                    targetPosition=angle, force=force,
                                    physicsClientId=self.client)

    def _settle(self, steps: int, hold: bool = False) -> None:
        start = audit.poses(self.client, self.task.objects)
        if hold:
            # Keep the body upright while camera/scripted actions drive only a subset of
            # joints. 200 force allowed H1-2's torso/legs to sag under the arm reaction
            # and eventually trap the controlled arm against the table.
            self._hold_motors()
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

    # -- 真 IK 可达性校正（批次 5）--------------------------------------------
    def _ik_solver(self):
        """position 口径的 IK（脚本档抓取不看腕朝向）：每个场景建一次。"""
        from teleoperation.simulation.arm_ik import PyBulletArmIK

        return PyBulletArmIK(self.robot_id, self.client, robot_type=self.robot,
                             sides=("left", "right"), rest_poses="current",
                             ik_mode="position", lock_others=True)

    def _joint_state(self) -> dict:
        """整机可动关节的当前角（``solve`` 会 teleport 手臂，量完必须还原）。"""
        out = {}
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client)
            if info[2] == p.JOINT_FIXED:
                continue
            out[index] = float(p.getJointState(self.robot_id, index,
                                               physicsClientId=self.client)[0])
        return out

    def _ik_residual_mm(self, ik, point, side: str = None) -> tuple:
        """这一点（世界系）用哪只手残差最小、残差多少。``side`` 指定时只量那只手。"""
        state = self._joint_state()
        best = (float("inf"), "")
        try:
            for name in ik.arm:
                if side and name != side:
                    continue
                ik.solve(name, tuple(float(v) for v in point), source="world")
                value = float(ik.residual_mm(name))
                if value < best[0]:
                    best = (value, name)
        finally:
            for index, value in state.items():
                p.resetJointState(self.robot_id, index, value,
                                  physicsClientId=self.client)
        return best


    def _reach_keys(self) -> list:
        """可达性校正要照顾的物体：除桌子外**全部**（含 mass=0 的目标标记）。

        ``_movable_keys`` 只收 mass>0，那是给「抬到台面上 / 退让」用的；可达性不同 ——
        ``pushcube`` 要把方块送到 ``target``，target 自己是 mass=0 的标记物，够不着就等于
        这条任务永远做不完（实测 target 在 y=+0.30 处 IK 残差 642 mm）。
        """
        return [key for key in self.task.objects if key != TABLE_KEY]

    def _layout_for_arm(self, forward: float = 0.26, drop: float = 0.22,
                        lateral: float = 0.02, spread: float = 0.12,
                        side_bias: float = 0.0,
                        table_min_x: float = 0.24,
                        rebuild_bench: bool = True) -> None:
        """按**这台臂的真实几何**重摆整幕（脚本档专用）。

        为什么不再用 ``solve_shift`` 的球模型：那个模型把物体放在 0.9×reach 的球面上，
        手臂伸直才勉强够到，压根没有「从上方接近」的余量；而且桌子会跟着漂到机器人身体里
        （实测桌面近沿 x=0.083 m，机器人胸口就在桌子里，``home`` 航点差 0.123 m）。

        这里只做三件事，全部按肩关节算：
        1. 物体**整组**平移：质心放到「肩前 ``forward``、横向不超过 ``lateral``」；
        2. 桌子跟着物体走，并把**桌面顶面**放到比肩低 ``drop``；近沿留在 x=0.24 m；
        3. 物体随桌子一起升降，落到桌面上（不重建任何物体，相对布置不变）。
        """
        if self.shoulder is None or self.robot_id is None:
            return
        keys = [key for key in self.task.objects if key != TABLE_KEY]
        if not keys:
            return
        shoulder = np.asarray(self.shoulder, dtype=float)
        points = np.stack([body_pose(self.client, self.task.objects[key])[0]
                           for key in keys])
        centre = points.mean(axis=0)
        # 先把「相对布置」等比压缩进 spread 半径：pushcube 的方块离目标 0.54 m，
        # 一条 0.41 m 的手臂一次搬运根本走不完（实测 carry 差 0.079 m / lower 0.138 m）。
        offsets = points[:, :2] - centre[:2]
        reach_xy = float(np.linalg.norm(offsets, axis=1).max()) if len(points) > 1 else 0.0
        scale = 1.0 if reach_xy <= spread or reach_xy <= 1e-9 else spread / reach_xy
        for key, offset in zip(keys, offsets):
            if scale >= 1.0:
                break
            target_xy = centre[:2] + offset * scale
            current = body_pose(self.client, self.task.objects[key])[0]
            move_body(self.client, self.task.objects[key],
                      np.array([target_xy[0] - float(current[0]),
                                target_xy[1] - float(current[1]), 0.0]))
        if scale < 1.0:
            points = np.stack([body_pose(self.client, self.task.objects[key])[0]
                               for key in keys])
            centre = points.mean(axis=0)
            print("[scene] layout_for_arm: relative spread %.3f -> %.3f m (x%.2f)"
                  % (reach_xy, reach_xy * scale, scale), flush=True)
        want_x = float(shoulder[0] + forward)
        # 横向按**整组中点**对齐肩（不是质心）：物体各自最多 ±spread，中点对齐后每个物体
        # 离肩的横向就 ≤spread，左右两条臂都够得着（实测可达带在 y≈±0.10）。
        y_mid = 0.5 * (float(points[:, 1].min()) + float(points[:, 1].max()))
        want_y = float(
            shoulder[1]
            + max(-lateral, min(lateral, y_mid - shoulder[1] + side_bias)))
        delta = np.array([want_x - centre[0], want_y - y_mid, 0.0])

        for key in keys:
            move_body(self.client, self.task.objects[key], delta)
        table = self.task.objects.get(TABLE_KEY)
        if table is not None:
            # 桌子整体跟着物体平移；平移后若近沿还伸进机器人身前 0.24 m 以内，再单独往前推
            move_body(self.client, table, np.array([delta[0], delta[1], 0.0]))
            low, high = aabb(self.client, table)
            object_min_x = min(
                (body_pose(self.client, self.task.objects[key])[0][0]
                 for key in keys), default=float(low[0]) + 0.10)
            # Keep a small support margin behind the frontmost object.  A 5 cm
            # margin used to pull the table all the way back to x=0.209 m even
            # when the caller explicitly requested x>=0.24 m.
            desired_min_x = min(float(table_min_x), object_min_x - 0.02)
            # The side-lying cup is 10 cm long along x.  Let its table move back
            # independently so it cannot begin only 2 cm from the near edge.
            if self.name == "uprearcup":
                desired_min_x = float(table_min_x)
            if (float(low[0]) < desired_min_x
                    or (self.name == "uprearcup"
                        and float(low[0]) > desired_min_x + 1e-4)):
                move_body(self.client, table,
                          np.array([desired_min_x - float(low[0]), 0.0, 0.0]))
                low, high = aabb(self.client, table)

            want_top = float(shoulder[2] - drop)
            shift_z = want_top - float(high[2])
            move_body(self.client, table, np.array([0.0, 0.0, shift_z]))
            for key in keys:
                move_body(self.client, self.task.objects[key], np.array([0.0, 0.0, shift_z]))
            self.report["notes"].append(
                "layout_for_arm: objects moved by %s, table top -> %.3f m, near edge %.3f m"
                % ([round(float(v), 3) for v in delta], want_top, float(low[0])))
            print("[scene] layout_for_arm: table top=%.3f near_edge=%.3f objects_dxy=%s"
                  % (want_top, float(low[0]), [round(float(v), 3) for v in delta[:2]]),
                  flush=True)
            # ★ 落地支撑必须按**新**桌面高度重建：旧支撑停在原桌面高度，等于在机器人胸口
            #   立了一堵墙（实测每个航点都差 ~0.12 m，手臂顶在墙上）。
            if rebuild_bench:
                self.cleanup()
                self._bench()

    def _separate_pickplace_camera(self, side: str = None) -> dict:
        """Put the cube in a deep right-side pick zone and the plate far forward.

        The original task puts the destination plate closer to the robot than the
        source cube.  Uniform scene compression then made them almost coincide,
        leaving no room to carry the cube forwards.  These offsets were measured
        with the H1-2 paper IK: both points stay below 3 mm residual and the
        source/destination centres are about 14 cm apart.
        """
        cube = self.task.objects.get("cube")
        plate = self.task.objects.get("plate")
        if cube is None or plate is None or self.shoulder is None:
            return {}
        sign = -1.0 if side == "right" else 1.0
        shoulder = np.asarray(self.shoulder, dtype=np.float64)
        cube_wanted = np.array([
            float(shoulder[0] + 0.32),
            float(shoulder[1] + sign * 0.10),
            float(body_pose(self.client, cube)[0][2]),
        ])
        plate_wanted = np.array([
            float(shoulder[0] + 0.38),
            float(shoulder[1] - sign * 0.04),
            float(body_pose(self.client, plate)[0][2]),
        ])
        for body, wanted in ((cube, cube_wanted), (plate, plate_wanted)):
            current = body_pose(self.client, body)[0]
            move_body(self.client, body, wanted - current)
        # The plate is a destination marker, not a manipulation target.  A light
        # 50 g plate is easy to knock to the table edge before the cube ever reaches
        # it, which removes the placement target from the reachable workspace.
        p.changeDynamics(plate, -1, mass=0.0, lateralFriction=1.2,
                         physicsClientId=self.client)
        self.report["camera_pickplace_plate_fixed"] = True
        spacing = float(np.linalg.norm(cube_wanted[:2] - plate_wanted[:2]))
        self.report["notes"].append(
            "camera pickplace: cube deep/right %s, plate far/forward %s,"
            " spacing %.3f m"
            % ([round(float(value), 3) for value in cube_wanted],
               [round(float(value), 3) for value in plate_wanted], spacing))
        self.report["camera_pickplace_spacing_m"] = round(spacing, 4)
        print("[scene] camera pickplace: cube right/deep, plate forward,"
              " spacing %.3f m" % spacing, flush=True)
        return {"cube": cube_wanted.tolist(), "plate": plate_wanted.tolist(),
                "spacing_m": spacing}

    def _stabilize_uprearcup_camera(self) -> dict:
        """Replace the smooth cylinder with one rigid, subtly lobed cup body."""
        cup = self.task.objects.get("cup")
        if cup is None:
            return {}
        position, orientation = body_pose(self.client, cup)
        half = [0.010, 0.008, 0.045]
        bump_offsets = [[0.0, 0.045, 0.0]]
        bump_count = len(bump_offsets)
        shape_types = [p.GEOM_CYLINDER] + [p.GEOM_BOX] * bump_count
        half_extents = [[0.0, 0.0, 0.0]] + [half] * bump_count
        frame_orientations = [[0.0, 0.0, 0.0, 1.0]] * (bump_count + 1)
        collision = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            radii=[0.04] + [0.0] * bump_count,
            lengths=[0.10] + [0.0] * bump_count,
            halfExtents=half_extents,
            collisionFramePositions=[[0.0, 0.0, 0.0]] + bump_offsets,
            collisionFrameOrientations=frame_orientations,
            physicsClientId=self.client)
        visual = p.createVisualShapeArray(
            shapeTypes=shape_types,
            radii=[0.04] + [0.0] * bump_count,
            lengths=[0.10] + [0.0] * bump_count,
            halfExtents=half_extents,
            rgbaColors=[[0.3, 0.7, 0.3, 1.0]] +
                       [[0.22, 0.55, 0.28, 1.0]] * bump_count,
            visualFramePositions=[[0.0, 0.0, 0.0]] + bump_offsets,
            visualFrameOrientations=frame_orientations,
            physicsClientId=self.client)
        p.removeBody(cup, physicsClientId=self.client)
        cup = p.createMultiBody(
            baseMass=0.20, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(value) for value in position],
            baseOrientation=[float(value) for value in orientation],
            physicsClientId=self.client)
        self.task.objects["cup"] = cup
        p.changeDynamics(
            cup, -1,
            lateralFriction=0.9,
            spinningFriction=0.005,
            rollingFriction=0.0,
            restitution=0.0,
            linearDamping=0.20,
            angularDamping=0.70,
            activationState=p.ACTIVATION_STATE_WAKE_UP,
            physicsClientId=self.client)
        settings = {
            "lateralFriction": 0.9,
            "spinningFriction": 0.005,
            "rollingFriction": 0.0,
            "restitution": 0.0,
            "linearDamping": 0.20,
            "angularDamping": 0.70,
            "compound_body": int(cup),
            "mass_kg": 0.20,
            "bump_half_extents_m": half,
            "bump_offsets_local_m": bump_offsets,
            "bump_count": len(bump_offsets),
        }
        self.report["camera_cup_stability"] = dict(settings)
        self.report["notes"].append(
            "camera uprearcup: rigid handle-like bump limits rolling")
        print("[scene] camera uprearcup: rigid anti-roll bump attached",
              flush=True)
        return settings

    def _place_uprearcup_camera_grasp(self, side: str = "right") -> dict:
        """Move the cup into the open hand's thumb/index/middle pinch region."""
        cup = self.task.objects.get("cup")
        if cup is None or self.robot_id is None:
            return {}
        prefix = "R_" if side == "right" else "L_"
        wanted = {
            prefix + "hand_base_link",
            prefix + "thumb_distal",
            prefix + "index_intermediate",
            prefix + "middle_intermediate",
        }
        positions = {}
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, index,
                                  physicsClientId=self.client)
            joint_name = info[1].decode("utf-8", errors="ignore")
            link_name = info[12].decode("utf-8", errors="ignore")
            link = int(info[16])
            if link < 0 or link_name not in wanted:
                continue
            positions[link_name] = np.asarray(
                p.getLinkState(
                    self.robot_id, link, computeForwardKinematics=True,
                    physicsClientId=self.client)[4],
                dtype=np.float64)
        if len(positions) != len(wanted):
            return {}
        palm = positions[prefix + "hand_base_link"]
        fingertip_center = (
            positions[prefix + "thumb_distal"]
            + positions[prefix + "index_intermediate"]
            + positions[prefix + "middle_intermediate"]) / 3.0
        target = np.array([
            palm[0] + 0.90 * (fingertip_center[0] - palm[0]),
            0.5 * (palm[1] + fingertip_center[1]),
            self._table_top() + 0.04,
        ])
        current, orientation = body_pose(self.client, cup)
        move_body(self.client, cup, target - current)
        details = {
            "side": side,
            "palm_m": palm.tolist(),
            "fingertip_center_m": fingertip_center.tolist(),
            "target_center_m": target.tolist(),
        }
        self.report["camera_cup_pre_grasp"] = details
        print("[scene] camera uprearcup: cup pre-grasp center %s"
              % [round(float(value), 3) for value in target], flush=True)
        return details

    def _place_pickplace_cube_camera_grasp(self, side: str = "right") -> dict:
        """Put the source cube into the same native-hand pinch region."""
        cube = self.task.objects.get("cube")
        if cube is None or self.robot_id is None:
            return {}
        prefix = "R_" if side == "right" else "L_"
        wanted = {
            prefix + "hand_base_link",
            prefix + "thumb_distal",
            prefix + "index_intermediate",
            prefix + "middle_intermediate",
        }
        positions = {}
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, index,
                                  physicsClientId=self.client)
            link_name = info[12].decode("utf-8", errors="ignore")
            link = int(info[16])
            if link < 0 or link_name not in wanted:
                continue
            positions[link_name] = np.asarray(
                p.getLinkState(
                    self.robot_id, link, computeForwardKinematics=True,
                    physicsClientId=self.client)[4],
                dtype=np.float64)
        if len(positions) != len(wanted):
            return {}
        palm = positions[prefix + "hand_base_link"]
        fingertip_center = (
            positions[prefix + "thumb_distal"]
            + positions[prefix + "index_intermediate"]
            + positions[prefix + "middle_intermediate"]) / 3.0
        target = np.array([
            palm[0] + 0.90 * (fingertip_center[0] - palm[0]),
            0.5 * (palm[1] + fingertip_center[1]),
            self._table_top() + 0.02,
        ])
        # A 1 cm inward correction keeps the smaller cube inside the verified
        # IK tolerance while remaining inside the native pinch envelope.
        target[0] -= 0.01
        table = self.task.objects.get(TABLE_KEY)
        if table is not None:
            low, _high = aabb(self.client, table)
            support_edge = float(target[0] - 0.055)
            if float(low[0]) > support_edge:
                move_body(self.client, table,
                          np.array([support_edge - float(low[0]), 0.0, 0.0]))
                self.report["camera_pickplace_table_edge_m"] = round(
                    support_edge, 4)
        current, orientation = body_pose(self.client, cube)
        move_body(self.client, cube, target - current)
        details = {
            "side": side,
            "palm_m": palm.tolist(),
            "fingertip_center_m": fingertip_center.tolist(),
            "target_center_m": target.tolist(),
        }
        self.report["camera_cube_pre_grasp"] = details
        label = "pickplace" if self.name == "pickplacecube" else self.name
        print("[scene] camera %s: cube pre-grasp center %s"
              % (label, [round(float(value), 3) for value in target]), flush=True)
        return details

    def _place_balltrashcan_camera(self, side: str = "right") -> dict:
        """Place the grasp ball in the native pinch center and fix the target can."""
        ball = self.task.objects.get("ball")
        can = self.task.objects.get("trashcan")
        if ball is None or can is None or self.robot_id is None:
            return {}
        prefix = "R_" if side == "right" else "L_"
        wanted = {
            prefix + "hand_base_link",
            prefix + "thumb_distal",
            prefix + "index_intermediate",
            prefix + "middle_intermediate",
        }
        positions = {}
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, index,
                                  physicsClientId=self.client)
            link_name = info[12].decode("utf-8", errors="ignore")
            link = int(info[16])
            if link < 0 or link_name not in wanted:
                continue
            positions[link_name] = np.asarray(
                p.getLinkState(
                    self.robot_id, link, computeForwardKinematics=True,
                    physicsClientId=self.client)[4],
                dtype=np.float64)
        if len(positions) != len(wanted):
            return {}
        palm = positions[prefix + "hand_base_link"]
        fingertip_center = (
            positions[prefix + "thumb_distal"]
            + positions[prefix + "index_intermediate"]
            + positions[prefix + "middle_intermediate"]) / 3.0
        target = np.array([
            palm[0] + 0.90 * (fingertip_center[0] - palm[0]),
            0.5 * (palm[1] + fingertip_center[1]),
            self._table_top() + 0.03,  # Ball radius; keep its bottom on the table.
        ])
        table = self.task.objects.get(TABLE_KEY)
        if table is not None:
            low, _high = aabb(self.client, table)
            support_edge = float(target[0] - 0.06)
            if float(low[0]) > support_edge:
                move_body(self.client, table,
                          np.array([support_edge - float(low[0]), 0.0, 0.0]))
        current, orientation = body_pose(self.client, ball)
        move_body(self.client, ball, target - current)
        p.changeDynamics(
            ball, -1,
            rollingFriction=0.12,
            spinningFriction=0.03,
            linearDamping=0.30,
            angularDamping=1.00,
            restitution=0.0,
            activationState=p.ACTIVATION_STATE_WAKE_UP,
            physicsClientId=self.client)
        can_position, can_orientation = body_pose(self.client, can)
        wall_half = [0.028, 0.008, 0.0675]
        bottom_half = [0.049, 0.049, 0.0075]
        wall_positions = []
        wall_orientations = []
        for angle in np.arange(8) * (np.pi / 4.0):
            wall_positions.append([
                float(0.049 * np.cos(angle)),
                float(0.049 * np.sin(angle)),
                0.0])
            wall_orientations.append(
                p.getQuaternionFromEuler([0.0, 0.0, float(angle)]))
        shape_types = [p.GEOM_BOX] * 9
        half_extents = [bottom_half] + [wall_half] * 8
        frame_positions = [[0.0, 0.0, -0.0675]] + wall_positions
        frame_orientations = [[0.0, 0.0, 0.0, 1.0]] + wall_orientations
        collision = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            collisionFramePositions=frame_positions,
            collisionFrameOrientations=frame_orientations,
            physicsClientId=self.client)
        visual = p.createVisualShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            rgbaColors=[[0.3, 0.3, 0.3, 1.0]] * 9,
            visualFramePositions=frame_positions,
            visualFrameOrientations=frame_orientations,
            physicsClientId=self.client)
        p.removeBody(can, physicsClientId=self.client)
        can = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(value) for value in can_position],
            baseOrientation=[float(value) for value in can_orientation],
            physicsClientId=self.client)
        self.task.objects["trashcan"] = can
        p.changeDynamics(can, -1, mass=0.0, lateralFriction=1.2,
                         physicsClientId=self.client)
        details = {
            "side": side,
            "palm_m": palm.tolist(),
            "fingertip_center_m": fingertip_center.tolist(),
            "target_center_m": target.tolist(),
            "trashcan_fixed": True,
            "trashcan_hollow": True,
            "wall_count": 8,
        }
        self.report["camera_balltrashcan_setup"] = details
        print("[scene] camera balltrashcan: ball pre-grasp %s; trashcan fixed"
              % [round(float(value), 3) for value in target], flush=True)
        return details


    def _pull_into_ik_workspace(self, tolerance_mm: float = 15.0,
                                max_shift_m: float = 0.30, rounds: int = 3,
                                side: str = None,
                                comfort_fraction: float = 0.80,
                                group_touch: float = 0.08) -> None:

        """把够不着的物体组沿「在桌面上就近挪」的方向试，直到**真 IK** 残差合格。

        为什么必须用 IK 而不是半径：``solve_shift`` 假设工作域是一个球，实测这台臂的
        可达区是一条斜带（``outputs/tmp_probe/workspace2.txt``：y=+0.10 一带残差 3-14 mm，
        y=-0.20 一带要到 x≈0.52 才 8 mm），球半径 0.55 m 上的点有一大半 IK 到不了
        （pushcube 的方块在 0.51 m 处残差 58-81 mm）。批次 2 的 ``in_reach`` 因此是乐观的。
        """
        if self.robot_id is None or not self.robot:
            return
        keys = self._reach_keys()
        if not keys:
            return

        ik = self._ik_solver()
        state = self._joint_state()
        try:
            groups = self._touch_groups(touch=float(group_touch))
            table = self.task.objects.get(TABLE_KEY)
            if table is not None:
                low, high = aabb(self.client, table)
            else:
                low, high = np.array([0.2, -0.8, 0.0]), np.array([1.2, 0.8, 1.0])
            ring = (0.02, 0.04, 0.07, 0.10, 0.14, 0.19, 0.25)
            angles = (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.57, -1.57,
                      2.09, -2.09, 2.62, -2.62, np.pi)
            # 「够得着」还不够：贴着可达边界（≈0.9 reach）的点虽然残差合格，但手臂伸直后
            # 没有余量从上方接近物体（实测 pushcube 的 hover 航点因此差 0.19 m）。
            # 所以还要求离肩不超过 0.80 * reach —— 留出肘部弯曲的余量。
            comfort = ((float(comfort_fraction) * float(self.reach))
                       if self.reach else None)

            moved = {}
            for group in groups:
                group = [key for key in group if key in keys]
                if not group:
                    continue
                for _round in range(int(rounds)):
                    worst_key, worst_value = None, 0.0
                    for key in group:
                        point = body_pose(self.client, self.task.objects[key])[0]
                        residual, _side = self._ik_residual_mm(ik, point, side=side)
                        far = (comfort is not None and self.shoulder is not None
                               and float(np.linalg.norm(np.asarray(point, dtype=float)
                                                        - self.shoulder)) > comfort)
                        if residual <= tolerance_mm and not far:
                            continue
                        score = residual if residual > tolerance_mm else tolerance_mm + 1.0
                        if score > worst_value:
                            worst_key, worst_value = key, score
                    if worst_key is None:
                        break

                    point = body_pose(self.client, self.task.objects[worst_key])[0]
                    chosen = None
                    for radius in ring:
                        if radius > max_shift_m:
                            break
                        for angle in angles:
                            delta = np.array([radius * np.cos(angle),
                                              radius * np.sin(angle), 0.0])
                            probe = np.asarray(point, dtype=float) + delta
                            if not (low[0] + 0.05 <= probe[0] <= high[0] - 0.05
                                    and low[1] + 0.05 <= probe[1] <= high[1] - 0.05):
                                continue
                            residual, _side = self._ik_residual_mm(ik, probe, side=side)
                            near = (comfort is None or self.shoulder is None
                                    or float(np.linalg.norm(
                                        np.asarray(probe, dtype=float) - self.shoulder))
                                    <= comfort)
                            if residual <= tolerance_mm and near:
                                chosen = delta
                                break
                        if chosen is not None:
                            break
                    if chosen is None:
                        message = ("ik workspace: %s 在 %.2f m 半径内找不到可达点（残差 %.0f mm）"
                                   % (worst_key, max_shift_m, worst_value))
                        self.report["notes"].append(message)
                        print("[scene] " + message, flush=True)
                        break
                    for key in group:
                        move_body(self.client, self.task.objects[key], chosen)
                    moved[worst_key] = round(float(moved.get(worst_key, 0.0)
                                                   + np.linalg.norm(chosen)), 4)
                    print("[scene] ik workspace: moved %s by %.3f m (was %.0f off)"
                          % (worst_key, float(np.linalg.norm(chosen)), worst_value),
                          flush=True)

            self._separate()
            final = {}
            for key in keys:
                point = body_pose(self.client, self.task.objects[key])[0]
                residual, side = self._ik_residual_mm(ik, point)
                final[key] = {"residual_mm": round(residual, 1), "side": side}
            self.report["ik_reach"] = {
                "tolerance_mm": float(tolerance_mm),
                "moved": moved,
                "objects": final,
                "ok": all(item["residual_mm"] <= tolerance_mm for item in final.values()),
            }
        finally:
            for index, value in state.items():
                p.resetJointState(self.robot_id, index, value,
                                  physicsClientId=self.client)


    # -- 对外 ---------------------------------------------------------------
    def prepare(self, settle_steps: int = SETTLE_STEPS, shift: bool = True,
                bench: bool = True, ik_workspace: bool = False,
                side: str = None, hold_robot: bool = False,
                layout_forward: float = 0.26,
                layout_drop: float = 0.22,
                layout_lateral: float = 0.02,
                layout_side_bias: float = 0.0,
                layout_table_min_x: float = 0.24,
                camera_pickplace_spacing: bool = False,
                camera_cube_pre_grasp: bool = False,
                camera_cup_stability: bool = False,
                camera_balltrashcan_setup: bool = False,
                ik_comfort_fraction: float = 0.80) -> dict:


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
        if ik_workspace:
            # 批次 5：球形模型把物体摆到了真 IK 够不到的地方 -> 先按真实臂几何重摆，
            # 再用 IK 残差做最后校正（两步都写进 scene report）
            self._layout_for_arm(forward=float(layout_forward),
                                 drop=float(layout_drop),
                                 lateral=float(layout_lateral),
                                 side_bias=float(layout_side_bias),
                                 table_min_x=float(layout_table_min_x))
            if camera_pickplace_spacing:
                self._separate_pickplace_camera(side=side)
            if camera_cup_stability:
                self._stabilize_uprearcup_camera()
            self._rest_on_table()
            self._separate()
            self._pull_into_ik_workspace(
                side=side, comfort_fraction=float(ik_comfort_fraction),
                group_touch=(0.0 if (camera_pickplace_spacing
                                     or camera_cube_pre_grasp) else 0.08))


        # Camera/replay actions only address arm + hand joints. Without holding the
        # rest of the body, gravity folds the legs into the arm path and the arm gets
        # blocked by its own robot before the task starts.
        self._settle(settle_steps, hold=(ik_workspace or hold_robot))


        # 静置后物体可能被挤出去几厘米（锅被两个手柄推），最后再收一次；
        # 只动 x/y、不再跑物理，drift 记的是"收之前"的漂移，不掩饰。
        clamp_exclusions = []
        if camera_pickplace_spacing:
            clamp_exclusions.extend(("plate", "cube"))
        elif camera_cube_pre_grasp:
            clamp_exclusions.append("cube")
        if camera_cup_stability:
            clamp_exclusions.append("cup")
        if camera_balltrashcan_setup:
            clamp_exclusions.extend(("ball", "trashcan"))
        self._clamp_to_reach(exclude_keys=tuple(clamp_exclusions))
        if camera_cup_stability:
            self._place_uprearcup_camera_grasp(side=side or "right")
        if camera_pickplace_spacing:
            self._place_pickplace_cube_camera_grasp(side=side or "right")
        elif camera_cube_pre_grasp:
            self._place_pickplace_cube_camera_grasp(side=side or "right")
        if camera_balltrashcan_setup:
            self._place_balltrashcan_camera(side=side or "right")
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


def sample_reach(client: int, body: int, robot: str, samples: int = 1200,
                 seed: int = 0) -> dict:
    """用 FK 采样量**真实**可达半径：``{side: 从肩关节到腕点的最大距离}``。

    为什么不能用连杆和：``measure_arm`` 原来把「肩 link 原点到肘」+「肘到腕」当半径
    （h1_2 = 0.5527 m），但那两段在静止位姿下并**不共线**，实测伸直也只有 0.4067 m
    （``outputs/tmp_probe/limits.txt``）；按 0.55 m 摆出来的物体，IK 残差 58-81 mm，
    手臂根本够不到（``outputs/tmp_probe/workspace2.txt``：y=-0.20 一带要 x≈0.52 才 8 mm）。
    这里按 URDF 限位随机采样 7 个臂关节，直接量腕点离肩多远 —— 这才是 IK 能用的球。
    """
    from teleoperation.simulation.arm_ik import PyBulletArmIK

    ik = PyBulletArmIK(body, client, robot_type=robot)
    rng = np.random.RandomState(int(seed))
    limits = {}
    for side in ik.arm:
        limits[side] = [(index, float(info[8]), float(info[9]))
                        for index, info in zip(ik.indices, ik.infos)
                        if index in {joint for joint, _name in ik.arm[side]}]
    best = {side: 0.0 for side in ik.arm}
    centre = {}
    neutral = {}
    for side in ik.arm:
        for index, _low, _high in limits[side]:
            neutral[index] = float(p.getJointState(body, index,
                                                   physicsClientId=client)[0])
        centre[side] = [round(float(v), 4) for v in ik.arm_points(side)["upper"]]
    try:
        for _ in range(int(samples)):
            for side in ik.arm:
                for index, low, high in limits[side]:
                    if not (low == low and high == high) or high <= low:
                        low, high = -3.14, 3.14
                    p.resetJointState(body, index, float(rng.uniform(low, high)),
                                      physicsClientId=client)
                points = ik.arm_points(side)
                radius = float(np.linalg.norm(np.asarray(points["wrist"])
                                              - np.asarray(points["upper"])))
                best[side] = max(best[side], radius)
    finally:
        for index, value in neutral.items():
            p.resetJointState(body, index, value, physicsClientId=client)
    return {"reach": {side: round(value, 4) for side, value in best.items()},
            "centre": centre}



def measure_arm(robot: str) -> dict:
    """两肩中点 + 单臂臂长（上臂 + 前臂），**半径用 FK 采样量**（见 ``sample_reach``）。

    用**两肩中点**而不是某一侧：``pushcube`` 的方块在 y=-0.2，离左肩 0.4 m、离右肩 0.0 m，
    只取"第一个 shoulder_pitch"会把可达判成不可达（实测 dz 被逼到 +1.00 m，台面抬到肩高）。
    """
    cached = _ARM_CACHE.get(str(robot))
    if cached is not None:
        return dict(cached)
    from teleoperation.simulation import RobotLoader


    client = p.connect(p.DIRECT)
    try:
        audit.fresh(client, ground=False)
        body = RobotLoader(client).load_robot(robot)["robot"]
        points = audit.link_points(client, body)
        sampled = sample_reach(client, body, robot)
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
    reach_sampled = max(sampled["reach"].values()) if sampled["reach"] else 0.0
    centres = [np.asarray(point, dtype=float) for point in sampled["centre"].values()]
    if centres:
        # 采样半径是**从肩关节轴**量出来的，中心必须用同一个点（不是名字匹配到的肩 link）
        centre = np.mean(np.stack(centres), axis=0)
    result = {"robot": robot,
              "reach_m": round(reach_sampled, 4) if reach_sampled > 0.0
              else round(upper + fore, 4),
              "reach_sampled_m": sampled["reach"],
              "reach_link_sum_m": round(upper + fore, 4),
              "upper_m": round(upper, 4),
              "fore_m": round(fore, 4),
              "shoulder_m": [round(float(v), 4) for v in centre],
              "shoulders": {name: [round(float(v), 4) for v in point]
                            for name, point in shoulders.items()}}
    _ARM_CACHE[str(robot)] = dict(result)
    return result



#: ``measure_arm`` 每次要跑一遍 FK 采样（~2 s），同一个进程里同一台机器人只量一次
_ARM_CACHE: dict = {}



#: 摄像头帧率的名义值（`frame_dt` 拿不到真实间隔时才用）：媒体管道实测 19-21 fps
CAMERA_NOMINAL_FPS = 20.0
#: 一个摄像头帧对应多少个物理步（12 步 = 0.05 s 仿真，与 20 fps 的相机同节奏）
CAMERA_SUBSTEPS = 12


class CameraPolicy:
    """摄像头 -> 手 18D + pose 手臂 -> 原装手关节 + PyBulletArmIK -> 原生动作向量。

    一条正在用的链路一行都不重写，只把它们接在 ``SimulationEnv`` 上::

        CameraFrameSource.next_frame()      # MediaPipe 手 21 点 + pose 肩/肘/腕
          -> RealtimeArmBridge.update()     # 相对重定向（默认 posture 方向口径）
          -> PyBulletArmIK.solve()          # 每侧 7 个臂关节角
          -> 按 env.action_joint_names 落位 -> env.task.apply_action()

    为什么每 ``substeps`` 步才读一次摄像头：MediaPipe 一帧要 30-50 ms，而 DIRECT 模式的
    物理步是 0.1 ms 级；每步读一次摄像头会把"你手的 1 秒"变成机器人的 0.02 秒。12 步 =
    0.05 s 仿真/帧，正好与 20 fps 的相机同节奏（实测 19.2 fps）。

    手部默认接到原生手：``MediaPipePalmLocalPipeline`` 的三帧窗口 -> L21 18D ->
    ``NativeHandAdapter`` -> H1-2 / GR1-T2 / G1 原装手。``hand_enabled=False`` 保留
    历史上“只驱动手臂、手指保持 reset 角度”的 A/B 行为。
    """

    def __init__(self, env, side: str = "right", camera_index: int = 0,
                 input_npz: str = None,
                 substeps: int = CAMERA_SUBSTEPS, calibrate_frames: int = 30,
                 arm_map: str = "posture", hand_model=None, pose_model=None,
                 preview: bool = False, hand_enabled: bool = True,
                 hand_backend: str = "auto", hand_checkpoint=None,
                 hand_smoothing: float = 0.25, hand_limit_mode: str = "clamp",
                 hand_device: str = "cpu", table_clearance_m: float = 0.06,
                 table_guard: bool = False,
                 grasp_assist: bool = True,
                 arm_force: float = 200.0, hand_force: float = 200.0,
                 ik_mode: str = "pose", wrist_pose: bool = True,
                 ik_solver: str = "paper",
                 target_speed_m_s: float = 0.60,
                 startup_hold_frames: int = 12,
                 startup_motion_m: float = 0.05,
                 startup_confirm_frames: int = 3,
                 target_deadband_m: float = 0.002,
                 freeze_objects: bool = True):
        from teleoperation.simulation.arm_ik import PyBulletArmIK
        from teleoperation.simulation.arm_profiles import profile_for
        from teleoperation.simulation.arm_realtime import RealtimeArmBridge
        from teleoperation.simulation.hand_mount import find_hand_base_link
        from teleoperation.simulation.realtime_teleop import (
            CameraFrameSource,
            ReplayFrameSource,
        )
        from teleoperation.applications.realtime_hand_sim import build_retargeter
        from teleoperation.paths import realtime_checkpoint
        from teleoperation.robots.native_hand import (
            NativeHandAdapter,
            PinchSignalStabilizer,
            pinch_signal_details,
        )
        from teleoperation.simulation.robot_loader import read_joint_ranges

        self.env = env
        self.side = str(side)
        self.substeps = max(1, int(substeps))
        self.input_npz = None if input_npz is None else str(input_npz)
        self.arm_map = str(arm_map)
        self.preview = bool(preview)
        self.hand_enabled = bool(hand_enabled)
        self.hand_backend = str(hand_backend)
        self.hand_limit_mode = str(hand_limit_mode)
        self.table_clearance_m = max(0.0, float(table_clearance_m))
        self.table_guard_enabled = bool(table_guard)
        self.grasp_assist_enabled = bool(grasp_assist)
        self.grasp_assist_excluded = grasp_assist_excluded(
            str(getattr(env, "task_name", "")))
        task_name = str(getattr(env, "task_name", ""))
        self.delayed_physics_object = (
            "cup" if task_name == "uprearcup"
            else "cube" if task_name in ("pickcube", "pickplacecube")
            else "ball" if task_name == "balltrashcan"
            else None)
        self.arm_force = max(1.0, float(arm_force))
        self.hand_force = max(1.0, float(hand_force))
        if ik_mode not in ("position", "pose"):
            raise ValueError("camera ik_mode must be 'position' or 'pose'")
        self.ik_mode = str(ik_mode)
        self.wrist_pose = bool(wrist_pose)
        self.ik_solver = str(ik_solver)
        self.target_speed_m_s = max(0.0, float(target_speed_m_s))
        self.startup_hold_frames = max(0, int(startup_hold_frames))
        self.startup_motion_m = max(0.0, float(startup_motion_m))
        self.startup_confirm_frames = max(1, int(startup_confirm_frames))
        self.target_deadband_m = max(0.0, float(target_deadband_m))
        self.tracking_frames = 0
        self.tracking_started = False
        self.startup_target_position = None
        self.startup_motion_frames = 0
        self.startup_target_motion_max_m = 0.0
        self.startup_arm_drift_max_m = 0.0
        self.startup_arm_drift_samples = 0
        self.freeze_objects = bool(freeze_objects)
        self.frozen_objects = {}
        self.last_frame_dt = 1.0 / float(CAMERA_NOMINAL_FPS)
        self.last_commanded_target = None
        self.target_slew_limited = 0
        self.table_guard_count = 0
        self.last_table_guard = {}
        self._window = "task_eval camera (q/Esc quits)"
        missing_names = []
        hand_model = Path(hand_model or (ROOT / "datasets" / "hand_landmarker.task"))
        pose_model = Path(pose_model or (ROOT / "datasets" / "pose_landmarker_lite.task"))
        if self.input_npz is not None:
            self.source = ReplayFrameSource(self.input_npz, "world")
        else:
            self.source = CameraFrameSource(
                model_asset_path=hand_model, camera_index=int(camera_index),
                width=640, height=480, fps=30, landmark_space="world",
                pose_model_asset_path=pose_model if pose_model.is_file() else None)
        profile = profile_for(env.robot_type)
        self.adapter = PyBulletArmIK(env.robot_id, env.client,
                                     robot_type=env.robot_type, sides=(self.side,),
                                     rest_poses="current", ik_mode=self.ik_mode,
                                     orientation_scope="wrist",
                                     max_wrist_joint_step_deg=20.0,
                                     lock_others=True,
                                     arm_force=self.arm_force,
                                     ik_solver=self.ik_solver)
        # 方向口径要的「机器人自己的量」：连杆长 + 加载时的肩/肘/腕（基座系）
        points = self.adapter.arm_points(self.side)
        self.startup_position = np.asarray(
            self.adapter.to_base(points["wrist"]), dtype=np.float64)
        shoulder = np.asarray(points["upper"], dtype=np.float64)
        elbow = np.asarray(points["elbow"], dtype=np.float64)
        wrist = np.asarray(points["wrist"], dtype=np.float64)
        links = {env.robot_type: {self.side: {
            "upper": float(np.linalg.norm(elbow - shoulder)),
            "fore": float(np.linalg.norm(wrist - elbow))}}}
        anchors = {env.robot_type: {self.side: {
            "shoulder": list(self.adapter.to_base(shoulder)),
            "elbow": list(self.adapter.to_base(elbow)),
            "wrist": list(self.adapter.to_base(wrist))}}}
        self.arm_anchor_base = np.asarray(
            anchors[env.robot_type][self.side]["shoulder"], dtype=np.float64)
        self.bridge = RealtimeArmBridge(
            [profile], sides=(self.side,), map_mode=self.arm_map,
            robot_links=links, anchors=anchors,
            orientation=("on" if self.wrist_pose else "off"),
            orientation_mode="hand_base", hand_roll_deg=180.0,
            calibration_frames=int(calibrate_frames), calibrate_mode="hold",
            jitter_filter=True)
        self.names = {name: index for index, name in enumerate(env.action_joint_names)}
        self.arm_names = set(env.robot_loader.arm_joints.get(self.side, []))
        for name in self.adapter.joint_values(self.side):
            if name not in self.names:
                missing_names.append(name)
        self.action = joint_positions(env)
        self.last_arm_target_world = None
        self.last_arm_world = None
        self.last_arm_error_mm = None
        self.hand_names = list(env.robot_loader.hand_joints.get(self.side, []))
        self.grasp_link, self.grasp_link_name = find_hand_base_link(
            env.robot_id, env.robot_type, self.side, env.client)
        self.grasp_constraint = None
        self.grasped_object = None
        self.grasp_attach_count = 0
        self.grasp_release_count = 0
        # Confirm on camera-frame time, not one 240 Hz physics frame.  Hand
        # classification commonly flickers for a frame while the fist stays closed.
        self.grasp_attach_confirm_steps = max(12, self.substeps * 2)
        self.grasp_release_confirm_steps = max(12, self.substeps * 8)
        self.grasp_max_attach_gap_m = 0.12
        self.grasp_candidate_steps = 0
        self.grasp_candidate_key = None
        self.grasp_open_steps = 0
        self.grasp_rejected_distance = 0
        self.grasp_feedback = {
            "state": "none", "object": None, "contacts": 0,
            "normal_force_n": 0.0}
        self._last_grasp_feedback_state = None
        self.contact_grip_active = False
        self.contact_grip_object = None
        self.contact_grip_targets = {}
        self.contact_grip_links = []
        self.contact_grip_count = 0
        self.contact_grip_open_steps = 0
        self.cup_velocity_clamp_count = 0
        self.hand_adapter = None
        self.hand_retargeter = None
        self.hand_backend_name = "off"
        self.hand_backend_note = "hand mapping disabled"
        self._pinch_signal = pinch_signal_details
        self._pinch_stabilizer = PinchSignalStabilizer(hold_frames=8)
        self.hand_valid_frames = 0
        self.hand_invalid_frames = 0
        self.hand_clipped = 0
        self.last_valid_hand = {}
        self.pinch_finger_frames = {}
        self.last_pinch = None
        self.last_raw_pinch = None
        self.max_pinch_strength = 0.0
        self.max_fist_strength = 0.0
        self.hand_calibration_status = {}
        if self.hand_enabled:
            checkpoint = Path(hand_checkpoint) if hand_checkpoint else realtime_checkpoint("world")
            self.hand_retargeter, self.hand_backend_name, self.hand_backend_note = \
                build_retargeter(self.hand_backend, checkpoint, str(hand_device),
                                 int(calibrate_frames), float(hand_smoothing), "world")
            ranges, _names = read_joint_ranges(env.robot_id, env.client)
            self.hand_adapter = NativeHandAdapter(
                env.robot_type, self.side, ranges, limit_mode=self.hand_limit_mode)
        self.steps = 0
        self.frames = 0
        self.hand_frames = {"left": 0, "right": 0}
        self.bad_sides = 0
        self.last_target = None
        self.last_info = {}
        self.last_hand = {}
        self.last_frame_at = None
        self.unmapped = missing_names
        self.stopped = False
        self.calibrated = False
        self._freeze_unattended_objects(env)
        print("[camera] 机器人这一侧臂由你的右手驱动（--side 可换）："
              "手往画面右侧移 = 机器人往它自己左边移（方块在它左边，y 从 -0.20 到 +0.30）；"
              "手往镜头方向收 = 机器人往前伸。看 PyBullet 窗口跟，别盯着字", flush=True)
        if self.hand_enabled:
            backend = (self.hand_backend_name
                       if self.hand_adapter is not None else "unavailable")
            print("[camera] hand -> native joints: %s (%s), %d mapped joints, limits=%s"
                  % (backend, self.hand_backend_note, len(self.hand_names),
                     self.hand_limit_mode), flush=True)
            print("[camera] hold this side's hand OPEN and fully visible during the first "
                  "%d frames: that is the hand-retargeter zero pose" % calibrate_frames,
                  flush=True)
        else:
            print("[camera] native hand mapping OFF: hand joints keep their reset angles",
                  flush=True)

    # ------------------------------------------------------------------ #
    def _freeze_unattended_objects(self, env) -> None:
        """Keep task objects still while the camera is calibrating or unattended."""
        if not self.freeze_objects:
            return
        for key, body in env.task.objects.items():
            if key == TABLE_KEY:
                continue
            try:
                mass = float(p.getDynamicsInfo(
                    body, -1, physicsClientId=env.client)[0])
                if mass <= 0.0:
                    continue
                p.resetBaseVelocity(
                    body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    physicsClientId=env.client)
                p.changeDynamics(
                    body, -1, mass=0.0, physicsClientId=env.client)
                self.frozen_objects[key] = (body, mass)
            except p.error:
                continue

    def _unfreeze_task_objects(self, env, only=None) -> None:
        if not self.frozen_objects:
            return
        selected = set(only) if only is not None else None
        for key, (body, mass) in list(self.frozen_objects.items()):
            if selected is not None and key not in selected:
                continue
            try:
                p.changeDynamics(
                    body, -1, mass=float(mass), physicsClientId=env.client)
                p.resetBaseVelocity(
                    body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    physicsClientId=env.client)
            except p.error:
                continue
            self.frozen_objects.pop(key, None)

    # ------------------------------------------------------------------ #
    def _draw_preview(self, raw, upper, target) -> bool:
        """现场对位窗口：手点 + 肩/肘/腕 + 目标是否解出来。返回 False = 用户按了 q。"""
        import cv2

        bgr = getattr(getattr(self.source, "_input", None), "last_bgr", None)
        if bgr is None:
            return True
        image = np.array(bgr, copy=True)
        height, width = image.shape[:2]
        for side, color in (("left", (0, 255, 0)), (self.side, (255, 0, 255))):
            points = (raw or {}).get(side)
            if points is None:
                continue
            for x, y in np.asarray(points, dtype=np.float64)[:, :2]:
                cv2.circle(image, (int(x * width), int(y * height)), 3, color, -1)
        arms = getattr(upper, "arms", {}) or {}
        for side in ("left", "right"):
            points = arms.get(side)
            if points is None:
                continue
            xy = np.asarray(points, dtype=np.float64)
            for index in range(2):
                cv2.line(image,
                         (int(xy[index, 0] * width), int(xy[index, 1] * height)),
                         (int(xy[index + 1, 0] * width), int(xy[index + 1, 1] * height)),
                         (0, 255, 255), 2)
        hits = {side: int((raw or {}).get(side) is not None) for side in ("left", "right")}
        pinch = self.last_pinch or {}
        pinch_text = "-"
        if pinch.get("finger") is not None:
            pinch_text = "%s/%.2f" % (
                pinch["finger"], float(pinch.get("strength", 0.0)))
        text = "L=%d R=%d hand=%s target=%s" % (
            hits["left"], hits["right"],
            "ok" if self.last_hand.get("valid") else self.last_hand.get("reason", "-"),
            "ok" if (target is not None and target.position is not None)
            else (target.reason if target is not None else "none"))
        pinch_line = "pinch=%s (raw %s)" % (
            pinch_text,
            "-" if not self.last_raw_pinch
            else "%s/%s" % (self.last_raw_pinch.get("finger"),
                            self.last_raw_pinch.get("strength")))
        feedback = self.grasp_feedback
        grasp_text = "grasp=%s %s contacts=%d force=%.1fN" % (
            feedback.get("state", "none"),
            feedback.get("object") or "-",
            int(feedback.get("contacts", 0)),
            float(feedback.get("normal_force_n", 0.0)))
        cv2.rectangle(image, (0, 0), (width, 72), (0, 0, 0), -1)
        cv2.putText(image, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 255, 0) if (hits["left"] or hits["right"]) else (0, 0, 255), 1)
        cv2.putText(image, pinch_line, (6, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 0), 1)
        grasp_color = ((0, 255, 255)
                       if feedback.get("state") == "attached"
                       else (0, 255, 0)
                       if feedback.get("state") == "contact"
                       else (180, 180, 180))
        cv2.putText(image, grasp_text, (6, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, grasp_color, 1)
        cv2.putText(image, "stand ~1 m, palm to camera | q/Esc quits",
                    (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(self._window, image)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    def _update_hand(self, window) -> None:
        """Retarget this side's hand and write native joint targets into ``action``."""
        if not self.hand_enabled or self.hand_adapter is None:
            return
        if (not getattr(self, "calibrated", False)
                or not self.tracking_started):
            # Keep both the arm and fingers still until the post-calibration hold
            # expires. Otherwise an early noisy hand frame can hit a nearby task
            # object before tracking has actually started.
            self.last_hand = {"valid": False, "reason": "startup hold"}
            return
        if window is None:
            self.hand_invalid_frames += 1
            self.last_hand = {"valid": False, "reason": "hand window is not ready"}
            return
        result = self.hand_retargeter.retarget(window)
        angles = getattr(result, self.side + "_angles")
        valid = bool(getattr(result, self.side + "_valid"))
        status = getattr(self.hand_retargeter, "calibration_status", None)
        if isinstance(status, dict):
            self.hand_calibration_status = {
                str(key): bool(value) for key, value in status.items()}
        if not valid or angles is None:
            self.hand_invalid_frames += 1
            self.last_hand = {"valid": False, "reason": "hand landmarks unusable"}
            return
        raw_frame = (getattr(getattr(self, "source", None), "last_raw_frame", None)
                     or {})
        raw_hand = raw_frame.get(f"world_{self.side}")
        image_hand = raw_frame.get(self.side)
        raw_details = self._pinch_signal(raw_hand, image_hand)
        raw_pinch = raw_details["selected"]
        pinch = self._pinch_stabilizer.update(raw_pinch)
        self.last_raw_pinch = {
            "finger": raw_pinch[0],
            "strength": round(float(raw_pinch[1]), 3),
            "ratio": (None if raw_pinch[2] is None
                      else round(float(raw_pinch[2]), 3)),
            "source": raw_details["source"],
            "world": raw_details["world"],
            "image": raw_details["image"],
        }
        targets, clipped = self.hand_adapter.map(
            angles.native_mapping_dofs(), pinch=pinch)
        action = np.array(self.action, dtype=np.float64, copy=True)
        written = {}
        for name, value in targets.items():
            slot = self.names.get(name)
            if slot is None:
                continue
            action[slot] = float(value)
            written[name] = float(value)
        if not written:
            self.hand_invalid_frames += 1
            self.last_hand = {
                "valid": False,
                "reason": "native hand mapping produced no joints in this action vector"}
            return
        self.action = action
        self.hand_valid_frames += 1
        self.hand_clipped += int(clipped)
        self.last_hand = {
            "valid": True,
            "backend": self.hand_backend_name,
            "limit_mode": self.hand_limit_mode,
            "mapped_joints": sorted(written),
            "clipped": int(clipped),
            "pinch_finger": pinch[0],
            "pinch_strength": round(float(pinch[1]), 3),
            "pinch_ratio": (None if pinch[2] is None
                            else round(float(pinch[2]), 3)),
            "calibrated": bool(getattr(
                self, "hand_calibration_status", {}).get(self.side, True)),
            "l21_angles": [round(float(value), 5)
                           for value in np.asarray(angles.values, dtype=np.float64)],
        }
        self.last_valid_hand = dict(self.last_hand)
        finger, strength, ratio = pinch
        if finger is not None:
            self.pinch_finger_frames[finger] = (
                self.pinch_finger_frames.get(finger, 0) + 1)
            self.max_pinch_strength = max(
                self.max_pinch_strength, float(strength))
            if finger == "fist":
                self.max_fist_strength = max(
                    self.max_fist_strength, float(strength))
        self.last_pinch = {
            "finger": finger,
            "strength": round(float(strength), 3),
            "ratio": None if ratio is None else round(float(ratio), 3),
        }

    def _guard_target_above_table(self, position):
        """Reject wrist commands below the tabletop while over the table footprint."""
        if not self.table_guard_enabled:
            self.last_table_guard = {}
            return position, False
        table = self.env.task.objects.get(TABLE_KEY)
        if table is None or position is None:
            self.last_table_guard = {}
            return position, False
        world = np.asarray(self.adapter.target_in_world(position, "base"),
                           dtype=np.float64)
        low, high = aabb(self.env.client, table)
        guarded, active = guard_table_target(
            world, (low, high), self.table_clearance_m, margin_m=0.05)
        if not active:
            self.last_table_guard = {}
            return position, False
        self.table_guard_count += 1
        self.last_table_guard = {
            "original_world": [round(float(value), 4) for value in world],
            "guarded_world": [round(float(value), 4) for value in guarded],
            "table_top_m": round(float(high[2]), 4),
            "clearance_m": self.table_clearance_m,
        }
        return tuple(float(value) for value in self.adapter.to_base(guarded)), True

    def _slew_target(self, position, start_position=None):
        """Limit one camera-frame wrist move to the configured Cartesian speed."""
        wanted = np.asarray(position, dtype=np.float64)
        if self.last_commanded_target is None:
            # The bridge has no command history before the first tracking frame.
            # Start from the robot's measured wrist, never from an arbitrary first
            # operator target: that first-frame jump is what swept objects off the
            # table in the live cup trial.
            self.last_commanded_target = np.asarray(
                wanted if start_position is None else start_position,
                dtype=np.float64).copy()
            return tuple(float(value) for value in self.last_commanded_target), False
        if self.target_speed_m_s <= 0.0:
            self.last_commanded_target = wanted.copy()
            return tuple(float(value) for value in wanted), False
        previous = np.asarray(self.last_commanded_target, dtype=np.float64)
        offset = wanted - previous
        distance = float(np.linalg.norm(offset))
        if distance <= self.target_deadband_m:
            return tuple(float(value) for value in previous), False
        limit = min(0.04, self.target_speed_m_s * self.last_frame_dt)
        changed = distance > limit and limit > 0.0
        if changed:
            wanted = previous + offset * (limit / distance)
            self.target_slew_limited += 1
        self.last_commanded_target = wanted.copy()
        return tuple(float(value) for value in wanted), changed

    def _update_startup_gate(self, target_position):
        """Return whether intentional post-calibration motion has been confirmed."""
        target_position = np.asarray(target_position, dtype=np.float64)
        if self.startup_target_position is None:
            # The robot's load-time wrist and the bridge's calibrated neutral
            # are not guaranteed to coincide.  Use the first post-calibration
            # target as the zero the operator must intentionally leave.
            self.startup_target_position = target_position.copy()
        displacement = float(np.linalg.norm(
            target_position - self.startup_target_position))
        if not self.tracking_started:
            self.startup_target_motion_max_m = max(
                self.startup_target_motion_max_m, displacement)
        if self.tracking_started:
            return True, displacement
        if self.tracking_frames <= self.startup_hold_frames:
            self.startup_motion_frames = 0
        elif displacement >= self.startup_motion_m:
            self.startup_motion_frames += 1
        else:
            self.startup_motion_frames = 0
        if (self.tracking_frames > self.startup_hold_frames
                and self.startup_motion_frames >= self.startup_confirm_frames):
            self.tracking_started = True
        return self.tracking_started, displacement

    def _read_frame(self) -> None:
        """读一帧摄像头 -> 重定向 -> IK -> 写进 ``self.action``。"""
        actual_before = None
        try:
            self.adapter.record_posture(self.side)
            actual_before = np.asarray(
                self.adapter.measure(self.side)[0], dtype=np.float64)
        except Exception:  # noqa: BLE001 - disconnect / first frame
            pass
        window = self.source.next_frame()
        if getattr(self.source, "exhausted", False):
            self.stopped = True
        self.frames += 1
        upper = self.source.arms()
        raw = self.source.last_raw_frame or {}
        self._update_hand(window)
        for side in ("left", "right"):
            self.hand_frames[side] += int(raw.get(side) is not None)
        now = time.time()
        delta = None if self.last_frame_at is None else max(now - self.last_frame_at, 1e-3)
        self.last_frame_at = now
        if delta is not None:
            self.last_frame_dt = delta
        targets = self.bridge.update(arms_world=upper, hands=raw, frame_dt=delta)
        target = targets[self.env.robot_type][self.side]
        self.last_target = target.describe()
        self._note_progress(raw, target)
        if self.preview and not self._draw_preview(raw, upper, target):
            self.stopped = True
        if target.position is None:
            self.bad_sides += 1
            return
        if self.calibrated:
            self.tracking_frames += 1
            was_tracking = self.tracking_started
            gate_open, displacement = self._update_startup_gate(target.position)
            if not gate_open:
                if actual_before is not None:
                    actual_base = np.asarray(
                        self.adapter.to_base(actual_before), dtype=np.float64)
                    self.startup_arm_drift_max_m = max(
                        self.startup_arm_drift_max_m,
                        float(np.linalg.norm(actual_base - self.startup_position)))
                    self.startup_arm_drift_samples += 1
                self.last_info = {
                    "valid": True,
                    "held": True,
                    "reason": "post-calibration startup hold",
                    "startup_motion_mm": round(displacement * 1000.0, 2),
                    "startup_motion_threshold_mm": round(
                        self.startup_motion_m * 1000.0, 2),
                    "startup_motion_confirm_frames": self.startup_motion_frames,
                    "startup_motion_confirm_required": self.startup_confirm_frames,
                    "startup_target_motion_max_mm": round(
                        self.startup_target_motion_max_m * 1000.0, 2),
                    "startup_arm_drift_max_mm": round(
                        self.startup_arm_drift_max_m * 1000.0, 2),
                    "startup_target_position": [
                        round(float(value), 4)
                        for value in self.startup_target_position],
                    "startup_pose": "strong motor hold",
                }
                return
            if not was_tracking:
                self._unfreeze_task_objects(
                    self.env, only=((self.delayed_physics_object,)
                                    if self.delayed_physics_object else None))
        start_base = (None if actual_before is None
                      else self.adapter.to_base(actual_before))
        wanted, target_slew_limited = self._slew_target(
            target.position, start_position=start_base)
        wanted, table_guarded = self._guard_target_above_table(wanted)
        if table_guarded:
            self.last_commanded_target = np.asarray(wanted, dtype=np.float64)
        current_shoulder = np.asarray(
            self.adapter.to_base(self.adapter.arm_points(self.side)["upper"]),
            dtype=np.float64)
        shoulder_shift = current_shoulder - self.arm_anchor_base
        wanted = tuple(float(value + shift) for value, shift in zip(wanted, shoulder_shift))
        elbow = (None if target.elbow is None else
                 tuple(float(value + shift) for value, shift in zip(target.elbow, shoulder_shift)))
        target_world = np.asarray(
            self.adapter.target_in_world(wanted, "base"), dtype=np.float64)
        orientation = None
        orientation_frame = "absolute"
        hand_frame = self.bridge.hand_frame_target()[self.side]
        if self.wrist_pose and hand_frame is not None:
            orientation = self.adapter.hand_target_orientation(
                self.side, hand_frame, source="base")
        elif self.wrist_pose:
            orientation = self.bridge.orientation_target()[self.side]
            orientation_frame = "neutral"
        bend = self.bridge.elbow_bend_target()[self.side]
        posture = (None if bend is None
                   else self.adapter.posture_from_bend(self.side, bend))
        values = self.adapter.solve(
            self.side, wanted, source="base", elbow=elbow,
            orientation=orientation, orientation_frame=orientation_frame,
            posture=posture, elbow_pass="last")
        if orientation is not None:
            values = self.adapter.refine_orientation(
                self.side, wanted, orientation, source="base",
                orientation_frame=orientation_frame, arm_values=values)
            if hand_frame is not None:
                self.adapter.last_hand_frame_base[self.side] = np.asarray(
                    hand_frame, dtype=np.float64).copy()
        solution_residual_mm = float(self.adapter.residual_mm(self.side))
        if actual_before is not None and self.last_arm_target_world is not None:
            self.last_arm_error_mm = round(1000.0 * float(np.linalg.norm(
                actual_before - np.asarray(self.last_arm_target_world))), 2)
            if self.frames % 20 == 0:
                print("[camera] wrist target=%s actual=%s lag=%.1f mm"
                      % ([round(float(v), 3) for v in self.last_arm_target_world],
                         [round(float(v), 3) for v in actual_before],
                         self.last_arm_error_mm), flush=True)
        self.last_arm_world = actual_before
        self.last_arm_target_world = target_world
        action = np.array(self.action, dtype=np.float64, copy=True)
        for name, value in values.items():
            slot = self.names.get(name)
            if slot is not None and slot < action.size:
                action[slot] = float(value)
        self.action = action
        self.last_info = {
            "valid": bool(target.valid), "held": bool(target.held),
            "reason": target.reason, "source": target.source,
            "left_ratio": round(self.hand_frames["left"] / max(self.frames, 1), 3),
            "right_ratio": round(self.hand_frames["right"] / max(self.frames, 1), 3),
            "hand_valid": bool(self.last_hand.get("valid")),
            "table_guard": table_guarded,
            "table_guard_count": self.table_guard_count,
            "target_slew_limited": target_slew_limited,
            "target_slew_limited_frames": self.target_slew_limited,
            "wrist_pose": self.wrist_pose,
            "hand_frame_source": ("hand_base" if hand_frame is not None
                                  else "wrist_relative"),
            "hand_frame_error_deg": round(
                float(self.adapter.hand_frame_error_deg(self.side)), 2),
            "shoulder_shift_mm": round(
                1000.0 * float(np.linalg.norm(shoulder_shift)), 2),
            "solution_residual_mm": round(solution_residual_mm, 2),
            "unmapped_joints": list(self.unmapped),
        }

    def _note_progress(self, raw, target) -> None:
        """现场提示：第一帧说清怎么站，标定中报进度，标定好之后只在掉线时提示。"""
        if self.frames == 1:
            print("[camera] 站到摄像头前约 1.0-1.5 m，上半身（肩到指尖）完整入镜；"
                  "手举到胸前别乱动，等下面出现 cal=ok 再开始动", flush=True)
        status = self.bridge.status()
        if status != "cal=ok":
            if self.frames % 5 == 0:
                print("[camera] %s  %d frames  hands L=%d R=%d" % (
                    status, self.frames,
                    int(raw.get("left") is not None),
                    int(raw.get("right") is not None)), flush=True)
            return
        if not self.calibrated:
            self.calibrated = True
            print("[camera] cal=ok（%d 帧标定完成）-> 机器人这一侧臂开始跟手" % self.frames,
                  flush=True)
        if target.position is None and (self.frames % 10) == 0:
            print("[camera] 这一帧没跟手：%s（%d 帧里 %d 次）" % (
                target.reason, self.frames, self.bad_sides), flush=True)

    def apply_action(self, env) -> None:
        """Apply arm and hand targets with separate motor-force budgets."""
        if self.tracking_started:
            self._update_contact_grip(env)
        for slot, (name, joint) in enumerate(zip(env.action_joint_names,
                                                 env.action_joint_indices)):
            if slot >= len(self.action):
                break
            # Position servos at the normal 200 N budget sag under gravity and
            # contact before the operator has intentionally started.  Lock the
            # unchanged startup pose firmly until the action gate opens.
            if not self.tracking_started:
                force = 2000.0
            else:
                force = self.arm_force if name in self.arm_names else self.hand_force
            p.setJointMotorControl2(
                env.robot_id, int(joint), p.POSITION_CONTROL,
                targetPosition=float(self.action[slot]), force=force,
                physicsClientId=env.client)
        if self.tracking_started:
            self._update_grasp_assist(env)

    def _update_contact_grip(self, env) -> None:
        """Stop finger closure after two distinct hand links touch an object.

        This is deliberately not a hand-object constraint: the object remains
        governed entirely by contact and friction.  It only emulates a compliant
        hand that stops closing when the surface is reached instead of driving
        every finger to its kinematic endpoint.
        """
        last_pinch = getattr(self, "last_pinch", None) or {}
        finger = last_pinch.get("finger")
        strength = float(last_pinch.get("strength") or 0.0)
        pinching = finger is not None and strength >= 0.25

        if getattr(self, "contact_grip_active", False):
            if pinching:
                self.contact_grip_open_steps = 0
            else:
                self.contact_grip_open_steps += 1
                release_steps = getattr(self, "grasp_release_confirm_steps", 96)
                if self.contact_grip_open_steps >= release_steps:
                    self.contact_grip_active = False
                    self.contact_grip_object = None
                    self.contact_grip_targets = {}
                    self.contact_grip_links = []
                    self.contact_grip_open_steps = 0
            if self.contact_grip_active:
                action = np.array(self.action, dtype=np.float64, copy=True)
                for name, value in self.contact_grip_targets.items():
                    slot = self.names.get(name)
                    if slot is not None and slot < action.size:
                        action[slot] = float(value)
                self.action = action
            return

        if not pinching or getattr(self, "grasp_constraint", None) is not None:
            return
        key = (getattr(self, "grasp_feedback", None) or {}).get("object")
        if not key or key == TABLE_KEY:
            return
        body = env.task.objects.get(key)
        if body is None:
            return
        try:
            contacts = p.getContactPoints(
                bodyA=env.robot_id, bodyB=body,
                physicsClientId=env.client)
        except p.error:
            return
        links = {int(contact[3]) for contact in contacts}
        names = []
        for link in links:
            info = p.getJointInfo(
                env.robot_id, link, physicsClientId=env.client)
            raw = info[12]
            names.append(raw.decode("utf-8", "ignore")
                         if isinstance(raw, bytes) else str(raw))
        lowered = [name.lower() for name in names]
        has_thumb = any("thumb" in name for name in lowered)
        has_finger = any(any(part in name for part in
                             ("index", "middle", "ring", "pinky"))
                         for name in lowered)
        if len(links) < 2 or not (has_thumb and has_finger):
            return

        targets = {}
        for name in self.hand_names:
            slot = self.names.get(name)
            if slot is None or slot >= len(env.action_joint_indices):
                continue
            joint = int(env.action_joint_indices[slot])
            targets[name] = float(p.getJointState(
                env.robot_id, joint, physicsClientId=env.client)[0])
        if not targets:
            return
        self.contact_grip_active = True
        self.contact_grip_object = str(key)
        self.contact_grip_targets = targets
        self.contact_grip_links = list(names)
        self.contact_grip_count += 1
        self.contact_grip_open_steps = 0
        print("[camera] contact grip stop: %s on %d hand links"
              % (key, len(links)), flush=True)
        action = np.array(self.action, dtype=np.float64, copy=True)
        for name, value in self.contact_grip_targets.items():
            slot = self.names.get(name)
            if slot is not None and slot < action.size:
                action[slot] = float(value)
        self.action = action

    def _update_grasp_assist(self, env) -> None:
        """Latch a physically contacted object only while the operator is closing."""
        self._update_contact_feedback(env)
        if not self.grasp_assist_enabled or self.grasp_link is None:
            return
        strength = float((self.last_pinch or {}).get("strength") or 0.0)
        if self.grasp_constraint is not None:
            hand_valid = bool((self.last_hand or {}).get("valid"))
            finger = (self.last_pinch or {}).get("finger")
            open_hand = (hand_valid and finger is None and strength < 0.10)
            if open_hand:
                self.grasp_open_steps += 1
                if self.grasp_open_steps >= self.grasp_release_confirm_steps:
                    p.removeConstraint(self.grasp_constraint,
                                       physicsClientId=env.client)
                    self.grasp_constraint = None
                    self.grasped_object = None
                    self.grasp_open_steps = 0
                    self.grasp_release_count += 1
                    print("[camera] grasp released: sustained open hand detected",
                          flush=True)
            else:
                self.grasp_open_steps = 0
            return
        if strength < 0.35:
            self.grasp_candidate_steps = 0
            self.grasp_candidate_key = None
            return
        candidates = []
        for key, body in env.task.objects.items():
            if key == TABLE_KEY or key in self.grasp_assist_excluded:
                continue
            try:
                if float(p.getDynamicsInfo(
                        body, -1, physicsClientId=env.client)[0]) <= 0.0:
                    continue
                contacts = p.getContactPoints(
                    bodyA=env.robot_id, bodyB=body,
                    physicsClientId=env.client)
            except p.error:
                continue
            if len(contacts) < 2:
                continue
            position = np.asarray(p.getBasePositionAndOrientation(
                body, physicsClientId=env.client)[0], dtype=np.float64)
            state = p.getLinkState(
                env.robot_id, self.grasp_link, computeForwardKinematics=True,
                physicsClientId=env.client)
            distance = float(np.linalg.norm(
                np.asarray(state[4], dtype=np.float64) - position))
            if distance > self.grasp_max_attach_gap_m:
                self.grasp_rejected_distance += 1
                continue
            candidates.append((distance, key, body, state))
        if not candidates:
            self.grasp_candidate_steps = 0
            self.grasp_candidate_key = None
            return
        _distance, key, body, hand_state = min(candidates, key=lambda item: item[0])
        if key == self.grasp_candidate_key:
            self.grasp_candidate_steps += 1
        else:
            self.grasp_candidate_key = key
            self.grasp_candidate_steps = 1
        if self.grasp_candidate_steps < self.grasp_attach_confirm_steps:
            return
        obj_pos, obj_orn = p.getBasePositionAndOrientation(
            body, physicsClientId=env.client)
        inverse_pos, inverse_orn = p.invertTransform(
            hand_state[4], hand_state[5], physicsClientId=env.client)
        rel_pos, rel_orn = p.multiplyTransforms(
            inverse_pos, inverse_orn, obj_pos, obj_orn, physicsClientId=env.client)
        self.grasp_constraint = p.createConstraint(
            env.robot_id, self.grasp_link, body, -1, p.JOINT_FIXED,
            [0, 0, 0], parentFramePosition=[0, 0, 0],
            childFramePosition=list(rel_pos),
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=list(rel_orn),
            physicsClientId=env.client)
        self.grasped_object = key
        self.grasp_attach_count += 1
        self.grasp_candidate_steps = 0
        self.grasp_candidate_key = None
        self.grasp_open_steps = 0
        print("[camera] grasp attached: %s (hand gap %.0f mm)"
              % (key, _distance * 1000.0), flush=True)

    def _update_contact_feedback(self, env) -> None:
        """Expose real hand-object contacts in both the preview and the terminal."""
        if not getattr(self, "tracking_started", False) or not p.isConnected(env.client):
            return
        best = None
        for key, body in env.task.objects.items():
            if key == TABLE_KEY:
                continue
            try:
                contacts = p.getContactPoints(
                    bodyA=env.robot_id, bodyB=body,
                    physicsClientId=env.client)
            except p.error:
                continue
            if not contacts:
                continue
            force = sum(float(contact[9]) for contact in contacts)
            score = (len(contacts), force)
            if best is None or score > best[0]:
                best = (score, key, len(contacts), force)
        state = "attached" if self.grasp_constraint is not None else (
            "contact" if best is not None else "none")
        if best is None and self.grasped_object is not None:
            key, count, force = self.grasped_object, 0, 0.0
        elif best is None:
            key, count, force = None, 0, 0.0
        else:
            _score, key, count, force = best
        feedback = {
            "state": state,
            "object": key,
            "contacts": int(count),
            "normal_force_n": round(float(force), 2),
        }
        self.grasp_feedback = feedback
        transition = (state, key)
        if transition != self._last_grasp_feedback_state:
            self._last_grasp_feedback_state = transition
            print("[camera] grasp feedback: %s object=%s contacts=%d force=%.1f N"
                  % (state, key, int(count), float(force)), flush=True)

    def __call__(self, step: int):
        """``run_trial`` 每步调一次；每 ``substeps`` 步读一帧摄像头。"""
        self.steps = int(step) + 1
        if (int(step) % self.substeps) == 0:
            self._read_frame()
        return self.action

    def post_step(self, env) -> None:
        """Damp only high-speed table contacts; never constrain a grasped cup."""
        if not self.tracking_started:
            return
        key = self.delayed_physics_object
        if key and key in self.frozen_objects:
            body = env.task.objects.get(key)
            if body is None:
                return
            strength = float((self.last_pinch or {}).get("strength") or 0.0)
            contacts = p.getContactPoints(
                bodyA=env.robot_id, bodyB=body,
                physicsClientId=env.client)
            if strength >= 0.35 and len(contacts) >= 2:
                self._unfreeze_task_objects(env, only=(key,))
                print("[camera] %s released to physics after confirmed hand contact"
                      % key, flush=True)
            return
        if str(getattr(env, "task_name", "")) != "uprearcup":
            return
        cup = env.task.objects.get("cup")
        if cup is None:
            return
        if (self.grasp_constraint is None and clamp_horizontal_velocity(
                env.client, cup, max_linear_m_s=0.30,
                max_angular_rad_s=1.5)):
            self.cup_velocity_clamp_count += 1

    def meta(self) -> dict:
        return {"camera": {"side": self.side, "substeps": self.substeps,
                           "input_npz": self.input_npz,
                           "arm_map": self.arm_map,
                           "calibration_frames": self.bridge.calibration_frames,
                           "hand_enabled": self.hand_enabled,
                           "hand_backend": self.hand_backend_name,
                           "hand_backend_note": self.hand_backend_note,
                           "hand_limit_mode": self.hand_limit_mode,
                           "hand_joints": self.hand_names,
                           "table_clearance_m": self.table_clearance_m,
                           "table_guard": self.table_guard_enabled,
                           "grasp_assist": self.grasp_assist_enabled,
                           "grasp_assist_excluded": sorted(
                               self.grasp_assist_excluded),
                           "arm_force": self.arm_force,
                           "hand_force": self.hand_force,
                           "ik_mode": self.ik_mode,
                           "ik_solver": self.ik_solver,
                           "wrist_pose": self.wrist_pose,
                           "target_speed_m_s": self.target_speed_m_s,
                           "startup_hold_frames": self.startup_hold_frames,
                           "startup_motion_m": self.startup_motion_m,
                           "startup_confirm_frames": self.startup_confirm_frames,
                           "target_deadband_m": self.target_deadband_m,
                           "freeze_objects": self.freeze_objects,
                           "unmapped_joints": list(self.unmapped)}}

    def report(self) -> dict:
        """trial 结束时落盘的摄像头侧证据（判据之外，用来解释失败）。"""
        connected = bool(p.isConnected(self.env.client))
        actual_hand = {}
        for name in self.hand_names:
            index = self.names.get(name)
            if index is None:
                continue
            try:
                actual_hand[name] = round(float(p.getJointState(
                    self.env.robot_id, self.env.action_joint_indices[index],
                    physicsClientId=self.env.client)[0]), 5)
            except Exception:
                break
        try:
            arm_posture = (self.adapter.posture_summary(self.side)
                           if connected else None)
        except p.error:
            arm_posture = None
        return {"frames": self.frames, "steps": self.steps,
                "hand_frames": dict(self.hand_frames),
                "hand_valid_frames": self.hand_valid_frames,
                "hand_invalid_frames": self.hand_invalid_frames,
                "hand_clipped_joints": self.hand_clipped,
                "hand_calibration_status": dict(self.hand_calibration_status),
                "last_hand": self.last_hand,
                "last_valid_hand": self.last_valid_hand,
                "pinch_finger_frames": dict(self.pinch_finger_frames),
                "last_pinch": self.last_pinch,
                "last_raw_pinch": self.last_raw_pinch,
                "max_pinch_strength": round(self.max_pinch_strength, 3),
                "max_fist_strength": round(self.max_fist_strength, 3),
                "actual_hand_joints": actual_hand,
                "table_guard_count": self.table_guard_count,
                "table_guard_enabled": self.table_guard_enabled,
                "grasp_assist": self.grasp_assist_enabled,
                "grasp_assist_excluded": sorted(self.grasp_assist_excluded),
                "grasp_attach_count": self.grasp_attach_count,
                "grasp_release_count": self.grasp_release_count,
                "grasp_attach_confirm_steps": self.grasp_attach_confirm_steps,
                "grasp_release_confirm_steps": self.grasp_release_confirm_steps,
                "grasp_max_attach_gap_m": self.grasp_max_attach_gap_m,
                "grasp_rejected_distance": self.grasp_rejected_distance,
                "grasp_feedback": dict(self.grasp_feedback),
                "contact_grip_active": self.contact_grip_active,
                "contact_grip_object": self.contact_grip_object,
                "contact_grip_count": self.contact_grip_count,
                "contact_grip_targets": dict(self.contact_grip_targets),
                "contact_grip_links": list(
                    getattr(self, "contact_grip_links", [])),
                "cup_velocity_clamp_count": self.cup_velocity_clamp_count,
                "grasped_object": self.grasped_object,
                "last_table_guard": self.last_table_guard,
                "arm_force": self.arm_force,
                "hand_force": self.hand_force,
                "ik_mode": self.ik_mode,
                "ik_solver": self.ik_solver,
                "wrist_pose": self.wrist_pose,
                "target_speed_m_s": self.target_speed_m_s,
                "startup_hold_frames": self.startup_hold_frames,
                "startup_motion_m": self.startup_motion_m,
                "startup_confirm_frames": self.startup_confirm_frames,
                "target_deadband_m": self.target_deadband_m,
                "tracking_frames": self.tracking_frames,
                "tracking_started": self.tracking_started,
                "startup_motion_frames": self.startup_motion_frames,
                "startup_target_motion_max_mm": round(
                    self.startup_target_motion_max_m * 1000.0, 2),
                "startup_arm_drift_max_mm": round(
                    self.startup_arm_drift_max_m * 1000.0, 2),
                "startup_arm_drift_samples": self.startup_arm_drift_samples,
                "startup_target_position": (
                    None if self.startup_target_position is None else
                    [round(float(value), 4)
                     for value in self.startup_target_position]),
                "freeze_objects": self.freeze_objects,
                "frozen_objects_remaining": sorted(self.frozen_objects),
                "cup_waits_for_contact": (
                    self.delayed_physics_object is not None),
                "delayed_physics_object": self.delayed_physics_object,
                "target_slew_limited_frames": self.target_slew_limited,
                "last_arm_target_world": (None if self.last_arm_target_world is None else
                                          [round(float(v), 4)
                                           for v in self.last_arm_target_world]),
                "last_arm_world": (None if self.last_arm_world is None else
                                   [round(float(v), 4) for v in self.last_arm_world]),
                "last_arm_error_mm": self.last_arm_error_mm,
                "hand_frame_source": self.last_info.get("hand_frame_source"),
                "hand_frame_error_deg": self.last_info.get("hand_frame_error_deg"),
                "shoulder_shift_mm": self.last_info.get("shoulder_shift_mm"),
                "solution_residual_mm": self.last_info.get(
                    "solution_residual_mm"),
                "solution_cost": dict(self.adapter.last_solution_cost.get(
                    self.side) or {}),
                "arm_posture": arm_posture,
                "held_frames": self.bad_sides,
                "left_ratio": round(self.hand_frames["left"] / max(self.frames, 1), 3),
                "right_ratio": round(self.hand_frames["right"] / max(self.frames, 1), 3),
                "bridge_counts": {key: (dict(value) if isinstance(value, dict) else value)
                                  for key, value in self.bridge.counts.items()},
                "last_target": self.last_target, "last_info": self.last_info}

    def close(self) -> None:
        try:
            if self.grasp_constraint is not None:
                p.removeConstraint(self.grasp_constraint,
                                   physicsClientId=self.env.client)
                self.grasp_constraint = None
            self._unfreeze_task_objects(self.env)
            if self.preview:
                import cv2

                cv2.destroyAllWindows()
                cv2.waitKey(1)
            self.source.release(close_landmarker=False)
        except Exception:  # noqa: BLE001 - 关摄像头失败不该盖住 trial 结果
            pass


def build_actions(policy: str, env, path: str = None, steps: int = None,
                  **options):
    """按 policy 造动作流；``None`` = 每步不下发动作（idle）。

    ``camera`` 返回的不是数组而是 ``CameraPolicy``（可调用对象）：它每读一帧摄像头就
    解一次 IK，动作在本步才算得出来，见 ``run_trial`` 对 callable 的支持。
    """
    if policy == "idle":
        return None, {"policy": "idle"}
    if policy == "camera":
        live = CameraPolicy(env, **options)
        return live, dict({"policy": "camera", "source": "mediapipe_pose"}, **live.meta())
    if policy == "scripted":
        from task_scripted import ScriptedPolicy
        live = ScriptedPolicy(env, task_name=env.task_name, **options)
        return live, dict({"policy": "scripted", "source": "waypoint+ik"}, **live.meta())

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
    try:
        return np.array([float(p.getJointState(env.robot_id, int(index),
                                               physicsClientId=env.client)[0])
                         for index in indices])
    except p.error:
        return np.zeros(0)


def run_trial(env, actions, max_steps: int, max_wall_s: float = None,
              success_hold_steps: int = SUCCESS_HOLD_STEPS,
              stop_on_success: bool = True) -> dict:
    """逐步推进并每帧查判据；**自己算仿真时间**（baseline 用 time.time()，空转也要等 300 s）。

    ``actions`` 可以是动作数组，也可以是 ``callable(step) -> action``：摄像头策略的动作
    只能边走边算（``CameraPolicy``），所以这里不要求整段先算出来。
    """
    progress = max(1, int(max_steps) // 10)
    success = False
    completion = None
    step = 0
    wall_hit = False
    success_streak = 0
    success_streak_max = 0
    success_rejected_reason = ""
    before = joint_positions(env)
    t0 = time.time()
    for step in range(int(max_steps)):
        if callable(actions):
            try:
                action = actions(step)
            except p.error as error:
                if "Not connected" not in str(error):
                    raise
                wall_hit = True
                break
            if getattr(actions, "stopped", False):
                wall_hit = True
                break
        elif actions is not None:
            action = actions[min(step, len(actions) - 1)]
        else:
            action = None
        if action is not None:
            if hasattr(actions, "apply_action"):
                actions.apply_action(env)
            else:
                env.task.apply_action(action, joint_indices=env.action_joint_indices)
        p.stepSimulation(physicsClientId=env.client)
        if hasattr(actions, "post_step"):
            actions.post_step(env)
        criterion = bool(env.task.check_success())
        supported, support_reason = task_objects_supported(env)
        if criterion and supported:
            success_streak += 1
            success_streak_max = max(success_streak_max, success_streak)
            if success_streak >= max(1, int(success_hold_steps)):
                success = True
                completion = (step + 1) * DT
                if stop_on_success:
                    break
                if success_streak == max(1, int(success_hold_steps)):
                    print("[DEMO] success reached at %.2f s; continuing until manual close"
                          % completion, flush=True)
        else:
            if criterion and support_reason:
                success_rejected_reason = support_reason
            success_streak = 0
        if max_wall_s is not None and (step % 24) == 23 and time.time() - t0 > max_wall_s:
            wall_hit = True
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
            "wall_s": round(time.time() - t0, 2),
            "wall_limit_hit": bool(wall_hit),
            "success_hold_steps": max(1, int(success_hold_steps)),
            "success_streak_max_steps": int(success_streak_max),
            "success_rejected_reason": success_rejected_reason or None}


def object_positions(env) -> dict:
    out = {}
    for key, body in env.task.objects.items():
        try:
            out[key] = [round(float(v), 4) for v in body_pose(env.client, body)[0]]
        except p.error:
            continue
    return out


def task_objects_supported(env) -> tuple:
    """Return ``(ok, reason)`` for dynamic objects still on the task bench."""
    table = env.task.objects.get(TABLE_KEY)
    if table is None:
        return True, ""
    try:
        _low, high = aabb(env.client, table)
    except p.error:
        return False, "table is not readable"
    floor = float(high[2]) - SUCCESS_TABLE_MARGIN_M
    for key, body in env.task.objects.items():
        if key == TABLE_KEY:
            continue
        try:
            if float(p.getDynamicsInfo(
                    body, -1, physicsClientId=env.client)[0]) <= 0.0:
                continue
            position = np.asarray(p.getBasePositionAndOrientation(
                body, physicsClientId=env.client)[0], dtype=np.float64)
        except p.error:
            return False, f"{key} is not readable"
        if float(position[2]) < floor:
            return False, f"{key} fell below the tabletop"
    return True, ""


def annotate_task_objects(env) -> list:
    """Draw object names and the cube -> target direction in the GUI."""
    if not getattr(env, "_render", False):
        return []
    ids = []
    for key, body in sorted(env.task.objects.items()):
        if key == TABLE_KEY:
            continue
        try:
            position = np.asarray(p.getBasePositionAndOrientation(
                body, physicsClientId=env.client)[0], dtype=np.float64)
            _low, high = aabb(env.client, body)
        except p.error:
            continue
        ids.append(p.addUserDebugText(
            key, [float(position[0]), float(position[1]), float(high[2] + 0.05)],
            textColorRGB=[1.0, 0.85, 0.2], textSize=1.1, lifeTime=0,
            physicsClientId=env.client))
    cube = env.task.objects.get("cube")
    target = env.task.objects.get("target")
    if cube is not None and target is not None:
        cube_pos = np.asarray(p.getBasePositionAndOrientation(
            cube, physicsClientId=env.client)[0], dtype=np.float64)
        target_pos = np.asarray(p.getBasePositionAndOrientation(
            target, physicsClientId=env.client)[0], dtype=np.float64)
        cube_pos[2] += 0.06
        target_pos[2] += 0.06
        ids.append(p.addUserDebugLine(
            cube_pos.tolist(), target_pos.tolist(),
            lineColorRGB=[0.1, 1.0, 0.25], lineWidth=4.0, lifeTime=0,
            physicsClientId=env.client))
    return ids


def robot_contact_summary(env) -> dict:
    """Robot link <-> task-object contacts at the end of a trial.

    The table contact is the diagnostic for "the arm is physically trapped":
    ``joint_travel_rad`` can be large while the wrist still cannot enter the
    requested point because a link is inside the table collider.
    """
    out = {}
    for key, body in env.task.objects.items():
        try:
            contacts = p.getContactPoints(
                bodyA=env.robot_id, bodyB=body, physicsClientId=env.client)
        except Exception:  # noqa: BLE001 - diagnostic only
            continue
        if not contacts:
            continue
        rows = []
        for contact in contacts:
            link = int(contact[3])
            try:
                info = p.getJointInfo(env.robot_id, link,
                                      physicsClientId=env.client)
                name = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
            except Exception:  # noqa: BLE001 - base link / disconnected
                name = "base" if link < 0 else str(link)
            rows.append({"robot_link": name,
                         "normal_force": round(float(contact[9]), 3)})
        out[key] = rows
    return out


def disable_robot_table_collision(env, key: str = TABLE_KEY,
                                  extra_bodies=()) -> int:
    """Keep task supports load-bearing without letting scene proxies trap the robot.

    The proxy table is one box spanning nearly the whole workspace. Its top is useful
    for supporting task objects, but collision with the robot arm blocks side raises.
    The added floor-to-table bench is another proxy collider and sits directly in the
    side-raise path, so it must be filtered too. Object/robot and ground/robot
    collisions remain enabled.
    """
    table = env.task.objects.get(key)
    if table is None:
        return 0
    scene_bodies = [table, *list(extra_bodies)]
    disabled = 0
    for link in range(-1, p.getNumJoints(env.robot_id, physicsClientId=env.client)):
        for body in scene_bodies:
            try:
                p.setCollisionFilterPair(
                    env.robot_id, body, int(link), -1, 0,
                    physicsClientId=env.client)
                disabled += 1
            except p.error:
                continue
    return disabled


def disable_robot_object_collision_except_hand(env, key: str,
                                               side: str = "right") -> int:
    """Disable arm/torso contacts with a delicate object while keeping finger contacts."""
    obj = env.task.objects.get(key)
    if obj is None:
        return 0
    hand_names = set(env.robot_loader.hand_joints.get(side, []))
    keep_links = set()
    all_links = []
    for index in range(p.getNumJoints(env.robot_id, physicsClientId=env.client)):
        info = p.getJointInfo(env.robot_id, index,
                              physicsClientId=env.client)
        joint_name = info[1].decode("utf-8", errors="ignore")
        link_index = int(info[16])
        all_links.append(link_index)
        if joint_name in hand_names:
            keep_links.add(link_index)
    disabled = 0
    for link in [-1] + all_links:
        if link in keep_links:
            continue
        try:
            p.setCollisionFilterPair(
                env.robot_id, obj, int(link), -1, 0,
                physicsClientId=env.client)
            disabled += 1
        except p.error:
            continue
    return disabled


def snapshot(env, path, target=None, yaw: float = 55.0, pitch: float = -22.0,
             distance: float = 1.9, width: int = 640, height: int = 480) -> bool:
    """离屏渲染一张 PNG（与 ``verify_three_robot_scene.render`` 同一套取景参数）。

    现场看 GUI 窗口之外还要一份可以贴进文档/报告的静态图；``ER_TINY_RENDERER`` 在
    DIRECT 与 GUI 两个客户端上都能出图，所以 ``--render`` 与否都能拍。
    """
    import cv2

    if target is None:
        positions = object_positions(env)
        target = positions.get("cube") or positions.get("table") or [0.4, 0.0, 0.9]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(target), distance=float(distance), yaw=float(yaw),
        pitch=float(pitch), roll=0, upAxisIndex=2)
    projection = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / float(height), nearVal=0.1,
        farVal=float(distance) * 4.0 + 10.0)
    image = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=projection,
                             renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)
    rgba = np.asarray(image[2]).reshape(height, width, 4).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)))


def evaluate_task(robot: str, task_name: str, trials: int, policy: str,
                  actions_path: str = None, max_steps: int = 14400,
                  settle_steps: int = SETTLE_STEPS, scene: bool = True,
                  render: bool = False, verbose: bool = True,
                  policy_options: dict = None, max_wall_s: float = None,
                  snapshot_dir: str = None,
                  success_hold_steps: int = SUCCESS_HOLD_STEPS,
                  stop_on_success: bool = True,
                  allow_gui_mouse_picking: bool = False) -> dict:
    """一个任务跑 N 个 trial，返回 report（含每个 trial 的 JSON 段）。"""
    from teleoperation.simulation.environment import SimulationEnv

    options = dict(policy_options or {})
    arm = measure_arm(robot) if scene else {}
    report = {"meta": {"robot": robot, "task": task_name, "trials": int(trials),
                       "policy": policy, "scene": bool(scene),
                       "settle_steps": int(settle_steps), "max_steps": int(max_steps),
                       "max_wall_s": max_wall_s,
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
        live = None
        task_scene = None
        try:
            env.reset(randomize=False)
            if render:
                # GUI picking can teleport task objects outside robot control.
                # Evaluation must accept only robot/contact-driven state changes.
                p.configureDebugVisualizer(
                    p.COV_ENABLE_MOUSE_PICKING,
                    1 if allow_gui_mouse_picking else 0,
                    physicsClientId=env.client)
            scene_report = None
            if scene:
                workspace = (camera_workspace_profile(task_name)
                             if policy == "camera" else {})
                scene_shoulder = arm.get("shoulder_m")
                if workspace:
                    position, orientation = p.getBasePositionAndOrientation(
                        env.robot_id, physicsClientId=env.client)
                    p.resetBasePositionAndOrientation(
                        env.robot_id,
                        [position[0], position[1],
                         position[2] - float(workspace["robot_drop_m"])],
                        orientation, physicsClientId=env.client)
                    scene_shoulder = np.asarray(
                        arm.get("shoulder_m"), dtype=np.float64).copy()
                    scene_shoulder[2] -= float(workspace["robot_drop_m"])
                task_scene = TaskScene(env.client, env.task, robot=robot,
                                      shoulder=scene_shoulder,
                                      reach=arm.get("reach_m"),
                                      name=task_name,
                                      robot_id=env.robot_id)
                scene_report = task_scene.prepare(
                                             settle_steps,
                                             ik_workspace=(policy in (
                                                 "scripted", "camera")),
                                             side=options.get("side"),
                                             hold_robot=(policy in (
                                                 "camera", "replay", "dummy")),
                                             layout_forward=(
                                                 float(workspace.get(
                                                     "layout_forward_m", 0.38))
                                                 if policy == "camera" else 0.26),
                                             layout_drop=(
                                                 max(0.0, float(scene_shoulder[2])
                                                     - float(workspace[
                                                         "table_top_m"]))
                                                 if workspace else
                                                 (0.18 if policy == "camera"
                                                  else 0.22)),
                                             layout_lateral=(
                                                 0.15 if policy == "camera" else 0.02),
                                             layout_side_bias=(
                                                 (-0.06 if options.get("side") == "right"
                                                  else 0.06)
                                                 if policy == "camera" else 0.0),
                                             layout_table_min_x=(
                                                 float(workspace.get(
                                                     "layout_table_min_x_m", 0.29))
                                                 if policy == "camera" else 0.24),
                                             camera_pickplace_spacing=(
                                                 policy == "camera"
                                                 and task_name == "pickplacecube"),
                                             camera_cube_pre_grasp=(
                                                 policy == "camera"
                                                 and task_name == "pickcube"),
                                             camera_cup_stability=(
                                                 policy == "camera"
                                                 and task_name == "uprearcup"),
                                             camera_balltrashcan_setup=(
                                                 policy == "camera"
                                                 and task_name == "balltrashcan"),
                                             ik_comfort_fraction=(
                                                 0.98 if policy == "camera"
                                                 and task_name == "pickplacecube"
                                                 else 0.80))
                if workspace:
                    scene_report["camera_workspace"] = dict(workspace)
                if policy == "camera":
                    if task_name == "uprearcup":
                        cup_contacts_disabled = (
                            disable_robot_object_collision_except_hand(
                                env, "cup", side=options.get("side", "right")))
                        scene_report["robot_cup_collision_links_disabled"] = (
                            cup_contacts_disabled)
                    disabled = disable_robot_table_collision(
                        env, extra_bodies=task_scene.extra_bodies)
                    scene_report["robot_table_collision_links_disabled"] = disabled
                    scene_report["robot_scene_collision_bodies"] = (
                        1 + len(task_scene.extra_bodies))
                    print("[scene] robot/scene-support collision disabled for %d "
                          "robot-link pairs; object and ground collisions remain on"
                          % disabled,
                          flush=True)

            initial_objects = object_positions(env)
            if {"cube", "target"} <= set(initial_objects):
                cube = np.asarray(initial_objects["cube"], dtype=np.float64)
                target = np.asarray(initial_objects["target"], dtype=np.float64)
                delta = target - cube
                print("[task] cube -> target: dx=%+.3f dy=%+.3f dz=%+.3f m"
                      % (delta[0], delta[1], delta[2]), flush=True)
            elif initial_objects:
                print("[task] initial objects: " + ", ".join(
                    "%s=%s" % (key, [round(float(v), 3) for v in value])
                    for key, value in sorted(initial_objects.items())
                    if key != TABLE_KEY), flush=True)
            annotate_task_objects(env)

            actions, policy_meta = build_actions(policy, env, actions_path, max_steps,
                                                 **options)
            live = actions if callable(actions) else None
            result = run_trial(
                env, actions, max_steps, max_wall_s=max_wall_s,
                success_hold_steps=success_hold_steps,
                stop_on_success=stop_on_success)
            trial = dict(result, trial=index, **policy_meta,
                         scene=scene_report, initial_objects=initial_objects,
                         final_objects=object_positions(env),
                         gui_mouse_picking_enabled=(
                             bool(allow_gui_mouse_picking) if render else None))
            if policy == "camera":
                trial["robot_object_contacts"] = robot_contact_summary(env)
            if scene_report is not None and scene_report.get("criteria_at_reset"):
                # 场景摆完判据就已经成立 -> 这条 trial 的成功是假的（布局把物体摆得太近），
                # 如实记成失败并写明原因，绝不让它进成功率。
                trial.update({"success": False, "completion_time_s": None,
                              "criteria_at_reset": True,
                              "note": "场景摆完后判据即为真（layout 把物体拉得太近）-> 不计成功"})

            if live is not None:
                try:
                    trial["camera" if policy == "camera" else policy + "_report"] = live.report()
                except Exception as error:  # report failure must not erase the trial
                    trial["camera" if policy == "camera" else policy + "_report"] = {
                        "error": f"{type(error).__name__}: {error}"}

            if snapshot_dir and p.isConnected(env.client):
                shot = Path(snapshot_dir) / ("%s_%s_t%d.png" % (task_name, robot, index))
                try:
                    if snapshot(env, shot):
                        trial["snapshot"] = str(shot)
                except p.error as error:
                    trial["snapshot_error"] = f"{type(error).__name__}: {error}"
            report["trials"].append(trial)
            if verbose:
                print("[%s] %s/%s trial %d: %s" % (
                    "OK  " if trial["success"] else "FAIL", task_name, robot, index,
                    ("success at %.2f s" % trial["completion_time_s"])
                    if trial["success"] else
                    "no success in %.1f s" % trial["sim_time_s"]), flush=True)
        finally:
            if live is not None:
                live.close()
            try:
                env.close()
            except p.error:
                # User closed the GUI window: the direct connection is already gone.
                pass
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
    parser.add_argument("--policy", default="idle",
                        choices=["idle", "dummy", "replay", "camera", "scripted"],
                        help="idle=不下发动作（查假成功）；dummy=正弦假动作；"
                             "replay=契约H 动作文件；camera=摄像头实时驱动（现场用）；"
                             "scripted=脚本化操作（无人：IK 到位 + 抓取约束，见 task_scripted.py）")

    parser.add_argument("--actions", default=None, help="--policy replay 的动作文件")
    parser.add_argument("--camera-index", type=int, default=0, help="--policy camera 的相机编号")
    parser.add_argument("--camera-input-npz", default=None,
                        help="--policy camera 用录制帧代替摄像头做可复现回放")
    parser.add_argument("--side", default="right", choices=["left", "right"],
                        help="--policy camera 驱动哪一侧手臂（默认 right）")
    parser.add_argument("--arm-map", default="posture",
                        choices=["chain", "segments", "posture"],
                        help="--policy camera 的重定向口径（默认 posture = 方向口径）")
    parser.add_argument("--substeps", type=int, default=CAMERA_SUBSTEPS,
                        help="--policy camera 一个摄像头帧对应几个物理步（默认 12 = 20 fps 同节奏）")
    parser.add_argument("--calibrate-frames", type=int, default=30,
                        help="--policy camera 的中性姿势标定帧数（默认 30，约 1.5 s）")
    parser.add_argument("--max-wall-s", type=float, default=None,
                        help="一个 trial 最多多少**墙钟**秒（--policy camera 建议 30-60）")
    parser.add_argument("--preview", action="store_true",
                        help="--policy camera 时弹摄像头预览窗（看手点/骨架/目标状态；q 退出）")
    parser.add_argument("--camera-hand", choices=("on", "off"), default="on",
                        help="--policy camera 同时驱动原装手（default on）；off 保留只驱动"
                             "手臂的旧 A/B 行为")
    parser.add_argument("--hand-backend", choices=("auto", "geometric", "checkpoint"),
                        default="auto", help="手部重定向后端（default auto）")
    parser.add_argument("--hand-checkpoint", default=None,
                        help="训练好的手部 palm-local checkpoint（auto 缺失时回退 geometric）")
    parser.add_argument("--hand-smoothing", type=float, default=0.25,
                        help="手部 18D 输出平滑系数 0..<1（default 0.25）")
    parser.add_argument("--hand-limit-mode", choices=("clamp", "rescale"),
                        default="clamp",
                        help="L21 -> 原装手限位处理：clamp 忠实截断（default），"
                             "rescale 按 L21 比例压到原生行程（视觉辅助，需在报告中声明）")
    parser.add_argument("--camera-table-clearance-mm", type=float, default=60.0,
                        help="启用 --camera-table-guard 时，腕目标在桌面投影内"
                             "至少离桌面这么高（default 60 mm）")
    parser.add_argument("--camera-table-guard", action="store_true",
                        help="把桌面投影内的腕目标强制抬到桌面上方；默认关闭，"
                             "因为摄像头任务已经关闭机器人-桌子碰撞，硬抬会逐帧跳变")
    parser.add_argument("--camera-arm-force", type=float, default=200.0,
                        help="--policy camera 手臂位置电机最大力（default 200，"
                             "与已验证的 quickcheck 链路一致）")
    parser.add_argument("--camera-hand-force", type=float, default=200.0,
                        help="--policy camera 手指位置电机最大力（default 200，"
                             "与已验证的 quickcheck 链路一致）")
    parser.add_argument("--camera-ik-mode", choices=("position", "pose"),
                        default="pose",
                        help="--policy camera 的臂 IK 模式（default pose，"
                             "让原生手掌朝向跟随操作者）")
    parser.add_argument("--camera-ik-solver", choices=("paper", "pybullet"),
                        default="paper",
                        help="--policy camera 的 IK 后端（default paper，"
                             "与 run_arm_quickcheck 一致）")
    parser.add_argument("--camera-wrist-pose", choices=("on", "off"), default="on",
                        help="--policy camera 是否把操作者手腕/掌面姿态传给 IK"
                             "（default on；off 为只跟位置的旧 A/B 行为）")
    parser.add_argument("--camera-grasp-assist", choices=("on", "off"), default="on",
                        help="--policy camera 在手指已产生多处接触且正在握拳时"
                             "锁定物体到手掌，松手解除（default on）")
    parser.add_argument("--camera-target-speed-mm-s", type=float, default=600.0,
                        help="--policy camera 腕目标最大跟随速度（default 600 mm/s）")
    parser.add_argument("--camera-startup-hold-frames", type=int, default=12,
                        help="标定完成后手臂保持当前姿势多少摄像头帧再开始跟随"
                             "（default 12，约 0.5-0.8 s）")
    parser.add_argument("--camera-startup-motion-mm", type=float, default=50.0,
                        help="标定后腕目标至少移动多少毫米才认定开始操作"
                             "（default 50 mm）")
    parser.add_argument("--camera-startup-confirm-frames", type=int, default=3,
                        help="超过启动位移后还需连续多少帧才开门"
                             "（default 3，约 0.15 s）")
    parser.add_argument("--camera-target-deadband-mm", type=float, default=2.0,
                        help="腕目标小于该变化时保持不动（default 2 mm）")
    parser.add_argument("--camera-freeze-objects", choices=("on", "off"),
                        default="on",
                        help="标定和启动保持期间冻结可动物体，开始跟随后再恢复物理"
                             "（default on）")
    parser.add_argument("--max-sim-s", type=float, default=DEFAULT_MAX_SIM_S,
                        help="一个 trial 最多多少**仿真**秒（默认 60）")
    parser.add_argument("--success-hold-ms", type=float, default=500.0,
                        help="任务判据必须连续成立多久才算成功（default 500 ms）")
    parser.add_argument("--demo-keep-open", action="store_true",
                        help="成功后继续运行，直到手动关闭窗口（演示用）")
    parser.add_argument("--settle-steps", type=int, default=SETTLE_STEPS,
                        help="场景摆好后空转多少步（默认 600 = 2.5 s）")
    parser.add_argument("--no-scene", action="store_true", help="不套场景层（baseline 原样跑）")
    parser.add_argument("--verify", action="store_true",
                        help="跑四项检查（30 个任务过场景层），不跑 trial")
    parser.add_argument("--render", action="store_true", help="GUI 渲染（默认 DIRECT）")
    parser.add_argument("--allow-gui-mouse-picking", action="store_true",
                        help="允许使用 PyBullet 鼠标拖动；仅用于演示，不能作为有效评测结果")
    parser.add_argument("--json", default=None, help="报告落盘路径")
    parser.add_argument("--snapshot-dir", default=None,
                        help="每个 trial 结束时离屏渲染一张 PNG 到这个目录（现场取证用）")
    parser.add_argument("--no-grasp-assist", action="store_true",
                        help="--policy scripted 关掉抓取约束（对照：纯接触抓不住）")
    parser.add_argument("--scripted-speed", type=float, default=None,
                        help="--policy scripted 末端速度 m/s（默认 0.35）")
    parser.add_argument("--scripted-ik-every", type=int, default=None,
                        help="--policy scripted 每几个物理步解一次 IK（默认 8）")
    parser.add_argument("--scripted-trace", action="store_true",
                        help="--policy scripted 打印每段到位情况（排障用）")
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
    success_hold_steps = max(
        1, int(round(max(0.0, float(args.success_hold_ms)) / 1000.0 / DT)))
    if args.policy == "scripted":
        from task_scripted import IK_EVERY_STEPS, SPEED_M_S
        options = {"side": args.side,
                   "grasp_assist": not args.no_grasp_assist,
                   "speed_m_s": (SPEED_M_S if args.scripted_speed is None
                                 else args.scripted_speed),
                   "ik_every": (IK_EVERY_STEPS if args.scripted_ik_every is None
                                else args.scripted_ik_every),
                   "trace": args.scripted_trace}

    else:
        options = {"side": args.side, "camera_index": args.camera_index,
                   "input_npz": args.camera_input_npz,
                   "substeps": args.substeps, "calibrate_frames": args.calibrate_frames,
                   "arm_map": args.arm_map, "preview": args.preview,
                   "hand_enabled": args.camera_hand == "on",
                   "hand_backend": args.hand_backend,
                   "hand_checkpoint": args.hand_checkpoint,
                   "hand_smoothing": args.hand_smoothing,
                   "hand_limit_mode": args.hand_limit_mode,
                   "table_clearance_m": max(
                       0.0, float(args.camera_table_clearance_mm)) / 1000.0,
                   "table_guard": bool(args.camera_table_guard),
                   "arm_force": max(1.0, float(args.camera_arm_force)),
                   "hand_force": max(1.0, float(args.camera_hand_force)),
                   "ik_mode": args.camera_ik_mode,
                   "ik_solver": args.camera_ik_solver,
                   "wrist_pose": args.camera_wrist_pose == "on",
                   "grasp_assist": args.camera_grasp_assist == "on",
                   "target_speed_m_s": max(
                       0.0, float(args.camera_target_speed_mm_s)) / 1000.0,
                   "startup_hold_frames": max(
                       0, int(args.camera_startup_hold_frames)),
                   "startup_motion_m": max(
                       0.0, float(args.camera_startup_motion_mm)) / 1000.0,
                   "startup_confirm_frames": max(
                       1, int(args.camera_startup_confirm_frames)),
                   "target_deadband_m": max(
                       0.0, float(args.camera_target_deadband_mm)) / 1000.0,
                   # A mass=0 object cannot be dragged by PyBullet's GUI
                   # picking tool.  Manual demos must keep the objects dynamic
                   # while formal evaluation keeps the freeze/startup gate.
                   "freeze_objects": camera_objects_frozen(
                       args.camera_freeze_objects,
                       args.allow_gui_mouse_picking)}

    reports = []
    for name in tasks:
        report = evaluate_task(args.robot, name, args.trials, args.policy,
                               actions_path=args.actions, max_steps=max_steps,
                               settle_steps=args.settle_steps,
                               scene=not args.no_scene, render=args.render,
                               policy_options=options,
                               max_wall_s=args.max_wall_s,
                               snapshot_dir=args.snapshot_dir,
                               success_hold_steps=success_hold_steps,
                               stop_on_success=not args.demo_keep_open,
                               allow_gui_mouse_picking=(
                                   args.allow_gui_mouse_picking))
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
