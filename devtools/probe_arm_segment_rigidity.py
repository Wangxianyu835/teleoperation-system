"""肩-肘 / 肘-腕到底是不是**刚体段**？——「方向配方」把肘目标放在 L1*上臂方向 处的前提。

用法::

    E:\\python3.11.7\\python.exe devtools\\probe_arm_segment_rigidity.py [--json out.json]

为什么（2026-10-14 现场「GR1-T2 前臂还是反」）：

200 帧回放实测 ``elbow_task_mm``（肘目标残差）gr1_t2 左 222 mm、右 157 mm，而同一帧的
``solution_elbow_mm`` 与它**逐帧相等** —— 说明手臂就停在 IK 给的那个解上（电机目标 = 解，
姿势在平衡点），所以不是「物理追不上」，是**解本身**就离目标那么远。

那就要问三件事，这个探针分别量：

1. **段是不是刚体**：配方拿加载时量到的 ``L1 = |肘 - 肩|``、``L2 = |腕 - 肘|`` 当常数，把肘
   目标放在「肩 + L1*上臂方向」。把每个臂关节单独拧开 ±60/±100 度再量同一对距离：如果两个
   距离随姿势变了，说明当锚点的那个 link（``*shoulder*`` / ``*elbow*``）不在旋转轴上，配方
   给的肘目标**几何上不可达**，IK 只能把误差摊到肘和腕上（正是 222 mm / 98 mm 那对数）。
2. **求解器能不能兑现一对「一定能到」的目标**：取这台机器人中性姿势自己的 FK 三点，按配方
   方式造一对目标（肩 + L1*u0，再加 L2*f0，u0/f0 取中性方向），从**拧开过的姿势**出发解一遍。
   这对目标由机器人自己的连杆长造出来，任何姿势下都应当可达；解不出来就是求解器/限位的问题。
3. **IK 把哪个帧送到目标点上**（2026-10-14 新增，§8.13.8 的第①条未查项）：``solve`` 交进去的是
   「连杆索引 + 世界点」，而 ``arm_points()`` 读 ``getLinkState()[4]``（URDF link frame = 关节
   原点）。只要 ``calculateInverseKinematics2`` 驱动的是 ``[0]``（惯性/质心帧），两个位置任务就
   带着一个**常量偏差**在解：腕那截小（残差 10-30 mm）、肘那截大（残差 70-234 mm），正是现场
   报告的形状。量法见 ``ik_frame_test``（恒等测试 + 沿基座 +X 的驱动测试），顺带记限位余量。

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: 单个关节拧开的角度（度）：够大就能暴露「锚点不在轴上」，又不至于撞限位。
SWEEP_DEG = (60.0, -60.0, 100.0, -100.0)


def distance_mm(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64)
                                - np.asarray(right, dtype=np.float64))) * 1000.0


def unit(end, start):
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-9 else vector / norm


def build(sides):
    """三机器人场景 + 手臂驱动（与 realtime_teleop 同一套构造路径）。"""
    from teleoperation.simulation.realtime_teleop import build_parser, build_scene

    args = build_parser().parse_args(["--headless"])
    args.scene = "robots"           # 手臂只存在于机器人场景（与 realtime_teleop 同一条路）
    return build_scene(args, tuple(sides))


def link_name(p, body, link_index, cid) -> str:
    try:
        info = p.getJointInfo(body, int(link_index), physicsClientId=cid)
        return str(info[12], "utf-8", "replace") if isinstance(info[12], bytes) else str(info[12])
    except Exception:                                   # noqa: BLE001 - 纯诊断
        return f"link{link_index}"


def rigid_anchor(adapter, side, p, cid):
    """肘关节的**父连杆原点** = 肘关节的父关节转轴 = 上臂的肩端（随姿势不变）。"""
    elbow_index = adapter.elbow_link[side]
    parent_link = int(p.getJointInfo(adapter.body, int(elbow_index),
                                     physicsClientId=cid)[16])
    point = tuple(float(value) for value in
                  p.getLinkState(adapter.body, parent_link,
                                 computeForwardKinematics=True,
                                 physicsClientId=cid)[4])
    return parent_link, point


def sweep_arm(adapter, side, p, cid):
    """把这一侧每个臂关节单独拧开，返回两段距离随姿势的最大偏差。

    三条口径一起量，好让报告自己说明为什么要换锚点：

    ``shoulder``  = 名字匹配到的第一个肩关节连杆（历史口径，报告/测试在用）
    ``upper``     = ``arm_points()`` 的刚体肩端（= 上面那个 rigid_anchor，修好之后的口径）
    ``wrist``     = ``arm_points()`` 的手腕点（ee 连杆；腕关节因此会让 L2 变）
    """
    joints = list(adapter.arm[side])
    neutral = {index: float(p.getJointState(adapter.body, index,
                                            physicsClientId=cid)[0])
               for index, _name in joints}
    points = adapter.arm_points(side)
    anchor_link, anchor_point = rigid_anchor(adapter, side, p, cid)
    upper0 = distance_mm(points["elbow"], points["shoulder"])
    fore0 = distance_mm(points["wrist"], points["elbow"])
    rigid0 = distance_mm(points["elbow"], anchor_point)
    builtin0 = distance_mm(points["elbow"], points["upper"])
    out = {"shoulder_link": link_name(p, adapter.body, adapter.shoulder_link[side], cid),
           "elbow_link": link_name(p, adapter.body, adapter.elbow_link[side], cid),
           "upper_link": link_name(p, adapter.body, adapter.upper_link[side], cid),
           "rigid_anchor_link": link_name(p, adapter.body, anchor_link, cid),
           "upper_mm": round(upper0, 2), "fore_mm": round(fore0, 2),
           "rigid_upper_mm": round(rigid0, 2),
           "builtin_upper_mm": round(builtin0, 2),
           #: 内置 ``upper`` 与「肘关节父连杆」是不是同一个点（口径自检，应为 0.000）。
           "builtin_vs_parent_mm": round(abs(builtin0 - rigid0), 3),
           "rigid_upper_sweep_mm": 0.0,
           "builtin_upper_sweep_mm": 0.0,
           "joints": {}, "breaks": {}}
    for index, name in joints:
        worst_upper, worst_fore, angle_at, worst_rigid = 0.0, 0.0, None, 0.0
        worst_builtin = 0.0
        for delta in SWEEP_DEG:
            for target_index, value in neutral.items():
                p.resetJointState(adapter.body, target_index, float(value),
                                  physicsClientId=cid)
            p.resetJointState(adapter.body, index,
                              float(neutral[index]) + math.radians(delta),
                              physicsClientId=cid)
            moved = adapter.arm_points(side)
            upper = distance_mm(moved["elbow"], moved["shoulder"])
            fore = distance_mm(moved["wrist"], moved["elbow"])
            builtin = distance_mm(moved["elbow"], moved["upper"])
            moved_anchor = tuple(float(value) for value in
                                 p.getLinkState(adapter.body, anchor_link,
                                                computeForwardKinematics=True,
                                                physicsClientId=cid)[4])
            rigid = distance_mm(moved["elbow"], moved_anchor)
            if abs(upper - upper0) > abs(worst_upper):
                worst_upper, angle_at = upper - upper0, delta
            if abs(fore - fore0) > abs(worst_fore):
                worst_fore = fore - fore0
            if abs(rigid - rigid0) > abs(worst_rigid):
                worst_rigid = rigid - rigid0
            if abs(builtin - builtin0) > abs(worst_builtin):
                worst_builtin = builtin - builtin0
        out["rigid_upper_sweep_mm"] = round(max(out["rigid_upper_sweep_mm"],
                                                abs(worst_rigid)), 2)
        out["builtin_upper_sweep_mm"] = round(max(out["builtin_upper_sweep_mm"],
                                                  abs(worst_builtin)), 2)
        out["joints"][name] = {"d_upper_mm": round(worst_upper, 2),
                               "d_fore_mm": round(worst_fore, 2),
                               "d_rigid_mm": round(worst_rigid, 2),
                               "d_builtin_mm": round(worst_builtin, 2),
                               "at_deg": angle_at}
        if abs(worst_upper) > 5.0:
            out["breaks"][name] = round(worst_upper, 2)
    for target_index, value in neutral.items():
        p.resetJointState(adapter.body, target_index, float(value), physicsClientId=cid)
    return out


def recipe_roll_test(adapter, side, p, cid, anchor_key, roll_deg=60.0):
    """**真实失败模式**：只把肩 roll 拧开 ``roll_deg``，再按某个锚点口径造配方目标，量残差。

    ``ArmSceneDriver._shoulder_shift`` 会把两个目标平移「当前锚点 - 加载锚点」，所以目标总是落在
    **当前**锚点 + ``L*u`` 处；要复现现场，A/B 两边都用「当前那一侧的锚点」，只换 key：

    - ``anchor_key="shoulder"``（历史口径）：锚点是上臂 pitch 连杆，**不随 roll 转动**（它自己的
      转轴在别的链节上），而 ``L1 = |肘 - 那个点|`` 是加载时量的 331/251/193 mm 常数。roll 一转，
      肘到**真正**肩端的距离就变了 62-102 mm，于是「肩 + L1*u」这个肘目标与「肘目标 + L2*v」这个
      腕目标**同时不可达**，7 自由度链只能折中 —— 现场看到的就是「前臂弯曲方向相反」。
    - ``anchor_key="upper"``（修好之后）：锚点是肘的父连杆原点（刚体肩端），到肘的距离与姿势无关，
      两个目标同时可达，残差应当回到毫米级。

    每个变体先把 ``elbow_recovery_wait`` 清零：探针要量的是「求解器能不能兑现」，不是上一个变体
    留下的节流窗口。
    """
    joints = list(adapter.arm[side])
    neutral = {index: float(p.getJointState(adapter.body, index, physicsClientId=cid)[0])
               for index, _name in joints}
    for index, _name in joints:                     # 回中性再量「加载时」的锚点与连杆
        p.resetJointState(adapter.body, index, float(neutral[index]), physicsClientId=cid)
    points0 = adapter.arm_points(side)
    anchor0 = np.asarray(points0[anchor_key], dtype=np.float64)
    u_rest = unit(points0["elbow"], points0[anchor_key])
    v_rest = unit(points0["wrist"], points0["elbow"])
    length_upper = distance_mm(points0["elbow"], points0[anchor_key]) / 1000.0
    length_fore = distance_mm(points0["wrist"], points0["elbow"]) / 1000.0
    roll_joint = next((index for index, name in joints
                       if "shoulder" in name.lower() and "roll" in name.lower()), None)
    if roll_joint is not None:
        p.resetJointState(adapter.body, roll_joint,
                          float(neutral[roll_joint]) + math.radians(roll_deg),
                          physicsClientId=cid)
    moved = adapter.arm_points(side)
    anchor_used = np.asarray(moved[anchor_key], dtype=np.float64)
    # 目标到**当前**锚点的距离：这才决定这对目标可不可达（加载时的锚点差多少就是上面那个 sweep）。
    reachable_gap_mm = distance_mm(anchor_used, moved["elbow"]) - distance_mm(anchor0,
                                                                            points0["elbow"])
    elbow_target = anchor_used + length_upper * u_rest
    wrist_target = elbow_target + length_fore * v_rest
    adapter.elbow_recovery_wait[side] = 0
    solution = adapter.solve(side, tuple(wrist_target), source="world",
                             elbow=tuple(elbow_target))
    adapter.apply(side, solution)
    achieved = adapter.arm_points(side)
    return {"anchor": anchor_key, "roll_deg": round(roll_deg, 1),
            "link_mm": round(length_upper * 1000.0, 1),
            "true_distance_mm": round(distance_mm(anchor_used, moved["elbow"]), 1),
            "unreachable_by_mm": round(reachable_gap_mm, 1),
            "elbow_mm": round(distance_mm(achieved["elbow"], elbow_target), 2),
            "wrist_mm": round(distance_mm(achieved["wrist"], wrist_target), 2)}


def reachable_pair_test(adapter, side, p, cid, anchor_key="upper"):
    """**求解器自检**：用机器人自己中性姿势的 FK 造一对目标，从拧开的姿势解一遍。

    两个锚点口径造出来的是**同一对目标点**（锚点 + 它自己的连杆长 * 它自己的方向 = 同一个肘/腕
    点），所以两边都应当是 0 mm —— 不是 0 就说明求解器/限位有问题，不是锚点问题（锚点问题看
    ``recipe_roll_test``）。每个变体先把 ``elbow_recovery_wait`` 清零，避免上一个变体留下的
    节流窗口把这一次求救挡掉。
    """
    joints = list(adapter.arm[side])
    neutral = {index: float(p.getJointState(adapter.body, index, physicsClientId=cid)[0])
               for index, _name in joints}
    points = adapter.arm_points(side)
    anchor = np.asarray(points[anchor_key], dtype=np.float64)
    upper0 = unit(points["elbow"], points[anchor_key])
    fore0 = unit(points["wrist"], points["elbow"])
    length_upper = distance_mm(points["elbow"], points[anchor_key]) / 1000.0
    length_fore = distance_mm(points["wrist"], points["elbow"]) / 1000.0
    for index, _name in joints:
        p.resetJointState(adapter.body, index,
                          float(neutral[index]) + math.radians(50.0), physicsClientId=cid)
    elbow_target = anchor + length_upper * upper0
    wrist_target = elbow_target + length_fore * fore0
    adapter.elbow_recovery_wait[side] = 0
    solution = adapter.solve(side, tuple(wrist_target), source="world",
                             elbow=tuple(elbow_target))
    adapter.apply(side, solution)
    achieved = adapter.arm_points(side)
    return {"anchor": anchor_key,
            "elbow_mm": round(distance_mm(achieved["elbow"], elbow_target), 2),
            "wrist_mm": round(distance_mm(achieved["wrist"], wrist_target), 2)}


def pose_tracking_test(adapter, side, p, cid, spreads=(5.0, 35.0),
                       variants=((False, 60), (True, 60), (True, 800))):
    """**求解器够不够**：目标 = 机器人自己一条合法姿势的 FK（精确可达），从另一条姿势解过去。

    现场工况（warm start、同一条臂、目标每帧只动一点）在这里近似成「中性 -> 差 ``spread`` 度的合法
    姿势」：目标一定可达（它就是 FK 出来的），于是残差只剩「求解器 / 限位」。``spread=5`` 是现场的
    逐帧步长量级，``spread=35`` 是大幅摆动。变体 = ``(是否锁住非本侧手臂关节, IK 迭代数)``：
    残差随迭代数明显降 = 求解器没收敛；把别的关节锁住之后降下来 = 求解器在拿身体去凑手臂的两个
    目标（``_locked_arguments``）；怎么都不降 = 目标被限位挡住（换配方口径没用）。

    支路恢复先关掉（``elbow_recovery_deg = 0``）：这里量的是求解器，不是恢复逻辑。
    """
    joints = list(adapter.arm[side])
    neutral = {index: float(p.getJointState(adapter.body, index, physicsClientId=cid)[0])
               for index, _name in joints}
    saved_recovery = adapter.elbow_recovery_deg
    saved_lock = adapter.ik_lock_others
    saved_iterations = adapter.ik_iterations
    adapter.elbow_recovery_deg = 0.0
    out = []
    try:
        for spread in spreads:
            for index, _name in joints:             # 目标姿势：每个臂关节 +spread，按限位裁剪
                info = p.getJointInfo(adapter.body, index, physicsClientId=cid)
                low, high = float(info[8]), float(info[9])
                wanted = float(neutral[index]) + math.radians(spread)
                p.resetJointState(adapter.body, index, min(max(wanted, low), high),
                                  physicsClientId=cid)
            target = adapter.arm_points(side)
            elbow_target = np.asarray(target["elbow"], dtype=np.float64)
            wrist_target = np.asarray(target["wrist"], dtype=np.float64)
            entry = {"spread_deg": round(spread, 1), "solves": []}
            for lock, count in variants:
                for index, _name in joints:         # warm start = 中性（= 现场的「上一帧」）
                    p.resetJointState(adapter.body, index, float(neutral[index]),
                                      physicsClientId=cid)
                adapter.ik_iterations = int(count)
                adapter.ik_lock_others = bool(lock)
                adapter.elbow_recovery_wait[side] = 0
                solution = adapter.solve(side, tuple(wrist_target), source="world",
                                         elbow=tuple(elbow_target), elbow_pass="last")
                achieved = adapter.arm_points(side)   # solve 结束时手臂正停在解上
                entry["solves"].append({
                    "lock_others": bool(lock),
                    "iterations": int(count),
                    "elbow_mm": round(distance_mm(achieved["elbow"], elbow_target), 2),
                    "wrist_mm": round(distance_mm(achieved["wrist"], wrist_target), 2)})
            out.append(entry)
    finally:
        adapter.elbow_recovery_deg = saved_recovery
        adapter.ik_lock_others = saved_lock
        adapter.ik_iterations = saved_iterations
        for index, _name in joints:
            p.resetJointState(adapter.body, index, float(neutral[index]),
                              physicsClientId=cid)
    return out


#: IK 帧口径测试把目标沿基座 +X 平移多少（mm）：够大能盖过数值噪声，又小到不会撞限位。
IK_FRAME_PROBE_MM = 20.0


def link_frames(p, body, link_index, cid) -> dict:
    """同一个连杆的**两个**官方帧（PyBullet 自己给的两个口径）。

    ``com``   = ``getLinkState()[0]``：惯性帧（质心）；
    ``frame`` = ``getLinkState()[4]``：URDF 的 link frame（= 关节原点）—— ``arm_points()`` /
    ``_branch_cost`` 读的是它；``solve`` 交进 IK 的那个点按哪一帧解释，官方文档没写。
    """
    state = p.getLinkState(body, int(link_index), computeForwardKinematics=True,
                           physicsClientId=cid)
    return {"com": np.asarray(state[0], dtype=np.float64),
            "frame": np.asarray(state[4], dtype=np.float64)}


def ik_frame_test(adapter, side, p, cid, probe_mm=IK_FRAME_PROBE_MM):
    """**IK 把哪个帧送到目标点上**：link frame（``arm_points`` 的口径）还是惯性帧？

    为什么必须量（§8.13.8 / §8.15.6 留下的第①条）：「把肘也当一个位置目标」这条路一直在
    「求解器没兑现」上打转 —— 迭代 60 提到 800 六侧全更差、锁住别的关节只把腕残差降下来，
    肘残差仍有 22-234 mm。剩下的可能只有两个：**两个任务的口径不同**，或目标被限位挡住。
    ``solve`` 交进去的是「连杆索引 + 世界点」，而 ``arm_points()`` 读 ``[4]``：只要 IK 驱动
    的是 ``[0]``，两个位置任务就带着一个**常量偏差**在解 —— 腕那一截小（手部连杆质心离原点
    近，残差 10-30 mm）、肘那一截大（前臂质心在前臂中间，残差 70-230 mm），正是现场报告里
    那个形状。结论只有三选一：``link_frame`` / ``com_frame`` / 看不出来。

    两条量法都不走物理（``solve`` 结束时手臂正停在解上）：

    1. **恒等测试**：目标 = 当前 FK 点、从当前姿势再解一遍。关节角应当一动不动；两个帧各自
       到目标的距离里，小的那个就是 IK 驱动的帧，另一个的差就是两个口径的常量偏差。
    2. **驱动测试**：目标沿**基座 +X** 平移 ``probe_mm`` 再解一遍，量两个帧各动了多少；位移
       与目标位移一致的那个帧就是 IK 驱动的帧 —— 这一条只看运动学，不受「解收敛没有」影响，
       比第 1 条硬。

    腕（``ee`` 连杆）与肘（``elbow_link``）各做一遍：现场报告里腕残差 10-30 mm、肘残差
    70-234 mm，若这两个连杆的 ``com``/``frame`` 间距之比正好是这个比，结论就是它。
    限位余量一并记下（第②条：这个合法姿势会不会被限位挡住）。
    """
    joints = list(adapter.arm[side])
    angles0 = {int(index): float(p.getJointState(adapter.body, index,
                                                 physicsClientId=cid)[0])
               for index, _name in joints}
    #: ``solve(..., posture=...)`` 吃的名字 -> 弧度，用来把 restPoses 换成「当前姿势」
    current_angles = {str(name): float(angles0[int(index)]) for index, name in joints}
    saved_recovery = adapter.elbow_recovery_deg
    adapter.elbow_recovery_deg = 0.0            # 量的是求解器，不是支路恢复
    adapter.elbow_recovery_wait[side] = 0

    def reset_pose():
        for index, _name in joints:
            p.resetJointState(adapter.body, index, angles0[int(index)],
                              physicsClientId=cid)

    def solve_pair(wrist_target, elbow_target, rest_current=False):
        """同一对目标解一遍；``rest_current`` 把零空间的 restPoses 换成当前姿势。

        这就是「求解器为什么不在解上」的判别式：``solve`` 默认把 ``restPoses`` 交成
        **加载时**的姿势（``_rest_poses(None)``），而现场入口（``apply_each``）从不传
        ``posture``。若恒等测试的残差在这一档塌到 ~0，说明那个常量偏差点不是链路帧口径，
        而是零空间项把解从位置任务上拉走了。
        """
        extra = {"posture": current_angles} if rest_current else {}
        adapter.solve(side, tuple(wrist_target), source="world",
                      elbow=tuple(elbow_target), elbow_pass="last", **extra)

    try:
        points = adapter.arm_points(side)
        targets = {"wrist": np.asarray(points["wrist"], dtype=np.float64),
                   "elbow": np.asarray(points["elbow"], dtype=np.float64)}
        links = {"wrist": int(adapter.ee[side][0]),
                 "elbow": int(adapter.elbow_link[side])}
        before = {name: link_frames(p, adapter.body, index, cid)
                  for name, index in links.items()}
        out = {"probe_mm": round(float(probe_mm), 3),
               "links": {name: link_name(p, adapter.body, index, cid)
                         for name, index in links.items()},
               "frame_offset_mm": {name: round(distance_mm(before[name]["com"],
                                                           before[name]["frame"]), 2)
                                   for name in links}}
        # ① 恒等测试：目标 = 当前 FK 三点。解对了就一动不动。
        def identity_once(rest_current):
            reset_pose()
            solve_pair(targets["wrist"], targets["elbow"], rest_current)
            after = {name: link_frames(p, adapter.body, index, cid)
                     for name, index in links.items()}
            moved = max((abs(math.degrees(float(p.getJointState(
                adapter.body, index, physicsClientId=cid)[0]) - angles0[int(index)]))
                for index, _name in joints), default=0.0)
            return {
                "joint_max_delta_deg": round(float(moved), 3),
                "distance_to_target_mm": {
                    name: {"from_com": round(distance_mm(after[name]["com"],
                                                        targets[name]), 2),
                           "from_frame": round(distance_mm(after[name]["frame"],
                                                           targets[name]), 2)}
                    for name in links}}

        out["identity"] = identity_once(False)
        # ①b 同一件事，只把零空间的 restPoses 从「加载姿势」换成「当前姿势」：
        # 残差塌下来 = 偏差来自零空间项，不是链路帧口径。
        out["identity_current_rest"] = identity_once(True)
        reset_pose()

        # ② 驱动测试：目标沿基座 +X 平移 probe_mm，看两个帧各动了多少。
        delta_base = np.array([float(probe_mm) / 1000.0, 0.0, 0.0], dtype=np.float64)

        def drive_once(rest_current):
            result = {}
            for name in ("wrist", "elbow"):
                reset_pose()
                shifted = np.asarray(adapter.to_world(tuple(
                    np.asarray(adapter.to_base(targets[name]), dtype=np.float64)
                    + delta_base)), dtype=np.float64)
                intended = shifted - targets[name]
                wrist_target = shifted if name == "wrist" else targets["wrist"]
                elbow_target = shifted if name == "elbow" else targets["elbow"]
                solve_pair(wrist_target, elbow_target, rest_current)
                after = link_frames(p, adapter.body, links[name], cid)
                record = {"intended_mm": round(float(np.linalg.norm(intended)) * 1000.0, 2)}
                for key in ("com", "frame"):
                    moved = after[key] - before[name][key]
                    record[key] = {
                        "moved_mm": round(float(np.linalg.norm(moved)) * 1000.0, 2),
                        "error_vs_intended_mm": round(float(np.linalg.norm(
                            moved - intended)) * 1000.0, 2)}
                record["drives"] = ("com" if record["com"]["error_vs_intended_mm"]
                                    < record["frame"]["error_vs_intended_mm"] else "frame")
                result[name] = record
            reset_pose()
            return result

        drive = drive_once(False)
        out["drive"] = drive
        out["ik_drives"] = {name: ("com_frame" if drive[name]["drives"] == "com"
                                   else "link_frame") for name in drive}
        # ②b 同样的驱动测试，零空间换成「当前姿势」：跟随率是不是就上来了。
        out["drive_current_rest"] = drive_once(True)

        # 限位余量：哪几个臂关节贴着限位（第②条的判据）。
        headroom = {}
        for index, name in joints:
            info = p.getJointInfo(adapter.body, index, physicsClientId=cid)
            low, high = float(info[8]), float(info[9])
            if high - low < 1e-9:
                continue
            headroom[str(name)] = round(min(math.degrees(angles0[int(index)] - low),
                                            math.degrees(high - angles0[int(index)])), 2)
        out["limit_headroom_deg"] = headroom
        out["min_limit_headroom_deg"] = min(headroom.values()) if headroom else None
        return out
    finally:
        adapter.elbow_recovery_deg = saved_recovery
        adapter.elbow_recovery_wait[side] = 0
        reset_pose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="把结果另外写一份 JSON")
    args = parser.parse_args()

    scene = build(("left", "right"))
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    driver = ArmSceneDriver(scene, sides=("left", "right"), track_posture=False,
                            slew_m=0.0)
    report = {}
    try:
        for robot_type, adapter in driver.ik.items():
            report[robot_type] = {}
            for side in ("left", "right"):
                entry = sweep_arm(adapter, side, scene.p, scene.cid)
                # A/B：同一套现场工况（只拧肩 roll），旧口径（名字匹配的肩点）vs 新口径（刚体肩端）。
                entry["roll_old_anchor"] = recipe_roll_test(adapter, side, scene.p,
                                                            scene.cid, "shoulder")
                entry["roll_new_anchor"] = recipe_roll_test(adapter, side, scene.p,
                                                            scene.cid, "upper")
                # 求解器自检：同一对「一定可达」的目标点，两个口径都应当是 0 mm。
                entry["solver_old_anchor"] = reachable_pair_test(adapter, side, scene.p,
                                                                 scene.cid, "shoulder")
                entry["solver_new_anchor"] = reachable_pair_test(adapter, side, scene.p,
                                                                 scene.cid, "upper")
                report[robot_type][side] = entry
                print(f"{robot_type} {side}: historical shoulder point = {entry['shoulder_link']}"
                      f" | rigid shoulder end (elbow's parent link) = {entry['upper_link']}"
                      f" | elbow = {entry['elbow_link']}", flush=True)
                print(f"    |elbow-shoulder|={entry['upper_mm']:.1f} mm "
                      f"(sweep max {max((abs(v['d_upper_mm']) for v in entry['joints'].values()), default=0.0):.1f} mm)"
                      f" | rigid |elbow-anchor|={entry['builtin_upper_mm']:.1f} mm "
                      f"(sweep max {entry['builtin_upper_sweep_mm']:.1f} mm)"
                      f" | |wrist-elbow|={entry['fore_mm']:.1f} mm", flush=True)
                print("    shoulder_roll +60 deg recipe (this is the field failure): "
                      f"old anchor (link {entry['roll_old_anchor']['link_mm']:.0f} mm, "
                      f"true {entry['roll_old_anchor']['true_distance_mm']:.0f} mm) -> "
                      f"elbow {entry['roll_old_anchor']['elbow_mm']:.1f} mm / "
                      f"wrist {entry['roll_old_anchor']['wrist_mm']:.1f} mm ; "
                      f"rigid anchor (link {entry['roll_new_anchor']['link_mm']:.0f} mm, "
                      f"true {entry['roll_new_anchor']['true_distance_mm']:.0f} mm) -> "
                      f"elbow {entry['roll_new_anchor']['elbow_mm']:.1f} mm / "
                      f"wrist {entry['roll_new_anchor']['wrist_mm']:.1f} mm", flush=True)
                print("    solver self-check (a pair that is exactly reachable): old anchor -> "
                      f"elbow {entry['solver_old_anchor']['elbow_mm']:.1f} mm / "
                      f"wrist {entry['solver_old_anchor']['wrist_mm']:.1f} mm ; rigid anchor -> "
                      f"elbow {entry['solver_new_anchor']['elbow_mm']:.1f} mm / "
                      f"wrist {entry['solver_new_anchor']['wrist_mm']:.1f} mm", flush=True)
                entry["track"] = pose_tracking_test(adapter, side, scene.p, scene.cid)
                for track in entry["track"]:
                    print("    solver tracking a reachable pose (spread %.0f deg): "
                          % track["spread_deg"]
                          + " ; ".join(f"{step['iterations']} iter lock="
                                       f"{'on' if step['lock_others'] else 'off'} -> elbow "
                                       f"{step['elbow_mm']:.1f} mm / wrist "
                                       f"{step['wrist_mm']:.1f} mm"
                                       for step in track["solves"]), flush=True)
                entry["ik_frame"] = ik_frame_test(adapter, side, scene.p, scene.cid)
                frame = entry["ik_frame"]
                for name in ("wrist", "elbow"):
                    step = frame["drive"][name]
                    ident = frame["identity"]["distance_to_target_mm"][name]
                    print(f"    IK end-effector frame ({name} link = "
                          f"{frame['links'][name]}): drives {frame['ik_drives'][name]} "
                          f"(for a {step['intended_mm']:.1f} mm target shift the link "
                          f"frame moved {step['frame']['moved_mm']:.1f} mm / the COM "
                          f"moved {step['com']['moved_mm']:.1f} mm) | COM-vs-frame gap "
                          f"{frame['frame_offset_mm'][name]:.1f} mm | identity (target = "
                          f"current FK) residual from frame "
                          f"{ident['from_frame']:.1f} mm / from COM "
                          f"{ident['from_com']:.1f} mm", flush=True)
                print("    identity solve moved the arm "
                      f"{frame['identity']['joint_max_delta_deg']:.2f} deg (restPoses = load "
                      f"pose) / {frame['identity_current_rest']['joint_max_delta_deg']:.2f} deg "
                      f"(restPoses = current pose) | tightest arm joint limit headroom "
                      f"{frame['min_limit_headroom_deg']} deg", flush=True)
                for name in ("wrist", "elbow"):
                    load = frame["drive"][name]
                    here = frame["drive_current_rest"][name]
                    print(f"    restPoses A/B ({name}): target +{load['intended_mm']:.1f} mm -> "
                          f"link frame moved {load['frame']['moved_mm']:.1f} mm with the load-pose "
                          f"restPoses / {here['frame']['moved_mm']:.1f} mm with the current-pose "
                          f"one (identity residual "
                          f"{frame['identity']['distance_to_target_mm'][name]['from_frame']:.1f} -> "
                          f"{frame['identity_current_rest']['distance_to_target_mm'][name]['from_frame']:.1f} "
                          f"mm)", flush=True)

                for name, delta in sorted(entry["joints"].items(),
                                          key=lambda item: -abs(item[1]["d_upper_mm"])):
                    if abs(delta["d_upper_mm"]) < 1.0 and abs(delta["d_fore_mm"]) < 1.0:
                        continue
                    print(f"    {name:34s} d|elbow-shoulder|="
                          f"{delta['d_upper_mm']:+8.2f} mm  d|wrist-elbow|="
                          f"{delta['d_fore_mm']:+8.2f} mm  d|elbow-rigid|="
                          f"{delta['d_builtin_mm']:+7.2f} mm  (at "
                          + (f"{delta['at_deg']:+.0f}" if delta["at_deg"] is not None
                             else "n/a") + " deg)")
    finally:
        report["scene"] = scene.summary() if hasattr(scene, "summary") else None
        scene.close()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
