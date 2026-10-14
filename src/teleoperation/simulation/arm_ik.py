"""Per-robot PyBullet arm IK: drive whichever arm the scene actually loaded.

Our simulation-side replacement for the TRON2A-only path in
``retargeting/arm/ik.py``.  The teammate's module stays untouched (rule 3); the
choice of arm happens *here*, keyed by ``robot_type``:

    profile = profile_for(robot_type)           # simulation/arm_profiles.py
    ik = PyBulletArmIK(body, cid, robot_type)   # this file
    ik.solve("right", target)                   # -> {joint name: angle}
    ik.apply("right", values)                   # -> POSITION_CONTROL that arm

The recipe is the one measured on 2026-10-10 (steady-state wrist residual 10-22 mm
on all three paper robots; see docs/multi-robot-arm.md):

1. the target is given in the robot's **base frame** and converted to world with
   the body's own base pose -- the base sits ``base_z`` metres above the plane, so
   a base-frame target that skips the conversion lands 866 mm low on H1-2;
2. ``calculateInverseKinematics`` runs with the model's own limits/ranges/rest
   poses (rest poses = the current joint angles, so the redundant 7 DoF arm stays
   near where it already is);
3. ``ik_mode`` comes from the profile: ``"pose"`` keeps the orientation constraint
   (h1_2 only), ``"position"`` drops it (gr1_t2 has no wrist link at all, and g1's
   wrist tracks the uncalibrated orientation convention poorly);
4. only that side's arm joints are written; every other joint stays under the
   scene's hold motors (see ``applications/arm_scene_driver.py``), so legs, waist
   and head do not drift.

What this is not: a hand-eye calibration.  The operator-to-robot mapping is the
per-robot tuning in ``arm_profiles.py``.  The orientation convention is not
calibrated either, which is why ``ik_mode="position"`` drops the constraint on the
two robots where the absolute convention did not hold up.  The way out is to stop
using an absolute orientation at all: ``solve(..., orientation_frame="neutral")``
takes the caller's quaternion as a **rotation relative to the build-time wrist
pose** (``self.neutral_orientation``) and composes

    R_target = R_neutral_ee * R_relative          (quaternion_multiply below)

That is the paper's "wrist 4x4 target" with the uncalibrated part factored out:
whatever the operator-to-robot axis convention is, it cancels inside the relative
rotation, and each robot gets its own neutral -- measured 2026-10-12, see
``docs/multi-robot-arm.md`` section 8.
"""

from __future__ import annotations

from dataclasses import replace
from math import acos, degrees, pi

import numpy as np

from teleoperation.simulation.arm_profiles import (
    SIDES,
    ArmProfile,
    profile_for,
    resolve_joint_names,
)

PHYSICS_DT = 1.0 / 240.0
ARM_FORCE = 200.0
IK_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5
#: 手臂链里的定位关键字：肩（上臂起点）与肘。三台的命名都含这两个词（实测 2026-10-12）。
SHOULDER_KEYWORD = "shoulder"
ELBOW_KEYWORD = "elbow"
#: 姿态统计里两两点积/距离的取样上限：长会话按步长抽稀，避免 O(n^2) 爆内存。
POSTURE_MAX_FRAMES = 600
#: ``solve(orientation_frame=...)`` 的两种口径。
ORIENTATION_FRAMES = ("absolute", "neutral")
#: 同时给了腕与肘两个位置目标时，肘任务放在第几遍：
#:
#: ``first``  第一遍就是 IK2（腕 + 肘），第二遍按 ``ik_mode`` 用腕的位姿收尾；
#: ``last``   前两遍只做腕，最后再交一次 IK2 把两个位置目标一起收住。
#:
#: 为什么要这个开关（实测 2026-10-13，``devtools/probe_elbow_task.py``）：``first`` 时第二遍
#: 的「腕姿态」收尾会把 IK2 摆好的肘拉回去（肘误差 185-256 mm、腕残差 107 mm），而 ``last``
#: 结束时两个位置目标同时满足（数字见 docs/multi-robot-arm.md 第 8.10 节）。
ELBOW_PASS_ORDERS = ("first", "last")
#: 肘位置任务的「支路恢复」阈值（**度**，判断量 = 前臂方向误差 + 0.5*上臂方向误差）：
#: 当前解的这两个方向误差超过它，就从加载姿势再解一遍，谁好要谁（0 = 关）。见
#: ``PyBulletArmIK._recover_elbow_branch``。
#:
#: 为什么需要它（2026-10-13 现场反馈「GR1-T2 前臂弯曲方向和我相反」，本机 200 帧回放实测）：
#: PyBullet 的 IK 从**当前**关节角出发，代价里除了位置还有「离 restPoses 多远」；某一帧把肘
#: 送进反向支路（肘在肩-腕连线另一侧）后，那个解离当前姿势更近，于是**每一帧都从那里再解**、
#: 再也出不来 —— gr1_t2 左臂 91% 的帧肘方向误差 > 30 度、29% 的帧前臂与要求方向夹角 > 90 度
#: （最差那一帧要的肘方向 (-0.14, +0.91, -0.39)、走到的却是 (+0.88, +0.31, +0.37)：肘甩到了
#: 腕的前面）。同一份数据 h1_2 右臂只有 5 度误差，所以不是目标错，是这一支路出不去。
#:
#: 判据是**方向**而不是毫米：第一次实现用「肘毫米 + 0.25*腕毫米」评分支路，g1 左臂从
#: 56.98 度降到 16.21 度（腕残差 245.6 -> 96.9 mm），但 gr1_t2 反而更差（前臂误差 72.8 ->
#: 85.5 度、反向前臂帧占比 0.34 -> 0.49）—— 毫米能同时被「肘近 30 mm、腕近 30 mm」满足，
#: 而人看到的是**方向**。改成方向后两个判据同一条，现场反馈的那一项就是判据本身。
ELBOW_RECOVERY_DEG = 25.0


def quaternion_multiply(left, right) -> tuple:
    """``left * right``，PyBullet 口径 ``(x, y, z, w)``：先转 ``right`` 再转 ``left``。

    用来把「相对中性」的姿态目标放回绝对姿态：``R(q_l * q_r) = R(q_l) @ R(q_r)``，
    等价于 PyBullet 的 ``multiplyTransforms([0,0,0], q_l, [0,0,0], q_r)[1]``（实测过）。
    """
    x1, y1, z1, w1 = (float(value) for value in left)
    x2, y2, z2, w2 = (float(value) for value in right)
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def _unit_difference(end, start):
    """``unit(end - start)``，两点重合（退化）时返回 ``None`` 而不是抛异常。"""
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-9 else vector / norm


def _direction_angle_deg(left, right) -> float:
    """两个单位向量的夹角（度）。"""
    dot = float(np.clip(np.dot(np.asarray(left, dtype=np.float64),
                               np.asarray(right, dtype=np.float64)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _percentile_stats(values) -> dict:
    """``{"median": .., "p90": ..}``；空输入给 ``None``（报告里必须能看出「没量到」）。"""
    if not values:
        return {"median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {"median": round(float(np.median(array)), 2),
            "p90": round(float(np.percentile(array, 90)), 2)}


def movable_joints(p, body, cid):
    """``(indices, infos)`` of every non-fixed joint, in PyBullet order."""
    indices, infos = [], []
    for index in range(p.getNumJoints(body, physicsClientId=cid)):
        info = p.getJointInfo(body, index, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED:
            continue
        indices.append(index)
        infos.append(info)
    return indices, infos


def joint_name(info) -> str:
    """Joint name of a ``getJointInfo`` record (bytes on older PyBullet builds)."""
    name = info[1]
    return name.decode() if isinstance(name, bytes) else str(name)


def link_name(info) -> str:
    """Child link name of a ``getJointInfo`` record."""
    name = info[12]
    return name.decode() if isinstance(name, bytes) else str(name)


def end_effector_index(p, body, side: str, cid, keywords=("wrist", "hand")):
    """``(index, link_name)`` of that side's hand mount (wrist/hand link).

    Naming differs per robot (measured 2026-10-10):
        h1_2   right_wrist_yaw_link      (arms end at the wrist)
        g1     right_wrist_yaw_link      (hand links exist but come later)
        gr1_t2 right_hand_pitch_link     (no wrist links at all)
    So: the last ``{side}_*keyword*`` link, keyword by keyword in the order the
    profile gives (most specific first).
    """
    for keyword in keywords:
        candidates = []
        for index in range(p.getNumJoints(body, physicsClientId=cid)):
            info = p.getJointInfo(body, index, physicsClientId=cid)
            name = link_name(info).lower()
            if name.startswith(side + "_") and keyword in name:
                candidates.append((index, link_name(info)))
        if candidates:
            return max(candidates)
    raise RuntimeError(
        f"no {side} link matching {list(keywords)} found on robot body {body}")




class PyBulletArmIK:
    """IK adapter for one loaded robot body: the arm follows, everything else is held.

    The body belongs to somebody else's client (``ThreeRobotHandScene``), so this
    class never loads URDFs, never sets gravity and never disconnects: it reads the
    base pose, solves IK and writes POSITION_CONTROL on that side's arm joints only.
    All other joints stay under the scene's own hold motors.

    The one thing it does touch is the arm itself: ``solve`` snaps that arm onto its
    first (position-only) solution with ``resetJointState`` so the second pass starts
    warm -- a single cold pass stalls 3-10x further from the target (see ``solve``).
    """

    def __init__(self, body, cid, robot_type=None, sides=SIDES, profile=None,
                 ik_mode=None, arm_force=ARM_FORCE, ik_iterations=IK_ITERATIONS,
                 residual_threshold=IK_RESIDUAL_THRESHOLD,
                 rest_poses: str = "current",
                 elbow_recovery_deg: float = ELBOW_RECOVERY_DEG):
        import pybullet as p

        if profile is None:
            if robot_type is None:
                raise ValueError("pass robot_type or profile")
            profile = profile_for(robot_type)
        if ik_mode is not None:
            profile = replace(profile, ik_mode=ik_mode)
        if rest_poses not in ("current", "zero"):
            raise ValueError(
                f"rest_poses must be 'current' or 'zero', got {rest_poses!r}")
        self.p = p
        self.body = body
        self.cid = int(cid)
        self.profile = profile
        self.robot_type = profile.robot_type
        self.ik_mode = profile.ik_mode
        self.rest_poses = rest_poses
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.arm_force = float(arm_force)
        self.ik_iterations = int(ik_iterations)
        self.residual_threshold = float(residual_threshold)

        # Every non-fixed joint in PyBullet order: the IK solution has one entry per
        # movable joint, so the mapping has to stay name-based (index 0 is a hip).
        self.indices, self.infos = movable_joints(p, body, self.cid)
        self.index_of = {joint_name(info): index
                         for index, info in zip(self.indices, self.infos)}
        self.name_of = {index: joint_name(info)
                        for index, info in zip(self.indices, self.infos)}

        self.arm = {}
        for side in self.sides:
            names = resolve_joint_names(profile, body, self.cid, side)
            missing = [name for name in names if name not in self.index_of]
            if missing:
                raise RuntimeError(
                    f"{self.robot_type}: {side} arm joints are not on the loaded "
                    f"body: {missing}")
            if len(names) != profile.expected_arm_dof:
                raise RuntimeError(
                    f"{self.robot_type}: {side} arm resolved to {len(names)} joints, "
                    f"expected {profile.expected_arm_dof}: {list(names)}")
            self.arm[side] = [(self.index_of[name], name) for name in names]
        self.arm_ids = {side: [index for index, _name in entries]
                        for side, entries in self.arm.items()}
        self.driven = {index for indices in self.arm_ids.values() for index in indices}
        self.ee = {side: end_effector_index(p, body, side, self.cid,
                                            profile.ee_keywords)
                   for side in self.sides}
        #: 「上臂起点（肩）」与「肘」的连杆引用。PyBullet 里子连杆的索引就是关节索引
        #: （实测 2026-10-12：h1_2 的 left_shoulder_pitch_joint 到 left_elbow_pitch_joint
        #: 量出 331 mm，正是整条上臂），所以直接拿关节索引当连杆索引用。
        self.shoulder_link = {side: self._limb_link(side, SHOULDER_KEYWORD)
                              for side in self.sides}
        self.elbow_link = {side: self._limb_link(side, ELBOW_KEYWORD)
                           for side in self.sides}
        #: ``record_posture()`` 每帧采样的姿势（每侧一串 dict），赛后统计肩/肘动没动。
        self.posture = {side: [] for side in self.sides}

        self.limits = self._joint_limits()
        #: Angle of every movable joint when this adapter was built (drift reference).
        self.hold = {index: float(p.getJointState(body, index,
                                                  physicsClientId=self.cid)[0])
                     for index in self.indices}
        #: Wrist orientation at build time; the default pose target keeps it.
        self.neutral_orientation = {side: self._orientation(side) for side in self.sides}
        #: Position of each movable joint inside a PyBullet IK solution vector.
        self.position_of = {index: position
                            for position, index in enumerate(self.indices)}
        self.last_target_world = {}
        self.last_orientation = {}
        self.last_posture = {}
        #: 这一帧交给「肘位置任务」的世界坐标点（``None`` = 没给，走单目标那条路）。
        self.last_elbow_world = {}
        #: 肘支路恢复（``_recover_elbow_branch``）：阈值（度，0 = 关）与每侧救回来的次数。
        self.elbow_recovery_deg = max(0.0, float(elbow_recovery_deg))
        self.elbow_recoveries = {}
        #: 每侧「算出更差 / 手跑掉」被拒的次数（报告用：说明这条路试了多少次、成了几次）。
        self.branch_rejections = {}
        #: 每侧最后一次恢复的前后数字（报告用；没恢复过就没有这个键）。
        self.last_branch_recovery = {}


    # ------------------------------------------------------------------ #
    def _joint_limits(self) -> dict:
        """IK bounds (lower / upper / range / rest) for every movable joint.

        A joint whose URDF limits are missing or reversed (PyBullet reports 0 / -1
        for continuous joints) gets a full turn instead -- same widening the
        validated probe uses.  ``rest_poses="current"`` starts the redundant 7 DoF
        chain from where it already is, which keeps the solution continuous from
        frame to frame; ``"zero"`` reproduces the probe's original rest poses.
        """
        p = self.p
        lower, upper, ranges, rest = [], [], [], []
        for index, info in zip(self.indices, self.infos):
            low, high = float(info[8]), float(info[9])
            if not (low == low and high == high) or high <= low:
                low, high = -2.0 * pi, 2.0 * pi
            lower.append(low)
            upper.append(high)
            ranges.append(max(high - low, 1e-6))
            if self.rest_poses == "current":
                rest.append(float(p.getJointState(self.body, index,
                                                  physicsClientId=self.cid)[0]))
            else:
                rest.append(0.0)
        return {"lower": lower, "upper": upper, "range": ranges, "rest": rest}

    def _orientation(self, side: str) -> tuple:
        state = self.p.getLinkState(self.body, self.ee[side][0],
                                    computeForwardKinematics=True,
                                    physicsClientId=self.cid)
        return tuple(float(value) for value in state[5])

    def _limb_link(self, side: str, keyword: str) -> int:
        """这一侧手臂里名字含 ``keyword`` 的第一个关节索引（= 它的子连杆索引）。

        名字对不上就直接报错：三台（h1_2 / g1 / gr1_t2）的肩与肘关节名都含这两个词，
        所以对不上说明 URDF 换了名字，姿态度量不该悄悄退化成「量别的东西」。
        """
        for index, name in self.arm[side]:
            if keyword in name.lower():
                return int(index)
        raise RuntimeError(
            f"{self.robot_type}/{side}: no arm joint matching {keyword!r} in "
            f"{[name for _index, name in self.arm[side]]}")

    def link_position(self, link_index: int) -> tuple:
        """World position of a link (``link_index`` = the joint that owns it)."""
        state = self.p.getLinkState(self.body, int(link_index),
                                    computeForwardKinematics=True,
                                    physicsClientId=self.cid)
        return tuple(float(value) for value in state[4])

    def arm_points(self, side: str) -> dict:
        """World ``{shoulder, elbow, wrist}`` of this arm, straight from forward kinematics.

        Used by the posture metrics ("did the shoulder and elbow take part?"), and by
        the driver to measure the left/right forearm gap.
        """
        return {"shoulder": self.link_position(self.shoulder_link[side]),
                "elbow": self.link_position(self.elbow_link[side]),
                "wrist": self.link_position(self.ee[side][0])}

    def record_posture(self, side: str) -> dict:
        """Sample this arm once: every arm joint angle plus the three FK points.

        Sampled *before* the next command is written, so what it reads is the pose the
        previous frame settled into -- the same instant the residual is measured at.

        The command that produced this pose is stored with it (``command_elbow`` /
        ``command_wrist``, both in this robot's **base** frame).  A pose alone cannot
        say whether the arm went the way it was asked to: 2026-10-13 现场反馈
        「GR1-T2 前臂弯曲方向和我相反」要的就是「要的方向 vs 走到的方向」这一对比，
        见 ``posture_summary`` 的 ``forearm_dir_error_deg`` / ``forearm_dir_dot``。
        """
        entry = {name: float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
                 for index, name in self.arm[side]}
        entry.update(self.arm_points(side))
        elbow_world = self.last_elbow_world.get(side)
        wrist_world = self.last_target_world.get(side)
        if elbow_world is not None and wrist_world is not None:
            entry["command_elbow"] = list(self.to_base(elbow_world))
            entry["command_wrist"] = list(self.to_base(wrist_world))
        self.posture[side].append(entry)
        return entry

    def clear_posture(self) -> None:
        """Drop the recorded posture samples (fresh A/B run)."""
        self.posture = {side: [] for side in self.sides}

    def posture_summary(self, side: str) -> dict:
        """What this arm did over the whole session: shoulder, upper arm, elbow.

        The paper does not constrain the elbow (it says the neutral regulariser plus the
        joint limits let it fall out); the symptom we are chasing is the opposite -- a
        shoulder that does not move and a forearm that folds onto the midline -- so the
        numbers here are chosen to expose exactly that, all of them needing *only* the
        robot (they work on a ``--input-npz`` replay too):

        ``shoulder_range_deg``       per shoulder joint: max - min joint angle (0 = frozen)
        ``shoulder_travel_deg``      the largest of the above (the headline number)
        ``upper_arm_dir_travel_deg`` how wide the upper-arm direction swept (2-point angle)
        ``elbow_span_mm``            how far the elbow travelled (max distance between samples)
        ``elbow_bend_deg``           shoulder-elbow-wrist angle, min/max
        ``upper_dir_error_deg``      asked vs achieved **upper-arm** direction (median / p90)
        ``forearm_dir_error_deg``    asked vs achieved **forearm** direction (median / p90)
        ``forearm_dir_dot``          median dot(achieved, asked) **per base-frame axis**: a
                                     负的某轴就是「这一轴反了」（现场 2026-10-13「前臂弯曲方向
                                     和我相反」的直接证据），X/Y/Z 分开看才知道是哪一根
        ``forearm_opposite_fraction``  夹角 > 90 度的帧占比（1.0 = 整场都在反向）
        """
        from teleoperation.simulation.arm_realtime import elbow_bend_degrees

        frames = self.posture[side]
        empty = {"frames": len(frames), "shoulder_range_deg": {}, "shoulder_travel_deg": 0.0,
                 "upper_arm_dir_travel_deg": 0.0, "elbow_span_mm": 0.0,
                 "elbow_bend_deg": {"min": None, "max": None},
                 "command_frames": 0, "upper_dir_error_deg": _percentile_stats([]),
                 "forearm_dir_error_deg": _percentile_stats([]),
                 "upper_struggle_fraction": None, "forearm_dir_dot": {},
                 "forearm_opposite_fraction": None, "worst_frame": None,
                 "elbow_recoveries": 0, "branch_rejections": 0,
                 "last_branch_recovery": None}
        if len(frames) < 2:
            return empty
        step = max(1, len(frames) // POSTURE_MAX_FRAMES)
        sampled = frames[::step]
        ranges = {}
        for index, name in self.arm[side]:
            if SHOULDER_KEYWORD not in name.lower():
                continue
            values = np.asarray([frame[name] for frame in frames], dtype=np.float64)
            ranges[name] = round(float(np.degrees(values.max() - values.min())), 2)
        shoulders = np.asarray([frame["shoulder"] for frame in sampled], dtype=np.float64)
        elbows = np.asarray([frame["elbow"] for frame in sampled], dtype=np.float64)
        upper = elbows - shoulders
        norms = np.linalg.norm(upper, axis=1)
        direction_travel = 0.0
        if np.all(norms > 1e-9):
            gram = (upper / norms[:, None]) @ (upper / norms[:, None]).T
            direction_travel = float(np.degrees(np.arccos(np.clip(gram.min(), -1.0, 1.0))))
        span = float(np.linalg.norm(elbows[:, None, :] - elbows[None, :, :], axis=-1).max()) * 1000.0
        bends = [elbow_bend_degrees(points["shoulder"], points["elbow"], points["wrist"])
                 for points in sampled]
        bends = [value for value in bends if np.isfinite(value)]
        # 目标 vs 走到的（两段方向，基座系）：方向口径只对「方向」负责，所以这里比对方向而不是
        # 腕点。采样是在写下一帧命令**之前**做的，所以 frame 里的 command_* 正是这一帧姿势
        # 对应的那条命令（见 record_posture）。
        upper_errors, fore_errors, fore_dots, opposite = [], [], [], 0
        struggled = 0
        worst = None
        for sampled_index, frame in enumerate(sampled):
            command_elbow = frame.get("command_elbow")
            command_wrist = frame.get("command_wrist")
            if command_elbow is None or command_wrist is None:
                continue
            shoulder = self.to_base(frame["shoulder"])
            elbow = self.to_base(frame["elbow"])
            wrist = self.to_base(frame["wrist"])
            achieved_u = _unit_difference(elbow, shoulder)
            achieved_f = _unit_difference(wrist, elbow)
            wanted_u = _unit_difference(command_elbow, shoulder)
            wanted_f = _unit_difference(command_wrist, command_elbow)
            if achieved_u is None or wanted_u is None:
                continue
            upper_error = _direction_angle_deg(achieved_u, wanted_u)
            upper_errors.append(upper_error)
            if upper_error > 30.0:
                struggled += 1
            if achieved_f is None or wanted_f is None:
                continue
            forearm_error = _direction_angle_deg(achieved_f, wanted_f)
            fore_errors.append(forearm_error)
            fore_dots.append(achieved_f * wanted_f)
            if forearm_error > 90.0:
                opposite += 1
            # 最差那一帧的原样向量：报告里能直接看出「要的是哪边、走到的是哪边」，
            # 不用再猜是符号错了还是根本到不了（2026-10-13 现场反馈的排障入口）。
            if worst is None or forearm_error > worst["forearm_error_deg"]:
                worst = {"sampled_frame": int(sampled_index), "forearm_error_deg": round(forearm_error, 2),
                         "upper_error_deg": round(upper_error, 2),
                         "wanted_fore": [round(float(value), 3) for value in wanted_f],
                         "achieved_fore": [round(float(value), 3) for value in achieved_f],
                         "wanted_upper": [round(float(value), 3) for value in wanted_u],
                         "achieved_upper": [round(float(value), 3) for value in achieved_u],
                         "elbow_target_m": [round(float(value), 3) for value in command_elbow],
                         "wrist_target_m": [round(float(value), 3) for value in command_wrist],
                         "shoulder_m": [round(float(value), 3) for value in shoulder]}
        dots = np.asarray(fore_dots, dtype=np.float64).reshape(-1, 3) if fore_dots else None
        return {"frames": len(frames),
                "shoulder_range_deg": ranges,
                "shoulder_travel_deg": round(max(ranges.values()) if ranges else 0.0, 2),
                "upper_arm_dir_travel_deg": round(direction_travel, 2),
                "elbow_span_mm": round(span, 2),
                "elbow_bend_deg": ({"min": round(min(bends), 1), "max": round(max(bends), 1)}
                                   if bends else {"min": None, "max": None}),
                "command_frames": len(fore_errors),
                "upper_dir_error_deg": _percentile_stats(upper_errors),
                "forearm_dir_error_deg": _percentile_stats(fore_errors),
                "upper_struggle_fraction": (None if not upper_errors
                                            else round(struggled / len(upper_errors), 3)),
                "forearm_dir_dot": ({} if dots is None else
                                    {name: round(float(np.median(dots[:, index])), 3)
                                     for index, name in enumerate("XYZ")}),
                "forearm_opposite_fraction": (None if not fore_errors
                                              else round(opposite / len(fore_errors), 3)),
                "worst_frame": worst,
                #: 这一侧救回过多少次反向支路（``_recover_elbow_branch``），以及最后一次的前后数字。
                "elbow_recoveries": int(self.elbow_recoveries.get(side, 0)),
                "branch_rejections": int(self.branch_rejections.get(side, 0)),
                "last_branch_recovery": self.last_branch_recovery.get(side)}

    def base_position(self) -> tuple:
        """World position of the loaded body's base."""
        base, _orientation = self.p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)
        return tuple(float(value) for value in base)

    def to_world(self, position) -> tuple:
        """Robot base-frame position -> world (base translation + base rotation).

        The rotation matters: GR1-T2 does not load with an identity base rotation.
        """
        values = tuple(float(value) for value in position)
        if len(values) != 3:
            raise ValueError(f"position must be (x, y, z), got {position!r}")
        _base, orientation = self.p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)
        rotation = self.p.getMatrixFromQuaternion(orientation)
        offset = [rotation[3 * row] * values[0] + rotation[3 * row + 1] * values[1]
                  + rotation[3 * row + 2] * values[2] for row in range(3)]
        base = self.base_position()
        return tuple(base[axis] + offset[axis] for axis in range(3))

    def to_base(self, position) -> tuple:
        """World position -> this robot's base frame (the inverse of ``to_world``).

        Needed because the elbow / wrist anchors a profile works with are base-frame
        points, while ``arm_points`` reports forward-kinematics points in world.
        """
        values = tuple(float(value) for value in position)
        if len(values) != 3:
            raise ValueError(f"position must be (x, y, z), got {position!r}")
        base, orientation = self.p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)
        inverse_position, inverse_orientation = self.p.invertTransform(base, orientation)
        local, _orientation = self.p.multiplyTransforms(
            inverse_position, inverse_orientation, list(values), [0.0, 0.0, 0.0, 1.0])
        return tuple(float(value) for value in local)

    def target_in_world(self, position, source: str = "base") -> tuple:
        """Resolve a target to world coordinates, honouring ``source``.

        ``"base"``     already in the robot's own base frame (per-robot reach);
        ``"operator"`` operator-space metres -> profile scale/offset -> base frame;
        ``"world"``    world coordinates, used as given.
        """
        if source == "world":
            return tuple(float(value) for value in position)
        if source == "operator":
            return self.to_world(self.profile.base_target_from_operator(position))
        if source == "base":
            return self.to_world(position)
        raise ValueError(
            f"source must be 'base', 'operator' or 'world', got {source!r}")


    # ------------------------------------------------------------------ #
    def _rest_poses(self, posture=None) -> list:
        """Null-space reference for the IK call: the build-time neutral, or a posture.

        PyBullet has no ``PostureTask``, but it does resolve the redundant 7 DoF chain
        by pulling the solution toward ``restPoses`` (the same null-space term
        ``rest_poses="current"`` uses to stay near the load-time neutral).  Handing in
        the *operator's* arm angles therefore turns that term into the paper's posture
        task: the wrist task still decides the position, the null-space term decides
        where the elbow / upper arm sit while doing it.

        ``posture`` is ``{joint name: angle}`` and may be partial; every name has to be
        a movable joint of this body, so a typo fails here instead of silently leaving
        the elbow where it was.
        """
        rest = list(self.limits["rest"])
        if not posture:
            return rest
        for name, value in dict(posture).items():
            index = self.index_of.get(str(name))
            if index is None:
                raise ValueError(
                    f"{self.robot_type}: posture joint {name!r} is not a movable joint "
                    f"of the loaded body")
            rest[self.position_of[index]] = float(value)
        return rest


    def _ik_call(self, side: str, world, orientation, arguments) -> list:
        """One ``calculateInverseKinematics`` call: world target -> full solution."""
        p = self.p
        if orientation is None:
            solution = p.calculateInverseKinematics(
                self.body, self.ee[side][0], world, **arguments)
        else:
            solution = p.calculateInverseKinematics(
                self.body, self.ee[side][0], world, orientation, **arguments)
        values = [float(value) for value in solution]
        if len(values) != len(self.indices):
            raise RuntimeError(
                f"IK returned {len(values)} values for {len(self.indices)} movable "
                f"joints (robot_type={self.robot_type})")
        return values

    def _ik_call2(self, side: str, world, elbow_world, arguments) -> list:
        """Two position targets in one call: the wrist *and* the elbow.

        This is the paper's ``FrameTask(手腕)`` + 肘部位置任务 的本机等价物：一条 7 自由度
        冗余链只给腕一个位置时，求解器会把整条臂留在原地（实测：只动大臂那一帧机器人纹丝
        不动），把**肘**也作为一个位置目标交进去，肩就必须真的转起来。姿态（orientation）
        只有腕能用，所以这一路是纯位置的；``solve`` 随后还会用腕的姿态做第二遍收尾。
        """
        p = self.p
        if not hasattr(p, "calculateInverseKinematics2"):
            raise RuntimeError(
                "this pybullet build has no calculateInverseKinematics2; the elbow "
                "position task needs it (upgrade pybullet or drop the elbow target)")
        solution = p.calculateInverseKinematics2(
            self.body,
            [self.ee[side][0], self.elbow_link[side]],
            [world, elbow_world],
            **arguments)
        values = [float(value) for value in solution]
        if len(values) != len(self.indices):
            raise RuntimeError(
                f"IK2 returned {len(values)} values for {len(self.indices)} movable "
                f"joints (robot_type={self.robot_type})")
        return values

    def _arm_values(self, side: str, solution) -> dict:
        return {name: solution[self.position_of[index]]
                for index, name in self.arm[side]}

    def _teleport(self, side: str, solution) -> None:
        """Snap this side's arm joints to a solution (no motors, no physics step).

        Between the two IK passes: PyBullet starts from the body's *current* angles,
        so handing the position solution back as the state is what makes the second
        pass converge (measured 2026-10-12).
        """
        for index, _name in self.arm[side]:
            self.p.resetJointState(self.body, index,
                                   float(solution[self.position_of[index]]),
                                   physicsClientId=self.cid)

    def _branch_cost(self, side: str, world, elbow_world) -> dict:
        """Current pose vs this frame's two targets, as the operator perceives it.

        The arm must already be **on** the solution being scored (``solve`` teleports
        before calling this): PyBullet's IK is not analytic here, so the only way to
        score a candidate is to put the arm there and read forward kinematics.

        ``cost_deg`` = 前臂方向误差 + 0.5*上臂方向误差（度）——这一帧「看起来像不像」的那两个数；
        ``elbow_mm`` / ``wrist_mm`` 照记，便于报告里区分「方向对但点差一点」和「方向就不对」。
        """
        points = self.arm_points(side)
        shoulder = np.asarray(points["shoulder"], dtype=np.float64)
        elbow = np.asarray(points["elbow"], dtype=np.float64)
        wrist = np.asarray(points["wrist"], dtype=np.float64)
        cost = {"elbow_mm": round(float(np.linalg.norm(
            elbow - np.asarray(elbow_world, dtype=np.float64))) * 1000.0, 3),
            "wrist_mm": round(float(np.linalg.norm(
                wrist - np.asarray(world, dtype=np.float64))) * 1000.0, 3)}
        achieved_u = _unit_difference(elbow, shoulder)
        achieved_f = _unit_difference(wrist, elbow)
        wanted_u = _unit_difference(elbow_world, shoulder)
        wanted_f = _unit_difference(world, elbow_world)
        upper = (None if achieved_u is None or wanted_u is None
                 else _direction_angle_deg(achieved_u, wanted_u))
        fore = (None if achieved_f is None or wanted_f is None
                else _direction_angle_deg(achieved_f, wanted_f))
        cost["upper_deg"] = None if upper is None else round(upper, 2)
        cost["forearm_deg"] = None if fore is None else round(fore, 2)
        cost["cost_deg"] = (None if fore is None else
                            round(fore + 0.5 * (upper or 0.0), 2))
        return cost

    def _ik_sequence(self, side, world, elbow_world, orientation, arguments,
                     elbow_pass, warm_start):
        """The historical pass sequence of ``solve``: ``(solution, target_orientation)``.

        Split out so the branch recovery can run the *same* sequence from the load
        pose.  Every pass teleports the arm before the next one (see ``_teleport``).
        """
        solution = (self._ik_call(side, world, None, arguments) if elbow_world is None
                    else self._ik_call2(side, world, elbow_world, arguments))
        target_orientation = None
        if self.ik_mode == "pose":
            if warm_start:
                self._teleport(side, solution)
                target_orientation = (self._orientation(side) if orientation is None
                                      else tuple(float(v) for v in orientation))
            else:
                target_orientation = (self.neutral_orientation[side]
                                      if orientation is None
                                      else tuple(float(v) for v in orientation))
            solution = self._ik_call(side, world, target_orientation, arguments)
        elif warm_start:
            # Position-only profiles still benefit: the second pass starts on target.
            self._teleport(side, solution)
            solution = self._ik_call(side, world, None, arguments)
        if elbow_world is not None and elbow_pass == "last":
            # 肘任务放在最后一遍：IK1 的收尾目标是「腕」，它会把 IK2 刚摆好的肘又拉回去
            # （实测 2026-10-13：肘误差 185-256 mm、腕残差 107 mm）。给了肘目标又要求最后一遍
            # 仍是腕的姿态时，就只能再交一次 IK2 把两个位置目标一起收住。
            self._teleport(side, solution)
            solution = self._ik_call2(side, world, elbow_world, arguments)
        return solution, target_orientation

    def _recover_elbow_branch(self, side, solution, world, elbow_world, orientation,
                              arguments, elbow_pass, warm_start):
        """肘落进**反向支路**时，从加载姿势再解一遍同一条序列，谁好要谁。

        为什么需要（2026-10-13 现场反馈 + 本机 200 帧回放实测，数字见
        ``ELBOW_RECOVERY_MM`` 与 docs/multi-robot-arm.md 第 8.12 节）：PyBullet 的 IK 从
        **当前**关节角出发，代价里除了位置还有「离 restPoses 多远」。某一帧把肘送进反向
        支路后，那个解离当前姿势更近，于是每一帧都从那里再解、再也出不来 —— 现场看到的就是
        「前臂弯曲方向和我的相反」。同一份数据里 h1_2 右臂只有 5 度误差，所以不是目标错。

        加载姿势（``restPoses``）是直臂下垂，两支在这里几乎重合，所以从它再解一遍会落到
        与当前姿势无关的那一支。两支都用**方向**量化（前臂 + 0.5*上臂，度），留小的那个；
        ``elbow_recovery_deg = 0`` 关掉整条路（A/B 用，行为与历史完全一致）。
        """
        self._teleport(side, solution)
        current = self._branch_cost(side, world, elbow_world)
        if (self.elbow_recovery_deg <= 0.0 or current["cost_deg"] is None
                or current["cost_deg"] <= self.elbow_recovery_deg):
            return solution
        self._teleport(side, self._rest_poses(None))
        candidate, _orientation = self._ik_sequence(side, world, elbow_world, orientation,
                                                    arguments, elbow_pass, warm_start)
        self._teleport(side, candidate)
        fresh = self._branch_cost(side, world, elbow_world)
        limit = max(1.5 * current["wrist_mm"], current["wrist_mm"] + 60.0)
        if (fresh["cost_deg"] is None or fresh["cost_deg"] > current["cost_deg"] - 1.0
                or fresh["wrist_mm"] > limit):
            # 新的一支没好 1 度以上，或者「方向好一点、手却跑掉」-> 不动。
            # 1 度门槛：两支几乎一样时别每帧来回换；腕残差护栏：第一版（只比方向）把 h1_2
            # 左臂的腕残差从 139 mm 抬到 457 mm、两前臂最小间隙掉到 5.18 mm（报告
            # ``arm_forearm_gap_mm``），那是操作者会看见的坏账，不能拿它换 4 度方向。
            self._teleport(side, solution)
            self.branch_rejections[side] = self.branch_rejections.get(side, 0) + 1
            return solution
        self.elbow_recoveries[side] = self.elbow_recoveries.get(side, 0) + 1
        self.last_branch_recovery[side] = {
            "before": current, "after": fresh,
            "cost_gain_deg": round(current["cost_deg"] - fresh["cost_deg"], 2),
            "count": self.elbow_recoveries[side]}
        return candidate

    def solve(self, side: str, position, orientation=None, source: str = "base",
              warm_start: bool = True, orientation_frame: str = "absolute",
              posture=None, elbow=None, elbow_pass: str = "first") -> dict:
        """IK for one side -> ``{joint name: target angle}`` for that arm only.

        Two passes, because one cold pass from a drooped arm stalls in a local
        minimum (measured in this scene at each robot's working point: h1_2 56 mm cold
        vs 18 mm warm, g1 59 vs 43, gr1_t2 46 vs 7):

        1. position-only pass, to get the wrist onto the target;
        2. snap the arm to that solution and solve again -- with the commanded
           orientation for ``ik_mode="pose"``, or, when the caller supplied none,
           with the orientation pass 1 already ended up in (self-consistent pose:
           the wrist keeps pointing where the position-driven arm points it).

        ``position`` is resolved to world through ``source``.  Note that pass 1
        leaves the arm joints at that solution (``resetJointState``); ``apply`` then
        servos from there in the same frame, which is what the realtime loop wants.
        ``warm_start=False`` reproduces the single cold pass, for A/B measurement.

        ``orientation`` is absolute by default.  With ``orientation_frame="neutral"``
        it is instead a rotation relative to this robot's build-time wrist pose, and
        the commanded pose becomes ``neutral_orientation[side] * orientation`` -- the
        neutral-relative framing the bridge emits (see ``arm_realtime.py``).  That is
        what makes one operator target usable by all three robots without a hand-eye
        calibration, because the joint axis convention cancels in the relative
        rotation.  Position-only profiles (``ik_mode="position"``) ignore it either
        way.

        ``posture`` (= ``{joint name: angle}``) replaces the null-space reference for
        this call only (see ``_rest_poses``).  That is what makes the elbow / upper arm
        follow the operator instead of whatever swivel the solver happened to keep, and
        it is the one thing the paper's ``PostureTask`` does that a plain position IK
        does not.  It is recorded in ``last_posture[side]`` for the report.

        ``elbow`` = an optional second position target (same frame as ``position``),
        resolved through the same ``source``.  Measured 2026-10-13: with only the wrist
        tasked, "只动大臂、手留在胸前"这一帧三条链都几乎不动（7 自由度冗余链把解留在原地），
        while ``restPoses``/``posture`` 的零空间authority 只有 7-22%（探针
        ``devtools/probe_posture_task.py``）。把肘当一个位置任务交进去，肩就必须转 —— 这是
        论文 ``FrameTask(腕) + 肘`` 的本机做法。``elbow`` 走的是纯位置的第一遍
        （``calculateInverseKinematics2``），第二遍仍按 ``ik_mode`` 用腕的姿态收尾；实际交给
        求解器的那个世界坐标记在 ``last_elbow_world[side]``。
        """
        if side not in self.arm:
            raise ValueError(
                f"{side!r} is not a driven side of this adapter ({list(self.arm)})")
        if orientation_frame not in ORIENTATION_FRAMES:
            raise ValueError(
                f"orientation_frame must be one of {ORIENTATION_FRAMES}, "
                f"got {orientation_frame!r}")
        if elbow_pass not in ELBOW_PASS_ORDERS:
            raise ValueError(
                f"elbow_pass must be one of {ELBOW_PASS_ORDERS}, "
                f"got {elbow_pass!r}")
        if orientation is not None and orientation_frame == "neutral":
            orientation = quaternion_multiply(self.neutral_orientation[side],
                                              tuple(float(v) for v in orientation))
        world = self.target_in_world(position, source)
        elbow_world = None if elbow is None else self.target_in_world(elbow, source)
        arguments = {
            "lowerLimits": self.limits["lower"],
            "upperLimits": self.limits["upper"],
            "jointRanges": self.limits["range"],
            "restPoses": self._rest_poses(posture),
            "maxNumIterations": self.ik_iterations,
            "residualThreshold": self.residual_threshold,
            "physicsClientId": self.cid,
        }
        solution, target_orientation = self._ik_sequence(
            side, world, elbow_world, orientation, arguments, elbow_pass, warm_start)
        if elbow_world is not None:
            # 反向支路恢复：见 _recover_elbow_branch（0 = 关，行为与历史一致）。
            solution = self._recover_elbow_branch(
                side, solution, world, elbow_world, orientation, arguments,
                elbow_pass, warm_start)
        self.last_target_world[side] = world
        self.last_orientation[side] = target_orientation
        #: What the null-space term was told this frame (empty = the build-time neutral).
        self.last_posture[side] = {str(name): float(value)
                                   for name, value in dict(posture or {}).items()}
        #: The elbow position task this frame (``None`` = only the wrist was tasked).
        self.last_elbow_world[side] = elbow_world
        return self._arm_values(side, solution)

    def apply(self, side: str, values) -> None:
        """Write one side's arm angles: ``{joint name: angle}`` or a URDF-order sequence."""
        if side not in self.arm:
            raise ValueError(
                f"{side!r} is not a driven side of this adapter ({list(self.arm)})")
        if isinstance(values, dict):
            targets = [(index, float(values[name])) for index, name in self.arm[side]
                       if name in values]
        else:
            sequence = [float(value) for value in values]
            if len(sequence) != len(self.arm[side]):
                raise ValueError(
                    f"{self.robot_type}/{side}: got {len(sequence)} values for "
                    f"{len(self.arm[side])} arm joints")
            targets = [(index, value) for (index, _name), value
                       in zip(self.arm[side], sequence)]
        for index, value in targets:
            self.p.setJointMotorControl2(
                self.body, index, self.p.POSITION_CONTROL, targetPosition=value,
                force=self.arm_force, physicsClientId=self.cid)

    def measure(self, side: str) -> tuple:
        """Actual world ``(position, quaternion)`` of that side's hand mount link."""
        state = self.p.getLinkState(self.body, self.ee[side][0],
                                    computeForwardKinematics=True,
                                    physicsClientId=self.cid)
        return (tuple(float(value) for value in state[4]),
                tuple(float(value) for value in state[5]))

    def residual_mm(self, side: str, target_world=None) -> float:
        """Distance (mm) between the target this side was given and where it got to."""
        target = self.last_target_world.get(side) if target_world is None else target_world
        if target is None:
            raise ValueError(f"{side}: no target yet -- call solve() first")
        position, _orientation = self.measure(side)
        return 1000.0 * sum((position[axis] - float(target[axis])) ** 2
                            for axis in range(3)) ** 0.5

    def orientation_error_deg(self, side: str, target_quaternion=None) -> float:
        """Angle (deg) between the commanded orientation and the measured one."""
        target = (self.last_orientation.get(side) if target_quaternion is None
                  else target_quaternion)
        if target is None:
            return 0.0
        _position, measured = self.measure(side)
        difference = self.p.getDifferenceQuaternion(measured, target)
        return degrees(2.0 * acos(min(1.0, abs(float(difference[3])))))

    def joint_values(self, side: str) -> dict:
        """Current angle of every arm joint this adapter drives, by joint name."""
        return {name: float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
                for index, name in self.arm[side]}

    def describe(self) -> dict:
        """JSON-able summary of what this adapter drives (for reports / --list)."""
        return {
            "robot": self.robot_type,
            "ik_mode": self.ik_mode,
            "rest_poses": self.rest_poses,
            "arm_force": self.arm_force,
            "driven_joints": len(self.driven),
            "metric_scale": self.profile.metric_scale,
            "metric_offset": list(self.profile.metric_offset),
            "end_effector": {side: {"index": index, "link": link}
                             for side, (index, link) in self.ee.items()},
            "arm_joints": {side: [name for _index, name in entries]
                           for side, entries in self.arm.items()},
            "limits_from_urdf": True,
        }



