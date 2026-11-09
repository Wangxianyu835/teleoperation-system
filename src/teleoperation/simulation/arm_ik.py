"""Per-robot PyBullet arm IK and bounded motor commands.

Targets can include wrist position, wrist orientation and elbow position.
Temporary joint states used during solving are restored before returning."""

from __future__ import annotations

from dataclasses import replace
from math import acos, degrees, pi, radians

import numpy as np

from teleoperation.simulation.arm_profiles import SIDES, profile_for, resolve_joint_names

PHYSICS_DT = 1.0 / 240.0
ARM_FORCE = 200.0
IK_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5
#: 手臂关节限位容差（rad）：IK 解落在 URDF 边界外这么一点点以内就**夹到边界**，不是丢整帧。
#: 三机 900 帧实测（2026-11-09，h1_2 + g1 + gr1_t2）：全部 209 次限位拒绝的超限量都在
#: 0.1~7.8 mrad，且都只落在腕关节（wrist_yaw / wrist_pitch / wrist_roll），属数值噪声；
#: 按 1e-5 拒绝等于让手臂在这些帧卡住上一姿势。真正不可达的目标不会走这条路 —— 它落在
#: ``pose_position_conflict``（位置残差 50~100 mm），不受本容差影响。
ARM_LIMIT_TOLERANCE = 1e-2
SHOULDER_KEYWORD = "shoulder"
ELBOW_KEYWORD = "elbow"
#: 「肘跟手」（``posture_from_bend``）在加载时量肘关节正方向用的探针角（度）：把该关节拧
#: +PROBE 一次，看「肩-肘-腕夹角」变多少。三台的肘关节正方向约定不一样（h1_2 与 gr1_t2 相反），
#: 所以符号只能量，不能写死 —— 写死就会在某一台上把「跟你一样弯」做成「反着弯」。
ELBOW_POSTURE_PROBE_DEG = 5.0
ELBOW_POSTURE_MIN_SLOPE = 0.2
#: Straight-arm bend boundary for the elbow null-space reference.
BEND_STRAIGHT_DEG = 180.0
#: 姿态统计里两两点积/距离的取样上限：长会话按步长抽稀，避免 O(n^2) 爆内存。
POSTURE_MAX_FRAMES = 600
#: ``solve(orientation_frame=...)`` 的两种口径。
ORIENTATION_FRAMES = ("absolute", "neutral")
#: first places the elbow task before wrist refinement; last places it afterwards.
ELBOW_PASS_ORDERS = ("first", "last")
#: Recovery threshold: forearm direction error + half the upper-arm error, in degrees.
ELBOW_RECOVERY_DEG = 25.0

#: Minimum direction-error improvement required to switch IK branch, in degrees.
ELBOW_RECOVERY_MARGIN_DEG = 8.0

#: Frames between recovery attempts, whether accepted or rejected.
ELBOW_RECOVERY_COOLDOWN_FRAMES = 12


def quaternion_multiply(left, right) -> tuple:
    """Compose xyzw quaternions as left * right (apply right, then left)."""
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


def _spike_stats(values) -> dict:
    """Return median, p90 and maximum statistics for sampled changes."""
    if not values:
        return {"median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"median": round(float(np.median(array)), 2),
            "p90": round(float(np.percentile(array, 90)), 2),
            "max": round(float(array.max()), 2)}


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
    """Resolve this side's wrist/hand mount link index and name from the loaded URDF."""
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

    ``solve`` temporarily seeds the arm with its position solution for a warm
    second pass, then restores every saved position and velocity. Only motors
    and physics stepping may move the displayed arm.
    """

    def __init__(self, body, cid, robot_type=None, sides=SIDES, profile=None,
                 ik_mode=None, arm_force=ARM_FORCE, ik_iterations=IK_ITERATIONS,
                 residual_threshold=IK_RESIDUAL_THRESHOLD,
                 rest_poses: str = "current",
                 elbow_recovery_deg: float = ELBOW_RECOVERY_DEG,
                 elbow_recovery_margin_deg: float = ELBOW_RECOVERY_MARGIN_DEG,
                 elbow_recovery_cooldown: int = ELBOW_RECOVERY_COOLDOWN_FRAMES,
                 lock_others: bool = False):
        import pybullet as p

        if profile is None:
            if robot_type is None:
                raise ValueError("pass robot_type or profile")
            profile = profile_for(robot_type)
        if ik_mode is not None:
            profile = replace(profile, ik_mode=ik_mode)
        if rest_poses not in ("current", "zero", "live", "loaded"):
            raise ValueError(
                f"rest_poses must be 'current', 'zero', 'live' or 'loaded', got {rest_poses!r}")
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
        self.ik_lock_others = bool(lock_others)

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
        self.shoulder_link = {side: self._limb_link(side, SHOULDER_KEYWORD)
                              for side in self.sides}
        self.elbow_link = {side: self._limb_link(side, ELBOW_KEYWORD)
                           for side in self.sides}
        self.upper_link = {side: self._rigid_upper_link(side) for side in self.sides}
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
        self.last_solve = {}
        self.last_posture = {}
        self.elbow_posture = {side: self._measure_elbow_posture(side)
                              for side in self.sides}
        #: 这一帧交给「肘位置任务」的世界坐标点（``None`` = 没给，走单目标那条路）。
        self.last_elbow_world = {}
        #: 肘支路恢复（``_recover_elbow_branch``）：阈值（度，0 = 关）与每侧救回来的次数。
        self.elbow_recovery_deg = max(0.0, float(elbow_recovery_deg))
        self.elbow_recovery_margin_deg = max(0.0, float(elbow_recovery_margin_deg))
        self.elbow_recovery_cooldown = max(0, int(elbow_recovery_cooldown))
        self.elbow_recovery_wait = {}
        self.elbow_recoveries = {}
        #: 每侧「算出更差 / 手跑掉」被拒的次数（报告用：说明这条路试了多少次、成了几次）。
        self.branch_rejections = {}
        #: 每侧最后一次恢复的前后数字（报告用；没恢复过就没有这个键）。
        self.last_branch_recovery = {}
        #: 这一帧**解**（不含物理）离两个目标点差多少：``_branch_cost`` 在把手臂 teleport 到解上
        #: 之后量的那个数，用来把「求解器没兑现」和「物理追不上」分开（见 ``posture_summary``
        #: 的 ``solution_elbow_mm`` / ``elbow_task_mm`` 这对数）。
        self.last_solution_cost = {}


    # ------------------------------------------------------------------ #
    def _joint_limits(self) -> dict:
        """IK bounds (lower / upper / range / rest) for every movable joint.

        Missing or reversed URDF limits receive a full-turn fallback range.
        This builds the initial reference list; _rest_poses resolves the configured
        current/live, loaded or zero reference on each solve.
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

    def _rigid_upper_link(self, side: str) -> int:
        """Return the elbow parent link, used as the rigid upper-arm shoulder anchor."""
        elbow = int(self.elbow_link[side])
        try:
            parent = int(self.p.getJointInfo(self.body, elbow,
                                             physicsClientId=self.cid)[16])
        except Exception:                                    # noqa: BLE001 - 兜底
            parent = 0
        if parent > 0:
            return parent
        candidates = [index for index, name in self.arm[side]
                      if SHOULDER_KEYWORD in name.lower()]
        return int(candidates[-1]) if candidates else elbow

    def arm_points(self, side: str) -> dict:
        """Return world shoulder, rigid upper anchor, elbow and wrist FK points."""
        return {"shoulder": self.link_position(self.shoulder_link[side]),
                "upper": self.link_position(self.upper_link[side]),
                "elbow": self.link_position(self.elbow_link[side]),
                "wrist": self.link_position(self.ee[side][0])}

    def _bend_deg(self, side: str) -> float:
        """这一侧手臂在肘处的夹角（度）：**与 ``arm_realtime.elbow_bend_degrees`` 同一个口径**
        （180 = 完全伸直，0 = 完全折起）。

        为什么要跟它一个口径：``posture_from_bend`` 把操作者的肘弯变化量直接交给肘关节，
        两个数必须能用同一把尺量，否则「你弯了 20 度」会被当成另一个东西。
        """
        points = self.arm_points(side)
        upper = (np.asarray(points["elbow"], dtype=np.float64)
                 - np.asarray(points["upper"], dtype=np.float64))
        lower = (np.asarray(points["elbow"], dtype=np.float64)
                 - np.asarray(points["wrist"], dtype=np.float64))
        norms = (float(np.linalg.norm(upper)), float(np.linalg.norm(lower)))
        if not all(np.isfinite(norms)) or min(norms) <= 0.0:
            return float("nan")
        cosine = float(np.dot(upper, lower)) / (norms[0] * norms[1])
        return float(degrees(acos(max(-1.0, min(1.0, cosine)))))

    def _measure_elbow_posture(self, side: str) -> tuple:
        """Measure elbow joint name and d(shoulder-elbow-wrist angle)/d(joint angle).

        Probe the joint temporarily and restore its state. Return no usable reference
        when the measured slope is too small."""
        entries = dict(self.arm[side])
        try:
            index = int(self.elbow_link[side])
        except (KeyError, TypeError):
            return (None, 0.0, None)
        name = entries.get(index)
        if name is None:
            return (None, 0.0, None)
        info = self.p.getJointInfo(self.body, index, physicsClientId=self.cid)
        if int(info[2]) == int(self.p.JOINT_FIXED):
            return (None, 0.0, None)
        base = float(self.p.getJointState(self.body, index,
                                          physicsClientId=self.cid)[0])
        before = self._bend_deg(side)
        chord = 0.0
        for direction in (1.0, -1.0):
            self.p.resetJointState(
                self.body, index,
                base + direction * radians(ELBOW_POSTURE_PROBE_DEG),
                physicsClientId=self.cid)
            after = self._bend_deg(side)
            if np.isfinite(before) and np.isfinite(after):
                candidate = direction * (float(after) - float(before)) / \
                    float(ELBOW_POSTURE_PROBE_DEG)
                if abs(candidate) > abs(chord):
                    chord = candidate
        # 原样恢复：这一步发生在加载之后、任何一帧之前，位置控制的电机目标还没被写过，
        # 所以恢复角度就等于恢复原状（``solve`` 里的 ``_teleport`` 也是这么切姿势的）。
        self.p.resetJointState(self.body, index, base, physicsClientId=self.cid)
        if not np.isfinite(before):
            return (str(name), 0.0, None)
        if abs(chord) < ELBOW_POSTURE_MIN_SLOPE:
            return (str(name), 0.0, float(before))
        return (str(name), chord, float(before))

    def record_posture(self, side: str) -> dict:
        """Sample arm joint angles and shoulder/elbow/wrist FK for session diagnostics."""
        entry = {name: float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
                 for index, name in self.arm[side]}
        entry.update(self.arm_points(side))
        wrist_world = self.last_target_world.get(side)
        if wrist_world is not None:
            entry["command_wrist"] = list(self.to_base(wrist_world))
        elbow_world = self.last_elbow_world.get(side)
        if elbow_world is not None:
            entry["command_elbow"] = list(self.to_base(elbow_world))
        solution_cost = self.last_solution_cost.get(side)
        if solution_cost is not None:
            # 同一帧「解」离目标多远：与下面 posture_summary 里从姿势量到的残差成对使用。
            entry["solution_cost"] = dict(solution_cost)
        self.posture[side].append(entry)
        return entry

    def clear_posture(self) -> None:
        """Drop the recorded posture samples (fresh A/B run)."""
        self.posture = {side: [] for side in self.sides}

    def posture_summary(self, side: str) -> dict:
        """Summarize sampled shoulder, upper-arm, forearm and elbow motion.

        Report direction errors, angle changes, coupling and branch recovery metrics."""
        from teleoperation.simulation.arm_realtime import elbow_bend_degrees

        frames = self.posture[side]
        empty = {"frames": len(frames), "shoulder_range_deg": {}, "shoulder_travel_deg": 0.0,
                 "upper_arm_dir_travel_deg": 0.0, "elbow_span_mm": 0.0,
                 "elbow_bend_deg": {"min": None, "max": None},
                 "command_frames": 0, "wrist_command_frames": 0, "command_note": None,
                 "upper_dir_error_deg": _percentile_stats([]),
                 "forearm_dir_error_deg": _percentile_stats([]),
                 "elbow_task_mm": _spike_stats([]), "wrist_task_mm": _spike_stats([]),
                 "solution_elbow_mm": _spike_stats([]),
                 "solution_wrist_mm": _spike_stats([]),
                 "upper_jerk_deg": _spike_stats([]), "forearm_jerk_deg": _spike_stats([]),
                 "elbow_jerk_mm": _spike_stats([]), "command_jerk_deg": _spike_stats([]),
                 "upper_struggle_fraction": None, "forearm_dir_dot": {},
                 "forearm_opposite_fraction": None, "worst_frame": None,
                 "worst_jerk_frame": None,
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
        shoulders = np.asarray([frame["upper"] for frame in sampled], dtype=np.float64)
        elbows = np.asarray([frame["elbow"] for frame in sampled], dtype=np.float64)
        upper = elbows - shoulders
        norms = np.linalg.norm(upper, axis=1)
        direction_travel = 0.0
        if np.all(norms > 1e-9):
            gram = (upper / norms[:, None]) @ (upper / norms[:, None]).T
            direction_travel = float(np.degrees(np.arccos(np.clip(gram.min(), -1.0, 1.0))))
        span = float(np.linalg.norm(elbows[:, None, :] - elbows[None, :, :], axis=-1).max()) * 1000.0
        # 肘弯曲角也从**刚体**三段量（upper-elbow-wrist）：用名字匹配的肩点会随 shoulder_roll
        # 摆动 48-102 mm，把「锚点错」算进弯曲角里。
        bends = [elbow_bend_degrees(points["upper"], points["elbow"], points["wrist"])
                 for points in sampled]
        bends = [value for value in bends if np.isfinite(value)]
        # 目标 vs 走到的（两段方向，基座系）：方向口径只对「方向」负责，所以这里比对方向而不是
        # 腕点。采样是在写下一帧命令**之前**做的，所以 frame 里的 command_* 正是这一帧姿势
        # 对应的那条命令（见 record_posture）。
        upper_errors, fore_errors, fore_dots, opposite = [], [], [], 0
        struggled = 0
        worst = None
        worst_jerk = None
        elbow_task, wrist_task = [], []
        solution_elbow, solution_wrist = [], []
        jerk_upper, jerk_fore, jerk_elbow, jerk_command = [], [], [], []
        previous_upper = previous_fore = previous_elbow = previous_wanted = None
        wrist_frames = 0
        command_note = None
        for sampled_index, frame in enumerate(sampled):
            command_elbow = frame.get("command_elbow")
            command_wrist = frame.get("command_wrist")
            # 腕命令点从第一次 solve 起每帧都在（见 record_posture）；肘命令点只有
            # ``--arm-elbow-task`` 打开时才记。缺肘命令**不等于**「这臂没收到命令」：
            # 这里照算 ``wrist_task_mm``，只把方向族跳过，并把原因写进 ``command_note``
            # （在这个修复之前是整段 continue，报告里只剩 None）。
            if command_wrist is None:
                continue
            wrist_frames += 1
            # 与 `_branch_cost` 同一口径：上臂方向用刚体肩端（`upper`）。命令点是从加载时的
            # 肩锚点推出来的，锚点换了之后 must 与记录点一致，否则误差会凭空多出 48-102 mm。
            shoulder = self.to_base(frame["upper"])
            elbow = self.to_base(frame["elbow"])
            wrist = self.to_base(frame["wrist"])
            wrist_task.append(float(np.linalg.norm(np.asarray(wrist, dtype=np.float64)
                                                   - np.asarray(command_wrist,
                                                                dtype=np.float64))) * 1000.0)
            if command_elbow is None:
                command_note = ("elbow command not recorded (--arm-elbow-task off): "
                                "direction metrics need it, wrist_task_mm does not")
                continue
            elbow_task.append(float(np.linalg.norm(np.asarray(elbow, dtype=np.float64)
                                                   - np.asarray(command_elbow,
                                                                dtype=np.float64))) * 1000.0)
            solution_cost = frame.get("solution_cost")
            if solution_cost is not None:
                solution_elbow.append(float(solution_cost["elbow_mm"]))
                solution_wrist.append(float(solution_cost["wrist_mm"]))
            achieved_u = _unit_difference(elbow, shoulder)
            achieved_f = _unit_difference(wrist, elbow)
            wanted_u = _unit_difference(command_elbow, shoulder)
            wanted_f = _unit_difference(command_wrist, command_elbow)
            if achieved_u is None or wanted_u is None:
                continue
            if previous_upper is not None:
                jerk_upper.append(_direction_angle_deg(previous_upper, achieved_u))
            previous_upper = achieved_u
            if previous_elbow is not None:
                jerk_elbow.append(float(np.linalg.norm(np.asarray(elbow, dtype=np.float64)
                                                       - previous_elbow)) * 1000.0)
            previous_elbow = np.asarray(elbow, dtype=np.float64)
            if achieved_f is not None and wanted_f is not None:
                if previous_fore is not None:
                    jerk = _direction_angle_deg(previous_fore, achieved_f)
                    jerk_fore.append(jerk)
                    if worst_jerk is None or jerk > worst_jerk["forearm_jerk_deg"]:
                        # 抽得最狠的那一帧的原样向量：一眼看出是「命令自己跳」还是「臂自己跳」。
                        worst_jerk = {
                            "sampled_frame": int(sampled_index),
                            "forearm_jerk_deg": round(jerk, 2),
                            "command_jerk_deg": (None if previous_wanted is None else
                                                 round(_direction_angle_deg(previous_wanted,
                                                                           wanted_f), 2)),
                            "previous_achieved_fore": [round(float(v), 3) for v in previous_fore],
                            "achieved_fore": [round(float(v), 3) for v in achieved_f],
                            "previous_wanted_fore": (None if previous_wanted is None else
                                                     [round(float(v), 3) for v in previous_wanted]),
                            "wanted_fore": [round(float(v), 3) for v in wanted_f],
                            "command_elbow": [round(float(v), 3) for v in command_elbow],
                            "command_wrist": [round(float(v), 3) for v in command_wrist]}
                previous_fore = achieved_f
                if previous_wanted is not None:
                    # 这一帧命令自己转了多少度：抖动是「臂的问题」还是「输入本来就跳」，靠它分。
                    jerk_command.append(_direction_angle_deg(previous_wanted, wanted_f))
                previous_wanted = wanted_f
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
                "wrist_command_frames": wrist_frames,
                # 方向族没数时写明原因（缺肘命令 / 一条命令都没记到），不留给读者一个裸的 None。
                "command_note": ("no command point recorded for this side"
                                 if wrist_frames == 0 else command_note),
                "elbow_task_mm": _spike_stats(elbow_task),
                "wrist_task_mm": _spike_stats(wrist_task),
                "solution_elbow_mm": _spike_stats(solution_elbow),
                "solution_wrist_mm": _spike_stats(solution_wrist),
                "upper_jerk_deg": _spike_stats(jerk_upper),
                "forearm_jerk_deg": _spike_stats(jerk_fore),
                "elbow_jerk_mm": _spike_stats(jerk_elbow),
                "command_jerk_deg": _spike_stats(jerk_command),
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
                "worst_jerk_frame": worst_jerk,
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
    def _live_rest_poses(self) -> list:
        """Read current joint angles for the IK null-space reference."""
        return [float(self.p.getJointState(self.body, index,
                                           physicsClientId=self.cid)[0])
                for index in self.indices]

    def _rest_poses(self, posture=None) -> list:
        """Build the IK null-space reference from the configured mode and optional posture.

        Current/live read actual joint angles; loaded uses the build-time pose;
        zero uses zero angles. Named posture values override the selected reference."""
        if not posture:
            return (self._live_rest_poses() if self.rest_poses in ("current", "live")
                    else list(self.limits["rest"]))
        rest = (self._live_rest_poses() if self.rest_poses in ("current", "live")
                else list(self.limits["rest"]))
        for name, value in dict(posture).items():
            index = self.index_of.get(str(name))
            if index is None:
                raise ValueError(
                    f"{self.robot_type}: posture joint {name!r} is not a movable joint "
                    f"of the loaded body")
            rest[self.position_of[index]] = float(value)
        return rest

    def posture_from_bend(self, side: str, bend_deg, neutral_bend=None) -> dict:
        """Convert operator elbow bend in degrees to a named robot joint reference.

        Use the measured joint slope and the robot load-time bend, or an explicitly
        provided neutral baseline. Clamp the reference to joint limits and the
        straight-arm boundary. Return an empty mapping without a usable slope."""
        name, slope, bend_at_hold = self.elbow_posture.get(side) or (None, 0.0, None)
        if not name or not slope or bend_deg is None or bend_at_hold is None:
            return {}
        bend = float(bend_deg)
        if not np.isfinite(bend):
            return {}
        baseline = float(bend_at_hold) if neutral_bend is None else float(neutral_bend)
        if not np.isfinite(baseline):
            return {}
        change = min(bend, BEND_STRAIGHT_DEG) - baseline
        index = self.index_of[str(name)]
        position = self.position_of[index]
        value = float(self.hold[index]) + radians(change / float(slope))
        low = float(self.limits["lower"][position])
        high = float(self.limits["upper"][position])
        return {str(name): min(max(value, low), high)}


    def _locked_arguments(self, side, arguments) -> dict:
        """Pin non-driven joints to their current angles in the IK bounds."""
        free = {index for index, _name in self.arm[side]}
        lower = list(arguments["lowerLimits"])
        upper = list(arguments["upperLimits"])
        for position, index in enumerate(self.indices):
            if index in free:
                continue
            angle = float(self.p.getJointState(self.body, index,
                                               physicsClientId=self.cid)[0])
            lower[position] = angle - 1e-6
            upper[position] = angle + 1e-6
        locked = dict(arguments)
        locked["lowerLimits"], locked["upperLimits"] = lower, upper
        # jointRanges 只影响步长缩放，不参与「能不能到」，保持原样。
        return locked


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
        """Solve simultaneous wrist and elbow position targets with PyBullet IK2."""
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
        """Return named arm angles with tolerance-sized overshoot pulled onto the bounds.

        The wrist branch accepts a value up to ``ARM_LIMIT_TOLERANCE`` outside the URDF limits
        so a saturated pose is not rejected on every frame, while command validation requires
        the exact limits (1e-6). Return only in-range values; keep larger violations untouched
        so the existing rejection path still refuses them.
        """
        values = {}
        for index, name in self.arm[side]:
            position = self.position_of[index]
            value = float(solution[position])
            low = float(self.limits["lower"][position])
            high = float(self.limits["upper"][position])
            if np.isfinite(value) and low - ARM_LIMIT_TOLERANCE <= value <= high + ARM_LIMIT_TOLERANCE:
                value = min(max(value, low), high)
            values[name] = value
        return values

    def _limit_violation(self, side: str, solution) -> dict:
        """Worst out-of-limit arm joint of a rejected solution (report diagnostic).

        The wrist branch rejects a pose whose arm values leave the URDF bounds by more than
        ``ARM_LIMIT_TOLERANCE``. Recording which joint, its value and the excess lets a report
        tell boundary noise (1e-6..1e-3 rad) from a genuinely unreachable target.
        """
        worst = None
        for index, name in self.arm[side]:
            position = self.position_of[index]
            value = float(solution[position])
            low = float(self.limits["lower"][position])
            high = float(self.limits["upper"][position])
            if not np.isfinite(value):
                return {"joint": name, "value": None, "lower": round(low, 6),
                        "upper": round(high, 6), "excess_rad": None}
            excess = value - high if value > high else low - value
            if excess > ARM_LIMIT_TOLERANCE and (worst is None or excess > worst["excess_rad"]):
                worst = {"joint": name, "value": round(value, 6),
                         "lower": round(low, 6), "upper": round(high, 6),
                         "excess_rad": round(float(excess), 8)}
        return worst or {}

    def _teleport(self, side: str, solution) -> None:
        """Temporarily set arm joint positions for the next IK pass, without stepping physics."""
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
        # 方向一律以**刚体**肩端（``upper`` = 肘关节父连杆原点）为基准：``shoulder`` 那个点在
        # 非中性姿势下会离开上臂（见 ``upper_link``），用它量方向会把「锚点错」算成「臂没动」。
        anchor = np.asarray(points["upper"], dtype=np.float64)
        elbow = np.asarray(points["elbow"], dtype=np.float64)
        wrist = np.asarray(points["wrist"], dtype=np.float64)
        cost = {"elbow_mm": round(float(np.linalg.norm(
            elbow - np.asarray(elbow_world, dtype=np.float64))) * 1000.0, 3),
            "wrist_mm": round(float(np.linalg.norm(
                wrist - np.asarray(world, dtype=np.float64))) * 1000.0, 3)}
        achieved_u = _unit_difference(elbow, anchor)
        achieved_f = _unit_difference(wrist, elbow)
        wanted_u = _unit_difference(elbow_world, anchor)
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
        """Solve wrist/elbow tasks while retaining position-only pose fallback.

        Temporary joint resets are enclosed by solve's state-restoration block.
        """
        solution = (self._ik_call(side, world, None, arguments) if elbow_world is None
                    else self._ik_call2(side, world, elbow_world, arguments))
        position_solution = solution
        target_orientation = None
        pose_error = position_error = None
        fallback = False
        if self.ik_mode == "pose" or orientation is not None:
            if warm_start:
                self._teleport(side, solution)
                if orientation is not None:
                    position_error = 1000 * sum(
                        (value - wanted) ** 2
                        for value, wanted in zip(self.measure(side)[0], world)) ** 0.5
                target_orientation = (self._orientation(side) if orientation is None
                                      else tuple(float(value) for value in orientation))
            else:
                target_orientation = (self.neutral_orientation[side] if orientation is None
                                      else tuple(float(value) for value in orientation))
            solution = self._ik_call(side, world, target_orientation, arguments)
            if warm_start and orientation is not None:
                self._teleport(side, solution)
                pose_error = 1000 * sum(
                    (value - wanted) ** 2
                    for value, wanted in zip(self.measure(side)[0], world)) ** 0.5
                if pose_error > self.profile.max_residual_mm and pose_error > position_error:
                    solution = position_solution
                    fallback = True
        elif warm_start:
            self._teleport(side, solution)
            solution = self._ik_call(side, world, None, arguments)
        if elbow_world is not None and elbow_pass == "last" and orientation is None:
            self._teleport(side, solution)
            solution = self._ik_call2(side, world, elbow_world, arguments)
        self._sequence_details = {
            "position_fallback": fallback,
            "position_candidate_residual_mm": position_error,
            "pose_candidate_residual_mm": pose_error,
        }
        return solution, target_orientation

    def _recover_elbow_branch(self, side, solution, world, elbow_world, orientation,
                              arguments, elbow_pass, warm_start):
        """Retry from the loaded pose when the direction-error threshold is exceeded.

        Require the configured improvement margin and respect the recovery cooldown."""
        self._teleport(side, solution)
        if self.elbow_recovery_wait.get(side, 0) > 0:
            # 冷却中：上一次刚尝试过（成功或被拒），这段时间就认当前这一支。新支路会从自己
            # warm start 保持住，所以冷却期不会掉回旧支路。
            self.elbow_recovery_wait[side] -= 1
            return solution
        current = self._branch_cost(side, world, elbow_world)
        if (self.elbow_recovery_deg <= 0.0 or current["cost_deg"] is None
                or current["cost_deg"] <= self.elbow_recovery_deg):
            return solution
        if self.elbow_recovery_cooldown > 0:
            self.elbow_recovery_wait[side] = self.elbow_recovery_cooldown
        # 起点固定用**加载姿势**（不是 ``_rest_poses(None)``）：这条路要的就是「从与当前姿势
        # 无关的那一支重解」，``rest_poses="live"`` 会把那行换成当前姿势，等于原地重解。
        self._teleport(side, list(self.limits["rest"]))
        candidate, _orientation = self._ik_sequence(side, world, elbow_world, orientation,
                                                    arguments, elbow_pass, warm_start)
        self._teleport(side, candidate)
        fresh = self._branch_cost(side, world, elbow_world)
        limit = max(1.5 * current["wrist_mm"], current["wrist_mm"] + 60.0)
        if (fresh["cost_deg"] is None
                or fresh["cost_deg"] > current["cost_deg"] - self.elbow_recovery_margin_deg
                or fresh["wrist_mm"] > limit):
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
        """Solve one arm and return {joint_name: target_angle}.

        Resolve position and elbow targets through source. Wrist orientation is
        world-absolute unless orientation_frame="neutral", which composes it with
        the robot build-time wrist orientation. Optional posture overrides the
        null-space reference. elbow_pass selects when the two-position task runs.
        Warm-start temporary joint positions and velocities are restored even on
        failure. Reject invalid or out-of-limit wrist solutions and bound large
        joint steps before returning the command."""
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
        if self.ik_lock_others:
            arguments = self._locked_arguments(side, arguments)
        saved = [(index, self.p.getJointState(self.body, index,
                                            physicsClientId=self.cid)[:2])
                 for index in self.indices]
        gui = self.p.getConnectionInfo(physicsClientId=self.cid).get('connectionMethod') == self.p.GUI
        if gui:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0, physicsClientId=self.cid)
        try:
            solution, target_orientation = self._ik_sequence(
                side, world, elbow_world, orientation, arguments, elbow_pass, warm_start)
            details = dict(self._sequence_details)
            if elbow_world is not None and orientation is None:
                original_solution = solution
                solution = self._recover_elbow_branch(
                    side, solution, world, elbow_world, orientation, arguments,
                    elbow_pass, warm_start)
                if solution is not original_solution:
                    details = dict(self._sequence_details)
                self.last_solution_cost[side] = self._branch_cost(side, world, elbow_world)
        finally:
            try:
                for index, (position_before, velocity_before) in saved:
                    self.p.resetJointState(self.body, index, position_before,
                                           targetVelocity=velocity_before,
                                           physicsClientId=self.cid)
            finally:
                if gui:
                    self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1, physicsClientId=self.cid)
        if orientation is not None:
            previous = {i: q for i, (q, _v) in saved}
            rejection = "pose_position_conflict" if details.get("position_fallback") else None
            largest_step = 0.
            for index, _name in self.arm[side]:
                k = self.position_of[index]
                value = solution[k]
                if not np.isfinite(value) or not self.limits["lower"][k] - ARM_LIMIT_TOLERANCE <= value <= self.limits["upper"][k] + ARM_LIMIT_TOLERANCE:
                    rejection = "nonfinite_or_limit"
                largest_step = max(largest_step, abs(value - previous[index]))
            if rejection:
                if rejection == "nonfinite_or_limit":
                    details["limit_violation"] = self._limit_violation(side, solution)
                solution = [previous[index] for index in self.indices]
            elif largest_step > .35:
                # Bound a branch-sized step without permanently rejecting an
                # otherwise reachable target on every subsequent frame.
                fraction = .35 / largest_step
                solution = [previous[i] + fraction * (value - previous[i])
                            for i, value in zip(self.indices, solution)]
                details["wrist_joint_step_scaled"] = fraction
            details["wrist_rejection"] = rejection
            details["wrist_pose_requested"] = True
        self.last_target_world[side] = world
        self.last_orientation[side] = target_orientation
        self.last_solve[side] = details
        self.last_posture[side] = {str(name): float(value)
                                   for name, value in dict(posture or {}).items()}
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

    def orientation_error_deg(self, side: str, target_quaternion=None) -> float | None:
        """Angle (deg) between the commanded orientation and the measured one."""
        target = (self.last_orientation.get(side) if target_quaternion is None
                  else target_quaternion)
        if target is None:
            return None
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
            "elbow_posture": {side: {
                "joint": name, "bend_slope": round(float(slope), 3),
                "bend_at_hold_deg": None if bend is None else round(float(bend), 2)}
                for side, (name, slope, bend) in self.elbow_posture.items()},
            "limits_from_urdf": True,
            "last_solve": dict(self.last_solve),
        }
