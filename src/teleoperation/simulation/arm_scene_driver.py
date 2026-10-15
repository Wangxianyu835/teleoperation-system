"""Drive the arms of the three-robot scene without editing the teammate's file.

``ThreeRobotHandScene`` holds every non-hand joint -- the 7+7 arm joints included --
at its load angle each frame, which is exactly why the arms look frozen.  That file
is teammate-owned (CONVENTIONS rule 3), so this wrapper detaches the arm joints from
``robot["hold_ids"]`` / ``robot["hold_targets"]`` after construction (both are plain
Python lists) and drives them from per-robot IK instead:

    scene = ThreeRobotHandScene(sides=("right",), gui=False)   # teammate's class
    driver = ArmSceneDriver(scene, sides=("right",))           # our layers
    for position in sweep:                                     # base-frame metres
        driver.apply("right", position, source="base")
        driver.step()                                          # scene.step()
    print(driver.summary(), driver.residuals_mm())

``robot_type`` is what selects the arm; nothing in the retargeting interfaces changes:

    ArmProfile      simulation/arm_profiles.py   per-robot tuning, ik_mode, ee link
      -> PyBulletArmIK  simulation/arm_ik.py      one solver per loaded robot
      -> ArmSceneDriver this file                 release holds, write motors, step

The hands keep working exactly as before: ``scene.apply(side, angles_18)`` is not
touched, and finger joints are never in the hold lists to begin with.  Finger
dynamics do not disturb the arm either -- the arm motors are the stiff ones here.

Why a wrapper instead of two lines in the scene: the scene is in the teammate's
baseline tree, so editing it would show up as a modified teammate file in
``devtools/preflight.ps1`` (rule 3).  The two-line version is described in
``docs/multi-robot-arm.md`` as a patch to hand back to whoever owns that file.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_ik import ARM_FORCE, ELBOW_PASS_ORDERS, PyBulletArmIK
from teleoperation.simulation.arm_profiles import (
    SIDES,
    describe,
    loadable_robots,
    profile_for,
    table,
)

#: 斜坡换算成「每帧一步」之后的硬上限（米）。摄像头循环掉速时 ``slew_m_per_s * dt`` 会长大，
#: 但一步几十厘米会把 IK 送回坏分支（这正是斜坡要修的东西），所以给它一个天花板。
#: 40 mm 是 2026-10-13 实测的拐点：40/60/80 mm 三档的横摆传递比都是 0.80（不再有改善），
#: 但 60 mm 的最大残差从 19.8 涨到 26.1 mm（判据 30 mm），所以取 40。
SLEW_MAX_STEP_M = 0.04
#: 「肘-腕」这一对（``--arm-map posture`` 的刚性配方）被斜坡压到请求长度的这个比例以下时，
#: 这一帧**保持上一帧下发的那一对**（两个点都不动：不跳、也不压扁），等操作者慢下来再继续跟。
#: 为什么不是在会压扁的帧上「按请求值一步到位」：那样症状更好（前臂反向帧 7 -> 1），但目标要跳
#: 一步，前臂抽动 max 从 26 度涨到 70 度 —— 而「手臂会抽」正是上一轮刚修掉的东西。保持那一对
#: 的版本：前臂反向帧 7 -> 3（剩下的 3 帧在未修复的版本里本来就有，逐帧数字相同）、抽动 26 度
#: 不变、残差不变、前臂间距还从 0.55 改善到 23.8 mm（g1）。数字见 docs/multi-robot-arm.md 8.18。
SLEW_PAIR_FLOOR_RATIO = 0.6


def slew_per_frame(speed_m_per_s: float, frame_dt: float,
                   ceiling_m: float = SLEW_MAX_STEP_M) -> float:
    """把「米/秒」的跟随速率换成「这一帧最多走多少米」（含天花板）。

    为什么不能写死 mm/帧：现场摄像头循环只有 8-12 fps，写死的 20 mm/帧 = 0.20 m/s，
    而实测录制里操作者手的每帧位移是「中位 37 mm、p90 147 mm」（10.05 fps -> p90 1.48 m/s），
    于是斜坡帧帧都在生效，机器人被按 13% 的手速拖着走，左右摆动直接消失（正弦实测传递比
    只有 0.43）。换成速率口径后 30 fps 与 10 fps 的跟随速度一致（30 fps 13.3 mm/帧、
    10 fps 40 mm/帧），摄像头再慢也不会给 IK 一步几十厘米。
    """
    if speed_m_per_s <= 0.0 or frame_dt <= 0.0:
        return 0.0
    return min(float(speed_m_per_s) * float(frame_dt), float(ceiling_m))


class ArmSceneDriver:
    """Arms of an already-built ``ThreeRobotHandScene`` follow per-robot IK targets.

    ``scene`` may hold arms, hands or both; only the sides given to ``sides`` are
    driven, and only the robots named in ``robots`` (``None`` = every loaded one).
    A robot left out keeps its hold untouched, i.e. stays frozen exactly as before.
    The scene keeps owning the physics client, the ground plane and the finger map.
    """

    def __init__(self, scene, sides=("right",), robots=None, arm_force=ARM_FORCE,
                 ik_mode=None, rest_poses: str = "current",
                 ik_iterations=None, residual_threshold=None,
                 track_posture: bool = True, slew_m: float = 0.0,
                 slew_m_per_s: float = 0.0, slew_pair_floor_ratio: float | None = None,
                 elbow_pass: str = "first",
                 elbow_recovery_deg=None, elbow_recovery_margin_deg=None,
                 elbow_recovery_cooldown=None, shoulder_anchors=None,
                 shoulder_anchor_key: str = "upper", lock_others: bool = False):
        self.scene = scene
        self.p = scene.p
        self.cid = scene.cid
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if not scene.robots:
            raise ValueError("the scene has no loaded robot to drive")
        #: 每帧采一次姿势（肩/肘/腕 + 肩关节角），赛后统计「肩到底动没动」。
        self.track_posture = bool(track_posture)
        self.ik = {}
        self.released = {}
        self.applied = {side: 0 for side in self.sides}
        #: 目标速率上限（米/帧）：0 = 关闭（目标一步到位）。见 ``_slew_target``。
        self.slew_m = max(0.0, float(slew_m))
        #: 跟随速率（米/秒）：非 0 时每帧用 ``update_slew(dt)`` 换算成 ``slew_m``。
        #: 摄像头循环帧率不固定（现场 8-12 fps），按速率给才不会"帧率一低就变慢"。
        self.slew_m_per_s = max(0.0, float(slew_m_per_s))
        #: 「肘-腕」这一对（``elbow_rigid``）的地板比例：斜坡一次不许把这一对压到请求长度的
        #: 这个比例以下，否则这一帧保持上一帧下发的那一对（见 ``SLEW_PAIR_FLOOR_RATIO`` 与 8.18）。
        #: 0 = 关护栏。``None``（默认）= 用 ``SLEW_PAIR_FLOOR_RATIO``，于是现场可以用
        #: ``--arm-slew-pair-floor 0`` 回到历史行为做 A/B，也可以给 0.4 让护栏少介入。
        self.slew_pair_floor_ratio = (
            SLEW_PAIR_FLOOR_RATIO if slew_pair_floor_ratio is None
            else max(0.0, float(slew_pair_floor_ratio)))
        #: 每台每侧最后一次**下发**的基座系目标（斜坡的起点）。
        self.commanded = {}
        #: 「肘-腕」这一对最后一次下发的值（``elbow_rigid`` 的护栏在会压扁的帧上保持它）。
        self.commanded_pair = {}
        #: 同时给了腕与肘两个位置目标时，肘任务放在第几遍（``arm_ik.ELBOW_PASS_ORDERS``）。
        #: 实测 2026-10-13：``first`` 时第二遍的腕姿态收尾会把肘拉回（肘误差 141-256 mm），
        #: ``last`` 结束时两个位置目标同时满足（见 devtools/probe_elbow_task.py）。
        if elbow_pass not in ELBOW_PASS_ORDERS:
            raise ValueError(
                f"elbow_pass must be one of {ELBOW_PASS_ORDERS}, got {elbow_pass!r}")
        self.elbow_pass = str(elbow_pass)
        #: 肘反向支路的恢复阈值（毫米）：None = 用 arm_ik 的默认值，0 = 关（A/B 用）。
        #: 见 ``PyBulletArmIK._recover_elbow_branch``（现场「前臂弯曲方向相反」的修法）。
        self.elbow_recovery_deg = elbow_recovery_deg
        #: 恢复的滞回（度）与「尝试过之后的冷却帧数」：0 = 关。滞回 + 冷却一起解决 2026-10-14
        #: 现场反馈的「整条手臂会抽」（1 度门槛 + 无冷却时每 6 帧换一次支路）。
        self.elbow_recovery_margin_deg = elbow_recovery_margin_deg
        self.elbow_recovery_cooldown = elbow_recovery_cooldown
        #: 加载时的肩锚点（基座系，米）：``{robot: {side: [x, y, z]}}``。非空时每一帧都把腕/肘
        #: 两个目标平移「当前肩 - 锚点」，见 ``_shoulder_shift``。
        self.shoulder_anchors = {str(name): {str(side): [float(value) for value in point]
                                             for side, point in dict(per_side or {}).items()
                                             if point is not None}
                                 for name, per_side in dict(shoulder_anchors or {}).items()}
        #: 上臂的「肩端」取哪个 FK 点：``"upper"`` = 肘的父连杆原点（刚体，默认）；
        #: ``"shoulder"`` = 名字匹配到的肩关节连杆（历史口径，只留给 A/B 与回退）。
        if shoulder_anchor_key not in ("upper", "shoulder"):
            raise ValueError("shoulder_anchor_key must be 'upper' or 'shoulder', got "
                             f"{shoulder_anchor_key!r}")
        self.shoulder_anchor_key = str(shoulder_anchor_key)
        #: 求解时是否把非本侧手臂的关节锁在当前角度（``arm_ik._locked_arguments``）。
        lock_others = bool(lock_others)
        self.lock_others = lock_others
        #: 每台每侧最后一次的平移量（毫米，报告用）。
        self.shoulder_shift_mm = {}
        #: 被斜坡削掉一部分的帧数（报告用：说明这一场有多少帧没一步到位）。
        self.slew_limited = 0
        #: 其中有多少帧为了让「肘-腕」这一对不退化而**放弃限速**（``elbow_rigid``，方向口径）。
        #: 报告里的 ``target_slew_pair_skips``：它说明护栏救了几帧（见 8.18）。
        self.slew_pair_skips = 0
        #: 每帧两只前臂的最近距离（毫米）：左右臂交叉 / 叠中线就是这个量掉下来。
        self.forearm_gaps = []
        wanted = None if robots is None else {str(name) for name in robots}
        for robot in scene.robots:
            robot_type = robot["type"]
            if wanted is not None and robot_type not in wanted:
                continue
            settings = {}
            if ik_iterations is not None:
                settings["ik_iterations"] = int(ik_iterations)
            if residual_threshold is not None:
                settings["residual_threshold"] = float(residual_threshold)
            if self.elbow_recovery_deg is not None:
                settings["elbow_recovery_deg"] = float(self.elbow_recovery_deg)
            if self.elbow_recovery_margin_deg is not None:
                settings["elbow_recovery_margin_deg"] = float(self.elbow_recovery_margin_deg)
            if self.elbow_recovery_cooldown is not None:
                settings["elbow_recovery_cooldown"] = int(self.elbow_recovery_cooldown)
            settings["lock_others"] = self.lock_others
            adapter = PyBulletArmIK(robot["body"], self.cid, robot_type,
                                    sides=self.sides, ik_mode=ik_mode,
                                    arm_force=arm_force, rest_poses=rest_poses,
                                    **settings)
            self.released[robot_type] = self._release_holds(robot, adapter)
            self.ik[robot_type] = adapter
        if not self.ik:
            raise ValueError(
                f"none of the requested robots {sorted(wanted)} is loaded; the scene "
                f"has {[robot['type'] for robot in scene.robots]}")

    def _release_holds(self, robot, adapter) -> dict:
        """Take the driven arm joints out of the scene's per-frame hold list.

        The scene stores these as plain lists, so filtering them in place is enough;
        the count is checked against what the adapter resolved, because a renamed key
        upstream would otherwise silently leave the arms held (frozen again).
        """
        body = robot["body"]
        wanted = set(adapter.driven)
        kept_ids, kept_targets, released = [], [], []
        for index, target in zip(robot["hold_ids"], robot["hold_targets"]):
            if index in wanted:
                released.append(adapter.name_of.get(index, str(index)))
            else:
                kept_ids.append(index)
                kept_targets.append(target)
        expected = len(adapter.driven)
        if len(released) != expected:
            raise RuntimeError(
                f"{robot['type']}: expected {expected} held arm joints, released "
                f"{len(released)}; the scene's hold bookkeeping changed (body "
                f"{body}) -- check three_robot_scene.py before trusting the arms")
        robot["hold_ids"] = kept_ids
        robot["hold_targets"] = kept_targets
        return {"joints": released, "count": len(released),
                "held_after": len(kept_ids)}


    # ------------------------------------------------------------------ #
    def apply(self, side: str, position, source: str = "base",
              orientation=None, orientation_frame: str = "absolute") -> dict:
        """One frame for one side with the *same* target for every robot.

        ``position`` is in the space named by ``source``; use
        :meth:`apply_each` when each robot needs its own base-frame target.
        ``orientation_frame`` is forwarded to every adapter: ``"neutral"`` means
        ``orientation`` is relative to that robot's own build-time wrist pose.
        """
        if side not in self.sides:
            raise ValueError(f"{side!r} is not driven by this driver ({list(self.sides)})")
        return self.apply_each(side, {robot_type: position for robot_type in self.ik},
                               source=source, orientation=orientation,
                               orientation_frame=orientation_frame)

    def set_shoulder_anchors(self, anchors) -> None:
        """加载后调用：``{robot: {side: {"shoulder": [x, y, z], ...}}}``（基座系，米）。

        只取 ``"shoulder"``；给空字典 = 关闭平移（与历史行为一致）。
        """
        self.shoulder_anchors = {}
        for name, per_side in dict(anchors or {}).items():
            entries = {}
            for side, entry in dict(per_side or {}).items():
                point = (entry or {}).get("shoulder") if isinstance(entry, dict) else None
                if point is not None:
                    entries[str(side)] = [float(value) for value in point]
            if entries:
                self.shoulder_anchors[str(name)] = entries

    def _shoulder_shift(self, robot_type, side, adapter):
        """``当前肩 FK - 加载时肩锚点``（基座系，米）；没有锚点给 ``None``。

        为什么需要：``arm_realtime`` 的肘/腕目标 = **加载时**的肩锚点 + 机器人连杆长*方向。
        机器人的躯干会被手臂反作用力推着走（这一场最差那帧 gr1_t2 的肩 FK 比锚点前移了 72 mm），
        于是「目标到**当前**肩的距离」不再等于上臂长：7 自由度链这时只能折中，肘被甩到别的
        位置 —— 现场看到的就是「前臂弯曲方向相反」。把两个目标一起平移
        「当前肩 - 锚点」之后，目标重新落在「离当前肩正好一根上臂」的球面上，方向也才是本来
        要的那个 M(你的方向)。数字见 docs/multi-robot-arm.md 第 8.12 节。

        2026-10-14 起默认「肩」指 ``arm_points()["upper"]``（= 肘关节父连杆原点，与肘刚性相连），
        不再是名字匹配的 ``"shoulder"``：后者不随 shoulder_roll 刚性移动，这一项会把
        「roll 转出来的 48-102 mm」误当成「躯干漂移」。锚点由 ``set_shoulder_anchors`` 写入，
        两边必须同一口径（``realtime_teleop.build_scene`` 也用同一个 key；``--arm-anchor
        shoulder`` 可以把这一行切回历史口径做 A/B）。
        """
        anchor = (self.shoulder_anchors.get(robot_type) or {}).get(side)
        if anchor is None:
            return None
        current = adapter.to_base(adapter.arm_points(side)[self.shoulder_anchor_key])
        return tuple(float(current[axis]) - float(anchor[axis]) for axis in range(3))

    def apply_each(self, side: str, targets, source: str = "base",
                   orientation=None, orientation_frame: str = "absolute",
                   elbows=None, elbow_bend_deg=None, elbow_rigid=None,
                   elbow_bend_baseline=None) -> dict:
        """One frame for one side with a target per robot: ``{robot_type: position}``.

        ``elbows`` = optional ``{robot_type: point}`` in the same space: the paper's
        elbow position task.  A robot missing from it gets no elbow target (wrist only),
        so an older caller keeps working unchanged.

        ``elbow_bend_deg`` = the operator's elbow bend this frame, in degrees
        (``arm_realtime``'s ``elbow_bend_target()``; 180 = straight).  It becomes each
        adapter's null-space posture reference for this frame
        (``PyBulletArmIK.posture_from_bend``) -- the paper's ``PostureTask``: the wrist
        and elbow tasks still decide *where*, this decides how the elbow bends while
        doing it, and each robot converts the same number through its own measured elbow
        joint and bend slope.  ``None`` (default) leaves the reference exactly where it
        was, so an older caller is byte-for-byte unchanged.

        ``elbow_bend_baseline`` = optional ``{robot_type: bend or None}`` that **moves the
        baseline** of that robot's posture reference: a robot listed with a number measures
        the operator's bend change against *that* number instead of against its own loaded
        bend.  The ``delta`` direction frame (``ArmProfile.posture_frame``) needs this,
        because there its "operator did not move" pose *is* the robot's loaded pose, so the
        baseline has to be the operator's neutral bend (``arm_realtime.elbow_bend_neutral``).
        Mixing the two puts a constant between the posture reference and the position task
        -- the 8.17.4 failure.  ``None`` (default) = every robot keeps its own loaded bend.

        ``elbow_rigid`` = optional collection of robot types whose elbow target forms a
        **rigid pair with the wrist target** (the ``--arm-map posture`` recipe:
        ``wrist = elbow + L_f * v``, see ``arm_realtime._posture_target``).  The slew cap
        below only walks the *wrist* in, so on a fast operator move the elbow target runs
        ahead and the two points come together: measured 2026-10-15 on the 200 frame
        replay (``--arm-slew-mm 40``) gr1_t2 left commanded forearm 251 mm -> **51 mm** on
        11 frames, and the solver folded the forearm the other way on 7 of them (128.7 deg)
        -- the "GR1-T2 大臂与小臂夹角反过来" the operator reports after a while.  For a
        robot listed here the cap is therefore **skipped and the pair that was sent on the
        previous frame is held** on the frames where it would squash the pair
        (``slew_pair_floor_ratio``, default ``SLEW_PAIR_FLOOR_RATIO``; **0 switches the guard
        off**, which is the historical behaviour and what an on-site A/B uses), and applies
        exactly as before on every other frame:
        measured 2026-10-15, gr1_t2 left commanded forearm never below 151 mm (was 51 mm),
        squashed frames 11 -> 0, achieved forearm pointing away from the command 7 -> 3
        frames (the 3 remaining frames are the older IK branch burst, identical in the
        unfixed run), jerk and residuals unchanged, and the g1 forearm gap improved from
        0.55 mm to 23.8 mm.  Robots left out (the ``--arm-elbow-task`` recipe, where the
        elbow is an independent position target) keep the historical behaviour
        byte-for-byte.  Two other variants were measured and rejected (translating the
        pair as a whole, and letting the cap through on the squashed frames): numbers in
        docs/multi-robot-arm.md 8.18.

        Returns ``{robot_type: {"ee": link, "ik_mode": ..., "joints": {name: angle}}}``
        so a demo, a test or a report can state exactly which arm was driven.
        """
        if side not in self.sides:
            raise ValueError(f"{side!r} is not driven by this driver ({list(self.sides)})")
        missing = sorted(robot_type for robot_type in self.ik
                         if robot_type not in targets)
        if missing:
            raise ValueError(f"no target given for {missing} (driven: {list(self.ik)})")
        elbows = dict(elbows or {})
        reports = {}
        for robot_type, adapter in self.ik.items():
            if self.track_posture:
                # 先采样：现在读到的是上一帧稳定下来的姿势（apply 只写电机目标）。
                adapter.record_posture(side)
            wanted = targets[robot_type]
            elbow = elbows.get(robot_type)
            rigid = (elbow is not None and elbow_rigid is not None
                     and str(robot_type) in elbow_rigid)
            if self.slew_m > 0.0 and source == "base":
                requested = tuple(float(value) for value in wanted)
                wanted = self._slew_target(robot_type, side, adapter, wanted)
                if rigid and self.slew_pair_floor_ratio > 0.0:
                    # 压扁护栏：斜坡只许把**腕**慢慢挪过去，不许把「肘-腕」这一对压成
                    # 退化目标（见 ``apply_each`` docstring 与 8.18）。会压扁就这一帧
                    # **保持上一帧下发的那一对**（两个点都不动）：不跳、不压扁，等操作者
                    # 慢下来再继续跟。第一帧还没有可保持的一对时按请求值一步到位。
                    shape = float(np.linalg.norm(
                        np.asarray(requested, dtype=np.float64)
                        - np.asarray(elbow, dtype=np.float64)))
                    narrowed = float(np.linalg.norm(
                        np.asarray(wanted, dtype=np.float64)
                        - np.asarray(elbow, dtype=np.float64)))
                    if narrowed < self.slew_pair_floor_ratio * shape:
                        held = self.commanded_pair.get((robot_type, side))
                        if held is None:
                            wanted = requested
                        else:
                            elbow, wanted = held
                        self.commanded[(robot_type, side)] = np.asarray(
                            wanted, dtype=np.float64)
                        self.slew_pair_skips += 1
                    else:
                        # 这一帧照旧限速：记下实际下发的这一对，供下一帧的护栏保持。
                        self.commanded_pair[(robot_type, side)] = (
                            elbow, tuple(float(value) for value in wanted))
            shift = (None if source != "base"
                     else self._shoulder_shift(robot_type, side, adapter))
            if shift is not None:
                # 平移放在斜坡**之后**：斜坡活在锚点系里（上一帧的命令也在锚点系），
                # 平移只是把这一帧的参考点从锚点换到当前肩。
                wanted = tuple(float(value) + shift[axis] for axis, value in enumerate(wanted))
                if elbow is not None:
                    elbow = tuple(float(value) + shift[axis]
                                  for axis, value in enumerate(elbow))
                self.shoulder_shift_mm.setdefault(robot_type, {})[side] = round(
                    1000.0 * float(np.linalg.norm(np.asarray(shift, dtype=np.float64))), 2)
            baseline = (None if elbow_bend_baseline is None
                        else elbow_bend_baseline.get(robot_type))
            posture = (None if elbow_bend_deg is None
                       else adapter.posture_from_bend(side, elbow_bend_deg,
                                                      neutral_bend=baseline))
            values = adapter.solve(side, wanted,
                                   orientation=orientation, source=source,
                                   orientation_frame=orientation_frame,
                                   elbow=elbow,
                                   elbow_pass=self.elbow_pass,
                                   posture=posture)
            adapter.apply(side, values)
            reports[robot_type] = {"ee": adapter.ee[side][1],
                                   "ik_mode": adapter.ik_mode,
                                   "shoulder_shift_mm": self.shoulder_shift_mm.get(
                                       robot_type, {}).get(side, 0.0),
                                   #: 这一帧交给零空间项的那个参考角（空的 = 历史行为）。
                                   "posture": {str(name): round(float(value), 5)
                                               for name, value in (posture or {}).items()},
                                   "joints": values}
        self.applied[side] += 1
        return reports

    # ------------------------------------------------------------------ #
    def update_slew(self, frame_dt: float) -> float:
        """按这一帧的实测时长把「米/秒」的跟随速率换算成本帧的步长上限（米/帧）。

        ``slew_m_per_s`` 为 0 时**不动** ``slew_m``：那种情况调用方用的是固定的
        ``--arm-slew-mm``（复现旧数字）或者干脆关掉了斜坡，这里绝不能把它改写成 0。
        返回本帧生效的 ``slew_m``（报告用）。
        """
        if self.slew_m_per_s <= 0.0:
            return self.slew_m
        self.slew_m = slew_per_frame(self.slew_m_per_s, frame_dt)
        return self.slew_m

    def _wrist_in_base(self, adapter, side: str):
        """That arm's wrist right now, in its **own** base frame (the ramp's start).

        ``PyBulletArmIK`` converts base -> world (``to_world``); the ramp needs the
        way back, and the base rotation matters (GR1-T2 does not load with an
        identity base rotation).
        """
        world = np.asarray(adapter.arm_points(side)["wrist"], dtype=np.float64)
        _base, orientation = self.p.getBasePositionAndOrientation(
            adapter.body, physicsClientId=self.cid)
        rotation = np.asarray(self.p.getMatrixFromQuaternion(orientation),
                              dtype=np.float64).reshape(3, 3)
        return rotation.T @ (world - np.asarray(adapter.base_position(),
                                               dtype=np.float64))

    def _slew_target(self, robot_type, side, adapter, position):
        """Rate-limit one commanded base-frame point to ``slew_m`` metres per call.

        Why this exists: the two-pass solver starts from the body's current angles, so a
        target that arrives in one large step can land it in a bad branch, and it then
        keeps solving into that branch every frame.  Measured on this machine 2026-10-13
        (gravity -9.81, 60 iterations / 1e-4, the app's own settings):

        * one command then hold (what an offline render or a probe does): h1_2 stops
          47.8 mm short with the two forearms 6.0 mm apart -- hands on the centre line --
          and g1 68.7 mm / gr1_t2 31.6 mm, all three past the 30 mm gate.  The same
          working point walked in at 20 mm/frame reaches h1_2 8.9 mm with an 83.5 mm gap
          (gr1_t2 6.7 / 200.0, g1 8.9 / 186.8).
        * replayed live session (outputs/tmp_realtime/live_demo_frames.npz, 300 frames,
          the app's own loop): cap off, h1_2 holds a constant 45 mm during frames
          105-165; at 20 mm/frame the same frames are 9 mm.

        So the first call of a side starts from the wrist's *current* position, and every
        later call starts from the previous command.  ``source`` must be ``"base"`` (the
        driver's one-target-per-robot contract); anything else passes through untouched.
        Frame-by-frame curve: E:\\cache\\tmp\\probe_slew_curve.py (tmp, not in git).
        """
        key = (robot_type, side)
        previous = self.commanded.get(key)
        if previous is None:
            previous = self._wrist_in_base(adapter, side)
        wanted = np.asarray(position, dtype=np.float64)
        offset = wanted - previous
        distance = float(np.linalg.norm(offset))
        if distance > self.slew_m:
            wanted = previous + offset * (self.slew_m / distance)
            self.slew_limited += 1
        self.commanded[key] = wanted
        return tuple(float(value) for value in wanted)

    def apply_operator(self, side: str, position, orientation=None,
                       orientation_frame: str = "absolute") -> dict:
        """``apply`` with operator-space metres: the profile's scale/offset is applied."""
        return self.apply(side, position, source="operator", orientation=orientation,
                          orientation_frame=orientation_frame)

    def step(self, frames: int = 1) -> None:
        """Advance the scene by ``frames`` rendered frames (``scene.step`` = 8 substeps).

        The left/right forearm gap is read *before* the step, i.e. from the pose the
        just-commanded frame settled into -- the same instant a posture sample covers.
        """
        for _ in range(max(1, int(frames))):
            if self.track_posture:
                gap = self.forearm_gap_mm()
                if gap:                       # 单侧驱动时量不出间隙，别记空帧
                    self.forearm_gaps.append(gap)
            self.scene.step()

    def forearm_gap_mm(self) -> dict:
        """``{robot: mm}``: distance between that robot's two forearms, right now.

        The forearm is ``elbow -> wrist`` (the segment that was folding onto the
        midline in the frames this session set out to fix).  Segment-to-segment
        distance, no collision shapes involved, so it works on any robot here.
        """
        if len(self.sides) < 2 or not all(side in adapter.arm for adapter in self.ik.values()
                                         for side in self.sides):
            return {}
        out = {}
        for robot_type, adapter in self.ik.items():
            left = adapter.arm_points("left") if "left" in self.sides else None
            right = adapter.arm_points("right") if "right" in self.sides else None
            if left is None or right is None:
                continue
            out[robot_type] = round(1000.0 * _segment_distance(
                left["elbow"], left["wrist"], right["elbow"], right["wrist"]), 2)
        return out

    def forearm_gap_summary(self) -> dict:
        """Worst (smallest) left/right forearm gap per robot over the session, plus the
        minimum across robots: the number that says "did the two arms cross"."""
        per_robot = {}
        for robot_type in self.ik:
            values = [entry[robot_type] for entry in self.forearm_gaps
                      if robot_type in entry]
            if values:
                per_robot[robot_type] = round(min(values), 2)
        worst = min(per_robot.values()) if per_robot else None
        return {"frames": len(self.forearm_gaps), "min_mm": per_robot,
                "worst_mm": None if worst is None else round(float(worst), 2)}

    def posture_summary(self, sides=None) -> dict:
        """``{robot: {side: posture metric}}`` -- see ``PyBulletArmIK.posture_summary``."""
        wanted = self.sides if sides is None else tuple(sides)
        return {robot_type: {side: adapter.posture_summary(side) for side in wanted}
                for robot_type, adapter in self.ik.items()}

    def residuals_mm(self, sides=None) -> dict:
        """``{robot: {side: mm}}``: target-to-wrist distance measured right now."""
        return self._per_side(
            lambda adapter, side: round(adapter.residual_mm(side), 3), sides)

    def orientation_errors_deg(self, sides=None) -> dict:
        """``{robot: {side: deg}}``: commanded vs measured wrist orientation."""
        return self._per_side(
            lambda adapter, side: round(adapter.orientation_error_deg(side), 2), sides)

    def _per_side(self, measure, sides=None) -> dict:
        wanted = self.sides if sides is None else tuple(sides)
        out = {}
        for robot_type, adapter in self.ik.items():
            for side in wanted:
                if side in adapter.last_target_world:
                    out.setdefault(robot_type, {})[side] = measure(adapter, side)
        return out

    def arm_values(self, sides=None) -> dict:
        """``{robot: {side: {joint: angle}}}`` of the driven arms (for reports)."""
        wanted = self.sides if sides is None else tuple(sides)
        return {robot_type: {side: adapter.joint_values(side) for side in wanted}
                for robot_type, adapter in self.ik.items()}

    def hold_drift(self) -> dict:
        """Worst drift (rad) of the joints the scene still holds: legs, waist, head.

        The held set can include hand joints the finger mapping does not cover (e.g.
        the whole left hand when only the right one is driven).  Those droop under
        gravity -- a hand-subsystem fact, not arm tracking -- so they are reported
        separately as ``hand_drift_rad``, the same split the landmark probe uses.
        """
        from teleoperation.robots.specification import _is_hand_joint

        p, cid = self.p, self.cid
        out = {}
        for robot in self.scene.robots:
            worst, name, hands = 0.0, None, 0.0
            for index, target in zip(robot["hold_ids"], robot["hold_targets"]):
                info = p.getJointInfo(robot["body"], index, physicsClientId=cid)
                label = info[1]
                label = label.decode() if isinstance(label, bytes) else str(label)
                angle = float(p.getJointState(robot["body"], index,
                                              physicsClientId=cid)[0])
                drift = abs(angle - float(target))
                if _is_hand_joint(label, "left") or _is_hand_joint(label, "right"):
                    hands = max(hands, drift)
                elif drift > worst:
                    worst, name = drift, label
            out[robot["type"]] = {"max_drift_rad": round(worst, 4), "joint": name,
                                  "hand_drift_rad": round(hands, 4),
                                  "held_joints": len(robot["hold_ids"])}
        return out


    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        """What this driver drives, robot by robot (released joints + IK settings)."""
        return {
            "sides": list(self.sides),
            "applied_frames": dict(self.applied),
            "released": self.released,
            "posture_tracking": bool(self.track_posture),
            "target_slew_mm": round(1000.0 * self.slew_m, 3),
            "target_slew_mm_per_s": round(1000.0 * self.slew_m_per_s, 3),
            "target_slew_ceiling_mm": round(1000.0 * SLEW_MAX_STEP_M, 3),
            "target_slew_limited_frames": self.slew_limited,
            "target_slew_pair_skips": self.slew_pair_skips,
            #: 护栏的地板比例（0 = 关）：现场调它 / 对照旧行为就是这一项。
            "target_slew_pair_floor_ratio": round(self.slew_pair_floor_ratio, 4),
            "robots": {robot_type: adapter.describe()
                       for robot_type, adapter in self.ik.items()},
        }

    def summary(self) -> dict:
        """Report payload: what moved, how well it tracked, what stayed put."""
        return {
            "scene": "three_robots+arm_ik",
            "sides": list(self.sides),
            "applied_frames": dict(self.applied),
            "released_arm_joints": {robot_type: entry["count"]
                                    for robot_type, entry in self.released.items()},
            "residual_mm": self.residuals_mm(),
            "orientation_error_deg": self.orientation_errors_deg(),
            # 姿势：肩/肘/上臂到底有没有参与（论文说肘不用管，这里量它有没有动）；
            # 前臂间隙：左右两条前臂离多近（叠中线 / 交叉就是这个量掉下来）。
            "posture": self.posture_summary(),
            "forearm_gap_mm": self.forearm_gap_summary(),
            "hold_drift": self.hold_drift(),
            # 斜坡：速率口径 + 本场实际用到的每帧上限（现场 8-12 fps，数字会大于 30 fps 时）。
            "target_slew_mm": round(1000.0 * self.slew_m, 3),
            "target_slew_mm_per_s": round(1000.0 * self.slew_m_per_s, 3),
            "target_slew_ceiling_mm": round(1000.0 * SLEW_MAX_STEP_M, 3),
            "target_slew_limited_frames": self.slew_limited,
            #: 其中有多少帧为了让「肘-腕」这一对不退化而放弃限速（``elbow_rigid``）；见 8.18。
            "target_slew_pair_skips": self.slew_pair_skips,
            #: 护栏的地板比例（0 = 关护栏 = 历史行为）：报告里一眼看出这一场用的哪一档。
            "target_slew_pair_floor_ratio": round(self.slew_pair_floor_ratio, 4),
            "hands": self.scene.summary(),
            "robots": self.describe()["robots"],
        }


def _segment_distance(p1, q1, p2, q2) -> float:
    """Shortest distance (metres) between segment ``p1-q1`` and segment ``p2-q2``.

    The textbook clamped closest-point solve (Ericson, *Real-Time Collision Detection*,
    5.1.9), pure numpy: no collision shapes, so it works on the ``elbow -> wrist``
    segments of any robot here, and on a ``--input-npz`` replay.
    """
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    q1 = np.asarray(q1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)
    q2 = np.asarray(q2, dtype=np.float64).reshape(3)
    d1, d2, offset = q1 - p1, q2 - p2, p1 - p2
    a, e = float(d1 @ d1), float(d2 @ d2)
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(offset))
    if a <= epsilon:
        s, t = 0.0, float(np.clip(float(d2 @ offset) / e, 0.0, 1.0))
    else:
        c = float(d1 @ offset)
        if e <= epsilon:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b, f = float(d1 @ d2), float(d2 @ offset)
            denominator = a * e - b * b
            s = 0.0 if denominator <= epsilon else float(
                np.clip((b * f - c * e) / denominator, 0.0, 1.0))
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    closest = (p1 + d1 * s, p2 + d2 * t)
    return float(np.linalg.norm(closest[0] - closest[1]))


def operator_wave(phase: float) -> tuple:
    """The 3-axis loop this tool sweeps: different phase per axis.

    A loop, not a line, so a single frozen axis shows up as a residual instead of
    hiding inside the motion.
    """
    return (math.sin(phase), 0.5 * math.sin(2.0 * phase), math.cos(phase))


def sweep_target(profile, amplitude: float, phase: float,
                 source: str = "base") -> tuple:
    """A smooth in-reach target: operator metres, or the profile's own working point."""
    wave = operator_wave(phase)
    if source == "operator":
        return tuple(amplitude * value for value in wave)
    offset = profile.metric_offset
    return tuple(offset[axis] + amplitude * wave[axis] for axis in range(3))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.simulation.arm_scene_driver",
        description="Drive the three-robot scene's arms per robot_type (PyBullet IK)")
    parser.add_argument("--robot", action="append", choices=loadable_robots(),
                        help="drive only these robots (repeatable; default: all loaded)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--amplitude", type=float, default=None,
                        help="half-range of the sweep in metres "
                             "(default: the profiles' own value)")
    parser.add_argument("--source", choices=("base", "operator"), default="base",
                        help="space of the sweep: robot base frame, or operator metres")
    parser.add_argument("--ik-mode", choices=("pose", "position"), default=None,
                        help="override the profile's ik_mode (A/B comparison)")
    parser.add_argument("--rest-poses", choices=("current", "zero", "live"),
                        default="current",
                        help="IK null-space reference: 'current' = where the arm was at "
                             "load time (a straight drooped arm -- it also *pulls* every "
                             "solution back there), 'zero' = all joints at 0, 'live' = "
                             "where the arm is right now (damps joint change instead of "
                             "dictating the elbow/upper arm; see arm_ik._live_rest_poses)")
    parser.add_argument("--arm-force", type=float, default=ARM_FORCE)
    parser.add_argument("--settle", type=int, default=15,
                        help="extra frames after the sweep before measuring")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--list-profiles", action="store_true",
                        help="print the per-robot table and exit")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _span_mm(points) -> float:
    """Largest single-axis span of a list of positions, in millimetres."""
    array = np.asarray(points, dtype=float).reshape(-1, 3)
    if array.size == 0:
        return 0.0
    return round(float(np.ptp(array, axis=0).max() * 1e3), 2)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_profiles:
        for name, entry in table().items():
            state = "loadable" if entry["loadable"] else "registered only"
            print(f"{name:<8} {state:<16} ik_mode={entry['ik_mode']:<9} "
                  f"scale={entry['metric_scale']:<5} "
                  f"offset={tuple(entry['metric_offset'])}")
            if not entry["loadable"]:
                print(f"{'':<8} reason: {entry['notes']}")
        return 0

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

    sides = tuple(SIDES) if args.side == "both" else (args.side,)
    scene = ThreeRobotHandScene(sides=sides, gui=not args.headless)
    try:
        driver = ArmSceneDriver(scene, sides=sides,
                                robots=tuple(args.robot) if args.robot else None,
                                arm_force=args.arm_force, ik_mode=args.ik_mode,
                                rest_poses=args.rest_poses)
    except BaseException:
        scene.close()
        raise
    try:
        for robot in sorted(driver.ik):
            adapter = driver.ik[robot]
            links = ", ".join(f"{side}:{link}" for side, (_i, link)
                              in sorted(adapter.ee.items()))
            print(f"  {robot:<8} ik_mode={adapter.ik_mode:<9} "
                  f"arm_joints={len(adapter.driven)} ee={links}")
        amplitude = args.amplitude
        if amplitude is None:
            values = [adapter.profile.sweep_amplitude_m
                      for adapter in driver.ik.values()]
            # One shared motion: base-frame targets are capped by the smallest working
            # box, operator metres get shrunk by the per-robot scales already.
            amplitude = max(values) if args.source == "operator" else min(values)
        frames = max(1, int(args.frames))
        measured = {robot: {side: [] for side in sides} for robot in driver.ik}
        targets = {robot: {side: [] for side in sides} for robot in driver.ik}
        for frame in range(frames):
            phase = 2.0 * math.pi * frame / frames
            for side in sides:
                if args.source == "operator":
                    driver.apply(side, sweep_target(None, amplitude, phase, "operator"),
                                 source="operator")
                else:
                    driver.apply_each(side, {
                        robot: sweep_target(adapter.profile, amplitude, phase)
                        for robot, adapter in driver.ik.items()}, source="base")
                for robot, adapter in driver.ik.items():
                    targets[robot][side].append(adapter.last_target_world[side])
                    measured[robot][side].append(adapter.measure(side)[0])
            driver.step()
        driver.step(max(0, int(args.settle)))
        payload = driver.summary()
        payload["sweep"] = {"amplitude_m": amplitude, "source": args.source,
                            "frames": frames, "settle_frames": int(args.settle)}
        payload["travel_mm"] = {robot: {side: _span_mm(measured[robot][side])
                                        for side in sides} for robot in driver.ik}
        payload["target_travel_mm"] = {robot: {side: _span_mm(targets[robot][side])
                                               for side in sides}
                                       for robot in driver.ik}
        for robot in sorted(driver.ik):
            adapter = driver.ik[robot]
            for side in sides:
                residual = payload["residual_mm"].get(robot, {}).get(side)
                print(f"  {robot:<8} {side:<5} residual={residual} mm "
                      f"travel={payload['travel_mm'][robot][side]:.1f} mm "
                      f"(target {payload['target_travel_mm'][robot][side]:.1f} mm) "
                      f"drift={payload['hold_drift'][robot]['max_drift_rad']:.4f} rad "
                      f"ee={adapter.ee[side][1]}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"report={args.report}")
        return 0
    finally:
        scene.close()


if __name__ == "__main__":
    raise SystemExit(main())




