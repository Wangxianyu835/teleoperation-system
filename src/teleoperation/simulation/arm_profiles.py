"""Per-robot arm profiles: pick the arm by ``robot_type``, not by hard-coding TRON2A.

Why this file exists
--------------------
The teammate's arm stack hard-codes TRON2A:

* ``robots/specification.py``  -> ``TRON2A_ARM_JOINTS`` / ``TRON2A_EE_LINKS``
* ``retargeting/arm/ik.py``    -> TRON2A joint order, TRON2A URDF, TRON2A calibration

Both are teammate-owned files (rule 3: do not edit them), so until now the three
paper robots could only be reached through a block of tuning constants buried in
``devtools/verify_arm_from_capture.py``.  This module lifts that knowledge into a
table keyed by ``robot_type``:

    profile = profile_for("h1_2")        # -> ArmProfile
    profile.ik_mode, profile.metric_scale, profile.metric_offset
    describe("gr1_t2")                   # -> JSON-able dict, for reports

``simulation/arm_ik.py`` (our PyBullet IK adapter) then solves for whichever robot
the scene loaded.  TRON2A is still registered here but with ``loadable=False``:
its URDF, calibration JSON and hand checkpoint live outside this repo, so it is
reported as unsupported instead of being silently driven with the paper robots'
convention.

The joint lists of the three paper robots are resolved from the loaded URDF at
runtime (``arm_joint_names``), so a profile stays valid if an upstream URDF renames
a joint; only the tuning numbers below are pinned to measured values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

SIDES = ("left", "right")

#: Robots whose URDF ships in ``assets/robots/from_teleopbench`` (local, not in git).
PAPER_ROBOTS = ("h1_2", "gr1_t2", "g1")

#: The teammate's arm as registered in ``retargeting/arm/ik.py``.  Kept visible on
#: purpose: this is the only place our side names it, and it stays ``loadable=False``.
TRON2A = "tron2a"


@dataclass(frozen=True)
class ArmProfile:
    """Everything the simulation layer needs to drive one robot's arm.

    ``metric_scale`` / ``metric_offset`` map an operator-space (metre) target to
    the robot's own base frame.  They are *tuning* numbers measured on this
    machine, not a hand-eye calibration (see ``docs/multi-robot-arm.md``).
    """

    robot_type: str
    #: Explicit joint names (URDF order); empty = resolve from the loaded model.
    side_joints: Mapping[str, tuple] = field(default_factory=dict)
    #: Link-name keywords used to find the hand mount, most specific first.
    ee_keywords: tuple = ("wrist", "hand")
    #: "pose" = constrain position and orientation, "position" = position only.
    ik_mode: str = "pose"
    #: Operator metre -> robot base-frame metre.
    metric_scale: float = 1.0
    #: Target centre in the robot's base frame (metres).
    metric_offset: tuple = (0.0, 0.0, 0.0)
    #: Realtime chain only: per-side travel clamp (metres, each base-frame axis).  Empty =
    #: fall back to ``sweep_amplitude_m``.  Measured by
    #: ``devtools/verify_realtime_arm_bridge.py --probe-travel`` (see docs section 7).
    realtime_travel_m: Mapping[str, float] = field(default_factory=dict)
    #: Realtime chain only: per-side neutral working point (metres, base frame).  Empty =
    #: use ``metric_offset``.  Measured by the same probe's ``--probe-lateral``.
    realtime_offset: Mapping[str, tuple] = field(default_factory=dict)
    #: Acceptance thresholds for the headless probe / tests.
    max_residual_mm: float = 30.0
    min_travel_mm: float = 10.0
    #: Sweep amplitude used by the demo and the probe (metres, base frame).
    sweep_amplitude_m: float = 0.05
    expected_arm_dof: int = 7
    #: False = registered but this repo cannot load/drive it.
    loadable: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.ik_mode not in ("pose", "position"):
            raise ValueError(
                f"{self.robot_type}: ik_mode must be 'pose' or 'position', "
                f"got {self.ik_mode!r}")
        if self.metric_scale <= 0.0:
            raise ValueError(f"{self.robot_type}: metric_scale must be > 0")
        if len(self.metric_offset) != 3:
            raise ValueError(f"{self.robot_type}: metric_offset must be (x, y, z)")
        for side, value in self.realtime_travel_m.items():
            if side not in SIDES:
                raise ValueError(
                    f"{self.robot_type}: realtime_travel_m key {side!r} is not one of {SIDES}")
            if not float(value) > 0.0:
                raise ValueError(
                    f"{self.robot_type}: realtime_travel_m[{side}] must be > 0, got {value!r}")
        for side, value in self.realtime_offset.items():
            if side not in SIDES:
                raise ValueError(
                    f"{self.robot_type}: realtime_offset key {side!r} is not one of {SIDES}")
            if len(tuple(value)) != 3:
                raise ValueError(
                    f"{self.robot_type}: realtime_offset[{side}] must be (x, y, z), got {value!r}")

    # ------------------------------------------------------------------ #
    def base_target_from_operator(self, position) -> tuple:
        """Operator-space (metre) target -> this robot's base frame.

        Pure arithmetic so it can be unit-tested without PyBullet; the base pose
        conversion (base frame -> world) belongs to ``PyBulletArmIK.to_world``.
        """
        values = tuple(float(value) for value in position)
        if len(values) != 3:
            raise ValueError(f"position must be (x, y, z), got {position!r}")
        return tuple(offset + self.metric_scale * value
                     for offset, value in zip(self.metric_offset, values))

    def reach(self, amplitude: float | None = None) -> tuple:
        """(min, max) corners of the box this profile sweeps inside the base frame."""
        amplitude = self.sweep_amplitude_m if amplitude is None else float(amplitude)
        low = tuple(value - amplitude for value in self.metric_offset)
        high = tuple(value + amplitude for value in self.metric_offset)
        return low, high

    # ------------------------------------------------------------------ #
    # 实时链路的每侧参数（实测覆盖值，不是标定）
    # ------------------------------------------------------------------ #
    def realtime_neutral(self, side: str) -> tuple:
        """这一侧的中性工作点（基座系，米）：实测覆盖值，缺省回落到 ``metric_offset``。"""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return tuple(float(value) for value in self.realtime_offset.get(side,
                                                                        self.metric_offset))

    def travel_limit(self, side: str) -> float:
        """这一侧每轴允许的最大位移（米）：实测覆盖值，缺省回落到 ``sweep_amplitude_m``。

        ``sweep_amplitude_m`` 是 ``devtools/verify_arm_multi_robot.py`` 验过的方框，
        所以在没有实测覆盖值的地方用它当上限是安全的。
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if side in self.realtime_travel_m:
            return float(self.realtime_travel_m[side])
        return float(self.sweep_amplitude_m)


#: Per-robot profiles.  Two measurement sets, both on this machine:
#: 2026-10-10 (devtools/verify_arm_from_capture.py: one robot, gravity 0, palm-derived
#: orientation): h1_2 10-16 mm, gr1_t2 22 mm, g1 18-33 mm; keeping the orientation
#: constraint helped on h1_2 and hurt on the other two.
#: 2026-10-12 (applications/three_robot_scene.py + simulation/arm_scene_driver.py, so
#: gravity -9.81, holds active, a full sweep at the profile amplitude, 40 frames, the
#: two-pass warm IK in PyBulletArmIK):
#:     h1_2   pose     steady mean 17.0 / max 23.8 mm, travel  99.5 of 100 mm
#:     gr1_t2 position steady mean  8.9 / max 13.3 mm, travel  60.5 of  60 mm
#:     g1     position steady mean  7.7 / max 14.9 mm, travel  59.6 of  60 mm
#:     arm force 200 / 500 / 1000 made no difference; non-driven joints drifted
#:     0.0003-0.0017 rad; the first frames after a cold start are 32-55 mm (the arm is
#:     still servoing in), hence the thresholds below describe steady state.
#: These are tuning values, not a hand-eye calibration; see docs/multi-robot-arm.md.
ARM_PROFILES: dict = {
    "h1_2": ArmProfile(
        robot_type="h1_2",
        ik_mode="pose",
        metric_scale=0.7,
        metric_offset=(0.24, 0.0, 0.19),
        # 2026-10-12 probe (both arms driven, sweep at the listed amplitude, 32 frames,
        # gravity on): left holds up to 0.10 m (mean 14.3 / max 27.8 mm), right up to
        # 0.16 m (mean 14.8 / max 29.8 mm); 0.20 m breaks both sides.
        realtime_travel_m={"left": 0.10, "right": 0.15},
        # 2026-10-12 lateral probe + real-session A/B (docs/multi-robot-arm.md 8.6):
        # with both hands on the centre line (y=0) both wrists are commanded to the *same*
        # point, so at rest the forearms converge in front of the chest and look crossed.
        # The shoulders sit at y=+-0.148.  Steady residual mean on the recorded 2280 frame
        # session: y=0 -> 16.8 (L) / 27.9 (R) mm, +-0.05 -> 22.7 / 22.7 mm, +-0.10 ->
        # 30.1 (L, over the ceiling) / 22.4 (R) mm.  +-0.05 leaves 10 cm between the
        # hands and keeps both sides inside max_residual_mm.
        realtime_offset={"left": (0.24, 0.05, 0.19), "right": (0.24, -0.05, 0.19)},
        max_residual_mm=30.0,
        min_travel_mm=10.0,
        sweep_amplitude_m=0.05,
        notes="7 DoF per arm; hand mount is right_wrist_yaw_link.  Sweep measured "
              "mean 17.0 / max 23.8 mm with the orientation constraint kept.",
    ),
    "gr1_t2": ArmProfile(
        robot_type="gr1_t2",
        ik_mode="position",
        metric_scale=0.4,
        metric_offset=(0.16, 0.0, 0.12),
        # Probe: left up to 0.10 m (mean 13.3 / max 28.2 mm), right up to 0.12 m
        # (mean 15.9 / max 26.3 mm); 0.16 m breaks both sides.
        realtime_travel_m={"left": 0.10, "right": 0.12},
        # Same reason as h1_2 (docs/multi-robot-arm.md 8.6): the shoulder line is at
        # y=+-0.095, so +-0.10 puts each hand under its own shoulder instead of both of
        # them on the centre line.  Real 2280 frame session: steady residual mean 7.8 ->
        # 18.4 (L) and 13.0 -> 13.4 (R) mm, both still inside max_residual_mm.
        realtime_offset={"left": (0.16, 0.10, 0.12), "right": (0.16, -0.10, 0.12)},
        max_residual_mm=25.0,
        min_travel_mm=6.0,
        sweep_amplitude_m=0.03,
        notes="no wrist link: the hand mount is right_hand_pitch_link, so the "
              "orientation constraint is dropped (ik_mode=position).  Sweep "
              "measured mean 8.9 / max 13.3 mm.",
    ),
    "g1": ArmProfile(
        robot_type="g1",
        ik_mode="position",
        metric_scale=0.4,
        metric_offset=(0.16, 0.0, 0.10),
        # g1 is the tight one: both sides only hold 0.04 m (left mean 19.8 / max 29.0 mm,
        # right mean 9.4 / max 21.6 mm); 0.06 m already spikes to 40+ mm.
        realtime_travel_m={"left": 0.04, "right": 0.04},
        # The left arm tracks much better with a lateral working point of +0.10 m: at
        # 0.04 m the mean drops from 31.2 mm (y=0) to 19.8 mm, and 0.06 m from 28.6 to
        # 19.9 mm.  The right arm's best point is the centre (18.3 mm), but -0.10 m only
        # costs +1.3 mm (19.6 mm) and puts the hand under its own shoulder (+-0.100),
        # which is what stops the right arm reaching across the chest at rest.  Real 2280
        # frame session: the left is unchanged at 22.5 mm (it already used +0.10) and the
        # right goes 11.8 -> 18.9 mm, still inside max_residual_mm.
        realtime_offset={"left": (0.16, 0.10, 0.10), "right": (0.16, -0.10, 0.10)},
        max_residual_mm=25.0,
        min_travel_mm=6.0,
        sweep_amplitude_m=0.03,
        notes="hand links exist but come later than the wrist, so the wrist is "
              "the clean IK target; position-only tracks better than the "
              "uncalibrated orientation convention.  Sweep measured mean 7.7 / "
              "max 14.9 mm.",
    ),
    # Registered only: this repo ships neither the TRON2A URDF nor its calibration
    # JSON nor the hand checkpoint, so the teammate's TRON2A path cannot run here.
    TRON2A: ArmProfile(
        robot_type=TRON2A,
        side_joints={},          # filled from the teammate's constants on demand
        ee_keywords=("grasper",),
        ik_mode="pose",
        metric_scale=1.0,
        metric_offset=(0.0, 0.0, 0.0),
        expected_arm_dof=7,
        loadable=False,
        notes="teammate's arm; needs third_party/tron2-robot-description/robot.urdf "
              "+ a calibration JSON + a hand checkpoint (none of them in this repo).",
    ),
}


def all_robots() -> tuple:
    """Every registered ``robot_type``, TRON2A included."""
    return tuple(ARM_PROFILES)


def loadable_robots() -> tuple:
    """The robots this repo can actually load and drive."""
    return tuple(name for name, profile in ARM_PROFILES.items() if profile.loadable)


def profile_for(robot_type: str) -> ArmProfile:
    """Profile of one ``robot_type``; raises with the known names on a typo."""
    try:
        profile = ARM_PROFILES[str(robot_type)]
    except KeyError:
        raise KeyError(
            f"unknown robot_type={robot_type!r}; registered: "
            f"{', '.join(all_robots())} (loadable here: "
            f"{', '.join(loadable_robots())})") from None
    if not profile.loadable:
        raise ValueError(
            f"robot_type={robot_type!r} is registered but not loadable in this "
            f"repo: {profile.notes}")
    return profile


def _tron2a_joint_names(side: str) -> tuple:
    """The teammate's TRON2A joint names, read (never written) from their spec."""
    from teleoperation.robots.specification import TRON2A_ARM_JOINTS

    return tuple(TRON2A_ARM_JOINTS[side])


def resolve_joint_names(profile: ArmProfile, body, cid, side: str) -> tuple:
    """Arm joint names of one side, in URDF order, resolved from the loaded body.

    Preference order: the profile's explicit list (TRON2A), else the teammate's
    ``_is_arm_joint`` naming rule applied to this body -- so one code path covers
    all three paper robots without a second hard-coded table.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    explicit = profile.side_joints.get(side) if profile.side_joints else None
    if explicit:
        return tuple(explicit)
    if profile.robot_type == TRON2A:
        return _tron2a_joint_names(side)
    import pybullet as p

    from teleoperation.robots.specification import _is_arm_joint

    names = []
    for index in range(p.getNumJoints(body, physicsClientId=cid)):
        info = p.getJointInfo(body, index, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED:
            continue
        name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
        if _is_arm_joint(name, side):
            names.append(name)
    return tuple(names)


def classify_arm_dof(body, cid) -> dict:
    """``{side: joint count}`` of the loaded body (7 per side is expected)."""
    import pybullet as p

    from teleoperation.robots.specification import _is_arm_joint

    counts = {side: 0 for side in SIDES}
    for index in range(p.getNumJoints(body, physicsClientId=cid)):
        info = p.getJointInfo(body, index, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED:
            continue
        name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
        for side in SIDES:
            if _is_arm_joint(name, side):
                counts[side] += 1
                break
    return counts


def describe(robot_type: str) -> dict:
    """JSON-able summary of one profile (for reports and ``--list-profiles``)."""
    profile = ARM_PROFILES[str(robot_type)]
    return {
        "robot_type": profile.robot_type,
        "loadable": profile.loadable,
        "ik_mode": profile.ik_mode,
        "metric_scale": profile.metric_scale,
        "metric_offset": list(profile.metric_offset),
        "realtime_offset": {side: list(profile.realtime_neutral(side)) for side in SIDES},
        "realtime_travel_m": {side: profile.travel_limit(side) for side in SIDES},
        "sweep_amplitude_m": profile.sweep_amplitude_m,
        "expected_arm_dof": profile.expected_arm_dof,
        "ee_keywords": list(profile.ee_keywords),
        "max_residual_mm": profile.max_residual_mm,
        "min_travel_mm": profile.min_travel_mm,
        "notes": profile.notes,
    }


def table() -> dict:
    """``{robot_type: describe(robot_type)}`` for every registered robot."""
    return {name: describe(name) for name in all_robots()}

