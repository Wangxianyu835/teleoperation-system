"""Per-robot arm configuration for H1-2, GR1-T2 and G1.

Profiles own axis conventions, target offsets, travel bounds and IK modes.
TRON2A uses its separate calibrated control route. Joint names for the three
scene robots are resolved from their loaded URDFs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

SIDES = ("left", "right")

#: Robots whose URDF ships in ``assets/robots``.
PAPER_ROBOTS = ("h1_2", "gr1_t2", "g1")

TRON2A = "tron2a"

#: Direction frames: mapped uses mapped directions; delta applies neutral-relative rotations.
POSTURE_FRAMES = ("mapped", "delta")


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
    posture_frame: str = "mapped"
    #: Operator metre -> robot base-frame metre.
    metric_scale: float = 1.0
    #: Target centre in the robot's base frame (metres).
    metric_offset: tuple = (0.0, 0.0, 0.0)
    #: Chain displacement limit per side and base-frame axis, in metres.
    #: An empty mapping falls back to sweep_amplitude_m.
    realtime_travel_m: Mapping[str, float] = field(default_factory=dict)
    #: Chain neutral working point per side in base-frame metres.
    #: An empty mapping falls back to metric_offset.
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
        if self.posture_frame not in POSTURE_FRAMES:
            raise ValueError(
                f"{self.robot_type}: posture_frame must be one of {POSTURE_FRAMES},"
                f" got {self.posture_frame!r}")
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

    def realtime_neutral(self, side: str) -> tuple:
        """Return the side-specific base-frame neutral point in metres."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        return tuple(float(value) for value in self.realtime_offset.get(side,
                                                                        self.metric_offset))

    def travel_limit(self, side: str) -> float:
        """Return the side-specific per-axis displacement limit in metres."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if side in self.realtime_travel_m:
            return float(self.realtime_travel_m[side])
        return float(self.sweep_amplitude_m)


#: Per-robot control profiles.
ARM_PROFILES: dict = {
    "h1_2": ArmProfile(
        robot_type="h1_2",
        ik_mode="pose",
        metric_scale=0.7,
        metric_offset=(0.24, 0.0, 0.19),
        realtime_travel_m={"left": 0.10, "right": 0.15},
        realtime_offset={"left": (0.24, 0.05, 0.19), "right": (0.24, -0.05, 0.19)},
        max_residual_mm=30.0,
        min_travel_mm=10.0,
        sweep_amplitude_m=0.05,
        notes='7 DoF per arm; hand mount is right_wrist_yaw_link.',
    ),
    "gr1_t2": ArmProfile(
        robot_type="gr1_t2",
        ik_mode="position",
        posture_frame="delta",
        metric_scale=0.4,
        metric_offset=(0.16, 0.0, 0.12),
        realtime_travel_m={"left": 0.10, "right": 0.12},
        realtime_offset={"left": (0.16, 0.10, 0.12), "right": (0.16, -0.10, 0.12)},
        max_residual_mm=25.0,
        min_travel_mm=6.0,
        sweep_amplitude_m=0.03,
        notes='no wrist link: the hand mount is right_hand_pitch_link, so the orientation constraint is dropped (ik_mode=position).',
    ),
    "g1": ArmProfile(
        robot_type="g1",
        ik_mode="position",
        metric_scale=0.4,
        metric_offset=(0.16, 0.0, 0.10),
        realtime_travel_m={"left": 0.04, "right": 0.06},
        realtime_offset={"left": (0.16, 0.10, 0.10), "right": (0.16, -0.10, 0.10)},
        max_residual_mm=25.0,
        min_travel_mm=6.0,
        sweep_amplitude_m=0.03,
        notes='hand links exist but come later than the wrist, so the wrist is the clean IK target; position-only tracks better than the uncalibrated orientation convention.',
    ),
    TRON2A: ArmProfile(
        robot_type=TRON2A,
        side_joints={},          # filled from the teammate's constants on demand
        ee_keywords=("grasper",),
        ik_mode="pose",
        metric_scale=1.0,
        metric_offset=(0.0, 0.0, 0.0),
        expected_arm_dof=7,
        loadable=False,
        notes="separate calibrated route; uses assets/robots/tron2a/robot.urdf "
              "and a calibration JSON; hand control also needs a hand checkpoint.",
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
    """Read TRON2A arm joint names from the robot specification."""
    from teleoperation.robots.specification import TRON2A_ARM_JOINTS

    return tuple(TRON2A_ARM_JOINTS[side])


def resolve_joint_names(profile: ArmProfile, body, cid, side: str) -> tuple:
    """Resolve one side's arm joint names in URDF order from the loaded body."""
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
        "posture_frame": profile.posture_frame,
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
