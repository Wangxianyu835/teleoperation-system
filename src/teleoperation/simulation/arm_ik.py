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
per-robot tuning in ``arm_profiles.py``, and the orientation convention is not
calibrated either -- hence ``ik_mode="position"`` where it does not hold up.
"""

from __future__ import annotations

from dataclasses import replace
from math import acos, degrees, pi

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
                 rest_poses: str = "current"):
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

    def solve(self, side: str, position, orientation=None, source: str = "base",
              warm_start: bool = True) -> dict:
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
        """
        if side not in self.arm:
            raise ValueError(
                f"{side!r} is not a driven side of this adapter ({list(self.arm)})")
        world = self.target_in_world(position, source)
        arguments = {
            "lowerLimits": self.limits["lower"],
            "upperLimits": self.limits["upper"],
            "jointRanges": self.limits["range"],
            "restPoses": self.limits["rest"],
            "maxNumIterations": self.ik_iterations,
            "residualThreshold": self.residual_threshold,
            "physicsClientId": self.cid,
        }
        solution = self._ik_call(side, world, None, arguments)
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
        self.last_target_world[side] = world
        self.last_orientation[side] = target_orientation
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



