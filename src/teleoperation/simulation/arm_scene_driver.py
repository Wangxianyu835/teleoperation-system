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

from teleoperation.simulation.arm_ik import ARM_FORCE, PyBulletArmIK
from teleoperation.simulation.arm_profiles import (
    SIDES,
    describe,
    loadable_robots,
    profile_for,
    table,
)


class ArmSceneDriver:
    """Arms of an already-built ``ThreeRobotHandScene`` follow per-robot IK targets.

    ``scene`` may hold arms, hands or both; only the sides given to ``sides`` are
    driven, and only the robots named in ``robots`` (``None`` = every loaded one).
    A robot left out keeps its hold untouched, i.e. stays frozen exactly as before.
    The scene keeps owning the physics client, the ground plane and the finger map.
    """

    def __init__(self, scene, sides=("right",), robots=None, arm_force=ARM_FORCE,
                 ik_mode=None, rest_poses: str = "current",
                 ik_iterations=None, residual_threshold=None):
        self.scene = scene
        self.p = scene.p
        self.cid = scene.cid
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if not scene.robots:
            raise ValueError("the scene has no loaded robot to drive")
        self.ik = {}
        self.released = {}
        self.applied = {side: 0 for side in self.sides}
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
              orientation=None) -> dict:
        """One frame for one side with the *same* target for every robot.

        ``position`` is in the space named by ``source``; use
        :meth:`apply_each` when each robot needs its own base-frame target.
        """
        if side not in self.sides:
            raise ValueError(f"{side!r} is not driven by this driver ({list(self.sides)})")
        return self.apply_each(side, {robot_type: position for robot_type in self.ik},
                               source=source, orientation=orientation)

    def apply_each(self, side: str, targets, source: str = "base",
                   orientation=None) -> dict:
        """One frame for one side with a target per robot: ``{robot_type: position}``.

        Returns ``{robot_type: {"ee": link, "ik_mode": ..., "joints": {name: angle}}}``
        so a demo, a test or a report can state exactly which arm was driven.
        """
        if side not in self.sides:
            raise ValueError(f"{side!r} is not driven by this driver ({list(self.sides)})")
        missing = sorted(robot_type for robot_type in self.ik
                         if robot_type not in targets)
        if missing:
            raise ValueError(f"no target given for {missing} (driven: {list(self.ik)})")
        reports = {}
        for robot_type, adapter in self.ik.items():
            values = adapter.solve(side, targets[robot_type],
                                   orientation=orientation, source=source)
            adapter.apply(side, values)
            reports[robot_type] = {"ee": adapter.ee[side][1],
                                   "ik_mode": adapter.ik_mode,
                                   "joints": values}
        self.applied[side] += 1
        return reports

    def apply_operator(self, side: str, position, orientation=None) -> dict:
        """``apply`` with operator-space metres: the profile's scale/offset is applied."""
        return self.apply(side, position, source="operator", orientation=orientation)

    def step(self, frames: int = 1) -> None:
        """Advance the scene by ``frames`` rendered frames (``scene.step`` = 8 substeps)."""
        for _ in range(max(1, int(frames))):
            self.scene.step()

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
            "hold_drift": self.hold_drift(),
            "hands": self.scene.summary(),
            "robots": self.describe()["robots"],
        }


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
    parser.add_argument("--rest-poses", choices=("current", "zero"), default="current")
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




