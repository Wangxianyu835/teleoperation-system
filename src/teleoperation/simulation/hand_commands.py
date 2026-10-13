"""Output filtering, rate limits and sampled L21 inter-finger collision checks.

The low-pass recurrence follows dex-retargeting's LPFilter. This layer operates
on native-side robot angles after prediction, leaving checkpoint semantics intact.
Collision queries are a simulator guard, not a hardware safety controller.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from teleoperation.contracts.hand import L21HandAngles
from teleoperation.data.urdf import parse_joint_limits
from teleoperation.paths import L21_ROOT
from teleoperation.robots.l21 import MAPPING_18
from teleoperation.retargeting.hand.postprocessing import KalmanSmoother
from teleoperation.cli.hand_command_options import add_hand_command_arguments


class L21CollisionGuard:
    """Check the actual URDF collision meshes along a sampled joint-space path.

Same-finger and palm pairs are excluded. The shipped neighboring metacarpal
meshes already touch at zero pose; only those tiny initial contacts receive a
recorded 0.1 mm tolerance. New/deeper inter-finger contacts remain forbidden.
"""

    def __init__(self, scene, side, margin=0.001, sample_step=0.02):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("collision margin must be finite and nonnegative")
        self.scene, self.side = scene, side
        self.p, self.cid, self.body = scene.p, scene.cid, scene.bodies[side]
        self.margin, self.sample_step = float(margin), float(sample_step)
        self.names = scene.joint_index[side]
        self.indices = [self.names[name] for name in MAPPING_18 if name]
        self.link_names = {}
        groups = {}
        for index in range(self.p.getNumJoints(self.body, physicsClientId=self.cid)):
            name = self.p.getJointInfo(self.body, index, physicsClientId=self.cid)[12].decode()
            finger = next((f for f in ("thumb", "index", "middle", "ring", "pinky")
                           if name.startswith(f + "_")), None)
            if finger and self.p.getCollisionShapeData(self.body, index, physicsClientId=self.cid):
                self.link_names[index], groups[index] = name, finger
        self.pairs = [(a, b) for a, b in itertools.combinations(groups, 2)
                      if groups[a] != groups[b]]
        if not self.pairs:
            raise RuntimeError("L21 collision meshes could not be found")
        self.floors = {pair: self.margin for pair in self.pairs}
        self.baseline_contacts = []
        for a, b in self.pairs:
            distance = self._distance(a, b)
            if distance is None or distance >= self.margin:
                continue
            names = (self.link_names[a], self.link_names[b])
            if not all(name.endswith("_metacarpals") for name in names) or distance < -0.0002:
                raise RuntimeError(f"Initial L21 pose is inside collision margin: {names}, {distance}")
            self.floors[a, b] = distance - 0.0001
            self.baseline_contacts.append({"links": list(names), "initial_distance_m": distance,
                                           "allowed_distance_m": self.floors[a, b]})
        self.blocked = 0
        self.last_contact = None

    def _distance(self, a, b):
        points = self.p.getClosestPoints(self.body, self.body, self.margin,
                                        linkIndexA=a, linkIndexB=b, physicsClientId=self.cid)
        return min((float(point[8]) for point in points), default=None)

    def _set_pose(self, angles):
        targets = self.scene.adapters[self.side].map(L21HandAngles(angles).native_mapping_dofs())
        previous = getattr(self, "_query_targets", None)
        for name, value in targets.items():
            if previous is not None and previous.get(name) == value:
                continue
            self.p.resetJointState(self.body, self.names[name], value, physicsClientId=self.cid)
        self._query_targets = targets

    def violations(self):
        """Inspect the current pose; callers must not confuse this with a guarantee."""
        boxes = {index: self.p.getAABB(self.body, index, physicsClientId=self.cid)
                 for index in self.link_names}
        for a, b in self.pairs:
            lo_a, hi_a = boxes[a]
            lo_b, hi_b = boxes[b]
            # Conservative broad phase before the mesh distance query.
            if any(lo_a[k] > hi_b[k] + self.margin or lo_b[k] > hi_a[k] + self.margin
                   for k in range(3)):
                continue
            distance = self._distance(a, b)
            if distance is not None and distance < self.floors[a, b] - 1e-9:
                return {"links": [self.link_names[a], self.link_names[b]],
                        "distance_m": distance, "required_distance_m": self.floors[a, b]}
        return None

    def project(self, previous, candidate):
        """Stop at the first unsafe segment, then restore the live simulator state."""
        saved = self.p.getJointStates(self.body, self.indices, physicsClientId=self.cid)
        self._query_targets = None
        try:
            self._set_pose(previous)
            contact = self.violations()
            if contact is not None:
                raise RuntimeError(f"Last accepted L21 command is no longer safe: {contact}")
            projected, contact = self._segment(previous, candidate)
            if contact is not None:
                self.blocked += 1
                self.last_contact = contact
                # Preserve feasible flexion rather than freezing the whole hand
                # when one MCP side-swing blocks the original joint-space ray.
                for dimension in (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 1, 4, 7, 10):
                    proposal = projected.copy()
                    proposal[dimension] = candidate[dimension]
                    projected, _ = self._segment(projected, proposal)
                # The scene applies all joints together. Recheck that simultaneous
                # path as well, rather than relying on the coordinate-wise path.
                projected, _ = self._segment(previous, projected)
            return projected
        finally:
            for index, state in zip(self.indices, saved):
                self.p.resetJointState(self.body, index, state[0], targetVelocity=state[1],
                                       physicsClientId=self.cid)
            self._query_targets = None

    def _segment(self, previous, candidate):
        delta = candidate - previous
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / self.sample_step)))
        safe_fraction = 0.0
        for index in range(1, steps + 1):
            fraction = index / steps
            self._set_pose(previous + fraction * delta)
            contact = self.violations()
            if contact is None:
                safe_fraction = fraction
                continue
            # Refine the first unsafe interval, rather than assuming a clear
            # endpoint means the whole motion is clear.
            low, high = safe_fraction, fraction
            for _ in range(8):
                middle = (low + high) / 2
                self._set_pose(previous + middle * delta)
                if self.violations() is None:
                    low = middle
                else:
                    high = middle
            return previous + low * delta, contact
        return candidate.copy(), None

    def summary(self):
        return {"scope": "same-hand different-finger URDF collision meshes",
                "pairs": len(self.pairs), "margin_m": self.margin,
                "path_sample_step_rad": self.sample_step, "blocked_commands": self.blocked,
                "last_contact": self.last_contact, "baseline_contacts": self.baseline_contacts}


class HandCommandPipeline:
    """Independent per-side output states seeded from the current hand position."""

    def __init__(self, sides, limits, initial=None, guards=None, filter_mode="lowpass",
                 alpha=0.2, max_velocity=1.5, kalman_q=1., kalman_r=0.01, clock=time.monotonic):
        if filter_mode not in ("lowpass", "kalman", "none"):
            raise ValueError("Unknown joint filter")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("filter alpha must be in (0, 1]")
        if not math.isfinite(max_velocity) or max_velocity <= 0:
            raise ValueError("joint max velocity must be finite and positive")
        if not math.isfinite(kalman_q) or kalman_q <= 0 or not math.isfinite(kalman_r) or kalman_r <= 0:
            raise ValueError("Kalman noise parameters must be finite and positive")
        self.sides, self.mode, self.alpha = tuple(sides), filter_mode, float(alpha)
        self.max_velocity, self.q, self.r = float(max_velocity), float(kalman_q), float(kalman_r)
        self.clock, self.limits, self.guards = clock, limits, guards or {}
        self.previous, self.filtered, self.times, self.kalman, self.stats = {}, {}, {}, {}, {}
        for side in self.sides:
            self.previous[side] = np.clip(np.asarray((initial or {}).get(side, np.zeros(18)), dtype=float),
                                          limits[side][:, 0], limits[side][:, 1])
            self.times[side] = None
            self.stats[side] = {"commands": 0, "held_frames": 0, "rate_limited": 0,
                                "limit_clipped": 0, "max_raw_delta_rad": 0., "max_command_delta_rad": 0.}
            self._seed(side)

    def _seed(self, side):
        self.filtered[side] = self.previous[side].copy()
        kf = KalmanSmoother(process_noise=self.q, measurement_noise=self.r, gate=0.,
                            max_velocity=self.max_velocity)
        kf.step(self.previous[side])
        self.kalman[side] = kf

    def missing(self, side):
        # Never extrapolate a missing hand or reset its next command to zero.
        self.stats[side]["held_frames"] += 1
        self.times[side] = None
        self._seed(side)

    def update(self, side, raw, timestamp=None):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (18,) or not np.isfinite(raw).all():
            raise ValueError("Robot hand command must be a finite 18-vector")
        now = self.clock() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("Command timestamp must be finite")
        before = self.previous[side]
        last_time = self.times[side]
        if last_time is not None and now <= last_time:
            return before.copy()
        elapsed = 1 / 30 if last_time is None else now - last_time
        if elapsed > 0.25:
            self._seed(side)
        # Long gaps must not permit a large reacquisition jump.
        dt = min(elapsed, 0.1)
        self.times[side] = now
        bounds = self.limits[side]
        target = np.clip(raw, bounds[:, 0], bounds[:, 1])
        stats = self.stats[side]
        stats["commands"] += 1
        stats["limit_clipped"] += int(not np.allclose(raw, target))
        stats["max_raw_delta_rad"] = max(stats["max_raw_delta_rad"], float(np.max(np.abs(target - before))))
        if self.mode == "lowpass":
            weight = 1 - (1 - self.alpha) ** (dt * 30)
            self.filtered[side] += weight * (target - self.filtered[side])
            filtered = self.filtered[side]
        elif self.mode == "kalman":
            self.kalman[side].dt = dt
            filtered = self.kalman[side].step(target)
        else:
            filtered = target
        filtered = np.clip(filtered, bounds[:, 0], bounds[:, 1])
        delta = filtered - before
        stats["rate_limited"] += int(np.any(np.abs(delta) > self.max_velocity * dt))
        # A common fraction preserves the direction of the joint-space segment.
        fraction = min(1., self.max_velocity * dt / max(float(np.max(np.abs(delta))), 1e-12))
        command = before + fraction * delta
        guard = self.guards.get(side)
        if guard is not None:
            projected = guard.project(before, command)
            if not np.allclose(projected, command, atol=1e-9):
                command = projected
                self.previous[side] = command.copy()
                self._seed(side)  # No accumulating filter error behind a blocked motion.
            else:
                command = projected
        self.previous[side] = command.copy()
        stats["max_command_delta_rad"] = max(stats["max_command_delta_rad"],
                                              float(np.max(np.abs(command - before))))
        return command.copy()

    def summary(self):
        return {"filter": self.mode, "alpha_at_30hz": self.alpha,
                "max_velocity_rad_s": self.max_velocity, "max_update_dt_s": 0.1,
                "kalman_q": self.q, "kalman_r": self.r,
                "sides": {side: dict(stats) for side, stats in self.stats.items()},
                "collision_guard": {side: guard.summary() for side, guard in self.guards.items()}}


def build_hand_commands(args, scene, sides):
    limits, initial, guards = {}, {}, {}
    l21 = hasattr(scene, "bodies") and hasattr(scene, "adapters")
    for side in sides:
        bounds = parse_joint_limits(L21_ROOT / side / f"linkerhand_l21_{side}.urdf")
        limits[side] = np.asarray([(0., 0.) if name is None else bounds.get(name, (0., 0.))
                                   for name in MAPPING_18], dtype=float)
        if l21:
            joints = scene.joint_values(side)
            initial[side] = np.asarray([joints.get(name, 0.) for name in MAPPING_18], dtype=float)
            if not args.no_collision_guard:
                guards[side] = L21CollisionGuard(scene, side, args.collision_margin_mm / 1000)
    commands = HandCommandPipeline(sides, limits, initial, guards, args.joint_filter,
                                   args.filter_alpha, args.joint_max_velocity, args.kalman_q, args.kalman_r)
    scope = "L21 inter-finger mesh checks" if guards else "no runtime mesh collision guard for this scene"
    print(f"hand commands: filter={args.joint_filter} speed<={args.joint_max_velocity} rad/s; {scope}", flush=True)
    return commands
