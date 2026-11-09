"""Output filtering, rate limits and sampled L21 inter-finger collision checks.

The low-pass recurrence follows dex-retargeting's LPFilter. This layer operates
on native-side robot angles after prediction, leaving checkpoint semantics intact.
Collision queries are a simulator guard, not a hardware safety controller.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from teleoperation.robots.l21 import L21_JOINT_ORDER
from teleoperation.cli.hand_command_options import add_hand_command_arguments


class L21CollisionGuard:
    """Check the actual URDF collision meshes along a sampled joint-space path.

Same-finger and palm pairs are excluded. The shipped neighboring metacarpal
meshes already touch at zero pose; those tiny initial contacts get the caller's
margin (``--collision-margin-mm``, 1 mm by default) instead of a fixed 0.1 mm.
New/deeper inter-finger contacts remain forbidden.
"""

    def __init__(self, scene, side, margin=0.001, sample_step=0.02):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("collision margin must be finite and nonnegative")
        self.scene, self.side = scene, side
        self.p, self.cid, self.body = scene.p, scene.cid, scene.bodies[side]
        self.margin, self.sample_step = float(margin), float(sample_step)
        self.names = scene.joint_index[side]
        self.indices = [self.names[name] for name in L21_JOINT_ORDER]
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
            self.floors[a, b] = distance - self.margin
            self.baseline_contacts.append({"links": list(names), "initial_distance_m": distance,
                                           "allowed_distance_m": self.floors[a, b]})
        self.blocked = 0
        self.last_contact = None

    def _distance(self, a, b):
        points = self.p.getClosestPoints(self.body, self.body, self.margin,
                                        linkIndexA=a, linkIndexB=b, physicsClientId=self.cid)
        return min((float(point[8]) for point in points), default=None)

    def _set_pose(self, angles):
        targets = self.scene.adapters[self.side].map(angles)
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
                for dimension in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 0, 3, 6, 9):
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


def build_hand_commands(args, scene, sides):
    from teleoperation.robots.hand_spec import load_hand_spec
    from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
    specs = {('l21',side):load_hand_spec('l21',side) for side in sides}
    initial, guards = {}, {}
    l21 = hasattr(scene, "bodies") and hasattr(scene, "adapters")
    for side in sides:
        key = ('l21',side)
        if l21:
            joints = scene.joint_values(side)
            initial[key] = np.asarray([joints.get(name, 0.) for name in specs[key].independent_joint_names], dtype=float)
            if not args.no_collision_guard:
                guards[key] = L21CollisionGuard(scene, side, args.collision_margin_mm / 1000)
    commands = JointCommandPipeline(specs,filter_mode=args.joint_filter,alpha=args.filter_alpha,
                                    max_velocity=args.joint_max_velocity,kalman_q=args.kalman_q,
                                    kalman_r=args.kalman_r,initial=initial,guards=guards)
    scope = "L21 inter-finger mesh checks" if guards else "no runtime mesh collision guard for this scene"
    print(f"hand commands: filter={args.joint_filter} speed<={args.joint_max_velocity} rad/s; {scope}", flush=True)
    return commands
