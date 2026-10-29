"""Sampled arm-vs-body self-distance guard; restores all queried joint states.

This is a command rejection layer for the available collision meshes, not a
collision-free IK planner or a guarantee about physical servo tracking.
"""
import numpy as np


class ArmCollisionGuard:
    def __init__(self, adapter):
        self.adapter = adapter
        self.p, self.body, self.cid = adapter.p, adapter.body, adapter.cid
        infos = {i: self.p.getJointInfo(self.body, i, physicsClientId=self.cid)
                 for i in range(self.p.getNumJoints(self.body, physicsClientId=self.cid))}
        self.parents = {i: info[16] for i, info in infos.items()}
        self.movable = {i for i, info in infos.items() if info[2] != self.p.JOINT_FIXED}
        self.shapes = {i for i in [-1, *infos]
                       if self.p.getCollisionShapeData(self.body, i, physicsClientId=self.cid)}
        self.hand_links = {}
        self.arm_links = {}
        for side, entries in adapter.arm.items():
            root = entries[0][0]
            hand_root = adapter.ee[side][0]
            self.arm_links[side] = {i for i in infos if self._descends(i, root)}
            self.hand_links[side] = {i for i in infos if self._descends(i, hand_root)}
        # Preserve the loaded model's existing mesh overlaps, but forbid making
        # those overlaps deeper. Other nonadjacent pairs require 3 mm clearance.
        self.floors = {}
        for pair, distance in self._distances().items():
            if self._eligible(*pair):
                self.floors[pair] = min(.003, distance - .0005)
        self.blocked = 0
        self.last_contact = None

    def _descends(self, link, ancestor):
        while link >= 0:
            if link == ancestor:
                return True
            link = self.parents.get(link, -1)
        return False

    def _near(self, a, b):
        # Count moving joints rather than mesh/fixed spacer links.
        for start, goal in ((a, b), (b, a)):
            count, link = 0, start
            while link >= 0:
                if link == goal:
                    return count <= 2
                count += int(link in self.movable)
                link = self.parents.get(link, -1)
                if count > 2:
                    break
        return False

    def _eligible(self, a, b, side=None):
        if a == b or a not in self.shapes or b not in self.shapes or self._near(a, b):
            return False
        if any(a in hand and b in hand for hand in self.hand_links.values()):
            return False  # Finger-vs-finger handling belongs to the hand guard.
        links = self.arm_links[side] if side is not None else set().union(*self.arm_links.values())
        return a in links or b in links

    def _distances(self):
        result = {}
        for contact in self.p.getClosestPoints(self.body, self.body, .003, physicsClientId=self.cid):
            pair = tuple(sorted((int(contact[3]), int(contact[4]))))
            result[pair] = min(result.get(pair, float("inf")), float(contact[8]))
        return result

    def project(self, side, candidate):
        adapter = self.adapter
        indices = adapter.arm_ids[side]
        saved = self.p.getJointStates(self.body, indices, physicsClientId=self.cid)
        previous = np.asarray([state[0] for state in saved])
        wanted = np.asarray([candidate[name] for _, name in adapter.arm[side]])
        safe = previous.copy()
        steps = max(1, int(np.ceil(np.max(np.abs(wanted - previous)) / .025)))
        try:
            for fraction in np.linspace(0., 1., steps + 1)[1:]:
                values = previous + fraction * (wanted - previous)
                for index, value in zip(indices, values):
                    self.p.resetJointState(self.body, index, float(value), physicsClientId=self.cid)
                for pair, distance in self._distances().items():
                    if self._eligible(*pair, side=side) and distance < self.floors.get(pair, .003) - 1e-6:
                        self.blocked += 1
                        self.last_contact = {"links": list(pair), "distance_m": distance,
                                             "required_distance_m": self.floors.get(pair, .003)}
                        return {name: float(value) for (_, name), value in zip(adapter.arm[side], safe)}
                safe = values.copy()
            return candidate
        finally:
            for index, state in zip(indices, saved):
                self.p.resetJointState(self.body, index, state[0], targetVelocity=state[1], physicsClientId=self.cid)

    def summary(self):
        return {"scope": "sampled nonadjacent arm-vs-body collision meshes",
                "shape_links": len(self.shapes), "blocked": self.blocked,
                "sample_step_rad": .025, "margin_m": .003,
                "last_contact": self.last_contact}
