"""Common camera/VR wrist observations -> existing arm driver, no finger changes."""
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from teleoperation.retargeting.arm.wrist_orientation import WristTracker


class WristSceneController:
    def __init__(self, driver, camera_to_base=None, routes=None):
        self.driver = driver
        self.routes = routes or {side: side for side in driver.sides}
        if camera_to_base is None:
            from teleoperation.simulation.arm_realtime import AxisMap
            camera_to_base = AxisMap().matrix()
        self.trackers = {side: WristTracker(camera_to_base) for side in self.routes}
        self.anchors = {}
        self.generations = {}
        self.axis_lines = {}
        self.history = deque(maxlen=1800)
        self.recorded_frames = 0
        self.states = {}
        self.orientations = {side: {name: None for name in driver.ik} for side in self.routes}

    def update(self, wrists, dt):
        for side, source in self.routes.items():
            tracker = self.trackers[side]
            tracker.update((wrists or {}).get(source), dt)
            if self.states.get(side) != tracker.status:
                print(f"wrist {side}: {tracker.status} {tracker.reason}", flush=True)
                self.states[side] = tracker.status
            if tracker.neutral is None:
                self.orientations[side] = {name: None for name in self.driver.ik}
                continue
            for name, adapter in self.driver.ik.items():
                key = (name, side)
                if self.generations.get(key) != tracker.generation:
                    p, q = adapter.measure(side)
                    self.anchors[key] = (np.asarray(adapter.to_base(p)), Rotation.from_quat(q))
                    self.generations[key] = tracker.generation
                _, initial = self.anchors[key]
                # The driver positions are base-local; IK orientations are world-space.
                _, base_q = adapter.p.getBasePositionAndOrientation(adapter.body, physicsClientId=adapter.cid)
                base = Rotation.from_quat(base_q)
                target = base * tracker.delta * base.inv() * initial
                self.orientations[side][name] = tuple(target.as_quat())
                self._draw_axes(adapter, side, target)
        return self.orientations

    def _draw_axes(self, adapter, side, target):
        draw = getattr(adapter.p, "addUserDebugLine", None)
        if draw is None:
            return
        point, actual = adapter.measure(side)
        origin = np.asarray(point)
        for label, rotation, length in (("actual", Rotation.from_quat(actual), .06),
                                         ("target", target, .1)):
            axes = rotation.as_matrix()
            for axis, color in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                key = (adapter.robot_type, side, label, axis)
                self.axis_lines[key] = draw(
                    origin.tolist(), (origin + length * axes[:, axis]).tolist(), color,
                    lineWidth=2 if label == "target" else 4,
                    replaceItemUniqueId=self.axis_lines.get(key, -1),
                    physicsClientId=adapter.cid)

    def apply_vr(self, frame, dt):
        """VR has wrist translation, no fabricated shoulder/elbow observations."""
        self.update(None if frame is None else frame.wrists, dt)
        for side, tracker in self.trackers.items():
            if tracker.neutral is None:
                continue
            targets = {name: tuple(self.anchors[(name, side)][0] + tracker.displacement)
                       for name in self.driver.ik}
            self.driver.apply_each(side, targets, source="base", orientation=self.orientations[side])

    def record(self):
        """Call after physics; retain the latest 1800 frames for manual acceptance."""
        self.recorded_frames += 1
        self.history.append({"frame": self.recorded_frames,
            "timestamps": {s: t.last_timestamp for s, t in self.trackers.items()},
            "states": {s: t.status for s, t in self.trackers.items()},
            "target_xyzw": {s: dict(values) for s, values in self.orientations.items()},
            "robots": self._measurements()})

    def _measurements(self):
        return {name: {side: {
                    "actual_xyzw": list(adapter.measure(side)[1]),
                    "orientation_error_deg": (adapter.orientation_error_deg(side, self.orientations[side][name])
                        if self.orientations[side][name] is not None else None),
                    "position_error_mm": (adapter.residual_mm(side)
                        if side in adapter.last_target_world else None),
                    "ik": adapter.last_solve.get(side)} for side in self.routes}
                    for name, adapter in self.driver.ik.items()}

    def summary(self):
        return {"sides": {s: t.summary() for s, t in self.trackers.items()},
                "target_xyzw": {s: dict(values) for s, values in self.orientations.items()},
                "alignment_anchors": {name + "/" + side: {"position_base_m": p.tolist(),
                    "orientation_world_xyzw": r.as_quat().tolist()}
                    for (name, side), (p, r) in self.anchors.items()},
                "robots": self._measurements(), "frames_recorded": self.recorded_frames,
                "applied_frames": dict(self.driver.applied),
                "history_capacity": self.history.maxlen, "history": list(self.history)}


def vr_arm_controller(scene, routes):
    """Standalone hand scenes have no arms; robot scenes follow wrists by default."""
    if not hasattr(scene, "robots"):
        return None
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
    driver = ArmSceneDriver(scene, sides=tuple(routes), lock_others=True,
                            rest_poses="current", elbow_recovery_deg=0)
    return WristSceneController(driver, routes=routes)
