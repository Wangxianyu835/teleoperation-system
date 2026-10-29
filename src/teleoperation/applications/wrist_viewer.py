"""Camera/VR wrist orientation viewer: no robot, hand retargeter or arm IK.

Run with ``python -m teleoperation.applications.wrist_viewer``.
The asymmetric paddle follows source-space spatial rotation at unit gain.
"""
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.observations import RawHandFrame
from teleoperation.inputs.wrist import camera_wrists, recorded_camera_wrist
from teleoperation.paths import DEFAULT_MEDIAPIPE_ASSET


class OrientationDisplay:
    """Only alignment; deliberately no gain, smoothing, joint or velocity limits."""
    def __init__(self, samples=10):
        self.sample_count = samples
        self.reset()

    def reset(self):
        self.samples = []
        self.neutral = None
        self.signature = None
        self.last_timestamp = None
        self.rotation = Rotation.identity()
        self.state, self.reason = "waiting", "no valid wrist observation"

    def update(self, observation):
        if observation is None or not observation.orientation_valid:
            self.state = "hold" if self.neutral is not None else "waiting"
            self.reason = "no wrist observation" if observation is None else observation.reason
            self.samples.clear()
            return
        signature = (observation.reference_frame, observation.local_frame, observation.source)
        if self.signature != signature:
            self.reset()
            self.signature = signature
        stamp = observation.timestamp
        if stamp is not None:
            if not np.isfinite(stamp) or self.last_timestamp is not None and stamp <= self.last_timestamp:
                self.state, self.reason = "hold", "waiting for a new sample"
                return
            self.last_timestamp = stamp
        # Both are explicit axis conventions, not measured camera extrinsics.
        if observation.reference_frame == "mediapipe_hand_world_axes":
            c = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
        elif observation.reference_frame == "visionpro_sdk_zup":
            c = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        else:
            self.state, self.reason = "hold", "unsupported source reference"
            return
        current = Rotation.from_quat(observation.orientation_xyzw)
        if self.neutral is None:
            if self.samples and (current * self.samples[0].inv()).magnitude() > np.deg2rad(8):
                self.samples.clear()
            self.samples.append(current)
            self.state, self.reason = "aligning", f"hold palm steady {len(self.samples)}/{self.sample_count}"
            if len(self.samples) < self.sample_count:
                return
            self.neutral = Rotation.from_quat([r.as_quat() for r in self.samples]).mean()
        self.rotation = Rotation.from_matrix(c @ (current * self.neutral.inv()).as_matrix() @ c.T)
        self.state, self.reason = "tracking", ""

    def snapshot(self):
        return {"state": self.state, "reason": self.reason, "source_frames": self.signature,
                "source_timestamp": self.last_timestamp,
                "neutral_xyzw": None if self.neutral is None else self.neutral.as_quat().tolist(),
                "display_xyzw": self.rotation.as_quat().tolist(),
                "rotation_from_neutral_deg": float(np.rad2deg(self.rotation.magnitude()))}


class WristBlocks:
    def __init__(self, sides):
        import pybullet as p
        self.p = p
        self.cid = p.connect(p.GUI)
        if self.cid < 0:
            raise RuntimeError("Cannot open wrist viewer window")
        self.sides = sides
        self.lines, self.labels, self.bodies, self.origins = {}, {}, {}, {}
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
        p.resetDebugVisualizerCamera(1.65 if len(sides) == 2 else 1.1,
                                    45, -28, [0, 0, .32], physicsClientId=self.cid)
        for index, side in enumerate(sides):
            origin = np.array([0., (index - (len(sides) - 1)/2) * .65, .32])
            self.origins[side] = origin
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.16, .10, .035],
                rgbaColor=[1., .65, .08, 1.], physicsClientId=self.cid)
            # Long orange palm, blue forward nose, pink lateral thumb and white
            # top marker make all three rotation directions recognizable.
            extras = [([.055, .06, .04], [.205, 0, 0], [.1, .45, 1., 1.]),
                      ([.065, .045, .03], [.03, .14 if side == "left" else -.14, 0], [1., .15, .45, 1.]),
                      ([.07, .035, .008], [-.045, .02, .043], [1., 1., 1., 1.])]
            shapes = [p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color,
                       physicsClientId=self.cid) for size, _, color in extras]
            self.bodies[side] = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual, basePosition=origin.tolist(),
                linkMasses=[0.] * 3, linkCollisionShapeIndices=[-1] * 3,
                linkVisualShapeIndices=shapes, linkPositions=[e[1] for e in extras],
                linkOrientations=[[0, 0, 0, 1]] * 3,
                linkInertialFramePositions=[[0, 0, 0]] * 3,
                linkInertialFrameOrientations=[[0, 0, 0, 1]] * 3,
                linkParentIndices=[0] * 3, linkJointTypes=[p.JOINT_FIXED] * 3,
                linkJointAxis=[[0, 0, 0]] * 3, physicsClientId=self.cid)
        p.addUserDebugText("WRIST ONLY | 1:1 | R: re-align | Q/Esc: exit",
                          [-.3, -.42, .77], textSize=1.2, physicsClientId=self.cid)

    def draw(self, side, display):
        p, origin = self.p, self.origins[side]
        p.resetBasePositionAndOrientation(self.bodies[side], origin.tolist(),
                                         display.rotation.as_quat().tolist(), physicsClientId=self.cid)
        axes = display.rotation.as_matrix()
        for index, color in enumerate(([1, 0, 0], [0, 1, 0], [0, .4, 1])):
            key = (side, index)
            endpoint = origin + .33 * axes[:, index]
            self.lines[key] = p.addUserDebugLine(origin.tolist(), endpoint.tolist(), color,
                lineWidth=5, replaceItemUniqueId=self.lines.get(key, -1), physicsClientId=self.cid)
            self.labels[key] = p.addUserDebugText("XYZ"[index], endpoint.tolist(), color,
                textSize=1.5, replaceItemUniqueId=self.labels.get(key, -1), physicsClientId=self.cid)
        self.labels[side] = p.addUserDebugText(
            f"{side.upper()} {display.state} | {np.rad2deg(display.rotation.magnitude()):.1f} deg",
            (origin + [0, 0, -.28]).tolist(), textColorRGB=[.2, .2, .2],
            textSize=1.2, replaceItemUniqueId=self.labels.get(side, -1), physicsClientId=self.cid)

    def keys(self):
        return self.p.getKeyboardEvents(physicsClientId=self.cid)

    def alive(self):
        return bool(self.p.isConnected(self.cid))

    def close(self):
        if self.alive():
            self.p.disconnect(self.cid)


class CameraReplay:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.index = 0

    def next_observation(self):
        i = self.index
        if i >= len(self.data["timestamps"]):
            raise StopIteration
        self.index += 1
        stamp = float(self.data["timestamps"][i])
        world = {s: self.data.get("world_" + s + "_hand", [None] * (i + 1))[i]
                 for s in ("left", "right")}
        wrists = camera_wrists(world, stamp)
        for side in wrists:
            key = side + "_wrist_orientation_xyzw"
            if key in self.data:
                wrists[side] = recorded_camera_wrist(self.data[key][i], stamp,
                    str(self.data["wrist_reference_frame"]), str(self.data["wrist_local_frame"]))
        return RawHandFrame({"left": None, "right": None}, stamp, wrists=wrists)

    def close(self):
        pass


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camera", "visionpro"), default="camera")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--ip", help="Vision Pro address or room code")
    parser.add_argument("--input-h5", type=Path, help="raw Vision Pro recording")
    parser.add_argument("--input-npz", type=Path, help="raw camera wrist recording")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--frames", type=int, default=0)
    return parser


def run(args):
    if args.frames < 0 or not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("frames must be nonnegative and fps must be positive")
    if args.source == "camera" and (args.ip or args.input_h5):
        raise ValueError("Camera uses --camera-index or --input-npz")
    if args.source == "visionpro" and (args.input_npz or bool(args.ip) == bool(args.input_h5)):
        raise ValueError("Vision Pro requires exactly one of --ip and --input-h5")
    if args.report and (args.report.exists() or args.report.resolve() in
            {p.resolve() for p in (args.input_h5, args.input_npz, args.model_asset_path) if p}):
        raise ValueError("Choose a new report path, distinct from inputs")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    displays = {s: OrientationDisplay() for s in sides}
    history = deque(maxlen=6000)
    processed, states, cv2 = 0, {}, None
    with ExitStack() as resources:
        if args.source == "camera":
            if args.input_npz:
                source = CameraReplay(args.input_npz)
            else:
                from teleoperation.inputs.mediapipe import MediaPipeCameraInput
                source = MediaPipeCameraInput(args.model_asset_path, args.camera_index,
                                              invalid_hand_as_missing=True)
                import cv2
                resources.callback(cv2.destroyAllWindows)
        else:
            from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
            source = VisionProReplayInput(args.input_h5, args.side) if args.input_h5 else VisionProInput(args.ip, args.side)
        resources.callback(source.close)
        scene = WristBlocks(sides)
        resources.callback(scene.close)
        print("Hold palm steady for 10 fresh samples. Unit gain, no output smoothing. R re-aligns; Q/Esc exits.", flush=True)
        try:
            while scene.alive() and (not args.frames or processed < args.frames):
                tick = time.monotonic()
                try:
                    frame = source.next_observation()
                except StopIteration:
                    break
                for side, display in displays.items():
                    display.update(None if frame is None else (frame.wrists or {}).get(side))
                    scene.draw(side, display)
                    state = (display.state, display.reason)
                    if states.get(side) != state:
                        print(side, *state, flush=True)
                        states[side] = state
                processed += 1
                history.append({"frame": processed, "sides": {s: d.snapshot() for s, d in displays.items()}})
                keys = scene.keys()
                pressed = lambda key: bool(keys.get(key, 0) & scene.p.KEY_WAS_TRIGGERED)
                quit_requested = any(pressed(k) for k in (ord("q"), 27))
                realign = pressed(ord("r"))
                if cv2 is not None:
                    image = source.last_bgr.copy()
                    for i, side in enumerate(sides):
                        cv2.putText(image, f"{side}: {displays[side].state} | 1:1 | R align | Q exit",
                                    (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1)
                    cv2.imshow("Wrist input (unmirrored)", image)
                    key = cv2.waitKey(1) & 255
                    quit_requested |= key in (ord("q"), 27)
                    realign |= key == ord("r")
                if realign:
                    for display in displays.values():
                        display.reset()
                if quit_requested:
                    break
                time.sleep(max(0., 1 / args.fps - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump({"entry": "wrist_viewer", "source": args.source, "gain": 1.,
                "smoothing": False, "robot": None, "ik": False, "frames": processed,
                "sides": {s: d.snapshot() for s, d in displays.items()},
                "history_capacity": history.maxlen, "history": list(history)}, stream, indent=2)
        print(f"report written: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    run(build_parser().parse_args())
