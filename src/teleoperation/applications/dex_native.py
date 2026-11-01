"""Native dex application composition and explicit Vision Pro calibration/I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from teleoperation.data.hand_recording import iter_dex_observations
from teleoperation.data.hand_joint_recording import HandJointResultWriter
from teleoperation.retargeting.hand.dex_native import DexInputAdapter, DexCalibration, DexNativeRetargeter, VECTOR_PARAMETERS


class LatestCameraInput:
    """One observation slot, overwritten by capture; never accumulate latency."""
    def __init__(self, source):
        import threading
        self.source=source
        self.condition=threading.Condition()
        self.stop=threading.Event()
        self.thread=None
        self.latest=None
        self.error=None
        self.sequence=0
        self.consumed=0
        self.operation=None

    @property
    def _cap(self):
        return self.source._cap

    @_cap.setter
    def _cap(self,value):
        self.source._cap=value

    def _read(self,operation,kwargs):
        import threading
        if self.thread is None:
            self.operation=(operation,kwargs)
            def capture():
                try:
                    while not self.stop.is_set():
                        observation=getattr(self.source,operation)(**kwargs)
                        with self.condition:
                            self.latest=observation
                            self.sequence+=1
                            self.condition.notify_all()
                except BaseException as error:
                    with self.condition:
                        self.error=error
                        self.condition.notify_all()
            self.thread=threading.Thread(target=capture,name="dex-latest-camera",daemon=True)
            self.thread.start()
        elif self.operation!=(operation,kwargs):
            raise ValueError("Latest camera source cannot mix acquisition profiles")
        with self.condition:
            ready=self.condition.wait_for(lambda:self.sequence>self.consumed or self.error is not None or self.stop.is_set(),timeout=5.)
            if self.error is not None:
                raise self.error
            if not ready or self.stop.is_set():
                raise RuntimeError("Latest camera capture stopped or timed out")
            self.consumed=self.sequence
            return self.latest

    def next_observation(self,**kwargs):
        return self._read("next_observation",kwargs)

    def next_teleop_observation(self):
        return self._read("next_teleop_observation",{})

    def release(self,**kwargs):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        # Unblock a pending camera read, but never destroy a landmarker while
        # the producer is still invoking it.
        capture=getattr(self.source,"_cap",None)
        if capture is not None:
            capture.release()
        if self.thread is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():
                raise RuntimeError("Camera inference thread did not close within five seconds")
        self.source.release(**kwargs)

    def close(self):
        return self.release(close_landmarker=True)




def canonical_window(frame):
    """Declare the explicit single-frame VR profile as a hand window."""
    from teleoperation.contracts.hand import HandWindow
    return HandWindow({side: None if points is None else points[None]
                       for side, points in frame.hands.items()}, frame.timestamp, frame.source, frame.metadata)






def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_frames(path, source):
    adapter = DexInputAdapter(source)
    return [adapter.adapt(raw) for raw in iter_dex_observations(path, source)]


def calibrate(args):
    if args.source != "visionpro":
        raise ValueError("Fixed dex calibration supports source visionpro only")
    frames = recording_frames(args.input, args.source)
    if not frames:
        raise ValueError("Empty calibration recording")
    calibration = DexCalibration.fit(frames, frames[0].timestamp + args.start, args.duration, file_hash(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(args.output)
    print(f"Saved fixed {calibration.profile} calibration: {args.output}")
    return 0


def command_pipeline(args, models):
    from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
    from teleoperation.robots.hand_spec import load_hand_spec
    return JointCommandPipeline({key: load_hand_spec(*key) for key in models}, args.joint_filter, args.filter_alpha,
        args.joint_max_velocity, args.kalman_q, args.kalman_r)


def visionpro_run(args):
    """Explicit native VR route. Device get_latest() supplies no backlog queue."""
    from contextlib import ExitStack
    import torch
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.inputs.visionpro import VisionProInput
    if args.scene != "robots" or args.dex_calibration is None:
        raise ValueError("VR native dex requires --scene robots and --dex-calibration")
    if args.checkpoint is not None or args.scale_factor != 1.:
        raise ValueError("VR native dex uses no checkpoint and requires scale-factor=1 (metres)")
    if args.source_hand not in (None, args.hand) or (args.hand == "both" and args.hand_mapping != "same"):
        raise ValueError("VR native dex uses same-side routing; both requires --hand-mapping same")
    if args.frames < 0 or not np.isfinite(args.fps) or args.fps < 0:
        raise ValueError("VR frames and fps must be nonnegative")
    sides = ("left", "right") if args.hand == "both" else (args.hand,)
    calibration = DexCalibration.load(args.dex_calibration)
    from teleoperation.retargeting.hand.dex_native import PROFILES
    if calibration.profile != PROFILES["visionpro"]:
        raise ValueError("Native VR requires its explicit VR calibration profile")
    output_h5 = args.output_h5 or (args.report.with_suffix(".h5") if args.report else None)
    destinations=[Path(p) for p in (output_h5,args.report) if p is not None]
    protected=[Path(p).resolve() for p in (args.input_h5,args.dex_calibration) if p is not None]
    if len({p.resolve() for p in destinations}) != len(destinations) or any(p.resolve() in protected or p.exists() for p in destinations):
        raise ValueError("Native VR destinations must be new and distinct from each other and inputs")
    torch.set_num_threads(1)
    retargeter = DexNativeRetargeter(calibration,sides=sides)
    commands = command_pipeline(args,retargeter.models)
    adapter = DexInputAdapter("visionpro")
    counts={"/".join(k):{"input_valid":0,"solved":0} for k in retargeter.models}
    timings={"capture":[],"preprocessing":[],"solve":[],"command":[],"render":[]}
    processed, previous_time, stopped = 0, None, "finished"
    wrist_report = {}
    wrist_control = None
    previous_tick = None
    started=time.monotonic()
    try:
        with ExitStack() as resources:
            scene=ThreeRobotHandScene(sides=sides,gui=not args.headless,steps_per_frame=args.sim_steps)
            resources.callback(scene.close)
            if len(scene.robots)!=3:
                raise ValueError("Native VR requires all three robot assets")
            from teleoperation.simulation.wrist_control import vr_arm_controller
            wrist_control = vr_arm_controller(scene, {s: s for s in sides})
            resources.callback(lambda: wrist_report.update(wrist_control.summary()))
            source=(iter_dex_observations(args.input_h5,"visionpro") if args.input_h5 else VisionProInput(args.ip,args.hand))
            resources.callback(source.close)
            writer=None
            if output_h5:
                output_h5.parent.mkdir(parents=True,exist_ok=True)
                writer=HandJointResultWriter(output_h5,retargeter.specs,{"source":"visionpro",
                    "profile":calibration.profile,"solver_parameters":VECTOR_PARAMETERS,
                    "calibration":vars(calibration),
                    "input_sha256":file_hash(args.input_h5) if args.input_h5 else None,
                    "timebase":"recording_seconds" if args.input_h5 else "live_seconds"})
                resources.callback(writer.close)
            while scene.alive() and (args.frames==0 or processed<args.frames):
                tick=time.monotonic()
                try:
                    raw=next(source) if args.input_h5 else source.next_observation()
                except StopIteration:
                    break
                timings["capture"].append((time.monotonic()-tick)*1000)
                if raw is None:
                    from teleoperation.contracts.observations import RawHandFrame
                    raw=RawHandFrame({"left":None,"right":None},time.time(),"visionpro",
                        geometry_encoding={s:"transforms25" for s in sides})
                if previous_time is not None and raw.timestamp<=previous_time:
                    raise ValueError("VR timestamps must strictly increase")
                wrist_dt = ((1/30 if previous_time is None else raw.timestamp-previous_time)
                            if args.input_h5 else (1/30 if previous_tick is None else tick-previous_tick))
                wrist_control.apply_vr(raw, wrist_dt)
                previous_tick = tick
                stamp=time.monotonic()
                frame=adapter.adapt(raw)
                timings["preprocessing"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                results=retargeter.retarget(canonical_window(frame))
                timings["solve"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                accepted={k:commands.update(v,raw.timestamp) for k,v in results.items()}
                for key,named in accepted.items():
                    scene.apply_native(named)
                    counts["/".join(key)]["input_valid"]+=int(frame.hands[key[1]] is not None)
                    counts["/".join(key)]["solved"]+=int(results[key].valid)
                if writer:
                    from teleoperation.retargeting.hand.hand_model import HandTargets
                    targets = {k: HandTargets(None, retargeter.targets(frame,k), raw.timestamp)
                               for k in results}
                    writer.append(raw.metadata.get("frame_id", processed),raw.timestamp,results,accepted,targets)
                timings["command"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                dt=1/30 if previous_time is None else min(.1,raw.timestamp-previous_time)
                scene.steps_per_frame=max(1,round(dt*240))
                scene.step()
                wrist_control.record()
                timings["render"].append((time.monotonic()-stamp)*1000)
                previous_time=raw.timestamp
                processed+=1
                if args.fps:
                    time.sleep(max(0.,1/args.fps-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        stopped="interrupted"
    report={"backend":"dex","source":"visionpro","profile":calibration.profile,"frames":processed,
        "wrists": wrist_report,
        "elapsed_seconds":time.monotonic()-started,"stopped":stopped,"hands":counts,
        "commands":commands.summary(),"source_queue":"get_latest, no accumulated frame queue",
        "timings_ms":{k:{"mean":float(np.mean(v)) if v else None,"p95":float(np.percentile(v,95)) if v else None} for k,v in timings.items()},
        "resources_closed":True,"live_device_acceptance":False if args.input_h5 else "requires operator five-minute checklist"}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2)
    print(f"Native VR frames={processed} hands={counts}")
    return 0 if any(v["solved"] for v in counts.values()) or any(wrist_report.get("applied_frames", {}).values()) else 1
