"""Native dex application composition: calibration, H5 export/replay and live I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from teleoperation.contracts.hand import NativeHandResult
from teleoperation.data.hand_recording import iter_dex_observations, NativeHandResultWriter, iter_native_result_commands
from teleoperation.retargeting.hand.dex_native import DexInputAdapter, DexCalibration, DexNativeRetargeter, VECTOR_PARAMETERS
from teleoperation.simulation.hand_commands import NativeHandCommandPipeline


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


def camera_window_frame(window):
    from teleoperation.contracts.hand import CanonicalHandFrame
    from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
    if window.metadata.get("coordinate_alignment") != WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT or window.metadata.get("units") != "m":
        raise ValueError("Dex window requires the existing canonical world metre profile")
    return CanonicalHandFrame({s: None if window.hands[s] is None else window.hands[s][-1]
        for s in ("left", "right")}, window.timestamp, "camera",
        {**window.metadata, "dex_input_profile": "dex_camera_world_v1"})


def native_sample_mapping(retargeter, side):
    return [f"{r}.{name}" for (r,s), model in retargeter.models.items() if s == side for name in model.joint_names]


def apply_native_window(window, retargeter, commands, scene, timestamp=None):
    latest = {s: [] for s in scene.sides}
    complete = {s: True for s in scene.sides}
    results = None if window is None else retargeter.retarget(camera_window_frame(window))
    for key, model in retargeter.models.items():
        if key[1] not in scene.sides:
            continue
        if results is None:
            retargeter.solvers[key].reset()
            commands.missing(key)
            complete[key[1]] = False
            continue
        result = results[key]
        accepted = commands.update(result, timestamp=timestamp)
        scene.apply_native(command_result(model, accepted, result.timestamp))
        complete[key[1]] &= result.valid
        latest[key[1]].extend(accepted[n] for n in model.joint_names)
    return {s: np.asarray(values) if complete[s] and values else None for s, values in latest.items()}


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
    frames = recording_frames(args.input, args.source)
    if not frames:
        raise ValueError("Empty calibration recording")
    calibration = DexCalibration.fit(frames, frames[0].timestamp + args.start, args.duration, file_hash(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(args.output)
    print(f"Saved fixed {calibration.profile} calibration: {args.output}")
    return 0


def command_pipeline(args, models):
    return NativeHandCommandPipeline(models, args.joint_filter, args.filter_alpha,
        args.joint_max_velocity, args.kalman_q, args.kalman_r)


def command_result(model, named, timestamp):
    return NativeHandResult(model.robot_type, model.side, model.joint_names,
        np.array([named[n] for n in model.joint_names]), timestamp, model.config_hash)


def export(args):
    import torch
    torch.set_num_threads(1)
    report_path = Path(getattr(args, "report", None) or args.output.with_suffix(".json"))
    paths = [args.input.resolve(), args.calibration.resolve(), args.output.resolve(), report_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, calibration, output and report paths must be distinct")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Choose unused native output and report paths")
    calibration = DexCalibration.load(args.calibration)
    retargeter = DexNativeRetargeter(calibration)
    commands = command_pipeline(args, retargeter.models)
    counts = {"/".join(k): {"input_valid": 0, "solved": 0} for k in retargeter.models}
    digest = file_hash(args.input)
    metadata = {"source": args.source, "input": str(args.input.resolve()), "input_sha256": digest,
                "calibration": vars(calibration), "command_timebase": "recording_seconds",
                "solver_parameters": VECTOR_PARAMETERS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed, frame_times = 0, []
    with NativeHandResultWriter(args.output, retargeter.models, metadata) as writer:
        adapter = DexInputAdapter(args.source)
        for observation in iter_dex_observations(args.input, args.source):
            frame = adapter.adapt(observation)
            results = retargeter.retarget(frame)
            accepted = {k: commands.update(result, frame.timestamp) for k, result in results.items()}
            targets = {k: retargeter.targets(frame, k) for k in results}
            writer.append(frame.timestamp, results, accepted, targets,
                digest == calibration.source_sha256 and calibration.segment[0] <= frame.timestamp <= calibration.segment[1])
            frame_times.append(sum(r.solve_ms for r in results.values()))
            for key, result in results.items():
                row = counts["/".join(key)]
                row["input_valid"] += int(frame.hands[key[1]] is not None)
                row["solved"] += int(result.valid)
            processed += 1
            if processed % 500 == 0:
                print(f"Native export frames={processed}",flush=True)
    report = {"frames": processed, "hands": counts, "commands": commands.summary(),
              "source": args.source, "input_sha256": digest,
              "solve_frame_p95_ms": float(np.percentile(frame_times, 95)) if frame_times else None}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Native dex export: {processed} frames -> {args.output}")
    return 0 if processed else 1


def replay(args):
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.robots.native_hand import NativeHandModel
    import itertools
    if not np.isfinite(args.fps) or args.fps < 0 or args.frames < 0:
        raise ValueError("Replay fps and frame limit must be nonnegative")
    models = {(r, s): NativeHandModel(r, s) for r in ("h1_2", "gr1_t2", "g1") for s in ("left", "right")}
    iterator = iter_native_result_commands(args.file, models)
    scene, count = None, 0
    try:
        first = next(iterator, None)
        if first is None:
            raise ValueError("Empty native result recording")
        scene = ThreeRobotHandScene(gui=args.render)
        if len(scene.robots) != 3:
            raise ValueError("Native six-hand replay requires all three robot assets")
        previous_time = first[0] - 1 / 30
        for timestamp, results in itertools.chain([first], iterator):
            if not scene.alive() or (args.frames and count >= args.frames):
                break
            started = time.monotonic()
            for result in results.values():
                scene.apply_native(result)
            scene.steps_per_frame = max(1, round(min(timestamp - previous_time, .1) * 240))
            scene.step()
            previous_time, count = timestamp, count + 1
            if args.fps:
                time.sleep(max(0., 1 / args.fps - (time.monotonic() - started)))
    finally:
        iterator.close()
        if scene is not None:
            scene.close()
    print(f"Native replay frames={count}")
    return 0 if count else 1


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
    started=time.monotonic()
    try:
        with ExitStack() as resources:
            scene=ThreeRobotHandScene(sides=sides,gui=not args.headless,steps_per_frame=args.sim_steps)
            resources.callback(scene.close)
            if len(scene.robots)!=3:
                raise ValueError("Native VR requires all three robot assets")
            source=(iter_dex_observations(args.input_h5,"visionpro") if args.input_h5 else VisionProInput(args.ip,args.hand))
            resources.callback(source.close)
            writer=None
            if output_h5:
                output_h5.parent.mkdir(parents=True,exist_ok=True)
                writer=NativeHandResultWriter(output_h5,retargeter.models,{"source":"visionpro",
                    "profile":calibration.profile,"solver_parameters":VECTOR_PARAMETERS,
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
                stamp=time.monotonic()
                frame=adapter.adapt(raw)
                timings["preprocessing"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                results=retargeter.retarget(frame)
                timings["solve"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                accepted={k:commands.update(v,raw.timestamp if args.input_h5 else None) for k,v in results.items()}
                for key,named in accepted.items():
                    scene.apply_native(command_result(retargeter.models[key],named,raw.timestamp))
                    counts["/".join(key)]["input_valid"]+=int(frame.hands[key[1]] is not None)
                    counts["/".join(key)]["solved"]+=int(results[key].valid)
                if writer:
                    writer.append(raw.timestamp,results,accepted,{k:retargeter.targets(frame,k) for k in results})
                timings["command"].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                dt=1/30 if previous_time is None else min(.1,raw.timestamp-previous_time)
                scene.steps_per_frame=max(1,round(dt*240))
                scene.step()
                timings["render"].append((time.monotonic()-stamp)*1000)
                previous_time=raw.timestamp
                processed+=1
                if args.fps:
                    time.sleep(max(0.,1/args.fps-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        stopped="interrupted"
    report={"backend":"dex","source":"visionpro","profile":calibration.profile,"frames":processed,
        "elapsed_seconds":time.monotonic()-started,"stopped":stopped,"hands":counts,
        "commands":commands.summary(),"source_queue":"get_latest, no accumulated frame queue",
        "timings_ms":{k:{"mean":float(np.mean(v)) if v else None,"p95":float(np.percentile(v,95)) if v else None} for k,v in timings.items()},
        "resources_closed":True,"live_device_acceptance":False if args.input_h5 else "requires operator five-minute checklist"}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2)
    print(f"Native VR frames={processed} hands={counts}")
    return 0 if any(v["solved"] for v in counts.values()) else 1
