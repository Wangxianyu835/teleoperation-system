"""Native dex application composition: calibration, H5 export/replay and live I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from teleoperation.contracts.hand import NativeHandResult
from teleoperation.data.hand_recording import iter_dex_observations, NativeHandResultWriter, iter_native_result_commands
from teleoperation.retargeting.hand.dex_native import DexInputAdapter, DexCalibration, DexNativeRetargeter
from teleoperation.simulation.hand_commands import NativeHandCommandPipeline


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
                "calibration": vars(calibration), "command_timebase": "recording_seconds"}
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
