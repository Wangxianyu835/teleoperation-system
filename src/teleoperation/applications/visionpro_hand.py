"""Upstream TransHandR Vision Pro -> compatible hand result -> existing scene."""
import argparse
from contextlib import ExitStack
import json
import time

import numpy as np

from teleoperation.applications.realtime_hand_sim import build_scene
from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline
from teleoperation.applications.visionpro_results import VRResultWriter
from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
from teleoperation.retargeting.hand.transhandr_vr import load_transhandr_vr_model, TransHandRVRRetargeter
from teleoperation.retargeting.hand.angles import canonical_to_native_angles
from teleoperation.simulation.hand_commands import build_hand_commands
from teleoperation.cli.visionpro_options import configure


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description="TransHandR upstream Vision Pro hand pipeline")
    return configure(parser)


def run(args):
    if args.frames < 0 or not np.isfinite(args.fps) or args.fps < 0:
        raise ValueError("frames and fps must be nonnegative")
    mapping = getattr(args, "hand_mapping", None)
    if args.hand == "both":
        if mapping != "same" or args.source_hand not in (None, "both"):
            raise ValueError("--hand both requires --hand-mapping same and --source-hand both (or omitted)")
        routes = {"left": "left", "right": "right"}
        source_hand = "both"
    else:
        if mapping is not None or args.source_hand == "both":
            raise ValueError("--hand-mapping and --source-hand both require --hand both")
        source_hand = args.source_hand or "left"
        routes = {args.hand: source_hand}
    output_h5 = getattr(args, "output_h5", None)
    if output_h5 is None and args.report:
        output_h5 = args.report.with_suffix(".h5")
    paths = [path.resolve() for path in (args.input_h5, args.checkpoint, args.report, output_h5)
             if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("input, checkpoint, report and output H5 paths must be different")
    if output_h5 is not None and output_h5.exists():
        raise FileExistsError(f"VR result already exists: {output_h5}; choose a new --output-h5/--report")
    if args.device == "cpu":
        import torch
        torch.set_num_threads(1)
    model = load_transhandr_vr_model(args.checkpoint, args.device)
    retargeters = {target: TransHandRVRRetargeter(model, args.device, source, target)
                  for target, source in routes.items()}
    pipeline = TransHandRVisionProPipeline(args.scale_factor)
    commands = None
    counts = {side: {"input_valid_frames": 0, "angle_frames": 0, "missing_or_warmup_frames": 0}
              for side in routes}
    processed = angle_frames = missing = 0
    previous_timestamp = None
    started = time.monotonic()
    with ExitStack() as resources:
        scene = build_scene(args, tuple(routes))
        resources.callback(scene.close)
        commands = build_hand_commands(args, scene, tuple(routes))
        if hasattr(scene, "frame_scene"):
            scene.frame_scene(args.view)
        source = (VisionProReplayInput(args.input_h5, source_hand) if args.input_h5 else
                  VisionProInput(args.ip, source_hand))
        resources.callback(source.close)
        writer = None
        if output_h5 is not None:
            writer = VRResultWriter(output_h5, routes, {
                "checkpoint": str(args.checkpoint), "input_h5": str(args.input_h5 or ""),
                "hand_mapping": mapping or "upstream_single",
                "raw_convention": "upstream_right_l21",
                "native_adaptation": "left_mcp_roll_signs" if mapping == "same" else "legacy_passthrough",
                "command_timebase": "recording" if args.input_h5 else "monotonic",
                "invalid_angles": "NaN for raw/native; command holds last accepted pose"})
            resources.callback(writer.close)
        routing = ", ".join(f"{source}_fingers -> {target} robot hand" for target, source in routes.items())
        print(f"VR upstream: {routing}; "
              f"axis remap + wrist origin + 3 frames; checkpoint={args.checkpoint}", flush=True)
        try:
            while scene.alive() and (args.frames == 0 or processed < args.frames):
                tick = time.monotonic()
                try:
                    observation = source.next_observation()
                except StopIteration:
                    break
                timestamp = float(observation.timestamp) if observation is not None else tick
                if args.input_h5:
                    if not np.isfinite(timestamp) or (previous_timestamp is not None and timestamp <= previous_timestamp):
                        raise ValueError("VR recording timestamps must be finite and strictly increasing")
                    previous_timestamp = timestamp
                window = pipeline.process_window(observation)
                records, any_valid = {}, False
                for side, source_side in routes.items():
                    input_valid = pipeline.current_hands[source_side] is not None
                    counts[side]["input_valid_frames"] += int(input_valid)
                    angles = (None if window is None else
                              getattr(retargeters[side].retarget(window), side + "_angles"))
                    raw = native = np.full(18, np.nan)
                    if angles is None:
                        counts[side]["missing_or_warmup_frames"] += 1
                        commands.missing(side)
                        command = commands.previous[side].copy()
                    else:
                        any_valid = True
                        counts[side]["angle_frames"] += 1
                        raw = np.asarray(angles.values, dtype=np.float64)
                        # Output boundary adaptation only. Never alter the upstream input/model.
                        native = canonical_to_native_angles(raw, side) if mapping == "same" else raw.copy()
                        command = commands.update(side, native, timestamp=timestamp if args.input_h5 else None)
                        scene.apply(side, command)
                    records[side] = {"input_valid": input_valid, "model_valid": angles is not None,
                                     "raw_angles": raw, "native_angles": native, "command_angles": command}
                if writer is not None:
                    frame_id = observation.metadata.get("frame_id", processed) if observation is not None else processed
                    writer.append(frame_id, timestamp, records)
                processed += 1
                angle_frames += int(any_valid)
                missing += int(not any_valid)
                scene.step()
                if args.fps:
                    time.sleep(max(0., 1 / args.fps - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            print("VR capture stopped.", flush=True)
    report = {"entry": "visionpro_hand", "upstream": "TransHandR/main_visionpro_realtime.py",
              "source_hand": source_hand, "target_hand": args.hand,
              "hand_mapping": mapping or "upstream_single", "routes": routes,
              "sides": counts, "output_h5": None if output_h5 is None else str(output_h5),
              "command_timebase": "recording" if args.input_h5 else "monotonic",
              "checkpoint": str(args.checkpoint), "processed_frames": processed,
              "angle_frames": angle_frames, "missing_or_warmup_frames": missing,
              "elapsed_seconds": time.monotonic() - started,
              "input_shape": [1, 3, 25, 3], "output_shape": [18],
              "preprocessing": "axis remap -> wrist relative -> chronological three frames -> fixed scale",
              "hand_commands": None if commands is None else commands.summary()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VR frames={processed} output_frames={angle_frames} missing/warmup={missing} sides={counts}", flush=True)
    if output_h5 is not None:
        print(f"VR per-frame results: {output_h5}", flush=True)
    return 0 if angle_frames else 1


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    try:
        return run(args)
    except Exception as error:
        print(f"[FAIL] Vision Pro pipeline: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
