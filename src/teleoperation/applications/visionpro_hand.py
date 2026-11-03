"""VR shortcut to common NN/dex control, plus explicit upstream compatibility."""
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
from teleoperation.contracts.hand_control import valid_joint_result
from teleoperation.simulation.hand_commands import build_hand_commands
from teleoperation.cli.visionpro_options import configure


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description="Vision Pro input with shared NN/dex hand retargeting")
    return configure(parser)


def run(args):
    if args.backend in ('nn', 'dex'):
        from argparse import Namespace
        from teleoperation.applications.hand_control import run as unified_run
        if args.dex_calibration is not None or args.scale_factor != 1.:
            raise ValueError('Unified NN/dex uses shared palm normalization, no fixed VR calibration/scale')
        if args.source_hand not in (None, args.hand) or args.hand_mapping not in (None, 'same'):
            raise ValueError('Unified hand control uses anatomical same-side routing')
        values = vars(args).copy()
        values['record_results'] = args.output_h5 or (args.report.with_suffix('.h5') if args.report else None)
        return unified_run(Namespace(**values))
    if args.hand_model != 'l21':
        raise ValueError('Upstream TransHandR weights target L21; use shared NN/dex for other hands')
    if args.scene == 'auto':
        args.scene = 'hands'
    if args.scene != "hands":
        raise ValueError("TransHandR VR requires --scene hands; use native dex for robots")
    if args.checkpoint is None:
        raise ValueError("TransHandR VR requires --checkpoint")
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
    previous_tick = None
    started = time.monotonic()
    with ExitStack() as resources:
        scene = build_scene(args, tuple(routes))
        resources.callback(scene.close)
        from teleoperation.simulation.wrist_control import vr_arm_controller
        wrist_control = vr_arm_controller(scene, routes)
        wrist_report = None
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
                "native_adaptation": "target_native_mcp_roll_signs",
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
                timestamp = float(observation.timestamp) if observation is not None else time.time()
                if args.input_h5:
                    if not np.isfinite(timestamp) or (previous_timestamp is not None and timestamp <= previous_timestamp):
                        raise ValueError("VR recording timestamps must be finite and strictly increasing")
                wrist_dt = ((1/30 if previous_timestamp is None else timestamp - previous_timestamp)
                            if args.input_h5 else (1/30 if previous_tick is None else tick - previous_tick))
                if wrist_control is not None:
                    wrist_control.apply_vr(observation, wrist_dt)
                window = pipeline.process_window(observation)
                previous_timestamp = timestamp
                previous_tick = tick
                records, any_valid = {}, False
                for side, source_side in routes.items():
                    input_valid = pipeline.current_hands[source_side] is not None
                    counts[side]["input_valid_frames"] += int(input_valid)
                    results = None if window is None else retargeters[side].retarget(window)
                    angles = valid_joint_result(results, "l21", side)
                    raw = np.full(18, np.nan)
                    native = np.full(17, np.nan)
                    if angles is None:
                        counts[side]["missing_or_warmup_frames"] += 1
                        accepted = commands.missing(('l21',side),timestamp=timestamp if args.input_h5 else None)
                    else:
                        any_valid = True
                        counts[side]["angle_frames"] += 1
                        raw = np.asarray(retargeters[side].last_raw_angles, dtype=np.float64)
                        # The adapter has already converted to the selected native hand.
                        native = angles.positions.copy()
                        accepted = commands.update(angles, timestamp=timestamp if args.input_h5 else None)
                    scene.apply_native(accepted)
                    records[side] = {"input_valid": input_valid, "model_valid": angles is not None,
                                     "raw_angles": raw, "native_positions": native, "command_positions": accepted.positions}
                if writer is not None:
                    frame_id = observation.metadata.get("frame_id", processed) if observation is not None else processed
                    writer.append(frame_id, timestamp, records)
                processed += 1
                angle_frames += int(any_valid)
                missing += int(not any_valid)
                scene.step()
                if wrist_control is not None:
                    wrist_control.record()
                if args.fps:
                    time.sleep(max(0., 1 / args.fps - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            print("VR capture stopped.", flush=True)
        if wrist_control is not None:
            wrist_report = wrist_control.summary()
    report = {"entry": "visionpro_hand", "upstream": "TransHandR/main_visionpro_realtime.py",
              "wrists": wrist_report,
              "source_hand": source_hand, "target_hand": args.hand,
              "hand_mapping": mapping or "upstream_single", "routes": routes,
              "sides": counts, "output_h5": None if output_h5 is None else str(output_h5),
              "command_timebase": "recording" if args.input_h5 else "monotonic",
              "checkpoint": str(args.checkpoint), "processed_frames": processed,
              "angle_frames": angle_frames, "missing_or_warmup_frames": missing,
              "elapsed_seconds": time.monotonic() - started,
              "input_shape": [1, 3, 25, 3], "output_shape": [18],
              "native_command_shape": [17],
              "preprocessing": "axis remap -> wrist relative -> chronological three frames -> fixed scale",
              "hand_commands": None if commands is None else commands.summary()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VR frames={processed} output_frames={angle_frames} missing/warmup={missing} sides={counts}", flush=True)
    if output_h5 is not None:
        print(f"VR per-frame results: {output_h5}", flush=True)
    wrist_applied = wrist_report is not None and any(wrist_report["applied_frames"].values())
    return 0 if angle_frames or wrist_applied else 1


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    try:
        return run(args)
    except Exception as error:
        print(f"[FAIL] Vision Pro pipeline: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
