"""Upstream TransHandR Vision Pro -> compatible hand result -> existing scene."""
import argparse
import json
import time

import numpy as np

from teleoperation.applications.realtime_hand_sim import build_scene
from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline
from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
from teleoperation.retargeting.hand.transhandr_vr import load_transhandr_vr_model, TransHandRVRRetargeter
from teleoperation.simulation.hand_commands import build_hand_commands
from teleoperation.cli.visionpro_options import configure


def build_parser(parser=None):
    parser = parser or argparse.ArgumentParser(description="TransHandR upstream Vision Pro hand pipeline")
    return configure(parser)


def run(args):
    if args.frames < 0 or not np.isfinite(args.fps) or args.fps < 0:
        raise ValueError("frames and fps must be nonnegative")
    if args.device == "cpu":
        import torch
        torch.set_num_threads(1)
    model = load_transhandr_vr_model(args.checkpoint, args.device)
    retargeter = TransHandRVRRetargeter(model, args.device, args.source_hand, args.hand)
    pipeline = TransHandRVisionProPipeline(args.scale_factor)
    scene = source = commands = None
    raw_angles, command_angles = [], []
    processed = missing = 0
    started = time.monotonic()
    try:
        scene = build_scene(args, (args.hand,))
        commands = build_hand_commands(args, scene, (args.hand,))
        if hasattr(scene, "frame_scene"):
            scene.frame_scene(args.view)
        source = (VisionProReplayInput(args.input_h5, args.source_hand) if args.input_h5 else
                  VisionProInput(args.ip, args.source_hand))
        print(f"VR upstream: {args.source_hand}_fingers -> {args.hand} robot hand; "
              f"axis remap + wrist origin + 3 frames; checkpoint={args.checkpoint}", flush=True)
        while scene.alive() and (args.frames == 0 or processed < args.frames):
            tick = time.monotonic()
            try:
                observation = source.next_observation()
            except StopIteration:
                break
            processed += 1
            window = pipeline.process_window(observation)
            angles = None if window is None else getattr(retargeter.retarget(window), args.hand + "_angles")
            if angles is None:
                missing += 1
                commands.missing(args.hand)
            else:
                raw = np.asarray(angles.values, dtype=np.float64)
                command = commands.update(args.hand, raw)
                scene.apply(args.hand, command)
                raw_angles.append(raw.copy())
                command_angles.append(command.copy())
            scene.step()
            if args.fps:
                time.sleep(max(0., 1 / args.fps - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("VR capture stopped.", flush=True)
    finally:
        if source is not None:
            source.close()
        if scene is not None:
            scene.close()
    report = {"entry": "visionpro_hand", "upstream": "TransHandR/main_visionpro_realtime.py",
              "source_hand": args.source_hand, "target_hand": args.hand,
              "checkpoint": str(args.checkpoint), "processed_frames": processed,
              "angle_frames": len(raw_angles), "missing_or_warmup_frames": missing,
              "elapsed_seconds": time.monotonic() - started,
              "input_shape": [1, 3, 25, 3], "output_shape": [18],
              "preprocessing": "axis remap -> wrist relative -> chronological three frames -> fixed scale",
              "hand_commands": None if commands is None else commands.summary()}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VR frames={processed} model_outputs={len(raw_angles)} missing/warmup={missing}", flush=True)
    return 0 if raw_angles else 1


def main(args=None):
    args = build_parser().parse_args() if args is None else args
    try:
        return run(args)
    except Exception as error:
        print(f"[FAIL] Vision Pro pipeline: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
