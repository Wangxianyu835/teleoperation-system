"""Capture and persist MediaPipe metre-based world landmarks only."""

from __future__ import annotations

from pathlib import Path

from teleoperation.data.world_landmark_recording import (
    write_world_landmark_recording,
)
from teleoperation.inputs.mediapipe import MediaPipeCameraInput


def run(args) -> int:
    frame_count = int(args.frames)
    if frame_count <= 0:
        raise ValueError("--frames must be positive")
    output = Path(args.output)
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--output must end with .h5 or .hdf5")

    captured = []
    try:
        with MediaPipeCameraInput(
            model_asset_path=args.model_asset_path,
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            invalid_hand_as_missing=True,
        ) as camera:
            for index in range(frame_count):
                frame = camera.next_frame(include_world=True)
                frame["frame_id"] = int(
                    frame.get("metadata", {}).get("frame_index", index)
                )
                captured.append(frame)
                if (index + 1) % 30 == 0:
                    print(f"captured={index + 1}/{frame_count}", flush=True)
    except KeyboardInterrupt:
        print("capture interrupted; saving collected frames", flush=True)

    if not captured:
        raise RuntimeError("No world landmark frames were captured")
    destination = write_world_landmark_recording(
        output,
        captured,
        metadata={
            "model_asset_path": args.model_asset_path,
            "camera_index": args.camera_index,
            "image_width": args.width,
            "image_height": args.height,
            "fps": args.fps,
            "capture_source": "teleoperation hand record-world",
        },
    )
    print(f"saved_world_landmarks={destination} frames={len(captured)}")
    return 0


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    return run(args)
