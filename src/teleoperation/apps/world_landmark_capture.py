"""Capture and persist MediaPipe metre-based world landmarks only."""

from __future__ import annotations

import os
from pathlib import Path
import time

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
    max_fps = float(getattr(args, "max_fps", 10))
    if max_fps < 0:
        raise ValueError("--max-fps must be zero or positive")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)

    captured = []
    preview = bool(getattr(args, "preview", True))
    window_name = "World Landmark Capture"
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
                frame_started = time.perf_counter()
                frame = camera.next_frame(include_world=True)
                frame["frame_id"] = int(
                    frame.get("metadata", {}).get("frame_index", index)
                )
                captured.append(frame)
                if preview:
                    cv2.imshow(window_name, camera.last_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if (index + 1) % 30 == 0:
                    print(f"captured={index + 1}/{frame_count}", flush=True)
                if max_fps > 0:
                    delay = (1.0 / max_fps) - (time.perf_counter() - frame_started)
                    if delay > 0:
                        time.sleep(delay)
    except KeyboardInterrupt:
        print("capture interrupted; saving collected frames", flush=True)
    finally:
        if preview:
            cv2.destroyAllWindows()

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
