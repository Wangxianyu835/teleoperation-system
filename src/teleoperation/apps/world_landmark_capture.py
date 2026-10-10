"""Capture and persist MediaPipe metre-based world landmarks only."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import time

from teleoperation.data.world_landmark_recording import (
    write_world_landmark_recording,
)
from teleoperation.inputs.mediapipe import MediaPipeCameraInput


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "world_landmarks"


def run(args, *, fast_exit: bool = False) -> int:
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
    camera = None
    try:
        camera = MediaPipeCameraInput(
            model_asset_path=args.model_asset_path,
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            invalid_hand_as_missing=True,
        )
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
        if camera is not None:
            camera.release(
                wait_for_mediapipe=False,
                close_landmarker=not fast_exit,
            )
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
    if fast_exit:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    return run(args, fast_exit=bool(getattr(args, "fast_exit", False)))


def _direct_args():
    parser = argparse.ArgumentParser(
        description="Record MediaPipe hand_world_landmarks (metres) to H5."
    )
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = DEFAULT_OUTPUT_DIR / f"world_landmarks_{stamp}.h5"
    return args


if __name__ == "__main__":
    raise SystemExit(run(_direct_args(), fast_exit=True))
