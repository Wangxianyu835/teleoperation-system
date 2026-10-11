"""Capture normalized and metre-based hand and shoulder/elbow/wrist data."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import time

import numpy as np

from teleoperation.data.hand_body_recording import write_hand_body_recording
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.inputs.pose_mediapipe import DEFAULT_POSE_VISIBILITY
from teleoperation.paths import DEFAULT_MEDIAPIPE_ASSET, DEFAULT_POSE_ASSET


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "hand_body_capture"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
POSE_CONNECTIONS = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5))


def _ensure_model(path: Path, urls: tuple[str, ...]) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    last_error = None
    for url in urls:
        try:
            print(f"downloading model: {url}", flush=True)
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as error:  # pragma: no cover - network fallback
            last_error = error
    raise FileNotFoundError(f"Could not download model {path}: {last_error}")


def _draw_landmarks(canvas, points, connections, color, label, show_index=False):
    if points is None:
        return
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0 or not np.isfinite(points).all():
        return
    height, width = canvas.shape[:2]
    import cv2

    for start, end in connections:
        if start >= len(points) or end >= len(points):
            continue
        x1, y1 = int(points[start, 0] * width), int(points[start, 1] * height)
        x2, y2 = int(points[end, 0] * width), int(points[end, 1] * height)
        if min(x1, y1, x2, y2) < 0:
            continue
        if max(x1, x2) >= width or max(y1, y2) >= height:
            continue
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
    for index, (x, y, _) in enumerate(points):
        px, py = int(x * width), int(y * height)
        if not (0 <= px < width and 0 <= py < height):
            continue
        cv2.circle(canvas, (px, py), 3, color, -1)
        if show_index:
            cv2.putText(
                canvas,
                str(index),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
    x, y = int(points[0, 0] * width), int(points[0, 1] * height)
    cv2.putText(
        canvas,
        label,
        (x + 6, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def _draw_preview(frame, data, window_name):
    import cv2

    canvas = frame.copy()
    _draw_landmarks(canvas, data.get("left"), HAND_CONNECTIONS, (0, 255, 0), "L-hand")
    _draw_landmarks(canvas, data.get("right"), HAND_CONNECTIONS, (0, 255, 0), "R-hand")
    _draw_landmarks(
        canvas,
        data.get("pose_keypoints"),
        POSE_CONNECTIONS,
        (255, 255, 0),
        "arms",
        show_index=True,
    )
    status = (
        f"L-hand={'OK' if data.get('left') is not None else '--'} "
        f"R-hand={'OK' if data.get('right') is not None else '--'} "
        f"L-arm={'OK' if data.get('left_arm_valid') else '--'} "
        f"R-arm={'OK' if data.get('right_arm_valid') else '--'}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(canvas, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(window_name, canvas)


def _normalise_capture_frame(frame):
    frame["left_hand_keypoints"] = frame.get("left")
    frame["right_hand_keypoints"] = frame.get("right")
    frame["left_world_landmarks"] = frame.get("world_left")
    frame["right_world_landmarks"] = frame.get("world_right")
    return frame


def run(args, *, fast_exit: bool = False) -> int:
    frame_count = int(args.frames)
    if frame_count <= 0:
        raise ValueError("--frames must be positive")
    output = Path(args.output)
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--output must end with .h5 or .hdf5")
    max_fps = float(getattr(args, "max_fps", 5))
    if max_fps < 0:
        raise ValueError("--max-fps must be zero or positive")
    pose_min_visibility = float(
        getattr(args, "pose_min_visibility", DEFAULT_POSE_VISIBILITY)
    )

    hand_model = _ensure_model(
        Path(args.model_asset_path),
        (
            "https://hf-mirror.com/google/mediapipe-models/resolve/main/hand_landmarker/float16/1/hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        ),
    )
    pose_model = _ensure_model(
        Path(args.pose_model_asset_path),
        (
            "https://hf-mirror.com/google/mediapipe-models/resolve/main/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        ),
    )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)

    captured = []
    preview = bool(getattr(args, "preview", True))
    window_name = "Hand + Arm Capture (q to stop)"
    camera = None
    try:
        camera = MediaPipeCameraInput(
            model_asset_path=hand_model,
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            invalid_hand_as_missing=True,
            pose_model_asset_path=pose_model,
            pose_min_visibility=pose_min_visibility,
        )
        for index in range(frame_count):
            frame_started = time.perf_counter()
            frame = _normalise_capture_frame(
                camera.next_frame(include_world=True, include_pose=True)
            )
            frame["frame_id"] = int(
                frame.get("metadata", {}).get("frame_index", index)
            )
            captured.append(frame)
            if preview:
                _draw_preview(camera.last_bgr, frame, window_name)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if (index + 1) % 10 == 0:
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
        raise RuntimeError("No hand/body frames were captured")
    destination = write_hand_body_recording(
        output,
        captured,
        metadata={
            "hand_model_asset_path": str(hand_model),
            "pose_model_asset_path": str(pose_model),
            "camera_index": args.camera_index,
            "fps": args.fps,
            "pose_min_visibility": pose_min_visibility,
            "capture_source": "teleoperation hand record-world",
        },
    )
    print(f"saved_hand_body_capture={destination} frames={len(captured)}")
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--pose-model-asset-path", type=Path, default=DEFAULT_POSE_ASSET)
    parser.add_argument(
        "--pose-min-visibility",
        type=float,
        default=DEFAULT_POSE_VISIBILITY,
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _direct_args():
    parser = argparse.ArgumentParser(
        description="Record normalized and metre-based hand and arm landmarks."
    )
    _add_arguments(parser)
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = DEFAULT_OUTPUT_DIR / f"hand_body_{stamp}.h5"
    return args


if __name__ == "__main__":
    raise SystemExit(run(_direct_args(), fast_exit=True))
