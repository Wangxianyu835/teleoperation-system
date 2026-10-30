"""Camera -> normalized and metre-based hand H5.

Run ``python -m teleoperation hand record-world --frames 30``.
Argument parsing belongs to teleoperation.cli; this module assembles capture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import time

import numpy as np

from teleoperation.data.hand_body_recording import write_hand_body_recording
from teleoperation.data.body_recording import capture_sample, save_body_recording
from teleoperation.applications.camera_teleop_source import CameraTeleopSource
from teleoperation.inputs.smplerx import SmplerXClient
from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.paths import DEFAULT_WORLD_CAPTURE_DIR

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


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
    for side in ('left', 'right'):
        _draw_landmarks(canvas, (data.get('arms') or {}).get(side), ((0, 1), (1, 2)),
                        (255, 180, 0), f'{side}-SMPL-X projection')
    status = (
        f"L-hand={'OK' if data.get('left') is not None else '--'} "
        f"R-hand={'OK' if data.get('right') is not None else '--'} "
        + ('SMPL-X reconstruction' if 'arms' in data else 'hand capture')
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


def _output_path(args) -> Path:
    if args.output is not None:
        return Path(args.output)
    started_at = datetime.now(timezone(timedelta(hours=8)))
    stem = f"hand_body_{started_at:%Y%m%d_%H%M%S}"
    directory = Path(getattr(args, "output_dir", DEFAULT_WORLD_CAPTURE_DIR))
    output = directory / f"{stem}.h5"
    index = 1
    while output.exists():
        output = directory / f"{stem}_{index:02d}.h5"
        index += 1
    return output


def run(args, *, fast_exit: bool | None = None) -> int:
    if fast_exit is None:
        fast_exit = bool(getattr(args, "fast_exit", False))
    frame_count = int(args.frames)
    if frame_count <= 0:
        raise ValueError("--frames must be positive")
    output = _output_path(args)
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--output must end with .h5 or .hdf5")
    max_fps = float(getattr(args, "max_fps", 5))
    if max_fps < 0:
        raise ValueError("--max-fps must be zero or positive")
    hand_model = _ensure_model(
        Path(args.model_asset_path),
        (
            "https://hf-mirror.com/google/mediapipe-models/resolve/main/hand_landmarker/float16/1/hand_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
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
    source = worker = None
    body_metadata = None
    arms_enabled = not bool(getattr(args, 'no_arms', False))
    timeout = float(getattr(args, 'startup_timeout', 120.))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('--startup-timeout must be positive and finite')
    try:
        if arms_enabled:
            worker = SmplerXWorkerTransport(
                worker_python=args.smplerx_worker_python, smplerx_root=args.smplerx_root,
                checkpoint=args.smplerx_checkpoint, body_model_dir=args.smplx_model_dir,
                model=args.smplerx_model)
        camera = MediaPipeCameraInput(
            model_asset_path=hand_model,
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            invalid_hand_as_missing=True,
        )
        if arms_enabled:
            session = f'record-{time.monotonic_ns()}'
            source = CameraTeleopSource(camera, camera, SmplerXClient(worker, session_id=session), session_id=session)
        startup_started = time.monotonic()
        while len(captured) < frame_count:
            index = len(captured)
            frame_started = time.perf_counter()
            if source is not None:
                raw = source.next_teleop_observation()
                if source.last_fault:
                    raise RuntimeError(f'SMPLer-X recording failed: {source.last_fault}')
                if source.last_result is not None and body_metadata is None:
                    result = source.last_result
                    body_metadata = dict(source='smplerx', model_id=result.model_id,
                                         model_hash=result.model_hash, time_unit='monotonic_s',
                                         rgb_storage='lossless_rgb_per_sample', session_id=session)
                frame = {**raw.hands.hands, 'arms': raw.upper_body.arms}
                if body_metadata is not None:
                    captured.append(capture_sample(source, raw))
                elif time.monotonic() - startup_started > timeout:
                    raise TimeoutError('SMPLer-X produced no body result before startup timeout')
            else:
                frame = _normalise_capture_frame(camera.next_frame(include_world=True))
                frame['frame_id'] = int(frame.get('metadata', {}).get('frame_index', index))
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
        if worker is not None:
            worker.close()
        if camera is not None:
            camera.release(
                wait_for_mediapipe=False,
                close_landmarker=not fast_exit,
            )
        if preview:
            cv2.destroyAllWindows()

    if not captured:
        raise RuntimeError("No hand/body frames were captured")
    writer = save_body_recording if arms_enabled else write_hand_body_recording
    destination = writer(
        output,
        captured,
        metadata={
            **(body_metadata or {}),
            "hand_model_asset_path": str(hand_model),
            "camera_index": args.camera_index,
            "fps": args.fps,
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
