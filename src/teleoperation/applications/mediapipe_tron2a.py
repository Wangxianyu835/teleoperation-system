# 类 RelativeControlClock[next_seconds],TimedCameraObservation（封装数据/流程）；函数 _configuration（内部辅助计算）；_preview（可视化）；run（入口/工作流）；链路：CLI/调用方→应用编排→输入/算法/存储/仿真
"""MediaPipe -> TRON2A calibration only. No robot execution or simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from teleoperation.data.calibration import write_calibration
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.retargeting.arm.calibration import (
    ArmCalibration, CameraCalibrationConfig, NeutralPoseCollector, tron2a_reference,
)


class RelativeControlClock:
    """Independent monotonic relative seconds; raw Unix ms stays untouched."""
    def __init__(self):
        self._origin = time.monotonic()
        self._last = None

    def next_seconds(self):
        value = max(0.0, time.monotonic() - self._origin)
        if self._last is not None:
            value = max(value, np.nextafter(self._last, np.inf))
        self._last = float(value)
        return self._last


@dataclass(frozen=True)
class TimedCameraObservation:
    raw: object
    control_relative_seconds: float


def _configuration(args):
    config = CameraCalibrationConfig(
        rotation=np.asarray(args.rotation, dtype=float).reshape(3, 3),
        scale=np.asarray(args.scale, dtype=float),
        translation=np.asarray(args.translation, dtype=float),
        safe_q={side: np.asarray(getattr(args, f"safe_q_{side}"), dtype=float)
                for side in ("left", "right")},
        workspace_min=np.asarray(args.workspace_min, dtype=float),
        workspace_max=np.asarray(args.workspace_max, dtype=float),
        sample_count=args.samples, min_visibility=args.pose_min_visibility,
        stability_tolerance=args.stability_tolerance,
        min_elbow_bend_degrees=args.min_elbow_bend,
        max_elbow_bend_degrees=args.max_elbow_bend,
        max_sample_gap=args.max_sample_gap,
        max_acceleration=args.max_acceleration, tracking_timeout=args.tracking_timeout,
    )
    for name in ("width", "height", "fps", "max_frames"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_frames < args.samples:
        raise ValueError("--max-frames must be at least --samples")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if args.camera_index < 0:
        raise ValueError("--camera-index must be non-negative")
    output = Path(args.output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end with .json")
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}; choose a new output")
    for name in ("model_asset_path", "pose_model_asset_path"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")
    # Validate URDF, safe joint bounds and FK before opening any camera.
    tron2a_reference(config, args.urdf)
    return config


def _preview(camera, timed, collector, window):
    import cv2
    from .world_landmark_capture import HAND_CONNECTIONS, _draw_landmarks

    raw = timed.raw
    canvas = camera.last_bgr.copy()
    for side, color in (("left", (0, 255, 0)), ("right", (255, 180, 0))):
        _draw_landmarks(canvas, raw.hands.hands[side], HAND_CONNECTIONS, color, f"{side}-hand")
        _draw_landmarks(canvas, raw.upper_body.arms[side], ((0, 1), (1, 2)), color,
                        f"{side}-arm", show_index=True)
    status = " ".join(
        f"{side}: hand={'OK' if raw.hands.hands[side] is not None else '--'} "
        f"arm={'OK' if raw.upper_body.valid[side] else '--'}"
        for side in ("left", "right")
    )
    for row, text in enumerate((status, collector.status,
                               f"relative seconds={timed.control_relative_seconds:.3f}; q/Esc cancels")):
        cv2.putText(canvas, text, (10, 23 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    cv2.imshow(window, canvas)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def run(args):
    """Check -> capture -> collect -> calculate -> persist -> close resources."""
    camera = None
    preview_open = False
    window = "TRON2A neutral calibration (camera only)"
    try:
        config = _configuration(args)
        collector = NeutralPoseCollector(config)
        print("Hold both arms neutral with elbows slightly bent. "
              "Rotation/scale are configured; this does not calibrate camera extrinsics.", flush=True)
        camera = MediaPipeCameraInput(
            model_asset_path=args.model_asset_path, camera_index=args.camera_index,
            width=args.width, height=args.height, fps=args.fps,
            invalid_hand_as_missing=True, pose_model_asset_path=args.pose_model_asset_path,
            pose_min_visibility=config.min_visibility,
        )
        clock = RelativeControlClock()
        for _ in range(args.max_frames):
            seconds = clock.next_seconds()
            if seconds >= args.timeout:
                raise RuntimeError(f"Calibration timed out: {collector.status}")
            raw = camera.next_teleop_observation()
            timed = TimedCameraObservation(raw, clock.next_seconds())
            collector.add(timed.raw, timed.control_relative_seconds)
            if args.preview:
                preview_open = True
                if _preview(camera, timed, collector, window):
                    print("Calibration cancelled; no file saved.")
                    return 130
            if collector.ready:
                calibration = collector.calibration(args.urdf)
                data = calibration.to_dict()
                data["metadata"].update({
                    "camera_index": args.camera_index,
                    "hand_model_asset_path": str(Path(args.model_asset_path).resolve()),
                    "pose_model_asset_path": str(Path(args.pose_model_asset_path).resolve()),
                })
                # Validate the exact serializable result before publishing it.
                ArmCalibration.from_dict(data)
                destination = write_calibration(args.output, data)
                ArmCalibration.load(destination)
                print(f"saved_calibration={destination} samples={len(collector.samples)}")
                return 0
        raise RuntimeError(f"No stable bilateral neutral window: {collector.status}")
    except KeyboardInterrupt:
        print("Calibration cancelled; no file saved.")
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Calibration failed: {error}")
        return 1
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            if preview_open:
                import cv2
                cv2.destroyWindow(window)
