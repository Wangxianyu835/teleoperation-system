"""Camera -> MediaPipe VIDEO -> raw left/right normalized 21-point hands.

Capture settings and extraction follow ``MediaPipe/hand capture media.py``.
No mirroring, coordinate transforms, temporal windows, or recording is applied.
Provide the original ``hand_landmarker.task`` via ``model_asset_path`` when it
is not in the working directory. Importing this module does not open a camera,
import OpenCV/MediaPipe, or download a model.

Demo (Ctrl+C stops capture)::

    python -m teleoperation hand realtime --model-asset-path PATH --frames 300
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import numpy as np

from .pose_mediapipe import DEFAULT_POSE_VISIBILITY, parse_pose_result


from teleoperation.paths import DEFAULT_MEDIAPIPE_ASSET

MODEL_PATH = DEFAULT_MEDIAPIPE_ASSET
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAM_FPS = 30
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
_RETAINED_FOR_PROCESS_EXIT = []


def _landmark_array(landmarks, side: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )
    if points.shape != (21, 3):
        raise ValueError(f"{side} landmarks must have shape (21, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{side} landmarks must be finite float32 values")
    return points


def _close_landmarker_in_background(landmarker) -> None:
    def close():
        try:
            landmarker.close()
        except Exception:
            pass

    threading.Thread(target=close, name="mediapipe-close", daemon=True).start()


def _retain_until_process_exit(resource) -> None:
    _RETAINED_FOR_PROCESS_EXIT.append(resource)


def parse_result(
    result, timestamp_ms: int, metadata: dict | None = None,
    *, invalid_hand_as_missing: bool = False, include_world: bool = False,
) -> dict:
    """Extract a single detection result; malformed detections raise ValueError.

``timestamp`` is the exact Unix millisecond integer passed to VIDEO detection.
Missing hands are None. Present hands retain the original normalized x/y/z
values and landmark order, with shape (21, 3) and dtype float32.
An opt-in realtime policy drops malformed landmarks for a known side, records
the error in metadata, and preserves the other side. Repeated known labels
invalidate that side instead of choosing one detection. Unknown labels still
raise; the default remains strict.

When ``include_world`` is true, the same detection result also exposes
``world_left`` and ``world_right`` from ``hand_world_landmarks``. Those values
remain in MediaPipe's metre-based world space.
"""
    hands = {"left": None, "right": None}
    world_hands = {"left": None, "right": None}
    seen_sides = set()
    invalid_reasons = {}
    try:
        landmarks = result.hand_landmarks
        handedness = result.handedness
        if len(landmarks) != len(handedness) or len(landmarks) > 2:
            raise ValueError("Expected matching landmarks/handedness for at most two hands")
        world_landmarks = None
        if include_world:
            world_landmarks = getattr(result, "hand_world_landmarks", None)
            if world_landmarks is None or len(world_landmarks) != len(landmarks):
                raise ValueError(
                    "MediaPipe world landmarks do not match hand landmarks"
                )
        for index, hand_landmarks in enumerate(landmarks):
            # Match the reference's first-category handedness lookup exactly.
            label = handedness[index][0].category_name
            if label not in ("Left", "Right"):
                raise ValueError(f"Unknown handedness: {label!r}")
            side = label.lower()
            if side in seen_sides:
                message = f"Duplicate handedness: {label}; refusing to overwrite a hand"
                if not invalid_hand_as_missing:
                    raise ValueError(message)
                hands[side] = None
                world_hands[side] = None
                invalid_reasons[side] = message
                continue
            seen_sides.add(side)
            try:
                hands[side] = _landmark_array(hand_landmarks, side)
                if include_world:
                    world_hands[side] = _landmark_array(
                        world_landmarks[index], f"{side} world"
                    )
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                if not invalid_hand_as_missing:
                    raise
                hands[side] = None
                world_hands[side] = None
                invalid_reasons[side] = str(error)
    except (AttributeError, IndexError, TypeError, OverflowError) as error:
        raise ValueError("Malformed MediaPipe landmarks or handedness") from error
    output = {
        **hands,
        "timestamp": timestamp_ms,
        "metadata": {
            **({} if metadata is None else metadata),
            "source_landmark_space": "mediapipe_normalized",
            "timestamp_unit": "unix_ms",
            **({"invalid_reasons": invalid_reasons} if invalid_reasons else {}),
        },
    }
    if include_world:
        output.update({
            "world_left": world_hands["left"],
            "world_right": world_hands["right"],
        })
        output["metadata"].update({
            "world_landmark_space": "mediapipe_world_meters",
            "world_length_unit": "m",
            "world_origin": "hand_geometric_center",
        })
    return output


class MediaPipeCameraInput:
    """Read one raw hand frame at a time; use a context manager or release()."""

    def __init__(
        self,
        model_asset_path: str | Path = MODEL_PATH,
        camera_index: int = 0,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        fps: int = CAM_FPS,
        *,
        invalid_hand_as_missing: bool = False,
        include_image: bool = False,
        pose_model_asset_path: str | Path | None = None,
        pose_min_visibility: float = DEFAULT_POSE_VISIBILITY,
    ):
        model_path = Path(model_asset_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {model_path}")
        pose_path = (
            None
            if pose_model_asset_path is None
            else Path(pose_model_asset_path).expanduser().resolve()
        )
        if pose_path is not None and not pose_path.is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {pose_path}")

        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp
        self._cap = None
        self._landmarker = None
        self._pose_landmarker = None
        self._camera_index = camera_index
        self._frame_index = 0
        self._last_timestamp_ms = None
        self.last_bgr = None
        self._invalid_hand_as_missing = invalid_hand_as_missing
        self._include_image = include_image
        self._pose_min_visibility = float(pose_min_visibility)
        previous = Path.cwd()
        try:
            # MediaPipe's native loader may reject a non-ASCII absolute path.
            # Loading the ASCII filename from its parent directory is robust on
            # Windows while keeping the same model and file.
            os.chdir(model_path.parent)
            try:
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path.name),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=MIN_DET_CONF,
                    min_hand_presence_confidence=MIN_DET_CONF,
                    min_tracking_confidence=MIN_TRK_CONF,
                )
                pose_options = None
                if pose_path is not None:
                    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=pose_path.name
                        ),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                        num_poses=1,
                        min_pose_detection_confidence=MIN_DET_CONF,
                        min_pose_presence_confidence=MIN_DET_CONF,
                        min_tracking_confidence=MIN_TRK_CONF,
                    )
                self._cap = cv2.VideoCapture(camera_index)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                if not self._cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {camera_index}")
                self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                if pose_options is not None:
                    os.chdir(pose_path.parent)
                    self._pose_landmarker = (
                        mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
                    )
            finally:
                os.chdir(previous)
        except BaseException:
            self.release()
            raise

    def next_frame(
        self, *, include_world: bool = False, include_pose: bool = False,
    ) -> dict:
        """Return a frame even with no hands; capture/detection errors raise.

On failure both resources are released. VIDEO timestamps use the reference's
wall-clock milliseconds, advanced by 1 ms only if the clock repeats or moves
backward, to satisfy the landmarker's increasing-timestamp requirement.
Set ``include_world=True`` to also return MediaPipe metre-based world landmarks
under ``world_left`` and ``world_right``. Set ``include_pose=True`` to add the
shoulder/elbow/wrist subset and its metric pose-world copy.
"""
        if self._cap is None or self._landmarker is None:
            raise RuntimeError("MediaPipe camera input is closed")
        if include_pose and self._pose_landmarker is None:
            raise RuntimeError("MediaPipe pose landmarker is not configured")
        try:
            success, frame = self._cap.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError(f"Frame read failed for camera {self._camera_index}")
            self.last_bgr = frame
            frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            if self._last_timestamp_ms is not None:
                timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            output = parse_result(result, timestamp_ms, {
                "camera_index": self._camera_index,
                "frame_index": self._frame_index,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
            }, invalid_hand_as_missing=self._invalid_hand_as_missing,
               include_world=include_world)
            if include_pose:
                pose_result = self._pose_landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
                pose = parse_pose_result(
                    pose_result,
                    min_visibility=self._pose_min_visibility,
                )
                pose_metadata = pose.pop("metadata")
                output.update(pose)
                output["metadata"].update(pose_metadata)
            self._last_timestamp_ms = timestamp_ms
            self._frame_index += 1
            if self._include_image:
                output["image_bgr"] = frame
            return output
        except BaseException:
            self.release()
            raise

    def next_observation(self, *, include_world: bool = False):
        from teleoperation.contracts.observations import RawHandFrame
        raw = self.next_frame(include_world=include_world)
        world_hands = None
        if include_world:
            world_hands = {
                side: raw.get(f"world_{side}") for side in ("left", "right")
            }
        return RawHandFrame(
            {side: raw[side] for side in ("left", "right")},
            raw["timestamp"],
            "mediapipe_approx",
            raw.get("metadata", {}),
            {"left": "landmarks21", "right": "landmarks21"},
            world_hands=world_hands,
        )

    def close(self): self.release()

    def release(
        self, *, wait_for_mediapipe: bool = True, close_landmarker: bool = True,
    ) -> None:
        """Release both resources, including partial initialization; idempotent.

        MediaPipe can block for tens of seconds while its native cleanup waits
        for a telemetry upload to time out. Collection callers that are about
        to exit may set ``wait_for_mediapipe=False`` to close that native object
        in a daemon thread instead of blocking the recording process. A caller
        that will immediately terminate the process may set
        ``close_landmarker=False`` as well.
        """
        cap, landmarker, pose_landmarker = (
            self._cap,
            self._landmarker,
            self._pose_landmarker,
        )
        self._cap = self._landmarker = self._pose_landmarker = None
        try:
            if cap is not None:
                cap.release()
        finally:
            for resource in (landmarker, pose_landmarker):
                if resource is None:
                    continue
                if not close_landmarker:
                    _retain_until_process_exit(resource)
                elif wait_for_mediapipe:
                    resource.close()
                else:
                    _close_landmarker_in_background(resource)

    def __enter__(self) -> "MediaPipeCameraInput":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
