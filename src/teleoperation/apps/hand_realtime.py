"""MediaPipe raw input -> per-frame palm-local alignment -> three-frame windows.

Identity tracking is disabled: MediaPipe handedness is used directly. Each
invalid side resets immediately and independently; no historical frame or basis
is reused. The demo loads palm_local_v2 and prints 18D outputs without sending
commands or recording data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from teleoperation.contracts.coordinates import PALM_LOCAL_COORDINATE_ALIGNMENT, PALM_LOCAL_METADATA
from teleoperation.retargeting.hand.coordinates import align_palm_local_coordinates, build_l21_reference_basis
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.retargeting.hand.topology import MEDIAPIPE_APPROX_SOURCE, ensure_hand25


from teleoperation.paths import DEFAULT_REALTIME_CHECKPOINT, DEFAULT_MEDIAPIPE_ASSET
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM, RUNTIME

PALM_LOCAL_V2_CHECKPOINT = DEFAULT_REALTIME_CHECKPOINT
SIDES = ("left", "right")


class MediaPipeCameraWorkflow:
    """Compose raw camera capture with palm-local realtime processing.

    next_input() retains the optional-window payload and relative seconds
    timestamp. Capture errors propagate after clearing history.
    The old identity thresholds are accepted for caller compatibility and have
    no effect. Scale and confidence must retain the raw-input defaults.
    """

    def __init__(
        self,
        model_asset_path: str | Path = "hand_landmarker.task",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        scale_factor: float = 1.0,
        min_confidence: float = 0.5,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        if scale_factor != 1.0:
            raise ValueError("MediaPipe palm-local realtime requires scale_factor=1.0")
        if min_confidence != 0.5:
            raise ValueError("MediaPipeCameraInput uses fixed min_confidence=0.5")
        self.model_asset_path = Path(model_asset_path)
        self.pipeline = MediaPipePalmLocalPipeline()
        self.processor = self.pipeline.processor
        self.last_raw_frame = None
        self._input = MediaPipeCameraInput(
            model_asset_path=model_asset_path, camera_index=camera_index,
            width=width, height=height, fps=fps,
            invalid_hand_as_missing=True,
        )
        self._start_time = time.time()

    def next_window(self):
        try:
            raw = self._input.next_observation()
            raw_frame = {**raw.hands, "timestamp": raw.timestamp, "metadata": dict(raw.metadata)}
            self.last_raw_frame = raw_frame
            frame = {
                **raw_frame,
                "timestamp": round(time.time() - self._start_time, 3),
                "metadata": {
                    **raw_frame.get("metadata", {}),
                    "raw_timestamp_ms": raw_frame["timestamp"],
                    "timestamp_unit": "relative_seconds",
                },
            }
            return self.pipeline.process_window(frame)
        except BaseException:
            self.last_raw_frame = None
            self.release()
            raise

    def next_input(self) -> dict | None:
        window = self.next_window()
        return None if window is None else window.to_payload()

    def release(self) -> None:
        self.pipeline.reset()
        self._input.close()

    def __enter__(self) -> "MediaPipeCameraWorkflow":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def load_palm_local_retargeter(checkpoint_path=PALM_LOCAL_V2_CHECKPOINT, device="cpu"):
    """Load the existing model and enforce the palm-local checkpoint contract."""
    from teleoperation.retargeting.hand.config import L21
    from teleoperation.retargeting.hand.predictor import create_twohand_retargeter

    return create_twohand_retargeter(
        L21.model_kwargs(), device, checkpoint_path=str(checkpoint_path),
        expected_coordinate_alignment=PALM_LOCAL_COORDINATE_ALIGNMENT,
    ).eval()


def run(args: argparse.Namespace) -> int:
    if args.frames < 0 or (args.frames == 0 and (not args.visualize or args.headless)):
        raise ValueError("--frames must be positive except for interactive visualization")
    if not args.visualize and (args.headless or args.snapshot is not None):
        raise ValueError("--headless and --snapshot require --visualize")
    for name in ("palm_radius", "robot_radius"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.snapshot is not None and args.snapshot.suffix.lower() != ".png":
        raise ValueError("--snapshot must end with .png")
    if args.visualize:
        from teleoperation.apps.hand_visualization import run as visualize
        return visualize(args)

    counts = {side: {"raw": 0, "outputs": 0, "resets": 0, "recoveries": 0} for side in SIDES}
    seen_output = {side: False for side in SIDES}
    awaiting_recovery = {side: False for side in SIDES}
    completed = 0
    started = None
    try:
        import torch

        if args.device == "cpu":
            torch.set_num_threads(1)
        model = load_palm_local_retargeter(args.checkpoint, args.device)
        retargeter = PoseTransformerRetargeter(model, args.device)
        print(f"checkpoint={args.checkpoint} alignment={PALM_LOCAL_COORDINATE_ALIGNMENT} identity_tracking=False", flush=True)
        print("Move hands into view, out of view, then back into view to check reset/recovery.", flush=True)
        with MediaPipeCameraWorkflow(args.model_asset_path, args.camera_index) as camera:
            started = time.perf_counter()
            for _ in range(args.frames):
                previous_streak = dict(camera.processor.valid_streak)
                window = camera.next_window()
                payload = None if window is None else window.to_payload()
                result = None if window is None else retargeter.retarget(window)
                predictions = {side: None if result is None or getattr(result, side + "_angles") is None else getattr(result, side + "_angles").values for side in SIDES}
                completed += 1
                descriptions = []
                for side in SIDES:
                    raw = camera.last_raw_frame[side]
                    palm = camera.processor.current_hands[side]
                    window = None if payload is None else payload["hands"][side]
                    angles = predictions[side]
                    streak = camera.processor.valid_streak[side]
                    event = ""
                    counts[side]["raw"] += int(raw is not None)
                    if streak == 0 and previous_streak[side] > 0:
                        counts[side]["resets"] += 1
                        awaiting_recovery[side] = seen_output[side]
                        event = " reset"
                    if angles is not None:
                        if angles.shape != (HAND_ANGLE_DIM,) or not np.isfinite(angles).all():
                            raise ValueError(f"Invalid {side} model output")
                        counts[side]["outputs"] += 1
                        seen_output[side] = True
                        if awaiting_recovery[side]:
                            counts[side]["recoveries"] += 1
                            awaiting_recovery[side] = False
                            event = " recovery_after_3_valid_frames"
                    arrays = (raw, palm, window, angles)
                    shapes = [None if value is None else value.shape for value in arrays]
                    finite = all(np.isfinite(value).all() for value in arrays if value is not None)
                    descriptions.append(
                        f"{side}: raw={shapes[0]} palm={shapes[1]} window={shapes[2]} "
                        f"angles={shapes[3]} finite={bool(finite)} streak={streak} "
                        f"invalid={camera.processor.invalid_reasons[side]!r}{event}"
                    )
                fps = completed / max(time.perf_counter() - started, 1e-9)
                print(f"frame={completed} timestamp={camera.last_raw_frame['timestamp']} unix_ms "
                      f"{' | '.join(descriptions)} FPS={fps:.1f}", flush=True)
    except KeyboardInterrupt:
        print("Capture stopped.")
    except Exception as error:
        print(f"Realtime smoke failed: {error}", flush=True)
        return 1
    elapsed = 0.0 if started is None else time.perf_counter() - started
    print(f"Captured={completed} counts={counts} FPS={completed / max(elapsed, 1e-9):.1f}; resources released")
    if not any(counts[side]["outputs"] for side in SIDES):
        print("Hand windows and real-hand inference were not observed; manual verification is still required.")
    elif not any(counts[side]["recoveries"] for side in SIDES):
        print("Real-hand dropout/recovery was not observed; manual verification is still required.")
    return 0

from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.observations import RawHandFrame


class MediaPipePalmLocalPipeline:
    def __init__(self):
        self.processor = MediaPipePalmLocalProcessor()
        self.buffer = TemporalBuffer()

    @property
    def current_hands(self): return self.processor.current_hands
    @property
    def valid_streak(self): return self.processor.valid_streak
    @property
    def invalid_reasons(self): return self.processor.invalid_reasons

    def process_window(self, frame):
        raw = frame if isinstance(frame, RawHandFrame) else RawHandFrame({side: frame[side] for side in SIDES}, frame.get("timestamp"), "mediapipe_approx", frame.get("metadata", {}))
        canonical = self.processor.process_frame(raw)
        return self.buffer.append(canonical)

    def process_frame(self, frame):
        window = self.process_window(frame)
        return None if window is None else window.to_payload()

    def reset(self): self.processor.reset(); self.buffer.reset()

from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
