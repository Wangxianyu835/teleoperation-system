"""Single-frame hand algorithms. Application/learning code owns temporal state."""
import numpy as np
from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.contracts.observations import RawHandFrame
from .tracking import HandIdentityTracker, _is_trackable_hand, DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from .topology import ensure_hand25, visionpro_fingers_to_points
from teleoperation.contracts.coordinates import PALM_LOCAL_METADATA
from .coordinates import align_palm_local_coordinates, build_l21_reference_basis


class CanonicalHandProcessor:
    def __init__(self, scale_factor=1.0, track_identity=True, max_center_displacement=DEFAULT_MAX_CENTER_DISPLACEMENT, max_shape_rmse=DEFAULT_MAX_SHAPE_RMSE, discard_untrackable=True):
        self.scale_factor = scale_factor
        self.track_identity = track_identity
        self.discard_untrackable = discard_untrackable
        self._identity_tracker = HandIdentityTracker(max_center_displacement, max_shape_rmse)

    def process_frame(self, frame: RawHandFrame):
        if not isinstance(frame, RawHandFrame): raise TypeError("Expected RawHandFrame")
        raw = dict(frame.hands)
        for side in ("left", "right"):
            if raw[side] is not None and frame.geometry_encoding.get(side) == "transforms25":
                raw[side] = visionpro_fingers_to_points(raw[side], side)
        tracked = self._identity_tracker.update(raw["left"], raw["right"]) if self.track_identity else raw
        return self._canonical(tracked, frame)

    def _canonical(self, tracked, frame):
        current = {side: None for side in ("left", "right")}
        for side, points in tracked.items():
            if points is not None and (not self.discard_untrackable or _is_trackable_hand(points)):
                current[side] = ensure_hand25(np.asarray(points, dtype=np.float32), scale_factor=self.scale_factor)
        return CanonicalHandFrame(current, frame.timestamp, frame.source, frame.metadata)

    def process_detections(self, detections, timestamp=None, source="unknown", metadata=None):
        if self.track_identity:
            tracked = self._identity_tracker.update_detections(detections)
        else:
            tracked = {"left": None, "right": None}
            for label, points in detections:
                if label not in tracked: raise ValueError(f"Invalid hand side: {label}")
                if tracked[label] is None: tracked[label] = points
        return self._canonical(tracked, RawHandFrame({"left": None, "right": None}, timestamp, source, metadata or {}))

    def reset(self): self._identity_tracker.reset()

SIDES = ("left", "right")
class MediaPipePalmLocalProcessor:
    """Camera-independent processing of the raw input's single-frame mapping."""

    def __init__(self):
        # Fatal robot-reference errors propagate before any camera is opened.
        # Each zero-pose robot basis is computed exactly once for this stream.
        self._robot_bases = {side: build_l21_reference_basis(side) for side in SIDES}
        
        self.current_hands = {side: None for side in SIDES}
        self.valid_streak = {side: 0 for side in SIDES}
        self.invalid_reasons = {side: None for side in SIDES}

    def process_frame(self, frame: RawHandFrame) -> CanonicalHandFrame:
        """Align each raw21 hand before appending to its own buffer.

        current_hands exposes this frame's aligned (25,3) values for smoke and
        equivalence checks. It never holds an old frame on invalid input.
        """
        current = {side: None for side in SIDES}
        for side in SIDES:
            raw = frame.hands[side]
            reason = frame.metadata.get("invalid_reasons", {}).get(side, "missing") if raw is None else None
            if raw is not None:
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        points = np.asarray(raw, dtype=np.float32)
                        if points.shape != (21, 3):
                            raise ValueError(f"raw shape must be (21,3), got {points.shape}")
                        if not np.isfinite(points).all():
                            raise ValueError("nonfinite raw landmarks")
                        points25 = ensure_hand25(points)
                        aligned = align_palm_local_coordinates(points25, self._robot_bases[side])
                    if aligned.shape != (25, 3) or not np.isfinite(aligned).all() or not np.any(aligned):
                        reason = "invalid or degenerate palm-local frame"
                    else:
                        current[side] = aligned
                except (TypeError, ValueError, OverflowError) as error:
                    reason = str(error)
            if current[side] is None:
                # update_canonical(None) alone does not clear old history.
                
                self.valid_streak[side] = 0
            else:
                self.valid_streak[side] = min(self.valid_streak[side] + 1, 3)
            self.invalid_reasons[side] = reason

        self.current_hands = current
        metadata = {
            **frame.metadata,
            **PALM_LOCAL_METADATA,
            "identity_tracking": False,
            "hand_valid": {side: current[side] is not None for side in SIDES},
            "invalid_reasons": dict(self.invalid_reasons),
        }
        # Already aligned: never use update(), which reconverts coordinates.
        return CanonicalHandFrame(current, frame.timestamp, "mediapipe_approx", metadata)

    def reset(self) -> None:
        self.current_hands = {side: None for side in SIDES}
        self.valid_streak = {side: 0 for side in SIDES}
        self.invalid_reasons = {side: None for side in SIDES}
