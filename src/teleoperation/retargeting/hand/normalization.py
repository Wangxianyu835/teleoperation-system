"""Causal palm-length normalization; no calibration, devices or robot geometry."""
import numpy as np

from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow
from teleoperation.contracts.coordinates import (
    WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT, WORLD_LANDMARK_SPACE, CANONICAL_HAND_FRAME,
    PALM_LENGTH_ALIGNMENT, PALM_LENGTH_PROFILE, validate_landmark_contract,
    palm_length_metadata,
)
from .coordinates import build_palm_basis

MIN_PALM_LENGTH_M = 1e-6


def _palm_frame_length(values):
    points = np.asarray(values,dtype=np.float64)
    if points.shape != (25,3) or not np.isfinite(points).all():
        raise ValueError('palm frame must contain finite hand25 points')
    points = points-points[0]
    length = float(np.linalg.norm(points[11]))
    if length <= MIN_PALM_LENGTH_M:
        raise ValueError('palm length does not exceed 1e-6 metres')
    build_palm_basis(points,(0,6,11))
    return length


def validate_metric_hand_frame(frame: CanonicalHandFrame) -> CanonicalHandFrame:
    """Reject invalid sides before buffering, including during initial warmup.

    This only validates metric geometry; normalization still occurs once at the
    complete window boundary. Later valid frames are never discarded for an
    older bad frame that should not have entered history.
    """
    if not isinstance(frame,CanonicalHandFrame):
        raise TypeError('Palm frame validation requires CanonicalHandFrame')
    if frame.metadata.get('coordinate_alignment') != WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT or frame.units != 'm':
        raise ValueError('Palm frame validation requires canonical world metres')
    validate_landmark_contract(WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT,frame.metadata,'Palm frame validation')
    hands, reasons = dict(frame.hands), dict(frame.invalid_reasons)
    for side, points in hands.items():
        if points is not None:
            try:
                _palm_frame_length(points)
            except ValueError as error:
                hands[side], reasons[side] = None, str(error)
    return CanonicalHandFrame(hands,frame.timestamp,frame.source,{**frame.metadata,'invalid_reasons':reasons})


def normalize_hand_window(window: HandWindow) -> HandWindow:
    if not isinstance(window, HandWindow):
        raise TypeError("Palm normalization requires HandWindow")
    alignment = window.metadata.get("coordinate_alignment")
    if alignment != WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT or window.units != "m":
        raise ValueError("Palm normalization requires canonical world metres, exactly once")
    validate_landmark_contract(alignment, window.metadata, "Palm normalization input")
    hands, lengths, reasons = {}, {}, dict(window.metadata.get("invalid_reasons", {}))
    for side in ("left", "right"):
        hands[side], lengths[side] = None, None
        values = window.hands[side]
        if values is None:
            reasons.setdefault(side, "missing or window not ready")
            continue
        if np.shape(values) != (3, 25, 3):
            raise ValueError("Palm normalization requires exactly three hand25 frames")
        points = np.asarray(values, dtype=np.float64)
        points = points - points[:, 0:1]
        try:
            frame_lengths = [_palm_frame_length(frame) for frame in points]
        except ValueError as error:
            reasons[side] = str(error)
            continue
        denominator = float(np.median(frame_lengths))
        hands[side] = (points / denominator).astype(np.float32)
        lengths[side], reasons[side] = denominator, None
    metadata = dict(window.metadata)
    metadata.pop("source_scale", None)
    metadata.update({
        **palm_length_metadata(),
        "source_length_unit": "m",
        "palm_length_m": lengths, "invalid_reasons": reasons,
    })
    return HandWindow(hands, window.timestamp, window.source, metadata)
