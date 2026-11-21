"""Classify MediaPipe world landmarks independently of the target hand."""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


DETECTOR_PARAMETERS = MappingProxyType({
    "minimum_palm_width_m": 1e-6,
    "index_preference_palm_widths": .10,
    "pinch_strength_span": .80,
    "curl_strength_span": .55,
    "minimum_strength": .10,
    "fist_minimum_curled_fingers": 3,
})
_FINGER_MCPS = np.array([5, 9, 13, 17])
_FINGER_TIPS = np.array([8, 12, 16, 20])


@dataclass(frozen=True)
class GraspIntent:
    kind: str
    strength: float
    valid: bool
    source_time: float
    reason: str = ""


def detect_grasp_intent(points21, *, timestamp: float) -> GraspIntent:
    """Classify a single source-side world21 frame using palm-width ratios."""
    stamp = float(timestamp)
    if not np.isfinite(stamp):
        raise ValueError("Grasp intent requires a finite source timestamp")
    if points21 is None:
        return GraspIntent("none", 0., False, stamp, "missing hand")
    points = np.asarray(points21, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return GraspIntent("none", 0., False, stamp, "invalid world21 landmarks")
    params = DETECTOR_PARAMETERS
    width = np.linalg.norm(points[5] - points[17])
    if width < params["minimum_palm_width_m"]:
        return GraspIntent("none", 0., False, stamp, "degenerate palm width")
    index, middle = np.linalg.norm(points[[8, 12]] - points[4], axis=1) / width
    use_index = index <= middle + params["index_preference_palm_widths"]
    ratio = index if use_index else middle
    pinch = float(np.clip((1 - ratio) / params["pinch_strength_span"], 0., 1.))
    spans = np.linalg.norm(points[_FINGER_TIPS] - points[_FINGER_MCPS], axis=1) / width
    curls = np.clip((1 - spans) / params["curl_strength_span"], 0., 1.)
    curl = float(np.median(curls))
    minimum = params["minimum_strength"]
    if (np.count_nonzero(curls >= minimum) >= params["fist_minimum_curled_fingers"]
            and curl >= max(minimum, pinch)):
        return GraspIntent("fist", curl, True, stamp)
    if pinch >= minimum:
        return GraspIntent("index" if use_index else "middle", pinch, True, stamp)
    return GraspIntent("none", 0., True, stamp)
