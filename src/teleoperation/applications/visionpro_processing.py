"""TransHandR VR preprocessing, adapted to RawHandFrame and HandWindow.

Reference: main_visionpro_realtime.py:1136-1300 in the user's TransHandR upstream.
The numerical path retains its axis remap, wrist origin, three-frame stack and
post-stack scaling. No camera palm alignment or identity reassignment is added.
"""
from collections import deque

import numpy as np

from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT
from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow
from teleoperation.contracts.observations import RawHandFrame


def transhandr_vr_points(fingers, side="left"):
    """Same numerical operations as the upstream VR data worker."""
    if side not in ("left", "right"):
        raise ValueError(f"Unknown VR hand side: {side}")
    fingers = np.asarray(fingers)
    if fingers.shape != (25, 4, 4) or not np.isfinite(fingers).all():
        raise ValueError("VR fingers must contain 25 finite 4x4 transforms")
    coordinates = []
    for transform_matrix in fingers:
        if side == "left":
            x = transform_matrix[1][3]
            y = -transform_matrix[2][3]
            z = transform_matrix[0][3]
        else:
            # The right-hand mapping from the upstream's adjacent branch/comment.
            x = -transform_matrix[1][3]
            y = transform_matrix[2][3]
            z = -transform_matrix[0][3]
        coordinates.append([x, y, z])
    coordinates = np.array(coordinates)
    wrist_pos = coordinates[0]
    relative_coordinates = coordinates - wrist_pos
    return relative_coordinates


class TransHandRVisionProPipeline:
    """Upstream rolling-three-frame logic with independent explicit side labels."""

    def __init__(self, scale_factor=1.0):
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("VR scale factor must be finite and positive")
        self.scale_factor = float(scale_factor)
        self._frames = {side: deque(maxlen=3) for side in ("left", "right")}
        self.current_hands = {side: None for side in self._frames}
        self.invalid_reasons = {side: None for side in self._frames}

    def process_frame(self, frame):
        if frame is None:
            self.reset()
            return None, None
        if not isinstance(frame, RawHandFrame) or frame.source != "visionpro":
            self.reset()
            raise ValueError("TransHandR VR preprocessing requires RawHandFrame(source='visionpro')")
        if "coordinate_alignment" in frame.metadata:
            self.reset()
            raise ValueError("VR preprocessing requires raw transforms; refusing repeated alignment")
        hands = {}
        for side in self._frames:
            fingers = frame.hands[side]
            reason = None
            points = None
            if fingers is not None:
                if frame.geometry_encoding.get(side) != "transforms25":
                    self.reset()
                    raise ValueError("VR geometry must explicitly declare transforms25")
                try:
                    points = transhandr_vr_points(fingers, side)
                    if not np.any(points):
                        points, reason = None, "zero/placeholder VR hand"
                except ValueError as error:
                    reason = str(error)
            else:
                reason = "missing VR hand"
            self.current_hands[side], self.invalid_reasons[side] = points, reason
            if points is None:
                self._frames[side].clear()
                hands[side] = None
            else:
                self._frames[side].append(points)
                hands[side] = self.model_window(side)
        metadata = {**frame.metadata, "coordinate_alignment": COORDINATE_ALIGNMENT,
                    "coordinate_frame": "l21", "source_landmark_space": "visionpro_world_meters",
                    "units": "m" if self.scale_factor == 1.0 else "scaled_m",
                    "origin": "wrist", "source_scale": self.scale_factor,
                    "vr_preprocessing": "transhandr_axis_wrist_three_frame_v1",
                    "identity_tracking": False, "invalid_reasons": dict(self.invalid_reasons)}
        canonical = CanonicalHandFrame(dict(self.current_hands), frame.timestamp, "visionpro", metadata)
        window = (None if all(value is None for value in hands.values()) else
                  HandWindow(hands, frame.timestamp, "visionpro", metadata))
        return window, canonical

    def model_window(self, side):
        if len(self._frames[side]) != 3:
            return None
        # Keep the original stack+transpose order and scale before float32 casting.
        vision_pro_data = np.stack(tuple(self._frames[side]), axis=1)
        reshaped_data = np.transpose(vision_pro_data, (1, 0, 2))
        reshaped_data = np.expand_dims(reshaped_data, axis=0)
        reshaped_data *= self.scale_factor
        return reshaped_data[0].astype(np.float32)

    def process_window(self, frame):
        return self.process_frame(frame)[0]

    def reset(self):
        for side in self._frames:
            self._frames[side].clear()
            self.current_hands[side] = None
            self.invalid_reasons[side] = None
