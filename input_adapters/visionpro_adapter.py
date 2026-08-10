"""Vision Pro input adapter for the canonical retargeting payload."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from input_adapters.hand_keypoints import HandWindowBuffer, VISIONPRO_SOURCE


class VisionProAdapter:
    """Wrap ``VisionProStreamer`` and emit canonical three-frame hand windows."""

    def __init__(
        self,
        ip: str,
        side: str = "left",
        scale_factor: float = 1.0,
        streamer: Any | None = None,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"Invalid Vision Pro hand side: {side}")

        self.side = side
        self._buffer = HandWindowBuffer(scale_factor=scale_factor)
        if streamer is None:
            from avp_stream import VisionProStreamer

            streamer = VisionProStreamer(ip=ip)
        self._streamer = streamer

    def next_input(self) -> dict | None:
        latest = self._streamer.get_latest()
        fingers_key = f"{self.side}_fingers"
        if fingers_key not in latest:
            return None

        points = visionpro_fingers_to_points(latest[fingers_key], side=self.side)
        kwargs = {"left_hand": None, "right_hand": None}
        kwargs[f"{self.side}_hand"] = points
        return self._buffer.update(
            timestamp=time.time(),
            source=VISIONPRO_SOURCE,
            metadata={"side": self.side},
            **kwargs,
        )


def visionpro_fingers_to_points(fingers: list | np.ndarray, side: str = "left") -> np.ndarray:
    """Extract 25 hand points from Vision Pro 4x4 transform matrices."""
    coordinates = []
    for transform_matrix in fingers:
        matrix = np.asarray(transform_matrix)
        if side == "right":
            x = -matrix[1][3]
            y = matrix[2][3]
            z = -matrix[0][3]
        else:
            x = matrix[1][3]
            y = -matrix[2][3]
            z = matrix[0][3]
        coordinates.append([x, y, z])

    return np.asarray(coordinates, dtype=np.float32)
