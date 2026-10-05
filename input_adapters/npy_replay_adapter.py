"""Replay MediaPipe/VisionPro-style hand keypoint logs from .npy files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from retargeting.hand_core import CanonicalHandProcessor
from retargeting.tracking import MEDIAPIPE_APPROX_SOURCE

NPY_REPLAY_SOURCE = "npy_replay"



class NpyReplayAdapter:
    """Read frame dictionaries and emit canonical retarget input payloads."""

    def __init__(self, path: str | Path, scale_factor: float = 1.0):
        self.path = Path(path)
        self.scale_factor = scale_factor
        self._frames = np.load(self.path, allow_pickle=True)
        self._processor = CanonicalHandProcessor(scale_factor=scale_factor, track_identity=False)
        self._index = 0

    def __iter__(self) -> Iterator[dict]:
        while True:
            payload = self.next_input()
            if payload is None:
                break
            yield payload

    def next_input(self) -> dict | None:
        while self._index < len(self._frames):
            frame = _as_frame_dict(self._frames[self._index])
            self._index += 1

            left_hand = frame.get("left_hand")
            right_hand = frame.get("right_hand")
            source = _source_for_frame(left_hand, right_hand)
            payload, _ = self._processor.process(
                left_hand=left_hand,
                right_hand=right_hand,
                timestamp=frame.get("timestamp"),
                source=source,
                metadata={
                    "path": str(self.path),
                    "frame_id": frame.get("frame_id", self._index - 1),
                    "adapter": NPY_REPLAY_SOURCE,
                },
            )
            if payload is not None:
                return payload

        return None


def _as_frame_dict(frame: object) -> dict:
    if isinstance(frame, dict):
        return frame
    if hasattr(frame, "item"):
        value = frame.item()
        if isinstance(value, dict):
            return value
    raise TypeError(f"Expected frame dict in .npy log, got {type(frame)!r}")


def _source_for_frame(left_hand: object, right_hand: object) -> str:
    for points in (left_hand, right_hand):
        if points is None:
            continue
        points_array = np.asarray(points)
        if points_array.shape == (21, 3):
            return MEDIAPIPE_APPROX_SOURCE
    return NPY_REPLAY_SOURCE
