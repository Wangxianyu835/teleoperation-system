"""Shared hand-only canonical processing boundary.

Source adapters provide raw 21/25 point arrays, side labels, timestamps, and
metadata.  This module owns identity continuity, canonical 25-point conversion,
wrist-relative normalization, and the chronological receptive-field window.
Coordinate-frame alignment is intentionally outside this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from retargeting.contracts import HAND_SIDES, build_retarget_input
from retargeting.tracking import (
    DEFAULT_MAX_CENTER_DISPLACEMENT,
    DEFAULT_MAX_SHAPE_RMSE,
    HandIdentityTracker,
    HandWindowBuffer,
    _is_trackable_hand,
    ensure_hand25,
)


HandPoints = np.ndarray | None


class _HandMetadataView:
    """Expose unresolved coordinate metadata without inferring values."""

    metadata: Mapping[str, Any]

    @property
    def coordinate_frame(self) -> str:
        return str(self.metadata.get("coordinate_frame", "unknown"))

    @property
    def units(self) -> str:
        return str(self.metadata.get("units", "unknown"))


@dataclass(frozen=True)
class RawHandFrame(_HandMetadataView):
    """One source frame before canonical hand processing."""

    hands: dict[str, HandPoints]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalHandFrame(_HandMetadataView):
    """One frame after topology conversion and wrist-relative normalization."""

    hands: dict[str, HandPoints]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandWindow(_HandMetadataView):
    """A chronological window of canonical hand frames."""

    hands: dict[str, np.ndarray | None]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return the existing retarget input mapping without changing values."""
        return build_retarget_input(
            source=self.source,
            timestamp=self.timestamp,
            left_hand=self.hands["left"],
            right_hand=self.hands["right"],
            metadata=dict(self.metadata),
        )


__all__ = [
    "CanonicalHandProcessor",
    "CanonicalHandFrame",
    "HandWindow",
    "RawHandFrame",
]


class CanonicalHandProcessor:
    """Process raw hand points into canonical three-frame hand payloads."""

    def __init__(
        self,
        scale_factor: float = 1.0,
        receptive_field: int = 3,
        track_identity: bool = True,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        self.track_identity = bool(track_identity)
        self._identity_tracker = HandIdentityTracker(
            max_center_displacement=max_center_displacement,
            max_shape_rmse=max_shape_rmse,
        )
        self._buffer = HandWindowBuffer(
            scale_factor=scale_factor,
            receptive_field=receptive_field,
        )

    def update(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Process one labeled frame and return a payload when a window is ready."""
        payload, _ = self.process(
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=timestamp,
            source=source,
            metadata=metadata,
        )
        return payload

    def process_frame(
        self,
        frame: RawHandFrame,
    ) -> tuple[HandWindow | None, CanonicalHandFrame]:
        """Process a typed raw frame through the shared hand boundary."""
        tracked = (
            self._identity_tracker.update(
                left_hand=frame.hands["left"],
                right_hand=frame.hands["right"],
            )
            if self.track_identity
            else dict(frame.hands)
        )
        return self._process_tracked(
            tracked=tracked,
            timestamp=frame.timestamp,
            source=frame.source,
            metadata=frame.metadata,
        )

    def process(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, np.ndarray]]:
        """Process one frame and also return its converted current-side points.

        The current-side mapping is used by offline dataset assembly for its
        newest-frame target.  The payload remains the public realtime result.
        """
        window, canonical = self.process_frame(
            RawHandFrame(
                hands={"left": left_hand, "right": right_hand},
                timestamp=timestamp,
                source=source,
                metadata={} if metadata is None else metadata,
            )
        )
        return (None if window is None else window.to_payload(), {
            side: points
            for side, points in canonical.hands.items()
            if points is not None
        })

    def _process_tracked(
        self,
        tracked: dict[str, np.ndarray | None],
        timestamp: float | None,
        source: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[HandWindow | None, CanonicalHandFrame]:
        current: dict[str, HandPoints] = {side: None for side in HAND_SIDES}
        for side in HAND_SIDES:
            points = tracked[side]
            if not _is_trackable_hand(points):
                self._buffer.reset_side(side)
                continue

            array = np.asarray(points, dtype=np.float32)
            current[side] = ensure_hand25(array, scale_factor=self._buffer.scale_factor)

        payload = self._buffer.update_canonical(
            left_hand=current["left"],
            right_hand=current["right"],
            timestamp=timestamp,
            source=source,
            metadata=metadata,
        )
        canonical = CanonicalHandFrame(
            hands=current,
            timestamp=timestamp,
            source=source,
            metadata={} if metadata is None else metadata,
        )
        if payload is None:
            return None, canonical
        return (
            HandWindow(
                hands=payload["hands"],
                timestamp=payload.get("timestamp"),
                source=payload["source"],
                metadata=payload.get("metadata", {}),
            ),
            canonical,
        )

    def update_detections(
        self,
        detections: list[tuple[str, np.ndarray]],
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Process source-labeled detections after identity assignment."""
        window, _ = self.process_detections(
            detections=detections,
            timestamp=timestamp,
            source=source,
            metadata=metadata,
        )
        return None if window is None else window.to_payload()

    def process_detections(
        self,
        detections: list[tuple[str, np.ndarray]],
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[HandWindow | None, CanonicalHandFrame]:
        """Process source-labeled detections and return payload/current points."""
        if not self.track_identity:
            labeled = {side: None for side in HAND_SIDES}
            for label, points in detections:
                if label not in labeled:
                    raise ValueError(f"Invalid hand side: {label}")
                if labeled[label] is None:
                    labeled[label] = points
            return self._process_tracked(
                tracked=labeled,
                timestamp=timestamp,
                source=source,
                metadata=metadata,
            )

        tracked = self._identity_tracker.update_detections(detections)
        # Route the already-assigned tracks through the common canonical stage
        # without applying identity assignment a second time.
        return self._process_tracked(
            tracked=tracked,
            timestamp=timestamp,
            source=source,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Reset identity state and both side windows."""
        self._identity_tracker.reset()
        self._buffer.reset()

    def reset_side(self, side: str) -> None:
        """Reset one side's canonical window."""
        self._buffer.reset_side(side)
