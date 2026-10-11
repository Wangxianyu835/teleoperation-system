"""Unprocessed sensor observations. Arrays retain their original precision."""
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    timestamp: float
    frame_id: int


@dataclass(frozen=True)
class RawHandFrame:
    hands: dict[str, np.ndarray | None]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    geometry_encoding: Mapping[str, str] = field(default_factory=dict)
    world_hands: dict[str, np.ndarray | None] | None = None

    def __post_init__(self):
        for side in ("left", "right"):
            if side not in self.hands:
                raise KeyError(f"RawHandFrame missing {side}")
            value = self.hands[side]
            encoding = self.geometry_encoding.get(side, "landmarks")
            if encoding not in ("landmarks", "landmarks21", "landmarks25", "transforms25"):
                raise ValueError(f"Unknown geometry encoding: {encoding}")
            if value is not None and encoding == "transforms25" and np.shape(value) != (25, 4, 4):
                raise ValueError("Vision Pro transforms must have shape (25,4,4)")
        if self.world_hands is not None:
            for side in ("left", "right"):
                if side not in self.world_hands:
                    raise KeyError(f"RawHandFrame world_hands missing {side}")
                value = self.world_hands[side]
                if value is not None and np.shape(value) != (21, 3):
                    raise ValueError(f"{side} world hand shape must be (21,3)")


    def validate(self, *, strict=True):
        for side in ("left", "right"):
            points = self.hands[side]
            if points is None: continue
            encoding = self.geometry_encoding.get(side, "landmarks")
            shape = np.shape(points)
            accepted = ((25, 4, 4),) if encoding == "transforms25" else ((21, 3),) if encoding == "landmarks21" else ((25, 3),) if encoding == "landmarks25" else ((21, 3), (25, 3))
            if shape not in accepted: raise ValueError(f"{side} raw hand shape {shape} conflicts with {encoding}")
            if strict and not np.isfinite(points).all(): raise ValueError(f"{side} raw hand contains NaN or Inf")
        if self.world_hands is not None:
            for side, points in self.world_hands.items():
                if points is None:
                    continue
                if np.shape(points) != (21, 3):
                    raise ValueError(f"{side} world hand shape must be (21,3)")
                if strict and not np.isfinite(points).all():
                    raise ValueError(f"{side} world hand contains NaN or Inf")
        return self


@dataclass(frozen=True)
class UpperBodyObservation:
    arms: dict[str, np.ndarray | None]
    valid: dict[str, bool]
    timestamp: float | None = None
    coordinate_frame: str = "unknown"
    units: str = "unknown"
    world_arms: dict[str, np.ndarray | None] | None = None

    def __post_init__(self):
        for side in ("left", "right"):
            if side not in self.arms or side not in self.valid:
                raise KeyError(f"UpperBodyObservation missing {side}")
            points = self.arms[side]
            if points is not None and np.shape(points) != (3, 3):
                raise ValueError(f"{side} shoulder/elbow/wrist must have shape (3,3)")
        if self.world_arms is not None:
            for side in ("left", "right"):
                if side not in self.world_arms:
                    raise KeyError(f"UpperBodyObservation world_arms missing {side}")
                points = self.world_arms[side]
                if points is not None and np.shape(points) != (3, 3):
                    raise ValueError(f"{side} world shoulder/elbow/wrist must have shape (3,3)")

    def validate(self, *, strict: bool = True):
        for side, points in self.arms.items():
            if self.valid[side] and points is None:
                raise ValueError(f"{side} valid observation is missing")
            if strict and points is not None and not np.isfinite(points).all():
                raise ValueError(f"{side} arm contains NaN or infinite values")
        if self.world_arms is not None:
            for side, points in self.world_arms.items():
                if points is None:
                    continue
                if np.shape(points) != (3, 3):
                    raise ValueError(f"{side} world shoulder/elbow/wrist must have shape (3,3)")
                if strict and not np.isfinite(points).all():
                    raise ValueError(f"{side} world arm contains NaN or infinite values")
        return self


@dataclass(frozen=True)
class RawTeleopObservation:
    hands: RawHandFrame
    upper_body: UpperBodyObservation

    def __post_init__(self):
        if not isinstance(self.hands, RawHandFrame) or not isinstance(self.upper_body, UpperBodyObservation):
            raise TypeError("RawTeleopObservation requires raw hands and an upper-body observation")


RawObservation = CameraFrame | RawHandFrame | UpperBodyObservation | RawTeleopObservation
