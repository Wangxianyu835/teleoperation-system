"""Device-neutral acquisition contracts.

The FFG source implements these interfaces today. A future Linker TA arm source
can implement the same lifecycle without changing the recorder or scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .frames import EXPECTED_FFG_CHANNELS


@dataclass(frozen=True)
class FFGSample:
    """One frame produced by a Linker FFG device."""

    host_time_ns: int
    sequence: int
    side: str
    positions_rad: np.ndarray
    force_response: np.ndarray | None = None
    device_time_ns: int | None = None
    valid: bool = True
    status: str = "tracking"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in {"left", "right", "unknown"}:
            raise ValueError(f"Unsupported hand side: {self.side}")
        if self.host_time_ns < 0 or self.sequence < 0:
            raise ValueError("Timestamp and sequence must be non-negative")

        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.shape != (EXPECTED_FFG_CHANNELS,):
            raise ValueError(
                f"FFG positions must have shape ({EXPECTED_FFG_CHANNELS},), "
                f"got {positions.shape}"
            )
        if self.valid and not np.isfinite(positions).all():
            raise ValueError("A valid FFG sample contains NaN or Inf")
        positions = positions.copy()
        positions.setflags(write=False)

        force = None
        if self.force_response is not None:
            force = np.asarray(self.force_response, dtype=np.float64).reshape(-1)
            if force.size != 5:
                raise ValueError(
                    f"FFG force response must contain 5 values, got {force.size}"
                )
            force = force.copy()
            force.setflags(write=False)

        metadata = dict(self.metadata)
        object.__setattr__(self, "positions_rad", positions)
        object.__setattr__(self, "force_response", force)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@runtime_checkable
class MotionSample(Protocol):
    """Common fields required from any future hand or arm sample."""

    host_time_ns: int
    sequence: int
    side: str
    valid: bool
    status: str
    metadata: Mapping[str, Any]


@runtime_checkable
class SampleSource(Protocol):
    """Generic lifecycle shared by FFG and future arm acquisition sources."""

    stream_id: str

    def start(self) -> None:
        """Open the device and begin receiving data."""

    def read(self, timeout_s: float = 0.2) -> MotionSample | None:
        """Return the next new sample, or None when the timeout expires."""

    def latest(self) -> MotionSample | None:
        """Return the latest sample without blocking."""

    def stop(self) -> None:
        """Stop acquisition and release hardware resources."""

    @property
    def is_running(self) -> bool:
        """Whether the source is actively receiving data."""


@runtime_checkable
class FFGSampleSource(SampleSource, Protocol):
    """Sample source whose concrete reads are FFG-shaped."""

    def read(self, timeout_s: float = 0.2) -> FFGSample | None:
        """Return the next FFG sample, or None when the timeout expires."""

    def latest(self) -> FFGSample | None:
        """Return the latest FFG sample without blocking."""


@runtime_checkable
class CaptureSink(Protocol):
    def append(self, sample: MotionSample) -> None:
        """Persist or display one sample."""
