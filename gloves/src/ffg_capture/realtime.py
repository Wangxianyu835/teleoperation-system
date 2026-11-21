"""Small realtime counters shared by the live capture workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .api import MotionSample


@dataclass
class RealtimeStats:
    max_gap_threshold_ms: float = 100.0
    frames: int = 0
    invalid_frames: int = 0
    long_gap_count: int = 0
    first_time_ns: int | None = None
    last_time_ns: int | None = None
    last_sequence: int | None = None
    max_gap_ms: float = 0.0

    def update(self, sample: MotionSample) -> None:
        self.frames += 1
        self.invalid_frames += int(not sample.valid)
        self.last_sequence = int(sample.sequence)
        if self.first_time_ns is None:
            self.first_time_ns = int(sample.host_time_ns)
        if self.last_time_ns is not None:
            gap_ms = (int(sample.host_time_ns) - self.last_time_ns) / 1e6
            if gap_ms > 0:
                self.max_gap_ms = max(self.max_gap_ms, gap_ms)
                self.long_gap_count += int(gap_ms > self.max_gap_threshold_ms)
        self.last_time_ns = int(sample.host_time_ns)

    @property
    def duration_s(self) -> float:
        if self.first_time_ns is None or self.last_time_ns is None:
            return 0.0
        return max(0.0, (self.last_time_ns - self.first_time_ns) / 1e9)

    @property
    def rate_hz(self) -> float | None:
        duration = self.duration_s
        if self.frames < 2 or duration <= 0:
            return None
        return (self.frames - 1) / duration

    @property
    def invalid_percent(self) -> float:
        if not self.frames:
            return 0.0
        return 100.0 * self.invalid_frames / self.frames

    def snapshot(self) -> dict[str, float | int | None]:
        rate = self.rate_hz
        return {
            "frames": self.frames,
            "duration_s": self.duration_s,
            "rate_hz": None if rate is None or not np.isfinite(rate) else float(rate),
            "invalid_frames": self.invalid_frames,
            "invalid_percent": self.invalid_percent,
            "max_gap_ms": self.max_gap_ms,
            "long_gap_count": self.long_gap_count,
            "last_sequence": self.last_sequence,
        }
