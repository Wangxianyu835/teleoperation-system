"""Hardware-free and live diagnostic routines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .api import FFGSampleSource
from .device import discover_serial_ports
from .frames import CHANNEL_NAMES


@dataclass(frozen=True)
class ProbeResult:
    success: bool
    frames_requested: int
    frames_received: int
    duration_s: float
    channel_count: int
    finite: bool
    changed: bool
    max_channel_peak_to_peak: float
    channels_with_motion: int
    report: dict[str, Any]


def probe_ffg(
    source: FFGSampleSource,
    *,
    frames: int = 120,
    timeout_s: float = 15.0,
    motion_check: bool = True,
) -> ProbeResult:
    positions: list[np.ndarray] = []
    started = time.monotonic()
    deadline = started + timeout_s
    first_sample_at = None
    while len(positions) < frames and time.monotonic() < deadline:
        sample = source.read(
            timeout_s=min(0.25, max(0.05, deadline - time.monotonic()))
        )
        if sample is None:
            continue
        if first_sample_at is None:
            first_sample_at = time.monotonic()
        positions.append(sample.positions_rad.copy())
    duration_s = 0.0 if first_sample_at is None else time.monotonic() - first_sample_at
    if not positions:
        report = {
            "success": False,
            "reason": "没有收到任何 FFG 数据帧",
            "available_ports": [item.as_dict() for item in discover_serial_ports()],
        }
        return ProbeResult(
            False,
            frames,
            0,
            duration_s,
            0,
            False,
            False,
            0.0,
            0,
            report,
        )

    matrix = np.vstack(positions)
    finite = bool(np.isfinite(matrix).all())
    channel_count = int(matrix.shape[1])
    peak_to_peak = np.ptp(matrix, axis=0)
    channels_with_motion = int(np.count_nonzero(peak_to_peak > 1e-4))
    changed = channels_with_motion > 0
    success = (
        len(positions) >= min(frames, 10)
        and channel_count == 21
        and finite
        and (changed or not motion_check)
    )
    report = {
        "success": success,
        "reason": (
            "连接和动作变化检查通过"
            if success
            else "数据有效，但未观察到预期的手部动作变化"
        ),
        "frames_requested": frames,
        "frames_received": len(positions),
        "duration_s": duration_s,
        "measured_rate_hz": (
            (len(positions) - 1) / duration_s if duration_s > 0 else None
        ),
        "channel_count": channel_count,
        "finite": finite,
        "changed": changed,
        "channels_with_motion": channels_with_motion,
        "max_channel_peak_to_peak_rad": float(np.max(peak_to_peak)),
        "channel_peak_to_peak_rad": {
            name: float(value) for name, value in zip(CHANNEL_NAMES, peak_to_peak)
        },
    }
    return ProbeResult(
        success,
        frames,
        len(positions),
        duration_s,
        channel_count,
        finite,
        changed,
        float(np.max(peak_to_peak)),
        channels_with_motion,
        report,
    )
