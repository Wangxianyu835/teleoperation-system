"""Data-quality metrics for recorded FFG streams."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import QualityConfig
from .frames import (
    EXPECTED_FFG_CHANNELS,
    FLEXION_CHANNEL_INDICES,
    SIDE_CHANNEL_INDICES,
)
from .replay import CaptureData


def _safe_float(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def analyze_capture(
    data: CaptureData,
    *,
    neutral_seconds: float = 5.0,
) -> dict[str, Any]:
    count = data.sample_count
    result: dict[str, Any] = {
        "file": str(data.path),
        "stream_id": data.stream_id,
        "schema_version": data.schema_version,
        "sample_count": count,
        "duration_s": 0.0,
        "actual_rate_hz": None,
        "interval_p05_ms": None,
        "interval_p95_ms": None,
        "max_interval_ms": None,
        "invalid_frames": int(count - np.count_nonzero(data.valid)),
        "invalid_fraction": None,
        "long_gap_count": 0,
        "channel_count": int(data.positions_rad.shape[1]) if count else 0,
        "all_finite": False,
        "has_motion": False,
        "overall_std_rad": None,
        "neutral_std_rad": None,
        "neutral_side_std_rad": None,
        "neutral_peak_to_peak_rad": None,
        "channel_min_rad": [],
        "channel_max_rad": [],
        "flexion_min_rad": [],
        "flexion_max_rad": [],
        "min_positive_step_rad": None,
        "saturation_fraction": None,
        "checks": {},
    }
    if count == 0:
        result["checks"] = {"has_frames": False}
        return result

    times = data.host_time_ns.astype(np.float64, copy=False)
    duration_s = max(0.0, (times[-1] - times[0]) / 1e9)
    intervals_ms = np.diff(times) / 1e6
    intervals_ms = intervals_ms[np.isfinite(intervals_ms)]
    valid_positions = data.positions_rad[np.isfinite(data.positions_rad).all(axis=1)]
    result["duration_s"] = duration_s
    result["actual_rate_hz"] = (
        _safe_float((count - 1) / duration_s) if duration_s > 0 else None
    )
    if intervals_ms.size:
        result["interval_p05_ms"] = _safe_float(np.percentile(intervals_ms, 5))
        result["interval_p95_ms"] = _safe_float(np.percentile(intervals_ms, 95))
        median_interval = float(np.median(intervals_ms))
        result["max_interval_ms"] = _safe_float(np.max(intervals_ms))
        result["long_gap_count"] = int(
            np.count_nonzero(intervals_ms > 3 * median_interval)
        )
    result["invalid_fraction"] = _safe_float(
        (count - np.count_nonzero(data.valid)) / count
    )
    result["all_finite"] = bool(np.isfinite(data.positions_rad[data.valid]).all())

    if valid_positions.size:
        flexion_positions = valid_positions[:, FLEXION_CHANNEL_INDICES]
        result["overall_std_rad"] = _safe_float(np.std(flexion_positions))
        result["has_motion"] = bool(np.ptp(flexion_positions, axis=0).max() > 1e-4)
        result["channel_min_rad"] = [
            _safe_float(item) for item in np.min(valid_positions, axis=0)
        ]
        result["channel_max_rad"] = [
            _safe_float(item) for item in np.max(valid_positions, axis=0)
        ]
        result["flexion_min_rad"] = [
            _safe_float(item) for item in np.min(flexion_positions, axis=0)
        ]
        result["flexion_max_rad"] = [
            _safe_float(item) for item in np.max(flexion_positions, axis=0)
        ]

        differences = np.diff(flexion_positions, axis=0)
        positive_steps = np.abs(differences[differences > 1e-10])
        if positive_steps.size:
            result["min_positive_step_rad"] = _safe_float(np.min(positive_steps))

        near_zero = np.isclose(flexion_positions, 0.0, atol=1e-4)
        near_max = np.isclose(flexion_positions, 2.0 * np.pi, atol=1e-4)
        result["saturation_fraction"] = _safe_float(
            np.count_nonzero(near_zero | near_max) / flexion_positions.size
        )

        neutral_count = int(
            np.searchsorted(times, times[0] + neutral_seconds * 1e9, side="right")
        )
        neutral = data.positions_rad[:neutral_count][data.valid[:neutral_count]]
        if neutral.size and neutral.shape[1] == EXPECTED_FFG_CHANNELS:
            result["neutral_std_rad"] = _safe_float(
                np.mean(np.std(neutral[:, FLEXION_CHANNEL_INDICES], axis=0))
            )
            result["neutral_side_std_rad"] = _safe_float(
                np.mean(np.std(neutral[:, SIDE_CHANNEL_INDICES], axis=0))
            )
            result["neutral_peak_to_peak_rad"] = _safe_float(
                np.mean(np.ptp(neutral[:, FLEXION_CHANNEL_INDICES], axis=0))
            )

    result["checks"] = {
        "has_frames": count >= 10,
        "channel_count": int(data.positions_rad.shape[1]) == EXPECTED_FFG_CHANNELS,
        "all_finite": bool(result["all_finite"]),
        "mostly_valid": result["invalid_fraction"] is not None
        and result["invalid_fraction"] <= 0.05,
        "has_measurable_motion": bool(result["has_motion"]),
        "no_large_gaps": int(result["long_gap_count"]) <= max(2, count // 100),
    }
    result["passed"] = bool(all(result["checks"].values()))
    return result


def grade_capture(data: CaptureData, config: QualityConfig) -> dict[str, Any]:
    """Return a compact PASS/WARN/FAIL report for the normal user workflow."""

    analysis = analyze_capture(data, neutral_seconds=config.neutral_seconds)
    failures: list[str] = []
    warnings: list[str] = []

    if analysis["sample_count"] == 0:
        failures.append("没有采集到数据帧")
    if analysis["channel_count"] != config.expected_channels:
        failures.append(
            f"通道数为 {analysis['channel_count']}，预期 {config.expected_channels}"
        )
    if not analysis["all_finite"]:
        failures.append("数据中包含 NaN 或 Inf")

    rate = analysis["actual_rate_hz"]
    if rate is None:
        failures.append("无法计算采样率")
    elif rate < config.expected_min_rate_hz * 0.5:
        failures.append(f"实测帧率过低：{rate:.1f} Hz")
    elif rate < config.expected_min_rate_hz:
        warnings.append(f"实测帧率偏低：{rate:.1f} Hz")

    invalid_fraction = analysis["invalid_fraction"]
    if invalid_fraction is None:
        failures.append("无法计算无效帧比例")
    elif invalid_fraction > config.max_invalid_fraction:
        message = f"无效帧比例偏高：{invalid_fraction * 100:.2f}%"
        (failures if invalid_fraction > 0.10 else warnings).append(message)

    if analysis["long_gap_count"] > max(2, analysis["sample_count"] // 100):
        warnings.append(f"检测到 {analysis['long_gap_count']} 个较长帧间隔")
    if not analysis["has_motion"]:
        warnings.append("没有观察到明显动作变化")

    motion_range = (
        max(
            (high - low)
            for low, high in zip(
                analysis["flexion_min_rad"],
                analysis["flexion_max_rad"],
                strict=False,
            )
            if low is not None and high is not None
        )
        if analysis["flexion_min_rad"]
        else None
    )
    if motion_range is not None and motion_range < config.min_motion_range_rad:
        warnings.append(f"动作范围偏小：{motion_range:.3f} rad")

    static_jitter = analysis["neutral_std_rad"]
    if static_jitter is not None and static_jitter > config.max_static_jitter_rad:
        warnings.append(f"静止抖动偏高：{static_jitter:.4f} rad")

    saturation = analysis["saturation_fraction"]
    if saturation is not None and saturation > config.max_saturation_fraction:
        warnings.append(f"端点饱和比例偏高：{saturation * 100:.1f}%")

    grade = "FAIL" if failures else "WARN" if warnings else "PASS"
    summary = {
        "rate_hz": rate,
        "invalid_percent": (
            None if invalid_fraction is None else invalid_fraction * 100.0
        ),
        "max_gap_ms": analysis["max_interval_ms"],
        "motion_range_rad": motion_range,
        "static_jitter_rad": static_jitter,
        "saturation_percent": (None if saturation is None else saturation * 100.0),
    }
    return {
        "grade": grade,
        "summary": summary,
        "failures": failures,
        "warnings": warnings,
        "analysis": analysis,
    }
