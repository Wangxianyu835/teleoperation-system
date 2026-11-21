"""Raw FFG anchor calibration independent of target robot hands."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .api import FFGSampleSource
from .config import CalibrationConfig
from .console import timestamp_label
from .frames import CHANNEL_NAMES, EXPECTED_FFG_CHANNELS


@dataclass(frozen=True)
class PoseAnchor:
    pose: str
    positions_rad: np.ndarray
    channel_std_rad: np.ndarray
    mean_variance: float
    warning_channels: tuple[str, ...]
    sample_count: int
    stable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose,
            "positions_rad": self.positions_rad.tolist(),
            "channel_std_rad": self.channel_std_rad.tolist(),
            "max_channel_std_rad": float(np.max(self.channel_std_rad)),
            "mean_variance": self.mean_variance,
            "warning_channels": list(self.warning_channels),
            "sample_count": self.sample_count,
            "stable": self.stable,
        }


@dataclass(frozen=True)
class CalibrationResult:
    side: str
    anchors: dict[str, PoseAnchor]
    path: Path

    @property
    def stable(self) -> bool:
        return all(anchor.stable for anchor in self.anchors.values())

    @property
    def warnings(self) -> list[str]:
        messages = []
        for pose, anchor in self.anchors.items():
            if anchor.warning_channels:
                messages.append(
                    f"{pose}: " + ", ".join(anchor.warning_channels) + " 抖动较大"
                )
        return messages


POSE_INSTRUCTIONS = {
    "fist": "请握拳，四指尽量握紧，拇指弯曲至中指附近",
    "opose": "请做 O 型手势，拇指与食指形成 O 型",
    "open": "请完全张开手掌，拇指尽量向后展开",
}


def capture_pose_anchor(
    source: FFGSampleSource,
    pose: str,
    config: CalibrationConfig,
) -> PoseAnchor:
    rolling: deque[tuple[float, np.ndarray]] = deque()
    stable_samples: list[np.ndarray] = []
    stable_started: float | None = None
    started = time.monotonic()
    last_message = -1.0
    last_mean_variance = float("inf")

    while time.monotonic() - started < config.max_wait_seconds:
        sample = source.read(timeout_s=0.2)
        if sample is None or not sample.valid:
            continue
        now = time.monotonic()
        elapsed = now - started
        rolling.append((now, sample.positions_rad.copy()))
        cutoff = now - config.stability_window_seconds
        while rolling and rolling[0][0] < cutoff:
            rolling.popleft()

        if elapsed < config.settle_seconds:
            if now - last_message >= 0.5:
                print(
                    f"\r  等待切换姿势 {elapsed:4.1f}/{config.settle_seconds:.1f}s",
                    end="",
                    flush=True,
                )
                last_message = now
            continue
        if len(rolling) < 5:
            continue

        window = np.vstack([value for _, value in rolling])
        last_mean_variance = float(np.var(window, axis=0).mean())
        if last_mean_variance <= config.max_mean_variance:
            if stable_started is None:
                stable_started = now
                stable_samples = []
            stable_samples.append(sample.positions_rad.copy())
            stable_duration = now - stable_started
            if now - last_message >= 0.25:
                print(
                    f"\r  稳定 {stable_duration:4.1f}/{config.hold_seconds:.1f}s "
                    f"mean_var={last_mean_variance:.4f}",
                    end="",
                    flush=True,
                )
                last_message = now
            if stable_duration >= config.hold_seconds:
                break
        else:
            stable_started = None
            stable_samples = []
            if now - last_message >= 0.5:
                print(
                    f"\r  等待稳定 mean_var={last_mean_variance:.4f} "
                    f"(门槛 {config.max_mean_variance:.4f})",
                    end="",
                    flush=True,
                )
                last_message = now

    print()
    if not stable_samples or len(stable_samples) < 5:
        raise RuntimeError(
            f"标定姿势 {pose} 在 {config.max_wait_seconds:.0f} 秒内未达到稳定条件，"
            f"最后 mean_var={last_mean_variance:.4f}"
        )
    matrix = np.vstack(stable_samples)
    if matrix.shape[1] != EXPECTED_FFG_CHANNELS:
        raise ValueError(f"标定数据通道数异常: {matrix.shape}")
    std = np.std(matrix, axis=0)
    mean_variance = float(np.var(matrix, axis=0).mean())
    warning_channels = tuple(
        name
        for name, value in zip(CHANNEL_NAMES, std, strict=True)
        if value > config.channel_std_warning_rad
    )
    return PoseAnchor(
        pose=pose,
        positions_rad=np.median(matrix, axis=0),
        channel_std_rad=std,
        mean_variance=mean_variance,
        warning_channels=warning_channels,
        sample_count=len(matrix),
        stable=bool(mean_variance <= config.max_mean_variance),
    )


def run_calibration(
    source: FFGSampleSource,
    config: CalibrationConfig,
    *,
    device_info: dict[str, Any],
    input_fn: Callable[[str], str] = input,
) -> CalibrationResult:
    poses = ("fist", "opose", "open") if config.show_fist else ("opose", "open")
    anchors: dict[str, PoseAnchor] = {}
    print("即将开始 FFG 原始位置锚点标定。")
    print(
        f"进入姿势后会先等待 {config.settle_seconds:.1f} 秒，"
        f"再要求连续稳定 {config.hold_seconds:.1f} 秒。"
    )
    for pose in poses:
        print()
        print(f"下一个姿势: {POSE_INSTRUCTIONS[pose]}")
        input_fn("准备好后按 Enter 开始...")
        print("请切换到该姿势并保持不动。程序会自动等待稳定。")
        anchors[pose] = capture_pose_anchor(
            source,
            pose,
            config,
        )
        anchor = anchors[pose]
        print(
            f"  {pose}: {anchor.sample_count} 帧，"
            f"mean_var={anchor.mean_variance:.6f}，"
            f"最大通道标准差={float(np.max(anchor.channel_std_rad)):.4f} rad，"
            f"稳定={anchor.stable}"
        )
        if anchor.warning_channels:
            print("    抖动警告: " + ", ".join(anchor.warning_channels))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    side = str(device_info.get("side", "unknown"))
    path = config.output_dir / f"ffg_{side}_calibration_{timestamp_label()}.json"
    payload = {
        "schema_version": "ffg_raw_calibration_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "rad",
        "channel_names": list(CHANNEL_NAMES),
        "device": device_info,
        "hold_seconds": config.hold_seconds,
        "settle_seconds": config.settle_seconds,
        "stability_window_seconds": config.stability_window_seconds,
        "stability_method": "mean_channel_variance",
        "max_mean_variance": config.max_mean_variance,
        "channel_std_warning_rad": config.channel_std_warning_rad,
        "anchors": {key: value.as_dict() for key, value in anchors.items()},
        "all_stable": all(value.stable for value in anchors.values()),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return CalibrationResult(side=side, anchors=anchors, path=path.resolve())


def load_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") not in {
        "ffg_raw_calibration_v1",
        "ffg_raw_calibration_v2",
    }:
        raise ValueError("Unsupported FFG calibration schema")
    return payload


def find_latest_calibration(
    directory: str | Path, side: str | None = None
) -> Path | None:
    root = Path(directory)
    pattern = (
        f"ffg_{side}_calibration_*.json"
        if side in {"left", "right"}
        else "ffg_*_calibration_*.json"
    )
    candidates = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime)
    return candidates[-1].resolve() if candidates else None


def resolve_calibration(
    config: CalibrationConfig,
    *,
    side: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    value = config.calibration_file.strip()
    if value.lower() == "auto":
        path = find_latest_calibration(config.output_dir, side)
    elif value:
        path = Path(value).expanduser().resolve()
    else:
        path = None
    return (path, load_calibration(path) if path else None)
