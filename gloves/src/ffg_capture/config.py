"""Configuration loading with paths anchored to the project root."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ffg_capture.yml"


def project_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class DeviceConfig:
    model: str = "linker_ffg"
    side: str = "right"
    port: str = "auto"
    baudrates: tuple[int, ...] = (2_000_000, 460_800)
    serial_query_interval_ms: float = 6.0
    serial_debug: bool = False
    skip_bluetooth_ports: bool = True

    def __post_init__(self) -> None:
        if self.side not in {"left", "right", "auto"}:
            raise ValueError("device.side must be left, right, or auto")
        if not self.baudrates:
            raise ValueError("device.baudrates cannot be empty")
        if not 3.0 <= self.serial_query_interval_ms <= 1_000.0:
            raise ValueError("device.serial_query_interval_ms must be 3..1000")


@dataclass(frozen=True)
class DiagnosticsConfig:
    sample_frames: int = 120
    timeout_s: float = 15.0
    motion_check: bool = True


@dataclass(frozen=True)
class RecordingConfig:
    output_dir: Path = field(default_factory=lambda: project_path("outputs/sessions"))
    default_duration_s: float = 0.0
    save_summary: bool = True
    save_final_plot: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    output_dir: Path = field(
        default_factory=lambda: project_path("outputs/calibration")
    )
    show_fist: bool = True
    hold_seconds: float = 5.0
    settle_seconds: float = 1.0
    stability_window_seconds: float = 1.0
    max_mean_variance: float = 0.03
    channel_std_warning_rad: float = 0.20
    max_wait_seconds: float = 20.0
    calibration_file: str = "auto"


@dataclass(frozen=True)
class VisualizationConfig:
    detailed: bool = False
    max_points: int = 3000
    dpi: int = 140


@dataclass(frozen=True)
class ReplayConfig:
    default_file: str = "auto"


@dataclass(frozen=True)
class RealtimeConfig:
    display_enabled: bool = True
    display_refresh_hz: float = 12.0
    status_interval_s: float = 2.0
    duration_s: float = 0.0
    debug_recording: bool = False


@dataclass(frozen=True)
class QualityConfig:
    expected_channels: int = 21
    neutral_seconds: float = 5.0
    expected_min_rate_hz: float = 20.0
    max_invalid_fraction: float = 0.01
    max_saturation_fraction: float = 0.20
    min_motion_range_rad: float = 0.05
    max_static_jitter_rad: float = 0.05


@dataclass(frozen=True)
class AppConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    logging_level: str = "INFO"
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("output_dir",):
            if key in value.get("recording", {}):
                value["recording"][key] = str(self.recording.output_dir)
            if key in value.get("calibration", {}):
                value["calibration"][key] = str(self.calibration.output_dir)
        value["source_path"] = str(self.source_path) if self.source_path else None
        value["device"]["baudrates"] = list(self.device.baudrates)
        return value

    def ensure_directories(self) -> None:
        for path in (
            self.recording.output_dir,
            self.calibration.output_dir,
            project_path("outputs/diagnostics"),
            project_path("outputs/reports"),
            project_path("outputs/plots"),
            project_path("outputs/logs"),
        ):
            path.mkdir(parents=True, exist_ok=True)


def _block(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration block '{key}' must be a mapping")
    return value


def _tuple_int(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("device.baudrates must be a list")
    return tuple(int(item) for item in value)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = project_path(
        path or os.environ.get("FFG_CAPTURE_CONFIG", DEFAULT_CONFIG_PATH)
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"FFG configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("FFG configuration root must be a mapping")

    device = _block(raw, "device")
    diagnostics = _block(raw, "diagnostics")
    recording = _block(raw, "recording")
    calibration = _block(raw, "calibration")
    visualization = _block(raw, "visualization")
    replay = _block(raw, "replay")
    realtime = _block(raw, "realtime")
    quality = _block(raw, "quality")
    logging_block = _block(raw, "logging")

    cfg = AppConfig(
        device=DeviceConfig(
            model=str(device.get("model", "linker_ffg")),
            side=str(device.get("side", "right")).lower(),
            port=str(device.get("port", "auto")),
            baudrates=_tuple_int(device.get("baudrates", [2_000_000, 460_800])),
            serial_query_interval_ms=max(
                3.0, float(device.get("serial_query_interval_ms", 6.0))
            ),
            serial_debug=bool(device.get("serial_debug", False)),
            skip_bluetooth_ports=bool(device.get("skip_bluetooth_ports", True)),
        ),
        diagnostics=DiagnosticsConfig(
            sample_frames=max(1, int(diagnostics.get("sample_frames", 120))),
            timeout_s=max(0.1, float(diagnostics.get("timeout_s", 15.0))),
            motion_check=bool(diagnostics.get("motion_check", True)),
        ),
        recording=RecordingConfig(
            output_dir=project_path(recording.get("output_dir", "outputs/sessions")),
            default_duration_s=max(0.0, float(recording.get("default_duration_s", 0))),
            save_summary=bool(recording.get("save_summary", True)),
            save_final_plot=bool(recording.get("save_final_plot", True)),
        ),
        calibration=CalibrationConfig(
            output_dir=project_path(
                calibration.get("output_dir", "outputs/calibration")
            ),
            show_fist=bool(calibration.get("show_fist", True)),
            hold_seconds=max(0.5, float(calibration.get("hold_seconds", 5.0))),
            settle_seconds=max(0.0, float(calibration.get("settle_seconds", 1.0))),
            stability_window_seconds=max(
                0.1, float(calibration.get("stability_window_seconds", 1.0))
            ),
            max_mean_variance=max(
                1e-9, float(calibration.get("max_mean_variance", 0.03))
            ),
            channel_std_warning_rad=max(
                0.0, float(calibration.get("channel_std_warning_rad", 0.20))
            ),
            max_wait_seconds=max(1.0, float(calibration.get("max_wait_seconds", 20.0))),
            calibration_file=str(calibration.get("calibration_file", "auto")),
        ),
        visualization=VisualizationConfig(
            detailed=bool(visualization.get("detailed", False)),
            max_points=max(100, int(visualization.get("max_points", 3000))),
            dpi=max(70, int(visualization.get("dpi", 140))),
        ),
        replay=ReplayConfig(default_file=str(replay.get("default_file", "auto"))),
        realtime=RealtimeConfig(
            display_enabled=bool(realtime.get("display_enabled", True)),
            display_refresh_hz=max(
                1.0, float(realtime.get("display_refresh_hz", 12.0))
            ),
            status_interval_s=max(0.5, float(realtime.get("status_interval_s", 2.0))),
            duration_s=max(0.0, float(realtime.get("duration_s", 0.0))),
            debug_recording=bool(realtime.get("debug_recording", False)),
        ),
        quality=QualityConfig(
            expected_channels=int(quality.get("expected_channels", 21)),
            neutral_seconds=max(0.0, float(quality.get("neutral_seconds", 5.0))),
            expected_min_rate_hz=max(
                0.1, float(quality.get("expected_min_rate_hz", 20.0))
            ),
            max_invalid_fraction=max(
                0.0, float(quality.get("max_invalid_fraction", 0.01))
            ),
            max_saturation_fraction=max(
                0.0, float(quality.get("max_saturation_fraction", 0.20))
            ),
            min_motion_range_rad=max(
                0.0, float(quality.get("min_motion_range_rad", 0.05))
            ),
            max_static_jitter_rad=max(
                0.0, float(quality.get("max_static_jitter_rad", 0.05))
            ),
        ),
        logging_level=str(logging_block.get("level", "INFO")).upper(),
        source_path=config_path,
    )
    cfg.ensure_directories()
    return cfg
