"""Versioned HDF5 recording for FFG and future motion streams."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .api import FFGSample
from .console import timestamp_label
from .frames import CHANNEL_NAMES, EXPECTED_FFG_CHANNELS

SCHEMA_VERSION = "linker_motion_capture_v1"


def create_session_path(
    output_dir: str | Path,
    stream_id: str,
    *,
    suffix: str = "",
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{stream_id}_{timestamp_label()}"
    if suffix:
        stem += f"_{suffix}"
    candidate = directory / f"{stem}.h5"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index:02d}.h5"
        index += 1
    return candidate


class CaptureRecorder:
    """Append-only writer with one stream group and room for future streams."""

    def __init__(
        self,
        path: str | Path,
        *,
        stream_id: str,
        device_info: dict[str, Any],
        calibration: dict[str, Any] | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream_id = stream_id
        self._file = h5py.File(self.path, "x")
        self._count = 0
        self._closed = False
        self._last_flush = datetime.now(timezone.utc)
        metadata = dict(session_metadata or {})
        metadata.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stream_id": stream_id,
                "channel_count": EXPECTED_FFG_CHANNELS,
                "channel_names": list(CHANNEL_NAMES),
                "position_unit": "rad",
                "layout_id": "ffg_raw21_5_4_4_4_4_v1",
            }
        )
        self._file.attrs["metadata_json"] = json.dumps(
            metadata, ensure_ascii=False, allow_nan=False
        )
        self._file.attrs["schema_version"] = SCHEMA_VERSION

        devices = self._file.create_group("devices")
        device_group = devices.create_group(stream_id)
        for key, value in device_info.items():
            device_group.attrs[key] = "" if value is None else str(value)

        calibrations = self._file.create_group("calibrations")
        if calibration:
            calibrations.attrs["active_calibration_json"] = json.dumps(
                calibration, ensure_ascii=False, allow_nan=False
            )

        streams = self._file.create_group("streams")
        group = streams.create_group(stream_id)
        group.attrs["channel_names_json"] = json.dumps(list(CHANNEL_NAMES))
        group.attrs["position_unit"] = "rad"
        group.attrs["device_kind"] = "linker_ffg"
        group.attrs["layout_id"] = "ffg_raw21_5_4_4_4_4_v1"
        self._group = group
        self._host_time = group.create_dataset(
            "host_time_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(min(1024, 1024),),
        )
        self._device_time = group.create_dataset(
            "device_time_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1024,),
        )
        self._sequence = group.create_dataset(
            "sequence",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1024,),
        )
        self._positions = group.create_dataset(
            "positions_rad",
            shape=(0, EXPECTED_FFG_CHANNELS),
            maxshape=(None, EXPECTED_FFG_CHANNELS),
            dtype=np.float64,
            chunks=(1024, EXPECTED_FFG_CHANNELS),
        )
        self._realforce = group.create_dataset(
            "realforce",
            shape=(0, 5),
            maxshape=(None, 5),
            dtype=np.float64,
            chunks=(1024, 5),
        )
        self._realforce_valid = group.create_dataset(
            "realforce_valid",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=(1024,),
        )
        self._valid = group.create_dataset(
            "valid",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=(1024,),
        )
        self._status = group.create_dataset(
            "status",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(1024,),
        )
        events = self._file.create_group("events")
        self._event_time = events.create_dataset(
            "time_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(128,),
        )
        self._event_sequence = events.create_dataset(
            "stream_sequence",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(128,),
        )
        self._event_label = events.create_dataset(
            "label",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(128,),
        )
        self._event_metadata = events.create_dataset(
            "metadata_json",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(128,),
        )
        self._event_count = 0
        self._file.flush()

    @property
    def count(self) -> int:
        return self._count

    def _resize(self, new_size: int) -> None:
        for dataset in (
            self._host_time,
            self._device_time,
            self._sequence,
            self._positions,
            self._realforce,
            self._realforce_valid,
            self._valid,
            self._status,
        ):
            dataset.resize((new_size, *dataset.shape[1:]))

    def append(self, sample: FFGSample) -> None:
        if self._closed:
            raise RuntimeError("Capture recorder is closed")
        index = self._count
        self._resize(index + 1)
        self._host_time[index] = int(sample.host_time_ns)
        self._device_time[index] = (
            -1 if sample.device_time_ns is None else int(sample.device_time_ns)
        )
        self._sequence[index] = int(sample.sequence)
        self._positions[index, :] = sample.positions_rad
        if sample.force_response is None:
            self._realforce[index, :] = np.nan
            self._realforce_valid[index] = False
        else:
            self._realforce[index, :] = sample.force_response
            self._realforce_valid[index] = True
        self._valid[index] = bool(sample.valid)
        self._status[index] = str(sample.status)
        self._count += 1

        now = datetime.now(timezone.utc)
        if (now - self._last_flush).total_seconds() >= 0.5:
            self._file.flush()
            self._last_flush = now

    def mark_event(
        self,
        label: str,
        *,
        host_time_ns: int | None = None,
        stream_sequence: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Capture recorder is closed")
        index = self._event_count
        for dataset in (
            self._event_time,
            self._event_sequence,
            self._event_label,
            self._event_metadata,
        ):
            dataset.resize((index + 1, *dataset.shape[1:]))
        self._event_time[index] = int(
            time.monotonic_ns() if host_time_ns is None else host_time_ns
        )
        self._event_sequence[index] = (
            self._sequence[-1]
            if stream_sequence is None and self._count
            else int(stream_sequence or 0)
        )
        self._event_label[index] = str(label)
        self._event_metadata[index] = json.dumps(
            metadata or {}, ensure_ascii=False, allow_nan=False
        )
        self._event_count += 1
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock_safe_file():
            # An interrupt can arrive after resizing one dataset but before the
            # complete frame is written. Keep only fully committed frames.
            for dataset in (
                self._host_time,
                self._device_time,
                self._sequence,
                self._positions,
                self._realforce,
                self._realforce_valid,
                self._valid,
                self._status,
            ):
                dataset.resize((self._count, *dataset.shape[1:]))
            self._group.attrs["sample_count"] = self._count
            self._file["events"].attrs["event_count"] = self._event_count
            self._group.attrs["closed_at_utc"] = now
            self._file.attrs["sample_count"] = self._count
            self._file.attrs["closed_at_utc"] = now
            self._file.flush()
            self._file.close()
        self._closed = True

    class _LockSafeFile:
        def __init__(self, handle: h5py.File) -> None:
            self.handle = handle

        def __enter__(self) -> h5py.File:
            return self.handle

        def __exit__(self, *_: object) -> None:
            return None

    def _lock_safe_file(self) -> "_LockSafeFile":
        return CaptureRecorder._LockSafeFile(self._file)

    def __enter__(self) -> "CaptureRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def write_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    summary_path = Path(path).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return summary_path.resolve()
