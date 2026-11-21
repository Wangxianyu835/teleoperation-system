"""Reading and streaming recorded motion capture sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from .api import FFGSample
from .recorder import SCHEMA_VERSION


@dataclass(frozen=True)
class CaptureData:
    path: Path
    stream_id: str
    schema_version: str
    host_time_ns: np.ndarray
    device_time_ns: np.ndarray
    sequence: np.ndarray
    positions_rad: np.ndarray
    realforce: np.ndarray
    realforce_valid: np.ndarray
    valid: np.ndarray
    status: tuple[str, ...]
    stream_metadata: dict[str, Any]
    device_metadata: dict[str, Any]
    session_metadata: dict[str, Any]
    events: tuple[dict[str, Any], ...] = ()

    @property
    def sample_count(self) -> int:
        return int(self.positions_rad.shape[0])

    @property
    def side(self) -> str:
        return self.stream_id.removeprefix("ffg_")


def list_streams(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if "streams" not in handle:
            return []
        return sorted(handle["streams"].keys())


def load_capture(path: str | Path, stream_id: str | None = None) -> CaptureData:
    target = Path(path).resolve()
    with h5py.File(target, "r") as handle:
        schema = str(handle.attrs.get("schema_version", ""))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported capture schema '{schema}', expected '{SCHEMA_VERSION}'"
            )
        streams = list(handle["streams"].keys())
        if not streams:
            raise ValueError("Capture file contains no streams")
        selected = stream_id or streams[0]
        if selected not in streams:
            raise ValueError(
                f"Stream '{selected}' not found; available streams: {', '.join(streams)}"
            )
        group = handle["streams"][selected]
        metadata_raw = str(handle.attrs.get("metadata_json", "{}"))
        stream_meta = {key: _decode_attr(value) for key, value in group.attrs.items()}
        device_meta = {}
        if "devices" in handle and selected in handle["devices"]:
            device_meta = {
                key: _decode_attr(value)
                for key, value in handle["devices"][selected].attrs.items()
            }
        host_time = np.asarray(group["host_time_ns"][()], dtype=np.int64)
        device_time = np.asarray(group["device_time_ns"][()], dtype=np.int64)
        sequence = np.asarray(group["sequence"][()], dtype=np.int64)
        positions = np.asarray(group["positions_rad"][()], dtype=np.float64)
        realforce = np.asarray(group["realforce"][()], dtype=np.float64)
        realforce_valid = np.asarray(group["realforce_valid"][()], dtype=np.bool_)
        valid = np.asarray(group["valid"][()], dtype=np.bool_)
        status_values = tuple(_decode_attr(item) for item in group["status"][()])
        lengths = {
            len(host_time),
            len(device_time),
            len(sequence),
            len(positions),
            len(realforce),
            len(realforce_valid),
            len(valid),
            len(status_values),
        }
        count = min(lengths) if lengths else 0
        trimmed = max(lengths, default=0) - count
        host_time = host_time[:count]
        device_time = device_time[:count]
        sequence = sequence[:count]
        positions = positions[:count]
        realforce = realforce[:count]
        realforce_valid = realforce_valid[:count]
        valid = valid[:count]
        status = status_values[:count]
        session_metadata = json.loads(metadata_raw)
        if trimmed:
            session_metadata["load_trimmed_trailing_rows"] = trimmed
        events: list[dict[str, Any]] = []
        if "events" in handle:
            event_group = handle["events"]
            event_count = int(event_group["time_ns"].shape[0])
            for index in range(event_count):
                events.append(
                    {
                        "time_ns": int(event_group["time_ns"][index]),
                        "stream_sequence": int(event_group["stream_sequence"][index]),
                        "label": _decode_attr(event_group["label"][index]),
                        "metadata": json.loads(
                            _decode_attr(event_group["metadata_json"][index])
                        ),
                    }
                )
        data = CaptureData(
            path=target,
            stream_id=selected,
            schema_version=schema,
            host_time_ns=host_time,
            device_time_ns=device_time,
            sequence=sequence,
            positions_rad=positions,
            realforce=realforce,
            realforce_valid=realforce_valid,
            valid=valid,
            status=status,
            stream_metadata=stream_meta,
            device_metadata=device_meta,
            session_metadata=session_metadata,
            events=tuple(events),
        )
    if data.positions_rad.ndim != 2 or data.positions_rad.shape[1] != 21:
        raise ValueError(
            f"Invalid FFG position matrix shape: {data.positions_rad.shape}"
        )
    if not (
        len(data.host_time_ns)
        == len(data.sequence)
        == len(data.positions_rad)
        == len(data.valid)
        == len(data.status)
    ):
        raise ValueError("Capture stream arrays have inconsistent lengths")
    return data


def iter_samples(data: CaptureData) -> Iterator[FFGSample]:
    for index in range(data.sample_count):
        force = (
            data.realforce[index].copy() if bool(data.realforce_valid[index]) else None
        )
        device_time = int(data.device_time_ns[index])
        yield FFGSample(
            host_time_ns=int(data.host_time_ns[index]),
            sequence=int(data.sequence[index]),
            side=data.side,
            positions_rad=data.positions_rad[index],
            force_response=force,
            device_time_ns=None if device_time < 0 else device_time,
            valid=bool(data.valid[index]),
            status=data.status[index],
            metadata={"device_kind": data.device_metadata.get("model", "linker_ffg")},
        )


def find_latest_session(directory: str | Path) -> Path:
    root = Path(directory)
    candidates = sorted(root.glob("*.h5"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No capture H5 files found under: {root.resolve()}")
    return candidates[-1].resolve()


def find_latest_in_directories(directories: list[str | Path]) -> Path:
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        try:
            return find_latest_session(root)
        except FileNotFoundError:
            continue
    searched = ", ".join(str(Path(item).resolve()) for item in directories)
    raise FileNotFoundError(f"No capture H5 files found. Searched: {searched}")


def slice_capture(data: CaptureData, start: int, end: int) -> CaptureData:
    start = max(0, int(start))
    end = min(data.sample_count, int(end))
    if end <= start:
        raise ValueError("Capture slice is empty")
    return CaptureData(
        path=data.path,
        stream_id=data.stream_id,
        schema_version=data.schema_version,
        host_time_ns=data.host_time_ns[start:end],
        device_time_ns=data.device_time_ns[start:end],
        sequence=data.sequence[start:end],
        positions_rad=data.positions_rad[start:end],
        realforce=data.realforce[start:end],
        realforce_valid=data.realforce_valid[start:end],
        valid=data.valid[start:end],
        status=data.status[start:end],
        stream_metadata=data.stream_metadata,
        device_metadata=data.device_metadata,
        session_metadata=data.session_metadata,
        events=data.events,
    )


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
