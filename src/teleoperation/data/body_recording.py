"""Versioned SMPLer-X body recording storage.

This format is deliberately separate from the historical hand/pose capture
format.  It stores the source frame identity and model provenance next to the
raw and derived body values, so a replay cannot silently reinterpret a Pose
recording as an SMPL-X result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from .transaction import atomic_output


SCHEMA_VERSION = "smplerx_body_recording_h5_v1"
REQUIRED_METADATA = ("source", "model_id", "model_hash", "time_unit")
LEGACY_MARKERS = ("mediapipe_pose", "pose_visibility", "pose_model")
ARRAY_FIELDS = (
    'rgb', 'observation_packet', 'worker_packet',
    "joints_root_m",
    "global_rotations",
    "image_joints",
    "body_pose",
    "shape",
    "wrist_quat_left",
    "wrist_quat_right",
)
SCALAR_FIELDS = (
    "sample_id",
    "capture_time_s",
    "completed_time_s",
    "valid",
    "reason",
    "state_left",
    "state_right",
)


@dataclass(frozen=True)
class BodyRecording:
    samples: tuple[dict, ...]
    metadata: dict


def _reject_legacy(value, *, context: str) -> None:
    text = str(value).lower()
    if any(marker in text for marker in LEGACY_MARKERS):
        raise ValueError(f"SMPL-X reader rejects legacy MediaPipe Pose body data: {context}")


def _metadata(metadata: Mapping | None, rows: list[Mapping]) -> dict:
    first = rows[0].get("metadata", {}) if rows else {}
    merged = {**dict(first), **({} if metadata is None else dict(metadata))}
    for key in REQUIRED_METADATA:
        if key not in merged:
            raise ValueError(f"SMPL-X body recording metadata requires {key}")
    merged.setdefault("schema_version", SCHEMA_VERSION)
    merged.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    if merged["source"] != "smplerx":
        raise ValueError("SMPL-X body recording source must be exactly 'smplerx'")
    if merged["time_unit"] != "monotonic_s":
        raise ValueError("SMPL-X body recording time_unit must be 'monotonic_s'")
    if not str(merged["model_hash"]).strip():
        raise ValueError("SMPL-X body recording requires a non-empty model_hash")
    for key, value in merged.items():
        _reject_legacy(value, context=f"metadata.{key}")
    return merged


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported recording value: {type(value)!r}")


def save_body_recording(
    path: str | Path,
    samples: Iterable[Mapping],
    metadata: Mapping | None = None,
) -> Path:
    rows = [dict(row) for row in samples]
    if not rows:
        raise ValueError("SMPL-X body recording requires at least one sample")
    attributes = _metadata(metadata, rows)
    for index, row in enumerate(rows):
        if "capture_time_s" not in row or not np.isfinite(float(row["capture_time_s"])):
            raise ValueError(f"sample {index} requires finite capture_time_s")
        if "sample_id" not in row:
            raise ValueError(f"sample {index} requires sample_id")
        for key, value in row.items():
            _reject_legacy(key, context=f"sample.{key}")
            _reject_legacy(value, context=f"sample.{key}")
    destination = Path(path)
    with atomic_output(destination) as temporary:
        with h5py.File(temporary, "w") as h5_file:
            h5_file.attrs["schema_version"] = SCHEMA_VERSION
            h5_file.attrs["source"] = "smplerx"
            h5_file.attrs["time_unit"] = "monotonic_s"
            h5_file.attrs["metadata_json"] = json.dumps(attributes, default=_json_default, sort_keys=True)
            if all('observation_packet' in row for row in rows):
                _write_hands(h5_file, rows)
            group = h5_file.create_group("samples")
            for index, row in enumerate(rows):
                sample = group.create_group(str(index))
                for key in SCALAR_FIELDS:
                    if key not in row:
                        continue
                    value = row[key]
                    if isinstance(value, (bool, np.bool_)):
                        sample.attrs[key] = bool(value)
                    elif isinstance(value, (int, float, np.integer, np.floating)):
                        sample.attrs[key] = value
                    else:
                        sample.attrs[key] = str(value)
                for key in ARRAY_FIELDS:
                    if key not in row or row[key] is None:
                        continue
                    array = np.asarray(row[key])
                    if not np.isfinite(array).all():
                        raise ValueError(f"sample {index} field {key} must be finite")
                    sample.create_dataset(key, data=array, compression='gzip' if key == 'rgb' else None)
                extra = {key: value for key, value in row.items()
                         if key not in SCALAR_FIELDS and key not in ARRAY_FIELDS and key != "metadata"}
                if extra:
                    sample.attrs["extra_json"] = json.dumps(extra, default=_json_default, sort_keys=True)
    return destination


def _write_hands(handle, rows):
    """Expose finger-only tables to the existing hand training/export tools."""
    observations = [replay_observation(row) for row in rows]
    handle.create_dataset('frame_ids', data=np.arange(len(rows), dtype=np.int64))
    handle.create_dataset('timestamps', data=[row['capture_time_s'] for row in rows])
    for side in ('left', 'right'):
        valid = np.ones(len(rows), dtype=bool)
        for suffix, attr in (('hand_keypoints', 'hands'), ('world_landmarks', 'world_hands')):
            values = np.full((len(rows), 21, 3), np.nan, dtype=np.float32)
            for index, raw in enumerate(observations):
                points = (getattr(raw.hands, attr) or {}).get(side)
                if points is not None and np.shape(points) == (21, 3) and np.isfinite(points).all():
                    values[index] = points
                else:
                    valid[index] = False
            handle.create_dataset(f'{side}_{suffix}', data=values)
        handle.create_dataset(f'{side}_valid', data=valid)
    handle.attrs.update(source_landmark_space='mediapipe_normalized',
                        hand_world_landmark_space='mediapipe_world_meters',
                        hand_processing_landmark_space='mediapipe_world_meters',
                        hand_world_origin='hand_geometric_center', length_unit='m')


def _read_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_body_recording(path: str | Path) -> BodyRecording:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SMPL-X body recording was not found: {source}")
    try:
        with h5py.File(source, "r") as h5_file:
            attributes = {key: _read_attr(value) for key, value in h5_file.attrs.items()}
            schema = attributes.get("schema_version")
            if schema != SCHEMA_VERSION:
                for key, value in attributes.items():
                    _reject_legacy(key, context=f"attribute.{key}")
                    _reject_legacy(value, context=f"attribute.{key}")
                raise ValueError(f"Unsupported body recording schema: {schema!r}")
            source_name = attributes.get("source")
            if source_name != "smplerx":
                raise ValueError("Body recording source must be 'smplerx'")
            if attributes.get("time_unit") != "monotonic_s":
                raise ValueError("Body recording time_unit must be 'monotonic_s'")
            metadata = json.loads(attributes.get("metadata_json", "{}"))
            for key in REQUIRED_METADATA:
                if key not in metadata:
                    raise ValueError(f"Body recording metadata requires {key}")
            _metadata(metadata, [])
            for key, value in metadata.items():
                _reject_legacy(value, context=f"metadata.{key}")
            group = h5_file.get("samples")
            if group is None or not group.keys():
                raise ValueError("Body recording contains no samples")
            rows = []
            for name in sorted(group.keys(), key=lambda value: int(value)):
                sample = group[name]
                row = {key: _read_attr(sample.attrs[key]) for key in SCALAR_FIELDS if key in sample.attrs}
                for key in ARRAY_FIELDS:
                    if key in sample:
                        row[key] = np.asarray(sample[key][:])
                if "extra_json" in sample.attrs:
                    row.update(json.loads(_read_attr(sample.attrs["extra_json"])))
                if 'sample_id' not in row or not np.isfinite(float(row.get('capture_time_s', float('nan')))):
                    raise ValueError('Body sample requires sample_id and finite capture_time_s')
                for key, value in row.items():
                    _reject_legacy(key, context=f"sample.{key}")
                    _reject_legacy(value, context=f"sample.{key}")
                rows.append(row)
    except OSError as error:
        raise ValueError("Body recording is not a supported SMPL-X HDF5 file") from error
    return BodyRecording(tuple(rows), {**metadata, "schema_version": schema})


def capture_sample(source, raw):
    """Preserve RGB, independent hand/body times, raw model arrays and output."""
    from teleoperation.inputs.smplerx import encode_packet
    frame = source.last_frame
    wrist_values = lambda values: None if values is None else {side: vars(value) for side, value in values.items()}
    packet = encode_packet('observation', {
        'hands': {**vars(raw.hands), 'wrists': wrist_values(raw.hands.wrists)},
        'upper_body': vars(raw.upper_body), 'wrists': wrist_values(raw.wrists)})
    row = {'sample_id': f'{source.session_id}:{frame.frame_id}',
           'capture_time_s': float(frame.timestamp), 'rgb': frame.rgb,
           'observation_packet': np.frombuffer(packet, dtype=np.uint8)}
    if source.last_result is not None:
        result = source.last_result
        row['completed_time_s'] = result.completed_time_s
        row['worker_packet'] = np.frombuffer(encode_packet('result', {'result': result}), dtype=np.uint8)
    return row


def replay_observation(row):
    from teleoperation.inputs.smplerx import decode_packet
    from teleoperation.contracts.observations import RawHandFrame, UpperBodyObservation, RawTeleopObservation
    from teleoperation.contracts.wrist import WristObservation
    packet = decode_packet(np.asarray(row['observation_packet'], dtype=np.uint8).tobytes())
    if packet['type'] != 'observation':
        raise ValueError('Body recording observation packet has wrong type')
    values = packet['payload']
    wrists = lambda mapping: None if mapping is None else {side: WristObservation(**value) for side, value in mapping.items()}
    hands = values['hands']
    hands['wrists'] = wrists(hands.get('wrists'))
    body = UpperBodyObservation(**values['upper_body'])
    if body.metadata.get('source') != 'smplerx':
        raise ValueError('Body replay requires explicit SMPL-X source')
    return RawTeleopObservation(RawHandFrame(**hands), body, wrists(values.get('wrists')))
