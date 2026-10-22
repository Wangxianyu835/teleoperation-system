"""Explicitly labelled offline input; does not modify legacy replay files."""
import json
from pathlib import Path

import numpy as np

def sample_seconds(timestamps, unit):
    """Normalize an explicitly declared timeline without modifying its source."""
    if unit not in ("seconds", "milliseconds"):
        raise ValueError("Specify timestamp unit: seconds or milliseconds")
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if not len(values):
        return values.copy()
    return (values - values[0]) / (1000 if unit == "milliseconds" else 1)


def load_validation_recording(path, timestamp_unit=None):
    if Path(path).suffix.lower() != ".npz":
        raise ValueError("Arm validation requires NPZ with shoulder/elbow/wrist arrays; hand-only H5 is not arm input")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.array(archive[key], copy=True) for key in archive.files}
    return validate_recording(data, timestamp_unit)


def validate_recording(data, timestamp_unit=None):
    required = ("timestamps", "left_arm", "right_arm")
    if any(key not in data for key in required):
        raise ValueError("Arm input requires timestamps, left_arm and right_arm (shoulder/elbow/wrist)")
    metadata = json.loads(str(np.asarray(data.get("metadata_json", "{}")).item()))
    if not isinstance(metadata, dict):
        raise ValueError("Recording metadata must be an object")
    if metadata.get("length_unit", "m") != "m":
        raise ValueError("Arm recordings must use world metres")
    if metadata.get("point_order", ["shoulder", "elbow", "wrist"]) != ["shoulder", "elbow", "wrist"]:
        raise ValueError("Arm point order must be shoulder/elbow/wrist")
    declared = metadata.get("timestamp_unit")
    if declared is not None and timestamp_unit is not None and declared != timestamp_unit:
        raise ValueError("Explicit timestamp unit conflicts with recording metadata")
    unit = timestamp_unit or declared
    seconds = sample_seconds(data["timestamps"], unit)
    if len(seconds) < 2:
        raise ValueError("Arm validation requires at least two frames")
    for side in ("left", "right"):
        if np.shape(data[side + "_arm"]) != (len(seconds), 3, 3):
            raise ValueError(f"{side}_arm must have shape (T,3,3)")
        for key in (side + "_hand", "world_" + side + "_hand"):
            if key in data and np.shape(data[key]) != (len(seconds), 21, 3):
                raise ValueError(f"{key} must have shape (T,21,3)")
    scenarios = np.asarray(data.get("scenario", ["imported"] * len(seconds)))
    if scenarios.shape != (len(seconds),) or scenarios.dtype.kind not in "US":
        raise ValueError("scenario must contain T text labels")
    return {"arrays": data, "seconds": seconds, "timestamp_unit": unit,
            "scenarios": scenarios.astype(str), "metadata": metadata}


def save_validation_recording(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    return path
