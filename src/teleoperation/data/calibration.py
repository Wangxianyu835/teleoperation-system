"""Calibration JSON persistence."""
import json
import os
from pathlib import Path
import tempfile


def read_calibration(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write_calibration(path, data, *, overwrite=False):
    """Publish complete, strict JSON atomically; existing files are protected.

    Serialize before touching the destination. A same-directory temporary file
    is synced and published with a non-overwriting hard link (or atomic replace
    when explicitly requested). Failed writes never expose partial JSON.
    """
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination = Path(path)
    if not overwrite and destination.exists():
        raise FileExistsError(f"Calibration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
