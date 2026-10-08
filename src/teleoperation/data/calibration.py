"""Calibration JSON persistence."""
import json
from pathlib import Path


def read_calibration(path): return json.loads(Path(path).read_text(encoding="utf-8"))
