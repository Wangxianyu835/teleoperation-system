"""Persistence for the existing NumPy observation recordings."""
import numpy as np


def read_recording(path): return np.load(path, allow_pickle=True)


def read_frame(frame):
    if isinstance(frame, dict): return frame
    if hasattr(frame, "item"):
        value = frame.item()
        if isinstance(value, dict): return value
    raise TypeError(f"Expected frame dict in .npy log, got {type(frame)!r}")
