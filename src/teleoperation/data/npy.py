"""Persistence for the existing NumPy observation recordings."""
import numpy as np


def read_recording(path): return np.load(path, allow_pickle=True)
