"""NumPy observation recordings; loading does not process coordinates."""
import numpy as np


def read_recording(path):
    return np.load(path, allow_pickle=True)


def read_arrays(path):
    with np.load(path, allow_pickle=True) as file:
        return {key: file[key] for key in file.files}
