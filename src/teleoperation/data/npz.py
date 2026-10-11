"""NPZ observation arrays, without coordinate processing."""
import numpy as np


def read_arrays(path):
    with np.load(path, allow_pickle=True) as file:
        return {key: file[key] for key in file.files}


def write_arrays(path, **arrays):
    np.savez(path, **arrays)
