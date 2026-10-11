# 函数 read_arrays（读取/解析）；write_arrays（保存/导出）；链路：应用/训练→文件读写→观测/命令数据
"""NPZ observation arrays, without coordinate processing."""
import numpy as np


def read_arrays(path):
    with np.load(path, allow_pickle=True) as file:
        return {key: file[key] for key in file.files}


def write_arrays(path, **arrays):
    np.savez(path, **arrays)
