# 函数 resolve_device（查询/选择）；链路：手部点→拓扑/坐标/时序处理→角度/模型结果
"""Shared device selection for training, export and offline diagnostics."""

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
