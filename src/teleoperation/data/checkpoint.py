"""Checkpoint serialization without interpretation or model construction."""
def read_checkpoint(path, map_location):
    import torch
    return torch.load(path, map_location=map_location)
