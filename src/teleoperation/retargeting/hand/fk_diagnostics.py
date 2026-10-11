import numpy as np
import torch
from .angles import angle18_to_nodes

def fk_positions(fk, angles):
    nodes = angle18_to_nodes(angles)
    with torch.no_grad():
        points = fk.forward(torch.from_numpy(nodes[None]))[2][0].cpu().numpy()
    if points.shape != (23, 3) or not np.isfinite(points).all():
        raise ValueError("L21 FK must produce finite (23,3) points")
    return points

