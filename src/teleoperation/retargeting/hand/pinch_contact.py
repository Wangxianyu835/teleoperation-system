"""Find a target-hand pose whose thumb and selected fingertip actually meet."""
from __future__ import annotations

import numpy as np


_CACHE = {}
_FINGER_TIPS = {"thumb": 0, "index": 1, "middle": 2}


def independent_pinch_pose(spec, finger="index", *, steps=350):
    """Return bounded independent joints minimizing thumb-to-finger distance.

    Dex optimizes all target fingertips together.  On the H1-2 and GR1-T2
    geometries that objective can leave the thumb and index more than 40 mm
    apart even when the operator's fingertips touch.  This function is used
    only for an explicit detected pinch, adding a target-space contact
    constraint while the object remains governed by ordinary physics.
    """
    if finger not in _FINGER_TIPS or finger == "thumb":
        raise ValueError("Pinch contact finger must be index or middle")
    if not np.isfinite(float(steps)) or int(steps) < 1:
        raise ValueError("Pinch optimization steps must be positive")
    key = (spec.hand_model, spec.side, spec.spec_hash, finger)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached.copy()

    import torch
    from teleoperation.retargeting.hand.hand_model import (
        DifferentiableHandModel,
    )

    model = DifferentiableHandModel(spec)
    bounds = spec.bounds
    low = torch.as_tensor(bounds[:, 0], dtype=torch.float64)
    high = torch.as_tensor(bounds[:, 1], dtype=torch.float64)
    middle = (low + high) / 2.0
    reference = torch.as_tensor(spec.q_reference, dtype=torch.float64)
    seeds = torch.stack((
        reference,
        middle,
        low,
        high,
        0.3 * low + 0.7 * high,
    ))
    unit = ((seeds - low) / (high - low) * 2.0 - 1.0).clamp(-0.999, 0.999)
    raw = torch.nn.Parameter(torch.atanh(unit))
    optimizer = torch.optim.Adam((raw,), lr=0.05)
    target_index = _FINGER_TIPS[finger]
    frames = (spec.root_link, spec.tip_frames[0],
              spec.tip_frames[target_index])

    for _ in range(int(steps)):
        optimizer.zero_grad()
        positions = low + (torch.tanh(raw) + 1.0) / 2.0 * (high - low)
        tips = model.positions(positions, frames)
        squared_gap = ((tips[:, 1] - tips[:, 2]) ** 2).sum(dim=-1)
        squared_gap.sum().backward()
        optimizer.step()

    with torch.no_grad():
        positions = low + (torch.tanh(raw) + 1.0) / 2.0 * (high - low)
        tips = model.positions(positions, frames)
        gaps = torch.linalg.norm(tips[:, 1] - tips[:, 2], dim=-1)
        best = int(torch.argmin(gaps))
        result = positions[best].detach().cpu().numpy()
        gap_m = float(gaps[best])
    if gap_m > 0.001:
        raise RuntimeError(
            "Could not find a sub-millimetre %s pinch pose for %s/%s "
            "(best %.3f mm)" % (
                finger, spec.hand_model, spec.side, gap_m * 1000.0))
    result.setflags(write=False)
    _CACHE[key] = result
    return result.copy()
