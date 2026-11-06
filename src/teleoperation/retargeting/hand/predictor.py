"""Two-hand retargeting wrapper built on the existing PoseTransformer."""

from __future__ import annotations

from typing import Any

import torch.nn as nn



def build_hand_model(model_kwargs: dict[str, Any]) -> nn.Module:
    """Create one hand PoseTransformer with the existing model implementation."""
    from teleoperation.retargeting.hand.transformer import PoseTransformer

    return PoseTransformer(**model_kwargs)


def build_joint_model(spec, architecture=None) -> nn.Module:
    """Create a native D-coordinate head from the target's effective limits."""
    from .config import L21Config
    kwargs = L21Config().model_kwargs()
    kwargs.update(architecture or {})
    kwargs.update(num_frame=3, in_num_joints=25, in_chans=3,
                  out_num_joint=spec.dof, out_chans=1,
                  angle_limit_rad=spec.bounds.tolist())
    model = build_hand_model(kwargs)
    model.model_kwargs = kwargs
    return model
