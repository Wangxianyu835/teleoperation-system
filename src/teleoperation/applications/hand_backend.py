"""Construct camera backends with explicit native targets and one input profile."""
from pathlib import Path

from teleoperation.robots.hand_spec import load_hand_spec


class CameraRetargeters:
    """Apply independently configured target models to the same human window."""

    def __init__(self, specs, retargeters):
        self.specs = dict(specs)
        self.retargeters = tuple(retargeters)

    def reset(self):
        for retargeter in self.retargeters:
            retargeter.reset()

    def retarget(self, window):
        results = {}
        for retargeter in self.retargeters:
            results.update(retargeter.retarget(window))
        if set(results) != set(self.specs):
            raise ValueError('Camera backend returned different target identities')
        return results


def build_camera_retargeter(hand_models, sides, backend, checkpoints=None, device='cpu'):
    """Validate native shared weights before allocating sensors or simulation."""
    hand_models, sides = tuple(hand_models), tuple(sides)
    if not hand_models or len(set(hand_models)) != len(hand_models):
        raise ValueError('Select unique target hand models')
    if not sides or len(set(sides)) != len(sides) or set(sides) - {'left', 'right'}:
        raise ValueError('Select unique left/right target sides')
    specs = {(name, side): load_hand_spec(name, side)
             for name in hand_models for side in sides}
    checkpoints = dict(checkpoints or {})
    if backend == 'dex':
        if checkpoints:
            raise ValueError('Camera dex does not use checkpoints or operator calibration')
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        return specs, NormalizedDexRetargeter(hand_models, sides)
    if backend != 'nn':
        raise ValueError('Camera hand backend must be nn or dex')
    if set(checkpoints) != set(hand_models):
        raise ValueError('NN requires exactly one shared checkpoint for every selected hand model')
    for path in checkpoints.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f'Hand checkpoint not found: {path}')
    from teleoperation.retargeting.hand.checkpoint import load_shared_joint_checkpoint
    from teleoperation.retargeting.hand.device import resolve_device
    from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
    device = resolve_device(device)
    retargeters = []
    for name in hand_models:
        canonical_spec = load_hand_spec(name, 'right')
        model, _ = load_shared_joint_checkpoint(checkpoints[name], canonical_spec, device)
        selected = {key: spec for key, spec in specs.items() if key[0] == name}
        retargeters.append(SharedJointNNRetargeter(selected, model, device,
                                                 canonical_spec=canonical_spec))
    return specs, CameraRetargeters(specs, retargeters)
