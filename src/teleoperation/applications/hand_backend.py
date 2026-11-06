"""Construct target-native hand backends independently of the input device."""
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


def build_hand_retargeter_from_specs(specs, backend, *, checkpoints=None, device='cpu'):
    """Construct native solvers once; source/profile validation belongs to callers."""
    specs = dict(specs)
    checkpoints = dict(checkpoints or {})
    if backend not in ('nn', 'dex'):
        raise ValueError('Hand backend must be nn or dex')
    if any(key != spec.key for key, spec in specs.items()):
        raise ValueError('Hand specification keys differ from target identities')
    selected = {}
    for (model, side), spec in specs.items():
        selected.setdefault(model, []).append(side)
    if backend == 'dex' and checkpoints:
        raise ValueError('Dex does not use checkpoints or operator calibration')
    if backend == 'nn' and set(checkpoints) != set(selected):
        raise ValueError('NN requires exactly one shared checkpoint for every selected hand model')
    if not specs:
        return CameraRetargeters({}, ())
    if backend == 'dex':
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        sides = next(iter(selected.values()))
        if all(value == sides for value in selected.values()):
            return NormalizedDexRetargeter(tuple(selected), tuple(sides))
        return CameraRetargeters(specs, [NormalizedDexRetargeter((name,), tuple(sides))
                                         for name, sides in selected.items()])
    for path in checkpoints.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f'Hand checkpoint not found: {path}')
    from teleoperation.retargeting.hand.checkpoint import load_shared_joint_checkpoint
    from teleoperation.retargeting.hand.device import resolve_device
    from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
    device = resolve_device(device)
    retargeters = []
    for name in selected:
        canonical_spec = load_hand_spec(name, 'right')
        model, _ = load_shared_joint_checkpoint(checkpoints[name], canonical_spec, device)
        per_specs = {key: spec for key, spec in specs.items() if key[0] == name}
        retargeters.append(SharedJointNNRetargeter(per_specs, model, device,
                                                 canonical_spec=canonical_spec))
    return retargeters[0] if len(retargeters) == 1 else CameraRetargeters(specs, retargeters)


def build_camera_retargeter(hand_models, sides, backend, checkpoints=None, device='cpu'):
    """Validate native shared weights before allocating sensors or simulation."""
    hand_models, sides = tuple(hand_models), tuple(sides)
    if not hand_models or len(set(hand_models)) != len(hand_models):
        raise ValueError('Select unique target hand models')
    if not sides or len(set(sides)) != len(sides) or set(sides) - {'left', 'right'}:
        raise ValueError('Select unique left/right target sides')
    specs = {(name, side): load_hand_spec(name, side)
             for name in hand_models for side in sides}
    return specs, build_hand_retargeter_from_specs(specs, backend, checkpoints=checkpoints, device=device)


def build_target_hand_retargeters(targets, backend, checkpoints=None, device='cpu'):
    """Infer exact enabled model/side pairs from whole-robot specifications."""
    specs = {hand.key: hand for target in targets for hand in target.hand_specs.values()}
    return specs, build_hand_retargeter_from_specs(specs, backend, checkpoints=checkpoints, device=device)
