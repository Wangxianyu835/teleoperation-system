import argparse
from pathlib import Path


def inspect_angle_h5(path):
    import h5py
    import numpy as np
    if not Path(path).is_file():
        raise FileNotFoundError(f'Angle H5 was not found: {path}')
    with h5py.File(path,'r') as stream:
        if stream.attrs.get('schema') == 'hand_joint_results_v1':
            from teleoperation.robots.hand_spec import load_hand_spec
            from teleoperation.data.hand_joint_recording import iter_joint_commands
            specs = {(name,side):load_hand_spec(name,side) for name in stream
                     if isinstance(stream[name],h5py.Group) for side in stream[name]}
            for _ in iter_joint_commands(path,specs):
                pass
            targets = {}
            for key,spec in specs.items():
                group = stream['/'.join(key)]
                valid,raw = np.asarray(group['valid']), np.asarray(group['raw_positions'])
                if raw.shape != (len(valid),spec.dof) or not np.isfinite(raw[valid]).all():
                    raise ValueError('Invalid raw joint result shape or nonfinite valid values')
                targets['/'.join(key)] = {'raw_shape':list(raw.shape),'command_shape':list(group['command_positions'].shape),
                                         'valid':int(valid.sum()),'invalid':int((~valid).sum()),'spec_hash':spec.spec_hash}
            return {'schema':'hand_joint_results_v1','path':str(path),'frames':len(stream['timestamps']),'targets':targets}
    raise ValueError('Retired angle recording: expected hand_joint_results_v1; record/export new data')


def run(args: argparse.Namespace) -> int:
    summary = inspect_angle_h5(args.angle_h5)
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
