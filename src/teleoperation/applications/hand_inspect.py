import argparse
from pathlib import Path
from teleoperation.contracts.constants import HAND_SIDES
from teleoperation.data.hand_h5 import read_angle_arrays
from teleoperation.retargeting.hand.diagnostics import summarize_angles


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
    arrays, attrs = read_angle_arrays(path)
    return summarize_angles(path, arrays, attrs)

def run(args: argparse.Namespace) -> int:
    summary = inspect_angle_h5(args.angle_h5)
    if summary.get('schema') == 'hand_joint_results_v1':
        import json
        print(json.dumps(summary,indent=2,ensure_ascii=False))
        return 0
    print(f"path={summary['path']}")
    print(f"frames={summary['frames']}")
    for side in HAND_SIDES:
        values = summary[side]
        print(f"{side}_shape={values['shape']}")
        print(f"{side}_valid={values['valid']}")
        print(f"{side}_invalid={values['invalid']}")
        print(f"{side}_nonfinite={values['nonfinite']}")
        print(f"{side}_out_of_limits={values['out_of_limits']}")
    for name, value in sorted(summary["attrs"].items()):
        print(f"attr.{name}={value}")
    return 0
