"""Compare methods on the same source, preprocessing and common valid times."""
import json

import h5py
import numpy as np
import torch

from teleoperation.data.hand_joint_recording import iter_joint_commands
from teleoperation.retargeting.hand.hand_model import DifferentiableHandModel


def _read(path,specs):
    # Reuse the command identity, bounds and coupling validation used by replay.
    for _ in iter_joint_commands(path,specs):
        pass
    with h5py.File(path,'r') as source:
        return (json.loads(source.attrs['metadata_json']),np.asarray(source['timestamps']),
                {key:{name:np.asarray(source['/'.join(key)][name]) for name in
                      ('raw_positions','command_positions','valid','target_vectors','solve_ms')} for key in specs})


def _errors(spec, q, target):
    fk=DifferentiableHandModel(spec)
    with torch.no_grad():
        points=np.concatenate([fk.vectors(torch.tensor(q[i:i+512],dtype=torch.float32)).numpy()
                               for i in range(0,len(q),512)])
    errors=np.linalg.norm(points-target,axis=-1)
    pinch=np.abs(np.linalg.norm(points[:,0]-points[:,1],axis=-1)-np.linalg.norm(target[:,0]-target[:,1],axis=-1))
    bounds=spec.bounds
    return {'tip_error_normalized_mean':float(errors.mean()),
            'tip_error_m_mean':float(errors.mean()*spec.reference_palm_length_m),
            'tip_error_m_p95':float(np.percentile(errors,95)*spec.reference_palm_length_m),
            'pinch_distance_error_normalized_mean':float(pinch.mean()),
            'pinch_distance_error_m_mean':float(pinch.mean()*spec.reference_palm_length_m),
            'limit_violations':int(np.sum(np.any((q<bounds[:,0]-1e-6)|(q>bounds[:,1]+1e-6),axis=1)))}


def compare_hand_methods(nn_path,dex_path,specs):
    a,ta,ra=_read(nn_path,specs)
    b,tb,rb=_read(dex_path,specs)
    if a.get('backend') != 'nn' or b.get('backend') != 'dex':
        raise ValueError('Supply NN results first and dex results second')
    for name in ('input_sha256','input_profile','command_policy'):
        if name not in a or a[name] != b.get(name):
            raise ValueError(f'Method comparison requires the same {name}')
    common,ia,ib=np.intersect1d(ta,tb,return_indices=True)
    report={'aligned_timestamps':len(common),'nn_frames':len(ta),'dex_frames':len(tb),
            'input_sha256':a['input_sha256'],'input_profile':a['input_profile'],
            'cross_operator_generalization_verified':False,
            'scope':'Recorded numerical comparison; invalid_frames includes window warmup and missing input.',
            'targets':{}}
    for key,spec in specs.items():
        valid=ra[key]['valid'][ia]&rb[key]['valid'][ib]
        left,right=ia[valid],ib[valid]
        if not len(left):
            raise ValueError(f'No common valid frames for {key}')
        if not np.allclose(ra[key]['target_vectors'][left],rb[key]['target_vectors'][right],atol=1e-7,rtol=0,equal_nan=False):
            raise ValueError('Methods did not use the same finite normalized targets')
        row={'common_valid_frames':len(left)}
        for method,times,data,indices in (('nn',ta,ra[key],left),('dex',tb,rb[key],right)):
            target=data['target_vectors'][indices]
            mapping={name:i for i,name in enumerate(spec.joint_names)}
            command=data['command_positions'][:,[mapping[n] for n in spec.independent_joint_names]]
            raw_metrics=_errors(spec,data['raw_positions'][indices],target)
            command_metrics=_errors(spec,command[indices],target)
            speeds=np.abs(np.diff(command,axis=0))/np.diff(times)[:,None]
            limits=np.asarray(a['command_policy']['max_velocity_rad_s']['/'.join(key)])
            command_metrics['speed_violations']=int(np.sum(np.any(speeds>limits+1e-5,axis=1)))
            valid_ms=data['solve_ms'][data['valid']]
            row[method]={'invalid_frames':int((~data['valid']).sum()),
                         'invalid_rate_including_warmup':float((~data['valid']).mean()),
                         'solve_ms_mean':float(valid_ms.mean()),'solve_ms_p95':float(np.percentile(valid_ms,95)),
                         'raw':raw_metrics,'command':command_metrics}
        report['targets']['/'.join(key)]=row
    return report
