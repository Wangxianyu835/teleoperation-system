"""Immutable, identity-checked arm morphology fitting artifacts."""
from dataclasses import dataclass,field
from pathlib import Path
import json
import numpy as np
from teleoperation.retargeting.arm.body_model import BodyModelSpec,reference_pose
from teleoperation.retargeting.arm.segment_scaling import SegmentScaling


@dataclass(frozen=True)
class ArmFittingProfile:
    body_spec: BodyModelSpec
    body_spec_hash: str
    robot_spec_hash: str
    betas: np.ndarray
    segment_scaling: SegmentScaling
    body_to_robot: np.ndarray
    wrist_rotation_offsets: dict
    pose_params: dict
    initial_rms: float
    final_rms: float
    per_point_error: np.ndarray
    iterations: int
    anchor: str = 'pelvis'
    robot_anchor: np.ndarray = field(default_factory=lambda:np.zeros(3))

    def to_dict(self):
        return dict(schema_version=1,body_spec=self.body_spec.to_dict(),body_spec_hash=self.body_spec_hash,
                    robot_spec_hash=self.robot_spec_hash,betas=self.betas.tolist(),
                    segment_scaling=self.segment_scaling.to_dict(),body_to_robot=self.body_to_robot.tolist(),
                    wrist_rotation_offsets={k:v.tolist() for k,v in self.wrist_rotation_offsets.items()},
                    pose_params={k:np.asarray(v).tolist() for k,v in self.pose_params.items()},
                    initial_rms=self.initial_rms,final_rms=self.final_rms,
                    per_point_error=self.per_point_error.tolist(),iterations=self.iterations,
                    anchor=self.anchor,robot_anchor=self.robot_anchor.tolist())


def save_arm_fitting(path,*,body_spec,robot_spec_hash,fit_result,segment_scaling,
                     body_to_robot=None,pose_params=None,wrist_rotation_offsets=None,
                     anchor='pelvis',robot_anchor=None):
    """Create once; returns a validated profile. Never replaces an existing file."""
    profile=ArmFittingProfile(body_spec,body_spec.hash,robot_spec_hash,np.asarray(fit_result.betas),
             segment_scaling,np.asarray(np.eye(3) if body_to_robot is None else body_to_robot),
             {k:np.asarray(v) for k,v in (wrist_rotation_offsets if wrist_rotation_offsets is not None
                                        else {'left':np.eye(3),'right':np.eye(3)}).items()},
             reference_pose() if pose_params is None else pose_params,
             fit_result.initial_rms,fit_result.final_rms,np.asarray(fit_result.per_point_error),fit_result.iterations,
             anchor,np.asarray(np.zeros(3) if robot_anchor is None else robot_anchor))
    payload=profile.to_dict()
    _validate(payload)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return profile


def _rotation(value,name):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,3) or not np.isfinite(value).all() or not np.allclose(value.T@value,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(value),1.,atol=1e-6):
        raise ValueError(f'{name} must be a proper rotation')


def _validate(value):
    if value.get('schema_version')!=1: raise ValueError('unsupported arm fitting schema')
    betas=np.asarray(value['betas'],dtype=float)
    if betas.shape!=(value['body_spec']['num_betas'],) or not np.isfinite(betas).all():
        raise ValueError('fitting beta dimensions do not match body model')
    for key in ('initial_rms','final_rms','per_point_error'):
        array=np.asarray(value[key],dtype=float)
        if not np.isfinite(array).all() or np.any(array<0): raise ValueError(f'invalid {key}')
    _rotation(value['body_to_robot'],'body_to_robot')
    if set(value['wrist_rotation_offsets'])!={'left','right'}: raise ValueError('both wrist rotation offsets required')
    for side,matrix in value['wrist_rotation_offsets'].items(): _rotation(matrix,f'{side} wrist offset')
    for name,pose in value['pose_params'].items():
        if not np.isfinite(np.asarray(pose,dtype=float)).all(): raise ValueError(f'nonfinite pose {name}')
    if value['iterations']<1: raise ValueError('iterations must be positive')
    scaling=SegmentScaling.from_dict(value['segment_scaling'])
    if value.get('anchor','pelvis')!=scaling.anchor: raise ValueError('profile/scaling anchor mismatch')
    anchor=np.asarray(value.get('robot_anchor',[0,0,0]),dtype=float)
    if anchor.shape!=(3,) or not np.isfinite(anchor).all(): raise ValueError('robot_anchor must be a finite three-vector')


def load_arm_fitting(path,*,body_spec=None,robot_spec_hash=None):
    """Load ArmFittingProfile and optionally enforce expected model/robot identities."""
    with Path(path).open(encoding='utf-8') as stream: value=json.load(stream)
    _validate(value)
    if body_spec is not None and value['body_spec_hash']!=body_spec.hash:
        raise ValueError('body model identity does not match fitting artifact')
    if robot_spec_hash is not None and value['robot_spec_hash']!=robot_spec_hash:
        raise ValueError('robot specification identity does not match fitting artifact')
    return ArmFittingProfile(BodyModelSpec.from_dict(value['body_spec']),value['body_spec_hash'],value['robot_spec_hash'],
        np.asarray(value['betas']),SegmentScaling.from_dict(value['segment_scaling']),np.asarray(value['body_to_robot']),
        {k:np.asarray(v) for k,v in value['wrist_rotation_offsets'].items()},value['pose_params'],
        value['initial_rms'],value['final_rms'],np.asarray(value['per_point_error']),value['iterations'],
        value.get('anchor','pelvis'),np.asarray(value.get('robot_anchor',[0,0,0]),dtype=float))
