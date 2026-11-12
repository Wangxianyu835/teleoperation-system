"""Declarative arm geometry shared by every camera retargeting robot.

Only kinematics is loaded: URDF visual/collision meshes are never requested.
Coordinates and wrist targets are expressed in the URDF root link frame.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from teleoperation.paths import ROBOTS_ROOT
from .specification import ARM_NEUTRAL, ROBOT_SPECS, TRON2A_ARM_JOINTS

# Joint order is a model declaration, never inferred by a fixed-DOF solver.
_ROBOT_ARMS = {
    'h1_2': ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_roll','wrist_pitch','wrist_yaw'),
    'gr1_t2': ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch'),
    'g1': ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'),
}
# H1/G1 forearms point forward at elbow zero; a geometric T-pose needs +pi/2.
_ANATOMICAL_ANCHOR = {'h1_2':'pelvis','gr1_t2':'pelvis','g1':'pelvis','tron2a_l21':'shoulder_center'}
_TPOSE_ELBOW = {'h1_2': np.pi/2, 'gr1_t2': 0., 'g1': np.pi/2, 'tron2a_l21': 0.}
_END_FRAMES = {'h1_2':'{side}_wrist_yaw_link','gr1_t2':'{side}_hand_pitch_link',
               'g1':'{side}_wrist_yaw_link','tron2a_l21':'grasper_{letter}_Link'}

@dataclass(frozen=True)
class ArmTaskConfig:
    position_cost: float = 1.0
    orientation_cost: float = 0.5
    lm_damping: float = 1e-4
    posture_cost: float = 1e-3
    gain: float = 1.0
    elbow_task: bool = False
    shared_joint_names: tuple = ()  # torso/head locked unless explicitly declared

    def __post_init__(self):
        for name in ('position_cost', 'orientation_cost', 'lm_damping', 'posture_cost'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or value<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.position_cost==0 and self.orientation_cost==0:
            raise ValueError('At least one wrist task cost must be positive')
        if not np.isscalar(self.gain) or not np.isfinite(self.gain) or not 0<self.gain<=1:
            raise ValueError('Task gain must be finite and within (0, 1]')
        if self.elbow_task is not False:
            raise ValueError('Elbow tasks are unsupported; elbow_task must be False')
        shared=tuple(self.shared_joint_names)
        if any(not isinstance(n,str) or not n for n in shared) or len(set(shared))!=len(shared):
            raise ValueError('Shared joints must have distinct nonempty names')
        object.__setattr__(self,'shared_joint_names',shared)

@dataclass(frozen=True)
class ArmSpec:
    robot_id: str
    urdf_path: Path
    base_link: str
    anatomical_anchor: str
    reference_up_axis: tuple | None
    arm_joint_names: object
    end_effector_frames: object
    wrist_offsets: object
    reference: object
    tpose: object
    spec_hash: str
    task_config: ArmTaskConfig
    landmark_frames: object
    _model: object

    @property
    def wrist_frames(self):
        return self.end_effector_frames

    def _forward_named(self, named_positions):
        import pinocchio as pin
        model=self._model
        q=pin.neutral(model)
        for name,value in named_positions.items():
            joint=model.joints[model.getJointId(name)]
            q[joint.idx_q]=value
        data=model.createData()
        pin.framesForwardKinematics(model,data,q)
        return data

    def wrist_tpose_rotations(self):
        """Return physical wrist rotations at the geometric T-pose in base axes.

        Includes each declared wrist offset and never changes an IK seed.
        """
        data=self._forward_named(self.tpose)
        return {side:(data.oMf[self._model.getFrameId(frame)].homogeneous @
                      self.wrist_offsets[side])[:3,:3].copy()
                for side,frame in self.end_effector_frames.items()}

    def landmark_positions(self, tpose=True):
        model=self._model
        data=self._forward_named(self.tpose if tpose else self.reference)
        result={'pelvis':np.zeros(3)}
        for key,frame in self.landmark_frames.items():
            result[key]=data.oMf[model.getFrameId(frame)].translation.copy()
        return result


def load_arm_spec(robot_id, urdf_path=None, sides=('left','right')):
    import pinocchio as pin
    if robot_id not in (*_ROBOT_ARMS,'tron2a_l21'):
        raise ValueError(f'Unknown arm robot: {robot_id}')
    sides=tuple(sides)
    if not sides or len(set(sides))!=len(sides) or set(sides)-{'left','right'}:
        raise ValueError('Select distinct left/right arm sides')
    path=Path(urdf_path or (ROBOTS_ROOT/'tron2a/robot.urdf' if robot_id=='tron2a_l21'
                          else ROBOTS_ROOT/ROBOT_SPECS[robot_id]['urdf'])).resolve()
    root=ET.parse(path).getroot()
    model=pin.buildModelFromUrdf(str(path))
    joints={j.get('name'):j for j in root.findall('joint')}
    children={j.find('child').get('link') for j in joints.values()}
    base=next(link.get('name') for link in root.findall('link') if link.get('name') not in children)
    names={s:tuple(TRON2A_ARM_JOINTS[s]) if robot_id=='tron2a_l21' else
           tuple(f'{s}_{stem}_joint' for stem in _ROBOT_ARMS[robot_id]) for s in sides}
    reference={}
    for jid in range(1,model.njoints):
        j=model.joints[jid]; name=model.names[jid]
        if j.nq!=1 or j.nv!=1:
            raise ValueError(f'URDF joint {name} must be scalar')
        value=ARM_NEUTRAL.get(name,0.)
        if robot_id=='tron2a_l21':
            if name.startswith('proximal_pitch_'): value=-.3
            if name.startswith('elbow_'): value=-1.2
        reference[name]=float(np.clip(value,model.lowerPositionLimit[j.idx_q],model.upperPositionLimit[j.idx_q]))
    tpose=reference.copy(); frames={}; landmarks={}; offsets={}
    for side,ordered in names.items():
        for name in ordered:
            if name not in reference: raise ValueError(f'Missing declared arm joint: {name}')
            tpose[name]=0.
        # URDF shoulder roll is positive outward on left and negative on right.
        tpose[ordered[1]]=np.pi/2*(1 if side=='left' else -1)
        elbow=next(n for n in ordered if 'elbow' in n)
        tpose[elbow]=_TPOSE_ELBOW[robot_id]
        for name in ordered:
            j=model.joints[model.getJointId(name)]
            tpose[name]=float(np.clip(tpose[name],model.lowerPositionLimit[j.idx_q],model.upperPositionLimit[j.idx_q]))
        ee=_END_FRAMES[robot_id].format(side=side,letter='L' if side=='left' else 'R')
        if not model.existFrame(ee): raise ValueError(f'Missing declared wrist frame: {ee}')
        frames[side]=ee
        # Use the shoulder roll origin: the pitch axis is inside the torso.
        landmarks[side+'_shoulder']=joints[ordered[1]].find('child').get('link')
        elbow=next(n for n in ordered if 'elbow' in n)
        landmarks[side+'_elbow']=joints[elbow].find('child').get('link')
        landmarks[side+'_wrist']=ee
        offset=np.eye(4); offset.setflags(write=False); offsets[side]=offset
    config=ArmTaskConfig()
    identity={'robot_id':robot_id,'anatomical_anchor':_ANATOMICAL_ANCHOR[robot_id],
              'reference_up_axis':(0.,0.,1.) if robot_id=='tron2a_l21' else None,'urdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'sides':sides,'joints':names,'frames':frames,'reference':reference,'tpose':tpose,
              'tasks':vars(config),'wrist_offsets':{s:o.tolist() for s,o in offsets.items()}}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    freeze=MappingProxyType
    return ArmSpec(robot_id,path,base,_ANATOMICAL_ANCHOR[robot_id],
                   (0.,0.,1.) if robot_id=='tron2a_l21' else None,freeze(names),freeze(frames),freeze(offsets),
                   freeze(reference),freeze(tpose),digest,config,freeze(landmarks),model)
