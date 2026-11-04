"""Explicit whole-robot identities and ordered, independently enabled parts."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from teleoperation.contracts.robot_control import RobotCommandFrame, RobotPartCommand
from teleoperation.contracts.constants import ROBOT_TARGETS
from teleoperation.paths import ROBOTS_ROOT, DEFAULT_TRON2A_URDF
from .specification import ROBOT_SPECS, TRON2A_ARM_JOINTS, _is_arm_joint



def resolve_robot_targets(robot=None,robots=None):
    if (robot is None)==(robots is None): raise ValueError('Select --robot or --robots explicitly')
    targets=(robot,) if robot is not None else tuple(robots)
    if (robots is not None and len(targets)<2) or len(set(targets))!=len(targets):
        raise ValueError('--robots requires at least two distinct targets')
    if not targets or any(n not in ROBOT_TARGETS for n in targets): raise ValueError('Unknown robot target')
    return targets


def enabled_parts(parts='both',side='both'):
    if parts not in ('both','arms','hands') or side not in ('both','left','right'):
        raise ValueError('Invalid parts/side')
    return tuple(s+'_'+p for s in (('left','right') if side=='both' else (side,))
                 for p in ('arm','hand') if parts=='both' or parts==p+'s')


@dataclass(frozen=True)
class PartSpec:
    joint_names: tuple
    limits: tuple
    reference: tuple


@dataclass(frozen=True)
class RobotControlSpec:
    robot_id: str
    hand_model: str
    urdf_path: Path | None
    parts: object
    hand_specs: object
    spec_hash: str
    display: str
    side: str

    def reference_frame(self,frame_id,timestamp):
        return RobotCommandFrame(self.robot_id,self.spec_hash,frame_id,timestamp,
            {key:RobotPartCommand(p.joint_names,p.reference,None,'none',False,'initialized') for key,p in self.parts.items()})

    def validate_frame(self,frame):
        if not isinstance(frame,RobotCommandFrame) or frame.robot_id!=self.robot_id or frame.spec_hash!=self.spec_hash or set(frame.parts)!=set(self.parts):
            raise ValueError('Robot command identity or parts differ from specification')
        for key,p in self.parts.items():
            part=frame.parts[key]
            if part.joint_names!=p.joint_names: raise ValueError('Part joint order differs from specification')
            limits=np.asarray(p.limits)
            if np.any(part.positions<limits[:,0]-1e-6) or np.any(part.positions>limits[:,1]+1e-6):
                raise ValueError('Part exceeds joint limits')
            if key.endswith('_hand'):
                hand=self.hand_specs[key[:key.index('_')]]
                named=dict(zip(part.joint_names,part.positions))
                expected=hand.expand([named[n] for n in hand.independent_joint_names])
                if any(abs(named[n]-expected[n])>1e-6 for n in named): raise ValueError('Hand coupling mismatch')


def load_robot_control_spec(robot_id,*,resource_overrides=None):
    if robot_id not in ROBOT_TARGETS: raise ValueError('Unknown robot target')
    options=dict(resource_overrides or {})
    display,side=options.get('parts','both'),options.get('side','both')
    selected=enabled_parts(display,side)
    hand_model='l21' if robot_id=='tron2a_l21' else robot_id
    path=Path(options.get('urdf_path') or (DEFAULT_TRON2A_URDF if robot_id=='tron2a_l21' else ROBOTS_ROOT/ROBOT_SPECS[robot_id]['urdf']))
    root=None
    if any(k.endswith('_arm') for k in selected): root=ET.parse(path).getroot()
    parts,hands,identity={},{},{}
    for key in selected:
        s,kind=key.split('_')
        if kind=='hand':
            from .hand_spec import load_hand_spec
            hand=load_hand_spec(hand_model,s); hands[s]=hand
            q=hand.expand(hand.q_reference)
            parts[key]=PartSpec(hand.joint_names,hand.joint_limits,tuple(q[n] for n in hand.joint_names))
            identity[key]=hand.spec_hash
        else:
            joints={j.get('name'):j for j in root.findall('joint')}
            names=TRON2A_ARM_JOINTS[s] if robot_id=='tron2a_l21' else tuple(n for n,j in joints.items() if j.get('type')!='fixed' and _is_arm_joint(n,s))
            if len(names)!=7: raise ValueError(f'{robot_id}/{s} must have seven arm joints')
            limits=tuple((float(joints[n].find('limit').get('lower')),float(joints[n].find('limit').get('upper'))) for n in names)
            parts[key]=PartSpec(tuple(names),limits,tuple(float(np.clip(0,*b)) for b in limits))
            identity[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    value={'robot_id':robot_id,'parts':{k:{'names':v.joint_names,'limits':v.limits,'asset':identity[k]} for k,v in parts.items()}}
    digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return RobotControlSpec(robot_id,hand_model,path if root is not None else None,
                           MappingProxyType(parts),MappingProxyType(hands),digest,display,side)
