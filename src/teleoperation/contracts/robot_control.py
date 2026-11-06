"""Immutable accepted robot commands, independent of devices and simulation."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import re
import numpy as np

PARTS = ('left_arm','right_arm','left_hand','right_hand')


@dataclass(frozen=True)
class RobotPartCommand:
    joint_names: tuple[str, ...]
    positions: np.ndarray
    source_timestamp: float | None
    source_timebase: str
    observed: bool
    status: str

    def __post_init__(self):
        names=tuple(self.joint_names)
        q=np.array(self.positions,dtype=float,copy=True)
        if not names or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names):
            raise ValueError('Part requires unique joint names')
        if q.shape!=(len(names),) or not np.isfinite(q).all():
            raise ValueError('Part requires finite ordered joint positions in radians')
        if self.source_timestamp is not None and not np.isfinite(self.source_timestamp):
            raise ValueError('Invalid source timestamp')
        if not isinstance(self.source_timebase,str) or not self.source_timebase:
            raise ValueError('Source timebase must be explicit')
        if not isinstance(self.observed,(bool,np.bool_)) or self.status not in ('tracking','holding','initialized'):
            raise ValueError('Invalid part observation status')
        if self.observed != (self.status=='tracking') or (self.observed and self.source_timestamp is None):
            raise ValueError('Tracking requires a source timestamp; held parts are not new observations')
        q.setflags(write=False)
        object.__setattr__(self,'joint_names',names)
        object.__setattr__(self,'positions',q)


@dataclass(frozen=True)
class RobotCommandFrame:
    robot_id: str
    spec_hash: str
    frame_id: int
    timestamp: float
    parts: Mapping[str,RobotPartCommand]

    def __post_init__(self):
        if not isinstance(self.robot_id,str) or not re.fullmatch(r'[a-z0-9_]+',self.robot_id):
            raise ValueError('Invalid robot identity')
        if not isinstance(self.spec_hash,str) or not re.fullmatch('[0-9a-f]{64}',self.spec_hash):
            raise ValueError('Invalid robot specification hash')
        if isinstance(self.frame_id,bool) or not isinstance(self.frame_id,(int,np.integer)) or self.frame_id<0:
            raise ValueError('frame_id must be a nonnegative integer')
        if not np.isfinite(self.timestamp) or self.timestamp<0:
            raise ValueError('Control timestamp must be finite nonnegative seconds')
        parts=dict(self.parts)
        if not parts or set(parts)-set(PARTS) or any(not isinstance(p,RobotPartCommand) for p in parts.values()):
            raise ValueError('Invalid robot command parts')
        # L21 uses identical left/right names in separate bodies; namespaces are per-part.
        object.__setattr__(self,'parts',MappingProxyType(parts))
