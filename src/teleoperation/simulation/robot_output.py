"""Validate complete control frames before actuation and advance each world once."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
import numpy as np


class RobotBackend(Protocol):
    spec: object
    physics_group: object
    def validate(self,frame): ...
    def write(self,frame): ...
    def read_state(self): ...
    def close(self): ...


@dataclass(frozen=True)
class OutputReceipt:
    frame_id: int
    timestamp: float
    statuses: object
    steps: object
    physics_dt: object


class RobotOutput:
    def __init__(self,backends):
        self.backends=dict(backends)
        if not self.backends or any(k!=v.spec.robot_id for k,v in self.backends.items()):
            raise ValueError('One backend is required for each explicit target')
        self.groups={v.physics_group.group_id:v.physics_group for v in self.backends.values()}
        self.last_time=self.last_frame=None
        self.last_receipt=None
        self.closed=False

    def submit(self,batch,*,steps):
        if self.closed: raise RuntimeError('Output is closed')
        if isinstance(steps,bool) or not isinstance(steps,(int,np.integer)) or steps<1:
            raise ValueError('Physics steps must be a positive integer')
        if set(batch)!=set(self.backends): raise ValueError('Output batch must cover selected targets exactly')
        identities={(f.frame_id,f.timestamp) for f in batch.values()}
        if len(identities)!=1: raise ValueError('Targets must share control frame and time')
        frame_id,timestamp=next(iter(identities))
        if self.last_time is not None and (timestamp<=self.last_time or frame_id<=self.last_frame):
            raise ValueError('Output control frames must strictly increase')
        for key,backend in self.backends.items():backend.validate(batch[key])
        statuses={k:'not_submitted' for k in batch}
        advanced={k:0 for k in self.groups}
        try:
            for key,backend in self.backends.items():
                backend.write(batch[key]); statuses[key]='submitted'
            for key,group in self.groups.items():
                group.advance(steps); advanced[key]=int(steps)
        except BaseException:
            self.last_receipt=OutputReceipt(frame_id,timestamp,MappingProxyType(statuses),
                MappingProxyType(advanced),MappingProxyType({k:g.time_step for k,g in self.groups.items()}))
            self.close()
            raise
        self.last_time,self.last_frame=timestamp,frame_id
        self.last_receipt=OutputReceipt(frame_id,timestamp,MappingProxyType(statuses),
            MappingProxyType(advanced),MappingProxyType({k:g.time_step for k,g in self.groups.items()}))
        return self.last_receipt

    def read_state(self):return {key:b.read_state() for key,b in self.backends.items()}

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            for backend in self.backends.values():backend.close()
        finally:
            for group in self.groups.values():group.close()
