"""Accepted whole-robot commands and actual dispatch receipts, without inference."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

import h5py
import numpy as np

from teleoperation.contracts.robot_control import RobotCommandFrame,RobotPartCommand
from teleoperation.robots.control_spec import load_robot_control_spec,enabled_parts
from teleoperation.simulation.robot_output import OutputReceipt

SCHEMA='robot_command_results_v1'
TEXT=h5py.string_dtype('utf-8')


def _text(value):return value.decode('utf-8') if isinstance(value,bytes) else str(value)


def _receipt_data(receipt):
    return dict(frame_id=int(receipt.frame_id),timestamp=float(receipt.timestamp),
                statuses=dict(receipt.statuses),steps=dict(receipt.steps),physics_dt=dict(receipt.physics_dt))


def _check_receipt(receipt,targets,frame_id,timestamp,*,complete=False):
    if isinstance(receipt.frame_id,bool) or not isinstance(receipt.frame_id,(int,np.integer)) or not np.isfinite(receipt.timestamp):
        raise ValueError('Invalid receipt frame/time')
    if receipt.frame_id!=frame_id or receipt.timestamp!=timestamp or set(receipt.statuses)!=set(targets):
        raise ValueError('Receipt identity differs from command frame')
    if not receipt.steps or set(receipt.steps)!=set(receipt.physics_dt):raise ValueError('Missing physics receipt')
    for key,steps in receipt.steps.items():
        if isinstance(steps,bool) or not isinstance(steps,(int,np.integer)) or steps<0:
            raise ValueError('Invalid recorded physics steps')
        if not np.isfinite(receipt.physics_dt[key]) or receipt.physics_dt[key]<=0:
            raise ValueError('Invalid recorded physics dt')
    if any(s not in ('submitted','not_submitted') for s in receipt.statuses.values()):
        raise ValueError('Invalid target dispatch status')
    if complete and (any(s!='submitted' for s in receipt.statuses.values()) or any(s<1 for s in receipt.steps.values())):
        raise ValueError('Recording contains incomplete dispatch')


class RobotCommandWriter:
    """Publish by an exclusive hard link only after a successful close.

    Failed/interrupted files retain their unique pending name and failed marker.
    Static validation failures leave the previous accepted recording row intact.
    """
    def __init__(self,path,specs,metadata):
        self.path=Path(path);self.specs=dict(specs)
        if not self.specs or any(key!=s.robot_id for key,s in self.specs.items()):
            raise ValueError('Explicit recording targets required')
        if self.path.exists():raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temporary_path=self.path.with_name(self.path.name+'.pending-'+uuid.uuid4().hex+'.h5')
        self.count=0;self.last_time=self.last_frame=None;self.closed=self.failed=False
        self.stream=h5py.File(self.temporary_path,'x');self.datasets=[];self.groups={}
        try:
            self.stream.attrs.update(schema=SCHEMA,angle_unit='rad',timestamp_unit='control_relative_seconds',
                recording_status='recording',targets_json=json.dumps(list(specs)),metadata_json=json.dumps(metadata))
            self._dataset(self.stream,'frame_ids',dtype='int64');self._dataset(self.stream,'control_timestamps')
            execution=self.stream.create_group('execution');self._dataset(execution,'receipts',dtype=TEXT)
            robots=self.stream.create_group('robots')
            for key,spec in specs.items():
                group=robots.create_group(key)
                asset=spec.urdf_path
                group.attrs.update(spec_hash=spec.spec_hash,parts=spec.display,side=spec.side,
                    enabled_parts=json.dumps(list(spec.parts)),urdf_path='' if asset is None else str(asset),
                    urdf_sha256='' if asset is None else hashlib.sha256(asset.read_bytes()).hexdigest())
                parts=group.create_group('parts')
                for name,part in spec.parts.items():
                    child=parts.create_group(name);self.groups[key,name]=child
                    hand=spec.hand_specs.get(name.split('_')[0]) if name.endswith('_hand') else None
                    child.attrs.update(joint_names=json.dumps(part.joint_names),limits=json.dumps(part.limits),
                        mimic_rules=json.dumps([] if hand is None else hand.mimic_rules),
                        hand_spec_hash='' if hand is None else hand.spec_hash,
                        hand_urdf_path='' if hand is None else str(hand.urdf_path))
                    self._dataset(child,'positions',(len(part.joint_names),))
                    self._dataset(child,'source_timestamp')
                    self._dataset(child,'observed',dtype='bool')
                    for field in ('source_timebase','status'):self._dataset(child,field,dtype=TEXT)
        except BaseException:
            self.abort('initialization failed');self.stream.close();self.closed=True
            raise

    def _dataset(self,parent,name,tail=(),dtype='float64'):
        dataset=parent.create_dataset(name,shape=(0,*tail),maxshape=(None,*tail),chunks=True,dtype=dtype)
        self.datasets.append(dataset)

    def append(self,batch,receipt):
        if self.closed or self.failed:raise RuntimeError('Recording is closed or failed')
        if set(batch)!=set(self.specs):raise ValueError('Recording batch must cover exact targets')
        identities={(frame.frame_id,frame.timestamp) for frame in batch.values()}
        if len(identities)!=1:raise ValueError('Recording batch has differing frame/time')
        frame_id,timestamp=next(iter(identities))
        if self.last_time is not None and (timestamp<=self.last_time or frame_id<=self.last_frame):
            raise ValueError('Recorded control time/frame must strictly increase')
        for key,spec in self.specs.items():spec.validate_frame(batch[key])
        _check_receipt(receipt,self.specs,frame_id,timestamp)
        encoded=json.dumps(_receipt_data(receipt))
        try:
            for dataset in self.datasets:dataset.resize(self.count+1,axis=0)
            self.stream['frame_ids'][self.count]=frame_id
            self.stream['control_timestamps'][self.count]=timestamp
            self.stream['execution/receipts'][self.count]=encoded
            for key,frame in batch.items():
                for name,part in frame.parts.items():
                    group=self.groups[key,name]
                    values=dict(positions=part.positions,source_timestamp=np.nan if part.source_timestamp is None else part.source_timestamp,
                        source_timebase=part.source_timebase,observed=part.observed,status=part.status)
                    for field,value in values.items():group[field][self.count]=value
            self.stream.flush()
        except BaseException:
            self.abort('row write failed');raise
        self.count+=1;self.last_frame=frame_id;self.last_time=timestamp
        if any(s!='submitted' for s in receipt.statuses.values()) or any(s<1 for s in receipt.steps.values()):
            self.abort('dispatch incomplete')

    def abort(self,reason):
        if not self.closed:
            self.failed=True;self.stream.attrs.update(recording_status='failed',failure_reason=str(reason));self.stream.flush()

    def close(self):
        if self.closed:return
        if not self.failed:self.stream.attrs['recording_status']='complete'
        self.stream.flush();self.stream.close();self.closed=True
        if self.failed:return
        try:
            os.link(self.temporary_path,self.path)  # Atomic, same volume, refuses overwrite.
        except BaseException:
            with h5py.File(self.temporary_path,'r+') as stream:
                stream.attrs.update(recording_status='failed',failure_reason='exclusive publication failed')
            raise
        self.temporary_path.unlink()

    def __enter__(self):return self
    def __exit__(self,kind,error,traceback):
        if error is not None:self.abort(f'{kind.__name__}: {error}')
        self.close()


def _read_row(stream,index,targets):
    frame_id=int(stream['frame_ids'][index]);timestamp=float(stream['control_timestamps'][index]);batch={}
    for key in targets:
        group=stream['robots'][key];parts={}
        for name,part in group['parts'].items():
            source=float(part['source_timestamp'][index])
            parts[name]=RobotPartCommand(tuple(json.loads(part.attrs['joint_names'])),part['positions'][index],
                None if np.isnan(source) else source,_text(part['source_timebase'][index]),
                bool(part['observed'][index]),_text(part['status'][index]))
            q=parts[name].positions;limits=np.asarray(json.loads(part.attrs['limits']))
            if limits.shape!=(len(q),2) or not np.isfinite(limits).all() or np.any(limits[:,0]>limits[:,1]):
                raise ValueError('Invalid recorded joint limits')
            if np.any(q<limits[:,0]-1e-6) or np.any(q>limits[:,1]+1e-6):raise ValueError('Recorded joint limits exceeded')
            named=dict(zip(parts[name].joint_names,q))
            for child,parent,multiplier,offset in json.loads(part.attrs['mimic_rules']):
                if abs(named[child]-(named[parent]*multiplier+offset))>1e-6:raise ValueError('Recorded hand coupling mismatch')
        batch[key]=RobotCommandFrame(key,_text(group.attrs['spec_hash']),frame_id,timestamp,parts)
    data=json.loads(_text(stream['execution/receipts'][index]))
    receipt=OutputReceipt(**data)
    return batch,receipt


@dataclass(frozen=True)
class RobotRecording:
    path: Path
    specs: dict
    metadata: dict
    frame_count: int

    def iter_frames(self):
        with h5py.File(self.path,'r') as stream:
            if stream.attrs.get('recording_status')!='complete':raise ValueError('Recording is not complete')
            for index in range(self.frame_count):yield _read_row(stream,index,self.specs)


def load_robot_recording(path,targets):
    """Validate every stored command before the caller creates a simulation.

    Only selected targets require local robot resources. Other stored targets
    are checked against their recorded layout, limits and coupling metadata.
    """
    targets=tuple(targets)
    if not targets or len(set(targets))!=len(targets):raise ValueError('Distinct replay targets required')
    with h5py.File(path,'r') as stream:
        try:
            if (stream.attrs.get('schema')!=SCHEMA or stream.attrs.get('recording_status')!='complete'
                or stream.attrs.get('angle_unit')!='rad' or stream.attrs.get('timestamp_unit')!='control_relative_seconds'):
                raise ValueError('Unsupported or incomplete robot recording')
            stored=json.loads(stream.attrs['targets_json'])
            if not stored or len(set(stored))!=len(stored) or set(stored)!=set(stream['robots']):raise ValueError('Invalid stored targets')
            if not set(targets)<=set(stored):raise ValueError('Requested robot is absent from recording')
            specs={};count=len(stream['frame_ids'])
            if stream['frame_ids'].dtype.kind not in 'iu' or stream['frame_ids'].shape!=(count,) or stream['control_timestamps'].shape!=(count,):
                raise ValueError('Invalid recording frame/time layout')
            if stream['execution/receipts'].shape!=(count,):raise ValueError('Receipt length differs from commands')
            for key in stored:
                group=stream['robots'][key]
                expected=set(enabled_parts(group.attrs['parts'],group.attrs['side']))
                if set(json.loads(group.attrs['enabled_parts']))!=expected or set(group['parts'])!=expected:
                    raise ValueError('Recorded parts differ from scene selection')
                for part in group['parts'].values():
                    names=json.loads(part.attrs['joint_names'])
                    for field in ('positions','source_timestamp','source_timebase','observed','status'):
                        expected_shape=(count,len(names)) if field=='positions' else (count,)
                        if part[field].shape!=expected_shape:raise ValueError('Incomplete command datasets')
                    if part['observed'].dtype.kind!='b':raise ValueError('Invalid observed flags')
                if key in targets:
                    options={'parts':group.attrs['parts'],'side':group.attrs['side']}
                    if key=='tron2a_l21' and group.attrs['urdf_path']:options['urdf_path']=group.attrs['urdf_path']
                    spec=load_robot_control_spec(key,resource_overrides=options)
                    if spec.spec_hash!=group.attrs['spec_hash']:raise ValueError('Replay robot specification differs')
                    if spec.urdf_path is not None and hashlib.sha256(spec.urdf_path.read_bytes()).hexdigest()!=group.attrs['urdf_sha256']:
                        raise ValueError('Recorded URDF identity differs')
                    specs[key]=spec
            previous=None;cadence=None
            for index in range(count):
                batch,receipt=_read_row(stream,index,stored)
                frame=next(iter(batch.values()))
                if previous is not None and (frame.frame_id<=previous.frame_id or frame.timestamp<=previous.timestamp):
                    raise ValueError('Recorded control time/frame must strictly increase')
                for key,spec in specs.items():spec.validate_frame(batch[key])
                _check_receipt(receipt,stored,frame.frame_id,frame.timestamp,complete=True)
                if len(set(receipt.steps.values()))!=1 or len(set(receipt.physics_dt.values()))!=1:
                    raise ValueError('Replay requires one common recorded physics cadence')
                current=dict(receipt.physics_dt)
                if cadence is not None and cadence!=current:raise ValueError('Recorded physics cadence changed')
                cadence=current
                previous=frame
            metadata=json.loads(stream.attrs['metadata_json'])
        except (KeyError,TypeError,IndexError) as error:raise ValueError('Malformed robot recording') from error
    return RobotRecording(Path(path),{key:specs[key] for key in targets},metadata,count)


def iter_robot_commands(path,specs):
    recording=load_robot_recording(path,specs)
    for key,spec in specs.items():
        if spec.spec_hash!=recording.specs[key].spec_hash:raise ValueError('Replay specification mismatch')
    for batch,_ in recording.iter_frames():yield batch
