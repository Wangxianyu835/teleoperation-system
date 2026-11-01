"""Versioned hand results; native independent solutions and accepted commands."""
import json
from pathlib import Path

import h5py
import numpy as np

from teleoperation.contracts.hand_control import HandJointCommand, HandJointResult


def _check_command(command, spec):
    if not isinstance(command, HandJointCommand) or command.key != spec.key or command.spec_hash != spec.spec_hash or command.joint_names != spec.joint_names:
        raise ValueError('Command target identity differs from recording specification')
    named = command.named_positions()
    expanded = spec.expand([named[n] for n in spec.independent_joint_names])
    if any(abs(named[n]-expanded[n]) > 1e-6 for n in spec.joint_names):
        raise ValueError('Recorded command violates native joint coupling')


class HandJointResultWriter:
    def __init__(self, path, specs, metadata):
        self.specs = dict(specs)
        if not self.specs or any(k != s.key for k,s in self.specs.items()):
            raise ValueError('Recording requires explicit target specifications')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = h5py.File(path,'x')
        self.count, self.last_time, self.datasets = 0, None, []
        try:
            self.stream.attrs.update(schema='hand_joint_results_v1',timestamp_unit='seconds',
                                     angle_unit='rad',metadata_json=json.dumps(metadata),recording_status='recording')
            self._dataset(self.stream,'timestamps')
            self._dataset(self.stream,'frame_ids',dtype='int64')
            self.groups = {}
            for key,spec in self.specs.items():
                group = self.stream.create_group('/'.join(key))
                self.groups[key] = group
                group.attrs.update(hand_model=spec.hand_model,side=spec.side,spec_hash=spec.spec_hash,
                                   independent_joint_names=json.dumps(spec.independent_joint_names),
                                   command_joint_names=json.dumps(spec.joint_names),
                                   reference_palm_length_m=spec.reference_palm_length_m,
                                   raw_coordinates='independent_native',command_coordinates='expanded_native')
                self._dataset(group,'raw_positions',(spec.dof,))
                self._dataset(group,'command_positions',(len(spec.joint_names),))
                self._dataset(group,'target_vectors',(len(spec.tip_frames),3))
                self._dataset(group,'target_positions_native_m',(len(spec.tip_frames),3))
                for name in ('solve_ms','palm_length_m'):
                    self._dataset(group,name)
                self._dataset(group,'valid',dtype='bool')
                self._dataset(group,'error',dtype=h5py.string_dtype('utf-8'))
        except BaseException:
            self.stream.close()
            raise

    def _dataset(self, parent, name, tail=(), dtype='float64'):
        dataset = parent.create_dataset(name,shape=(0,*tail),maxshape=(None,*tail),chunks=(64,*tail),dtype=dtype)
        self.datasets.append(dataset)

    def append(self, frame_id, timestamp, results, commands, targets=None, palm_lengths=None):
        if not np.isfinite(timestamp) or self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Result timestamps must strictly increase in seconds')
        if set(results) != set(self.specs) or set(commands) != set(self.specs):
            raise ValueError('Frame must cover every configured target')
        rows = {}
        for key,spec in self.specs.items():
            result, command = results[key], commands[key]
            if not isinstance(result,HandJointResult) or result.key != key or result.spec_hash != spec.spec_hash or result.joint_names != spec.independent_joint_names or result.timestamp != timestamp:
                raise ValueError('Result identity/time differs from recording specification')
            _check_command(command,spec)
            if command.timestamp != timestamp:
                raise ValueError('Command timestamp differs from recorded frame')
            target = (targets or {}).get(key)
            nan_targets = np.full((len(spec.tip_frames),3),np.nan)
            rows[key] = dict(raw_positions=result.positions if result.valid else np.full(spec.dof,np.nan),
                             command_positions=command.positions,valid=result.valid,error=result.error or '',solve_ms=result.solve_ms,
                             target_vectors=nan_targets if target is None or target.vectors is None else target.vectors,
                             target_positions_native_m=nan_targets if target is None or target.positions_native_m is None else target.positions_native_m,
                             palm_length_m=(palm_lengths or {}).get(spec.side) or np.nan)
        for dataset in self.datasets:
            dataset.resize(self.count+1,axis=0)
        self.stream['timestamps'][self.count] = timestamp
        self.stream['frame_ids'][self.count] = frame_id
        for key,row in rows.items():
            for name,value in row.items():
                self.groups[key][name][self.count] = value
        self.last_time, self.count = timestamp, self.count+1

    def close(self, success=True):
        if self.stream is not None:
            self.stream.attrs.update(frames=self.count,recording_status='complete' if success else 'interrupted')
            self.stream.close()
            self.stream = None

    def __enter__(self): return self

    def __exit__(self, kind, *args): self.close(success=kind is None)


def iter_joint_commands(path, specs):
    """Replay only explicitly matched target assets, ordered joints and coupling."""
    with h5py.File(path,'r') as stream:
        if stream.attrs.get('schema') != 'hand_joint_results_v1' or stream.attrs.get('timestamp_unit') != 'seconds' or stream.attrs.get('angle_unit') != 'rad':
            raise ValueError('Not a hand joint command recording')
        times = np.asarray(stream['timestamps'])
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid command timebase')
        actual = {(name,side) for name in stream if isinstance(stream[name],h5py.Group) for side in stream[name]}
        if actual != set(specs):
            raise ValueError('Replay target selection differs from recording')
        for key,spec in specs.items():
            group = stream['/'.join(key)]
            if group.attrs.get('spec_hash') != spec.spec_hash or tuple(json.loads(group.attrs['command_joint_names'])) != spec.joint_names or tuple(json.loads(group.attrs['independent_joint_names'])) != spec.independent_joint_names:
                raise ValueError('Replay target specification differs from recording')
            if group['command_positions'].shape != (len(times),len(spec.joint_names)):
                raise ValueError('Replay command shape mismatch')
        for index,timestamp in enumerate(times):
            commands = {}
            for key,spec in specs.items():
                command = HandJointCommand(*key,spec.spec_hash,spec.joint_names,
                                           np.asarray(stream['/'.join(key)]['command_positions'][index]),float(timestamp))
                _check_command(command,spec)
                commands[key] = command
            yield float(timestamp), commands
