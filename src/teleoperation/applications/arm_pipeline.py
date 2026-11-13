"""One observation update, many bounded control updates, receipt-driven commits."""
from dataclasses import dataclass
import numpy as np

from teleoperation.retargeting.arm.pink_ik import PinkArmIK
from teleoperation.retargeting.arm.body_model import smplerx_pose_in_model_frame


BODY_INDICES = {'pelvis': 0, 'head': 15, 'left_shoulder': 16,
                'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
                'left_wrist': 20, 'right_wrist': 21}


@dataclass(frozen=True)
class ArmUpdate:
    candidate: object
    source_sequence: int
    source_time: float
    expires_time: float
    source_is_new: bool
    active_sides: tuple


class ArmPipeline:
    def __init__(self, specs, profiles, body_model, *, initial=None,
                 max_age=.3, max_dt=.02, max_catchup=.1, solver='quadprog', enabled_sides=None):
        if set(specs) != set(profiles) or not specs:
            raise ValueError('Each selected robot requires a fitting profile')
        if getattr(body_model.spec, 'model_type', 'smplx') != 'smplx':
            raise ValueError('SMPLer-X camera arms require an SMPL-X (smplx) fitting and reconstruction model')
        if any(spec.task_config.shared_joint_names for spec in specs.values()):
            raise ValueError('Shared joints require a supported output command contract; keep shared_joint_names empty')
        if not np.isfinite([max_age, max_dt, max_catchup]).all() or min(max_age,max_dt,max_catchup)<=0:
            raise ValueError('Target age and integration bounds must be finite and positive')
        if max_dt > .02:
            raise ValueError('PINK integration steps must not exceed 20 ms')
        self.specs, self.profiles, self.body_model = dict(specs), dict(profiles), body_model
        selected = {name:tuple(spec.arm_joint_names) for name,spec in specs.items()} if enabled_sides is None else {
            name:tuple(sides) for name,sides in enabled_sides.items()}
        if set(selected)!=set(specs) or any(len(set(sides))!=len(sides) or
                set(sides)-set(specs[name].arm_joint_names) for name,sides in selected.items()):
            raise ValueError('Enabled arm sides must match the selected robot specifications')
        self.enabled_sides = selected
        self.solvers = {name: PinkArmIK(spec, initial=(initial or {}).get(name), solver=solver)
                        for name, spec in specs.items()}
        self.max_age,self.max_dt,self.max_catchup = float(max_age),float(max_dt),float(max_catchup)
        self.targets = {}
        self._session = None
        self._sequence = -1
        self._source_time = None
        self._expires = None
        self._valid = {}
        self._committed_sequences = {name: None for name in specs}
        self._pending = {}
        self.last_errors = {}
        self.reconstruction_count = 0
        body_hash = getattr(body_model.spec,'hash',None)
        for name, profile in profiles.items():
            if profile.robot_spec_hash != specs[name].spec_hash:
                raise ValueError(f'Fitting robot specification mismatch: {name}')
            if body_hash is not None and getattr(profile,'body_spec_hash',body_hash) != body_hash:
                raise ValueError('Fitting body specification differs from reconstruction model')
            if np.asarray(profile.betas).shape != (body_model.spec.num_betas,):
                raise ValueError('Fitting shape dimension differs from reconstruction model')

    def ingest(self, result, *, now_s, control_time=None):
        if result is None:
            return False
        stamp = result.frame_id.capture_time_s
        if not np.isfinite(now_s) or not 0 <= now_s-stamp <= self.max_age:
            return False
        session = result.frame_id.session_id
        if self._session is not None and session != self._session:
            # A new operator/session requires an explicit reset, never an implicit jump.
            raise ValueError('Body session changed; reset the arm pipeline explicitly')
        if result.frame_id.sequence <= self._sequence:
            return False
        if self._source_time is not None and stamp <= self._source_time:
            return False
        # Reconstruct and map before accepting source metadata: a failed input
        # must never refresh the expiry or identity of the previous geometry.
        generated = {}
        reconstructed = False
        if any(result.valid.values()):
            for key in ('smplx_root_pose','smplx_body_pose'):
                if key not in result.arrays:
                    raise ValueError(f'Body result lacks raw pose parameter {key}; re-record body input')
            names = tuple(self.specs)
            betas = np.stack([self.profiles[name].betas for name in names])
            joints, rotations = self.body_model.reconstruct(smplerx_pose_in_model_frame(result.arrays), betas)
            for index,name in enumerate(names):
                spec, profile = self.specs[name], self.profiles[name]
                transform = np.asarray(profile.body_to_robot)
                available = {'pelvis', *profile.segment_scaling.scales}
                anchor = ((joints[index,16]+joints[index,17])/2
                          if getattr(profile,'anchor','pelvis')=='shoulder_center' else joints[index,0])
                points = {key: transform @ (joints[index,j]-anchor) for key,j in BODY_INDICES.items()
                          if key in available}
                scaled = profile.segment_scaling.apply(points)
                generated[name] = {}
                for side in spec.arm_joint_names:
                    target = np.eye(4)
                    target[:3,3] = scaled[side+'_wrist']+getattr(profile,'robot_anchor',np.zeros(3))
                    offset = getattr(profile,'wrist_rotation_offsets',{}).get(side,np.eye(3))
                    target[:3,:3] = transform @ rotations[index,BODY_INDICES[side+'_wrist']] @ offset
                    generated[name][side] = target
            reconstructed = True
        self.targets = generated
        self._session = session
        self._sequence = result.frame_id.sequence
        self._source_time = stamp
        self._expires = (now_s if control_time is None else control_time)+self.max_age-(now_s-stamp)
        self._valid = dict(result.valid)
        self.reconstruction_count += int(reconstructed)
        return True

    def prepare(self, control_time, *, dt):
        if not np.isfinite([control_time,dt]).all() or dt <= 0:
            raise ValueError('Control time and dt must be finite; dt must be positive')
        self.discard()
        if dt > self.max_catchup:
            self.last_errors = {name:'control_gap_hold' for name in self.specs}
            return {}
        if self._expires is None or control_time > self._expires:
            self.last_errors = {name:'target_expired' for name in self.specs}
            return {}
        # Each call represents one bounded physical control step. The session owns substeps.
        if dt > self.max_dt+1e-12:
            raise ValueError('Split elapsed control time into bounded substeps before prepare')
        self.last_errors = {}
        for name,ik in self.solvers.items():
            active = tuple(side for side in self.enabled_sides[name] if self._valid.get(side,False))
            if not active or name not in self.targets:
                continue
            try:
                candidate = ik.prepare(self.targets[name],dt,active_sides=active)
            except (RuntimeError,ValueError) as error:
                self.last_errors[name] = str(error)
                continue
            self._pending[name] = ArmUpdate(candidate,self._sequence,self._source_time,
                self._expires,self._committed_sequences[name]!=self._sequence,active)
        return dict(self._pending)

    def commit(self, names=None):
        for name in tuple(self._pending) if names is None else tuple(names):
            update = self._pending.pop(name)
            self.solvers[name].commit(update.candidate)
            self._committed_sequences[name] = update.source_sequence

    def discard(self):
        self._pending.clear()

    def reset(self, actual_named):
        for name,ik in self.solvers.items():
            ik.reset(actual_named[name])
        self.targets.clear()
        self._session = None
        self._sequence = -1
        self._source_time = self._expires = None
        self._valid = {}
        self._committed_sequences = {name:None for name in self.specs}
        self.discard()
