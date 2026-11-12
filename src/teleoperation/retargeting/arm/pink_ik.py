"""Transactional, bounded differential PINK IK; no simulator side effects."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np

class PinkIKError(RuntimeError):
    """The QP backend could not produce a finite, bounded proposal."""

@dataclass(frozen=True)
class PinkIKCandidate:
    q: np.ndarray
    joint_values: object
    diagnostics: object
    _owner: object
    _generation: int

class _HoldJoints:
    """Hard equality on inactive tangent coordinates (zero velocity).

    PINK VelocityLimit deliberately excludes zero bounds; equality constraints
    preserve the inactive arm exactly without a conflicting soft posture task.
    """
    gain=1.0
    def __init__(self, model, indices):
        self.projection=np.eye(model.nv)[indices]
    def compute_jacobian(self, configuration):
        return self.projection
    def compute_error(self, configuration):
        return np.zeros(len(self.projection))

class PinkArmIK:
    def __init__(self, spec, initial=None, solver='quadprog'):
        import pinocchio as pin
        import pink
        self.spec=spec; self.solver=solver
        self._owner=object(); self._generation=0
        full=spec._model
        qfull=pin.neutral(full)
        for name,value in spec.reference.items(): qfull[full.joints[full.getJointId(name)].idx_q]=value
        enabled={n for names in spec.arm_joint_names.values() for n in names}
        shared=set(spec.task_config.shared_joint_names)
        if shared & enabled or any(not full.existJointName(n) for n in shared):
            raise ValueError('Shared joints must exist in the URDF and be separate from arm joints')
        enabled.update(shared)
        locked=[jid for jid in range(1,full.njoints) if full.names[jid] not in enabled]
        self.model=pin.buildReducedModel(full,locked,qfull)
        self._q=pin.neutral(self.model)
        for name in self.model.names[1:]:
            j=self.model.joints[self.model.getJointId(name)]
            self._q[j.idx_q]=spec.reference[name]
        if not np.all(np.isfinite(self.model.velocityLimit)) or np.any(self.model.velocityLimit<=0):
            raise ValueError('Enabled joints require positive finite URDF velocity bounds')
        self._reference=self._q.copy()
        self.reset(initial or {})

    @property
    def q(self):
        return self._q.copy()

    def _validate_q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid reduced-model configuration')
        if np.any(q<self.model.lowerPositionLimit-1e-9) or np.any(q>self.model.upperPositionLimit+1e-9):
            raise ValueError('Joint configuration exceeds URDF bounds')
        return q

    def reset(self, named_positions):
        """Synchronize only explicitly named joints after an external receipt."""
        q=self._q.copy()
        for name,value in named_positions.items():
            if not self.model.existJointName(name):
                raise ValueError(f'Cannot reset unknown or locked joint: {name}')
            j=self.model.joints[self.model.getJointId(name)]
            q[j.idx_q]=float(value)
        self._validate_q(q)
        self._q=q; self._generation+=1

    def joint_values(self,q=None):
        q=self._q if q is None else self._validate_q(q)
        return {side:{name:float(q[self.model.joints[self.model.getJointId(name)].idx_q])
                      for name in names} for side,names in self.spec.arm_joint_names.items()}

    def measure(self,q=None):
        import pinocchio as pin
        q=self._q if q is None else self._validate_q(q)
        data=self.model.createData()
        pin.framesForwardKinematics(self.model,data,q)
        return {side:data.oMf[self.model.getFrameId(frame)].homogeneous.copy() @ self.spec.wrist_offsets[side]
                for side,frame in self.spec.end_effector_frames.items()}

    def prepare(self,targets,dt,active_sides=None):
        import pinocchio as pin
        import pink
        from pink.tasks import FrameTask, PostureTask
        dt=float(dt)
        if not np.isfinite(dt) or not 0<dt<=.02:
            raise ValueError('IK dt must be finite and within (0, 0.02] seconds')
        if set(targets)-set(self.spec.arm_joint_names): raise ValueError('Target side not enabled in ArmSpec')
        checked={}
        for side,target in targets.items():
            target=np.asarray(target,dtype=float)
            if target.shape!=(4,4) or not np.all(np.isfinite(target)) or not np.allclose(target[3],[0,0,0,1],atol=1e-8):
                raise ValueError('Wrist target must be a finite homogeneous 4x4 transform')
            rotation=target[:3,:3]
            if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1,atol=1e-6):
                raise ValueError('Wrist target rotation must belong to SO(3)')
            checked[side]=target.copy()
        active=tuple(checked) if active_sides is None else tuple(active_sides)
        if len(set(active))!=len(active) or set(active)-set(self.spec.arm_joint_names):
            raise ValueError('Invalid active arm sides')
        active=tuple(s for s in active if s in checked)
        active_names={n for s in active for n in self.spec.arm_joint_names[s]}
        # Shared joints are disabled when fewer than all enabled sides are active.
        if set(active)=={'left','right'}: active_names.update(self.spec.task_config.shared_joint_names)
        held=[self.model.joints[jid].idx_v for jid in range(1,self.model.njoints)
              if self.model.names[jid] not in active_names]
        if not active:
            q=self._q.copy()
        else:
            cfg=pink.Configuration(self.model,self.model.createData(),self._q.copy())
            params=self.spec.task_config
            tasks=[]
            for side in active:
                task=FrameTask(self.spec.end_effector_frames[side],position_cost=params.position_cost,
                               orientation_cost=params.orientation_cost,lm_damping=params.lm_damping,gain=params.gain)
                target=checked[side]@np.linalg.inv(self.spec.wrist_offsets[side])
                task.set_target(pin.SE3(target[:3,:3],target[:3,3]))
                tasks.append(task)
            posture=PostureTask(cost=params.posture_cost,gain=params.gain)
            posture.set_target(self._reference)
            tasks.append(posture)
            constraints=[_HoldJoints(self.model,held)] if held else []
            try:
                velocity=pink.solve_ik(cfg,tasks,dt,solver=self.solver,constraints=constraints)
                if not np.all(np.isfinite(velocity)) or np.any(np.abs(velocity)>self.model.velocityLimit+1e-7):
                    raise ValueError('QP velocity violates URDF limits')
                q=pin.integrate(self.model,self._q,velocity*dt)
                # Remove only equality-solver roundoff from held scalar joints.
                for jid in range(1,self.model.njoints):
                    j=self.model.joints[jid]
                    if j.idx_v in held: q[j.idx_q]=self._q[j.idx_q]
                self._validate_q(q)
            except Exception as exc:
                raise PinkIKError(f'{self.spec.robot_id}: PINK IK failed: {exc}') from exc
        measured=self.measure(q)
        errors={side:{'position_error_m':float(np.linalg.norm(measured[side][:3,3]-checked[side][:3,3])),
                      'orientation_error_rad':float(np.linalg.norm(pin.log3(measured[side][:3,:3].T@checked[side][:3,:3])))}
                for side in active}
        diagnostics={'solver':self.solver,'active_sides':active,'dt':dt,'errors':errors,
                     'max_velocity_rad_s':float(np.max(np.abs(q-self._q))/dt)}
        q.setflags(write=False)
        values=MappingProxyType({s:MappingProxyType(v) for s,v in self.joint_values(q).items()})
        return PinkIKCandidate(q,values,MappingProxyType(diagnostics),self._owner,self._generation)

    def commit(self,candidate):
        """Commit only after the caller has received accepted output."""
        if not isinstance(candidate,PinkIKCandidate) or candidate._owner is not self._owner or candidate._generation!=self._generation:
            raise ValueError('Foreign or stale PINK candidate')
        self._q=self._validate_q(candidate.q).copy()
        self._generation+=1
