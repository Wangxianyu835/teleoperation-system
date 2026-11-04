"""Adapt existing accepted commands to the common actuator boundary.

Legacy capture, solver tuning, HUD and result files stay with their callers.
This adapter does not retarget, filter or acquire another observation.
"""
from dataclasses import replace
from types import SimpleNamespace

from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_backends import SimulationGroup, NativeRobotBackend
from teleoperation.simulation.robot_output import RobotOutput
from .robot_commands import RobotCommandAssembler


class AcceptedRobotOutput:
    def __init__(self,backends):
        self.specs={key:b.spec for key,b in backends.items()}
        self.output=RobotOutput(backends)
        actual=self.output.read_state()
        initial={}
        for key,spec in self.specs.items():
            frame=spec.reference_frame(0,0.)
            initial[key]=replace(frame,parts={name:replace(part,positions=actual[key][name])
                                               for name,part in frame.parts.items()})
        self.assembler=RobotCommandAssembler(self.specs,initial)

    @classmethod
    def native(cls,scene,robots,side,driver=None):
        group=SimulationGroup(scene,owns_scene=False,advance_frame=None if driver is None else driver.step)
        specs={name:load_robot_control_spec(name,resource_overrides={
            'parts':'hands' if driver is None else 'both','side':side}) for name in robots}
        return cls({name:NativeRobotBackend(spec,scene,group,driver) for name,spec in specs.items()})

    @classmethod
    def tron(cls,world,urdf,with_hands,initial_arm):
        # The borrowed world keeps ownership and its reference-hand update.
        scene=SimpleNamespace(cid=world.client,
                              step=lambda: [world.step() for _ in range(scene.steps_per_frame)])
        group=SimulationGroup(scene,owns_scene=False)
        spec=load_robot_control_spec('tron2a_l21',resource_overrides={
            'parts':'both' if with_hands else 'arms','urdf_path':urdf})
        # Constructor reset the arms to initial_arm and attached hands to zero.
        backend=_TronWorldBinding(spec,world,group,initial_arm)
        return cls({'tron2a_l21':backend})

    def submit(self,frame_id,seconds,*,hands=None,arms=None,source_timestamp=None,
               source_timebase='monotonic_s',observed=None,steps=8):
        updates={key:{} for key in self.specs}
        for (model,side),command in (hands or {}).items():
            key=next(name for name,spec in self.specs.items() if spec.hand_model==model)
            valid=source_timestamp is not None and (observed is None or observed.get((key,side+'_hand'),False))
            updates[key][side+'_hand']=RobotPartCommand(command.joint_names,command.positions,
                source_timestamp,source_timebase,valid,'tracking' if valid else 'holding')
        for side,reports in (arms or {}).items():
            for key,report in reports.items():
                part=self.specs[key].parts[side+'_arm']
                valid=source_timestamp is not None and (observed is None or observed.get((key,side+'_arm'),False))
                updates[key][side+'_arm']=RobotPartCommand(part.joint_names,
                    [report['joints'][name] for name in part.joint_names],source_timestamp,
                    source_timebase,valid,'tracking' if valid else 'holding')
        batch={key:self.assembler.assemble(key,frame_id,seconds,updates[key]) for key in self.specs}
        return self.output.submit(batch,steps=steps)

    def close(self):self.output.close()


class _TronWorldBinding:
    """Keep the legacy world's exact motor gains and mounted-hand behavior."""
    def __init__(self,spec,world,group,initial_arm):
        self.spec,self.world,self.physics_group=spec,world,group
        import numpy as np
        self.initial={key:initial_arm[key.split('_')[0]] if key.endswith('_arm') else np.zeros(len(part.joint_names))
                      for key,part in spec.parts.items()}

    def validate(self,frame):self.spec.validate_frame(frame)

    def write(self,frame):
        if self.spec.display=='arms':
            self.world.apply_arms({s:frame.parts[s+'_arm'].positions for s in ('left','right')})
        else:
            from teleoperation.contracts.commands import Tron2AL21Command
            self.world.apply(Tron2AL21Command(timestamp=frame.timestamp,
                **{key:part.positions for key,part in frame.parts.items()},
                **{key+'_valid':part.observed for key,part in frame.parts.items()}))

    def read_state(self):return self.initial
    def close(self):pass
