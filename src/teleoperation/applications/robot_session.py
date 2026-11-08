"""One observation fans out to independent per-target command producers."""
from dataclasses import replace
from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.contracts.observations import RawTeleopObservation
from teleoperation.applications.hand_processing import UnifiedHandPipeline
from teleoperation.applications.hand_backend import build_target_hand_retargeters
from teleoperation.applications.robot_commands import RobotCommandAssembler
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.simulation.robot_output import RobotOutput


class RobotControlSession:
    def __init__(self,backends,*,backend='dex',checkpoints=None,device='cpu',
                 arm_calibrate_frames=15,source_kind='camera',command_options=None,hand_selection=None):
        self.backends=dict(backends); self.output=RobotOutput(backends)
        self.specs={n:b.spec for n,b in backends.items()}; self.source_kind=source_kind
        initial={}
        for name,b in backends.items():
            base=b.spec.reference_frame(0,0.); values=b.read_state()
            initial[name]=replace(base,parts={k:replace(p,positions=values[k]) for k,p in base.parts.items()})
        self.assembler=RobotCommandAssembler(self.specs,initial)
        self.pipeline=UnifiedHandPipeline()
        self.hand_specs,self.retargeter=(hand_selection if hand_selection is not None else
            build_target_hand_retargeters(self.specs.values(),backend,checkpoints or {},device))
        hand_initial={}
        for name,spec in self.specs.items():
            for side,hand in spec.hand_specs.items():
                p=initial[name].parts[side+'_hand']; named=dict(zip(p.joint_names,p.positions))
                hand_initial[hand.key]=[named[n] for n in hand.independent_joint_names]
        self.commands=JointCommandPipeline(self.hand_specs,initial=hand_initial,**(command_options or {}))
        self.bridges={}; self.wrists={}; self.previous_time=None
        for name,b in backends.items():
            if b.driver is None:continue
            if source_kind=='visionpro':
                b.driver.lock_others=True
                for adapter in b.driver.ik.values():
                    adapter.ik_lock_others=True
            from teleoperation.simulation.wrist_control import WristSceneController
            self.wrists[name]=WristSceneController(b.driver)
            if source_kind=='camera':
                from teleoperation.simulation.arm_realtime import RealtimeArmBridge
                from teleoperation.simulation.arm_profiles import profile_for
                from teleoperation.applications.camera_scene_arm import measure_camera_arm_geometry
                links,anchors=measure_camera_arm_geometry(b.driver)
                b.driver.set_shoulder_anchors(anchors)
                self.bridges[name]=RealtimeArmBridge([profile_for(name)],sides=b.driver.sides,
                    calibration_frames=arm_calibrate_frames,orientation='off',map_mode='posture',
                    robot_links=links,anchors=anchors)

    def next_batch(self,raw,frame_id,control_seconds):
        hands=raw.hands if isinstance(raw,RawTeleopObservation) else raw
        dt=1/30 if self.previous_time is None else control_seconds-self.previous_time
        if dt<=0:raise ValueError('Session control time must strictly increase')
        updates={n:{} for n in self.specs}
        window=self.pipeline.process_window(hands) if self.hand_specs else None
        results={} if window is None else self.retargeter.retarget(window)
        for name,spec in self.specs.items():
            for side,hand in spec.hand_specs.items():
                result=results.get(hand.key)
                command=self.commands.missing(hand.key,control_seconds) if result is None else self.commands.update(result,control_seconds)
                valid=result is not None and result.valid
                if valid:
                    updates[name][side+'_hand']=RobotPartCommand(command.joint_names,command.positions,
                        hands.timestamp,hands.metadata.get('timestamp_unit','sensor_seconds'),True,'tracking')
            b=self.backends[name]
            if name=='tron2a_l21' and b.arm_controller is not None:
                if not isinstance(raw,RawTeleopObservation):raise ValueError('TRON2A arms require body observations')
                valid=b.arm_controller.update(raw.upper_body,control_seconds,raw.wrists)
                for side in ('left','right'):
                    key=side+'_arm'
                    if key not in spec.parts:continue
                    observed=bool(valid[side] and raw.upper_body.valid.get(side,False))
                    updates[name][key]=RobotPartCommand(spec.parts[key].joint_names,b.arm_controller.controllers[side].q,
                        raw.upper_body.timestamp,'body_monotonic_s',observed,'tracking' if observed else 'holding')
            if b.driver is None:continue
            if self.source_kind=='visionpro':
                produced=self.wrists[name].solve_vr(hands,dt)
            else:
                if not isinstance(raw,RawTeleopObservation):raise ValueError('Camera arm control requires body observations')
                from teleoperation.applications.camera_scene_arm import solve_camera_arm_frame
                _,produced=solve_camera_arm_frame(self.bridges[name],b.driver,self.wrists[name],raw.upper_body,raw.wrists,dt)
                source_time=raw.upper_body.timestamp; timebase='body_monotonic_s'
            for side,reports in produced.items():
                report=reports.get(name)
                if self.source_kind=='visionpro':
                    wrist=None if hands.wrists is None else hands.wrists.get(side)
                    source_time=None if wrist is None else wrist.timestamp
                    timebase='visionpro_wrist_s'
                if report is None or source_time is None:continue
                if self.source_kind=='camera' and not raw.upper_body.valid.get(side,False):continue
                if self.source_kind=='visionpro':
                    tracker=self.wrists[name].trackers[side]
                    if (hands.wrists is None or hands.wrists.get(side) is None
                            or not hands.wrists[side].orientation_valid
                            or tracker.status not in ('tracking','reacquiring')):continue
                names=spec.parts[side+'_arm'].joint_names
                updates[name][side+'_arm']=RobotPartCommand(names,[report['joints'][n] for n in names],source_time,timebase,True,'tracking')
        batch={n:self.assembler.assemble(n,frame_id,control_seconds,updates[n]) for n in self.specs}
        self.previous_time=control_seconds
        return batch

    def close(self):self.output.close()
