"""One observation fans out to independent per-target command producers."""
from collections import Counter
from dataclasses import replace
from types import MappingProxyType
from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.contracts.observations import RawTeleopObservation
from teleoperation.applications.native_hand_processing import UnifiedHandPipeline
from teleoperation.applications.hand_backend import build_target_hand_retargeters
from teleoperation.applications.robot_commands import RobotCommandAssembler
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.simulation.robot_output import RobotOutput


class RobotControlSession:
    def __init__(self,backends,*,backend='dex',checkpoints=None,device='cpu',
                 arm_calibrate_frames=15,source_kind='camera',command_options=None,hand_selection=None,
                 grasp_intent='off',camera_arms=None):
        if grasp_intent not in ('off', 'pinch-fist'):
            raise ValueError('Unknown grasp-intent mode')
        if grasp_intent != 'off' and (source_kind != 'camera' or not any(
                backend.spec.hand_specs for backend in backends.values())):
            raise ValueError('grasp-intent requires camera control with hands enabled')
        self._grasp_mode = grasp_intent
        self._grasp_intents = MappingProxyType({})
        self._grasp_source_time = None
        self._grasp_counts = ({side: Counter() for side in ('left', 'right')}
                              if grasp_intent != 'off' else None)
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
        self.wrists={}; self.previous_time=None
        self.camera_arms=camera_arms
        self._pending_batch=None
        self._pending_time=None
        if source_kind=='camera' and any(k.endswith('_arm') for spec in self.specs.values() for k in spec.parts):
            if camera_arms is None:
                raise ValueError('Camera arms require the MediaPipe/Paper controller')
        if source_kind=='visionpro':
            from teleoperation.simulation.wrist_control import WristSceneController
            for name,b in backends.items():
                if b.driver is None:continue
                b.driver.lock_others=True
                for adapter in b.driver.ik.values():adapter.ik_lock_others=True
                self.wrists[name]=WristSceneController(b.driver)

    @property
    def grasp_intents(self):
        """Latest immutable intents keyed by actual camera side."""
        return self._grasp_intents

    def grasp_intent_summary(self):
        report = {'mode': self._grasp_mode}
        if self._grasp_counts is not None:
            from teleoperation.retargeting.hand.grasp_intent import DETECTOR_PARAMETERS
            report.update(parameters=dict(DETECTOR_PARAMETERS),
                          counts={side: dict(counts) for side, counts in self._grasp_counts.items()},
                          spec_hashes={'/'.join(key): spec.spec_hash for key, spec in self.hand_specs.items()})
        return report

    def _update_grasp_intents(self, hands):
        from teleoperation.retargeting.hand.grasp_intent import detect_grasp_intent

        stamp = hands.timestamp
        if self._grasp_source_time is not None and stamp <= self._grasp_source_time:
            return
        points = hands.world_hands
        intents = {side: detect_grasp_intent(None if points is None else points[side], timestamp=stamp)
                   for side in self._grasp_counts}
        for side, intent in intents.items():
            self._grasp_counts[side][intent.kind if intent.valid else 'invalid'] += 1
        self._grasp_intents = MappingProxyType(intents)
        self._grasp_source_time = stamp

    def next_batch(self,raw,frame_id,control_seconds,*,new_observation=True,source_now_s=None,
                   integration_dt=None, motion_allowed=True):
        if not motion_allowed:
            raw, new_observation = None, False
        hands=raw.hands if isinstance(raw,RawTeleopObservation) else raw
        dt=.01 if self.previous_time is None else control_seconds-self.previous_time
        if integration_dt is not None:dt=integration_dt
        if dt<=0:raise ValueError('Session control time must strictly increase')
        updates={n:{} for n in self.specs}
        window=self.pipeline.process_window(hands) if self.hand_specs and new_observation and hands is not None else None
        results={} if window is None else self.retargeter.retarget(window)
        if (self._grasp_counts is not None and new_observation and hands is not None
                and hands.source != 'visionpro'):
            self._update_grasp_intents(hands)
        for name,spec in self.specs.items():
            for side,hand in spec.hand_specs.items():
                # The assembler holds the accepted command between camera
                # arrivals. Waiting must not reset filters or input timing.
                if not new_observation:
                    continue
                result=results.get(hand.key)
                command=self.commands.missing(hand.key,control_seconds) if result is None else self.commands.update(result,control_seconds)
                valid=result is not None and result.valid
                if valid:
                    updates[name][side+'_hand']=RobotPartCommand(command.joint_names,command.positions,
                        hands.timestamp,hands.metadata.get('timestamp_unit','sensor_seconds'),True,'tracking')
            if self.source_kind=='camera':
                continue
            b=self.backends[name]
            if b.driver is None:continue
            produced=self.wrists[name].solve_vr(hands,dt)
            for side,reports in produced.items():
                report=reports.get(name)
                if self.source_kind=='visionpro':
                    wrist=None if hands.wrists is None else hands.wrists.get(side)
                    source_time=None if wrist is None else wrist.timestamp
                    timebase='visionpro_wrist_s'
                if report is None or source_time is None:continue
                # Omit rejected updates so the assembler retains its last legal command.
                if not report.get('accepted',True):continue
                if self.source_kind=='camera' and not raw.upper_body.valid.get(side,False):continue
                if self.source_kind=='visionpro':
                    tracker=self.wrists[name].trackers[side]
                    if (hands.wrists is None or hands.wrists.get(side) is None
                            or not hands.wrists[side].orientation_valid
                            or tracker.status not in ('tracking','reacquiring')):continue
                names=spec.parts[side+'_arm'].joint_names
                updates[name][side+'_arm']=RobotPartCommand(names,[report['joints'][n] for n in names],source_time,timebase,True,'tracking')
        if self.camera_arms is not None and motion_allowed and new_observation:
            for name, parts in self.camera_arms.solve(raw, dt).items():
                updates[name].update(parts)
        batch=MappingProxyType({n:self.assembler.prepare(n,frame_id,control_seconds,updates[n]) for n in self.specs})
        self._pending_batch=batch
        self._pending_time=control_seconds
        return batch

    def submit(self,batch,*,steps):
        if batch is not self._pending_batch:
            raise ValueError('Submit the current prepared session batch')
        try:
            receipt=self.output.submit(batch,steps=steps)
        except BaseException:
            self._pending_batch=None
            raise
        for frame in batch.values():self.assembler.commit(frame)
        self.previous_time=self._pending_time
        self._pending_batch=None
        return receipt

    def close(self):self.output.close()
