"""Shared camera/replayed-pose control for every declarative robot arm target."""
from dataclasses import replace
from pathlib import Path
from contextlib import ExitStack
import json
import threading
import time
import numpy as np


def build_arm_pipeline(args, specs, backends):
    from teleoperation.robots.arm_spec import load_arm_spec
    from teleoperation.data.arm_fitting import load_arm_fitting
    from teleoperation.retargeting.arm.body_model import BodyModel
    from .arm_pipeline import ArmPipeline
    arm_specs={name:load_arm_spec(name,spec.urdf_path) for name,spec in specs.items()}
    profiles={name:load_arm_fitting(Path(args.arm_fitting_dir)/(name+'.json'),
                 robot_spec_hash=spec.spec_hash) for name,spec in arm_specs.items()}
    identities={profile.body_spec_hash for profile in profiles.values()}
    if len(identities)!=1:raise ValueError('All robot fittings must use one shared body model specification')
    body_spec=next(iter(profiles.values())).body_spec
    if args.smplx_module_dir is not None:
        body_spec=replace(body_spec,module_dir=str(args.smplx_module_dir))
    if body_spec.hash!=next(iter(identities)):
        raise ValueError('Body model files changed since fitting; refit explicitly')
    body=BodyModel(body_spec,device=args.body_device)
    body.layer  # Load and verify required resources before the control loop.
    initial={name:{joint:float(q) for key,values in backends[name].read_state().items()
                   if key.endswith('_arm') for joint,q in zip(specs[name].parts[key].joint_names,values)}
             for name in specs}
    sides={name:tuple(key.split('_')[0] for key in spec.parts if key.endswith('_arm'))
           for name,spec in specs.items()}
    return ArmPipeline(arm_specs,profiles,body,initial=initial,max_age=args.target_max_age,
                       enabled_sides=sides)


class ObservationPump:
    """Capture never blocks the fixed-rate actuator loop; retain pending body arrivals."""
    def __init__(self,source):
        self.source=source
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.latest=None
        self.body_result=None
        self.fault=None
        self.count=0
        self.thread=threading.Thread(target=self._run,name='arm-observation',daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                raw=self.source.next_teleop_observation()
                if getattr(self.source,'last_fault',None):
                    raise RuntimeError(f'Body worker failed: {self.source.last_fault}')
                with self.lock:
                    self.latest=raw
                    if raw.body_result is not None:self.body_result=raw.body_result
                    self.count+=1
        except BaseException as error:
            self.fault=error

    def take(self):
        if self.fault is not None:raise self.fault
        with self.lock:
            raw=self.latest
            self.latest=None
            if raw is not None:
                raw=replace(raw,body_result=self.body_result)
                self.body_result=None
            return raw

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)
        close=getattr(self.source,'close',None) or self.source.release
        close()
        self.thread.join(timeout=2.)


def run_camera_control(args,targets,checkpoints):
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from .hand_backend import build_target_hand_retargeters
    from .robot_session import RobotControlSession
    from .robot_control import _open_source
    specs={name:load_robot_control_spec(name,resource_overrides={
        'parts':args.parts,'side':args.side,
        **({'urdf_path':args.tron2a_urdf} if name=='tron2a_l21' and args.tron2a_urdf else {})}) for name in targets}
    hand_selection=build_target_hand_retargeters(specs.values(),args.backend,checkpoints,args.device)
    with ExitStack() as resources:
        group=SimulationGroup(gui=not args.headless);resources.callback(group.close)
        backends={name:RobotPartSceneBuilder().build(spec,group=group,solve_arms=False,
            base_pose=(index*2.2,0.,.8) if len(specs)>1 else None,geometry=args.geometry)
            for index,(name,spec) in enumerate(specs.items())}
        pipeline=build_arm_pipeline(args,specs,backends)
        session=RobotControlSession(backends,backend=args.backend,checkpoints=checkpoints,
            device=args.device,source_kind='camera',hand_selection=hand_selection,arm_pipeline=pipeline,
            command_options={'filter_mode':args.joint_filter,'alpha':args.filter_alpha,
                'max_velocity':args.joint_max_velocity,'kalman_q':args.kalman_q,'kalman_r':args.kalman_r})
        resources.callback(session.close)
        source=_open_source(args)
        replaying=args.input_body is not None
        pump=None
        if replaying:
            resources.callback(source.release)
            try: upcoming=source.next_teleop_observation()
            except StopIteration:raise ValueError('Body recording has no observations')
            source_origin=float(upcoming.hands.timestamp)
        else:
            pump=ObservationPump(source);resources.callback(pump.close)
            source_origin=None
            ready_deadline=time.perf_counter()+args.body_startup_timeout
            first_live=None
            while time.perf_counter()<ready_deadline:
                sample=pump.take()
                if sample is not None and sample.body_result is not None:
                    first_live=sample
                    break
                time.sleep(.01)
            if first_live is None:
                raise TimeoutError('No accepted SMPL pose before body startup timeout')
        writer=None
        if args.record_results is not None:
            from teleoperation.data.robot_command_recording import RobotCommandWriter
            config={name:b.execution_config for name,b in backends.items() if hasattr(b,'execution_config')}
            writer=resources.enter_context(RobotCommandWriter(args.record_results,specs,
                {'source':'camera','parts':args.parts,'side':args.side,'geometry':args.geometry,
                 'backend':args.backend,'arm_method':'smpl_segment_pink_v1',
                 'execution_config':config,'body_spec_hash':pipeline.body_model.spec.hash,
                 'fitting_hashes':{name:p.robot_spec_hash for name,p in pipeline.profiles.items()}}))
        count=observations=0
        period=1/args.control_hz
        started=time.perf_counter()
        previous_tick=None
        physics_remainder=0.
        durations=[]
        last_raw=None
        exhausted=False
        source_previous=None
        try:
            while group.scene.alive() and (not args.frames or count<args.frames):
                wall_tick=time.perf_counter()
                seconds=count*period if replaying else wall_tick-started
                dt=period if previous_tick is None else seconds-previous_tick
                if dt<=0:continue
                if not replaying and dt>pipeline.max_catchup:
                    # Do not integrate accumulated stall time. Old targets are revoked.
                    pipeline.targets.clear();pipeline._valid={}
                raw=None
                if replaying:
                    source_now=source_origin+seconds
                    while not exhausted and float(upcoming.hands.timestamp)<=source_now+1e-9:
                        arrival=upcoming
                        arrival_time=float(arrival.hands.timestamp)
                        if source_previous is not None and arrival_time<=source_previous:
                            raise ValueError('Recorded source time must strictly increase')
                        source_previous=arrival_time
                        if arrival.body_result is not None:
                            pipeline.ingest(arrival.body_result,now_s=source_now,control_time=seconds)
                        last_raw=arrival;observations+=1;raw=arrival
                        try:upcoming=source.next_teleop_observation()
                        except StopIteration:exhausted=True
                    if exhausted and raw is None:break
                    # Ingest already occurred above so every new sample (not just the last) is handled.
                    if raw is not None:raw=replace(raw,body_result=None)
                else:
                    source_now=time.monotonic()
                    raw=first_live if count==0 else pump.take()
                    if raw is not None:last_raw=raw;observations+=1
                # Bounded substeps are each physically submitted, not merely integrated in memory.
                # Replay has an exact logical period. Live missed deadlines do not
                # accumulate an unbounded catch-up queue or feed a large dt to IK.
                integration_span=dt if replaying else period
                steps=max(1,int(np.ceil((min(integration_span,pipeline.max_catchup)-1e-12)/pipeline.max_dt)))
                sub_dt=min(integration_span,pipeline.max_catchup)/steps
                for sub in range(steps):
                    control_time=seconds-(steps-1-sub)*sub_dt
                    batch=session.next_batch(raw if sub==0 else None,count,control_time,
                        new_observation=raw is not None and sub==0,source_now_s=source_now,
                        integration_dt=sub_dt)
                    physics_remainder+=sub_dt/group.time_step
                    physics_steps=max(1,int(physics_remainder+1e-9))
                    physics_remainder-=physics_steps
                    try:receipt=session.submit(batch,steps=physics_steps)
                    except BaseException:
                        if writer is not None and session.output.last_receipt is not None:
                            writer.append(batch,session.output.last_receipt)
                        raise
                    if writer is not None:writer.append(batch,receipt)
                    count+=1
                previous_tick=seconds
                durations.append(time.perf_counter()-wall_tick)
                if not replaying or not args.headless:
                    time.sleep(max(0.,started+(count*period)-time.perf_counter()))
        except KeyboardInterrupt:
            if writer is not None:writer.abort('keyboard interrupt')
        if not pipeline.reconstruction_count:
            raise ValueError('No accepted raw SMPL pose arrived; old derived-point recordings cannot run the new arm method')
        report={'targets':list(targets),'parts':args.parts,'side':args.side,'source':'camera',
            'arm_method':'smpl_segment_pink_v1','frames':count,'observations':observations,
            'body_reconstructions':pipeline.reconstruction_count,'control_hz_target':args.control_hz,
            'physics_frames':group.scene.frames,'hardware_verified':False,
            'control_ms':{f'p{p}':float(np.percentile(durations,p)*1000) for p in (50,95,99)} if durations else {},
            'deadline_misses':sum(value>period for value in durations),
            'last_errors':pipeline.last_errors}
        if args.report:
            Path(args.report).parent.mkdir(parents=True,exist_ok=True)
            with Path(args.report).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
        print(json.dumps(report))
        return 0
