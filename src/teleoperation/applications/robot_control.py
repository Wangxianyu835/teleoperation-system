"""Unified robot control and replay application boundary."""
from pathlib import Path
import numpy as np
from teleoperation.robots.control_spec import resolve_robot_targets


def preflight(args):
    targets=resolve_robot_targets(args.robot,args.robots)
    if getattr(args, 'preview', False) and (args.source != 'camera' or args.headless):
        raise ValueError('--preview requires a live camera and windowed control')
    if getattr(args,'grasp_intent','off')!='off' and (args.source!='camera' or args.parts=='arms'):
        raise ValueError('--grasp-intent requires camera control with hands enabled')
    if args.frames<0 or not np.isfinite(args.fps) or args.fps<0: raise ValueError('Invalid frames/fps')
    if args.arm_calibrate_frames<1 or args.width<1 or args.height<1:raise ValueError('Invalid calibration/image dimensions')
    from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
    JointCommandPipeline({},filter_mode=args.joint_filter,alpha=args.filter_alpha,
        max_velocity=args.joint_max_velocity,kalman_q=args.kalman_q,kalman_r=args.kalman_r)
    if args.source=='visionpro':
        if (args.ip is None)==(args.input_h5 is None):
            raise ValueError('Vision Pro requires exactly one of --ip or --input-h5')
        if 'tron2a_l21' in targets and args.parts!='hands':
            raise ValueError('TRON2A VR arm mapping is not supported; choose hands only')
    elif args.ip is not None or args.input_h5 is not None:
        raise ValueError('VR options require --source visionpro')
    arms=args.parts!='hands'
    if args.source == 'camera' and arms and 'tron2a_l21' in targets:
        raise ValueError('Unified MediaPipe camera arms support h1_2, gr1_t2 and g1; use dual camera for calibrated TRON2A')
    if args.tron2a_calibration is not None and ('tron2a_l21' not in targets or not arms):
        raise ValueError('TRON2A calibration only applies to its enabled arms')
    pairs=args.hand_checkpoint or []
    if len(dict(pairs))!=len(pairs): raise ValueError('Duplicate hand checkpoints')
    checkpoints={name:Path(path) for name,path in pairs}
    models={('l21' if n=='tron2a_l21' else n) for n in targets} if args.parts!='arms' else set()
    if (args.backend=='dex' or not models) and checkpoints: raise ValueError('No NN weights apply to this configuration')
    if args.backend=='nn' and models!=set(checkpoints): raise ValueError('Provide one shared checkpoint per selected hand model')
    for path in [*checkpoints.values(),args.input_h5,args.tron2a_calibration]:
        if path is not None and not path.is_file(): raise FileNotFoundError(path)
    for path in (args.record_results,args.report):
        if path is not None and path.exists(): raise FileExistsError(path)
    destinations=[p.resolve() for p in (args.record_results,args.report) if p is not None]
    inputs=[p.resolve() for p in (*checkpoints.values(),args.input_h5,args.tron2a_calibration) if p is not None]
    if len(destinations)!=len(set(destinations)) or set(destinations).intersection(inputs):
        raise ValueError('Outputs must differ from inputs and each other')
    if args.tron2a_urdf is not None and ('tron2a_l21' not in targets or not arms):
        raise ValueError('TRON2A URDF applies only to selected TRON2A arms')
    return targets,checkpoints


def validate_camera_resources(args):
    from .realtime_hand_sim import resolve_model_asset
    args.model_asset_path = resolve_model_asset(args.model_asset_path)
    if args.parts != 'hands' and not args.pose_model_asset_path.is_file():
        raise FileNotFoundError(args.pose_model_asset_path)


def run(args):
    import time
    import json
    from contextlib import ExitStack
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from .robot_session import RobotControlSession
    targets,checkpoints=preflight(args)
    if args.source=='camera':
        validate_camera_resources(args)
    specs={n:load_robot_control_spec(n,resource_overrides={'parts':args.parts,'side':args.side,
             **({'urdf_path':args.tron2a_urdf} if n=='tron2a_l21' and args.tron2a_urdf else {})}) for n in targets}
    # Load/check weights before devices or GUI. The session reuses this selection.
    from .hand_backend import build_target_hand_retargeters
    hand_selection=build_target_hand_retargeters(specs.values(),args.backend,checkpoints,args.device)
    calibration=None
    if args.tron2a_calibration is not None:
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        calibration=ArmCalibration.load(args.tron2a_calibration)
        if calibration.metadata.get('capture_kind')=='synthetic_validation_only':
            raise ValueError('Synthetic calibration is for offline validation only')
    with ExitStack() as resources:
        group=SimulationGroup(gui=not args.headless); resources.callback(group.close)
        backends={n:RobotPartSceneBuilder().build(s,group=group,
                     base_pose=(i*2.2,0.,.8) if len(specs)>1 else None,
                     calibration=calibration if n=='tron2a_l21' else None,geometry=args.geometry)
                  for i,(n,s) in enumerate(specs.items())}
        camera_arms = None
        if args.source == 'camera' and args.parts != 'hands':
            from .mediapipe_robot_arms import MediaPipeRobotArms
            camera_arms = MediaPipeRobotArms(backends, args)
        session=RobotControlSession(backends,camera_arms=camera_arms,backend=args.backend,checkpoints=checkpoints,device=args.device,
                    arm_calibrate_frames=args.arm_calibrate_frames,source_kind=args.source,hand_selection=hand_selection,
                    grasp_intent=getattr(args,'grasp_intent','off'),
                    command_options={'filter_mode':args.joint_filter,'alpha':args.filter_alpha,
                        'max_velocity':args.joint_max_velocity,'kalman_q':args.kalman_q,'kalman_r':args.kalman_r})
        resources.callback(session.close)
        writer=None
        if args.record_results is not None:
            from teleoperation.data.robot_command_recording import RobotCommandWriter
            metadata={'source':args.source,'parts':args.parts,'side':args.side,'geometry':args.geometry,
                      'backend':args.backend,'execution_config':{}}
            from teleoperation.runtime_metadata import runtime_metadata
            metadata['runtime']=runtime_metadata()
            if calibration is not None:
                metadata['execution_config']['tron2a_l21']={key:{s:v.tolist() for s,v in getattr(calibration,key).items()}
                    for key in ('safe_q','hand_mount_pose')}
            writer=resources.enter_context(RobotCommandWriter(args.record_results,specs,metadata))
        source=_open_source(args); resources.callback(source.close if hasattr(source,'close') else source.release)
        from .camera_preview import open_control_preview
        preview = open_control_preview(args, resources, source)
        count=0; origin=None; previous=None; started=time.perf_counter(); replaying=args.input_h5 is not None
        try:
            while group.scene.alive() and (not args.frames or count<args.frames):
                tick=time.perf_counter()
                try:raw=source.next_teleop_observation() if hasattr(source,'next_teleop_observation') else source.next_observation()
                except StopIteration:break
                if preview is not None and preview():
                    break
                from teleoperation.contracts.observations import RawHandFrame,RawTeleopObservation
                if raw is None:raw=RawHandFrame({'left':None,'right':None},time.time(),'visionpro')
                hands=raw.hands if isinstance(raw,RawTeleopObservation) else raw
                stamp=float(hands.timestamp)
                if replaying:
                    if previous is not None and stamp<=previous:raise ValueError('Recorded source time must strictly increase')
                    if origin is None:origin=stamp
                    seconds=stamp-origin
                else:seconds=time.perf_counter()-started
                batch=session.next_batch(raw,count,seconds)
                dt=1/30 if previous is None else (stamp-previous if replaying else seconds-last_seconds)
                try:receipt=session.submit(batch,steps=max(1,min(24,round(min(dt,.1)*240))))
                except BaseException:
                    if writer is not None and session.output.last_receipt is not None and session.output.last_receipt.frame_id==count:
                        writer.append(batch,session.output.last_receipt)
                    raise
                if writer is not None:writer.append(batch,receipt)
                previous=stamp; last_seconds=seconds; count+=1
                if args.fps and not (replaying and args.headless):time.sleep(max(0.,1/args.fps-(time.perf_counter()-tick)))
        except KeyboardInterrupt:
            if writer is not None:writer.abort('keyboard interrupt')
        from teleoperation.runtime_metadata import runtime_metadata
        report={'runtime':runtime_metadata(),'targets':list(targets),'parts':args.parts,'side':args.side,'source':args.source,
                'arm_mode':'off' if args.parts=='hands' else 'wrist' if args.source=='visionpro' else 'mediapipe-' + args.arm_ik_solver,'frames':count,
                'physics_frames':group.scene.frames,'hardware_verified':False,
                'grasp_intent':session.grasp_intent_summary()}
        if camera_arms is not None:
            report['arms'] = camera_arms.summary()
        if writer is not None:
            writer.set_metadata('grasp_intent', report['grasp_intent'])
            writer.set_metadata('arm_mode', report['arm_mode'])
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            with args.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print(json.dumps(report))
    return 0


def _open_source(args):
    if args.source == 'visionpro':
        from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
        return VisionProReplayInput(args.input_h5, args.side) if args.input_h5 else VisionProInput(args.ip, args.side)
    from teleoperation.inputs.mediapipe import MediaPipeCameraInput
    camera = MediaPipeCameraInput(args.model_asset_path, args.camera_index,
        width=args.width, height=args.height, fps=max(1,round(args.fps or 30)),
        invalid_hand_as_missing=True,
        pose_model_asset_path=args.pose_model_asset_path if args.parts != 'hands' else None)
    return _CameraSource(camera, args.parts != 'hands')


class _CameraSource:
    def __init__(self, camera, arms):
        self.camera = self._input = camera
        self.arms = arms
        self.last_raw_frame = None

    def next_teleop_observation(self):
        from dataclasses import replace
        if self.arms:
            raw = self.camera.next_teleop_observation()
            hands = raw.hands
        else:
            raw = None
            hands = self.camera.next_observation(include_world=True)
        stamp = float(hands.timestamp) / 1000.
        hands = replace(hands, timestamp=stamp, metadata={**hands.metadata, 'timestamp_unit':'unix_seconds'})
        self.last_raw_frame = {**hands.hands, 'timestamp':stamp}
        if raw is None:
            return hands
        return replace(raw, hands=hands, upper_body=replace(raw.upper_body, timestamp=stamp))

    def close(self):
        self.camera.close()



def replay(args):
    import json
    import time
    from contextlib import ExitStack
    from teleoperation.data.robot_command_recording import load_robot_recording
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from teleoperation.simulation.robot_output import RobotOutput
    targets=resolve_robot_targets(args.robot,args.robots)
    if args.report is not None and args.report.exists():raise FileExistsError(args.report)
    if args.report is not None and args.report.resolve()==args.input.resolve():raise ValueError('Report must differ from input')
    recording=load_robot_recording(args.input,targets)
    configs=recording.metadata.get('execution_config',{})
    for name,spec in recording.specs.items():
        if name=='tron2a_l21' and spec.display!='hands':
            config=configs.get(name)
            if not isinstance(config,dict):raise ValueError('Missing recorded TRON2A scene configuration')
            for side in ('left','right'):
                q=np.asarray(config['safe_q'][side],dtype=float);pose=np.asarray(config['hand_mount_pose'][side],dtype=float)
                if q.shape!=(7,) or pose.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-5:
                    raise ValueError('Invalid recorded TRON2A execution pose')
    with ExitStack() as resources:
        group=SimulationGroup(gui=not args.headless);resources.callback(group.close)
        backends={name:RobotPartSceneBuilder().build(spec,group=group,execution_only=True,
            execution_config=configs.get(name),geometry=args.geometry,
            base_pose=(i*2.2,0.,.8) if len(targets)>1 else None)
            for i,(name,spec) in enumerate(recording.specs.items())}
        output=RobotOutput(backends);resources.callback(output.close)
        count=steps_total=0;previous=None;started=time.perf_counter();first=None
        try:
            for batch,recorded_receipt in recording.iter_frames():
                if not group.scene.alive():break
                seconds=next(iter(batch.values())).timestamp
                if first is None:first=seconds
                if not args.headless:time.sleep(max(0.,seconds-first-(time.perf_counter()-started)))
                dt=next(iter(recorded_receipt.physics_dt.values()))
                if previous is not None and dt!=previous:raise ValueError('Changing physics dt is unsupported')
                if previous is None:
                    group.time_step=dt;group.scene.p.setTimeStep(dt,physicsClientId=group.scene.cid)
                steps=next(iter(recorded_receipt.steps.values()))
                output.submit(batch,steps=steps)
                previous=dt;count+=1;steps_total+=steps
        except KeyboardInterrupt:pass
        report={'targets':list(targets),'frames':count,'physics_steps':steps_total,
            'scene_selection':{name:{'parts':s.display,'side':s.side} for name,s in recording.specs.items()},
            'recorded_geometry':recording.metadata.get('geometry'),'geometry':args.geometry,
            'command_source':'accepted_recording','hardware_verified':False}
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            with args.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print(json.dumps(report))
    return 0
