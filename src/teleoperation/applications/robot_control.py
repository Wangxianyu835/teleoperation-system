"""Unified robot control and replay application boundary."""
from pathlib import Path
import numpy as np
from teleoperation.robots.control_spec import resolve_robot_targets


def preflight(args):
    targets=resolve_robot_targets(args.robot,args.robots)
    if args.frames<0 or not np.isfinite(args.fps) or args.fps<0: raise ValueError('Invalid frames/fps')
    if args.arm_calibrate_frames<1 or args.width<1 or args.height<1:raise ValueError('Invalid calibration/image dimensions')
    from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
    JointCommandPipeline({},filter_mode=args.joint_filter,alpha=args.filter_alpha,
        max_velocity=args.joint_max_velocity,kalman_q=args.kalman_q,kalman_r=args.kalman_r)
    if args.source=='visionpro':
        if (args.ip is None)==(args.input_h5 is None) or args.input_body is not None:
            raise ValueError('Vision Pro requires exactly one of --ip or --input-h5')
        if 'tron2a_l21' in targets and args.parts!='hands':
            raise ValueError('TRON2A VR arm mapping is not supported; choose hands only')
    elif args.ip is not None or args.input_h5 is not None:
        raise ValueError('VR options require --source visionpro')
    arms=args.parts!='hands'
    if args.tron2a_calibration is not None and ('tron2a_l21' not in targets or not arms):
        raise ValueError('TRON2A calibration only applies to its enabled arms')
    if 'tron2a_l21' in targets and arms and args.tron2a_calibration is None:
        raise ValueError('TRON2A arms require --tron2a-calibration')
    pairs=args.hand_checkpoint or []
    if len(dict(pairs))!=len(pairs): raise ValueError('Duplicate hand checkpoints')
    checkpoints={name:Path(path) for name,path in pairs}
    models={('l21' if n=='tron2a_l21' else n) for n in targets} if args.parts!='arms' else set()
    if (args.backend=='dex' or not models) and checkpoints: raise ValueError('No NN weights apply to this configuration')
    if args.backend=='nn' and models!=set(checkpoints): raise ValueError('Provide one shared checkpoint per selected hand model')
    for path in [*checkpoints.values(),args.input_h5,args.input_body,args.tron2a_calibration]:
        if path is not None and not path.is_file(): raise FileNotFoundError(path)
    for path in (args.record_results,args.report):
        if path is not None and path.exists(): raise FileExistsError(path)
    destinations=[p.resolve() for p in (args.record_results,args.report) if p is not None]
    inputs=[p.resolve() for p in (*checkpoints.values(),args.input_h5,args.input_body,args.tron2a_calibration) if p is not None]
    if len(destinations)!=len(set(destinations)) or set(destinations).intersection(inputs):
        raise ValueError('Outputs must differ from inputs and each other')
    if args.tron2a_urdf is not None and ('tron2a_l21' not in targets or not arms):
        raise ValueError('TRON2A URDF applies only to selected TRON2A arms')
    return targets,checkpoints


def run(args):
    import time
    import json
    from contextlib import ExitStack
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from .robot_session import RobotControlSession
    targets,checkpoints=preflight(args)
    specs={n:load_robot_control_spec(n,resource_overrides={'parts':args.parts,'side':args.side,
             **({'urdf_path':args.tron2a_urdf} if n=='tron2a_l21' and args.tron2a_urdf else {})}) for n in targets}
    # Load/check weights before devices or GUI. The session reuses this selection.
    from .hand_backend import build_target_hand_retargeters
    hand_selection=build_target_hand_retargeters(specs.values(),args.backend,checkpoints,args.device)
    if args.source=='camera' and args.input_body is None:
        from .realtime_hand_sim import resolve_model_asset
        args.model_asset_path=resolve_model_asset(args.model_asset_path)
        if args.parts!='hands':
            for name in ('smplerx_worker_python','smplerx_root','smplerx_checkpoint','smplx_model_dir'):
                if not Path(getattr(args,name)).exists():raise FileNotFoundError(getattr(args,name))
    if args.record_results is not None:raise ValueError('Complete command recording is not connected yet')
    calibration=None
    if args.tron2a_calibration is not None:
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        calibration=ArmCalibration.load(args.tron2a_calibration)
        if args.input_body is None and calibration.metadata.get('capture_kind')=='synthetic_validation_only':
            raise ValueError('Synthetic calibration is for offline validation only')
    with ExitStack() as resources:
        group=SimulationGroup(gui=not args.headless); resources.callback(group.close)
        backends={n:RobotPartSceneBuilder().build(s,group=group,
                     base_pose=(i*2.2,0.,.8) if len(specs)>1 else None,
                     calibration=calibration if n=='tron2a_l21' else None,geometry=args.geometry)
                  for i,(n,s) in enumerate(specs.items())}
        session=RobotControlSession(backends,backend=args.backend,checkpoints=checkpoints,device=args.device,
                    arm_calibrate_frames=args.arm_calibrate_frames,source_kind=args.source,hand_selection=hand_selection,
                    command_options={'filter_mode':args.joint_filter,'alpha':args.filter_alpha,
                        'max_velocity':args.joint_max_velocity,'kalman_q':args.kalman_q,'kalman_r':args.kalman_r})
        resources.callback(session.close)
        source=_open_source(args); resources.callback(source.close if hasattr(source,'close') else source.release)
        count=0; origin=None; previous=None; started=time.monotonic(); replaying=args.input_h5 is not None or args.input_body is not None
        try:
            while group.scene.alive() and (not args.frames or count<args.frames):
                tick=time.monotonic()
                try:raw=source.next_teleop_observation() if hasattr(source,'next_teleop_observation') else source.next_observation()
                except StopIteration:break
                from teleoperation.contracts.observations import RawHandFrame,RawTeleopObservation
                if raw is None:raw=RawHandFrame({'left':None,'right':None},time.time(),'visionpro')
                hands=raw.hands if isinstance(raw,RawTeleopObservation) else raw
                stamp=float(hands.timestamp)
                if replaying:
                    if previous is not None and stamp<=previous:raise ValueError('Recorded source time must strictly increase')
                    if origin is None:origin=stamp
                    seconds=stamp-origin
                else:seconds=time.monotonic()-started
                batch=session.next_batch(raw,count,seconds)
                dt=1/30 if previous is None else (stamp-previous if replaying else seconds-last_seconds)
                receipt=session.output.submit(batch,steps=max(1,min(24,round(min(dt,.1)*240))))
                previous=stamp; last_seconds=seconds; count+=1
                if args.fps and not (replaying and args.headless):time.sleep(max(0.,1/args.fps-(time.monotonic()-tick)))
        except KeyboardInterrupt:pass
        report={'targets':list(targets),'parts':args.parts,'side':args.side,'source':args.source,
                'arm_mode':'off' if args.parts=='hands' else 'wrist' if args.source=='visionpro' else 'body','frames':count,
                'physics_frames':group.scene.frames,'hardware_verified':False}
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            with args.report.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
        print(json.dumps(report))
    return 0


def _open_source(args):
    if args.input_body is not None:
        from .body_replay_source import BodyReplaySource
        return BodyReplaySource(args.input_body)
    if args.source=='visionpro':
        from teleoperation.inputs.visionpro import VisionProInput,VisionProReplayInput
        return VisionProReplayInput(args.input_h5,args.side) if args.input_h5 else VisionProInput(args.ip,args.side)
    from teleoperation.inputs.mediapipe import MediaPipeCameraInput
    camera=MediaPipeCameraInput(args.model_asset_path,args.camera_index,width=args.width,height=args.height,
                               fps=max(1,round(args.fps or 30)),invalid_hand_as_missing=True)
    if args.parts=='hands':return _CameraHandsSource(camera)
    worker=None
    try:
        from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport
        from teleoperation.inputs.smplerx import SmplerXClient
        from .camera_teleop_source import CameraTeleopSource
        worker=SmplerXWorkerTransport(worker_python=args.smplerx_worker_python,smplerx_root=args.smplerx_root,
            checkpoint=args.smplerx_checkpoint,body_model_dir=args.smplx_model_dir,model=args.smplerx_model)
        import time
        session_id=f'robot-{time.monotonic_ns()}'
        return CameraTeleopSource(camera,camera,SmplerXClient(worker,session_id=session_id),session_id=session_id)
    except BaseException:
        if worker is not None:worker.close()
        camera.close();raise


class _CameraHandsSource:
    def __init__(self,camera):self.camera=camera
    def next_observation(self):return self.camera.process_sample(self.camera.capture_sample())
    def close(self):self.camera.close()


def replay(args):
    resolve_robot_targets(args.robot,args.robots)
    raise RuntimeError('Robot command replay is not connected yet')
