"""Unified robot control and replay application boundary."""
from pathlib import Path
import numpy as np
from teleoperation.robots.control_spec import resolve_robot_targets


def preflight(args):
    targets=resolve_robot_targets(args.robot,args.robots)
    if args.frames<0 or not np.isfinite(args.fps) or args.fps<0: raise ValueError('Invalid frames/fps')
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
    return targets,checkpoints


def run(args):
    preflight(args)
    raise RuntimeError('Robot control session is not connected yet')


def replay(args):
    resolve_robot_targets(args.robot,args.robots)
    raise RuntimeError('Robot command replay is not connected yet')
