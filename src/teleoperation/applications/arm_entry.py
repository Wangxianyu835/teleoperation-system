"""Argument-only adapters from legacy camera commands to unified robot control."""
import argparse
from pathlib import Path
from teleoperation.cli.robot_options import configure


def camera_arguments(args, robots, parts):
    robots=tuple(robots)
    if not robots or len(set(robots))!=len(robots):
        raise ValueError('Choose distinct targets explicitly with --robot NAME')
    if parts!='hands' and getattr(args,'input_npz',None) is not None:
        raise ValueError('Arm replay requires raw SMPL pose parameters in --input-body; derived-point NPZ is unsupported')
    if parts!='hands' and getattr(args,'arm_fitting_dir',None) is None:
        raise ValueError('Camera arms require --arm-fitting-dir; run arm fit for each selected robot')
    parser=argparse.ArgumentParser(add_help=False)
    configure(parser)
    base=parser.parse_args(['--robot',robots[0]])
    for key,value in vars(args).items():
        if hasattr(base,key) and key not in ('robot','robots','parts','source','_handler'):
            if value is not None:setattr(base,key,value)
    base.robot=robots[0] if len(robots)==1 else None
    base.robots=robots if len(robots)>1 else None
    base.parts=parts; base.source='camera'
    if hasattr(args,'gui'):base.headless=not args.gui
    if hasattr(args,'display'):base.geometry=args.display
    if robots==('tron2a_l21',):
        base.tron2a_urdf=Path(args.urdf)
        # Old extrinsic-neutral calibration is not a fitted SMPL body profile.
        base.tron2a_calibration=None
        if parts=='both':
            checkpoint=getattr(args,'checkpoint',None)
            if checkpoint is None:raise ValueError('--with-hands requires --checkpoint')
            base.backend='nn';base.hand_checkpoint=[('l21',str(checkpoint))]
        else:
            base.backend='dex';base.hand_checkpoint=None
    return base


def run_camera_alias(args, robots, parts):
    from teleoperation.applications.robot_control import run
    return run(camera_arguments(args,robots,parts))
