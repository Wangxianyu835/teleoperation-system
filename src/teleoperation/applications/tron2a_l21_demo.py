"""Preset TRON2A/L21 motion through the shared scene and command output."""
from __future__ import annotations
import argparse
from dataclasses import replace
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from teleoperation.paths import DEFAULT_TRON2A_URDF
from teleoperation.robots.specification import TRON2A_L21_MOUNT_POSE


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,default=DEFAULT_TRON2A_URDF)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--display',choices=('full','simplified'),default='simplified')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--static',action='store_true',help='Hold the initial pose for visual inspection')
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds<=0:
        parser.error('--seconds must be finite and positive')
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from teleoperation.simulation.robot_output import RobotOutput
    calibration=SimpleNamespace(
        safe_q={s:np.array([0,0,0,-.5,0,0,0.]) for s in ('left','right')},
        hand_mount_pose={s:np.array(TRON2A_L21_MOUNT_POSE) for s in ('left','right')})
    if args.calibration is not None:
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        calibration=ArmCalibration.load(args.calibration)
    spec=load_robot_control_spec('tron2a_l21',resource_overrides={'urdf_path':args.urdf})
    group=SimulationGroup(gui=not args.headless)
    try:
        backend=RobotPartSceneBuilder().build(spec,group=group,calibration=calibration,geometry=args.display)
        output=RobotOutput({'tron2a_l21':backend})
        start=time.perf_counter()
        for index in range(max(1,round(args.seconds/group.time_step))):
            if not group.alive():break
            t=index*group.time_step
            frame=spec.reference_frame(index,t)
            parts={}
            for key,part in frame.parts.items():
                side,kind=key.split('_')
                q=np.array(calibration.safe_q[side] if kind=='arm' else part.positions,copy=True)
                if not args.static:
                    if kind=='arm':q[3]+=.15*math.sin(2*math.pi*t/4)
                    else:q[[1,2,4,5,7,8,10,11,14,15,16]]=.4*(1-math.cos(2*math.pi*t/4))
                low,high=np.asarray(spec.parts[key].limits).T
                parts[key]=replace(part,positions=np.clip(q,low,high),source_timestamp=t,
                                   source_timebase='demo_seconds',observed=True,status='tracking')
            output.submit({'tron2a_l21':replace(frame,parts=parts)},steps=1)
            if not args.headless:time.sleep(max(0.,start+(index+1)*group.time_step-time.perf_counter()))
        if group.alive():
            actual=backend.read_state()
            for side in ('left','right'):
                print(f"{side}: elbow={actual[side+'_arm'][3]:.3f} rad, index_pitch={actual[side+'_hand'][1]:.3f} rad")
    except KeyboardInterrupt:
        pass
    finally:
        group.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
