"""Read historical 48D H5 files and execute them through the shared robot output."""
from dataclasses import replace
import math
import time
from teleoperation.data.command_h5 import read_robot_commands


def main(args=None):
    if args is None:
        raise TypeError('Use teleoperation.cli.main() to parse command arguments')
    if not math.isfinite(args.fps) or args.fps<=0:
        raise ValueError('Replay FPS must be finite and positive')
    from teleoperation.retargeting.arm.calibration import ArmCalibration
    from teleoperation.robots.control_spec import load_robot_control_spec
    from teleoperation.simulation.robot_backends import SimulationGroup
    from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
    from teleoperation.simulation.robot_output import RobotOutput
    timestamps,arrays=read_robot_commands(args.command_h5)
    calibration=ArmCalibration.load(args.calibration)
    spec=load_robot_control_spec('tron2a_l21',resource_overrides={'urdf_path':args.urdf})
    frames=[]
    for index,stamp in enumerate(timestamps):
        frame=spec.reference_frame(index,index/args.fps)
        parts={}
        for key,part in frame.parts.items():
            valid=bool(arrays[key+'_valid'][index])
            parts[key]=replace(part,positions=arrays[key+'_q'][index],source_timestamp=float(stamp),
                               source_timebase='legacy_file_seconds',observed=valid,
                               status='tracking' if valid else 'holding')
        frame=replace(frame,parts=parts);spec.validate_frame(frame);frames.append(frame)
    group=SimulationGroup(gui=not args.headless)
    try:
        backend=RobotPartSceneBuilder().build(spec,group=group,execution_only=True,
            execution_config={'safe_q':calibration.safe_q,'hand_mount_pose':calibration.hand_mount_pose},
            geometry=args.display)
        output=RobotOutput({'tron2a_l21':backend});index=0
        while group.alive():
            for frame in frames:
                if not group.alive():break
                output.submit({'tron2a_l21':replace(frame,frame_id=index,timestamp=index/args.fps)},
                              steps=max(1,round(240/args.fps)))
                index+=1
                if not args.headless:time.sleep(1/args.fps)
            if not args.loop or not frames:break
    except KeyboardInterrupt:
        pass
    finally:
        group.close()
    return 0
