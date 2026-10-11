# 函数 main（入口/工作流）；链路：CLI/调用方→应用编排→输入/算法/存储/仿真
from teleoperation.applications.dual_export import export_robot_commands


def main(args):
    count = export_robot_commands(args.observations, args.angle_h5, args.calibration, args.output, args.urdf)
    print(f"output={args.output}")
    print(f"frames={count}")
    return 0
