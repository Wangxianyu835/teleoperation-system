import argparse
from pathlib import Path
from teleoperation.contracts.constants import HAND_SIDES
from teleoperation.data.hand_h5 import read_angle_arrays
from teleoperation.retargeting.hand.diagnostics import summarize_angles


def inspect_angle_h5(path):
    arrays, attrs = read_angle_arrays(path)
    return summarize_angles(path, arrays, attrs)

def run(args: argparse.Namespace) -> int:
    summary = inspect_angle_h5(args.angle_h5)
    print(f"path={summary['path']}")
    print(f"frames={summary['frames']}")
    for side in HAND_SIDES:
        values = summary[side]
        print(f"{side}_shape={values['shape']}")
        print(f"{side}_valid={values['valid']}")
        print(f"{side}_invalid={values['invalid']}")
        print(f"{side}_nonfinite={values['nonfinite']}")
        print(f"{side}_out_of_limits={values['out_of_limits']}")
    for name, value in sorted(summary["attrs"].items()):
        print(f"attr.{name}={value}")
    return 0
