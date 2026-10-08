"""Replay a 48D Tron2AL21Command H5 in the TRON2A DACH PyBullet scene."""

from __future__ import annotations

import argparse
import time

import h5py
import numpy as np

from teleoperation.retargeting.arm.calibration import ArmCalibration
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    timestamps, arrays = read_robot_commands(args.command_h5)
    world = Tron2ADualArmWorld(args.urdf, hand_mount_pose=ArmCalibration.load(args.calibration).hand_mount_pose, gui=True)
    try:
        while True:
            for index, timestamp in enumerate(timestamps):
                command = Tron2AL21Command(float(timestamp), arrays["left_arm_q"][index], arrays["left_hand_q"][index], arrays["right_arm_q"][index], arrays["right_hand_q"][index], bool(arrays["left_arm_valid"][index]), bool(arrays["left_hand_valid"][index]), bool(arrays["right_arm_valid"][index]), bool(arrays["right_hand_valid"][index]))
                world.apply(command)
                for _ in range(max(1, round(240 / args.fps))):
                    world.step()
                    time.sleep(1.0 / 240.0)
            if not args.loop:
                break
    finally:
        world.close()
    return 0


from teleoperation.data.command_h5 import read_robot_commands
