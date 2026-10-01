"""Replay a 48D RobotCommand H5 in the TRON2A DACH PyBullet scene."""

from __future__ import annotations

import argparse
import time

import h5py
import numpy as np

from retargeting.arm import ArmCalibration
from retargeting.command import RobotCommand
from simulation.pybullet_world import Tron2ADualArmWorld


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay TRON2A + L21 RobotCommand H5 in PyBullet")
    parser.add_argument("--command-h5", required=True)
    parser.add_argument("--urdf", default="third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    with h5py.File(args.command_h5, "r") as h5_file:
        timestamps = np.asarray(h5_file["timestamps"][:], dtype=np.float64)
        arrays = {name: np.asarray(h5_file[name][:]) for name in ("left_arm_q", "left_hand_q", "right_arm_q", "right_hand_q", "left_arm_valid", "left_hand_valid", "right_arm_valid", "right_hand_valid")}
    world = Tron2ADualArmWorld(args.urdf, hand_mount_pose=ArmCalibration.load(args.calibration).hand_mount_pose, gui=True)
    try:
        while True:
            for index, timestamp in enumerate(timestamps):
                command = RobotCommand(float(timestamp), arrays["left_arm_q"][index], arrays["left_hand_q"][index], arrays["right_arm_q"][index], arrays["right_hand_q"][index], bool(arrays["left_arm_valid"][index]), bool(arrays["left_hand_valid"][index]), bool(arrays["right_arm_valid"][index]), bool(arrays["right_hand_valid"][index]))
                world.apply(command)
                for _ in range(max(1, round(240 / args.fps))):
                    world.step()
                    time.sleep(1.0 / 240.0)
            if not args.loop:
                break
    finally:
        world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
