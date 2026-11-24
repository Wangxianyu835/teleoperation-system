"""Standalone TRON2A + L21 PyBullet demonstration, without tracking hardware."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.paths import DEFAULT_TRON2A_URDF
from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_TRON2A_URDF)
    parser.add_argument("--calibration", type=Path, help="optional ArmCalibration JSON with hand mount poses")
    parser.add_argument("--display", choices=("full", "simplified"), default="simplified")
    parser.add_argument("--headless", action="store_true", help="run without a GUI and without real-time delays")
    parser.add_argument("--seconds", type=float, default=10.0, help="simulation duration in seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    # A known valid demonstration pose, not a measured robot calibration.
    neutral = {side: np.array([0., 0., 0., -0.5, 0., 0., 0.]) for side in ("left", "right")}
    mounts = None
    if args.calibration is not None:
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        calibration = ArmCalibration.load(args.calibration)
        neutral = calibration.safe_q
        mounts = calibration.hand_mount_pose

    world = Tron2ADualArmWorld(args.urdf, gui=not args.headless, display=args.display,
                               safe_q=neutral, hand_mount_pose=mounts)
    try:
        # Keep the example inside each arm's actual URDF limits, including when
        # a user supplies a different neutral pose.
        bounds = {side: np.array([world.joint_info[i][8:10] for i in world.arm_indices[side]])
                  for side in ("left", "right")}
        hand_bounds = {side: np.array([world.p.getJointInfo(body, i, physicsClientId=world.client)[8:10]
                                      for i in world.hand_indices[side]])
                       for side, body in world.hand_bodies.items()}
        start = time.perf_counter()
        steps = max(1, round(args.seconds / world.time_step))
        for step in range(steps):
            if not world.is_connected():
                break
            t = step * world.time_step
            arms, hands = {}, {}
            for side in ("left", "right"):
                arm = np.array(neutral[side], dtype=float, copy=True)
                arm[3] += 0.15 * math.sin(2.0 * math.pi * t / 4.0)
                arms[side] = np.clip(arm, bounds[side][:, 0], bounds[side][:, 1])
                # Flex pitch joints, leaving splay/roll at zero.
                hand = np.zeros(17)
                hand[[1, 2, 4, 5, 7, 8, 10, 11, 14, 15, 16]] = 0.4 * (1.0 - math.cos(2.0 * math.pi * t / 4.0))
                hands[side] = np.clip(hand, hand_bounds[side][:, 0], hand_bounds[side][:, 1])
            world.apply(Tron2AL21Command(t, arms["left"], hands["left"], arms["right"], hands["right"],
                                         True, True, True, True))
            world.step()
            if not args.headless:
                time.sleep(max(0., start + (step + 1) * world.time_step - time.perf_counter()))
        if world.is_connected():
            state = world.actual_state()
            for side in ("left", "right"):
                hand_state = world.p.getJointStates(world.hand_bodies[side], world.hand_indices[side], physicsClientId=world.client)
                print(f"{side}: elbow={state['arms'][side]['q'][3]:.3f} rad, "
                      f"index_pitch={hand_state[1][0]:.3f} rad")
    except KeyboardInterrupt:
        pass
    finally:
        world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
