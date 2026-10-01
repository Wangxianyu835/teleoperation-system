"""Run a user-supplied observation iterator through the dual-arm teleoperator."""

from __future__ import annotations

import argparse
import importlib

from retargeting.config import DEFAULT_CHECKPOINT, L21
from retargeting.model import create_twohand_retargeter
from retargeting.realtime_dual_teleop import DualTeleoperator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a camera/VR payload iterator through TRON2A + L21 teleoperation")
    parser.add_argument("--adapter", required=True, help="module:factory returning an iterable of canonical payloads")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--urdf", default="third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    module_name, separator, factory_name = args.adapter.partition(":")
    if not separator:
        raise ValueError("--adapter must be formatted as module:factory")
    iterator = getattr(importlib.import_module(module_name), factory_name)()
    hand = create_twohand_retargeter(L21.model_kwargs(), args.device, str(args.checkpoint))
    teleoperator = DualTeleoperator(hand, args.device, args.calibration, args.urdf)
    for payload in iterator:
        command = teleoperator.update(payload)
        print(command.timestamp, command.left_arm_valid, command.right_arm_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
