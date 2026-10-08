"""Run a user-supplied observation iterator through the dual-arm teleoperator."""

from __future__ import annotations

import argparse
import importlib

from teleoperation.paths import DEFAULT_CHECKPOINT
from teleoperation.retargeting.hand.config import L21
from teleoperation.retargeting.hand.predictor import create_twohand_retargeter
from teleoperation.apps.dual import DualTeleoperator


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    module_name, separator, factory_name = args.adapter.partition(":")
    if not separator:
        raise ValueError("--adapter must be formatted as module:factory")
    iterator = getattr(importlib.import_module(module_name), factory_name)()
    if not hasattr(iterator, "next_observation"):
        iterator = iter(iterator)
    teleoperator = None
    try:
        hand = create_twohand_retargeter(L21.model_kwargs(), args.device, str(args.checkpoint))
        teleoperator = DualTeleoperator(hand, args.device, args.calibration, args.urdf)
        while True:
            try:
                payload = iterator.next_observation() if hasattr(iterator, "next_observation") else next(iterator)
            except StopIteration:
                break
            if payload is None:
                continue
            command = teleoperator.update(payload)
            print(command.timestamp, command.left_arm_valid, command.right_arm_valid)
    finally:
        if teleoperator is not None: teleoperator.close()
        close = getattr(iterator, "close", None)
        if close is not None: close()
    return 0


