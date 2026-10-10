"""Validate a hand .npy log against the canonical realtime input contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teleoperation.paths import PROJECT_ROOT

from teleoperation.contracts.validation import validate_retarget_input
from teleoperation.applications.hand_processing import NpyReplayWorkflow


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    adapter = NpyReplayWorkflow(args.path)
    payload_count = 0
    first_payload = None
    for payload in adapter:
        validate_retarget_input(payload)
        payload_count += 1
        if first_payload is None:
            first_payload = payload

    print(f"path={args.path}")
    print(f"payloads={payload_count}")
    if first_payload is None:
        print("result=no_complete_three_frame_window")
        return 1

    print(f"source={first_payload['source']}")
    for side in ("left", "right"):
        hand = first_payload["hands"][side]
        print(f"{side}_shape={None if hand is None else hand.shape}")
    print("result=ok")
    return 0


