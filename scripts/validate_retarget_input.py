"""Validate a hand .npy log against the canonical realtime input contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.retarget_io import validate_retarget_input
from input_adapters.npy_replay_adapter import NpyReplayAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a hand .npy replay file")
    args = parser.parse_args()

    adapter = NpyReplayAdapter(args.path)
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


if __name__ == "__main__":
    raise SystemExit(main())
