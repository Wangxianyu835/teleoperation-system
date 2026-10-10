"""Direct runner for MediaPipe world landmark data collection.

PyCharm: open this file and click Run. The default settings record 300 frames
from camera 0 and save a timestamped H5 under outputs/world_landmarks/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "hand_landmarker.task"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "world_landmarks"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record MediaPipe hand_world_landmarks (metres) to H5."
    )
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the camera window while recording (press q to stop early)",
    )
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = DEFAULT_OUTPUT_DIR / f"world_landmarks_{stamp}.h5"
    return args


def main() -> int:
    from teleoperation.apps.world_landmark_capture import run

    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
