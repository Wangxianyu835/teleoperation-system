"""Command-line entry point for exported angle H5 inspection.

用法（两种写法等价）：

    python inspect_angle_h5.py --angle-h5 tmp_motion/angles.h5
    python inspect_angle_h5.py tmp_motion/angles.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retargeting.inspect import run


def main() -> int:
    parser = argparse.ArgumentParser(prog="inspect_angle_h5.py")
    parser.add_argument("angle_h5", nargs="?", type=Path, help="exported angle H5 path")
    parser.add_argument("--angle-h5", dest="angle_h5_option", type=Path, help="exported angle H5 path")
    args = parser.parse_args()
    target = args.angle_h5_option or args.angle_h5
    if target is None:
        parser.error("the following arguments are required: --angle-h5")
    return run(argparse.Namespace(angle_h5=target))


if __name__ == "__main__":
    raise SystemExit(main())
