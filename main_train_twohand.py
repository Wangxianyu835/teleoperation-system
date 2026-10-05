"""Compatibility entry; use python -m retargeting train."""

import sys
from retargeting.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
