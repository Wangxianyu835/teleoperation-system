"""Compatibility entry; use python -m retargeting export."""

import sys
from retargeting.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export", *sys.argv[1:]]))
