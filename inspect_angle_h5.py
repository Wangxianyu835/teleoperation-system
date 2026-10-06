"""Command-line entry point for exported angle H5 inspection."""

import sys

from retargeting.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["inspect", *sys.argv[1:]]))
