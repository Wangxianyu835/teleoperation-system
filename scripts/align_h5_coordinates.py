"""Compatibility wrapper for the canonical palm-local preprocessing CLI.

Use python -m retargeting align; legacy matrix APIs remain in coordinates.py.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retargeting.cli import main
from retargeting.preprocessing import align_h5

if __name__ == "__main__":
    raise SystemExit(main(["align", *sys.argv[1:]]))
