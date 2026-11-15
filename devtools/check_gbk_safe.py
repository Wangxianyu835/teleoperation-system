"""Standalone entry for the same maintained GBK scanner as the public CLI."""
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from teleoperation.cli import main as cli_main
    return cli_main(['tools', 'check-gbk-safe', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())
