"""Check the isolated SMPLer-X runtime without importing model libraries.

The checker only validates paths and writes a machine-readable report.  The
actual model smoke test is run by the worker environment after the official
SMPLer-X source and weights are installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the SMPLer-X worker assets")
    parser.add_argument("--check", action="store_true", help="return non-zero when an asset is missing")
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--smplerx-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--model", default="s32", choices=("s32",))
    parser.add_argument("--report", type=Path, required=True)
    return parser


def inspect_assets(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "worker-python": args.worker_python,
        "smplerx-root": args.smplerx_root,
        "checkpoint": args.checkpoint,
        "body-model-dir": args.body_model_dir,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    return {
        "status": "ready" if not missing else "missing",
        "model": args.model,
        "python": sys.version,
        "checker_python": str(Path(sys.executable).resolve()),
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "missing": missing,
        "model_smoke": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    report = inspect_assets(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["missing"]:
        for item in report["missing"]:
            print(f"MISSING {item}", file=sys.stderr)
        return 2 if args.check else 0
    print(f"READY model={args.model} worker={args.worker_python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
