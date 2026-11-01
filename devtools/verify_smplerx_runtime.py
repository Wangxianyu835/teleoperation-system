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
import subprocess
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
    parser.add_argument('--input-image', type=Path, help='run real single-image inference in the worker environment')
    parser.add_argument('--input-video', type=Path, help='benchmark chronological video frames in the isolated worker')
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--duration-s', type=float, default=0.)
    return parser


def inspect_assets(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "worker-python": args.worker_python,
        "smplerx-root": args.smplerx_root,
        "checkpoint": args.checkpoint,
        "body-model-dir": args.body_model_dir,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from external.smplerx_worker.adapter import SmplerXAssets
    assets = SmplerXAssets(args.smplerx_root, args.checkpoint, args.body_model_dir, args.model)
    for name, path in assets.body_files().items():
        if not path.is_file():
            missing.append(f'{name}: {path}')
    return {
        "status": "assets_present" if not missing else "missing",
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
    code = 0
    if args.input_image is not None and args.input_video is not None:
        raise ValueError('Choose image or video input')
    if not report['missing'] and (args.input_image is not None or args.input_video is not None):
        benchmark = args.input_video is not None or args.duration_s > 0
        command = [str(args.worker_python), str(Path(__file__).resolve().parents[1] /
                   ('external/smplerx_worker/benchmark.py' if benchmark else 'external/smplerx_worker/smoke.py')), '--smplerx-root', str(args.smplerx_root),
                   '--checkpoint', str(args.checkpoint), '--body-model-dir', str(args.body_model_dir),
                   '--input-video' if args.input_video else '--input-image', str((args.input_video or args.input_image).resolve())]
        if benchmark:
            command += ['--samples', str(args.samples), '--duration-s', str(args.duration_s)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=max(300, args.duration_s + 180))
        report['worker_stderr'] = completed.stderr[-12000:]
        code = completed.returncode
        try:
            report['model_smoke'] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report['model_smoke'] = {'status': 'failed', 'reason': 'worker did not return a JSON report'}
            code = code or 1
        report['status'] = 'smoke_passed' if code == 0 else 'smoke_failed'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["missing"]:
        for item in report["missing"]:
            print(f"MISSING {item}", file=sys.stderr)
        return 2 if args.check else 0
    print(f"{report['status'].upper()} model={args.model} worker={args.worker_python}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
