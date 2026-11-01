"""Standalone raw-world camera recorder using the maintained hand record workflow."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from teleoperation.paths import DEFAULT_MEDIAPIPE_ASSET, PROJECT_ROOT

MODEL_ASSET_CANDIDATES = (DEFAULT_MEDIAPIPE_ASSET,)


def default_model_asset():
    if not DEFAULT_MEDIAPIPE_ASSET.is_file():
        raise FileNotFoundError(f'MediaPipe model not found: {DEFAULT_MEDIAPIPE_ASSET}')
    return DEFAULT_MEDIAPIPE_ASSET


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-h5', type=Path)
    parser.add_argument('--aligned-output', type=Path)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--hand-side', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--model-asset-path', type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument('--no-preview', action='store_true')
    return parser


def run(args):
    from teleoperation.applications.offline_h5_record import run as record
    import math
    if args.frames < 0 or not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError('frames and seconds must be finite and nonnegative')
    if args.frames == 0 and args.seconds == 0 and args.no_preview:
        raise ValueError('Headless capture requires a positive frame or duration limit')
    values = vars(args).copy()
    values.update(output=args.output_h5, output_dir=PROJECT_ROOT/'datasets/raw',
                  landmark_space='world')
    return record(SimpleNamespace(**values))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
