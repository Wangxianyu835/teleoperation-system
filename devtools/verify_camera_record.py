"""Record current world H5 and verify its aligned three-frame training windows."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))


def main():
    from teleoperation.applications.camera_record import build_parser as camera_parser, run
    from teleoperation.learning.dataset import TwoHandH5Dataset
    from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--model-asset-path', type=Path)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'outputs/camera_verification')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    raw, aligned = args.out_dir/f'camera_{stamp}.h5', args.out_dir/f'aligned_{stamp}.h5'
    options = ['--output-h5',str(raw),'--aligned-output',str(aligned),'--frames',str(args.frames),
               '--seconds',str(args.seconds),'--camera-index',str(args.camera_index),'--no-preview']
    if args.model_asset_path:
        options += ['--model-asset-path',str(args.model_asset_path)]
    status = run(camera_parser().parse_args(options))
    if status:
        return status
    data = TwoHandH5Dataset(aligned,input_profile=PALM_LENGTH_PROFILE)
    print(json.dumps({'raw':str(raw),'aligned':str(aligned),'frames':data.frame_count,
                      'valid_windows':data.side_counts(),'input_profile':PALM_LENGTH_PROFILE}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
