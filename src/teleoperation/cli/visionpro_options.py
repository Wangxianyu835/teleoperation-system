"""VR CLI construction without importing models, sensors or simulation."""
from pathlib import Path

from .hand_command_options import add_hand_command_arguments


def configure(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ip", help="Vision Pro Tracking Streamer IP or room code")
    source.add_argument("--input-h5", type=Path, help="existing raw Vision Pro transform recording")
    parser.add_argument("--backend", choices=("nn", "dex", "transhandr"), default="nn",
                        help='nn/dex share the RGB hand contract; transhandr is explicit upstream compatibility')
    from teleoperation.contracts.hand_models import HAND_MODELS
    from teleoperation.contracts.coordinates import UNIFIED_HAND_PROFILE
    parser.add_argument('--hand-model', choices=HAND_MODELS, default='l21')
    parser.set_defaults(source='visionpro', input_profile=UNIFIED_HAND_PROFILE,
                        model_asset_path=None, camera_index=0)
    parser.add_argument("--dex-calibration", type=Path, help="retired for unified control; NN/dex uses shared window normalization")
    parser.add_argument("--checkpoint", type=Path, help="shared target NN weights, or explicit TransHandR upstream weights")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-hand", choices=("left", "right", "both"),
                        help="sensor side (default: left for single-hand, both for dual-hand)")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="right",
                        help="destination robot hand (upstream L21 URDF is right)")
    parser.add_argument("--hand-mapping", choices=("same",),
                        help="explicit same-side routing; required with --hand both")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene", choices=("auto", "hands", "robots"), default="auto",
                        help="auto selects L21 hands or the selected native robot; TransHandR supports hands only")
    parser.add_argument("--frames", type=int, default=0, help="0 = until stream end or Ctrl+C")
    parser.add_argument("--fps", type=float, default=30., help="polling upper bound; 0 = unthrottled replay")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-steps", type=int, default=8)
    parser.add_argument("--view", choices=("full", "front"), default="full")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-h5", type=Path,
                        help="per-frame upstream raw18 and native17 diagnostics (TransHandR), or native joint results (dex); default: report path with .h5 suffix")
    add_hand_command_arguments(parser)
    return parser
