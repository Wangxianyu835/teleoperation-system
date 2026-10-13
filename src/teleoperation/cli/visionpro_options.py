"""VR CLI construction without importing models, sensors or simulation."""
from pathlib import Path

from .hand_command_options import add_hand_command_arguments


def configure(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ip", help="Vision Pro Tracking Streamer IP or room code")
    source.add_argument("--input-h5", type=Path, help="existing raw Vision Pro transform recording")
    parser.add_argument("--checkpoint", type=Path, required=True, help="original TransHandR VR model_pos checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-hand", choices=("left", "right"), default="left",
                        help="sensor side (upstream defaults to left_fingers)")
    parser.add_argument("--hand", choices=("left", "right"), default="right",
                        help="destination robot hand (upstream L21 URDF is right)")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene", choices=("hands", "robots"), default="hands")
    parser.add_argument("--frames", type=int, default=0, help="0 = until stream end or Ctrl+C")
    parser.add_argument("--fps", type=float, default=30., help="polling upper bound; 0 = unthrottled replay")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-steps", type=int, default=8)
    parser.add_argument("--view", choices=("full", "front"), default="full")
    parser.add_argument("--report", type=Path)
    add_hand_command_arguments(parser)
    return parser
