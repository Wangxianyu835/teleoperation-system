"""Dependency-free options shared by realtime hand consumers."""


def add_hand_command_arguments(parser):
    parser.add_argument("--joint-filter", choices=("lowpass", "kalman", "none"),
                        default="lowpass", help="robot output filter for every hand backend")
    parser.add_argument("--filter-alpha", type=float, default=0.2,
                        help="low-pass new-sample weight at 30 Hz; smaller is smoother")
    parser.add_argument("--joint-max-velocity", type=float, default=1.5,
                        help="maximum commanded finger joint speed in rad/s")
    parser.add_argument("--kalman-q", type=float, default=1.0)
    parser.add_argument("--kalman-r", type=float, default=0.01)
    parser.add_argument("--collision-margin-mm", type=float, default=1.0,
                        help="inter-finger separation for the L21 hands scene")
    parser.add_argument("--no-collision-guard", action="store_true",
                        help="disable L21 inter-finger checks for diagnostic comparisons")
