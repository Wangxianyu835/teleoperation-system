"""Unified robot selection with the preprint camera-arm boundary."""
from pathlib import Path
from teleoperation.contracts.constants import ROBOT_TARGETS
from teleoperation.paths import DEFAULT_POSE_ASSET
from .hand_command_options import add_hand_command_arguments


def configure(parser, *, replay=False):
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument('--robot', choices=ROBOT_TARGETS)
    targets.add_argument('--robots', choices=ROBOT_TARGETS, nargs='+')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--geometry', choices=('full','simplified'), default='full')
    parser.add_argument('--report', type=Path)
    if replay:
        parser.add_argument('--input', type=Path, required=True)
        parser.set_defaults(_handler='teleoperation.applications.robot_control:replay')
        return
    parser.add_argument('--parts', choices=('both','arms','hands'), default='both')
    parser.add_argument('--side', choices=('both','left','right'), default='both')
    parser.add_argument('--source', choices=('camera','visionpro'), default='camera')
    parser.add_argument('--backend', choices=('nn','dex'), default='dex')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--grasp-intent', choices=('off','pinch-fist'), default='off')
    parser.add_argument('--hand-checkpoint', nargs=2, action='append', metavar=('MODEL','PATH'))
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--ip')
    source.add_argument('--input-h5', type=Path)
    parser.add_argument('--tron2a-calibration', type=Path)
    parser.add_argument('--tron2a-urdf', type=Path)
    parser.add_argument('--tron2a-orientation-mode', choices=('fixed','forearm'), default='fixed')
    parser.add_argument('--pose-min-visibility', type=float, default=None,
                        help='Pose visibility threshold; TRON2A defaults to its calibration')
    parser.add_argument('--model-asset-path', type=Path)
    parser.add_argument('--pose-model-asset-path', type=Path, default=DEFAULT_POSE_ASSET)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--record-results', type=Path)
    parser.add_argument('--arm-calibrate-frames', type=int, default=30)
    parser.add_argument('--arm-calibrate-mode', choices=('hold','track'), default='track')
    parser.add_argument('--arm-ik-solver', choices=('paper','pybullet'), default='paper')
    parser.add_argument('--arm-orientation-scope', choices=('wrist','arm'), default='wrist')
    add_hand_command_arguments(parser)
    parser.set_defaults(_handler='teleoperation.applications.robot_control:run')
