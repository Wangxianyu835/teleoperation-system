"""Argument schemas only: no model, device or simulator imports."""
import argparse
from pathlib import Path
from teleoperation.paths import *
from teleoperation.retargeting.hand.config import L21, RUNTIME
from .smplerx_options import add_smplerx_arguments


def hand_dex_calibrate(parser):
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", choices=("visionpro",), default="visionpro")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0., help="seconds after first recorded frame")
    parser.add_argument("--duration", type=float, default=2.)
    parser.set_defaults(_handler="teleoperation.applications.dex_native:calibrate")




ROOT = str(PROJECT_ROOT)

def robot_control(parser):
    from .robot_options import configure
    configure(parser)

def robot_replay(parser):
    from .robot_options import configure
    configure(parser,replay=True)

def hand_record(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--landmark-space", choices=("world",), default="world")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path,
                             help="explicit raw filename, overriding automatic time/side naming")
    destination.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "datasets" / "raw",
                             help="directory for camera_world_hand_data_<side>_<YYYYMMDD_HHMMSS>.h5")
    parser.add_argument("--aligned-output", type=Path,
                        help="training H5 path; default: aligned_<raw filename>")
    parser.add_argument("--hand-side", choices=("left", "right", "both"), default="both",
                        help="hand(s) to record and include in automatic filename (default: both)")
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="requested camera FPS")
    parser.add_argument("--no-preview", action="store_true",
                        help="record without the camera/landmark preview window")
    parser.add_argument("--frames", type=int, default=0,
                        help="capture N frames; 0 records until q/Esc, window close or Ctrl+C")
    parser.set_defaults(_handler="teleoperation.applications.offline_h5_record:run")

def hand_align(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--landmark-space", choices=("world",), default="world")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(_handler="teleoperation.applications.hand_align:run")


def hand_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    from teleoperation.paths import DEFAULT_HAND_MODEL_ROOT
    from teleoperation.contracts.constants import HAND_MODELS
    parser.add_argument("--hand-model", choices=HAND_MODELS, nargs='+', required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_HAND_MODEL_ROOT)
    from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
    parser.add_argument('--input-profile',choices=(PALM_LENGTH_PROFILE,),default=PALM_LENGTH_PROFILE)
    parser.add_argument("--resume", type=Path, help="explicit model_last.pth from this target/run")
    parser.add_argument('--init-checkpoint',type=Path,help='strictly compatible new target weights; does not resume training state')
    parser.add_argument("--epochs", type=int, default=L21.training.epochs)
    parser.add_argument("--batch-size", type=int, default=L21.training.batch_size)
    parser.add_argument("--learning-rate", type=float, default=L21.training.learning_rate)
    parser.add_argument("--val-ratio", type=float, default=L21.training.val_ratio)
    parser.add_argument("--early-stopping-patience", type=int, default=L21.training.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=L21.training.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=RUNTIME.device)
    parser.set_defaults(_handler="teleoperation.applications.training:run")


def hand_export(parser: argparse.ArgumentParser) -> None:
    from .hand_joint_options import add_target_arguments
    from .hand_command_options import add_hand_command_arguments
    add_target_arguments(parser)
    add_hand_command_arguments(parser)
    parser.add_argument('--input', type=Path, required=True, help='canonical world metre human H5')
    parser.add_argument('--output', type=Path, required=True)
    parser.set_defaults(_handler='teleoperation.applications.hand_control:export')


def hand_control(parser):
    from .hand_joint_options import add_target_arguments
    from .hand_command_options import add_hand_command_arguments
    add_target_arguments(parser)
    add_hand_command_arguments(parser)
    from teleoperation.contracts.coordinates import UNIFIED_HAND_PROFILE
    parser.set_defaults(input_profile=UNIFIED_HAND_PROFILE)
    parser.add_argument('--source',choices=('camera','visionpro'),default='camera')
    vr = parser.add_mutually_exclusive_group()
    vr.add_argument('--input-h5',type=Path,help='raw Vision Pro transform recording')
    vr.add_argument('--ip',help='Vision Pro streamer address')
    parser.add_argument('--fps',type=float,default=0.,help='loop upper bound; 0 is unthrottled')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--scene',choices=('auto','hands','robots'),default='auto')
    parser.add_argument('--model-asset-path',type=Path)
    parser.add_argument('--camera-index',type=int,default=0)
    parser.add_argument('--frames',type=int,default=0,help='0 runs until interruption/window close')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--record-results',type=Path)
    parser.set_defaults(_handler='teleoperation.applications.hand_control:run')


def hand_list_models(parser):
    parser.set_defaults(_handler='teleoperation.applications.hand_control:list_models')


def replay_hand_joints(parser):
    from teleoperation.contracts.constants import HAND_MODELS
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--hand-model',choices=HAND_MODELS,required=True)
    parser.add_argument('--hand',choices=('left','right','both'),default='right')
    parser.add_argument('--headless',action='store_true')
    parser.set_defaults(_handler='teleoperation.applications.hand_control:replay')


def hand_inspect(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--angle-h5", type=Path, required=True)
    parser.set_defaults(_handler="teleoperation.applications.hand_inspect:run")




def hand_record_world(parser: argparse.ArgumentParser) -> None:
    parser.description = "Record shared RGB, MediaPipe Hands and SMPL-X body observations."
    add_smplerx_arguments(parser)
    parser.add_argument('--no-arms', action='store_true', help='record fingers only; do not start SMPLer-X')
    parser.add_argument('--startup-timeout', type=float, default=120., help='seconds to wait for first body result')
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--frames", type=int, default=30)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path,
                             help="explicit H5 filename, overriding automatic time naming")
    destination.add_argument("--output-dir", type=Path, default=DEFAULT_WORLD_CAPTURE_DIR,
                             help="directory for hand_body_<YYYYMMDD_HHMMSS>.h5")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=5.0,
        help="maximum capture rate; use 0 for unthrottled capture",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the camera window while recording (press q to stop early)",
    )
    parser.set_defaults(
        _handler="teleoperation.applications.world_landmark_capture:run",
        fast_exit=True,
    )


def sim_run(parser):
    parser = parser
    parser.add_argument('--task', type=str, default=None,
                        help='任务名 (e.g., pushcube, pickcube)')
    parser.add_argument('--robot', type=str, default='h1_2',
                        choices=['h1_2', 'gr1_t2', 'g1'],
                        help='机器人类型')
    parser.add_argument('--no-render', action='store_true',
                        help='无GUI渲染（无头模式）')
    parser.add_argument('--no-record', action='store_true',
                        help='不记录数据')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true',
                       help='运行 3 次正弦波演示；默认 pushcube，可用 --task 指定任务')
    modes.add_argument('--benchmark', action='store_true',
                       help='对 Level 1 任务进行基准测试；不能与 --task 同用')
    parser.add_argument('--trials', type=int, default=5,
                        help='基准测试每个任务的试验次数')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='数据存储目录')
    parser.set_defaults(_handler="teleoperation.applications.benchmark:main")

def dual_calibrate_camera(parser):
    parser.description = (
        "Capture stable bilateral shoulder-relative SMPL-X neutral wrists and "
        "TRON2A URDF FK references. Camera preview only; no robot control. "
        "Rotation and scale must be configured; neutral capture does not estimate extrinsics."
    )
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index, help="camera device number")
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET, help="Hands .task model (must exist)")
    add_smplerx_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="new calibration JSON; existing files are never overwritten")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_TRON2A_URDF, help="TRON2A URDF for FK and joint limits; meshes not required")
    for side in ("left", "right"):
        parser.add_argument(f"--safe-q-{side}", type=float, nargs=7, required=True,
                            help=f"{side} reference radians: proximal pitch/roll/yaw, elbow, wrist yaw/pitch/roll; must satisfy URDF limits")
    parser.add_argument("--rotation", type=float, nargs=9, required=True,
                        help="explicit proper 3x3 rotation from SMPL-X camera axes to base_Link, row-major; not automatically estimated")
    parser.add_argument("--scale", type=float, nargs=3, default=[1., 1., 1.], help="positive x/y/z mapping scales (default 1 1 1; configured, not estimated)")
    parser.add_argument("--translation", type=float, nargs=3, default=[0., 0., 0.], help="extra robot-frame offset in metres (default zero)")
    parser.add_argument("--workspace-min", type=float, nargs=3, required=True, help="minimum base_Link x/y/z in metres")
    parser.add_argument("--workspace-max", type=float, nargs=3, required=True, help="maximum base_Link x/y/z in metres")
    parser.add_argument("--samples", type=int, default=60, help="consecutive stable bilateral valid samples, >=3")
    parser.add_argument("--stability-tolerance", type=float, default=0.015, help="maximum elbow/wrist distance from window mean, in shoulder-relative metres")
    parser.add_argument("--min-elbow-bend", type=float, default=10., help="minimum bend away from straight, degrees; prevents degenerate forearm plane")
    parser.add_argument("--max-elbow-bend", type=float, default=100., help="maximum allowed neutral elbow bend, degrees")
    parser.add_argument("--max-sample-gap", type=float, default=0.5, help="maximum gap in relative seconds before restarting collection")
    parser.add_argument("--max-acceleration", type=float, default=20., help="positive stored joint acceleration limit, rad/s^2; no control in this command")
    parser.add_argument("--tracking-timeout", type=float, default=0.5, help="positive stored tracking timeout in seconds")
    parser.add_argument("--max-frames", type=int, default=3000, help="abort after this many camera frames")
    parser.add_argument("--timeout", type=float, default=120., help="abort collection after relative seconds")
    parser.add_argument("--width", type=int, default=640, help="requested camera width")
    parser.add_argument("--height", type=int, default=480, help="requested camera height")
    parser.add_argument("--fps", type=int, default=30, help="requested capture FPS")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True, help="show unmirrored skeleton/status; q/Esc/window close cancels without saving")
    parser.set_defaults(_handler="teleoperation.applications.mediapipe_tron2a:run")


def dual_camera(parser):
    parser.description = "Realtime camera SMPLer-X body -> selected robot simulation: arms only or dual arms and L21 hands. Requires real neutral calibration."
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--arms-only", action="store_true", help="arm-only simulation; no hand model required")
    modes.add_argument("--with-hands", action="store_true", help="full control using one canonical shared native L21 checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="canonical shared native L21 checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=RUNTIME.camera_device)
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET, help="MediaPipe Hands detector .task")
    add_smplerx_arguments(parser)
    parser.add_argument("--calibration", type=Path, required=True, help="real dual calibrate-camera JSON; synthetic fixtures are rejected")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_TRON2A_URDF)
    parser.add_argument("--orientation-mode", choices=("fixed", "forearm"), default="fixed")
    parser.add_argument("--display", choices=("full", "simplified"), default="full", help="full requires all meshes; explicitly choose simplified kinematic preview without contact acceptance")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True, help="camera arm/status preview; q/Esc/window close exits")
    parser.add_argument("--attach-hands", action=argparse.BooleanOptionalAction, default=None, help="arms-only defaults off (zero reference geometry); with-hands requires on")
    parser.add_argument("--frames", type=int, default=0, help="0 runs until exit; positive number limits capture frames")
    parser.add_argument("--simulation-steps", type=int, default=8, help="PyBullet 1/240 s steps per camera frame; default 8 for ~30 FPS")
    parser.add_argument("--report", type=Path, default=Path("outputs/dual/camera_report.json"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.set_defaults(_handler="teleoperation.applications.mediapipe_tron2a:run_camera")


def dual_export(parser):
    parser = parser
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--hand-joints", dest="angle_h5", metavar="HAND_JOINTS", type=Path, required=True, help="hand_joint_results_v1 accepted L21 commands")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_TRON2A_URDF)
    parser.set_defaults(_handler="teleoperation.applications.dual_export:main")

def dual_realtime(parser):
    parser = parser
    parser.add_argument("--adapter", required=True,
                        help="module:factory returning an object with next_observation() or an iterable of RawTeleopObservation")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_TRON2A_URDF))
    parser.add_argument("--checkpoint", type=Path, required=True, help="canonical shared native L21 checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.set_defaults(_handler="teleoperation.applications.dual_realtime:main")

def dual_replay(parser):
    parser = parser
    parser.add_argument("--command-h5", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_TRON2A_URDF))
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.set_defaults(_handler="teleoperation.applications.dual_replay:main")

def replay_actions(parser):
    ap = parser
    ap.add_argument('--file', type=str, default=None,
                    help='动作序列文件 (.npz/.h5/.hdf5)，契约H 格式')
    ap.add_argument('--dummy', action='store_true',
                    help='用假数据（正弦摆动）而非真实文件')
    ap.add_argument('--save-actions', type=str, default=None,
                    help='把动作序列另存为契约H 示例文件')
    ap.add_argument('--describe', action='store_true',
                    help='只打印动作空间定义后退出')
    ap.add_argument('--robot', type=str, default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--task', type=str, default='pushcube')
    ap.add_argument('--render', action='store_true', help='开启 GUI 可视化')
    ap.add_argument('--no-render', action='store_true', help='无头模式（默认）')
    ap.add_argument('--record', action='store_true', help='记录数据到 HDF5')
    ap.add_argument('--randomize', action='store_true',
                    help='回放时启用域随机化（默认关闭，保证可复现）')
    ap.add_argument('--steps', type=int, default=480,
                    help='--dummy 模式生成的步数')
    parser.set_defaults(_handler="teleoperation.applications.replay.replay_actions:main")




def hand_vr(parser):
    from teleoperation.cli.visionpro_options import configure
    configure(parser)
    parser.set_defaults(_handler="teleoperation.applications.visionpro_hand:main")


def tools_show_robots(parser):
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.show_robots:main")

def tools_show_hand(parser):
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.show_hand:main")

def tools_check_environment(parser):
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.check_environment:main")

def tools_check_camera(parser):
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.check_camera:main")

def tools_check_gbk_safe(parser):
    ap = parser
    ap.add_argument('--strict', action='store_true',
                    help='有违规时返回退出码 1（用于 CI / 提交前检查）')
    ap.add_argument('--root', default='.', help='扫描根目录（默认当前）')
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.check_gbk_safe:main")


def tools_diagnose_hand_coordinates(parser):
    parser = parser
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.diagnose_hand_coordinates:main")




def tools_visualize_hand_keypoints(parser):
    parser = parser
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--frame-start",
        type=int,
        help="First frame in a continuous inclusive range.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Last frame in a continuous inclusive range.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "picture")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot stored keypoints directly instead of training-processed 25-point keypoints.",
    )
    parser.add_argument(
        "--receptive-field",
        type=int,
        default=1,
        help="History window for processed H5 visualization; use 3 to match training windows.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip frames with no valid hand keypoints instead of stopping.",
    )
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.visualize_hand_keypoints:main")

def tools_visualize_l21_fk(parser):
    parser = parser
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "picture")
    parser.add_argument(
        "--pose",
        choices=("zero", "curl"),
        default="zero",
        help="Use zero angles or a small curled pose.",
    )
    parser.add_argument(
        "--label-names",
        action="store_true",
        help="Label each point with its joint name instead of its index.",
    )
    parser.set_defaults(_handler="teleoperation.applications.diagnostics.visualize_l21_fk:main")
