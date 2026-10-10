"""Argument schemas only: no model, device or simulator imports."""
import argparse
import os
from pathlib import Path
from teleoperation.paths import *
from teleoperation.retargeting.hand.config import L21, RUNTIME, ANGLE_LIMITS
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
PALM_LOCAL_V2_CHECKPOINT = DEFAULT_REALTIME_CHECKPOINT
ROOT = str(PROJECT_ROOT)

def hand_align(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(_handler="teleoperation.apps.hand:align")


def hand_train(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_WARMSTART_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=L21.training.epochs)
    parser.add_argument("--batch-size", type=int, default=L21.training.batch_size)
    parser.add_argument("--learning-rate", type=float, default=L21.training.learning_rate)
    parser.add_argument("--val-ratio", type=float, default=L21.training.val_ratio)
    parser.add_argument("--early-stopping-patience", type=int, default=L21.training.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=L21.training.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=RUNTIME.device)
    parser.set_defaults(_handler="teleoperation.apps.training:run")


def hand_export(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_H5,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_H5,
    )
    parser.add_argument("--batch-size", type=int, default=RUNTIME.export_batch_size)
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=L21.training.source_scale,
    )
    parser.add_argument(
        "--disable-identity-tracking",
        action="store_true",
        help="trust recorded left/right labels without temporal reassignment",
    )
    parser.add_argument(
        "--max-center-displacement",
        type=float,
        default=DEFAULT_MAX_CENTER_DISPLACEMENT,
        help="maximum normalized palm-center displacement between frames",
    )
    parser.add_argument(
        "--max-shape-rmse",
        type=float,
        default=DEFAULT_MAX_SHAPE_RMSE,
        help="maximum wrist-relative landmark RMSE between frames",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=RUNTIME.device,
    )
    parser.set_defaults(_handler="teleoperation.apps.hand:export")


def hand_inspect(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--angle-h5", type=Path, required=True)
    parser.set_defaults(_handler="teleoperation.apps.hand:inspect")


def hand_realtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--checkpoint", type=Path, default=PALM_LOCAL_V2_CHECKPOINT)
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=RUNTIME.camera_device)
    parser.add_argument("--frames", type=int, default=RUNTIME.camera_frames,
                        help="capture N frames; 0 runs until quit in visualization mode")
    parser.add_argument("--visualize", action="store_true", help="show raw hands, palm windows and L21 FK")
    parser.add_argument("--view", choices=("yz", "xz", "xy", "iso"), default=RUNTIME.view)
    parser.add_argument("--palm-radius", type=float, default=RUNTIME.palm_radius)
    parser.add_argument("--robot-radius", type=float, default=RUNTIME.robot_radius)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--headless", action="store_true", help="render without opening a GUI")
    parser.add_argument("--snapshot", type=Path, help="save the final rendered frame as PNG")
    parser.set_defaults(_handler="teleoperation.apps.hand:realtime")


def hand_record_world(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-asset-path", type=Path, default=DEFAULT_MEDIAPIPE_ASSET)
    parser.add_argument("--camera-index", type=int, default=RUNTIME.camera_index)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the camera window while recording (press q to stop early)",
    )
    parser.set_defaults(_handler="teleoperation.apps.world_landmark_capture:main")


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
    parser.set_defaults(_handler="teleoperation.apps.benchmark:main")

def sim_demo_joints(parser):
    parser.set_defaults(_handler="teleoperation.apps.demo.joints:main")

def dual_export(parser):
    parser = parser
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--angle-h5", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_TRON2A_URDF)
    parser.set_defaults(_handler="teleoperation.apps.dual_export_cli:main")

def dual_realtime(parser):
    parser = parser
    parser.add_argument("--adapter", required=True, help="module:factory returning an InputSource or iterable of raw combined observations")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_TRON2A_URDF))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.set_defaults(_handler="teleoperation.apps.dual_realtime:main")

def dual_replay(parser):
    parser = parser
    parser.add_argument("--command-h5", required=True)
    parser.add_argument("--urdf", default=str(DEFAULT_TRON2A_URDF))
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.set_defaults(_handler="teleoperation.apps.dual_replay:main")

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
    parser.set_defaults(_handler="teleoperation.apps.replay.replay_actions:main")

def replay_native_hand(parser):
    ap = parser
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--robot', default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--hand', default='both',
                    choices=['left', 'right', 'both'])
    ap.add_argument('--render', action='store_true', help='GUI 可视化')
    ap.add_argument('--view', default='full',
                    choices=['full', 'front', 'side', 'hands', 'left-hand', 'right-hand'],
                    help='full=整机 / front=正面 / side=侧面 / hands=双手 / '
                         'left-hand=左手特写 / right-hand=右手特写')
    ap.add_argument('--loop', type=int, default=1,
                    help='播放几遍（默认 1；设 0 表示无限循环，方便演示）')
    ap.add_argument('--limit-mode', default='clamp',
                    choices=['clamp', 'rescale'],
                    help='数据超出原装手限位时怎么办：'
                         'clamp=截断（默认，忠实映射）；'
                         'rescale=按比例缩放（保运动形状但改变语义，'
                         '用于判断"是不是被限位卡住了"）')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--start-frame', type=int, default=0,
                    help='start at this zero-based recording frame')
    ap.add_argument('--end-frame', type=int,
                    help='stop before this zero-based frame (default: recording end)')
    ap.add_argument('--substeps', type=int, default=0,
                    help='每帧数据推进多少个物理步（默认 0 = 自动匹配数据时间，'
                         '约 8 步）。设小了会变成慢动作且关节滞后')
    ap.add_argument('--report', action='store_true',
                    help='只打印映射报告，不启动回放')
    ap.add_argument('--no-repair', action='store_true')
    parser.set_defaults(_handler="teleoperation.apps.replay.replay_hand_native:main")

def replay_l21_hand(parser):
    ap = parser
    ap.add_argument('--file', type=str,
                    default=os.path.join(ROOT, 'datasets', 'raw',
                                         'retarget_twohand_153542.h5'),
                    help='队友输出的 h5 文件')
    ap.add_argument('--hand', choices=['left', 'right', 'both'],
                    default='right')
    ap.add_argument('--check', action='store_true',
                    help='只做静态检查，不渲染')
    ap.add_argument('--render', action='store_true',
                    help='GUI 可视化回放（不加则只做静态检查）')
    ap.add_argument('--headless-replay', action='store_true',
                    help='无头模式回放（不开窗口，用于自动验证）')
    ap.add_argument('--smooth', type=int, default=0,
                    help='滑动平均窗口（奇数，如 5）')
    ap.add_argument('--kalman', action='store_true',
                    help='启用卡尔曼滤波（搭配 --kalman-q/--kalman-r 调参）')
    ap.add_argument('--kalman-q', type=float, default=1.0,
                    help='卡尔曼过程噪声 q（默认 1.0）')
    ap.add_argument('--kalman-r', type=float, default=0.01,
                    help='卡尔曼观测噪声 r（默认 0.01）')
    ap.add_argument('--gate', type=float, default=0.0,
                    help='创新门控倍数（默认 0=关闭；仅孤立尖峰数据才需要）')
    ap.add_argument('--analyze', action='store_true',
                    help='打印跳变性质诊断（孤立尖峰 vs 成片真实运动）')
    ap.add_argument('--no-repair', action='store_true',
                    help='不修复「整帧异常」坏帧（默认会自动检测并插值修复）')
    ap.add_argument('--compare', action='store_true',
                    help='对比 原始/移动平均/卡尔曼 的平滑指标')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='数据帧率（用于滤波器 dt 与指标换算，默认 30）')
    ap.add_argument('--speed', type=float, default=1.0, help='播放倍速')
    parser.set_defaults(_handler="teleoperation.apps.replay.replay_hand_angles:main")

def replay_mounted_hand(parser):
    ap = parser
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--robot', default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--hand', default='both',
                    choices=['left', 'right', 'both'])
    ap.add_argument('--render', action='store_true', help='GUI 可视化')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--mount-offset', nargs=3, type=float,
                    default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'),
                    help='在「手基座 link」坐标系里的安装平移（米）')
    ap.add_argument('--mount-rpy', nargs=3, type=float,
                    default=[0.0, 0.0, 0.0], metavar=('R', 'P', 'Y'),
                    help='安装旋转（弧度）')
    ap.add_argument('--no-hide-robot-hand', action='store_true',
                    help='不隐藏机器人自带的手（便于对比位置）')
    ap.add_argument('--no-auto-mount', action='store_true',
                    help='关闭自动安装朝向（改用 --mount-rpy 手动值）')
    ap.add_argument('--no-fix-jitter', action='store_true',
                    help='不修「手抖」（默认会重标 l21 的质量/惯量）')
    ap.add_argument('--dt', type=float, default=1.0 / 1000.0,
                    help='物理时间步（默认 1/1000；l21 惯量极小需要小步长）')
    ap.add_argument('--substeps', type=int, default=8,
                    help='每帧数据推进多少个物理步（默认 8）。'
                         '0 = 自动让仿真时间正好等于数据时间'
                         '（最准但慢 4 倍）。'
                         '实测关节 29 ms 就能走完 1.5 rad，'
                         '所以 8 步足够跟上，视觉上只滞后约 4 帧')
    parser.set_defaults(_handler="teleoperation.apps.replay.replay_hand_on_robot:main")

def tools_show_robots(parser):
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.show_robots:main")

def tools_show_hand(parser):
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.show_hand:main")

def tools_check_environment(parser):
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.check_environment:main")

def tools_check_camera(parser):
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.check_camera:main")

def tools_check_gbk_safe(parser):
    ap = parser
    ap.add_argument('--strict', action='store_true',
                    help='有违规时返回退出码 1（用于 CI / 提交前检查）')
    ap.add_argument('--root', default='.', help='扫描根目录（默认当前）')
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.check_gbk_safe:main")

def tools_compare_training_hand_pose(parser):
    parser = parser
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--before-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--after-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("picture/training_compare.png"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.compare_training_hand_pose:main")

def tools_diagnose_hand_coordinates(parser):
    parser = parser
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.diagnose_hand_coordinates:main")

def tools_make_sample_data(parser):
    ap = parser
    ap.add_argument('--kind', choices=['hand', 'actions', 'all'],
                    default='all')
    ap.add_argument('--side', choices=['left', 'right'], default='right')
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--robot', type=str, default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--task', type=str, default='pushcube')
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.make_sample_data:main")

def tools_show_all_hands(parser):
    ap = parser
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--view', default='full', choices=['full', 'front'])
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--loop', type=int, default=1,
                    help='播放几遍（0 = 无限循环）')
    ap.add_argument('--substeps', type=int, default=0)
    ap.add_argument('--no-repair', action='store_true')
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.show_hands_all:main")

def tools_validate_retarget_input(parser):
    parser = parser
    parser.add_argument("path", type=Path, help="Path to a hand .npy replay file")
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.validate_retarget_input:main")

def tools_verify_hand_pipeline(parser):
    ap = parser
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--render', action='store_true', help='最后打开 GUI')
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.verify_hand_pipeline:main")

def tools_verify_p0(parser):
    parser = parser
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.verify_p0:main")

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
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.visualize_hand_keypoints:main")

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
    parser.set_defaults(_handler="teleoperation.apps.diagnostics.visualize_l21_fk:main")
