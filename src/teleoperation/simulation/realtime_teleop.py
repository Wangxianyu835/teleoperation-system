"""Live RGB or SMPL-X recording to H1-2 / GR1-T2 / G1 simulation.

A single CameraTeleopSource distributes RGB to MediaPipe Hands and an isolated
SMPLer-X worker. Each retains its own capture timestamp. Live and recorded body
observations share apply_camera_arm_frame, the existing bridge, wrist controller,
IK and bounded motor commands. --input-body replays new H5 observations;
--input-npz is finger-only. See docs/smplerx-arm.md for runnable commands.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
from teleoperation.applications.hand_backend import build_camera_retargeter
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.applications.camera_preview import PREVIEW_HINT_ROBOTS, _window_pos
from teleoperation.applications.realtime_support import _describe, _drain_in_background, _loop_fps, _median_interval
from teleoperation.applications.realtime_hand_sim import resolve_model_asset
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.inputs.smplerx import SmplerXClient
from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport
from teleoperation.applications.camera_teleop_source import CameraTeleopSource
from teleoperation.retargeting.hand.config import RUNTIME
from teleoperation.simulation.arm_preview import (
    ARM_PREVIEW_WINDOW,
    build_arm_preview,
)
from teleoperation.simulation.arm_profiles import loadable_robots, profile_for
from teleoperation.simulation.arm_ik import (ELBOW_PASS_ORDERS, ELBOW_RECOVERY_COOLDOWN_FRAMES,
                                             ELBOW_RECOVERY_DEG, ELBOW_RECOVERY_MARGIN_DEG)
from teleoperation.simulation.arm_realtime import (
    CALIBRATE_MODES,
    DEFAULT_LATERAL_GUARD_M,
    MAP_MODES,
    MAX_DIRECTION_RATE_DEG_S,
    TRAVEL_MODES,
    AxisMap,
    RealtimeArmBridge,
    load_neutral,
    save_neutral,
)
from teleoperation.simulation.arm_scene_driver import SLEW_MAX_STEP_M, ArmSceneDriver
from teleoperation.simulation.scene_appearance import apply_appearance
from teleoperation.simulation.hand_commands import add_hand_command_arguments

SIDES = ("left", "right")
ARM_ENTRY = "realtime_teleop"
#: 摄像头循环名义帧长（秒）：第一帧还没有"上一帧"可比，斜坡按它换算，
#: 这样第一帧也是 20 mm/帧（旧口径），不会退化成"一步跳到工作点"。
NOMINAL_FRAME_DT = 1.0 / 30.0
#: Default wrist target speed in millimetres/second.
DEFAULT_ARM_SLEW_MM_PER_S = 600.0
#: HUD 里把来源缩成一个字母：P = pose 手臂（主），H = 手部兜底，- = 无数据（保持上一帧）。
SOURCE_SHORT = {"smplerx": "S", "hand": "H", "none": "-"}


class ArmGainAction(argparse.Action):
    """Parse a robot name and a finite positive gain before opening devices."""

    def __call__(self, parser, namespace, values, option_string=None):
        name, value = values
        if name not in loadable_robots():
            raise argparse.ArgumentError(self, f"unknown robot {name!r}")
        try:
            gain = float(value)
        except ValueError:
            raise argparse.ArgumentError(self, "gain must be a finite positive number") from None
        if not np.isfinite(gain) or gain <= 0:
            raise argparse.ArgumentError(self, "gain must be a finite positive number")
        pairs = list(getattr(namespace, self.dest, None) or [])
        pairs.append((name, gain))
        setattr(namespace, self.dest, pairs)


class SimulationStepClock:
    """Match live physics time to elapsed time; keep offline steps deterministic."""

    def __init__(self, fixed_steps=8, realtime=True, clock=time.perf_counter):
        if fixed_steps < 1:
            raise ValueError("simulation steps must be positive")
        self.fixed_steps, self.realtime, self.clock = int(fixed_steps), realtime, clock
        self.previous = clock()
        self.fraction = 0.0
        self.total_steps = self.capped_frames = 0

    def next_steps(self):
        if self.realtime:
            now = self.clock()
            elapsed = max(0.0, now - self.previous)
            self.previous = now
            # Do not replay a long freeze as an abrupt burst of physics updates.
            self.capped_frames += int(elapsed > 0.1)
            ticks = min(elapsed, 0.1) * 240 + self.fraction
            steps = max(1, int(ticks))
            self.fraction = max(0.0, ticks - steps)
        else:
            steps = self.fixed_steps
        self.total_steps += steps
        return steps


class CameraFrameSource:
    """Shared capture adapter for independent finger and SMPL-X body output."""

    def __init__(self, model_asset_path, camera_index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30, landmark_space: str = "world",
                 smplerx_client=None, session_id="camera", arms_enabled=False):
        self.model_asset_path = Path(model_asset_path)
        self.landmark_space = landmark_space
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space, input_profile=PALM_LENGTH_PROFILE)
        self.processor = self.pipeline.processor
        self.last_raw_frame = None
        self.last_observation = None
        self._input = MediaPipeCameraInput(
            model_asset_path=model_asset_path, camera_index=camera_index,
            width=width, height=height, fps=fps, invalid_hand_as_missing=True)
        self._source = CameraTeleopSource(
            self._input, self._input, smplerx_client, session_id=session_id,
            arms_enabled=arms_enabled)
        self.body_enabled = bool(arms_enabled)
        self._start_timestamp = None
        self._last_capture_time_s = self._capture_dt = None

    def next_frame(self):
        """采集一帧；返回手部三帧窗口（还没攒够就是 ``None``）。

        ``last_raw_frame`` 是这一帧的手部（归一化 + ``world_*``），``last_observation``
        是同一帧的完整观测（含 pose 手臂），两者同源且同一时间戳。
        """
        try:
            observation = self._source.next_teleop_observation()
        except StopIteration:
            self.exhausted = True
            return None
        self.last_observation = observation
        raw = observation.hands
        stamp = float(raw.timestamp)
        previous = getattr(self, '_last_capture_time_s', None)
        if previous is not None and stamp <= previous:
            raise ValueError('Camera capture times must increase')
        self._capture_dt = None if previous is None else stamp - previous
        self._last_capture_time_s = stamp
        if self._start_timestamp is None:
            self._start_timestamp = float(raw.timestamp)
        raw_frame = {**raw.hands, "timestamp": raw.timestamp, "wrists": None,
                     "metadata": dict(raw.metadata)}
        self.last_raw_frame = raw_frame
        if raw.world_hands is not None:
            raw_frame.update({"world_" + side: raw.world_hands.get(side) for side in SIDES})
        frame = {
            **raw_frame,
            "timestamp": float(raw.timestamp) - self._start_timestamp,
            "metadata": {
                **raw_frame.get("metadata", {}),
                "capture_time_s": raw_frame["timestamp"],
                "timestamp_unit": "relative_seconds",
            },
        }
        if raw.world_hands is not None:
            frame.update({"world_" + side: raw.world_hands[side] for side in SIDES})
        diagnostics = getattr(self, "diagnostics", None)
        with (nullcontext() if diagnostics is None else diagnostics.stage("hand_preprocessing")):
            if raw.world_hands is None and all(value is None for value in raw.hands.values()):
                self.pipeline.reset()
                return None
            return self.pipeline.process_window(frame)

    @property
    def frame_dt(self):
        return self._capture_dt

    def arms(self):
        """The tracked SMPLer-X shoulder/elbow/wrist observation."""
        observation = self.last_observation
        return None if observation is None or not self.body_enabled else observation.upper_body

    def wrists(self):
        observation = self.last_observation
        return None if observation is None or not self.body_enabled else observation.wrists

    def release(self, **kwargs):
        return self._source.release()


class BodyReplayFrameSource(CameraFrameSource):
    """Run a new body recording through the same hand/control adapters as live input."""
    def __init__(self, path, landmark_space='world'):
        from teleoperation.applications.body_replay_source import BodyReplaySource
        self._source = BodyReplaySource(path)
        self.frames = self._source.frames
        self.exhausted = False
        self.landmark_space = landmark_space
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space, input_profile=PALM_LENGTH_PROFILE)
        self.processor = self.pipeline.processor
        self.last_raw_frame = self.last_observation = None
        self.body_enabled = True
        self._start_timestamp = None
        self._last_capture_time_s = self._capture_dt = None


class ReplayFrameSource:
    """Finger-only NPZ debug input. Legacy arm and wrist fields are rejected."""

    def __init__(self, path, landmark_space: str = "world"):
        # 一次性读进内存：np.load 的句柄会锁住文件（Windows 上连删临时目录都不行），
        # 而这里的帧本来就只有几十 MB 级，读完之后不被文件句柄牵着走更安全。
        with np.load(str(path)) as handle:
            self.data = {key: np.asarray(handle[key]) for key in handle.files}
        legacy = [key for key in self.data if "pose" in key.lower() or key in {
            "left_arm", "right_arm", "wrist_reference_frame", "wrist_local_frame",
            "left_wrist_orientation_xyzw", "right_wrist_orientation_xyzw"}]
        if legacy:
            raise ValueError(
                "Legacy MediaPipe Pose arm NPZ is not accepted; replay an SMPL-X body recording"
            )
        self.frames = int(np.asarray(self.data["timestamps"]).shape[0])
        self.landmark_space = landmark_space
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space, input_profile=PALM_LENGTH_PROFILE)
        self.processor = self.pipeline.processor
        self.last_raw_frame = None
        self.last_observation = None
        self.exhausted = False
        self.index = 0

    def _points(self, key, index, shape):
        if key not in self.data:
            return None
        values = np.asarray(self.data[key][index], dtype=np.float64)
        if values.shape != tuple(shape) or not np.isfinite(values).all():
            return None
        return values

    def next_frame(self):
        """下一帧的手部窗口；录完了就把 ``exhausted`` 置 True 并返回 ``None``。"""
        if self.index >= self.frames:
            self.exhausted = True
            return None
        index, self.index = self.index, self.index + 1
        timestamp = float(np.asarray(self.data["timestamps"])[index])
        raw_frame = {"timestamp": timestamp,
                     "metadata": {"source": "npz_replay",
                                  "timestamp_unit": "relative_seconds",
                                  "source_landmark_space": "mediapipe_normalized",
                                  "world_landmark_space": "mediapipe_world_meters",
                                  "world_length_unit": "m",
                                  "world_origin": "hand_geometric_center",
                                  "source_scale": 1.0}}
        for side in SIDES:
            raw_frame[side] = self._points(f"{side}_hand", index, (21, 3))
            raw_frame[f"world_{side}"] = self._points(f"world_{side}_hand", index, (21, 3))
        # NaN（= 那帧没检到）照样报个原因，和实时链路 inputs/mediapipe.py 同一套口径。
        missing = {side: "npz replay: hand not detected in this frame"
                   for side in SIDES if raw_frame[side] is None}
        if missing:
            raw_frame["metadata"]["invalid_reasons"] = missing
        world_missing = {side: "npz replay: no world landmarks in this frame"
                         for side in SIDES if raw_frame[f"world_{side}"] is None}
        if world_missing:
            raw_frame["metadata"]["world_invalid_reasons"] = world_missing
        self.last_raw_frame = raw_frame
        raw_frame["wrists"] = None
        diagnostics = getattr(self, "diagnostics", None)
        with (nullcontext() if diagnostics is None else diagnostics.stage("hand_preprocessing")):
            return self.pipeline.process_window(raw_frame)

    def arms(self):
        return None

    def wrists(self):
        return None

    def release(self, **kwargs):
        """回放没有设备要关；保留接口以便和 ``CameraFrameSource`` 互换。"""
        return None


def build_scene(args, sides):
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    return ThreeRobotHandScene(sides, gui=not args.headless,
                               steps_per_frame=args.sim_steps,
                               robot_types=args.robot)


def build_parser() -> argparse.ArgumentParser:
    """world米制、两侧；机器人必须显式选择，可重复选择进行对比。"""
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.simulation.realtime_teleop",
        description="Realtime camera -> hand angles + arm targets -> H1-2 / GR1-T2 / G1")
    parser.add_argument("--backend", choices=("nn", "dex"), default="dex")
    parser.add_argument("--diagnostics", action="store_true", help="read-only arm timings and state statistics")
    parser.add_argument("--diagnostic-trace", type=Path, help="stream diagnostic CSV; implies --diagnostics; must be a new file")
    parser.add_argument("--hand-checkpoint", nargs=2, action="append", default=None,
                        metavar=("MODEL", "PATH"), help="target-specific NN checkpoint (repeatable)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=RUNTIME.camera_device)
    parser.add_argument("--model-asset-path", type=Path, default=None)
    parser.add_argument("--smplerx-worker-python", type=Path, default=None)
    parser.add_argument("--smplerx-root", type=Path, default=None)
    parser.add_argument("--smplerx-checkpoint", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=None)
    parser.add_argument("--smplerx-model", choices=("s32",), default="s32")
    from teleoperation.cli.smplerx_options import smplerx_defaults
    parser.set_defaults(**smplerx_defaults())
    parser.add_argument("--pose-model-asset-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pose-min-visibility", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--landmark-space", choices=("world",), default="world")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--robot", action="append", choices=loadable_robots(), default=None,
                        help="explicit target (required at run time; repeat for comparison)")
    parser.add_argument("--no-arms", action="store_true",
                        help="baseline behaviour: drive the hands only, arms stay held")
    # Accepted only so existing launcher argument lists still parse. Wrist
    # observations now drive orientation automatically; no enable switch.
    parser.add_argument("--arm-orientation", choices=("off", "auto", "on"), default="auto",
                        help=argparse.SUPPRESS)
    parser.add_argument("--arm-elbow-follow", choices=("off", "operator"), default="off",
                        help='IK null-space reference: off uses --arm-rest-poses; operator maps operator elbow bend to each robot elbow joint.')
    parser.add_argument("--arm-axis-order", default=None, metavar="ZXY",
                        help="robot axes taken from operator axes (default from arm_realtime)")
    parser.add_argument("--arm-axis-signs", default=None, metavar="-1,1,-1")
    parser.add_argument("--arm-gain", nargs=2, action=ArmGainAction, default=None,
                        metavar=("ROBOT", "GAIN"), help="override metric_scale for one robot")
    parser.add_argument("--arm-map", choices=MAP_MODES, default="posture",
                        help='Arm mapping: posture (default) maps upper/forearm directions using robot link lengths; chain maps shoulder-to-wrist displacement; segments scales each segment separately. Joint limits and IK constrain reach.')
    parser.add_argument("--arm-elbow-task", action="store_true",
                        help='Add an elbow position target for segments mapping. Posture mapping already supplies one. Off by default.')
    parser.add_argument("--arm-elbow-pass", choices=ELBOW_PASS_ORDERS, default="last",
                        help='IK pass carrying the elbow position target: first or last (default last).')
    parser.add_argument("--arm-elbow-gain", type=float, default=1.0, metavar="0-1",
                        help='Fraction of upper-arm displacement used for the segments elbow target (default 1.0).')
    parser.add_argument("--arm-posture-gain", type=float, default=1.0, metavar="0-1",
                        help='Direction interpolation gain for posture mapping: 1 follows mapped directions, 0 stays at robot neutral (default 1).')
    parser.add_argument("--arm-posture-frame", choices=("auto", "mapped", "delta"),
                        default="auto",
                        help='Posture direction frame: auto uses mapped for h1_2/g1 and delta for gr1_t2; mapped uses axis-mapped directions; delta applies operator neutral-relative rotations to robot neutral directions.')
    parser.add_argument("--arm-ik-scope", choices=("arm", "body"), default="arm",
                        help='IK variables: arm (default) pins non-driven joints; body includes all movable joints.')
    parser.add_argument("--arm-rest-poses", choices=("current", "zero", "live", "loaded"),
                        default="current",
                        help='IK null-space reference: current (default) reads actual joint angles; live is an alias; loaded uses the build-time pose; zero uses zero angles.')
    parser.add_argument("--arm-anchor", choices=("upper", "shoulder"), default="upper",
                        help='Upper-arm shoulder anchor: upper (default) is the elbow parent-link origin; shoulder is the name-matched shoulder link.')
    parser.add_argument("--arm-no-shoulder-follow", action="store_true",
                        help='Disable translating elbow/wrist targets with the current FK shoulder relative to its loaded anchor.')
    parser.add_argument("--arm-direction-rate-deg-s", type=float,
                        default=MAX_DIRECTION_RATE_DEG_S, metavar="DEG/S",
                        help='Maximum operator direction rate in degrees/second (default 600); hold rejected jumps for a bounded number of frames. 0 disables the guard.')
    parser.add_argument("--arm-elbow-recovery-deg", type=float, default=ELBOW_RECOVERY_DEG,
                        metavar="DEG",
                        help='Direction-error threshold for retrying IK from the loaded pose (default 25 degrees, 0 disables recovery).')
    parser.add_argument("--arm-elbow-recovery-margin-deg", type=float,
                        default=ELBOW_RECOVERY_MARGIN_DEG, metavar="DEG",
                        help='Required direction-error improvement before switching IK branch (default 8 degrees).')
    parser.add_argument("--arm-elbow-recovery-frames", type=int,
                        default=ELBOW_RECOVERY_COOLDOWN_FRAMES, metavar="N",
                        help='Frames to wait after each branch recovery attempt (default 12, 0 retries every frame).')

    parser.add_argument("--arm-travel-mm", type=float, default=None,
                        help="override the per-axis travel clamp for every robot (mm)")
    parser.add_argument("--arm-travel-mode", choices=TRAVEL_MODES, default="clip",
                        help='Displacement travel limiting: clip clamps each axis independently; scale shrinks the complete displacement to fit. Does not apply to posture mapping.')
    parser.add_argument("--arm-lateral-guard-mm", type=float,
                        default=1000.0 * DEFAULT_LATERAL_GUARD_M, metavar="MM",
                        help='Keep wrist targets on their own side: left y >= MM, right y <= -MM (default 0 disables the guard).')
    parser.add_argument("--no-arm-gain-quality-guard", action="store_true",
                        help='Disable plausibility and side-symmetry checks for calibrated operator arm lengths.')
    parser.add_argument("--arm-slew-mm-s", type=float, default=DEFAULT_ARM_SLEW_MM_PER_S,
                        metavar="MM/S",
                        help='Wrist target speed limit in mm/s (default 600), converted using capture intervals and capped at 40 mm per step. 0 disables slew limiting.')
    parser.add_argument("--arm-slew-mm", type=float, default=None, metavar="MM",
                        help='Fixed target step limit in mm per frame, overriding --arm-slew-mm-s.')
    parser.add_argument("--arm-slew-pair-floor", type=float, default=None, metavar="RATIO",
                        help='Minimum retained length ratio for a rigid elbow/wrist target pair; hold the previous pair below this ratio (default 0.6, 0 disables the guard).')
    parser.add_argument("--arm-ik-iterations", type=int, default=60,
                        help='PyBullet IK maximum iterations per pass (default 60).')
    parser.add_argument("--arm-ik-residual-threshold", type=float, default=1e-4,
                        help="PyBullet IK residualThreshold per pass (default 1e-4)")
    parser.add_argument("--arm-calibrate-frames", type=int, default=30,
                        help="neutral frames per side before the neutral is final (default 30). "
                             "With --arm-calibrate-mode hold the arms also stay on the working "
                             "point until then; with track (default) they follow from the first "
                             "frame that sees your arm")
    parser.add_argument("--arm-calibrate-mode", choices=CALIBRATE_MODES, default="track",
                        help='Neutral calibration: hold waits for a complete window; track (default) follows using the available sample median while the window converges.')
    parser.add_argument("--neutral-json", type=Path, default=None,
                        help="reuse a neutral pose saved by --save-neutral")
    parser.add_argument("--save-neutral", type=Path, default=None,
                        help="write this run's neutral pose so the next run can reuse it")
    parser.add_argument("--arm-trace", action="store_true",
                        help="print the operator-side delta per axis (use this to check the axes)")
    parser.add_argument("--dump-npz", type=Path, default=None,
                        help="record hands-only debug frames; it is not an arm replay format")
    parser.add_argument("--arm-frame-dump", type=Path, default=None,
                        help="record every frame's arm samples (shoulder/upper/elbow/wrist in the "
                             "robot base frame plus the command that produced them) to an .npz; "
                             "devtools/show_arm_direction.py turns two of those into A/B figures")
    parser.add_argument("--input-npz", type=Path, default=None,
                        help="replay a recorded session instead of the camera (no camera needed)")
    parser.add_argument('--input-body', type=Path, help='replay a new SMPL-X H5 recording, without camera or model')
    parser.add_argument('--dump-body', type=Path, help='save shared RGB, raw SMPL-X, Hands and control observations')
    parser.add_argument("--sim-steps", type=int, default=8,
                        help="physics steps per offline frame or with --fixed-sim-steps (default 8)")
    parser.add_argument("--fixed-sim-steps", action="store_true",
                        help="use fixed physics steps for live capture too; default follows elapsed time")
    parser.add_argument("--view", choices=("full", "front"), default="full")
    parser.add_argument("--g1-color", type=float, nargs=3, default=(0.65, 0.80, 0.95),
                        metavar=("R", "G", "B"), help="G1 display RGB, each in [0, 1]")
    parser.add_argument("--original-g1-color", action="store_true",
                        help="keep the original G1 URDF materials")
    parser.add_argument("--light-position", type=float, nargs=3, default=(2., -4., 6.),
                        metavar=("X", "Y", "Z"), help="PyBullet GUI light position")
    parser.add_argument("--shadows", action="store_true",
                        help="enable GUI shadows (off by default for visibility)")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames; 0 runs until Ctrl+C or window close")
    parser.add_argument("--headless", action="store_true")
    add_hand_command_arguments(parser)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-no-mirror", action="store_true")
    parser.add_argument("--preview-window-pos", type=_window_pos, default=(20, 60),
                        metavar="X,Y", help="screen position for the camera window (or 'none')")
    parser.add_argument("--preview-no-topmost", action="store_true")
    parser.add_argument("--preview-dump-dir", type=Path, default=None)
    parser.add_argument("--preview-dump-frames", type=int, default=5)
    parser.add_argument("--no-arm-overlay", action="store_true",
                        help="do not draw the arm skeleton/numbers in the camera window "
                             "(the window then matches the baseline hand-only preview)")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def arm_hud(bridge, driver, targets) -> str:
    """HUD 里的手臂段：中性标定进度 + 每侧来源（P/H/-）+ 每台机器人的末端残差。"""
    if bridge is None:
        return "arm=off (hands only)"
    per_side = next(iter(targets.values())) if targets else {}
    trace = bridge.trace()
    sources = " ".join(
        f"{side[0].upper()}={SOURCE_SHORT.get(getattr(target, 'source', 'none'), '?')}/"
        f"{trace.get(side, {}).get('tracking_state', 'hold') if target.held else 'tracking'}"
        for side, target in sorted(per_side.items()))
    residuals = ""
    if driver is not None:
        values = driver.residuals_mm()
        text = " ".join(f"{name}={max(entry.values()):.0f}"
                        for name, entry in sorted(values.items()) if entry)
        residuals = f" res(mm) {text}"
    clamped = [f"{name}/{side[0].upper()}" for name, entries in sorted(targets.items())
               for side, target in sorted(entries.items()) if target.info.get("clamped")]
    limit_text = "" if not clamped else " CLAMP=" + ",".join(clamped)

    pinned = sum((bridge.counts.get("clamped") or {}).values())
    guarded = sum((bridge.counts.get("guarded") or {}).values())
    tail = f" sat={pinned}" + ("" if not guarded else f" guard={guarded}")
    return f"{bridge.status()} arm {sources}{residuals}{limit_text}{tail}"


def arm_lines(bridge, sides) -> dict:
    """采集窗口里每侧一行手臂数字：来源字母 + 相对中性姿势的三轴位移（厘米）+ 臂长 + 肘角。

    数据就是 ``bridge.trace()``（``--arm-trace`` 打的同一份）；窗口只显示，不参与计算。
    ``bridge`` 为 ``None``（``--no-arms``）时给空字典，调用方不用分叉。
    """
    if bridge is None:
        return {}
    lines = {}
    for side, entry in sorted(bridge.trace().items()):
        if side not in sides:
            continue
        letter = SOURCE_SHORT.get(entry.get("source", "none"), "?")
        delta = entry.get("delta_op_cm")
        moved = ("--" if delta is None
                 else "[" + ",".join(f"{float(value):+5.1f}" for value in delta) + "]")
        reach = entry.get("operator_reach_m")
        elbow = entry.get("elbow_bend_deg")
        reason = str(entry.get("reason") or "")
        age = entry.get('age_s')
        age_text = '--' if age is None else f'{age:.2f}s'
        lines[side] = (f"{side} {letter} d(cm){moved} "
                       f"{entry.get('tracking_state', 'hold')} age={age_text} "
                       f"reach={'--' if reach is None else f'{float(reach):.3f}'}m "
                       f"elbow={'--' if elbow is None else f'{float(elbow):.0f}'}deg"
                       f"{'' if not reason else ' ' + reason[:44]}")
    return lines


def print_arm_state(processed, bridge, driver, targets, profiles) -> None:
    """Print operator displacement and per-robot arm targets for --arm-trace."""
    trace = bridge.trace()
    for side, entry in sorted(trace.items()):
        delta = entry["delta_op_cm"]
        text = ("n/a" if delta is None
                else "[" + ", ".join(f"{value:+6.1f}" for value in delta) + "]")
        print(f"  arm trace frame={processed} {side:<5} src={entry['source']:<4} "
              f"state={entry['tracking_state']} held={entry['held']} fresh={entry['fresh']} "
              f"age_s={entry['age_s']} sample={entry['sample_id']} "
              f"delta_op_cm={text} reach={entry['operator_reach_m']} "
              f"elbow={entry['elbow_bend_deg']} reason={entry['reason'] or '-'}",
              flush=True)
    for robot, per_side in sorted(targets.items()):
        fragments = []
        for side, target in sorted(per_side.items()):
            offset = profiles[robot].realtime_neutral(side)
            if target.position is None:
                fragments.append(f"{side}=--")
                continue
            delta = [round((target.position[axis] - offset[axis]) * 1000.0, 1)
                     for axis in range(3)]
            flag = " CLAMP" if target.info.get("clamped") else ""
            fragments.append(f"{side}=[{delta[0]:+5.0f},{delta[1]:+5.0f},{delta[2]:+5.0f}]mm"
                             f" {target.source}{flag}")
        print(f"  arm {robot:>7}: " + "  ".join(fragments), flush=True)


def _record_frame(dumped, raw_frame, arms) -> None:
    """Record hands-only debug frames; arm dumps are not a supported source."""
    for side in SIDES:
        hand = None if raw_frame is None else raw_frame.get(side)
        world = None if raw_frame is None else raw_frame.get(f"world_{side}")
        arm = None
        if arms is not None and arms.world_arms:
            arm = arms.world_arms.get(side)
        dumped["hand"][side].append(np.full((21, 3), np.nan, dtype=np.float32)
                                    if hand is None
                                    else np.asarray(hand, dtype=np.float32))
        dumped["world"][side].append(np.full((21, 3), np.nan, dtype=np.float32)
                                     if world is None
                                     else np.asarray(world, dtype=np.float32))
    dumped["timestamps"].append(float(0.0 if raw_frame is None
                                     else raw_frame.get("timestamp", 0.0)))


def _save_dump(path, dumped) -> Path:
    """把记录写成 npz；键名与 ``devtools/verify_realtime_arm_bridge.py`` 的读取口径一致。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        timestamps=np.asarray(dumped["timestamps"], dtype=np.float64),
        left_hand=np.stack(dumped["hand"]["left"]),
        right_hand=np.stack(dumped["hand"]["right"]),
        world_left_hand=np.stack(dumped["world"]["left"]),
        world_right_hand=np.stack(dumped["world"]["right"]))
    return target


def _collect_arm_frames(driver, robots, sides) -> dict:
    """Collect base-frame arm FK arrays while the physics client is connected."""
    arrays = {}
    for name in robots:
        adapter = driver.ik.get(name)
        if adapter is None:
            continue
        for side in sides:
            frames = list(adapter.posture.get(side) or [])
            points, command_elbows, command_wrists = [], [], []
            for frame in frames:
                points.append([adapter.to_base(frame[key])
                               for key in ("shoulder", "upper", "elbow", "wrist")])
                for key, sink in (("command_elbow", command_elbows),
                                  ("command_wrist", command_wrists)):
                    value = frame.get(key)
                    sink.append(np.full(3, np.nan, dtype=np.float64) if value is None
                                else np.asarray(value, dtype=np.float64))
            arrays[f"{name}_{side}_points"] = np.asarray(
                points, dtype=np.float64).reshape(-1, 4, 3)
            arrays[f"{name}_{side}_cmd_elbow"] = np.asarray(
                command_elbows, dtype=np.float64).reshape(-1, 3)
            arrays[f"{name}_{side}_cmd_wrist"] = np.asarray(
                command_wrists, dtype=np.float64).reshape(-1, 3)
    return arrays


def _save_arm_frames(path, arrays) -> Path:
    """把 ``_collect_arm_frames`` 读到的数组写成 npz（不碰 PyBullet，断连后也能写）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target




def run(args) -> int:
    """采集 -> 手 + 臂 -> 仿真；0 = 至少一侧给出过输出，1 = 全程没有手也没有臂。"""
    if not args.robot or len(set(args.robot))!=len(args.robot):
        raise ValueError('Choose distinct targets explicitly with --robot NAME (repeat for comparison)')
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    diagnostics_enabled = bool(getattr(args, "diagnostics", False) or getattr(args, "diagnostic_trace", None))
    if diagnostics_enabled:
        trace_path = getattr(args, "diagnostic_trace", None)
        protected = [getattr(args, key, None) for key in
                     ("input_npz", "input_body", "dump_body", "model_asset_path", "smplerx_checkpoint", "smplx_model_dir", "neutral_json", "dump_npz", "save_neutral")]
        protected.extend(path for _model, path in (args.hand_checkpoint or []))
        destinations = [value for value in (trace_path, args.report) if value is not None]
        if len({Path(path).resolve() for path in destinations}) != len(destinations):
            raise ValueError("Diagnostic CSV and report paths must differ")
        if any(Path(out).resolve() == Path(source).resolve() for out in destinations
               for source in protected if source is not None):
            raise ValueError("Diagnostic output must not overwrite an input or another output")
    input_body = getattr(args, 'input_body', None)
    dump_body = getattr(args, 'dump_body', None)
    replaying = args.input_npz is not None or input_body is not None
    outputs = [Path(value).resolve() for value in (dump_body, args.dump_npz, args.report,
               getattr(args, 'diagnostic_trace', None), args.save_neutral, args.arm_frame_dump) if value is not None]
    inputs = [Path(value).resolve() for value in (input_body, args.input_npz,
              args.model_asset_path, args.neutral_json, args.smplerx_checkpoint) if value is not None]
    inputs.extend(Path(path).resolve() for _model, path in (args.hand_checkpoint or []))
    if len(set(outputs)) != len(outputs) or set(outputs).intersection(inputs):
        raise ValueError('Output must not overwrite an input or another output')
    if args.input_npz is not None and input_body is not None:
        raise ValueError('Choose either --input-body or --input-npz')
    if dump_body is not None and (replaying or args.no_arms):
        raise ValueError('--dump-body requires live SMPL-X arm capture')
    asset = resolve_model_asset(args.model_asset_path) if not replaying else None
    robots = tuple(args.robot)
    profiles = {name: profile_for(name) for name in robots}
    arms_wanted = not args.no_arms
    if args.pose_model_asset_path is not None or args.pose_min_visibility is not None:
        raise ValueError("MediaPipe Pose arm options were removed; configure SMPLer-X")
    if args.input_npz is not None and arms_wanted:
        raise ValueError("Legacy NPZ arm replay is removed; replay an SMPL-X body recording")
    if not replaying and arms_wanted:
        for name in ("smplerx_worker_python", "smplerx_root", "smplerx_checkpoint", "smplx_model_dir"):
            value = getattr(args, name, None)
            if value is None:
                raise ValueError(f"--{name.replace('_', '-')} is required when arms are enabled")
            if not Path(value).exists():
                raise FileNotFoundError(f"{name} not found: {value}")
    if args.dump_npz is not None and arms_wanted:
        raise ValueError("--dump-npz is a hands-only debug format; use SMPL-X body recording for arms")
    checkpoints = {}
    for model, path in args.hand_checkpoint or []:
        if model not in robots or model in checkpoints:
            raise ValueError("Each --hand-checkpoint must name a distinct active robot")
        checkpoints[model] = Path(path)
    specs, retargeter = build_camera_retargeter(
        robots, sides, args.backend, checkpoints=checkpoints or None, device=args.device)
    backend = args.backend
    note = "target-native independent joints; palm-length normalized camera input"
    scene = camera = preview = driver = bridge = diagnostics = None
    transport = None
    wrist_control = None
    wrist_summary = None
    fixed_slew_m = None
    arm_failures = {}
    body_rows = []
    body_metadata = None
    processed = 0
    started = None
    marks = []
    samples = {side: [] for side in sides}
    invalid = {side: 0 for side in sides}
    raw_seen = {side: 0 for side in sides}
    native_samples = {key: [] for key in specs}
    native_invalid = {key: 0 for key in specs}
    dumped = {"hand": {side: [] for side in SIDES},
              "world": {side: [] for side in SIDES},
              "timestamps": []}
    targets = {}
    try:
        if diagnostics_enabled:
            from teleoperation.simulation.arm_diagnostics import ArmDiagnostics
            diagnostics = ArmDiagnostics(getattr(args, "diagnostic_trace", None))
        args.scene = "robots"                      # 手臂只存在于机器人场景
        scene = build_scene(args, sides)
        commands = JointCommandPipeline(specs, args.joint_filter, args.filter_alpha,
                                        args.joint_max_velocity, args.kalman_q, args.kalman_r)
        appearance = apply_appearance(
            scene, None if args.original_g1_color else args.g1_color,
            args.light_position, args.shadows)
        if hasattr(scene, "frame_scene"):
            scene.frame_scene(args.view)
        if arms_wanted:
            fixed_slew_m = (None if args.arm_slew_mm is None
                            else max(0.0, float(args.arm_slew_mm)) / 1000.0)
            slew_speed_m_per_s = (0.0 if fixed_slew_m is not None
                                  else max(0.0, float(args.arm_slew_mm_s)) / 1000.0)
            driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                    ik_iterations=args.arm_ik_iterations,
                                    residual_threshold=args.arm_ik_residual_threshold,
                                    rest_poses=args.arm_rest_poses,
                                    diagnostics=diagnostics,
                                    slew_m=(0.0 if fixed_slew_m is None else fixed_slew_m),
                                    slew_m_per_s=slew_speed_m_per_s,
                                    slew_pair_floor_ratio=args.arm_slew_pair_floor,
                                    elbow_pass=args.arm_elbow_pass,
                                    elbow_recovery_deg=args.arm_elbow_recovery_deg,
                                    elbow_recovery_margin_deg=args.arm_elbow_recovery_margin_deg,
                                    elbow_recovery_cooldown=args.arm_elbow_recovery_frames,
                                    shoulder_anchor_key=args.arm_anchor,
                                    lock_others=(args.arm_ik_scope == "arm"))
            robot_links, robot_anchors = {}, {}
            if args.arm_map in ("segments", "posture"):
                from teleoperation.applications.camera_scene_arm import measure_camera_arm_geometry
                robot_links, robot_anchors = measure_camera_arm_geometry(driver, args.arm_anchor)
                if not args.arm_no_shoulder_follow:
                    # 肘/腕目标按「当前肩 - 加载锚点」平移（见 ArmSceneDriver._shoulder_shift）：
                    # 躯干被手臂反作用力推着走时，目标仍然落在「离当前肩正好一根上臂」的球面上。
                    driver.set_shoulder_anchors(robot_anchors)
                elbow_note = ("elbow task: built by the direction recipe (pass "
                              f"{args.arm_elbow_pass})"
                              if args.arm_map == "posture" else
                              f"elbow task {'on' if args.arm_elbow_task else 'off'}")
                pass_note = ("" if args.arm_map == "posture" else
                             (f" (pass {args.arm_elbow_pass})"
                              if args.arm_elbow_task else ""))
                frame_note = ("" if args.arm_posture_frame != "auto" else
                              " (" + ", ".join(f"{name}:{profiles[name].posture_frame}"
                                               for name in robots) + ")")

                print(f"arm map: {args.arm_map} | shoulder end = {args.arm_anchor} "
                      f"({'rigid: the elbow parent link' if args.arm_anchor == 'upper' else 'historical name-matched link'})"
                      f" | ik scope = {args.arm_ik_scope}"
                      f" | rest poses = {args.arm_rest_poses}"
                      f" | elbow follow = {args.arm_elbow_follow}"
                      f" | posture frame = {args.arm_posture_frame}{frame_note}"
                      f" | measured links (m) "
                      + " ".join(f"{name}:{side[0]}" +
                                 f" {robot_links[name][side]['upper']:.3f}"
                                 f"+{robot_links[name][side]['fore']:.3f}"
                                 for name in robots for side in sides)
                      + f" | {elbow_note}"
                      + pass_note
                      + (f" | elbow branch recovery: >{args.arm_elbow_recovery_deg:.0f} deg"
                         f" (margin {args.arm_elbow_recovery_margin_deg:.0f} deg,"
                         f" retry every {args.arm_elbow_recovery_frames} frames)"

                         if args.arm_elbow_recovery_deg > 0.0 else
                         " | elbow branch recovery: off"),
                      flush=True)
            else:
                print(f"arm map: chain (no elbow task: a wrist target only) | ik scope = "
                      f"{args.arm_ik_scope} | in this recipe the upper arm cannot move on its "
                      f"own and the per-axis travel clamp caps the arm (g1 40 mm); pass "
                      f"--arm-map posture for the direction recipe", flush=True)
            gains = {str(pair[0]): float(pair[1]) for pair in (args.arm_gain or [])}
            axis_map = AxisMap()
            if args.arm_axis_order and args.arm_axis_signs:
                axis_map = AxisMap.parse(args.arm_axis_order, args.arm_axis_signs)
            elif args.arm_axis_order:
                axis_map = AxisMap.parse(args.arm_axis_order,
                                         ",".join(f"{sign:+.0f}" for sign in axis_map.signs))
            elif args.arm_axis_signs:
                axis_map = AxisMap.parse("".join(axis_map.order), args.arm_axis_signs)
            neutral = load_neutral(args.neutral_json) if args.neutral_json else None
            bridge = RealtimeArmBridge(
                [profiles[name] for name in robots], sides=sides, axis_map=axis_map,
                calibration_frames=args.arm_calibrate_frames,
                calibrate_mode=args.arm_calibrate_mode,
                orientation="off", gains=gains,
                max_travel_m=(None if args.arm_travel_mm is None
                              else float(args.arm_travel_mm) / 1000.0),
                travel_mode=args.arm_travel_mode,
                lateral_guard_m=max(0.0, float(args.arm_lateral_guard_mm)) / 1000.0,
                gain_quality_guard=not args.no_arm_gain_quality_guard,
                neutral=neutral,
                map_mode=args.arm_map, robot_links=robot_links,
                elbow_task=args.arm_elbow_task, elbow_gain=args.arm_elbow_gain,
                posture_gain=args.arm_posture_gain,
                posture_frame=(None if args.arm_posture_frame == "auto"
                               else args.arm_posture_frame),
                max_direction_rate_deg_s=args.arm_direction_rate_deg_s,
                anchors=robot_anchors)
            from teleoperation.simulation.wrist_control import WristSceneController
            wrist_control = WristSceneController(driver, axis_map.matrix())
            print(f"arms: {', '.join(robots)} | ik_mode "
                  f"{({name: profiles[name].ik_mode for name in robots})} | "
                  f"orientation input wrist (automatic) | travel limit "
                  f"{bridge.travel_limits()} m ({args.arm_travel_mode}) | "
                  f"lateral guard {args.arm_lateral_guard_mm} mm | "
                  f"neutral mode {args.arm_calibrate_mode} | "
                  f"neutral {'reused from ' + str(args.neutral_json) if neutral else str(args.arm_calibrate_frames) + ' frames'} | "
                  + (f"target slew {fixed_slew_m * 1000.0:.1f} mm/frame (fixed)"
                     if fixed_slew_m is not None else
                     f"target slew {args.arm_slew_mm_s:.0f} mm/s "
                     f"(ceiling {SLEW_MAX_STEP_M * 1000.0:.0f} mm/frame)")
                  + f" | pair floor {driver.slew_pair_floor_ratio:.2f}"
                  + ("" if driver.slew_pair_floor_ratio > 0.0 else " (off)"),
                  flush=True)
        if input_body is not None:
            camera = BodyReplayFrameSource(input_body, args.landmark_space)
        elif args.input_npz is not None:
            camera = ReplayFrameSource(args.input_npz, args.landmark_space)
            print(f"replay: {args.input_npz} ({camera.frames} frames, "
                  f"landmark_space={args.landmark_space}); no camera is opened", flush=True)
        else:
            transport = None
            client = None
            if arms_wanted:
                transport = SmplerXWorkerTransport(
                    worker_python=args.smplerx_worker_python,
                    smplerx_root=args.smplerx_root,
                    checkpoint=args.smplerx_checkpoint,
                    body_model_dir=args.smplx_model_dir,
                    model=args.smplerx_model,
                )
                client = SmplerXClient(transport, session_id=f"realtime-{time.monotonic_ns()}")
            camera = CameraFrameSource(asset, args.camera_index, args.width, args.height,
                                       args.fps, args.landmark_space,
                                       client, client.session_id if client is not None else "camera",
                                       arms_wanted)
        preview = build_arm_preview(args, camera, sides)
        camera.diagnostics = diagnostics
        if not args.headless:
            print(PREVIEW_HINT_ROBOTS, flush=True)
            print(f"camera window {ARM_PREVIEW_WINDOW!r}: raw frame + 21 hand points + both "
                  f"arms (shoulder/elbow/wrist); the PyBullet window shows the three robots.",
                  flush=True)
        if bridge is not None:
            if args.arm_calibrate_mode == "track":
                print("keep a neutral arm pose (elbows slightly bent, hands in front of the "
                      f"chest) for about {args.arm_calibrate_frames} frames per side; the arms "
                      "already follow from the first frame that sees your arm, and the neutral "
                      "keeps converging while you hold still.", flush=True)
            else:
                print("keep a neutral arm pose (elbows slightly bent, hands in front of the "
                      f"chest) for about {args.arm_calibrate_frames} frames per side; until "
                      "then the arms hold their working point.", flush=True)
        started = time.perf_counter()
        simulation_clock = SimulationStepClock(args.sim_steps,
            realtime=not replaying and not args.fixed_sim_steps)
        from teleoperation.applications.accepted_robot_output import AcceptedRobotOutput
        robot_execution=AcceptedRobotOutput.native(scene,robots,args.side,driver)

        while True:
            if args.frames and processed >= args.frames:
                break
            if not scene.alive():
                print("PyBullet window closed; stopping.", flush=True)
                break
            if diagnostics is not None:
                diagnostics.begin_frame()
                capture_started = time.perf_counter()
            window = camera.next_frame()
            if getattr(camera, "exhausted", False):
                print("replay finished (all recorded frames were replayed).", flush=True)
                break
            processed += 1
            if dump_body is not None:
                from teleoperation.data.body_recording import capture_sample
                source = camera._source
                body_rows.append(capture_sample(source, camera.last_observation))
                if source.last_result is not None:
                    result = source.last_result
                    body_metadata = dict(source='smplerx', model_id=result.model_id,
                                         model_hash=result.model_hash, time_unit='monotonic_s')
            if diagnostics is not None:
                capture_ms = (time.perf_counter() - capture_started) * 1000
                diagnostics.record_ms("capture_detection", max(0., capture_ms - diagnostics.frame_timings.get("hand_preprocessing", 0.)))
                hand_started = time.perf_counter()
            raw_frame = camera.last_raw_frame
            for side in sides:
                if raw_frame is not None and raw_frame.get(side) is not None:
                    raw_seen[side] += 1
            latest = {side: None for side in sides}
            command_timestamp = float(raw_frame['timestamp']) if replaying else None
            results = {} if window is None else retargeter.retarget(window)
            side_values = {side: [] for side in sides}
            side_valid = {side: True for side in sides}
            accepted_hands={}
            for key, spec in specs.items():
                result = results.get(key)
                command = (commands.missing(key, timestamp=command_timestamp) if result is None
                           else commands.update(result, timestamp=command_timestamp))
                accepted_hands[key]=command
                if result is None or not result.valid:
                    native_invalid[key] += 1
                    side_valid[key[1]] = False
                else:
                    native_samples[key].append(result.positions.copy())
                side_values[key[1]].extend(command.positions)
            for side in sides:
                if side_valid[side] and side_values[side]:
                    latest[side] = np.asarray(side_values[side])
                    samples[side].append(latest[side])
                else:
                    invalid[side] += 1
            targets = {}
            accepted_arms={}
            if diagnostics is not None:
                diagnostics.record_ms("hand_inference_output", (time.perf_counter() - hand_started) * 1000)
            if bridge is not None:
                frame_dt = (args.sim_steps / 240.0 if args.input_npz is not None else
                            getattr(camera, 'frame_dt', None) or NOMINAL_FRAME_DT)
                from teleoperation.applications.camera_scene_arm import solve_camera_arm_frame
                def arm_error(side, error):
                    if side not in arm_failures:
                        arm_failures[side] = f'{type(error).__name__}: {error}'
                        print(f'[WARN] {side}: arm command held: {arm_failures[side]}', flush=True)
                with (nullcontext() if diagnostics is None else diagnostics.stage('arm_mapping')):
                    targets,accepted_arms = solve_camera_arm_frame(bridge, driver, wrist_control, camera.arms(),
                        camera.wrists(), frame_dt, elbow_follow=args.arm_elbow_follow, on_error=arm_error)
            control_seconds=(processed-1)*args.sim_steps/240. if replaying else time.perf_counter()-started
            stamp=None if raw_frame is None else float(raw_frame['timestamp'])
            observed={(model,side+'_hand'):key in results and results[key].valid
                      for key in specs for model,side in (key,)}
            body=camera.arms()
            for model in robots:
                for side in sides:
                    observed[model,side+'_arm']=body is not None and body.valid.get(side,False)
            with (nullcontext() if diagnostics is None else diagnostics.stage('physics')):
                robot_execution.submit(processed-1,control_seconds,hands=accepted_hands,arms=accepted_arms,
                    source_timestamp=stamp,observed=observed,steps=simulation_clock.next_steps())
            if wrist_control is not None:wrist_control.record()
            if args.dump_npz is not None:
                _record_frame(dumped, raw_frame, camera.arms())
            if preview is not None:
                preview_started = time.perf_counter() if diagnostics is not None else None
                body = camera.arms()
                # 不在这里挑手臂点：arm_preview.update() 自己把形状不对 / 含 NaN 的一侧
                # 当作"没有"（与桥的丢帧保持同口径），这里只负责把 None 原样传下去。
                wrist_lines = arm_lines(bridge, sides)
                if wrist_control is not None:
                    for side in sides:
                        wrist_lines[side] = wrist_lines.get(side, "") + " wrist=" + wrist_control.trackers[side].status
                preview.update(raw_frame, latest, arm_hud(bridge, driver, targets),
                               arms=None if body is None else body.arms,
                               lines=wrist_lines)
                if diagnostics is not None:
                    diagnostics.record_ms("preview", (time.perf_counter() - preview_started) * 1000)
            if diagnostics is not None:
                diagnostics.capture(processed - 1, driver, bridge, targets, arms=camera.arms(),
                    raw_timestamp=None if raw_frame is None else raw_frame.get("timestamp"),
                    timestamp_unit="monotonic_s" if args.input_npz is None else "relative_seconds",
                    sample_time_s=(camera._source.last_time_s - camera._start_timestamp if input_body is not None
                                   else time.perf_counter() - started if args.input_npz is None else None))
            if preview is not None and preview.quit_requested:
                print("camera window closed; stopping.", flush=True)
                break
            marks.append(time.perf_counter())
            if args.print_every and processed % args.print_every == 0:
                counts = " ".join(f"{side}={len(samples[side])}" for side in sides)
                print(f"frame={processed} fps={_loop_fps(marks):.1f} hand valid={counts} "
                      f"invalid={sum(invalid.values())} | {arm_hud(bridge, driver, targets)}",
                      flush=True)
                if args.arm_trace and bridge is not None:
                    print_arm_state(processed, bridge, driver, targets, profiles)
        final = {side: scene.joint_values(side) for side in sides}
        # 注意：这些都要在 finally 断开之前读 —— PyBullet 断连后再 getLinkState 会抛
        # "Not connected to physics server"。
        arm_residuals = {} if driver is None else driver.residuals_mm()
        arm_hold_drift = {} if driver is None else driver.hold_drift()
        arm_posture = {} if driver is None else driver.posture_summary()
        arm_gap = {} if driver is None else driver.forearm_gap_summary()
        arm_summary = None if driver is None else driver.summary()
        wrist_summary = None if wrist_control is None else wrist_control.summary()
        # --arm-frame-dump 的数据也必须在这里读（to_base 要查基座位姿）：
        # 放在 finally 之后就是「断连后再问 PyBullet」，见 _collect_arm_frames 的说明。
        arm_frames = (None if (args.arm_frame_dump is None or driver is None)
                      else _collect_arm_frames(driver, robots, sides))
        scene_summary = scene.summary() if hasattr(scene, "summary") else None
        elapsed = time.perf_counter() - started
    finally:
        failure = sys.exc_info()[1]
        try:
            if diagnostics is not None:
                diagnostics.close()
        finally:
            if preview is not None:
                _drain_in_background(preview.close, "camera preview close")
            if camera is not None:
                _drain_in_background(camera.release, "camera release")
            elif transport is not None:
                transport.close()
            if scene is not None:
                _drain_in_background(scene.close, "PyBullet disconnect")
        if failure is not None and diagnostics is not None and args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"entry": ARM_ENTRY, "exit_reason": "error",
                "error": f"{type(failure).__name__}: {failure}", "diagnostics": diagnostics.summary()},
                indent=2, allow_nan=False), encoding="utf-8")
        if body_rows and body_metadata is not None:
            from teleoperation.data.body_recording import save_body_recording
            save_body_recording(dump_body, body_rows, body_metadata)

    fps = _loop_fps(marks)
    print(f"\nprocessed={processed} frames in {elapsed:.2f} s  fps={fps:.1f}  "
          f"median frame interval {_median_interval(marks) * 1000:.1f} ms")
    body_worker = (camera._source.worker_summary() if hasattr(camera, '_source')
                   and hasattr(camera._source, 'worker_summary') else None)
    if body_worker is not None and body_worker['enabled']:
        print(f"body worker: results={body_worker['results_received']} "
              f"valid={body_worker['valid_results']} reasons={body_worker['result_reasons']} "
              f"fault={body_worker['fault']}", flush=True)
    if scene_summary is not None:
        print(f"scene={scene_summary['scene']} sim_frames={scene_summary['sim_frames']} "
              f"applied={scene_summary['applied_frames']}", flush=True)
    for robot, entry in sorted(arm_residuals.items()):
        if entry:
            text = " ".join(f"{side}={value:.1f}" for side, value in sorted(entry.items()))
            print(f"  arm {robot}: wrist residual mm {text}", flush=True)
    for robot, per_side in sorted(arm_posture.items()):
        for side, entry in sorted(per_side.items()):
            if not entry or entry.get("frames", 0) < 2:
                continue
            bend = entry["elbow_bend_deg"]
            print(f"  arm {robot}/{side}: shoulder travel {entry['shoulder_travel_deg']:.1f} deg, "
                  f"upper arm swept {entry['upper_arm_dir_travel_deg']:.1f} deg, "
                  f"elbow moved {entry['elbow_span_mm']:.0f} mm, "
                  f"elbow bend {bend['min']}-{bend['max']} deg", flush=True)
    if arm_gap:
        text = " ".join(f"{robot}={value:.0f}" for robot, value in sorted(
            (arm_gap.get("min_mm") or {}).items()))
        print(f"  arm forearm gap (worst): {text} mm "
              f"[worst {arm_gap.get('worst_mm')} mm over {arm_gap.get('frames')} frames]",
              flush=True)
    if arm_hold_drift:
        print(f"  arm hold drift: {json.dumps(arm_hold_drift, ensure_ascii=False)}", flush=True)
    if bridge is not None:
        for side, entry in sorted((bridge.summary().get("arm_length_trust") or {}).items()):
            if not entry.get("trusted"):
                print(f"  arm gain guard [{side}]: {entry.get('reason')}; using "
                      f"profile.metric_scale instead of arm-length normalisation", flush=True)
        arm_guarded_total_print = sum((bridge.counts.get("guarded") or {}).values())
        if arm_guarded_total_print:
            print(f"  arm lateral guard: {arm_guarded_total_print} frames had a wrist target "
                  f"pushed back to its own side "
                  f"(limit {1000.0 * bridge.lateral_guard_m:.0f} mm)", flush=True)
    if preview is not None:
        print(f"camera preview: window={'on' if preview.show else 'off'} "
              f"mirror={'on' if preview.mirror else 'off'} "
              f"arm_overlay={'on' if getattr(preview, 'overlay', False) else 'off'} "
              f"dumped_png={preview.dumped}"
              f"{'' if preview.rect is None else f' rect={preview.rect}'}"
              f"{'' if not preview.note else ' note=' + preview.note}", flush=True)
    arm_valid_total = 0 if bridge is None else sum(bridge.counts["valid"].values())
    arm_clamped_total = 0 if bridge is None else sum(bridge.counts["clamped"].values())
    arm_guarded_total = 0 if bridge is None else sum(bridge.counts["guarded"].values())
    arm_gain_guarded_total = (0 if bridge is None
                              else sum(bridge.counts["gain_guarded"].values()))
    report = {
        "entry": ARM_ENTRY,
        "backend": backend,
        "hand_commands": commands.summary(),
        "landmark_space": args.landmark_space,
        "length_unit": "m" if args.landmark_space == "world" else "normalized",
        "backend_note": note,
        "input_profile": PALM_LENGTH_PROFILE,
        "hand_targets": {"/".join(key): {
            "dof": spec.dof, "joint_names": list(spec.independent_joint_names),
            "movable_joint_names": list(spec.joint_names), "spec_hash": spec.spec_hash,
            "valid_frames": len(native_samples[key]), "invalid_frames": native_invalid[key],
            "angles": _describe(native_samples[key], spec.independent_joint_names) if native_samples[key] else None,
        } for key, spec in specs.items()},
        "model_asset": None if asset is None else str(asset),
        "smplerx_model": None if not arms_wanted else args.smplerx_model,
        "smplerx_checkpoint": None if not arms_wanted or replaying else str(Path(args.smplerx_checkpoint).resolve()),
        "input_body": None if input_body is None else str(input_body),
        "body_source": camera._source.metrics.summary() if hasattr(camera, '_source') else None,
        "body_worker": body_worker,
        "camera_index": args.camera_index,
        "input_npz": None if args.input_npz is None else str(args.input_npz),
        "headless": bool(args.headless),
        "scene": args.scene,
        "scene_summary": scene_summary,
        "appearance": appearance,
        "arms_enabled": bool(arms_wanted),
        "robots": list(robots),
        "arm_failures": arm_failures,
        "simulation_timing": {
            "mode": "elapsed" if simulation_clock.realtime else "fixed",
            "physics_steps": simulation_clock.total_steps,
            "simulated_seconds": simulation_clock.total_steps / 240,
            "capped_frames": simulation_clock.capped_frames,
        },
        "arm_targets_valid_frames": arm_valid_total,
        "arm_clamped_frames": arm_clamped_total,
        "arm_guarded_frames": arm_guarded_total,
        "arm_gain_guarded_frames": arm_gain_guarded_total,
        "arm_travel_mode": args.arm_travel_mode,
        "arm_lateral_guard_mm": float(args.arm_lateral_guard_mm),
        "arm_slew_mm_per_s": (None if fixed_slew_m is not None
                              else float(args.arm_slew_mm_s)),
        "arm_slew_mm_fixed": (None if fixed_slew_m is None else fixed_slew_m * 1000.0),
        "arm_slew_ceiling_mm": SLEW_MAX_STEP_M * 1000.0,
        "arm_slew_pair_floor_ratio": (None if driver is None
                                      else round(driver.slew_pair_floor_ratio, 4)),
        "arm_posture_frame": args.arm_posture_frame,
        "arm_slew_mm_last_frame": (None if driver is None
                                   else round(1000.0 * driver.slew_m, 3)),
        "arm": arm_summary,
        "wrists": wrist_summary,
        "arm_posture": arm_posture,
        "arm_forearm_gap_mm": arm_gap,
        "bridge": (None if bridge is None else bridge.summary()),
        "arm_last_targets": {
            robot: {side: target.describe() for side, target in sorted(per_side.items())}
            for robot, per_side in sorted(targets.items())},
        "preview": None if preview is None else {
            "window": bool(preview.show),
            "mirrored": bool(preview.mirror),
            "arm_overlay": bool(getattr(preview, "overlay", False)),
            "dumped_png": int(preview.dumped),
            "dump_dir": None if preview.dump_dir is None else str(preview.dump_dir),
            "window_name": preview.window_name,
            "window_pos": None if preview.window_pos is None else list(preview.window_pos),
            "window_rect": None if preview.rect is None else list(preview.rect),
            "topmost": bool(preview.topmost),
            "note": preview.note,
        },
        "processed_frames": processed,
        "elapsed_seconds": round(float(elapsed), 3),
        "fps": round(float(fps), 2),
        "median_frame_interval_ms": round(float(_median_interval(marks) * 1000), 2),
        "sides": {
            side: {
                "raw_detected_frames": raw_seen[side],
                "valid_frames": len(samples[side]),
                "invalid_frames": invalid[side],
                "angles": _describe(samples[side], [f"{model}.{name}" for (model, target_side), spec in specs.items()
                                                           if target_side == side for name in spec.joint_names]) if samples[side] else None,
                "final_joint_values": final.get(side),
            }
            for side in sides
        },
    }
    for side in sides:
        entry = report["sides"][side]
        print(f"  hand {side}: raw={entry['raw_detected_frames']}/{processed} "
              f"valid={entry['valid_frames']} invalid={entry['invalid_frames']}")
        if processed >= 30 and entry["raw_detected_frames"] < processed * 0.1:
            print(f"[WARN] {side} hand detection is below 10%: keep the whole hand large "
                  "enough in the color image, with visible fingers and good lighting.", flush=True)
        elif processed >= 30 and entry["valid_frames"] == 0:
            print(f"[WARN] {side} hand was detected but produced no angles: check valid "
                  "valid three-frame windows and metric world landmarks.", flush=True)
    if bridge is not None:
        trace = bridge.trace()
        for side, entry in sorted(trace.items()):
            print(f"  arm  {side}: source={entry['source']} "
                  f"delta_op_cm={entry['delta_op_cm']} reach_m={entry['operator_reach_m']} "
                  f"elbow_deg={entry['elbow_bend_deg']}")
    if bridge is not None:
        held_reasons = bridge.counts["held_reason"]
        if any(sum(kinds.values()) for kinds in held_reasons.values()):
            shown = []
            for side, kinds in sorted(held_reasons.items()):
                counts = ", ".join(f"{kind}={count}" for kind, count in kinds.items() if count)
                shown.append(f"{side} [{counts or 'none'}]")
            print("  arm  held frames: " + " | ".join(shown), flush=True)
        if bridge.calibrating:
            if args.arm_calibrate_mode == "track":
                # track 下「中性没攒够」不是故障：手臂已经在跟手，只是基准还是临时中位数。
                print(f"[WARN] the neutral is still a temporary median ({bridge.status()}): the "
                      f"arms follow anyway (--arm-calibrate-mode track); keep both arms visible "
                      f"and hold still to finish it, or rerun with --arm-calibrate-mode hold to "
                      f"gate on the full neutral.", flush=True)
            else:
                print(f"[WARN] the neutral calibration never completed ({bridge.status()}): "
                      "check the held-frame breakdown above and keep both arms visible in the "
                      "camera window.", flush=True)
    if args.save_neutral and bridge is not None:
        if bridge.calibrating:
            print("[WARN] --save-neutral skipped: the neutral calibration never completed",
                  flush=True)
        else:
            path = save_neutral(args.save_neutral, bridge.neutral_record(),
                                {"entry": ARM_ENTRY, "axis_map": bridge.axis_map.describe()})
            print(f"neutral pose saved: {path}", flush=True)
    if args.dump_npz is not None and dumped["timestamps"]:
        print(f"raw frames dumped: {_save_dump(args.dump_npz, dumped)} "
              f"({len(dumped['timestamps'])} frames)", flush=True)
    if arm_frames is not None:
        print(f"arm frames dumped: {_save_arm_frames(args.arm_frame_dump, arm_frames)}",
              flush=True)
    if args.report:
        if diagnostics is not None:
            report["diagnostics"] = diagnostics.summary()
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {path}")
    if diagnostics is not None:
        print(f"diagnostics: {diagnostics.frames} frames, {diagnostics.rows} arm rows; host processing timings only", flush=True)
    if not any(samples[side] for side in sides) and arm_valid_total == 0:
        if any(raw_seen[side] for side in sides):
            print("[FAIL] MediaPipe detected a hand but produced no angles: put the whole hand "
                  "in the middle of the frame with valid metric world landmarks for three frames.")
        else:
            print("[FAIL] no raw hand detection at all: check the camera window - the hand must "
                  "really be visible (lighting, distance, not only at the frame edge).")
        return 1
    return 0


def main(args=None) -> int:
    """CLI 边界：参数补齐 -> run()；异常在这里变成 ``[FAIL]`` + 退出码 1。"""
    if args is None:
        args = build_parser().parse_args()
    else:
        # 被 launcher / 测试用现成的 args 调用时，补齐它没声明的可选参数，
        # 保证两条调用方式行为一致（与基线入口同一套做法）。
        for name, value in vars(build_parser().parse_args([])).items():
            if not hasattr(args, name):
                setattr(args, name, value)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nCapture stopped by user.", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary reports instead of raising
        print(f"[FAIL] realtime teleop stopped: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
