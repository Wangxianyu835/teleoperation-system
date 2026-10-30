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
from teleoperation.contracts.hand_control import valid_joint_result
from teleoperation.retargeting.hand.angles import l21_legacy_values
from teleoperation.applications.realtime_hand_sim import (
    PREVIEW_HINT_ROBOTS,
    _describe,
    _drain_in_background,
    _loop_fps,
    _median_interval,
    _window_pos,
    build_retargeter,
    build_scene,
    resolve_model_asset,
)
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.inputs.smplerx import SmplerXClient
from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport
from teleoperation.applications.camera_teleop_source import CameraTeleopSource
from teleoperation.paths import realtime_checkpoint
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM, RUNTIME
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
from teleoperation.simulation.hand_commands import add_hand_command_arguments, build_hand_commands, apply_legacy_l21_mapping

SIDES = ("left", "right")
ARM_ENTRY = "realtime_teleop"
#: 摄像头循环名义帧长（秒）：第一帧还没有"上一帧"可比，斜坡按它换算，
#: 这样第一帧也是 20 mm/帧（旧口径），不会退化成"一步跳到工作点"。
NOMINAL_FRAME_DT = 1.0 / 30.0
#: 斜坡默认速率（mm/s）：600 mm/s 在 30 fps 下正好是 20 mm/帧（与 2026-10-13 实测口径同速），
#: 现场 10-12 fps 下按帧换算成 50-60 mm/帧、再被 SLEW_MAX_STEP_M（40 mm）压到 40 mm/帧 ——
#: 40 mm 是实测拐点（横摆传递 0.80，再快没有改善而残差变大）。见 docs/multi-robot-arm.md §8.9。
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
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space)
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
        raw_frame = {**raw.hands, "timestamp": raw.timestamp, "wrists": raw.wrists,
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
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space)
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
        self.pipeline = MediaPipePalmLocalPipeline(landmark_space)
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
        # 元数据不是装饰：队友的处理器在 world 口径下会校验
        # ``world_landmark_space`` / ``world_length_unit`` / ``source_scale``
        # （retargeting/hand/processing.py 的 process_frame），而 --dump-npz 只存数组，
        # 所以回放时按 inputs/mediapipe.py 的同一口径把这些声明补回来；
        # 少一项整场在第一帧就 ValueError（这是实测踩到的，见 docs 第 8 节）。
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
        from teleoperation.inputs.wrist import camera_wrists
        raw_frame["wrists"] = camera_wrists(
            {side: raw_frame.get("world_" + side) for side in SIDES}, timestamp)
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


def build_parser() -> argparse.ArgumentParser:
    """所有开关；默认值都往"能直接跑起来"调（world 米制、两侧、三台全上）。"""
    parser = argparse.ArgumentParser(
        prog="python -m teleoperation.simulation.realtime_teleop",
        description="Realtime camera -> hand angles + arm targets -> H1-2 / GR1-T2 / G1")
    parser.add_argument("--backend", choices=("auto", "geometric", "checkpoint", "dex"),
                        default="auto")
    parser.add_argument("--dex-calibration", type=Path, help="fixed camera dex calibration JSON")
    parser.add_argument("--diagnostics", action="store_true", help="read-only arm timings and state statistics")
    parser.add_argument("--diagnostic-trace", type=Path, help="stream diagnostic CSV; implies --diagnostics; must be a new file")
    parser.add_argument("--checkpoint", type=Path, default=None)
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
    parser.add_argument("--landmark-space", choices=("world", "normalized"), default="world")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--robot", action="append", choices=loadable_robots(), default=None,
                        help="drive only these robots (repeatable; default: all three)")
    parser.add_argument("--no-arms", action="store_true",
                        help="baseline behaviour: drive the hands only, arms stay held")
    # Accepted only so existing launcher argument lists still parse. Wrist
    # observations now drive orientation automatically; no enable switch.
    parser.add_argument("--arm-orientation", choices=("off", "auto", "on"), default="auto",
                        help=argparse.SUPPRESS)
    parser.add_argument("--arm-elbow-follow", choices=("off", "operator"), default="off",
                        help="null-space posture reference for the arm IK. 'off' (default) = the "
                             "historical behaviour: the build-time neutral, or --arm-rest-poses. "
                             "'operator' = hand the solver the operator's elbow bend, relative to "
                             "the neutral calibration, on this robot's own elbow joint "
                             "(PyBulletArmIK.posture_from_bend): PyBullet has no PostureTask, but "
                             "its restPoses term is the same null-space knob, and that term is what "
                             "pulls the elbow back to the load pose (measured 2026-10-14: an exactly "
                             "reachable wrist+elbow target pair came out 6.6-140.1 mm off with the "
                             "load pose as the reference). This reference moves with you and stays "
                             "consistent with the wrist+elbow tasks, so it cannot fight them; a zero "
                             "bend delta reproduces the historical behaviour exactly")
    parser.add_argument("--arm-axis-order", default=None, metavar="ZXY",
                        help="robot axes taken from operator axes (default from arm_realtime)")
    parser.add_argument("--arm-axis-signs", default=None, metavar="-1,1,-1")
    parser.add_argument("--arm-gain", nargs=2, action=ArmGainAction, default=None,
                        metavar=("ROBOT", "GAIN"), help="override metric_scale for one robot")
    parser.add_argument("--arm-map", choices=MAP_MODES, default="posture",
                        help="how operator motion becomes a target. 'posture' (default since "
                             "2026-10-14: the field round of 2026-10-13/14 accepted this one and "
                             "rejected both displacement recipes) = move the DIRECTION instead of "
                             "a displacement, with the robot's own measured bone lengths: "
                             "elbow = shoulder_anchor + L_u*M(unit(你的上臂)), wrist = elbow + "
                             "L_f*M(unit(你的前臂)). Your 40 deg shoulder swing is then 40 deg on "
                             "every robot, the per-axis travel clamp does not apply, and both "
                             "segment lengths are preserved; joint limits and IK still constrain reach. 'chain' (the historical default, "
                             "kept for A/B: pass it explicitly) = wrist minus shoulder, one gain. "
                             "It cannot move the shoulder when only the upper arm swings "
                             "(measured 0.0 mm of target travel on all three robots) and its "
                             "per-axis clamp caps the arm at 40-150 mm, which in the field looked "
                             "like 'the upper arm will not rise' and 'the arms will not open'; it "
                             "also hands the IK no elbow target, so the bending direction is the "
                             "solver's own choice (the GR1-T2 complaint). 'segments' = the "
                             "paper's scale_joints: one term per bone, scaled by the robot's own "
                             "measured link lengths, which is also what the elbow task needs; it "
                             "cancels again when the robot's two bone ratios match yours "
                             "(g1 0.193/0.184). See docs/multi-robot-arm.md 8.10 / 8.11 / 8.15")
    parser.add_argument("--arm-elbow-task", action="store_true",
                        help="also hand the IK an elbow position target (the paper's elbow "
                             "task; needs --arm-map segments; --arm-map posture builds the "
                             "elbow target by itself, so this flag is not needed there). "
                             "Off by default. Measured "
                             "0.01-0.06 deg of upper-arm motion without it (the redundant "
                             "7 DoF chain keeps its configuration, which is why the upper "
                             "arm 'does not move') vs 7-33 deg with it. Cost: the wrist "
                             "residual grows when the two targets disagree, so try it and "
                             "watch --arm-trace / the report")
    parser.add_argument("--arm-elbow-pass", choices=ELBOW_PASS_ORDERS, default="last",
                        help="which IK pass carries the elbow target (default 'last': with "
                             "'first' the wrist-pose finishing pass pulls the elbow back "
                             "again, measured elbow error 141-256 mm)")
    parser.add_argument("--arm-elbow-gain", type=float, default=1.0, metavar="0-1",
                        help="how much of the upper-arm displacement to hand the elbow "
                             "target (default 1.0 = the whole term). PyBullet gives the two "
                             "position targets equal weight, so this is the only knob on the "
                             "trade: measured 2026-10-13 on a real recording (120 frames, "
                             "h1_2 right) 1.0 gave 83.98 deg of shoulder travel at 65.75 mm "
                             "wrist residual (ceiling 30) where the wrist-only run gave 49.93 "
                             "deg at 6.38 mm. Lower it to buy wrist accuracy back")
    parser.add_argument("--arm-posture-gain", type=float, default=1.0, metavar="0-1",
                        help="direction recipe only (--arm-map posture): 1.0 (default) = the "
                             "robot's upper arm / forearm point exactly where yours point; "
                             "0.5 = stop halfway between the robot's own neutral direction "
                             "and yours; 0 = stay at the robot's neutral. It is an angle "
                             "blend (slerp), not a displacement scale: your 40 deg shoulder "
                             "swing stays 40 deg on every robot either way")
    parser.add_argument("--arm-posture-frame", choices=("auto", "mapped", "delta"),
                        default="auto",
                        help="direction recipe only (--arm-map posture): which frame the "
                             "operator's directions are carried in. 'auto' (default) = each "
                             "robot uses its own ArmProfile.posture_frame, which is 'mapped' "
                             "for h1_2/g1 and 'delta' for gr1_t2 (measured on the same 200 "
                             "frame replay: gr1_t2 elbow residual median 197 -> 26 mm left / "
                             "138 -> 50 mm right, while h1_2 right would go 10 -> 46 deg the "
                             "other way -- so it is per robot, not global). 'mapped' = the "
                             "historical absolute frame (the robot points where you point; "
                             "assumes your neutral pose and its loaded pose agree). 'delta' = "
                             "your rotation from your own neutral is applied to the robot's "
                             "own neutral direction (you did not move = it stays in its own "
                             "loaded pose). See docs/multi-robot-arm.md 8.19")
    parser.add_argument("--arm-ik-scope", choices=("arm", "body"), default="arm",
                        help="which joints the IK may move (default 'arm'): 'arm' pins every "
                             "non-arm joint to its current angle (arm_ik._locked_arguments), "
                             "'body' lets PyBullet solve over *all* movable joints with "
                             "restPoses pulling them (the historical behaviour, kept for A/B). "
                             "Measured 2026-10-14 on outputs/tmp_realtime/live_long_frames.npz, "
                             "200 frames, --arm-map posture: 'arm' makes the forearm-reversed "
                             "fraction 0.0 on all six sides (body: 0.099 gr1_t2 left / 0.041 g1 "
                             "left / 0.006 gr1_t2 right), wrist residual 134.6 -> 21.2 mm "
                             "(gr1_t2 left) and 74.9 -> 34.8 (h1_2 left), forearm direction "
                             "error 66.0 -> 44.7 deg (gr1_t2 left); elbow jerk maxima rise "
                             "(13.1 -> 42.7 deg on h1_2 left) because the arm now actually "
                             "follows the operator, whose recorded direction jumps reach 153 deg "
                             "in one frame. 3 repeats of 'arm' gave identical numbers (opposite "
                             "fraction 0.0 on all six sides); the solver has 7 variables instead "
                             "of ~55 so it is not slower, but no speedup is claimed -- fps over "
                             "these runs varied 2.5-12.7 with machine load under both scopes")
    parser.add_argument("--arm-rest-poses", choices=("current", "zero", "live", "loaded"),
                        default="current",
                        help="IK null-space reference: current (default) reads this frame's "
                             "joint angles; live is an alias; loaded uses the build-time "
                             "neutral from the source branch; zero uses zero angles. "
                             "Source-branch historical comparisons used its old current "
                             "meaning, now named loaded. See docs/branch-integration-2026-10-27.md")
    parser.add_argument("--arm-anchor", choices=("upper", "shoulder"), default="upper",
                        help="which FK point counts as the upper arm's shoulder end, used both "
                             "by the direction recipe's elbow/wrist targets and by the "
                             "shoulder-follow shift (default 'upper' = the elbow's parent link "
                             "origin, rigid to the elbow; 'shoulder' = the name-matched shoulder "
                             "joint link, the historical behaviour kept for A/B). Measured "
                             "2026-10-14 with devtools/probe_arm_segment_rigidity.py: the "
                             "name-matched point drifts 48.5-101.7 mm when shoulder_roll turns "
                             "(gr1_t2 left +101.6 mm) while the parent link origin is 0.00 mm, "
                             "so the old recipe asked for an elbow target off the reachable "
                             "shell and the solver had to compromise (the field 'GR1-T2 forearm "
                             "still reversed', 222 mm elbow residual)")
    parser.add_argument("--arm-no-shoulder-follow", action="store_true",
                        help="turn off shoulder following for the elbow/wrist targets (default: "
                             "on for the recipes that measure anchors). The commanded elbow is "
                             "'load-time shoulder anchor + upper-arm length * your direction', a "
                             "point fixed to the robot BASE; when the torso is pushed by the arm "
                             "reaction forces the distance from the CURRENT shoulder stops being "
                             "one upper arm, so the 7 DoF chain compromises and the elbow lands "
                             "somewhere else (the field 'bends the opposite way'). With it on, "
                             "both targets are shifted by 'current FK shoulder - anchor' every "
                             "frame. Use this flag to reproduce the old numbers")
    parser.add_argument("--arm-direction-rate-deg-s", type=float,
                        default=MAX_DIRECTION_RATE_DEG_S, metavar="DEG/S",
                        help="operator direction glitch guard (default %.0f deg/s, 0 = off). "
                             "A single-frame direction change faster than a human joint can "
                             "turn is a landmark jump, not your arm: measured on the recording "
                             "(outputs/tmp_realtime/live_long_frames.npz) the median step is "
                             "4.7-6.9 deg at ~10 fps but the MAX is 153 deg. The direction "
                             "recipe copies directions 1:1, so such a jump asks the IK to swing "
                             "the elbow 153 deg in one frame -- it either flips branch or jerks "
                             "(the field 'H1-2 twitches at the wrist'). Over the limit the "
                             "previous direction is held and counted in counts.direction_glitch"
                             % MAX_DIRECTION_RATE_DEG_S)
    # 这条的 help 用 f-string 而不是 "% CONST"（与上面两条不同）：argparse 会对每条 help
    # 再做一次 % 格式化（_format_action），"% CONST" 那一道会把 "91%%" 吃成 "91%"，argparse
    # 再撞上裸 % 就抛 TypeError: %o format ...，整个 --help 直接崩（2026-10-14 实测）。
    # 所以最终字符串里的字面百分号必须是 %%，也就是这里不能再有第二道 % 格式化。
    parser.add_argument("--arm-elbow-recovery-deg", type=float, default=ELBOW_RECOVERY_DEG,
                        metavar="DEG",
                        help=f"elbow branch recovery threshold in degrees (default "
                             f"{ELBOW_RECOVERY_DEG:.0f}, "
                             "0 = off). The IK starts from the current pose and pays for "
                             "distance to restPoses, so once a frame drops the elbow into the "
                             "flipped branch every later frame is solved from there and never "
                             "comes back -- the field complaint 'GR1-T2 bends the opposite way' "
                             "(200-frame replay, 2026-10-13: gr1_t2 left 91%% of frames with "
                             ">30 deg elbow error, 29%% with the forearm more than 90 deg away "
                             "from the asked direction; h1_2 right was 5 deg). Above this "
                             "threshold the same pass sequence is re-solved from the load "
                             "pose and the branch with the smaller forearm+0.5*upper direction "
                             "error (degrees) wins")
    parser.add_argument("--arm-elbow-recovery-margin-deg", type=float,
                        default=ELBOW_RECOVERY_MARGIN_DEG, metavar="DEG",
                        help="hysteresis of the branch recovery in degrees (default %.0f, "
                             "0 = accept any improvement). Measured 2026-10-14: with a 1 deg "
                             "margin the recovery fired every ~6 frames (h1_2 left: 19 "
                             "recoveries in 123 commanded frames) and the field saw the whole "
                             "arm twitch; 8 deg keeps one branch" % ELBOW_RECOVERY_MARGIN_DEG)
    parser.add_argument("--arm-elbow-recovery-frames", type=int,
                        default=ELBOW_RECOVERY_COOLDOWN_FRAMES, metavar="N",
                        help="after any recovery attempt (accepted or rejected), wait N "
                             "frames before trying again (default %d, 0 = try every frame). "
                             "The accepted branch warm-starts from itself, so the cooldown "
                             "does not fall back -- it only stops the per-frame flipping and "
                             "the duplicate IK solves" % ELBOW_RECOVERY_COOLDOWN_FRAMES)

    parser.add_argument("--arm-travel-mm", type=float, default=None,
                        help="override the per-axis travel clamp for every robot (mm)")
    parser.add_argument("--arm-travel-mode", choices=TRAVEL_MODES, default="clip",
                        help="what to do when the requested wrist target exceeds the travel "
                             "limit. 'clip' = cut each axis on its own (default since "
                             "2026-10-13: measured on a live composite step of "
                             "(-9.3, +6.4, +24.5) cm, where only op_z is over the limit, it "
                             "keeps g1's lateral wrist travel at 3.44 cm / shoulder roll 5.5 "
                             "deg / elbow 7.1 deg). 'scale' = shrink the whole step so its "
                             "longest axis lands exactly on the limit, direction kept: one "
                             "over-limit axis (the jittery MediaPipe depth) then shrinks every "
                             "axis, which is what made the robot look like it only ever swung "
                             "up and down (same step: g1 lateral 1.31 cm, roll 2.0 deg, elbow "
                             "2.8 deg). See docs/multi-robot-arm.md 8.8")
    parser.add_argument("--arm-lateral-guard-mm", type=float,
                        default=1000.0 * DEFAULT_LATERAL_GUARD_M, metavar="MM",
                        help="keep each wrist target on its own side of the body: left wrist "
                             "y >= +MM, right wrist y <= -MM (**default 0 = off** since "
                             "2026-10-13 evening; 20 was the daytime default). It was added "
                             "because h1_2's neutral working point is only y=+-0.05 m while the "
                             "per-axis travel limit is 0.10 / 0.15 m, so the targets can legally "
                             "cross the midline and the two forearms touch (measured on a 1560 "
                             "frame session: gap down to 2.21 mm). Cost measured 2026-10-13: "
                             "swing the left hand +-9 cm and h1_2's lateral swing drops from "
                             "0.49 to 0.43 of the operator's, and a 9.0 cm inward request lands "
                             "at 2.9 cm instead of 6.3 cm -- that half of the swing is what the "
                             "operator complained was missing, and the crossing itself is the "
                             "operator's own hand crossing the midline")
    parser.add_argument("--no-arm-gain-quality-guard", action="store_true",
                        help="trust the calibrated arm length even when it is not humanly "
                             "possible. In the measured live_long session the pose "
                             "calibration gave 0.2742 m (left) / 0.3976 m (right), 31%% apart "
                             "-> gain h1_2 left 1.4 (the 2x cap), and then every axis of "
                             "every frame slammed into the travel limit")
    parser.add_argument("--arm-slew-mm-s", type=float, default=DEFAULT_ARM_SLEW_MM_PER_S,
                        metavar="MM/S",
                        help="how fast the commanded wrist target may follow the hand, as a "
                             "**speed** (mm/s, default 600 = 20 mm per 1/30 s frame). The app "
                             "converts it every frame with the loop's measured interval, so a "
                             "10 fps camera gets 40 mm/frame (the SLEW_MAX_STEP_M ceiling) "
                             "instead of 20: the recorded operator's hand moves 37 mm/frame "
                             "(median) / 147 mm/frame (p90) at 10.05 fps, so a fixed "
                             "20 mm/frame cap put the robot on 13%% of the hand speed and the "
                             "measured lateral swing transfer was only 0.43 (0.80 with this "
                             "default). 0 = off (jump straight there; the two-pass IK can then "
                             "lock into a bad branch)")
    parser.add_argument("--arm-slew-mm", type=float, default=None, metavar="MM",
                        help="fixed per-frame cap (mm), overriding --arm-slew-mm-s. Give this "
                             "to reproduce a 2026-10-13 number exactly (that session ran "
                             "20 mm/frame at 10 fps = 200 mm/s); leave it None to let the app "
                             "scale the speed with the measured frame time")
    parser.add_argument("--arm-slew-pair-floor", type=float, default=None, metavar="RATIO",
                        help="floor for the rigid elbow-wrist pair (the --arm-map posture "
                             "recipe): the cap may not shrink that pair below RATIO of the "
                             "requested one, otherwise that frame keeps the pair sent on the "
                             "previous frame (see docs/multi-robot-arm.md 8.18). 0 = the guard "
                             "off, i.e. the historical behaviour, which is what an on-site A/B "
                             "uses; leave it None to use arm_scene_driver.SLEW_PAIR_FLOOR_RATIO "
                             "(0.6)")
    parser.add_argument("--arm-ik-iterations", type=int, default=60,
                        help="PyBullet IK maxNumIterations per pass (default 60: measured "
                             "2026-10-12, 60/1e-4 beats 200/1e-5 in residual AND speed here)")
    parser.add_argument("--arm-ik-residual-threshold", type=float, default=1e-4,
                        help="PyBullet IK residualThreshold per pass (default 1e-4)")
    parser.add_argument("--calibration-frames", type=int, default=15,
                        help="open-hand calibration frames for the geometric backend")
    parser.add_argument("--arm-calibrate-frames", type=int, default=30,
                        help="neutral frames per side before the neutral is final (default 30). "
                             "With --arm-calibrate-mode hold the arms also stay on the working "
                             "point until then; with track (default) they follow from the first "
                             "frame that sees your arm")
    parser.add_argument("--arm-calibrate-mode", choices=CALIBRATE_MODES, default="track",
                        help="hold = stay on the working point until the neutral has enough "
                             "frames (historical); track = start following from the first frame "
                             "that sees your arm and let the neutral converge (default). See "
                             "CALIBRATE_MODES in arm_realtime.py for the 2026-10-14 field evidence")
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
    parser.add_argument("--smoothing", type=float, default=0.0)
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
    sources = " ".join(
        f"{side[0].upper()}={SOURCE_SHORT.get(getattr(target, 'source', 'none'), '?')}"
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

    # sat / guard 是累计帧数：sat 一直涨说明手已经跑出机器人行程（目标被钉住，
    # 再动也不跟）；guard 是横向护栏把手腕挡回本侧的帧数。现场用这两个数一眼分辨
    # "手在动但机器人不跟"到底是行程不够还是护栏在挡。
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
        lines[side] = (f"{side} {letter} d(cm){moved} "
                       f"reach={'--' if reach is None else f'{float(reach):.3f}'}m "
                       f"elbow={'--' if elbow is None else f'{float(elbow):.0f}'}deg"
                       f"{'' if not reason else ' ' + reason[:44]}")
    return lines


def print_arm_state(processed, bridge, driver, targets, profiles) -> None:
    """``--arm-trace`` 的现场证据：你动了多少（厘米）以及机器人被要求动到哪（毫米）。"""
    trace = bridge.trace()
    for side, entry in sorted(trace.items()):
        delta = entry["delta_op_cm"]
        text = ("n/a" if delta is None
                else "[" + ", ".join(f"{value:+6.1f}" for value in delta) + "]")
        print(f"  arm trace frame={processed} {side:<5} src={entry['source']:<4} "
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
        wrist = ((raw_frame or {}).get("wrists") or {}).get(side)
        dumped.setdefault("wrist", {s: [] for s in SIDES})[side].append(
            wrist.orientation_xyzw if wrist is not None and wrist.orientation_valid else np.full(4, np.nan))
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
    """逐帧手臂样本（机器人**基座系**）先读进内存，供 ``_save_arm_frames`` 落盘。

    数据源是 ``PyBulletArmIK.record_posture``：每次写下一帧命令**之前**采样一次，所以第 i 条
    记录就是第 i 帧物理结算后的姿势 —— 与报告 ``arm.posture`` 的指标同源（同一份列表）。

    键名 ``{robot}_{side}_{field}``：

    ``points``      (N, 4, 3)，四行按 ``(shoulder, upper, elbow, wrist)`` 排
                    （``upper`` = 肘的父连杆原点 = 刚体肩端，方向配方用的锚点）
    ``cmd_elbow``   (N, 3)，产生这一帧姿势的肘命令点（基座系；没命令的帧全 NaN）
    ``cmd_wrist``   (N, 3)，同上，腕命令点

    为什么存基座系而不是世界系：三台机器人摆位不同，基座系里「上/下/左/右」才是机器人
    自己的身体坐标，两张图叠在一起看方向才不会互相误导（报告里的方向指标也是基座系）。

    **必须在 PyBullet 还连着的时候调用**（``adapter.to_base`` 会查基座位姿）：调用点在
    ``run()`` 的 ``finally`` 里断开连接**之前**。2026-10-14 现场把落盘放在 finally 之后，
    ``--arm-frame-dump`` 每次都以 ``Not connected to physics server`` 结束（断开被丢到
    后台线程，于是变成「有时成功有时失败」的竞态），这里按同一份数据改成先读后写。
    """
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


def rigid_elbow_robots(per_side) -> set:
    """哪些机器人的「肘目标」与腕目标是**刚性**成对的（斜坡必须把两点整体平移）。

    只有 ``--arm-map posture`` 的配方是这样：``wrist = elbow + L_f * v``
    （``arm_realtime._posture_target`` 推出来的两个点，随 ``info["elbow_target"] ==
    "posture"`` 标记）。``--arm-elbow-task`` 的肘目标是独立的位置目标（只打 ``elbow_task``，
    没有 ``elbow_target`` 键），它那条路的 A/B 数字在 docs/multi-robot-arm.md 8.15，
    必须逐字不变。

    为什么这件事要紧：斜坡（``ArmSceneDriver._slew_target``）只削**腕**那一点，不限速的
    肘目标会比腕先走；一对刚性目标于是被拉近，命令自己的前臂方向被压成噪声（实测 200 帧
    回放：前臂段 250.6 -> 51 mm，11 帧），IK 就把前臂折到反方向 —— 现场「跑一会 GR1-T2
    的大臂与小臂夹角反过来」。数字与复现命令见 docs/multi-robot-arm.md 8.18。
    """
    return {name for name, target in per_side.items()
            if getattr(target, "elbow", None) is not None
            and (target.info or {}).get("elbow_target") == "posture"}


def run(args) -> int:
    """采集 -> 手 + 臂 -> 仿真；0 = 至少一侧给出过输出，1 = 全程没有手也没有臂。"""
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    diagnostics_enabled = bool(getattr(args, "diagnostics", False) or getattr(args, "diagnostic_trace", None))
    if diagnostics_enabled:
        trace_path = getattr(args, "diagnostic_trace", None)
        protected = [getattr(args, key, None) for key in
                     ("input_npz", "input_body", "dump_body", "checkpoint", "model_asset_path", "smplerx_checkpoint", "smplx_model_dir", "neutral_json", "dump_npz", "save_neutral")]
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
    inputs = [Path(value).resolve() for value in (input_body, args.input_npz, args.checkpoint,
              args.model_asset_path, args.neutral_json, args.smplerx_checkpoint) if value is not None]
    if len(set(outputs)) != len(outputs) or set(outputs).intersection(inputs):
        raise ValueError('Output must not overwrite an input or another output')
    if args.input_npz is not None and input_body is not None:
        raise ValueError('Choose either --input-body or --input-npz')
    if dump_body is not None and (replaying or args.no_arms):
        raise ValueError('--dump-body requires live SMPL-X arm capture')
    asset = resolve_model_asset(args.model_asset_path) if not replaying else None
    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else realtime_checkpoint(args.landmark_space))
    robots = tuple(args.robot) if args.robot else loadable_robots()
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
    dex_options = ({"dex_calibration": args.dex_calibration, "native_sides": sides} if args.backend == "dex" else {})
    retargeter, backend, note = build_retargeter(
        args.backend, checkpoint, args.device, args.calibration_frames,
        args.smoothing, args.landmark_space, **dex_options)
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
        if backend == "dex":
            from teleoperation.applications.dex_native import command_pipeline, apply_native_window, native_sample_mapping
            commands = command_pipeline(args, retargeter.models)
        else:
            commands = build_hand_commands(args, scene, sides)
        appearance = apply_appearance(
            scene, None if args.original_g1_color else args.g1_color,
            args.light_position, args.shadows)
        if hasattr(scene, "frame_scene"):
            scene.frame_scene(args.view)
        if arms_wanted:
            # 斜坡两种口径（见 §8.9）：给了 --arm-slew-mm 就用固定 mm/帧（复现旧数字），
            # 否则按 --arm-slew-mm-s 的速率每帧换算 —— 现场 8-12 fps 下固定 20 mm/帧
            # 只有 0.2 m/s，而录制里操作者手速中位就是 0.37 m/s，摆动会被整段削掉。
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
            # segments 口径与肘任务要的「机器人自己的量」：实测连杆长 + 中性腕/肘点（基座系）。
            # 加载后量一次即可，之后每帧只是加减法（见 arm_realtime.MAP_MODES）。
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
                # 方向口径搬谁的方向（8.19）：auto 时逐台摊开，读日志的人不必去 profile 表里对照。
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
                # chain 口径（历史默认）：不量连杆长、也不给肘位置目标。这一行必须打出来 ——
                # 2026-10-14 现场那次跑的就是默认 chain，日志里一行口径提示都没有，于是
                # 「大臂抬不起来 / 张不开怀抱 / GR1-T2 前臂反」被读成「代码被改坏了」
                # （实测证据与 A/B 见 docs/multi-robot-arm.md 8.15）。
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
                  # 压扁护栏（见 arm_scene_driver.SLEW_PAIR_FLOOR_RATIO 与 8.18）：
                  # 0 = 关（历史行为）。现场一眼看出这一场跑的是哪一档。
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
            if backend == "dex":
                command_timestamp = processed * args.sim_steps / 240 if args.input_npz is not None else None
                latest = apply_native_window(window, retargeter, commands, scene, command_timestamp)
                for side in sides:
                    if latest[side] is None:
                        invalid[side] += 1
                    else:
                        samples[side].append(latest[side])
            elif window is not None:
                result = retargeter.retarget(window)
                for side in sides:
                    angles = valid_joint_result(result, "l21", side)
                    if angles is None:
                        invalid[side] += 1
                        commands.missing(('l21',side))
                        continue
                    command = commands.update(angles)
                    apply_legacy_l21_mapping(scene, command)
                    values = command.positions
                    samples[side].append(values)
                    latest[side] = values
            else:
                for side in sides:
                    commands.missing(('l21',side))
            targets = {}
            if diagnostics is not None:
                diagnostics.record_ms("hand_inference_output", (time.perf_counter() - hand_started) * 1000)
            if bridge is not None:
                # 这一帧的真实时长先算出来：斜坡的速率口径与关键点跳变护栏都要用它
                # （现场摄像头 8-12 fps，名义 1/30 s 会把 600 度/秒算成 20 度/帧）。
                # Legacy NPZ replay uses the explicit fixed physics interval.
                # Its timestamps have no guaranteed unit; host processing speed
                # must not alter direction guards or target slew limiting.
                frame_dt = (args.sim_steps / 240.0 if args.input_npz is not None else
                            getattr(camera, 'frame_dt', None) or NOMINAL_FRAME_DT)
                from teleoperation.applications.camera_scene_arm import apply_camera_arm_frame
                def arm_error(side, error):
                    if side not in arm_failures:
                        arm_failures[side] = f'{type(error).__name__}: {error}'
                        print(f'[WARN] {side}: arm command held: {arm_failures[side]}', flush=True)
                with (nullcontext() if diagnostics is None else diagnostics.stage('arm_mapping')):
                    targets = apply_camera_arm_frame(bridge, driver, wrist_control, camera.arms(),
                        camera.wrists(), frame_dt, elbow_follow=args.arm_elbow_follow, on_error=arm_error)
                scene.steps_per_frame = simulation_clock.next_steps()
                with (nullcontext() if diagnostics is None else diagnostics.stage("physics")):
                    driver.step(1)
                wrist_control.record()
            else:
                scene.steps_per_frame = simulation_clock.next_steps()
                with (nullcontext() if diagnostics is None else diagnostics.stage("physics")):
                    scene.step()
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
    if scene_summary is not None:
        print(f"scene={scene_summary['scene']} sim_frames={scene_summary['sim_frames']} "
              f"applied={scene_summary['applied_frames']} "
              f"clipped={scene_summary['clipped_dimensions']}", flush=True)
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
        "model_asset": None if asset is None else str(asset),
        "smplerx_model": None if not arms_wanted else args.smplerx_model,
        "smplerx_checkpoint": None if not arms_wanted or replaying else str(Path(args.smplerx_checkpoint).resolve()),
        "input_body": None if input_body is None else str(input_body),
        "body_source": camera._source.metrics.summary() if hasattr(camera, '_source') else None,
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
        # 方向口径搬谁的方向：auto（每台按 profile，见 bridge.posture_frame_by_robot）/
        # mapped / delta。见 docs/multi-robot-arm.md 8.19。
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
                "angles": _describe(samples[side], native_sample_mapping(retargeter, side) if backend == "dex" else None) if samples[side] else None,
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
                  "three-frame windows and open-hand calibration.", flush=True)
    if bridge is not None:
        trace = bridge.trace()
        for side, entry in sorted(trace.items()):
            print(f"  arm  {side}: source={entry['source']} "
                  f"delta_op_cm={entry['delta_op_cm']} reach_m={entry['operator_reach_m']} "
                  f"elbow_deg={entry['elbow_bend_deg']}")
    if bridge is not None:
        # 「这一帧为什么没跟手」的分布（HOLD_REASONS）：现场判断"手臂不动"的第一手证据 ——
        # 2026-10-14 那场只看 HUD 的 cal=15/30 是猜不出原因的，这一行直接给出分类。
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
                  "in the middle of the frame and hold it open for about "
                  f"{args.calibration_frames} frames to finish calibration.")
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
    # 与基线入口一致：先跑完 main()（统计/报告都写完），flush 之后直接 os._exit，
    # 免得卡在 MediaPipe landmarker.close()（本机实测约 42 s）上。
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)








