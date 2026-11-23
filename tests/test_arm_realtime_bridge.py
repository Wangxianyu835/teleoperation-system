"""unit tests: simulation/arm_realtime.py (relative arm retargeting) + the replay source.

这些用例**不碰 PyBullet、不开摄像头**：桥本身是纯 numpy，所以能在这里把相对映射、
中性标定、丢帧保持、兜底来源、每侧行程都算清楚。仿真侧"手臂真的跟上了"的判据在
``devtools/verify_realtime_arm_bridge.py``（那里才有残差、行程、漂移的实测数字）。

运行::

    $env:PYTHONPATH = "$pwd\\src"
    E:\\python3.11.7\\python.exe -m unittest tests.test_arm_realtime_bridge -v
"""
from __future__ import annotations

import importlib.util
import csv
import io
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from teleoperation.simulation.arm_profiles import PAPER_ROBOTS, profile_for
from teleoperation.simulation.arm_realtime import (
    CALIBRATE_MODES,
    DEFAULT_LATERAL_GUARD_M,
    GLITCH_HOLD_FRAMES,
    HOLD_REASONS,
    MAX_DIRECTION_RATE_DEG_S,
    MAX_ORIENTATION_RATE_DEG_S,
    MAX_ORIENTATION_STEP_DEG,
    MIN_ELBOW_PLANE_SINE,
    ORIENTATION_MODES,
    PALM_HOLD_S,
    PALM_LATE_FRAMES,
    PALM_LATE_SPREAD_DEG,
    PALM_MAX_SPREAD_DEG,
    SYMMETRY_MAX_SAMPLES,
    SYMMETRY_SAME_SIDE_DEG,
    AxisMap,
    AdaptiveArmPointFilter,
    AdaptiveRotationFilter,
    RealtimeArmBridge,
    anatomical_hand_frame,
    arm_reach,
    elbow_bend_degrees,
    load_neutral,
    palm_orientation_frame,
    quaternion_angle_deg,
    relative_palm_rotation,
    relative_palm_wrist_rotation,
    relative_wrist_rotation,
    save_neutral,
    wrist_orientation_frame,
)
from teleoperation.simulation.realtime_teleop import (
    WRIST_TRACE_COLUMNS,
    CameraFrameSource,
    FrameBudget,
    ReplayFrameSource,
    WristTrace,
    build_parser,
    orientation_source_label,
)

SIDES = ("left", "right")
PROFILES = [profile_for(name) for name in PAPER_ROBOTS]
#: 合成操作者的手臂：上臂 + 前臂 = 0.30 + 0.30 = 0.60 m，正好等于归一化参考臂长，
#: 因此 gain 应当等于 profile.metric_scale（断言不用再乘一个系数）。
UPPER_ARM = np.array([0.0, 0.30, 0.0])
FORE_ARM = np.array([0.0, 0.30, 0.0])


def arm(side: str, wrist_offset=(0.0, 0.0, 0.0), body_shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """(肩, 肘, 腕) 米制：``wrist_offset`` 只动腕，``body_shift`` 把整个人搬走。"""
    sign = 1.0 if side == "left" else -1.0
    shift = np.asarray(body_shift, dtype=np.float64)
    shoulder = np.array([0.17 * sign, -0.35, 0.0]) + shift
    elbow = shoulder + UPPER_ARM
    wrist = elbow + FORE_ARM + np.asarray(wrist_offset, dtype=np.float64)
    return np.stack((shoulder, elbow, wrist))


def open_hand(centre=(0.5, 0.5, 0.0), span: float = 1.0) -> np.ndarray:
    """一只掌面可张成的手（21 点，归一化口径），用于兜底来源与回放用例。"""
    wrist = np.asarray(centre, dtype=np.float64)
    offsets = (
        (0.02, 0.06), (0.04, 0.03), (0.06, 0.01), (0.07, -0.01),        # thumb
        (0.03, -0.03), (0.04, -0.08), (0.04, -0.11), (0.04, -0.13),     # index
        (0.00, -0.03), (0.00, -0.09), (0.00, -0.12), (0.00, -0.14),     # middle
        (-0.03, -0.03), (-0.04, -0.08), (-0.04, -0.11), (-0.04, -0.12),  # ring
        (-0.05, -0.02), (-0.06, -0.06), (-0.07, -0.08), (-0.07, -0.10),  # pinky
    )
    points = [wrist]
    for x, y in offsets:
        points.append(wrist + np.array([x, y, 0.0]) * span)
    return np.asarray(points, dtype=np.float64)


def bent_arm(side: str, wrist_offset=(0.0, 0.06, 0.10)) -> np.ndarray:
    """米制 (肩, 肘, 腕)，但肘是弯的：前臂不平行上臂，所以手臂平面能张成（有姿态）。

    默认偏移 (0, 0.06, 0.10) 与上臂 (0, 0.30, 0) 的夹角约 24 度，远大于
    ``MIN_ELBOW_PLANE_SINE``，所以这一帧一定给得出腕坐标系。
    """
    sign = 1.0 if side == "left" else -1.0
    shoulder = np.array([0.17 * sign, -0.35, 0.0])
    elbow = shoulder + UPPER_ARM
    wrist = elbow + FORE_ARM + np.asarray(wrist_offset, dtype=np.float64)
    return np.stack((shoulder, elbow, wrist))


def axis_angle(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues 公式：绕 ``axis`` 转 ``angle_rad`` 的 3x3 旋转（测试里造「动过的手」）。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * cross + (1.0 - np.cos(angle_rad)) * (cross @ cross)


def rotate_arm(points, rotation, origin) -> np.ndarray:
    """把整条手臂绕 ``origin`` 转 ``rotation``（用来造「你转了一下前臂」的帧）。"""
    array = np.asarray(points, dtype=np.float64)
    return (rotation @ (array - np.asarray(origin, dtype=np.float64)).T).T + np.asarray(origin)


def _quaternion_angle_deg(quaternion) -> float:
    """四元数 ``(x, y, z, w)`` 的旋转角（度）：取最短那一侧（``q`` 与 ``-q`` 是同一个姿态）。"""
    values = np.asarray(quaternion, dtype=np.float64)
    w = min(1.0, abs(float(values[3])))
    return float(np.degrees(2.0 * np.arccos(w)))


def twisted_hand(side: str = "right", angle_deg: float = 40.0) -> np.ndarray:
    """掌面绕**前臂轴**转过 ``angle_deg`` 度的手（旋前/旋后）；腕仍落在 ``bent_arm`` 的腕上。

    掌面基的 x 取自 pose 的前臂方向、z 取自掌面法向，所以只转手掌 = 只改 z，
    相对旋转正好是绕前臂轴的 ``angle_deg`` 度 —— 这一维三点基完全量不到（见 PALM_HOLD_S）。
    """
    arm_points = bent_arm(side)
    hand = open_hand()
    axis = arm_points[2] - arm_points[1]
    return rotate_arm(hand, axis_angle(axis, np.radians(angle_deg)), hand[0])


def calibrated(sides=SIDES, frames: int = 4, **kwargs) -> RealtimeArmBridge:
    """建桥并喂 ``frames`` 帧中性姿势，让标定恰好完成。"""
    bridge = RealtimeArmBridge(PROFILES, sides=sides, calibration_frames=frames, **kwargs)
    for _ in range(frames):
        bridge.update(arms_world={side: arm(side) for side in sides}, hands=None)
    return bridge



class AxisMapTests(unittest.TestCase):
    """轴向约定：默认映射把"手往前"接到机器人 +X，"图像向右"接到机器人 +Y。"""

    def test_default_map_maps_operator_axes_to_robot_axes(self):
        matrix = AxisMap().matrix()
        np.testing.assert_allclose(matrix @ np.array([0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(matrix @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])

    def test_rejects_invalid_order_and_signs(self):
        with self.assertRaises(ValueError):
            AxisMap(("X", "Y", "Y"), (1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            AxisMap(("X", "Y", "Z"), (1.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            AxisMap(("X", "Y", "Z"), (1.0, 1.0))

    def test_parse_accepts_the_cli_forms(self):
        self.assertEqual(AxisMap.parse("ZXY", "-1,1,-1"), AxisMap())
        self.assertEqual(AxisMap.parse("z x y", "-1 1 -1"), AxisMap())
        self.assertEqual(AxisMap.parse("XYZ", "1,1,1").describe()["signs"], [1.0, 1.0, 1.0])

    def test_describe_is_json_able_and_readable(self):
        described = json.loads(json.dumps(AxisMap().describe()))
        self.assertEqual(described["order"], ["Z", "X", "Y"])
        self.assertEqual(described["robot_axes_from_operator"][0], "X_robot = -1 * Z_op")


class GeometryTests(unittest.TestCase):
    """肘弯角与臂长：纯诊断量，用来对照"你的肘弯了多少、机器人弯了多少"。"""

    def test_elbow_bend_is_180_when_straight_and_0_when_folded(self):
        straight = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.6, 0.0]])
        self.assertAlmostEqual(elbow_bend_degrees(*straight), 180.0, places=4)
        folded = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.0]])
        self.assertAlmostEqual(elbow_bend_degrees(*folded), 0.0, places=4)

    def test_elbow_bend_is_nan_for_degenerate_input(self):
        self.assertTrue(np.isnan(elbow_bend_degrees([0, 0, 0], [0, 0, 0], [1, 0, 0])))

    def test_arm_reach_is_the_sum_of_both_segments(self):
        self.assertAlmostEqual(arm_reach(arm("right")), 0.60, places=9)
        self.assertTrue(np.isnan(arm_reach(None)))
        self.assertTrue(np.isnan(arm_reach(np.full((3, 3), np.nan))))
        # 全零的输入在长度上"合法"（0 米），是桥的 MIN_OPERATOR_REACH_M 把它判成不可用
        self.assertEqual(arm_reach(np.zeros((3, 3))), 0.0)



class CalibrationTests(unittest.TestCase):
    """中性标定：攒够帧数才完成，取中位数，坏帧不计入。"""

    def test_calibration_needs_n_valid_frames_and_uses_the_median(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=3)
        self.assertTrue(bridge.calibrating)
        for _ in range(2):
            targets = bridge.update(arms_world={"right": arm("right")})
        self.assertTrue(bridge.calibrating)
        self.assertEqual(bridge.calibration_progress()["right"], (2, 3))
        entry = targets["h1_2"]["right"]
        self.assertTrue(entry.held and not entry.valid)
        self.assertIn("calibrating 2/3", entry.reason)
        bridge.update(arms_world={"right": arm("right", wrist_offset=(0.03, 0.0, 0.0))})
        self.assertFalse(bridge.calibrating)
        # 中位数：第 3 帧的位移不改变中性「腕 - 肩」（三帧里取中位）
        self.assertAlmostEqual(bridge.neutral["right"]["reach"], 0.60, places=6)
        self.assertEqual(bridge.neutral["right"]["samples"], 3)

    def test_invalid_frames_do_not_count_towards_calibration(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=2)
        bridge.update(arms_world={"right": None})
        bridge.update(arms_world={"right": np.full((3, 3), np.nan)})
        self.assertTrue(bridge.calibrating)

    def test_a_loaded_neutral_skips_calibration(self):
        payload = {"sides": {"left": {"offset": [0.0, 0.6, 0.0], "reach": 0.6,
                                      "samples": 9, "complete": True},
                             "right": {"offset": [0.0, 0.6, 0.0], "reach": 0.6,
                                       "samples": 9, "complete": True}}}
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, neutral=payload)
        self.assertFalse(bridge.calibrating)
        self.assertEqual(bridge.status(), "cal=ok")

    def test_status_reports_progress_while_calibrating(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4)
        bridge.update(arms_world={side: arm(side) for side in SIDES})
        self.assertEqual(bridge.status(), "cal=left=1/4 right=1/4")


class CalibrateModeTests(unittest.TestCase):
    """还没攒够中性帧时跟不跟手（``--arm-calibrate-mode``）。

    2026-10-14 现场：85% 的帧没有任何可用手臂数据，``hold`` 下标定进度停在
    ``cal=left=15/30 right=16/30`` 整整 4 分钟，三台机器人一动不动 —— 用户看到的就是
    「手臂卡在胸前举着」。``track`` 让「看见一次就动」，代价只是开头几帧还没有基准。
    """

    def test_hold_is_the_constructor_default_and_keeps_the_working_point(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4)
        self.assertEqual(bridge.calibrate_mode, "hold")
        moved = {"right": arm("right", wrist_offset=(0.0, 0.0, 0.08))}
        pinned = bridge.update(arms_world=moved)["h1_2"]["right"]
        self.assertTrue(pinned.held and not pinned.valid)
        # 改前的行为逐字保留：目标就是这一侧的 profile 中性工作点。
        reference = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4)
        working = reference.update(arms_world={"right": None})["h1_2"]["right"]
        np.testing.assert_allclose(pinned.position, working.position, atol=0.0)

    def test_track_mode_follows_the_operator_before_the_neutral_is_complete(self):
        # palm_wait_frames=0：这些夹具不给手点，等手点的语义归 PalmWaitTests 管，
        # 这里只钉「中和没攒够也跟手」。
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=30,
                                   calibrate_mode="track", palm_wait_frames=0)
        hold = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=30)
        for _ in range(3):                       # 现场要求站定 30 帧，这里站 3 帧
            bridge.update(arms_world={"right": arm("right")})
            hold.update(arms_world={"right": arm("right")})
        self.assertTrue(bridge.calibrating)      # 30 帧远没攒够
        moved = {"right": arm("right", wrist_offset=(0.0, 0.0, 0.08))}
        tracking = bridge.update(arms_world=moved)["h1_2"]["right"]
        pinned = hold.update(arms_world=moved)["h1_2"]["right"]
        self.assertTrue(tracking.valid and not tracking.held)
        self.assertEqual(tracking.reason, "")
        self.assertTrue(pinned.held and not pinned.valid)
        # 同一帧、同一个操作者动作：track 已经离开工作点，hold 还在原地。
        self.assertGreater(np.linalg.norm(np.asarray(tracking.position, dtype=np.float64)
                                          - np.asarray(pinned.position, dtype=np.float64)),
                           0.01)
        self.assertEqual(bridge.counts["held"]["right"], 0)
        self.assertEqual(bridge.status(), "cal=track right=4/30")

    def test_track_and_hold_converge_to_the_same_neutral(self):
        frames = [arm("right")] * 3 + [arm("right", wrist_offset=(0.0, 0.0, 0.02))]
        tracking = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4,
                                     calibrate_mode="track", palm_wait_frames=0)
        hold = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4)
        for frame in frames:
            tracking.update(arms_world={"right": frame})
            hold.update(arms_world={"right": frame})
        self.assertTrue(tracking.neutral["right"]["complete"])
        self.assertFalse(tracking.calibrating)
        # 同一批样本、同一个中位数：攒够之后的临时基准与终值没有区别。
        np.testing.assert_allclose(tracking.neutral["right"]["offset"],
                                   hold.neutral["right"]["offset"], atol=0.0)
        np.testing.assert_allclose(tracking.neutral["right"]["upper"],
                                   hold.neutral["right"]["upper"], atol=0.0)
        self.assertEqual(tracking.neutral["right"]["samples"],
                         hold.neutral["right"]["samples"])

    def test_held_frames_are_classified_for_the_field_log(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=2,
                                   calibrate_mode="track", palm_wait_frames=0)
        bridge.update(arms_world={"right": None})              # 一条数据都没有
        bridge.update(arms_world={"right": arm("right")})      # 有效
        bridge.update(arms_world={"right": arm("right")})      # 标定恰好攒够
        bridge.update(arms_world={"right": None})              # 保持上一帧
        squashed = np.stack((np.zeros(3), np.array([0.0, 0.02, 0.0]),
                             np.array([0.0, 0.04, 0.0])))
        bridge.update(arms_world={"right": squashed})           # 被臂长判据拒
        reasons = bridge.counts["held_reason"]["right"]
        self.assertEqual(reasons["calibrating"], 1)
        self.assertEqual(reasons["no_pose_landmarks"], 1)
        self.assertEqual(reasons["reach_below_min"], 1)
        # 报告里能直接读到同一份计数（现场拿到报告不用再翻 HUD 日志）。
        self.assertEqual(bridge.summary()["counts"]["held_reason"]["right"], reasons)
        self.assertEqual(sorted(reasons), sorted(HOLD_REASONS))

    def test_a_bad_calibrate_mode_is_refused(self):
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, calibrate_mode="still")
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_calibrate_mode, CALIBRATE_MODES[1])   # track 是现场默认
        hold = build_parser().parse_args(["--arm-calibrate-mode", "hold"])
        self.assertEqual(hold.arm_calibrate_mode, CALIBRATE_MODES[0])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--arm-calibrate-mode", "still"])


class RelativeMappingTests(unittest.TestCase):
    """相对重定向：中性点、方向、站位无关、臂长归一化、每侧行程夹取。"""

    def test_a_neutral_frame_sits_on_the_profile_neutral_point(self):
        bridge = calibrated(sides=("right",))
        targets = bridge.update(arms_world={"right": arm("right")})
        for name in PAPER_ROBOTS:
            entry = targets[name]["right"]
            self.assertTrue(entry.valid)
            self.assertFalse(entry.held)
            self.assertEqual(entry.info["source"], "pose")
            np.testing.assert_allclose(entry.position,
                                       profile_for(name).realtime_neutral("right"), atol=1e-9)

    def test_moving_the_wrist_away_from_the_camera_pushes_robot_x(self):
        bridge = calibrated(sides=("right",))
        targets = bridge.update(arms_world={"right": arm("right", wrist_offset=(0.0, 0.0, 0.10))})
        entry = targets["h1_2"]["right"]
        neutral = entry.info["neutral_m"]
        gain = profile_for("h1_2").metric_scale          # 臂长 == 参考值 -> 系数 1.0
        self.assertAlmostEqual(entry.info["gain"], gain, places=6)
        self.assertAlmostEqual(entry.position[0], neutral[0] - gain * 0.10, places=9)
        self.assertAlmostEqual(entry.position[1], neutral[1], places=9)
        self.assertAlmostEqual(entry.position[2], neutral[2], places=9)
        self.assertFalse(entry.info["clamped"])

    def test_whole_body_translation_does_not_move_the_robot(self):
        """站位移动时肩和腕一起动，机器人不该被带跑（这正是不需要手眼标定的那一条）。"""
        bridge = calibrated(sides=("right",))
        still = bridge.update(arms_world={"right": arm("right")})["g1"]["right"].position
        moved = bridge.update(
            arms_world={"right": arm("right", body_shift=(0.25, 0.15, -0.30))})["g1"]["right"]
        np.testing.assert_allclose(moved.position, still, atol=1e-9)

    def test_shorter_arms_get_a_larger_gain_capped_by_the_band(self):
        # 这一例专门测「短臂 -> 归一化放大」：合成臂长 0.30 m 正好压在可信下限上，
        # 显式放宽到 0.15 m，免得这条断言和 MIN_PLAUSIBLE_ARM_REACH_M 的边界耦合。
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=1,
                                   min_plausible_reach_m=0.15)
        short = np.stack((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.15, 0.0]),
                          np.array([0.0, 0.30, 0.0])))
        bridge.update(arms_world={"right": short})
        entry = bridge.update(arms_world={"right": short})["h1_2"]["right"]
        self.assertEqual(entry.info["gain_source"], "neutral")
        # 0.60 / 0.30 = 2.0，正好压在允许区间上限
        self.assertAlmostEqual(entry.info["gain"], profile_for("h1_2").metric_scale * 2.0,
                               places=6)

    def test_arm_length_normalisation_can_be_switched_off(self):
        bridge = calibrated(sides=("right",), normalise_arm_length=False)
        entry = bridge.update(arms_world={"right": arm("right")})["g1"]["right"]
        self.assertAlmostEqual(entry.info["gain"], profile_for("g1").metric_scale, places=6)

    def test_travel_is_clamped_per_axis_and_reported(self):
        bridge = calibrated(sides=("right",))
        entry = bridge.update(
            arms_world={"right": arm("right", wrist_offset=(0.50, 0.0, 0.0))})["g1"]["right"]
        limit = profile_for("g1").travel_limit("right")
        delta = np.asarray(entry.position) - np.asarray(entry.info["neutral_m"])
        self.assertTrue(entry.info["clamped"])
        self.assertAlmostEqual(abs(delta[1]), limit, places=9)
        # 夹取计数按 (机器人, 侧) 记：三台各算一次
        self.assertEqual(bridge.counts["clamped"]["right"], len(PAPER_ROBOTS))

    def test_the_override_flag_replaces_the_per_side_limit(self):
        bridge = calibrated(sides=("right",), max_travel_m=0.02)
        entry = bridge.update(
            arms_world={"right": arm("right", wrist_offset=(0.50, 0.0, 0.0))})["h1_2"]["right"]
        delta = np.asarray(entry.position) - np.asarray(entry.info["neutral_m"])
        self.assertAlmostEqual(abs(delta[1]), 0.02, places=9)
        self.assertEqual(entry.info["travel_limit_m"], 0.02)



class TravelModeAndGuardTests(unittest.TestCase):
    """2026-10-13 三个护栏：行程等比缩 / 横向护栏 / 标定可信度。

    现场那场 1560 帧的旧口径实测（outputs\\tmp_guard\\ab_baseline.json）：
    行程夹取 6599 次（每帧每轴都夹）、h1_2 肩关节行程 322 度、前臂间距最小 2.21 mm。
    """

    def test_defaults_match_the_documented_guards(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES)
        self.assertEqual(bridge.travel_mode, "clip")
        self.assertEqual(DEFAULT_LATERAL_GUARD_M, 0.0)      # 2026-10-13 晚：默认不拦横向
        self.assertAlmostEqual(bridge.lateral_guard_m, DEFAULT_LATERAL_GUARD_M, places=9)
        self.assertTrue(bridge.gain_quality_guard)
        report = bridge.summary()
        self.assertEqual(report["travel_mode"], "clip")
        for key in ("guarded", "gain_guarded"):
            self.assertEqual(set(report["counts"][key]), set(SIDES))

    def test_constructor_rejects_bad_guard_settings(self):
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, travel_mode="squash")
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, lateral_guard_m=-0.01)
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, min_plausible_reach_m=0.0)

    def test_the_entry_parser_exposes_the_guard_flags(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_travel_mode, "clip")
        self.assertAlmostEqual(args.arm_lateral_guard_mm,
                               1000.0 * DEFAULT_LATERAL_GUARD_M, places=9)
        self.assertEqual(args.arm_lateral_guard_mm, 0.0)        # 默认不拦横向
        self.assertFalse(args.no_arm_gain_quality_guard)
        legacy = build_parser().parse_args(["--arm-travel-mode", "scale",
                                            "--arm-lateral-guard-mm", "20",
                                            "--no-arm-gain-quality-guard"])
        self.assertEqual(legacy.arm_travel_mode, "scale")
        self.assertEqual(legacy.arm_lateral_guard_mm, 20.0)     # 恢复旧护栏
        self.assertTrue(legacy.no_arm_gain_quality_guard)

    def test_the_entry_parser_exposes_the_palm_hold_window(self):
        """``--arm-palm-hold-ms``：默认 500 ms，``0`` 关掉（A/B 基线）。"""
        args = build_parser().parse_args([])
        self.assertAlmostEqual(args.arm_palm_hold_ms, 1000.0 * PALM_HOLD_S, places=9)
        self.assertEqual(args.arm_palm_hold_ms, 500.0)
        off = build_parser().parse_args(["--arm-palm-hold-ms", "0"])
        self.assertEqual(off.arm_palm_hold_ms, 0.0)

    def test_the_entry_parser_defaults_the_slew_to_a_speed(self):
        """斜坡默认是速率（mm/s），固定 mm/帧 只作为复现旧数字的 override。"""
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_slew_mm, None)
        self.assertGreater(args.arm_slew_mm_s, 0.0)
        # 30 fps 下换算回来就是 2026-10-13 那次实测用的 20 mm/帧。
        self.assertAlmostEqual(args.arm_slew_mm_s / 30.0, 20.0, places=9)
        fixed = build_parser().parse_args(["--arm-slew-mm", "20"])
        self.assertEqual(fixed.arm_slew_mm, 20.0)

    @staticmethod
    def _overflow(bridge, side="right", offset=(0.50, 0.20, 0.30), robot="g1"):
        """一个「手跑出机器人行程」的请求：三个轴都超上限，返回 (目标, 中性, 请求)。"""
        entry = bridge.update(
            arms_world={side: arm(side, wrist_offset=offset)})[robot][side]
        neutral = np.asarray(entry.info["neutral_m"], dtype=np.float64)
        wanted = AxisMap().apply(np.asarray(offset, dtype=np.float64))
        return entry, neutral, wanted * float(entry.info["gain"])

    def test_scale_mode_keeps_the_direction_and_lands_the_longest_axis_on_the_limit(self):
        bridge = calibrated(sides=("right",), travel_mode="scale")
        entry, neutral, wanted = self._overflow(bridge)
        limit = float(entry.info["travel_limit_m"])
        self.assertTrue(entry.info["clamped"])
        self.assertGreater(float(np.max(np.abs(wanted))), limit)
        delta = np.asarray(entry.position) - neutral
        np.testing.assert_allclose(delta, wanted * (limit / np.max(np.abs(wanted))),
                                   atol=1e-9)
        self.assertAlmostEqual(float(np.max(np.abs(delta))), limit, places=9)
        self.assertEqual(bridge.counts["clamped"]["right"], len(PAPER_ROBOTS))

    def test_clip_mode_still_pins_every_axis_on_the_box_edge(self):
        bridge = calibrated(sides=("right",), travel_mode="clip")
        entry, neutral, wanted = self._overflow(bridge)
        limit = float(entry.info["travel_limit_m"])
        delta = np.asarray(entry.position) - neutral
        self.assertTrue(bool(np.all(np.abs(wanted) > limit)))
        np.testing.assert_allclose(np.abs(delta), np.full(3, limit), atol=1e-9)
        scaled = wanted * (limit / np.max(np.abs(wanted)))
        self.assertGreater(float(np.linalg.norm(delta)), float(np.linalg.norm(scaled)))

    def test_the_default_mode_only_cuts_the_axis_that_is_over_the_limit(self):
        """回归（2026-10-13 现场"横向摆动丢失"）：只有一个轴超限时，别的轴必须原样保留。

        现场那次复合位移 (-9.3, +6.4, +24.5) cm 里只有 op_z（纵深，MediaPipe 最抖的轴）
        超限。``scale`` 把整条位移等比压小，于是横向 9.0 cm 的请求只剩 0.42 倍；``clip``
        （现在的默认）只压超限的那一轴，横向按 gain 原样过去。
        """
        offset = (0.09, 0.0, 0.24)          # 操作者系：横向 9 cm + 纵深 24 cm
        deltas = {}
        for mode in ("clip", "scale"):
            bridge = calibrated(sides=("right",), travel_mode=mode)
            entry = bridge.update(
                arms_world={"right": arm("right", wrist_offset=offset)})["g1"]["right"]
            neutral = np.asarray(entry.info["neutral_m"], dtype=np.float64)
            deltas[mode] = np.asarray(entry.position) - neutral
            gain = float(entry.info["gain"])
            limit = float(entry.info["travel_limit_m"])
            self.assertTrue(entry.info["clamped"])
        wanted = AxisMap().apply(np.asarray(offset, dtype=np.float64)) * gain
        self.assertGreater(abs(wanted[0]), limit)                  # 纵深确实超限
        self.assertLess(abs(wanted[1]), limit)                     # 横向没超限
        # 超限轴两模式都钉在上限；横向：clip 原样、scale 被拖小。
        self.assertAlmostEqual(abs(deltas["clip"][0]), limit, places=9)
        self.assertAlmostEqual(abs(deltas["scale"][0]), limit, places=9)
        self.assertAlmostEqual(deltas["clip"][1], wanted[1], places=9)
        self.assertLess(abs(deltas["scale"][1]), abs(deltas["clip"][1]))
        self.assertAlmostEqual(abs(deltas["scale"][1] / deltas["clip"][1]), limit / abs(wanted[0]),
                               places=9)


    def test_the_lateral_guard_keeps_the_left_wrist_on_the_left_side(self):
        # 护栏默认已经是 0（2026-10-13 晚，见 §8.9），这里显式给 20 mm 测机制本身。
        bridge = calibrated(sides=("left",), lateral_guard_m=0.02)
        entry = bridge.update(
            arms_world={"left": arm("left", wrist_offset=(-0.50, 0.0, 0.0))})["h1_2"]["left"]
        neutral = np.asarray(entry.info["neutral_m"], dtype=np.float64)
        wanted = AxisMap().apply(np.array([-0.50, 0.0, 0.0])) * float(entry.info["gain"])
        wanted = wanted * (float(entry.info["travel_limit_m"]) / np.max(np.abs(wanted)))
        self.assertLess(neutral[1] + wanted[1], 0.0)     # 不加护栏时腕目标会到身体右侧
        self.assertTrue(entry.info["guarded"])
        self.assertAlmostEqual(entry.position[1], 0.02, places=9)
        self.assertGreaterEqual(bridge.counts["guarded"]["left"], 1)

    def test_the_default_guard_is_off_so_the_target_really_does_cross_the_midline(self):
        """默认不再拦横向（2026-10-13 晚的决定）：跨中线是操作者自己把右手横过身体中线。

        代价写在 §8.9：要恢复"窗口里永不跨中线"就给 `--arm-lateral-guard-mm 20`。
        """
        bridge = calibrated(sides=("left",))
        self.assertEqual(bridge.lateral_guard_m, 0.0)
        entry = bridge.update(
            arms_world={"left": arm("left", wrist_offset=(-0.50, 0.0, 0.0))})["h1_2"]["left"]
        self.assertFalse(entry.info["guarded"])
        self.assertLess(entry.position[1], 0.0)          # 前臂贴到 2.21 mm 的机制就是这个
        self.assertEqual(bridge.counts["guarded"]["left"], 0)

    def test_a_plausible_calibration_is_still_normalised(self):
        bridge = calibrated(sides=("right",))
        entry = bridge.update(arms_world={"right": arm("right")})["g1"]["right"]
        self.assertTrue(entry.info["gain_trusted"])
        self.assertEqual(entry.info["gain_source"], "neutral")
        self.assertEqual(bridge.counts["gain_guarded"]["right"], 0)

    def test_an_implausible_calibration_switches_the_normalisation_off(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=2)
        tiny = np.stack((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.12, 0.0]),
                         np.array([0.0, 0.24, 0.0])))       # 肩到腕 0.24 m：不是人的手臂
        for _ in range(2):
            bridge.update(arms_world={"right": tiny})       # 标定帧：不计入护栏计数
        before = bridge.counts["gain_guarded"]["right"]
        entry = bridge.update(arms_world={"right": tiny})["h1_2"]["right"]
        self.assertEqual(bridge.counts["gain_guarded"]["right"] - before,
                         len(PAPER_ROBOTS))
        self.assertFalse(entry.info["gain_trusted"])
        self.assertEqual(entry.info["gain_source"], "metric_scale")
        self.assertAlmostEqual(entry.info["gain"], profile_for("h1_2").metric_scale,
                               places=6)
        self.assertIn("below", entry.info["gain_guard_reason"])
        self.assertFalse(bridge.summary()["arm_length_trust"]["right"]["trusted"])

    def test_left_and_right_reaches_that_disagree_are_not_trusted(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=1)
        short = np.stack((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.16, 0.0]),
                          np.array([0.0, 0.32, 0.0])))      # 0.32 m 本身合格，但和 0.60 差 47%
        entry = None
        for _ in range(2):
            entry = bridge.update(arms_world={"left": arm("left"),
                                              "right": short})["h1_2"]["left"]
        self.assertFalse(entry.info["gain_trusted"])
        self.assertIn("differ by", entry.info["gain_guard_reason"])
        self.assertAlmostEqual(entry.info["gain"], profile_for("h1_2").metric_scale,
                               places=6)


class HoldAndFallbackTests(unittest.TestCase):
    """退化输入不猜：保持上一帧、或用既有「21 点 -> 手腕」兜底。"""

    def test_missing_pose_arm_holds_the_previous_target_exactly(self):
        bridge = calibrated(sides=("right",))
        first = bridge.update(arms_world={"right": arm("right", wrist_offset=(0.0, 0.0, 0.05))})
        second = bridge.update(arms_world={"right": None})
        previous = first["gr1_t2"]["right"]
        entry = second["gr1_t2"]["right"]
        self.assertTrue(entry.held and not entry.valid)
        self.assertEqual(entry.info["source"], "pose")      # 保持的是上一帧那次的来源
        self.assertIn("no usable pose arm landmarks", entry.reason)
        np.testing.assert_allclose(entry.position, previous.position, atol=0.0)
        self.assertGreaterEqual(bridge.counts["held"]["right"], 1)

    def test_a_too_short_arm_frame_is_rejected_and_held(self):
        bridge = calibrated(sides=("right",))
        bridge.update(arms_world={"right": arm("right", wrist_offset=(0.0, 0.0, 0.05))})
        squashed = np.stack((np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.02, 0.0]),
                             np.array([0.0, 0.04, 0.0])))
        entry = bridge.update(arms_world={"right": squashed})["h1_2"]["right"]
        self.assertTrue(entry.held)
        self.assertIn("below", entry.reason)

    def test_the_hand_fallback_is_used_when_pose_is_missing(self):
        bridge = calibrated(sides=("right",))
        targets = bridge.update(arms_world={"right": None}, hands={"right": open_hand()})
        entry = targets["h1_2"]["right"]
        self.assertEqual(entry.source, "hand")
        self.assertTrue(entry.valid and not entry.held)
        self.assertIn("fallback_reason", entry.info)
        self.assertAlmostEqual(entry.info["fallback_scale"], 0.7, places=9)

    def test_the_fallback_can_be_switched_off(self):
        """关掉兜底后，pose 缺失时保持的是上一帧的 pose 目标 —— 绝不悄悄换成手部映射。"""
        bridge = calibrated(sides=("right",), fallback_hand=False)
        targets = bridge.update(arms_world={"right": None}, hands={"right": open_hand()})
        entry = targets["h1_2"]["right"]
        self.assertTrue(entry.held)
        self.assertEqual(entry.info["source"], "pose")
        self.assertNotEqual(entry.source, "hand")

    def test_before_any_data_every_robot_sits_on_its_neutral_point(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=5)
        targets = bridge.update()
        for name in PAPER_ROBOTS:
            for side in SIDES:
                entry = targets[name][side]
                self.assertTrue(entry.held)
                self.assertFalse(entry.valid)
                np.testing.assert_allclose(entry.position,
                                           profile_for(name).realtime_neutral(side), atol=1e-9)
                self.assertIsNone(entry.orientation)



class ElbowBendTargetTests(unittest.TestCase):
    """``elbow_bend_target``：``--arm-elbow-follow operator`` 的那一份「操作者姿势」输入。

    口径与 ``elbow_bend_degrees`` 一致（180 = 完全伸直，0 = 完全折起），给的是**绝对**角：
    位置任务命令的形状就是这个角，而每台机器人加载姿势下自己的夹角各不相同，所以由
    ``PyBulletArmIK.posture_from_bend`` 拿绝对角减掉自己那个数（本项目现场录制：操作者的
    中性肘弯 138.9 度，而 g1 / gr1_t2 / h1_2 加载姿势下自己是 172.9 / 111.2 / 155.2 度）。
    """

    def test_nothing_is_offered_before_a_neutral_exists(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES)
        self.assertEqual(bridge.elbow_bend_target(), {"left": None, "right": None})
        bridge.update(arms_world={side: arm(side) for side in SIDES}, hands=None)
        self.assertEqual(bridge.elbow_bend_target(), {"left": None, "right": None})

    def test_a_straight_arm_reads_180(self):
        bridge = calibrated()
        for side in SIDES:
            self.assertAlmostEqual(bridge.elbow_bend_target()[side], 180.0, places=1)

    def test_a_bent_arm_reads_the_bend_it_has(self):
        bridge = calibrated()
        bridge.update(arms_world={side: bent_arm(side) for side in SIDES}, hands=None)
        bends = bridge.elbow_bend_target()
        for side in SIDES:
            # bent_arm 的前臂偏 (0, 0.06, 0.10)：与上臂夹角 15.5 度，所以 180 -> 164.5
            self.assertAlmostEqual(bends[side], 164.5, delta=0.5)

    def test_a_frame_without_pose_arms_keeps_the_last_bend(self):
        """断流那一帧沿用上一帧的姿势（与目标保持同口径）。

        必须这样：否则每丢一帧参考就弹回加载姿势，而「一断流肘就掉回去」正是现场那一件事。
        """
        bridge = calibrated()
        bridge.update(arms_world={side: bent_arm(side) for side in SIDES}, hands=None)
        before = bridge.elbow_bend_target()
        bridge.update(arms_world={"left": None, "right": None}, hands=None)
        self.assertEqual(bridge.elbow_bend_target(), before)

    def test_the_summary_reports_the_neutral_bend_and_this_frame_bend(self):
        bridge = calibrated()
        bridge.update(arms_world={side: bent_arm(side) for side in SIDES}, hands=None)
        payload = json.loads(json.dumps(bridge.summary()))
        for side in SIDES:
            self.assertAlmostEqual(payload["elbow_bend_deg"][side]["neutral"], 180.0,
                                   places=1)
            self.assertAlmostEqual(payload["elbow_bend_deg"][side]["current"], 164.5,
                                   delta=0.5)


class ReportAndIoTests(unittest.TestCase):
    """报告 / 诊断 / 中性记录的读写，以及构造参数校验。"""

    def test_summary_and_trace_are_json_able(self):
        bridge = calibrated(sides=("right",))
        bridge.update(arms_world={"right": arm("right", wrist_offset=(0.02, 0.0, 0.03))})
        payload = json.loads(json.dumps(bridge.summary()))
        self.assertIn("axis_map", payload)
        self.assertEqual(set(payload["travel_limit_m"]), set(PAPER_ROBOTS))
        self.assertEqual(set(payload["counts"]["valid"]), {"right"})
        trace = bridge.trace()
        self.assertEqual(set(trace), {"right"})
        self.assertEqual(trace["right"]["source"], "pose")
        self.assertEqual(len(trace["right"]["delta_op_cm"]), 3)
        self.assertAlmostEqual(trace["right"]["operator_reach_m"], 0.60, delta=0.01)

    def test_save_and_load_neutral_round_trip(self):
        bridge = calibrated(sides=SIDES)
        with tempfile.TemporaryDirectory() as folder:
            path = save_neutral(Path(folder) / "neutral.json", bridge.neutral_record())
            loaded = load_neutral(path)
        self.assertTrue(loaded["sides"]["right"]["complete"])
        np.testing.assert_allclose(loaded["sides"]["right"]["offset"], [0.0, 0.6, 0.0],
                                   atol=1e-6)
        reused = RealtimeArmBridge(PROFILES, sides=SIDES, neutral=loaded)
        self.assertFalse(reused.calibrating)

    def test_reset_can_keep_or_drop_the_calibration(self):
        bridge = calibrated(sides=("right",))
        bridge.reset()
        self.assertFalse(bridge.calibrating)
        self.assertEqual(bridge.frames, 0)
        bridge.reset(recalibrate=True)
        self.assertTrue(bridge.calibrating)

    def test_constructor_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=("middle",))
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, orientation="sometimes")
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=0)
        with self.assertRaises(TypeError):
            RealtimeArmBridge(["h1_2"], sides=SIDES)
        with self.assertRaises(ValueError):
            RealtimeArmBridge([], sides=SIDES)



class EntryAndReplayTests(unittest.TestCase):
    """入口参数默认值 + 回放源：不用相机也能把整条循环喂起来。"""

    def test_entry_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.side, "both")
        self.assertEqual(args.landmark_space, "world")
        self.assertEqual(args.arm_ik_iterations, 60)
        self.assertEqual(args.arm_calibrate_frames, 30)
        self.assertFalse(args.no_arms)
        self.assertIsNone(args.input_npz)

    def test_replay_source_feeds_frames_then_stops(self):
        frames = 3
        hand = open_hand()
        payload = {
            "timestamps": np.arange(frames) / 30.0,
            "left_hand": np.stack([hand] * frames),
            "right_hand": np.stack([np.full((21, 3), np.nan)] * frames),
            "world_left_hand": np.stack([(hand - hand[0]) * 0.19] * frames),
            "world_right_hand": np.stack([np.full((21, 3), np.nan)] * frames),
            "left_arm": np.stack([arm("left")] * frames),
            "right_arm": np.stack([np.full((3, 3), np.nan)] * frames),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.npz"
            np.savez_compressed(path, **payload)
            source = ReplayFrameSource(path, "normalized")
            self.assertEqual(source.frames, frames)
            for index in range(frames):
                source.next_frame()            # 手部窗口可能为 None，但不应抛异常
                self.assertEqual(source.last_raw_frame["timestamp"], index / 30.0)
            self.assertFalse(source.exhausted)
            self.assertIsNone(source.next_frame())
            self.assertTrue(source.exhausted)
            observation = source.arms()
        self.assertTrue(observation.valid["left"])
        self.assertFalse(observation.valid["right"])
        np.testing.assert_allclose(observation.world_arms["left"], arm("left"), atol=1e-6)
        self.assertIsNone(observation.world_arms["right"])
        self.assertIsNone(source.release())

    def test_camera_source_keeps_the_world_hands_that_the_bridge_reads(self):
        """回归用例（2026-11-16 现场）：``last_raw_frame`` 必须带 ``world_*``。

        ``run()`` 把 ``camera.last_raw_frame`` 同时喂给 ``RealtimeArmBridge`` 和
        ``--wrist-trace``；桥靠 `world_left/right` 造掌面基（``orientation_palm``），
        trace 靠它写 ``palm_space=world``。曾经 ``self.last_raw_frame`` 在合并
        ``world_*`` **之前**赋值，于是现场 985 帧里 ``palm_space`` 全是
        ``normalized``、``ori_source`` 全是 ``three_point`` —— 掌面基一次都没启用。
        回放源是合并后才赋值的，所以只有这条用例能拦住它。

        不碰相机：只替换 ``_input`` / ``pipeline`` 两个协作者。
        """
        def observation():
            hands = SimpleNamespace(
                hands={"left": np.zeros((21, 3)), "right": None},
                world_hands={"left": np.full((21, 3), 0.19), "right": None},
                timestamp=1234, metadata={"source": "stub"})
            return SimpleNamespace(hands=hands)

        source = object.__new__(CameraFrameSource)
        source.pose_enabled = True
        source._start_time = 0.0
        source.last_observation = None
        source._input = SimpleNamespace(next_teleop_observation=observation)
        source.pipeline = SimpleNamespace(process_window=lambda frame: frame)
        source.next_frame()

        self.assertIn("world_left", source.last_raw_frame)
        np.testing.assert_allclose(source.last_raw_frame["world_left"], 0.19)
        # 这一侧这一帧没检到，就必须是 None（不能伪造一个零阵）。
        self.assertIsNone(source.last_raw_frame["world_right"])

    def test_world_replay_declares_the_metadata_the_pipeline_demands(self):
        """world 口径回放必须自带队友处理器要的三个声明（world 米制 + source_scale）。

        回归用例：缺这几项时 ``MediaPipePalmLocalProcessor.process_frame`` 会在第一帧抛
        ``ValueError: World input must declare mediapipe_world_meters and
        world_length_unit='m'``，整场回放直接退出码 1（实测踩到，见 docs 第 8 节）。
        """
        frames = 3
        hand = open_hand()
        missing = np.full((21, 3), np.nan)
        payload = {
            "timestamps": np.arange(frames) / 30.0,
            "left_hand": np.stack([hand] * frames),
            "right_hand": np.stack([missing] * frames),
            "world_left_hand": np.stack([(hand - hand[0]) * 0.19] * frames),
            "world_right_hand": np.stack([missing] * frames),
            "left_arm": np.stack([arm("left")] * frames),
            "right_arm": np.stack([arm("right")] * frames),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.npz"
            np.savez_compressed(path, **payload)
            source = ReplayFrameSource(path, "world")
            for _ in range(frames):
                source.next_frame()          # 修好之前这里就抛 ValueError
            metadata = source.last_raw_frame["metadata"]
        self.assertEqual(metadata["source_landmark_space"], "mediapipe_normalized")
        self.assertEqual(metadata["world_landmark_space"], "mediapipe_world_meters")
        self.assertEqual(metadata["world_length_unit"], "m")
        self.assertEqual(metadata["source_scale"], 1.0)
        self.assertIn("right", metadata["world_invalid_reasons"])
        self.assertNotIn("left", metadata["world_invalid_reasons"])

    def test_world_replay_can_emit_native_palm_length_windows(self):
        """Dex consumes the same normalized three-frame contract as native hands."""
        from teleoperation.contracts.native_coordinates import (
            PALM_LENGTH_ALIGNMENT,
            PALM_LENGTH_PROFILE,
        )

        frames = 3
        hand = open_hand()
        missing = np.full((21, 3), np.nan)
        payload = {
            "timestamps": np.arange(frames) / 30.0,
            "left_hand": np.stack([hand] * frames),
            "right_hand": np.stack([missing] * frames),
            "world_left_hand": np.stack([(hand - hand[0]) * 0.19] * frames),
            "world_right_hand": np.stack([missing] * frames),
            "left_arm": np.stack([arm("left")] * frames),
            "right_arm": np.stack([arm("right")] * frames),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frames.npz"
            np.savez_compressed(path, **payload)
            source = ReplayFrameSource(
                path, "world", hand_input_profile=PALM_LENGTH_PROFILE)
            window = None
            for _ in range(frames):
                window = source.next_frame()
        self.assertIsNotNone(window)
        self.assertEqual(window.hands["left"].shape, (3, 25, 3))
        self.assertEqual(window.metadata["coordinate_alignment"],
                         PALM_LENGTH_ALIGNMENT)
        self.assertEqual(window.metadata["input_profile"], PALM_LENGTH_PROFILE)
        self.assertEqual(window.units, "1")


class WristOrientationFrameTests(unittest.TestCase):
    """腕坐标系（三点 -> 3x3）：右手系正交，直臂时说「不知道」。

    这是论文「手腕完整位姿」约束里我们做得到的那部分（见 arm_realtime 模块 docstring）。
    """

    def test_the_frame_is_orthonormal_right_handed_and_points_down_the_forearm(self):
        points = bent_arm("right")
        frame = wrist_orientation_frame(points)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=12)
        forearm = points[2] - points[1]
        np.testing.assert_allclose(frame[:, 0], forearm / np.linalg.norm(forearm),
                                   atol=1e-12)
        # z 是上臂 x 前臂：与上臂、前臂都垂直（肘的拐向）
        upper = points[1] - points[0]
        self.assertAlmostEqual(float(frame[:, 2] @ upper), 0.0, places=12)
        self.assertAlmostEqual(float(frame[:, 2] @ forearm), 0.0, places=12)

    def test_a_straight_or_squashed_arm_has_no_planar_frame(self):
        shoulder = np.array([0.17, -0.35, 0.0])
        straight = np.stack((shoulder, shoulder + UPPER_ARM, shoulder + 2.0 * UPPER_ARM))
        self.assertIsNone(wrist_orientation_frame(straight))
        self.assertIsNone(wrist_orientation_frame(None))
        self.assertIsNone(wrist_orientation_frame(np.zeros((3, 3))))

    def test_a_rotation_of_the_whole_arm_comes_back_as_the_relative_rotation(self):
        neutral = bent_arm("right")
        reference = wrist_orientation_frame(neutral)
        # 绕「中性腕坐标系的 z 轴」转 40 度：世界系里的轴是 reference[:, 2]，所以
        # 相对旋转在**中性帧自己的基**里就是绕 z 转 40 度（这正是我们要发给机器人的
        # 那个量：操作者的前臂 z 对应机器人中性 EE 的 z）。
        rotation = axis_angle(reference[:, 2], np.radians(40.0))
        moved = rotate_arm(neutral, rotation, neutral[0])
        relative = relative_wrist_rotation(moved, reference)
        np.testing.assert_allclose(relative, axis_angle((0.0, 0.0, 1.0), np.radians(40.0)),
                                   atol=1e-9)
        # 没动的时候是单位阵 -> 相对姿态是零旋转
        np.testing.assert_allclose(relative_wrist_rotation(neutral, reference), np.eye(3),
                                   atol=1e-12)

    def test_the_relative_rotation_needs_both_a_neutral_frame_and_a_valid_arm(self):
        self.assertIsNone(relative_wrist_rotation(bent_arm("right"), None))
        self.assertIsNone(relative_wrist_rotation(None, np.eye(3)))


class NeutralOrientationTests(unittest.TestCase):
    """中性记录里多存一份「腕坐标系」，姿态目标就成了**相对中性**的四元数。"""

    def _calibrated_on_a_bent_arm(self, frames: int = 4, **kwargs) -> RealtimeArmBridge:
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=frames,
                                   **kwargs)
        for _ in range(frames):
            bridge.update(arms_world={"right": bent_arm("right")})
        return bridge

    def test_sitting_still_gives_an_identity_relative_orientation(self):
        bridge = self._calibrated_on_a_bent_arm()
        entry = bridge.update(arms_world={"right": bent_arm("right")})["h1_2"]["right"]
        self.assertTrue(entry.valid)
        np.testing.assert_allclose(entry.orientation, (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        self.assertGreaterEqual(bridge.counts["oriented"]["right"], 1)
        self.assertTrue(bridge.summary()["neutral"]["right"]["orientation_recorded"])

    def test_rotating_the_forearm_sends_that_relative_rotation(self):
        from teleoperation.retargeting.arm.wrist_targets import (
            rotation_matrix_to_quaternion)

        # 这条钉的是**基的数学**（三点基下 35 度就该是 35 度），所以把角速度限速关掉：
        # 默认 frame_dt=1/30 s、540 度/秒 = 18 度/帧，一帧转 35 度会被削（见
        # test_a_wrist_that_jumps_is_rate_limited 那条）。限速本身另有四条用例。
        bridge = self._calibrated_on_a_bent_arm(max_orientation_rate_deg_s=0.0)
        neutral = bent_arm("right")
        reference = wrist_orientation_frame(neutral)
        # 绕中性帧的 z 轴（= 肘的拐向）转 -35 度；相对旋转在中性帧基里就是绕 z 转 -35 度。
        rotation = axis_angle(reference[:, 2], np.radians(-35.0))
        moved = rotate_arm(neutral, rotation, neutral[0])
        entry = bridge.update(arms_world={"right": moved})["h1_2"]["right"]
        expected = rotation_matrix_to_quaternion(
            axis_angle((0.0, 0.0, 1.0), np.radians(-35.0)))
        # 四元数有正负两种写法：取与期望同号的那个比较
        sign = 1.0 if float(np.dot(entry.orientation, expected)) >= 0.0 else -1.0
        np.testing.assert_allclose(np.asarray(entry.orientation) * sign, expected,
                                   atol=1e-6)
        # 姿态与位置互不干扰：位置仍是「中性 + 位移」
        self.assertIsNotNone(entry.position)
        self.assertIn("delta_base_m", entry.info)

    def test_only_a_pose_profile_gets_the_orientation_target(self):
        bridge = self._calibrated_on_a_bent_arm()
        neutral = bent_arm("right")
        rotation = axis_angle(wrist_orientation_frame(neutral)[:, 2], np.radians(30.0))
        moved = rotate_arm(neutral, rotation, neutral[0])
        targets = bridge.update(arms_world={"right": moved})
        self.assertIsNotNone(targets["h1_2"]["right"].orientation)
        for name in ("gr1_t2", "g1"):        # position-only profiles
            with self.subTest(robot=name):
                self.assertIsNone(targets[name]["right"].orientation)

    def test_the_robot_independent_orientation_target_is_available(self):
        """``orientation_target()`` 给的是「操作者那一份」，与哪台机器人无关。

        回归：调用方曾经用 ``per_side[sorted(per_side)[0]].orientation`` 取姿态，而
        ``sorted(['g1','gr1_t2','h1_2'])[0]`` 是 ``g1``（position-only -> None），
        于是 h1_2 的 pose 约束被静默丢掉（2026-10-12 现场那次就是这样）。
        """
        bridge = self._calibrated_on_a_bent_arm()
        neutral = bent_arm("right")
        rotation = axis_angle(wrist_orientation_frame(neutral)[:, 2], np.radians(25.0))
        targets = bridge.update(arms_world={"right": rotate_arm(neutral, rotation, neutral[0])})
        by_side = bridge.orientation_target()
        self.assertIsNotNone(by_side["right"])
        np.testing.assert_allclose(by_side["right"], targets["h1_2"]["right"].orientation,
                                   atol=1e-12)
        # 机器人无关：g1 / gr1_t2 的 target 里没有（position-only），但这一份有
        self.assertIsNone(targets["g1"]["right"].orientation)
        # orientation="off" 时一律 None
        bridge_off = RealtimeArmBridge(PROFILES, sides=("right",), orientation="off",
                                       neutral=bridge.neutral_record())
        bridge_off.update(arms_world={"right": neutral})
        self.assertIsNone(bridge_off.orientation_target()["right"])

    def test_a_dropped_hand_frame_reuses_the_last_palm_rotation(self):
        """掌面基丢一帧不要换基：换基等于把腕目标在「拧了 40 度」和「没拧」之间来回拨。

        现场取证见 ``arm_realtime.PALM_HOLD_S``（2026-11-16：``ori_source`` 逐帧在
        ``palm_normal`` / ``three_point`` 之间切，g1 左腕 pitch 在 -1.512 / +1.619 rad
        两颗限位间跳 = 抽搐）。
        """
        bridge = self._calibrated_with_a_palm_basis()
        twisted = self._palm_frame(bridge, {"world_right": twisted_hand()})
        self.assertEqual(twisted.info["orientation_source"], "palm_normal")
        # 掌面基量得到旋前：这一帧相对中性是绕前臂轴 40 度左右的旋转
        angle = _quaternion_angle_deg(twisted.orientation)
        self.assertGreater(angle, 30.0)
        self.assertLess(angle, 50.0)
        held = self._palm_frame(bridge, None)
        self.assertEqual(held.info["orientation_source"], "palm_hold")
        np.testing.assert_allclose(np.asarray(held.orientation),
                                   np.asarray(twisted.orientation), atol=1e-12)
        self.assertEqual(bridge.counts["palm_held"]["right"], 1)

    def test_the_palm_rotation_is_dropped_back_to_three_point_after_the_window(self):
        """窗口过了才如实退回三点基：不能无限保持，否则腕会锁在旧姿态上不跟手。"""
        bridge = self._calibrated_with_a_palm_basis()          # 每帧 0.1 s，窗口 0.5 s
        self._palm_frame(bridge, {"world_right": twisted_hand()})
        sources = [self._palm_frame(bridge, None).info["orientation_source"]
                   for _ in range(6)]                          # 年龄 0.1 .. 0.6 s
        self.assertEqual(sources[0], "palm_hold")              # 0.1 s
        self.assertEqual(sources[4], "palm_hold")              # 0.5 s（含边界）
        self.assertEqual(sources[5], "three_point")            # 0.6 s
        entry = self._palm_frame(bridge, None)
        self.assertEqual(entry.info["orientation_source"], "three_point")
        # 手臂一直没动，三点基看到的相对旋转就是单位阵（旋前它量不到）
        np.testing.assert_allclose(np.asarray(entry.orientation),
                                   (0.0, 0.0, 0.0, 1.0), atol=1e-9)

    def test_a_zero_window_goes_straight_back_to_three_point(self):
        """``palm_hold_s=0`` = 关（A/B 基线，与 2026-11-16 之前逐字一致）。"""
        bridge = self._calibrated_with_a_palm_basis(palm_hold_s=0.0)
        self._palm_frame(bridge, {"world_right": twisted_hand()})
        dropped = self._palm_frame(bridge, None)
        self.assertEqual(dropped.info["orientation_source"], "three_point")
        np.testing.assert_allclose(np.asarray(dropped.orientation),
                                   (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        self.assertEqual(bridge.counts["palm_held"]["right"], 0)

    def test_a_run_without_a_palm_basis_never_fabricates_one(self):
        """标定时没有手点 -> 中性里没有掌面基；之后丢帧也不许编一个出来。"""
        bridge = self._calibrated_on_a_bent_arm()
        entry = bridge.update(arms_world={"right": bent_arm("right")},
                              hands=None)["h1_2"]["right"]
        self.assertEqual(entry.info["orientation_source"], "three_point")
        self.assertEqual(bridge.counts["palm_held"]["right"], 0)

    def test_reset_forgets_the_held_palm_rotation(self):
        bridge = self._calibrated_with_a_palm_basis()
        self._palm_frame(bridge, {"world_right": twisted_hand()})
        bridge.reset()
        entry = self._palm_frame(bridge, None)
        self.assertEqual(entry.info["orientation_source"], "three_point")
        self.assertEqual(bridge.counts["palm_held"]["right"], 0)

    def _calibrated_with_a_palm_basis(self, palm_hold_s: float = PALM_HOLD_S,
                                      frame_dt: float = 0.1) -> RealtimeArmBridge:
        """标定时手也在画面里（未拧的掌面）-> 中性记录带上掌面基（``orientation_palm``）。

        这一步只钉**基**（掌面法向 / 保持窗口），所以把两条姿态上限都留在关的状态：
        40 度拧腕在 0.1 s 一帧下会被速率那条削成 54 度、被每帧步长那条削成 30 度，
        混进来就分不清「基算得对不对」和「限幅削了多少」（限幅自己见
        ``OrientationSlewTests`` / ``OrientationStepCapTests``）。
        """
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4,
                                   palm_hold_s=palm_hold_s, frame_dt=frame_dt,
                                   max_orientation_rate_deg_s=0.0,
                                   max_orientation_step_deg=0.0)
        for _ in range(4):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands={"world_right": open_hand()}, frame_dt=frame_dt)
        return bridge

    def _palm_frame(self, bridge, hands):
        """手臂不动、只手部数据变的一帧（``hands=None`` = 这一帧没手点）。"""
        return bridge.update(arms_world={"right": bent_arm("right")},
                             hands=hands, frame_dt=0.1)["h1_2"]["right"]

    def test_the_hand_fallback_carries_no_relative_orientation(self):
        bridge = self._calibrated_on_a_bent_arm()
        entry = bridge.update(arms_world={"right": None},
                              hands={"right": open_hand()})["h1_2"]["right"]
        self.assertEqual(entry.source, "hand")
        self.assertIsNone(entry.orientation)
        self.assertIn("no elbow", entry.info["orientation_source"])


class AdaptiveJitterFilterTests(unittest.TestCase):
    """MediaPipe 慢速噪声的 3 帧中值 + 自适应低通。"""

    DT = 0.14

    @staticmethod
    def _axis_quaternion(degrees_value):
        return np.array([np.sin(np.radians(degrees_value) / 2.0),
                         0.0, 0.0, np.cos(np.radians(degrees_value) / 2.0)])

    def test_a_single_pose_point_spike_is_removed(self):
        filter_ = AdaptiveArmPointFilter()
        base = np.array([[0.0, 0.0, 0.0],
                         [0.1, 0.0, 0.0],
                         [0.2, 0.0, 0.0]])
        for _ in range(3):
            filtered = filter_.update("right", base, self.DT)
        spike = base.copy()
        spike[2, 2] += 0.02
        filtered = filter_.update("right", spike, self.DT)
        np.testing.assert_allclose(filtered, base, atol=1e-12)

    def test_two_frames_of_real_motion_pass_the_point_filter(self):
        filter_ = AdaptiveArmPointFilter()
        base = np.array([[0.0, 0.0, 0.0],
                         [0.1, 0.0, 0.0],
                         [0.2, 0.0, 0.0]])
        stepped = base + np.array([0.0, 0.12, 0.0])
        for _ in range(3):
            filter_.update("right", base, self.DT)
        filter_.update("right", stepped, self.DT)
        filtered = filter_.update("right", stepped, self.DT)
        self.assertGreater(float(filtered[1, 1] - base[1, 1]), 0.09)

    def test_a_single_rotation_spike_is_removed_but_motion_passes(self):
        filter_ = AdaptiveRotationFilter()
        identity = self._axis_quaternion(0.0)
        turned = self._axis_quaternion(40.0)
        for _ in range(3):
            filter_.update("right", identity, self.DT)
        held = filter_.update("right", turned, self.DT)
        self.assertLess(quaternion_angle_deg(held, identity), 1e-6)
        filter_.update("right", turned, self.DT)
        moved = filter_.update("right", turned, self.DT)
        self.assertGreater(quaternion_angle_deg(moved, identity), 30.0)

    def test_the_cli_exposes_the_filter_and_defaults_it_on(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_jitter_filter, "on")


class WristLocalOrientationTests(unittest.TestCase):
    """``orientation_mode="wrist"``：前臂方向归位置任务，腕只跟局部旋转。

    现场「手臂映射准了，但手腕几乎没作用」的剩余机制是：旧口径把整只手相对中性的
    旋转原样交给只转腕的 IK。操作者正常摆臂时，前臂方向本来就会变，腕于是一直替
    肩/肘补偿，真正的手腕旋前/屈伸反而被限位和位置误差压住。
    """

    FRAMES = 4
    RATE_OFF = {"max_orientation_rate_deg_s": 0.0,
                "max_orientation_step_deg": 0.0}

    def _bridge(self, mode="wrist", with_hand=True):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",),
                                   calibration_frames=self.FRAMES,
                                   orientation="on", orientation_mode=mode,
                                   **self.RATE_OFF)
        for _ in range(self.FRAMES):
            bridge.update(
                arms_world={"right": bent_arm("right")},
                hands={"world_right": open_hand()} if with_hand else None,
                frame_dt=0.1)
        return bridge

    def _target(self, bridge, arm_points, hand_points):
        return bridge.update(
            arms_world={"right": arm_points},
            hands={"world_right": hand_points},
            frame_dt=0.1)["h1_2"]["right"]

    def test_rigid_arm_motion_keeps_the_local_wrist_at_zero(self):
        """抬臂/摆肘时手与前臂一起转：局部腕角必须仍是 0，不能塞给腕关节。"""
        neutral_arm = bent_arm("right")
        neutral_hand = open_hand()
        rotation = axis_angle((0.3, 0.2, 0.93), 0.9)
        moved_arm = rotate_arm(neutral_arm, rotation, neutral_arm[0])
        moved_hand = rotate_arm(neutral_hand, rotation, neutral_arm[0])

        local = self._target(self._bridge("wrist"), moved_arm, moved_hand)
        self.assertEqual(local.info["orientation_source"], "palm_normal")
        self.assertEqual(local.info["orientation_mode"], "wrist")
        self.assertLess(_quaternion_angle_deg(local.orientation), 1e-6)

        historic = self._target(self._bridge("hand"), moved_arm, moved_hand)
        self.assertEqual(historic.info["orientation_mode"], "hand")
        self.assertGreater(_quaternion_angle_deg(historic.orientation), 30.0)

    def test_a_real_hand_twist_is_still_sent_in_wrist_mode(self):
        """只绕前臂轴拧手：局部腕旋转保持不变，仍然要有 40 度。"""
        target = self._target(self._bridge("wrist"), bent_arm("right"),
                              twisted_hand("right", 40.0))
        self.assertEqual(target.info["orientation_source"], "palm_normal")
        self.assertAlmostEqual(_quaternion_angle_deg(target.orientation), 40.0,
                               delta=1.0)

    def test_both_hands_put_a_positive_twist_on_the_same_local_axis(self):
        """左手关键点绕序是镜像的；局部公式仍必须让两侧 +40° 落在同一个 x 轴。"""
        for side in ("left", "right"):
            bridge = RealtimeArmBridge(
                PROFILES, sides=(side,), calibration_frames=self.FRAMES,
                orientation="on", orientation_mode="wrist", **self.RATE_OFF)
            for _ in range(self.FRAMES):
                bridge.update(
                    arms_world={side: bent_arm(side)},
                    hands={f"world_{side}": open_hand()},
                    frame_dt=0.1)
            target = bridge.update(
                arms_world={side: bent_arm(side)},
                hands={f"world_{side}": twisted_hand(side, 40.0)},
                frame_dt=0.1)["h1_2"][side]
            with self.subTest(side=side):
                self.assertAlmostEqual(_quaternion_angle_deg(target.orientation),
                                       40.0, delta=1.0)
                self.assertGreater(float(target.orientation[0]), 0.30)
                self.assertLess(abs(float(target.orientation[1])), 1e-9)
                self.assertLess(abs(float(target.orientation[2])), 1e-9)

    def test_without_a_palm_the_local_mode_does_not_fabricate_three_point_motion(self):
        """三点基没有旋前维；局部口径宁可不发姿态，也不把前臂运动冒充成腕运动。"""
        target = self._target(self._bridge("wrist", with_hand=False),
                              bent_arm("right"), open_hand())
        self.assertIsNone(target.orientation)
        self.assertIsNone(target.info["orientation_source"])

    def test_a_late_reference_anchors_the_local_wrist_at_the_late_pose(self):
        """迟到补采发生在手臂已经移动后：那一刻的局部腕角仍必须是 0。"""
        bridge = RealtimeArmBridge(
            PROFILES, sides=("right",), calibration_frames=self.FRAMES,
            orientation="on", orientation_mode="wrist",
            palm_wait_frames=0, **self.RATE_OFF)
        for _ in range(self.FRAMES):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands=None, frame_dt=0.1)
        neutral = bent_arm("right")
        moved_arm = rotate_arm(
            neutral, axis_angle((0.2, 0.7, 0.3), 1.4), neutral[0])
        last = None
        for _ in range(PALM_LATE_FRAMES):
            last = bridge.update(
                arms_world={"right": moved_arm},
                hands={"world_right": open_hand()},
                frame_dt=0.1)["h1_2"]["right"]
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_reference_source"], "late")
        self.assertTrue(entry["orientation_palm_local_recorded"])
        self.assertLess(_quaternion_angle_deg(last.orientation), 1e-6)
        hand = open_hand()
        twisted_hand_points = rotate_arm(
            hand,
            axis_angle(moved_arm[2] - moved_arm[1], np.radians(40.0)),
            hand[0])
        twisted = bridge.update(
            arms_world={"right": moved_arm},
            hands={"world_right": twisted_hand_points},
            frame_dt=0.1)["h1_2"]["right"]
        self.assertAlmostEqual(_quaternion_angle_deg(twisted.orientation),
                               40.0, delta=1.0)

    def test_the_mode_name_is_validated(self):
        self.assertEqual(ORIENTATION_MODES, ("hand_base", "wrist", "hand"))
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=("right",),
                              orientation_mode="handish")
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_orientation_mode, "hand_base")
        self.assertEqual(args.arm_orientation_scope, "wrist")

    def test_the_helper_is_identity_under_a_rigid_transform(self):
        arm = bent_arm("right")
        hand = open_hand()
        forearm = wrist_orientation_frame(arm)
        palm = palm_orientation_frame(hand, arm, "right")
        rotation = axis_angle((0.0, 1.0, 0.0), 0.7)
        moved_arm = rotate_arm(arm, rotation, arm[0])
        moved_hand = rotate_arm(hand, rotation, arm[0])
        relative = relative_palm_wrist_rotation(
            moved_hand, moved_arm, forearm.T @ palm, "right")
        self.assertIsNotNone(relative)
        np.testing.assert_allclose(relative, np.eye(3), atol=1e-12)

    def test_hand_base_mode_exposes_the_current_palm_frame(self):
        """Absolute hand mode must give IK the current palm axes, not only a delta."""
        bridge = RealtimeArmBridge(
            PROFILES, sides=("right",), calibration_frames=self.FRAMES,
            orientation="on", orientation_mode="hand_base", **self.RATE_OFF)
        for _ in range(self.FRAMES):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands={"world_right": open_hand()}, frame_dt=0.1)
        hand = twisted_hand("right", 25.0)
        arm = bent_arm("right")
        bridge.update(arms_world={"right": arm},
                      hands={"world_right": hand}, frame_dt=0.1)
        expected = (bridge.axis_map.matrix()
                    @ anatomical_hand_frame(hand, arm, "right")
                    @ np.diag([1.0, -1.0, -1.0]))
        np.testing.assert_allclose(
            bridge.hand_frame_target()["right"], expected, atol=1e-12)
        bridge.update(arms_world={"right": arm}, hands=None, frame_dt=0.1)
        self.assertIsNone(bridge.hand_frame_target()["right"])

    def test_the_anatomical_frame_puts_thumb_side_on_positive_y_both_hands(self):
        for side in ("left", "right"):
            hand = open_hand()
            frame = anatomical_hand_frame(hand, bent_arm(side), side)
            thumb = hand[4] - hand[0]
            thumb /= np.linalg.norm(thumb)
            with self.subTest(side=side):
                self.assertGreater(float(frame[:, 1] @ thumb), 0.95)
                self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=12)


class OrientationSlewTests(unittest.TestCase):
    """腕姿态目标的角速度上限（``MAX_ORIENTATION_RATE_DEG_S`` / ``--arm-orientation-rate-deg-s``）。

    现场取证见那个常量：``PALM_HOLD_S`` 把「换基」那一下治住之后，同一场里仍有 1-2% 的掌面
    帧自己就是坏的（实测逐帧角速度 p99 740-833 度/秒、最大 1252，而人手腕约 500-700），
    外加「掌面基丢超窗口后退回三点基」那 2.6-3.0 rad 的真空跳。两者都不该在一帧里落到腕上。
    """

    def _bridge(self, **kwargs) -> RealtimeArmBridge:
        """带掌面基的桥，每帧 0.1 s（于是默认 540 度/秒 = 54 度/帧）。"""
        kwargs.setdefault("frame_dt", 0.1)
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4, **kwargs)
        for _ in range(4):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands={"world_right": open_hand()}, frame_dt=kwargs["frame_dt"])
        return bridge

    def _frame(self, bridge, hands, frame_dt: float = 0.1):
        return bridge.update(arms_world={"right": bent_arm("right")},
                             hands=hands, frame_dt=frame_dt)["h1_2"]["right"]

    def test_a_wrist_that_jumps_is_rate_limited(self):
        """一帧里手「拧了 150 度」（关键点跳变）-> 只放行 54 度，不是 150 度。"""
        # 这一步只钉**速率**那条：帧长 0.1 s -> 540 度/秒 = 54 度/帧，正好在每帧步长上限
        # （默认 30 度）之上，所以显式关掉步长那条，免得两条上限混在一起看不出是哪条削的。
        bridge = self._bridge(max_orientation_step_deg=0.0)
        self._frame(bridge, {"world_right": open_hand()})          # 中性：0 度
        jumped = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        angle = _quaternion_angle_deg(jumped.orientation)
        self.assertAlmostEqual(angle, 54.0, delta=1.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 1)
        # 同一条链路关掉限速时这一帧就是 150 度 —— 差别全在限速上，不是基算错了。
        raw_bridge = self._bridge(max_orientation_rate_deg_s=0.0)
        self._frame(raw_bridge, {"world_right": open_hand()})
        raw = self._frame(raw_bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(raw.orientation), 150.0, delta=1.0)
        self.assertEqual(raw_bridge.counts["orientation_clamped"]["right"], 0)

    def test_the_slew_lets_a_real_twist_through_untouched(self):
        """真手速之内（30 度/帧 = 300 度/秒）一个数都不许改：限速不是平滑。"""
        # 关掉每帧步长上限：这一步要证明的是「速率那条不介入」，30 度正好撞在步长上限上。
        bridge = self._bridge(max_orientation_step_deg=0.0)
        self._frame(bridge, {"world_right": open_hand()})
        for expected in (30.0, 60.0, 90.0):
            entry = self._frame(bridge, {"world_right": twisted_hand(angle_deg=expected)})
            self.assertAlmostEqual(_quaternion_angle_deg(entry.orientation), expected,
                                   delta=1e-6)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 0)

    def test_the_limit_is_scaled_by_the_real_frame_dt(self):
        """半秒一帧时 540 度/秒 = 270 度/帧，150 度就该原样过去；33 ms 一帧则只有 18 度。"""
        # 只钉速率那条：每帧步长上限（默认 30 度）与「按 dt 换算」是两件事，这里把它关掉，
        # 否则半秒一帧会被步长削成 30 度（那正是新加那条的行为，见 OrientationStepCapTests）。
        slow = self._bridge(max_orientation_step_deg=0.0)
        self._frame(slow, {"world_right": open_hand()}, frame_dt=0.5)
        passed = self._frame(slow, {"world_right": twisted_hand(angle_deg=150.0)}, frame_dt=0.5)
        self.assertAlmostEqual(_quaternion_angle_deg(passed.orientation), 150.0, delta=1.0)
        self.assertEqual(slow.counts["orientation_clamped"]["right"], 0)
        fast = self._bridge(max_orientation_step_deg=0.0)
        self._frame(fast, {"world_right": open_hand()}, frame_dt=1.0 / 30.0)
        cut = self._frame(fast, {"world_right": twisted_hand(angle_deg=150.0)},
                          frame_dt=1.0 / 30.0)
        self.assertAlmostEqual(_quaternion_angle_deg(cut.orientation), 18.0, delta=0.5)
        self.assertEqual(fast.counts["orientation_clamped"]["right"], 1)

    def test_a_clamped_spike_does_not_bounce_back(self):
        """尖峰后被削掉的角度**不累积**：手回到中性，目标一步回到 0，不会过冲到来回振。"""
        bridge = self._bridge()
        self._frame(bridge, {"world_right": open_hand()})
        self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        back = self._frame(bridge, {"world_right": open_hand()})
        self.assertLessEqual(_quaternion_angle_deg(back.orientation), 1.0)
        settled = self._frame(bridge, {"world_right": open_hand()})
        self.assertLessEqual(_quaternion_angle_deg(settled.orientation), 1e-6)

    def test_the_hold_window_keeps_the_raw_measurement_not_the_clamped_one(self):
        """`_palm_hold` 必须存**原始测量**：存削过的值 = 一次坏帧把随后的保持一起压低。"""
        # 只钉速率那条（0.1 s -> 54 度），步长那条关掉，否则两次都被削成 30 度看不出区别。
        bridge = self._bridge(max_orientation_step_deg=0.0)
        self._frame(bridge, {"world_right": open_hand()})
        cut = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(cut.orientation), 54.0, delta=1.0)
        # 下一帧丢手点 -> 沿用「上一次掌面旋转」= 原始的 150 度，再被限速削一次
        held = self._frame(bridge, None)
        self.assertEqual(held.info["orientation_source"], "palm_hold")
        self.assertEqual(bridge.counts["palm_held"]["right"], 1)
        self.assertGreater(_quaternion_angle_deg(held.orientation), 100.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 2)

    def test_a_zero_rate_turns_the_slew_off(self):
        """``max_orientation_rate_deg_s=0`` = 关（A/B 基线，与 2026-11-16 之前逐字一致）。"""
        bridge = self._bridge(max_orientation_rate_deg_s=0.0)
        self._frame(bridge, {"world_right": open_hand()})
        entry = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(entry.orientation), 150.0, delta=1.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 0)

    def test_reset_forgets_the_slew_baseline(self):
        """``reset()`` 之后第一帧不跟上一场比（换人 / 重标定后大幅转向是正常的）。"""
        bridge = self._bridge()
        self._frame(bridge, {"world_right": open_hand()})
        self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        bridge.reset()
        entry = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 0)
        self.assertIsNotNone(entry.orientation)

    def test_the_entry_parser_exposes_the_slew_flag(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_orientation_rate_deg_s, MAX_ORIENTATION_RATE_DEG_S)
        args = build_parser().parse_args(["--arm-orientation-rate-deg-s", "0"])
        self.assertEqual(args.arm_orientation_rate_deg_s, 0.0)


class OrientationStepCapTests(unittest.TestCase):
    """每帧步长上限（``MAX_ORIENTATION_STEP_DEG`` / ``--arm-orientation-step-deg``）。

    现场取证见那个常量：速率上限是**帧率盲**的 —— 2026-11-16 晚那场实跑只有 5.92 fps
    （``median_frame_interval_ms`` 155.4，``outputs/tmp_realtime/arm_run_20261116_213650.json``），
    540 度/秒 x 0.155 s = **84 度/帧**，于是一帧 150 度的坏掌面帧仍被放行 84 度。这一条与
    帧长无关：任何帧率下一帧都不超过 30 度。下面用真实的 155 ms 帧长钉住这两件事。
    """

    #: 现场实测的帧长（5.92 fps）。用它而不是 0.1 s，是因为这条上限存在的理由就是低帧率。
    FIELD_DT = 0.155

    def _bridge(self, **kwargs) -> RealtimeArmBridge:
        kwargs.setdefault("frame_dt", self.FIELD_DT)
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4, **kwargs)
        for _ in range(4):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands={"world_right": open_hand()}, frame_dt=kwargs["frame_dt"])
        return bridge

    def _frame(self, bridge, hands, frame_dt: float = None):
        return bridge.update(arms_world={"right": bent_arm("right")},
                             hands=hands,
                             frame_dt=self.FIELD_DT if frame_dt is None
                             else frame_dt)["h1_2"]["right"]

    def test_at_6_fps_a_150_deg_jump_moves_30_deg_not_84(self):
        """5.92 fps 下速率那条会放行 84 度，步长那条把它压到 30 度。"""
        bridge = self._bridge()
        self._frame(bridge, {"world_right": open_hand()})
        jumped = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(jumped.orientation),
                               MAX_ORIENTATION_STEP_DEG, delta=1.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 1)
        # 同一条链路关掉步长那条 -> 回到 540 x 0.155 = 83.7 度：差别全在步长上限上。
        rate_only = self._bridge(max_orientation_step_deg=0.0)
        self._frame(rate_only, {"world_right": open_hand()})
        bigger = self._frame(rate_only, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(bigger.orientation), 83.7, delta=1.5)

    def test_above_18_fps_the_step_cap_never_binds(self):
        """30 fps 下速率只放 18 度/帧，30 度的步长上限一次都不该介入。"""
        bridge = self._bridge(frame_dt=1.0 / 30.0)
        self._frame(bridge, {"world_right": open_hand()}, frame_dt=1.0 / 30.0)
        cut = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)},
                          frame_dt=1.0 / 30.0)
        self.assertAlmostEqual(_quaternion_angle_deg(cut.orientation), 18.0, delta=0.5)
        # 真手速（15 度/帧 = 450 度/秒）一个数都不改，两条上限都不介入。
        slow = self._bridge(frame_dt=1.0 / 30.0)
        self._frame(slow, {"world_right": open_hand()}, frame_dt=1.0 / 30.0)
        real = self._frame(slow, {"world_right": twisted_hand(angle_deg=15.0)},
                           frame_dt=1.0 / 30.0)
        self.assertAlmostEqual(_quaternion_angle_deg(real.orientation), 15.0, delta=1e-6)
        self.assertEqual(slow.counts["orientation_clamped"]["right"], 0)

    def test_a_sustained_spike_is_walked_out_at_the_cap_per_frame(self):
        """连续 3 帧都要求 150 度 -> 90 度（每帧 30），而不是一帧 84 度再被拉回。"""
        bridge = self._bridge()
        self._frame(bridge, {"world_right": open_hand()})
        angles = [_quaternion_angle_deg(self._frame(
            bridge, {"world_right": twisted_hand(angle_deg=150.0)}).orientation)
            for _ in range(3)]
        for index, angle in enumerate(angles, start=1):
            self.assertAlmostEqual(angle, MAX_ORIENTATION_STEP_DEG * index, delta=1.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 3)

    def test_a_zero_step_keeps_the_rate_limit_only(self):
        """``max_orientation_step_deg=0`` = 只按速率（这条新上限的关法）。"""
        bridge = self._bridge(max_orientation_step_deg=0.0)
        self._frame(bridge, {"world_right": open_hand()})
        entry = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(entry.orientation), 83.7, delta=1.5)

    def test_a_zero_rate_switches_the_whole_mechanism_off(self):
        """``max_orientation_rate_deg_s=0`` 时步长上限**也不能**单独接管（A/B 基线逐位透传）。"""
        bridge = self._bridge(max_orientation_rate_deg_s=0.0)
        self._frame(bridge, {"world_right": open_hand()})
        entry = self._frame(bridge, {"world_right": twisted_hand(angle_deg=150.0)})
        self.assertAlmostEqual(_quaternion_angle_deg(entry.orientation), 150.0, delta=1.0)
        self.assertEqual(bridge.counts["orientation_clamped"]["right"], 0)

    def test_the_entry_parser_exposes_the_step_cap(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_orientation_step_deg, MAX_ORIENTATION_STEP_DEG)
        off = build_parser().parse_args(["--arm-orientation-step-deg", "0"])
        self.assertEqual(off.arm_orientation_step_deg, 0.0)
        custom = build_parser().parse_args(["--arm-orientation-step-deg", "12.5"])
        self.assertEqual(custom.arm_orientation_step_deg, 12.5)

    def test_the_constructor_clamps_a_negative_step_to_off(self):
        """负数是「关」，不是反向限幅。"""
        bridge = self._bridge(max_orientation_step_deg=-5.0)
        self.assertEqual(bridge.max_orientation_step_deg, 0.0)


class FrameBudgetTests(unittest.TestCase):
    """``--frame-budget-json``（``FrameBudget``）：回答「5.92 fps 花在哪」。

    2026-11-16 晚那场实跑只有 5.92 fps，而报告里只有整帧的 ``median_frame_interval_ms``，
    分不出是取帧/姿态模型、IK、物理步进还是渲染。这一类只钉「算得对、写得进、不给旗子
    就不构造」，判读交给现场那一张表。
    """

    def test_stats_and_share_are_the_plain_arithmetic(self):
        budget = FrameBudget(Path(tempfile.gettempdir()) / "frame_budget_probe.json")
        budget.samples["ik"] = [0.010, 0.020]          # 10 ms / 20 ms -> 均值 15
        budget.samples["capture"] = [0.005, 0.005]      # 均值 5
        budget.intervals = [0.050, 0.050]               # 整帧均值 50 ms -> 20 fps
        data = budget.summary()
        self.assertEqual(data["frames"], 2)
        self.assertAlmostEqual(data["fps"], 20.0, places=6)
        self.assertAlmostEqual(data["frame"]["mean_ms"], 50.0, places=6)
        self.assertAlmostEqual(data["stages"]["ik"]["mean_ms"], 15.0, places=6)
        self.assertAlmostEqual(data["stages"]["ik"]["p50_ms"], 10.0, places=6)
        self.assertAlmostEqual(data["stages"]["ik"]["max_ms"], 20.0, places=6)
        self.assertAlmostEqual(data["share_percent"]["ik"], 30.0, places=6)
        self.assertAlmostEqual(data["share_percent"]["capture"], 10.0, places=6)
        # 没记过的段必须是 0，不是缺键：报告里要能一眼看出「这段没跑」。
        self.assertEqual(data["stages"]["preview"]["count"], 0)
        self.assertEqual(data["stages"]["preview"]["mean_ms"], 0.0)

    def test_an_empty_budget_does_not_divide_by_zero(self):
        """一帧都没跑完（比如刚启动就关窗）：fps 是 None，不是 ZeroDivisionError。"""
        data = FrameBudget(Path(tempfile.gettempdir()) / "frame_budget_empty.json").summary()
        self.assertEqual(data["frames"], 0)
        self.assertIsNone(data["fps"])
        self.assertEqual(data["stages"]["ik"]["mean_ms"], 0.0)

    def test_lap_returns_the_next_starting_point(self):
        budget = FrameBudget(Path(tempfile.gettempdir()) / "frame_budget_lap.json")
        start = budget.lap("capture", 0.0)
        self.assertGreater(start, 0.0)
        self.assertEqual(len(budget.samples["capture"]), 1)
        self.assertEqual(len(budget.samples["ik"]), 0)

    def test_table_prints_one_line_per_stage(self):
        budget = FrameBudget(Path(tempfile.gettempdir()) / "frame_budget_table.json")
        budget.samples["ik"] = [0.020]
        budget.intervals = [0.100]
        text = FrameBudget.table(budget.summary())
        lines = text.splitlines()
        self.assertIn("frame budget: 1 frames", lines[0])
        self.assertIn("10.00 fps", lines[0])
        for stage in FrameBudget.STAGES:
            self.assertTrue(any(line.strip().startswith(stage) for line in lines[1:]),
                            f"missing stage row: {stage}")

    def test_save_writes_the_same_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.json"
            budget = FrameBudget(path)
            budget.samples["ik"] = [0.020]
            budget.intervals = [0.100]
            data = budget.save()
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["fps"], data["fps"])

    def test_the_entry_parser_exposes_the_budget_flag(self):
        self.assertIsNone(build_parser().parse_args([]).frame_budget_json)
        args = build_parser().parse_args(["--frame-budget-json", "budget.json"])
        self.assertEqual(str(args.frame_budget_json), "budget.json")


class NeutralOrientationRecordTests(unittest.TestCase):
    """老中性记录不带姿态 -> 退化成位置模式；新记录带姿态 -> 存读一致。"""

    def test_an_old_neutral_record_still_works_as_position_only(self):
        payload = {"format": "realtime_arm_neutral_v1",
                   "sides": {"right": {"offset": [0.0, 0.6, 0.0], "reach": 0.6,
                                       "samples": 4, "complete": True}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old_neutral.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_neutral(path)
            self.assertTrue(loaded["sides"]["right"]["complete"])
            self.assertIsNone(loaded["sides"]["right"]["orientation"])
            bridge = RealtimeArmBridge(PROFILES, sides=("right",), neutral=loaded)
            entry = bridge.update(arms_world={"right": bent_arm("right")})["h1_2"]["right"]
        self.assertTrue(entry.valid)
        self.assertIsNone(entry.orientation)
        self.assertEqual(bridge.counts["oriented"]["right"], 0)

    def test_the_neutral_frame_survives_a_save_and_load_round_trip(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=4)
        for _ in range(4):
            bridge.update(arms_world={"right": bent_arm("right")})
        reference = wrist_orientation_frame(bent_arm("right"))
        with tempfile.TemporaryDirectory() as folder:
            path = save_neutral(Path(folder) / "neutral.json", bridge.neutral_record())
            loaded = load_neutral(path)
            reused = RealtimeArmBridge(PROFILES, sides=("right",), neutral=loaded)
            entry = reused.update(arms_world={"right": bent_arm("right")})["h1_2"]["right"]
        np.testing.assert_allclose(loaded["sides"]["right"]["orientation"], reference,
                                   atol=1e-9)
        np.testing.assert_allclose(entry.orientation, (0.0, 0.0, 0.0, 1.0), atol=1e-9)


# --------------------------------------------------------------------------- #
# 2026-10-13 现场症状的两条出口：两段式映射（--arm-map segments）与肘位置任务
# --------------------------------------------------------------------------- #
def segment_links(upper: float = 0.30, fore: float = 0.20) -> dict:
    """每台机器人每侧一个假连杆长（米）：只测两段式映射的算术，不接仿真。"""
    return {name: {side: {"upper": upper, "fore": fore} for side in SIDES}
            for name in PAPER_ROBOTS}


def segment_anchors() -> dict:
    """每台每侧的中性肘/腕点（基座系，米）：只测肘目标长得对不对。"""
    return {name: {side: {"elbow": [0.10, 0.20, 0.05], "wrist": [0.10, 0.45, 0.05]}
                   for side in SIDES}
            for name in PAPER_ROBOTS}


def swung_bent_arm(side: str, angle: float) -> np.ndarray:
    """弯肘中性姿势上把上臂绕「肩->腕」轴转 angle、腕点不动：现场那个「只摆大臂」。

    必须用弯肘的姿势：直臂时上臂方向就**在**「肩->腕」轴上，绕它转等于没转（位移恒为 0），
    那是几何事实，不是 bug。现场「握拳贴胸」正是弯肘的。
    """
    points = bent_arm(side)
    shoulder, elbow, wrist = points
    return np.stack((shoulder,
                     shoulder + axis_angle(wrist - shoulder, angle) @ (elbow - shoulder),
                     wrist))


def calibrated_bent(sides=SIDES, frames: int = 4, **kwargs) -> RealtimeArmBridge:
    """同 ``calibrated()``，但中性姿势是**弯肘**的（见 ``swung_bent_arm``）。"""
    bridge = RealtimeArmBridge(PROFILES, sides=sides, calibration_frames=frames, **kwargs)
    for _ in range(frames):
        bridge.update(arms_world={side: bent_arm(side) for side in sides})
    return bridge


class SegmentsMappingTests(unittest.TestCase):
    """只动大臂这一帧：chain 给不出目标（现场零反应），segments 给得出；肘任务再把它交给 IK。"""

    def test_only_swinging_the_upper_arm_gives_an_empty_target_with_chain(self):
        # 现场症状的复现：手留在胸前、只把肘摆开，历史口径下目标**一点都不动**。
        bridge = calibrated_bent(sides=("right",), map_mode="chain")
        neutral = profile_for("h1_2").realtime_neutral("right")
        entry = bridge.update(arms_world={"right": swung_bent_arm("right", 0.8)})["h1_2"]["right"]
        self.assertEqual(entry.info["map"], "chain")
        np.testing.assert_allclose(entry.position, neutral, atol=1e-9)

    def test_the_same_frame_does_move_the_target_with_segments(self):
        bridge = calibrated_bent(sides=("right",), map_mode="segments",
                                 robot_links=segment_links())
        neutral = np.asarray(profile_for("h1_2").realtime_neutral("right"), dtype=float)
        entry = bridge.update(arms_world={"right": swung_bent_arm("right", 0.8)})["h1_2"]["right"]
        self.assertEqual(entry.info["map"], "segments")
        self.assertGreater(float(np.linalg.norm(np.asarray(entry.position) - neutral)),
                           0.005)

    def test_segments_falls_back_to_chain_when_link_lengths_are_missing(self):
        bridge = calibrated_bent(sides=("right",), map_mode="segments")
        entry = bridge.update(arms_world={"right": arm("right")})["h1_2"]["right"]
        self.assertEqual(entry.info["map"], "chain (segments unavailable)")

    def test_elbow_target_is_the_upper_arm_term_added_to_the_anchor(self):
        bridge = calibrated_bent(sides=("right",), map_mode="segments",
                                 robot_links=segment_links(), elbow_task=True,
                                 anchors=segment_anchors())
        entry = bridge.update(arms_world={"right": swung_bent_arm("right", 0.8)})["h1_2"]["right"]
        anchor = np.asarray(segment_anchors()["h1_2"]["right"]["elbow"], dtype=float)
        self.assertEqual(entry.info["elbow_task"], "on")
        self.assertIsNotNone(entry.elbow)
        # info 里那份是给报告用的、保留 4 位小数，所以容差给 2e-4 而不是 1e-5。
        np.testing.assert_allclose(np.asarray(entry.elbow) - anchor,
                                   np.asarray(entry.info["elbow_delta_base_m"]),
                                   atol=2e-4)
        # 肘目标动了几十毫米，而同一帧的腕目标几乎不动 —— 这正是「只摆大臂」该有的样子。
        self.assertGreater(float(np.linalg.norm(np.asarray(entry.elbow) - anchor)), 0.005)

    def test_elbow_target_survives_a_neutral_record_round_trip(self):
        bridge = calibrated_bent(sides=("right",), map_mode="segments",
                                 robot_links=segment_links(), elbow_task=True,
                                 anchors=segment_anchors())
        with tempfile.TemporaryDirectory() as folder:
            save_neutral(Path(folder) / "neutral.json", bridge.neutral_record())
            loaded = load_neutral(Path(folder) / "neutral.json")
        self.assertIsNotNone(loaded["sides"]["right"]["upper"])
        self.assertIsNotNone(loaded["sides"]["right"]["fore"])
        reused = RealtimeArmBridge(PROFILES, sides=("right",), map_mode="segments",
                                   robot_links=segment_links(), elbow_task=True,
                                   anchors=segment_anchors(), neutral=loaded)
        entry = reused.update(arms_world={"right": swung_bent_arm("right", 0.8)})["h1_2"]["right"]
        self.assertEqual(entry.info["map"], "segments")
        self.assertEqual(entry.info["elbow_task"], "on")

    def test_elbow_task_without_the_segments_map_is_refused(self):
        # 两个目标必须出自同一条按连杆长缩放的链，否则它们互相打架（实测腕残差 107 mm）。
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=("right",), elbow_task=True)

    def test_elbow_task_without_anchor_is_recorded_not_guessed(self):
        bridge = calibrated_bent(sides=("right",), map_mode="segments",
                                 robot_links=segment_links(), elbow_task=True)
        entry = bridge.update(arms_world={"right": swung_bent_arm("right", 0.8)})["h1_2"]["right"]
        self.assertIsNone(entry.elbow)
        self.assertTrue(entry.info["elbow_task"].startswith("off"))


# --------------------------------------------------------------------------- #
# 2026-11-17：k（机器人段长 / 你的段长）的左右对称性与「坏标定不许进链路」
# --------------------------------------------------------------------------- #
def foreshortened_arm(side: str, upper: float = 0.30, fore: float = 0.20) -> np.ndarray:
    """段长可调的米制 (肩, 肘, 腕)：``fore=0.06`` 复现"深度压扁"的那一场标定。

    现场依据（2026-11-17 三场 live 的 ``segment_scales``）：同一个人同一天量出来的
    ``k_前臂`` 是 1.02 / 1.34 / **3.79**（h1_2 前臂 0.229 m ÷ 你的 0.060 m），而同一份
    记录的右臂一直是 0.99（0.23 m）。一个人左右前臂不可能差 4 倍 —— 所以「左手不如右手」
    的头号嫌疑是**标定那一场左臂的关键点被深度压扁**，不是求解器也不是机器人。
    """
    sign = 1.0 if side == "left" else -1.0
    shoulder = np.array([0.17 * sign, -0.35, 0.0])
    elbow = shoulder + np.array([0.0, upper, 0.0])
    wrist = elbow + np.array([0.0, fore, 0.0])
    return np.stack((shoulder, elbow, wrist))


class SegmentScaleTrustTests(unittest.TestCase):
    """``k = 机器人段长 / 你的段长``：姿势变了它不变，标定坏了它不许生效。"""

    def test_the_same_arm_held_differently_gives_the_same_k(self):
        # 绕肩转 45.8 度是**刚体**运动：段长不变，k 就不该变。
        straight = calibrated_bent(sides=SIDES, robot_links=segment_links())
        rotated = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                    robot_links=segment_links())
        for _ in range(4):
            rotated.update(arms_world={"left": swung_bent_arm("left", 0.8),
                                       "right": bent_arm("right")})
        for name in PAPER_ROBOTS:
            profile = profile_for(name)
            with self.subTest(robot=name):
                np.testing.assert_allclose(rotated._segment_scales(profile, "left"),
                                           straight._segment_scales(profile, "right"),
                                           atol=1e-9)

    def test_both_sides_of_a_symmetric_neutral_give_the_same_k(self):
        bridge = calibrated_bent(sides=SIDES, robot_links=segment_links())
        for name in PAPER_ROBOTS:
            profile = profile_for(name)
            with self.subTest(robot=name):
                np.testing.assert_allclose(bridge._segment_scales(profile, "left"),
                                           bridge._segment_scales(profile, "right"),
                                           atol=1e-9)
                self.assertEqual(bridge.segment_scale_note("left"), "")
                self.assertEqual(bridge.segment_scale_note("right"), "")

    def test_a_foreshortened_forearm_disables_that_side_only(self):
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.06),
                                      "right": foreshortened_arm("right", fore=0.20)})
        profile = profile_for("h1_2")
        # 0.06 m 的前臂 -> h1_2 的 k 会是 3.8 倍：判它不可用，退回整段 gain。
        self.assertIsNone(bridge._segment_scales(profile, "left"))
        self.assertIn("forearm", bridge.segment_scale_note("left"))
        self.assertIn("0.060", bridge.segment_scale_note("left"))
        # 另一侧照旧 —— 左右互不牵连（判据只看**你的**段长，与机器人无关）。
        np.testing.assert_allclose(bridge._segment_scales(profile, "right"),
                                   (0.30 / 0.30, 0.20 / 0.20), atol=1e-9)
        self.assertEqual(bridge.segment_scale_note("right"), "")
        # 报告里必须留痕：事后才知道「那一场 k 为什么没生效」。
        notes = bridge.summary()["segment_scale_notes"]
        self.assertEqual(notes["right"], "")
        self.assertIn("0.060", notes["left"])

    def test_a_bad_side_falls_back_to_the_whole_segment_recipe(self):
        bridge = RealtimeArmBridge(PROFILES, sides=("left",), calibration_frames=4,
                                   map_mode="segments", robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.06)})
        entry = bridge.update(arms_world={"left": swung_bent_arm("left", 0.8)})["h1_2"]["left"]
        # 不是崩、不是静默 —— 如实写清楚这一帧走的是哪条口径。
        self.assertEqual(entry.info["map"], "chain (segments unavailable)")
        self.assertTrue(np.isfinite(entry.position).all())

    def test_the_live_loop_warns_once_per_side(self):
        # 现场那条 [WARN] 是一次性的：静默退回才是最坏的结果（人只会觉得"今天这机器不对劲"）。
        from teleoperation.simulation.realtime_teleop import _warn_bad_segment_scales

        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.06),
                                      "right": foreshortened_arm("right", fore=0.20)})
        warned = set()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _warn_bad_segment_scales(bridge, SIDES, warned)
            _warn_bad_segment_scales(bridge, SIDES, warned)
        text = output.getvalue()
        self.assertEqual(text.count("[WARN] segment scales OFF"), 1)
        self.assertIn("(left)", text)
        self.assertEqual(warned, {"left"})
    def test_a_symmetric_neutral_has_no_asymmetry_note(self):
        bridge = calibrated_bent(sides=SIDES, robot_links=segment_links())
        self.assertEqual(bridge.segment_asymmetry_note(), "")
        self.assertEqual(bridge.summary()["segment_asymmetry_note"], "")

    def test_an_in_band_but_lopsided_forearm_is_flagged(self):
        # 现场最常见的一次坏标定：0.17 vs 0.23 m —— **两端都在带宽内**，带宽护栏一声不吭，
        # 但 k_前臂 从 0.99 变 1.35，前臂位移被放大 35%（那一场左 residual 是右的 1.9 倍）。
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.17),
                                      "right": foreshortened_arm("right", fore=0.23)})
        # 带宽护栏不管这一档（两个值都在 0.12-0.45 里）。
        self.assertEqual(bridge.segment_scale_note("left"), "")
        self.assertEqual(bridge.segment_scale_note("right"), "")
        note = bridge.segment_asymmetry_note()
        self.assertIn("1.35x", note)
        self.assertIn("forearm", note)
        self.assertIn("0.170/0.230", note)
        # 深度压扁只会让人量得**更短**，所以短的那一侧才是嫌疑人。
        self.assertIn("(left) is the suspect", note)
        self.assertEqual(bridge.summary()["segment_asymmetry_note"], note)

    def test_asymmetry_is_reported_but_never_changes_the_control(self):
        # 只告警不改控制：单次采集分不清"左手量短了"还是"右手量长了"，静默换配方
        # 就是又一次把配方偷偷换掉。两侧的 k 必须照旧可用。
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   map_mode="segments", robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.17),
                                      "right": foreshortened_arm("right", fore=0.23)})
        profile = profile_for("h1_2")
        self.assertIsNotNone(bridge._segment_scales(profile, "left"))
        self.assertIsNotNone(bridge._segment_scales(profile, "right"))
        self.assertTrue(bridge.segment_asymmetry_note())
        entry = bridge.update(arms_world={side: swung_bent_arm(side, 0.8)
                                          for side in SIDES})["h1_2"]
        for side in SIDES:
            self.assertNotIn("unavailable", entry[side].info["map"])

    def test_a_lopsided_upper_arm_names_that_side_not_the_forearm(self):
        # 文案里「短的是嫌疑人」必须按**最不对称的那一段**判：上臂不对称、前臂对称时，
        # 若还按前臂说「the shorter one」就会把人指到错的地方去改标定。
        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", upper=0.22, fore=0.23),
                                      "right": foreshortened_arm("right", upper=0.34,
                                                                 fore=0.23)})
        note = bridge.segment_asymmetry_note()
        self.assertIn("upper arm", note)
        self.assertIn("the shorter one (left) is the suspect", note)
        # 反向：上臂左长右短 -> 嫌疑人换成右侧（不是恒定指向左边）。
        flipped = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                    robot_links=segment_links())
        for _ in range(4):
            flipped.update(arms_world={"left": foreshortened_arm("left", upper=0.34,
                                                                 fore=0.23),
                                       "right": foreshortened_arm("right", upper=0.22,
                                                                  fore=0.23)})
        self.assertIn("the shorter one (right) is the suspect",
                      flipped.segment_asymmetry_note())

    def test_the_asymmetry_warning_needs_both_sides_and_fires_once(self):
        from teleoperation.simulation.realtime_teleop import _warn_asymmetric_segment_scales

        bridge = RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=4,
                                   robot_links=segment_links())
        for _ in range(4):
            bridge.update(arms_world={"left": foreshortened_arm("left", fore=0.17),
                                      "right": foreshortened_arm("right", fore=0.23)})
        warned = set()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            # 只跑一侧时不该喊：比的是"你的左臂 vs 你的右臂"。
            _warn_asymmetric_segment_scales(bridge, ("left",), warned)
            self.assertEqual(output.getvalue(), "")
            _warn_asymmetric_segment_scales(bridge, SIDES, warned)
            _warn_asymmetric_segment_scales(bridge, SIDES, warned)
        text = output.getvalue()
        self.assertEqual(text.count("[WARN] left/right calibration mismatch"), 1)
        self.assertIn("1.35x", text)




# --------------------------------------------------------------------------- #
# 2026-10-13 晚：方向口径（--arm-map posture）—— 现场判定两种位移口径都不像之后加的
# --------------------------------------------------------------------------- #
#: 每台一套**不同**的假连杆长（米，与实测量级一致）：方向口径的幅度不该随臂长变。
POSTURE_LINKS = {"g1": (0.193, 0.184), "gr1_t2": (0.280, 0.250), "h1_2": (0.331, 0.229)}
#: 这些用例喂帧用的帧长（秒）：现场摄像头 8-12 fps，比名义 1/30 s 更接近真值。
#: 跳变护栏的上限按帧长换算（600 度/秒 x 0.25 s = 150 度/帧），所以「一帧转 40-60 度」
#: 在这里是正常手速、不该被拦；护栏本身另有一组用例（DirectionGlitchGuardTests）。
TEST_FRAME_DT = 0.25


def posture_links() -> dict:
    """每台每侧一套假连杆长（米）：只测方向口径的算术，不接仿真。"""
    return {name: {side: {"upper": POSTURE_LINKS[name][0],
                          "fore": POSTURE_LINKS[name][1]}
                   for side in SIDES}
            for name in PAPER_ROBOTS}


def posture_anchors() -> dict:
    """每台每侧的中性肩/肘/腕点（基座系，米）：只测方向口径的算术，不接仿真。"""
    entries = {}
    for name in PAPER_ROBOTS:
        entries[name] = {}
        for side in SIDES:
            sign = 1.0 if side == "left" else -1.0
            shoulder = np.array([0.0, 0.20 * sign, 1.20])
            entries[name][side] = {
                "shoulder": list(shoulder),
                "elbow": list(shoulder + np.array([0.0, 0.0, -0.20])),
                "wrist": list(shoulder + np.array([0.10, 0.0, -0.20]))}
    return entries


def unit(vector) -> np.ndarray:
    """测试里的单位化小工具（方向断言全用它，避免重复写三遍）。"""
    values = np.asarray(vector, dtype=np.float64)
    return values / float(np.linalg.norm(values))


def raised_arm(side: str, angle: float) -> np.ndarray:
    """把**上臂方向**绕 x 轴转 ``angle``（x 轴垂直于上臂，所以方向真的转了 angle）。

    不要用 ``swung_bent_arm`` 量方向：那个动作是「绕肩->腕轴摆肘」，上臂方向与那根轴只差
    约 8.5 度，所以方向只转得到 5.9 度（它量的是位移口径下的目标位移，不是方向）。前臂
    向量保持不动、跟着肘走，所以这一帧的 ``operator_fore_deg`` 是 0。
    """
    points = bent_arm(side)
    shoulder, elbow, wrist = points
    upper = axis_angle(np.array([1.0, 0.0, 0.0]), angle) @ (elbow - shoulder)
    return np.stack((shoulder, shoulder + upper, shoulder + upper + (wrist - elbow)))


def posture_bridge(side: str = "right", **kwargs) -> RealtimeArmBridge:
    """方向口径的桥，中性姿势是弯肘的（上臂与前臂方向不同，才测得出两个方向）。

    ``frame_dt=TEST_FRAME_DT``：现场摄像头 8-12 fps，一帧几百度/秒是正常手速；用名义 1/30 s
    会把「一帧转 40-60 度」也判成关键点跳变（护栏的上限是 600 度/秒），那测的就不是方向口径了。
    """
    kwargs.setdefault("frame_dt", TEST_FRAME_DT)
    return calibrated_bent(sides=(side,), map_mode="posture",
                           robot_links=posture_links(), anchors=posture_anchors(),
                           **kwargs)


class PostureMappingTests(unittest.TestCase):
    """方向口径：你的方向一比一搬过去，长度用机器人自己的 —— 幅度与臂长无关。"""

    def test_robot_arm_directions_follow_the_operators_verbatim(self):
        # 现场症状的根因是「幅度被短基线/行程夹掉」，所以这里断言的是**方向**：
        # 同一帧、三台臂长完全不同的机器人，上臂方向都等于你的方向（过 axis_map）。
        frame = raised_arm("right", np.radians(40.0))
        for name in PAPER_ROBOTS:
            with self.subTest(robot=name):
                bridge = posture_bridge()
                entry = bridge.update(arms_world={"right": frame})[name]["right"]
                self.assertEqual(entry.info["map"], "posture")
                here = np.asarray(posture_anchors()[name]["right"]["shoulder"], dtype=float)
                expected = unit(bridge.axis_map.apply(frame[1] - frame[0]))
                np.testing.assert_allclose(
                    unit(np.asarray(entry.elbow, dtype=float) - here), expected, atol=1e-9)
                expected_fore = unit(bridge.axis_map.apply(frame[2] - frame[1]))
                np.testing.assert_allclose(
                    unit(np.asarray(entry.position, dtype=float)
                        - np.asarray(entry.elbow, dtype=float)), expected_fore, atol=1e-9)
                # 「你转 40 度，机器人就转 40 度」：与它是 g1 还是 h1_2 无关。
                self.assertAlmostEqual(entry.info["operator_upper_deg"], 40.0, places=2)
                self.assertAlmostEqual(entry.info["robot_upper_deg"], 40.0, places=2)
                self.assertIsNotNone(entry.elbow)

    def test_target_lengths_are_the_robots_own_bones(self):
        # 两根骨头的长度是机器人自己量出来的（不是你的、也不是被 gain 缩放的）。
        for name in PAPER_ROBOTS:
            with self.subTest(robot=name):
                bridge = posture_bridge()
                entry = bridge.update(
                    arms_world={"right": raised_arm("right", 0.6)})[name]["right"]
                here = np.asarray(posture_anchors()[name]["right"]["shoulder"], dtype=float)
                upper = float(np.linalg.norm(np.asarray(entry.elbow, dtype=float) - here))
                fore = float(np.linalg.norm(np.asarray(entry.position, dtype=float)
                                            - np.asarray(entry.elbow, dtype=float)))
                self.assertAlmostEqual(upper, POSTURE_LINKS[name][0], places=9)
                self.assertAlmostEqual(fore, POSTURE_LINKS[name][1], places=9)

    def test_both_targets_are_reachable_together(self):
        # 腕目标与肘目标同出一源（腕 - 肘 正好是机器人的前臂向量），所以 IK2 不会互相拉扯；
        # 位移口径下两个目标各自被逐轴夹断，正是现场「抽抽」的来源。
        bridge = posture_bridge()
        entry = bridge.update(arms_world={"right": raised_arm("right", 1.1)})["h1_2"]["right"]
        here = np.asarray(posture_anchors()["h1_2"]["right"]["shoulder"], dtype=float)
        wrist = np.asarray(entry.position, dtype=float)
        elbow = np.asarray(entry.elbow, dtype=float)
        ceiling = POSTURE_LINKS["h1_2"][0] + POSTURE_LINKS["h1_2"][1]
        self.assertLessEqual(float(np.linalg.norm(wrist - here)), ceiling + 1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(wrist - elbow)),
                               POSTURE_LINKS["h1_2"][1], places=9)

    def test_amplitude_is_not_squeezed_by_the_travel_ceiling(self):
        # 现场症状：g1 每轴 40 mm 的行程上限把大臂压得看不出来。方向口径下目标位移是
        # 「你的转角 x 机器人自己的上臂长」，落在大臂自己的球面上，不受那个上限约束。
        bridge = posture_bridge()
        rest = bridge.update(arms_world={"right": raised_arm("right", 0.0)})["g1"]["right"]
        swung = bridge.update(
            arms_world={"right": raised_arm("right", np.radians(60.0))})["g1"]["right"]
        travel = float(np.linalg.norm(np.asarray(swung.position)
                                      - np.asarray(rest.position)))
        self.assertGreater(travel, 2.0 * profile_for("g1").travel_limit("right"))
        self.assertFalse(swung.info["travel_limit_applied"])
        self.assertFalse(swung.info["clamped"])

    def test_zero_gain_keeps_the_robots_own_neutral_direction(self):
        bridge = posture_bridge(posture_gain=0.0)
        entry = bridge.update(arms_world={"right": raised_arm("right", 0.9)})["h1_2"]["right"]
        anchors = posture_anchors()["h1_2"]["right"]
        here = np.asarray(anchors["shoulder"], dtype=float)
        rest = np.asarray(anchors["elbow"], dtype=float) - here
        np.testing.assert_allclose(unit(np.asarray(entry.elbow, dtype=float) - here),
                                   unit(rest), atol=1e-9)
        self.assertEqual(entry.info["robot_upper_deg"], 0.0)

    def test_gain_out_of_range_is_refused(self):
        with self.assertRaises(ValueError):
            RealtimeArmBridge(PROFILES, sides=("right",), map_mode="posture",
                              posture_gain=1.5)

    def test_elbow_target_needs_no_elbow_task_flag(self):
        # 方向口径下肘目标由口径自己给出：--arm-elbow-task 不需要，给了也不冲突。
        entry = posture_bridge().update(
            arms_world={"right": raised_arm("right", 0.6)})["h1_2"]["right"]
        self.assertEqual(entry.info["elbow_target"], "posture")
        other = posture_bridge(elbow_task=True).update(
            arms_world={"right": raised_arm("right", 0.6)})["h1_2"]["right"]
        self.assertEqual(other.info["map"], "posture")
        np.testing.assert_allclose(other.elbow, entry.elbow, atol=1e-12)

    def test_missing_shoulder_anchor_falls_back_to_chain_with_a_reason(self):
        anchors = {name: {side: {"elbow": [0.10, 0.20, 0.05],
                                 "wrist": [0.10, 0.45, 0.05]}
                          for side in SIDES}
                   for name in PAPER_ROBOTS}
        bridge = calibrated_bent(sides=("right",), map_mode="posture",
                                 robot_links=posture_links(), anchors=anchors)
        entry = bridge.update(arms_world={"right": raised_arm("right", 0.8)})["h1_2"]["right"]
        self.assertIn("posture unavailable", entry.info["map"])
        self.assertIn("shoulder anchor", entry.info["map"])
        self.assertIsNone(entry.elbow)

    def test_summary_reports_the_gain_and_which_sides_can_run_it(self):
        summary = posture_bridge(posture_gain=0.5).summary()
        self.assertEqual(summary["map_mode"], "posture")
        self.assertEqual(summary["posture_gain"], 0.5)
        self.assertTrue(all(summary["posture_available"]["g1"].values()))
        self.assertIn("right", summary["shoulder_anchors_m"]["g1"])
        self.assertIn("posture", summary["counts"])

    def test_the_live_entry_defaults_to_the_direction_recipe(self):
        """2026-10-14 现场第三轮：默认 chain 时「大臂抬不起来 / 张不开怀抱 / GR1-T2 前臂反」
        三个症状全回来了（chain 不给肘目标，肩只被逐轴行程夹到 40-150 mm）。入口默认改 posture，
        显式 ``--arm-map chain`` 仍然要能进（A/B 与老记录靠它复现）。
        """
        self.assertEqual(build_parser().parse_args([]).arm_map, "posture")
        self.assertEqual(build_parser().parse_args(["--arm-map", "chain"]).arm_map, "chain")
        self.assertEqual(build_parser().parse_args(["--arm-map", "segments"]).arm_map,
                         "segments")


# --------------------------------------------------------------------------- #
# 2026-10-13 现场第二个症状（H1-2 手臂与手掌交界处偶尔抽搐）的修法：关键点跳变护栏。
# --------------------------------------------------------------------------- #
class DirectionGlitchGuardTests(unittest.TestCase):
    """单帧方向变化超过人手臂角速度上限 -> 判为关键点跳变，保持上一帧。

    依据（``outputs/tmp_realtime/check_arm_jitter.py`` 量同一份录音）：每帧方向变化中位
    4.7-6.9 度、p90 21-35 度、**最大 153 度**。方向口径 1:1 搬方向，于是跳变直接变成
    「一帧里把肘转 153 度」的请求，IK 只好翻支路或抽一下。
    """

    def _rotated(self, side, degrees, **kwargs):
        """把这一侧的手臂绕肩整体转 ``degrees`` 度后的一帧（上臂/前臂方向都跟着转）。"""
        points = arm(side, **kwargs)
        rotation = axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(degrees))
        return rotate_arm(points, rotation, points[0])

    def _bridge(self, **kwargs) -> RealtimeArmBridge:
        """名义 1/30 s 的桥：护栏上限 = 600/30 = 20 度/帧，所以 120 度的跳变必然触发。"""
        return posture_bridge(frame_dt=1.0 / 30.0, **kwargs)

    def test_a_single_frame_spike_is_held_and_counted(self):
        bridge = self._bridge()
        first = bridge.update(arms_world={"right": bent_arm("right")})["h1_2"]["right"]
        # 名义帧长 1/30 s -> 上限 600/30 = 20 度；转 120 度必然触发
        spike = bridge.update(arms_world={"right": self._rotated("right", 120.0)})["h1_2"]["right"]
        self.assertEqual(bridge.counts["direction_glitch"]["right"], 1)
        self.assertGreater(spike.info["operator_direction_glitch_deg"], 100.0)
        # 保持上一帧 -> 肘/腕两个目标与上一帧逐位相同
        np.testing.assert_allclose(spike.position, first.position, atol=0.0)
        np.testing.assert_allclose(spike.elbow, first.elbow, atol=0.0)

    def test_a_persistent_change_is_accepted_after_the_hold_limit(self):
        bridge = self._bridge()
        bridge.update(arms_world={"right": bent_arm("right")})
        targets = [bridge.update(arms_world={"right": self._rotated("right", 120.0)})["h1_2"]["right"]
                   for _ in range(GLITCH_HOLD_FRAMES + 1)]
        self.assertEqual(bridge.counts["direction_glitch"]["right"], GLITCH_HOLD_FRAMES)
        self.assertEqual(bridge.counts["direction_glitch_kept"]["right"], 1)
        # 放过去之后必须真的动了，不能一直卡在旧方向上
        moved = np.linalg.norm(np.asarray(targets[-1].position, dtype=float)
                               - np.asarray(targets[0].position, dtype=float))
        self.assertGreater(moved, 0.01)

    def test_a_rate_of_zero_turns_the_guard_off(self):
        bridge = self._bridge(max_direction_rate_deg_s=0.0)
        first = bridge.update(arms_world={"right": bent_arm("right")})["h1_2"]["right"]
        spike = bridge.update(arms_world={"right": self._rotated("right", 120.0)})["h1_2"]["right"]
        self.assertEqual(bridge.counts["direction_glitch"]["right"], 0)
        self.assertGreater(np.linalg.norm(np.asarray(spike.position, dtype=float)
                                          - np.asarray(first.position, dtype=float)), 0.01)

    def test_a_data_gap_resets_the_baseline_instead_of_faking_a_glitch(self):
        bridge = self._bridge()
        bridge.update(arms_world={"right": bent_arm("right")})
        bridge.update(arms_world={"right": None})            # 断流一帧（没检到）
        entry = bridge.update(arms_world={"right": self._rotated("right", 120.0)})["h1_2"]["right"]
        self.assertEqual(bridge.counts["direction_glitch"]["right"], 0)
        self.assertEqual(entry.info["operator_direction_glitch_deg"], 0.0)

    def test_the_frame_dt_scales_the_limit(self):
        # 半秒一帧时 600 度/秒 = 300 度/帧，所以 120 度是正常移动，不该拦。
        slow = self._bridge()
        slow.update(arms_world={"right": bent_arm("right")}, frame_dt=0.5)
        slow.update(arms_world={"right": self._rotated("right", 120.0)}, frame_dt=0.5)
        self.assertEqual(slow.counts["direction_glitch"]["right"], 0)
        # 33 ms 一帧时同一动作就是 3600 度/秒，必须拦。
        fast = self._bridge()
        fast.update(arms_world={"right": bent_arm("right")}, frame_dt=1.0 / 30.0)
        fast.update(arms_world={"right": self._rotated("right", 120.0)}, frame_dt=1.0 / 30.0)
        self.assertEqual(fast.counts["direction_glitch"]["right"], 1)

    def test_the_entry_parser_exposes_the_guard_flag(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.arm_direction_rate_deg_s, MAX_DIRECTION_RATE_DEG_S)
        args = build_parser().parse_args(["--arm-direction-rate-deg-s", "0"])
        self.assertEqual(args.arm_direction_rate_deg_s, 0.0)


class HelpTextTests(unittest.TestCase):
    """``--help`` 必须真的打得开（全入口，不只手臂那几个参数）。

    argparse 在渲染帮助时会对每条 ``help`` 再做一次 ``%`` 格式化
    （``_format_action`` -> ``_get_help_string(action) % params``）。所以 help 字符串里
    **任何**剩下的裸 ``%`` 都会抛 ``TypeError: %o format: an integer is required, not
    dict`` —— 2026-10-14 现场那条命令就是这么崩的：``--help`` 退出码 1，用户根本查不到新参数
    （``--arm-calibrate-mode``）。凶手是 ``--arm-elbow-recovery-deg`` 那段 ``91%%``：
    它自己先用 ``"% CONST"`` 格式化过一道，``%%`` 被吃成一个 ``%``，argparse 再看到就崩。

    修法是那一块改成 f-string，字面百分号只留 ``%%`` 给 argparse。本测试就是那条不变量的
    看门人：``format_help()`` 不抛异常、参数在帮助里、而且最后渲染出来的是一个 ``%``
    而不是两个。
    """

    def test_help_renders_and_lists_the_arm_options(self):
        text = build_parser().format_help()      # 裸 '%' 会在这里抛 TypeError
        for option in ("--arm-calibrate-mode", "--arm-calibrate-frames",
                       "--arm-elbow-recovery-deg", "--arm-map", "--arm-travel-mode"):
            self.assertIn(option, text)

    def test_a_literal_percent_survives_as_one_percent(self):
        text = build_parser().format_help()
        # argparse 把 %% 还原成 %：报告数字里的百分号必须只出现一次。
        self.assertIn("91% of frames", text)
        self.assertNotIn("%%", text)

    def test_the_calibrate_mode_choices_are_listed(self):
        text = build_parser().format_help()
        self.assertIn("--arm-calibrate-mode {hold,track}", text)
        self.assertIn("--arm-calibrate-frames", text)


DEVTOOLS = Path(__file__).resolve().parents[1] / "devtools"


def load_arm_launcher():
    """按路径加载 ``devtools/run_realtime_arm_windows.py``。

    它不在安装包里，而且它 ``import run_realtime_windows``（同目录），所以先把
    ``devtools`` 放进 ``sys.path``；加载方式与 ``tests/test_arm_preview.py`` 里那套一致。
    2026-11-09 起现场档（``--elbow-follow``）也住在这个文件里，原先的薄封装
    ``run_realtime_arm_elbow_follow_windows.py`` 已删掉。
    """
    if str(DEVTOOLS) not in sys.path:
        sys.path.insert(0, str(DEVTOOLS))
    spec = importlib.util.spec_from_file_location(
        "run_realtime_arm_windows", DEVTOOLS / "run_realtime_arm_windows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ElbowFollowPresetTests(unittest.TestCase):
    """启动器的默认口径是 2026-10-15 的现场档，``--elbow-follow`` 只负责换成入口参数。

    ``--arm-elbow-follow`` 的语义在 ``arm_ik`` / ``arm_scene_driver`` 侧已由
    ``ElbowPostureTests`` 覆盖；这里钉的是"到底把哪个参数交给了入口"。
    """

    def setUp(self):
        self.launcher = load_arm_launcher()

    def test_default_run_is_this_build(self):
        self.assertEqual(self.launcher.apply_elbow_preset([]),
                         ["--arm-elbow-follow", "operator"])
        self.assertEqual(self.launcher.apply_elbow_preset(["--robot", "g1"]),
                         ["--robot", "g1", "--arm-elbow-follow", "operator"])

    def test_the_off_switches_are_the_historical_side(self):
        self.assertEqual(self.launcher.apply_elbow_preset(["--off"]),
                         ["--arm-elbow-follow", "off"])
        self.assertEqual(self.launcher.apply_elbow_preset(["--elbow-follow", "off"]),
                         ["--arm-elbow-follow", "off"])
        self.assertNotIn("--off", self.launcher.apply_elbow_preset(["--off"]))

    def test_the_switch_becomes_the_entry_flag(self):
        self.assertEqual(
            self.launcher.apply_elbow_preset(["--elbow-follow", "operator"]),
            ["--arm-elbow-follow", "operator"])
        self.assertEqual(self.launcher.apply_elbow_preset(["--elbow-follow=off"]),
                         ["--arm-elbow-follow", "off"])

    def test_the_switch_never_reaches_the_entry_parser(self):
        pinned = self.launcher.apply_elbow_preset(
            ["--robot", "g1", "--elbow-follow", "operator", "--side", "right"])
        self.assertNotIn("--elbow-follow", pinned)   # 糖不能漏给入口的 parser
        self.assertEqual(pinned[-2:], ["--arm-elbow-follow", "operator"])
        self.assertIn("--robot", pinned)
        self.assertIn("--side", pinned)

    def test_an_explicit_choice_is_not_overridden(self):
        self.assertEqual(
            self.launcher.apply_elbow_preset(
                ["--arm-elbow-follow", "off", "--elbow-follow", "operator"]),
            ["--arm-elbow-follow", "off"])
        self.assertEqual(self.launcher.apply_elbow_preset(["--arm-elbow-follow=off"]),
                         ["--arm-elbow-follow=off"])

    def test_the_printed_mode_follows_the_last_flag(self):
        self.assertEqual(
            self.launcher.elbow_follow_value(
                ["--arm-elbow-follow=off", "--arm-elbow-follow", "operator"]),
            "operator")
        self.assertEqual(self.launcher.elbow_follow_value([]), "operator")
        self.assertEqual(self.launcher.elbow_follow_value(["--arm-elbow-follow", "off"]),
                         "off")

    def test_the_pinned_choice_is_accepted_by_the_entry_parser(self):
        args = build_parser().parse_args(self.launcher.apply_elbow_preset([]))
        self.assertEqual(args.arm_elbow_follow, self.launcher.ELBOW_BUILD)


def load_quickcheck_launcher():
    """按路径加载 ``devtools/run_arm_quickcheck.py``（现场入口只许 import，不许改）。"""
    if str(DEVTOOLS) not in sys.path:
        sys.path.insert(0, str(DEVTOOLS))
    spec = importlib.util.spec_from_file_location(
        "run_arm_quickcheck", DEVTOOLS / "run_arm_quickcheck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuickcheckLauncherTests(unittest.TestCase):
    """``devtools/run_arm_quickcheck.py``：不碰现场入口，也能把现场档跑起来并打一屏读数。

    为什么要有它（2026-11-17）：用户要求 ``devtools/run_realtime_arm_windows.py`` 保持 HEAD
    的样子（它工作区里曾钉着一段 ``WRIST_PRESET``，已 ``git checkout --`` 回退），于是入口默认
    又回到 ``--arm-orientation off``，也就是"手腕不动"。新文件把现场档**显式**补进参数表，
    这里钉住三件事：补的就是那四项、你自己写的优先、摘要读不到报告也不许抛。
    """

    def setUp(self):
        self.launcher = load_quickcheck_launcher()

    def test_it_pins_the_field_build(self):
        self.assertEqual(self.launcher.with_field_flags([]),
                         ["--arm-elbow-follow", "operator",
                          "--arm-orientation", "on",
                          "--arm-orientation-scope", "wrist",
                          "--arm-orientation-mode", "hand_base",
                          "--arm-hand-roll-deg", "180",
                          "--arm-jitter-filter", "on",
                          "--arm-ik-mode", "pose"])
        args = build_parser().parse_args(self.launcher.with_field_flags([]))
        self.assertEqual(args.arm_elbow_follow, "operator")
        self.assertEqual(args.arm_orientation, "on")
        self.assertEqual(args.arm_orientation_scope, "wrist")
        self.assertEqual(args.arm_orientation_mode, "hand_base")
        self.assertEqual(args.arm_hand_roll_deg, 180.0)
        self.assertEqual(args.arm_jitter_filter, "on")
        self.assertEqual(args.arm_ik_mode, "pose")

    def test_your_own_arm_flags_win(self):
        argv = self.launcher.with_field_flags(["--arm-orientation-scope", "arm",
                                               "--arm-orientation=off"])
        self.assertEqual(argv.count("--arm-orientation-scope"), 1)
        self.assertEqual(argv[argv.index("--arm-orientation-scope") + 1], "arm")
        self.assertNotIn("--arm-orientation", argv)     # =off 也算写了，不许再补 on
        self.assertIn("--arm-ik-mode", argv)            # 没写的那几项照补

    def test_both_spellings_of_frames_and_report_count_as_written(self):
        self.assertTrue(self.launcher.has_flag(["--frames", "150"], "--frames"))
        self.assertTrue(self.launcher.has_flag(["--frames=150"], "--frames"))
        self.assertFalse(self.launcher.has_flag(["--frame", "150"], "--frames"))
        self.assertTrue(self.launcher.has_flag(["--report=out.json"], "--report"))

    def test_the_digest_survives_a_missing_report(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.launcher.digest(Path("no_such_quickcheck_report.json"))
        self.assertIn("no report to read", stream.getvalue())

    def test_the_digest_shows_symmetry_and_the_late_palm_basis(self):
        report = Path(tempfile.mkdtemp()) / "quickcheck.json"
        report.write_text(json.dumps({
            "processed_frames": 900, "fps": 12.5, "elapsed_seconds": 72.0,
            "bridge": {
                "operator_symmetry": {"frames": 800, "median_deg": 8.5, "p90_deg": 20.0,
                                      "max_deg": 120.0, "same_side_frames": 3,
                                      "note": "both arms were mirror images"},
                "neutral": {"left": {"palm_reference_source": "late",
                                     "palm_reference_frame": 367,
                                     "palm_reference_pose_deg": 132.8,
                                     "palm_spread_deg": 4.8, "palm_outliers": 1}},
                "counts": {"palm_late": {"left": 1, "right": 0}},
            },
            "arm": {"hand_orientation_error_deg": {
                "h1_2": {"left": 3.2, "right": 4.8}}},
        }, ensure_ascii=False), encoding="utf-8")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.launcher.digest(report)
        text = stream.getvalue()
        self.assertIn("median 8.50 deg", text)
        self.assertIn("3/800 frames same-side", text)
        self.assertIn("late @frame 367, 132.8 deg", text)               # 迟到基准的偏置要露出来
        self.assertIn("left palm basis was learned 132.8 deg", text)     # 超 60 度必须报警
        self.assertIn("palm/hand axes", text)
        self.assertIn("h1_2 3.2/4.8 deg", text)


class WristTraceColumnTests(unittest.TestCase):
    """``--wrist-trace`` 是取证格式：表头与每一行的列数必须一致，``ori_source`` 要落地。

    这类错不会抛异常 —— 少写一格只会让现场读出来的每个字段静默错位一格，所以用一条
    「行宽 == 表头宽」的断言钉住它（2026-11-16 加 ``ori_source`` 时就是靠它兜的底）。
    """

    HEADER = WRIST_TRACE_COLUMNS

    class _Adapter:
        """只提供 ``WristTrace.frame`` 用到的五个方法。"""

        def to_base(self, point):
            return np.asarray(point, dtype=np.float64)

        def arm_points(self, side):
            return {"wrist": np.zeros(3)}

        def residual_mm(self, side):
            return 12.5

        def orientation_error_deg(self, side):
            return 1.5

        def joint_values(self, side):
            return {"right_elbow_roll_joint": 0.2, "right_wrist_pitch_joint": 0.1,
                    "right_wrist_yaw_joint": -0.3}

    class _Driver:
        def __init__(self):
            self.ik = {"h1_2": WristTraceColumnTests._Adapter()}

    @staticmethod
    def _rows(folder, pending, orientation=None, raw_frame=None):
        trace = WristTrace(Path(folder) / "trace.csv")
        trace.frame(7, 0.231, pending, WristTraceColumnTests._Driver(),
                    orientation, raw_frame)
        trace.close()
        with (Path(folder) / "trace.csv").open("r", encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_a_measured_frame_writes_one_cell_per_column(self):
        pending = {"right": {"status": "ok",
                             "positions": {"h1_2": np.array([0.24, -0.05, 0.19])},
                             "reports": {"h1_2": object()},
                             "ori_source": "palm_normal"}}
        # 相对中性转了 90 度：四元数 (x, y, z, w)，WristTrace 取 [3] 当余弦。
        half = np.radians(45.0)
        orientation = {"right": np.array([0.0, 0.0, np.sin(half), np.cos(half)])}
        with tempfile.TemporaryDirectory() as folder:
            rows = self._rows(folder, pending, orientation,
                              {"world_right": np.zeros((21, 3))})
        header, row = rows[0], rows[1]
        self.assertEqual(header, list(self.HEADER))
        self.assertEqual(len(row), len(self.HEADER))
        cells = dict(zip(header, row))
        self.assertEqual(cells["ori_source"], "palm_normal")
        self.assertEqual(cells["ori_sent"], "1")
        self.assertAlmostEqual(float(cells["ori_turn_deg"]), 90.0, places=1)
        self.assertEqual(cells["wrist_names"].split(";")[0], "right_elbow_roll_joint")

    def test_a_frame_without_a_robot_stays_aligned(self):
        pending = {"right": {"status": "no_target", "ori_source": "three_point"}}
        with tempfile.TemporaryDirectory() as folder:
            rows = self._rows(folder, pending)
        header, row = rows[0], rows[1]
        self.assertEqual(len(row), len(self.HEADER))
        cells = dict(zip(header, row))
        self.assertEqual((cells["robot"], cells["status"]), ("-", "no_target"))
        self.assertEqual(cells["ori_source"], "three_point")
        self.assertEqual(cells["ori_sent"], "0")

    def test_the_label_normalises_what_the_bridge_writes(self):
        def targets(*sources):
            return {name: SimpleNamespace(info={"orientation_source": source})
                    for name, source in zip(("h1_2", "g1", "gr1_t2"), sources)}

        self.assertEqual(orientation_source_label(targets("palm_normal")), "palm_normal")
        # palm_hold 不能被 startswith("palm") 吞成 palm_normal：它意味着这一帧手点丢了。
        self.assertEqual(orientation_source_label(targets("palm_hold")), "palm_hold")
        self.assertEqual(orientation_source_label(targets("three_point")), "three_point")
        self.assertEqual(orientation_source_label(targets(None)), "none")
        self.assertEqual(orientation_source_label(targets()),
                         "none")
        # 手部兜底：桥写的是整句话，CSV 里必须是短标签且不含逗号。
        fallback = orientation_source_label(
            targets("none (hand fallback has no elbow)"))
        self.assertEqual(fallback, "none_hand_fallback")
        self.assertNotIn(",", fallback)
        # 真出现不一致就如实连起来，不静默挑一个。
        self.assertEqual(orientation_source_label(targets("palm_normal", "three_point")),
                         "mixed(palm_normal|three_point)")


if __name__ == "__main__":
    unittest.main()






class PalmBasisVisibilityTests(unittest.TestCase):
    """「中性里有没有掌面基」必须可见（2026-11-16 晚那场整场丢旋前的教训）。

    机制：掌面基要跟中性里的 ``orientation_palm`` 相减，而中性是标定那几十帧的均值；标定时手
    不在画面里 -> ``orientation_palm`` 是 None -> 之后再怎么把手伸进画面，也只能一路退回
    ``three_point``（对旋前不敏感 = 拧腕不跟）。当时唯一的线索是 ``--wrist-trace`` 的
    ``ori_source`` 全是 ``three_point``，现场只看到「手腕不跟」，看不出是标定时机的问题。
    所以：标定期间拿到掌面基的帧数进报告（``palm_samples`` / ``orientation_palm_recorded``），
    并且在 HUD 旁边一次性喊出来（``realtime_teleop._warn_missing_palm_basis``）。
    """

    FRAMES = 4

    def _bridge(self, with_hands: bool) -> RealtimeArmBridge:
        bridge = RealtimeArmBridge(PROFILES, sides=("right",),
                                   calibration_frames=self.FRAMES, frame_dt=0.1)
        for _ in range(self.FRAMES):
            bridge.update(arms_world={"right": bent_arm("right")},
                          hands={"world_right": open_hand()} if with_hands else None,
                          frame_dt=0.1)
        return bridge

    def test_the_palm_frame_count_matches_the_basis_that_was_recorded(self):
        with_palm = self._bridge(True)
        entry = with_palm.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], self.FRAMES)
        self.assertTrue(entry["orientation_palm_recorded"])
        # 同一次标定里三点基也照样记（两条基不能混用，两份都要在）。
        self.assertTrue(entry["orientation_recorded"])

    def test_a_calibration_without_hands_reports_zero_and_stays_three_point(self):
        blind = self._bridge(False)
        entry = blind.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], 0)
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertTrue(entry["orientation_recorded"])
        # 后面把手伸进来也补不回来：中性里没有掌面基，整场只能三点基。
        update = blind.update(arms_world={"right": bent_arm("right")},
                              hands={"world_right": open_hand()}, frame_dt=0.1)
        self.assertEqual(update["h1_2"]["right"].info["orientation_source"], "three_point")

    def test_the_warning_fires_once_per_side_and_only_when_the_basis_is_missing(self):
        import io
        from contextlib import redirect_stdout

        from teleoperation.simulation.realtime_teleop import _warn_missing_palm_basis

        blind = self._bridge(False)
        warned = set()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _warn_missing_palm_basis(blind, ("right",), warned)
            _warn_missing_palm_basis(blind, ("right",), warned)
        text = buffer.getvalue()
        self.assertEqual(text.count("[WARN]"), 1, text)
        self.assertIn("right", text)
        self.assertIn("palm_samples=0", text)
        self.assertEqual(warned, {"right"})

        with_palm = self._bridge(True)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _warn_missing_palm_basis(with_palm, ("right",), set())
        self.assertEqual(buffer.getvalue(), "")


class PalmBasisTrustTests(unittest.TestCase):
    """掌面基的**可信度闸门**：样本太少或彼此不一致 -> 就当没有（退回三点基）。

    2026-11-17 现场 15:40 那场（``outputs/tmp_realtime/arm_run_20261117_154001.json``）：
    30 帧标定里左手 ``palm_samples=0``、右手 12，右手拿这 12 帧的均值当基准，于是同一场
    ``orientation_error_deg`` 左 0.6-1.6 度 / 右 **24.9-122.5 度**，h1_2 右腕三个关节
    ``at_limit: true``（左三个全在中段），``residual_mm`` 左 2.5-9.4 / 右 45-256 mm ——
    现场看到的就是「右手卡住不动、手腕怎么拧都不对」。判据见
    ``arm_realtime.PALM_MIN_SAMPLE_FRACTION`` / ``PALM_MAX_SPREAD_DEG``。
    """

    FRAMES = 6

    @staticmethod
    def _hand(twist_deg: float = 0.0):
        """掌面朝相机的手；``twist_deg`` 绕 world y 俯仰一下 = 换一个掌面朝向。"""
        hand = open_hand()
        if not twist_deg:
            return hand
        return rotate_arm(hand, axis_angle((0.0, 1.0, 0.0), np.radians(twist_deg)),
                          (0.5, 0.5, 0.0))

    def _bridge(self, palm_frames: int, frames: int = None, twist_deg: float = 0.0):
        """前 ``palm_frames`` 帧给手（奇数帧带 ``twist_deg``），其余帧只有手臂。"""
        frames = self.FRAMES if frames is None else frames
        bridge = RealtimeArmBridge(PROFILES, sides=("right",),
                                   calibration_frames=frames, frame_dt=0.1)
        for index in range(frames):
            hand = None
            if index < palm_frames:
                hand = {"world_right": self._hand(0.0 if index % 2 == 0 else twist_deg)}
            bridge.update(arms_world={"right": bent_arm("right")}, hands=hand, frame_dt=0.1)
        return bridge

    def test_a_basis_from_too_few_frames_is_rejected_with_the_count_in_the_note(self):
        # 10 帧标定、只有 4 帧看得见手：低于一半 -> 不算数。
        bridge = self._bridge(palm_frames=4, frames=10)
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], 4)
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertIn("4/10", entry["palm_note"])
        self.assertIn("calibration frames", entry["palm_note"])
        # 这一帧还不能补：迟到基准要**连续** 8 帧稳定的掌面基（见 PALM_LATE_FRAMES），
        # 在那之前如实退回三点基（不会把腕推进限位）。
        update = bridge.update(arms_world={"right": bent_arm("right")},
                               hands={"world_right": self._hand()}, frame_dt=0.1)
        self.assertEqual(update["h1_2"]["right"].info["orientation_source"], "three_point")

    def test_a_basis_whose_samples_disagree_is_rejected_with_the_spread_in_the_note(self):
        bridge = self._bridge(palm_frames=self.FRAMES, twist_deg=90.0)
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], self.FRAMES)
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertGreater(entry["palm_spread_deg"], 25.0)
        self.assertIn("disagree", entry["palm_note"])
        self.assertIn("deg", entry["palm_note"])

    def test_a_steady_basis_is_kept_and_reports_a_small_spread(self):
        bridge = self._bridge(palm_frames=self.FRAMES)
        entry = bridge.summary()["neutral"]["right"]
        self.assertTrue(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_note"], "")
        self.assertLessEqual(entry["palm_spread_deg"], 1.0)
        self.assertEqual(bridge.palm_basis_note("right"), "")

    def test_the_verdict_and_the_spread_survive_save_and_load(self):
        rejected = self._bridge(palm_frames=self.FRAMES, twist_deg=90.0)
        fresh = RealtimeArmBridge(PROFILES, sides=("right",),
                                  calibration_frames=self.FRAMES, frame_dt=0.1)
        fresh.load_neutral(rejected.neutral_record())
        note = fresh.palm_basis_note("right")
        self.assertIn("disagree", note)
        entry = fresh.summary()["neutral"]["right"]
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertGreater(entry["palm_spread_deg"], 25.0)
        self.assertEqual(entry["palm_note"], note)

    def test_a_few_flipped_samples_no_longer_kill_the_basis(self):
        """多数簇（``_palm_cluster``）：8 帧看得见手、其中 2 帧掌面差了 45 度 -> 仍然可用。

        现场那两场（15:40 / 16:00）的散度 55 / 165 度不是「人转了手」，是那几帧的法向取反。
        把正反两拨一起平均得到的既不正也不反，于是闸门把**整场**的旋前判死；现在只把散出去
        的那几帧记成 ``palm_outliers``，多数一致的仍然可用。
        """
        frames = 10
        bridge = RealtimeArmBridge(PROFILES, sides=("right",),
                                   calibration_frames=frames, frame_dt=0.1)
        for index in range(frames):
            if index >= 8:
                hand = None                                   # 最后两帧手不在画面里
            elif index >= 6:
                hand = {"world_right": self._hand(90.0)}      # 这两帧掌面转过 45 度（翻号/半遮挡）
            else:
                hand = {"world_right": self._hand()}
            bridge.update(arms_world={"right": bent_arm("right")}, hands=hand, frame_dt=0.1)
        entry = bridge.summary()["neutral"]["right"]
        self.assertTrue(entry["orientation_palm_recorded"], entry["palm_note"])
        self.assertEqual(entry["palm_note"], "")
        self.assertEqual(entry["palm_reference_source"], "neutral")
        self.assertEqual(entry["palm_outliers"], 2)
        # ``palm_spread_deg`` 是**全部**样本相对基准的最大夹角：它如实报到 90 度，
        # 说明确实有帧被判出簇（``palm_outliers``）；判死与否只看「簇有多大 + 簇内散度」。
        self.assertGreater(entry["palm_spread_deg"], PALM_MAX_SPREAD_DEG)

    def test_clustering_does_not_loosen_the_frame_count_guard(self):
        """10 帧里只有 4 帧看得见手、且两两不一致 -> 最大簇只有 2 帧，仍然判死。

        聚簇只能挡「少数坏帧」，不能把「手只露了一下」变成可用：12/30 帧凑出来的基准
        会把六个腕关节推进限位（见 ``PALM_MIN_SAMPLE_FRACTION``）。
        """
        frames = 10
        bridge = RealtimeArmBridge(PROFILES, sides=("right",),
                                   calibration_frames=frames, frame_dt=0.1)
        for index in range(frames):
            if index >= 4:
                hand = None
            elif index % 2:
                hand = {"world_right": self._hand(90.0)}
            else:
                hand = {"world_right": self._hand()}
            bridge.update(arms_world={"right": bent_arm("right")}, hands=hand, frame_dt=0.1)
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], 4)
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertIn("2/10", entry["palm_note"])

    def test_the_warning_says_the_basis_was_rejected_and_why(self):
        import io
        from contextlib import redirect_stdout

        from teleoperation.simulation.realtime_teleop import _warn_missing_palm_basis

        bridge = self._bridge(palm_frames=self.FRAMES, twist_deg=90.0)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _warn_missing_palm_basis(bridge, ("right",), set())
        text = buffer.getvalue()
        self.assertEqual(text.count("[WARN]"), 1, text)
        self.assertIn("palm_samples=6", text)
        self.assertIn("basis rejected", text)
        self.assertIn("disagree", text)


class PalmLateReferenceTests(unittest.TestCase):
    """中性里没有掌面基时，运行中补采一次（``arm_realtime.PALM_LATE_FRAMES``）。

    这一条对应现场那句「手腕还是不会转」。取证（``outputs/tmp_realtime/live_run_1600_wrist.csv``，
    883 帧 ×3 台 ×2 侧）：``ori_source`` 是 ``palm_normal`` 的只有 **3 行**（右 0 行），
    其余全是 ``three_point``；而三点基的第三轴取「上臂 x 前臂」的手臂平面法向，
    **对绕前臂的自转完全不敏感**。也就是说，只要中性那次没采到掌面基，这一整场
    「手怎么拧、机器人手腕一动不动」是数学上注定的，不是桥的问题。
    """

    FRAMES = 6

    def _bridge(self, **kwargs):
        kwargs.setdefault("calibrate_mode", "hold")
        kwargs.setdefault("palm_wait_frames", 0)
        return RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=self.FRAMES,
                                 frame_dt=0.1, **kwargs)

    def _feed(self, bridge, count, hand=None):
        """喂 ``count`` 帧；``hand`` 是那一侧的 world 手（``None`` = 这一帧没手点）。"""
        hands = None if hand is None else {"world_right": hand}
        last = None
        for _ in range(count):
            last = bridge.update(arms_world={"right": bent_arm("right")},
                                 hands=hands, frame_dt=0.1)
        return last

    def test_a_neutral_without_a_palm_basis_gets_one_on_the_fly(self):
        # 关掉「每帧最多走多少度」那条限幅（下一条用例专门测它），这里只想看姿态目标本身。
        bridge = self._bridge(max_orientation_step_deg=0.0)
        self._feed(bridge, self.FRAMES)                    # 手臂在、手还没进画面
        self.assertFalse(bridge.calibrating)
        entry = bridge.summary()["neutral"]["right"]
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_reference_source"], "")
        self.assertEqual(bridge.palm_reference_note("right"), "")
        # 手进来并站住 8 帧 -> 补采（第 8 帧就生效：采完之后同一帧就按掌面基算）
        last = self._feed(bridge, PALM_LATE_FRAMES, hand=open_hand())
        entry = bridge.summary()["neutral"]["right"]
        self.assertTrue(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_reference_source"], "late")
        self.assertEqual(entry["palm_reference_frame"], self.FRAMES + PALM_LATE_FRAMES)
        self.assertEqual(entry["palm_note"], "")
        self.assertEqual(bridge.counts["palm_late"]["right"], 1)
        self.assertEqual(last["h1_2"]["right"].info["orientation_source"], "palm_normal")
        self.assertIn("palm reference acquired", bridge.palm_reference_note("right"))
        # 零点取在「此刻」：这一帧与标定姿势的前臂方向差 ~0 度 -> 腕不带固定偏移。
        self.assertIsNotNone(entry["palm_reference_pose_deg"])
        self.assertLess(entry["palm_reference_pose_deg"], 1.0)
        self.assertIn("deg away from the calibration pose", bridge.palm_reference_note("right"))
        # **这就是「腕会转」**：拧 40 度，姿态目标真的跟到 40 度（改前是 three_point/0 度）。
        twisted = bridge.update(arms_world={"right": bent_arm("right")},
                                hands={"world_right": twisted_hand("right", 40.0)},
                                frame_dt=0.1)["h1_2"]["right"]
        self.assertEqual(twisted.info["orientation_source"], "palm_normal")
        angle = _quaternion_angle_deg(twisted.orientation)
        self.assertGreater(angle, 30.0)
        self.assertLess(angle, 50.0)

    def test_the_window_must_be_consecutive_so_a_dropped_frame_restarts_it(self):
        bridge = self._bridge()
        self._feed(bridge, self.FRAMES)
        self._feed(bridge, PALM_LATE_FRAMES - 1, hand=open_hand())
        self._feed(bridge, 1)                              # 手丢一帧
        self._feed(bridge, PALM_LATE_FRAMES - 1, hand=open_hand())
        entry = bridge.summary()["neutral"]["right"]
        self.assertFalse(entry["orientation_palm_recorded"], "7 + 7 帧不是连续 8 帧")
        self._feed(bridge, 1, hand=open_hand())
        self.assertEqual(bridge.summary()["neutral"]["right"]["palm_reference_source"], "late")

    def test_a_hand_that_keeps_moving_never_becomes_a_reference(self):
        """零点只有站在不动时才谈得上：来回拧手的帧永远凑不满一个稳定窗口。"""
        bridge = self._bridge()
        self._feed(bridge, self.FRAMES)
        for index in range(20):
            hand = (open_hand() if index % 2 == 0 else twisted_hand("right", 90.0))
            self._feed(bridge, 1, hand=hand)
        entry = bridge.summary()["neutral"]["right"]
        self.assertFalse(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_reference_source"], "")
        self.assertEqual(bridge.counts["palm_late"]["right"], 0)
        self.assertGreater(PALM_LATE_SPREAD_DEG, 0.0)      # 判据是「几帧彼此一致」，不是恒真

    def test_a_neutral_that_has_a_palm_basis_never_gets_a_late_one(self):
        bridge = self._bridge()
        self._feed(bridge, self.FRAMES, hand=open_hand())
        self.assertEqual(bridge.summary()["neutral"]["right"]["palm_reference_source"], "neutral")
        self._feed(bridge, PALM_LATE_FRAMES + 2, hand=open_hand())
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_reference_source"], "neutral")
        self.assertEqual(bridge.counts["palm_late"]["right"], 0)
        self.assertEqual(bridge.palm_reference_note("right"), "")

    def test_the_cli_reports_the_late_reference_instead_of_a_permanent_warning(self):
        from teleoperation.simulation.realtime_teleop import _warn_missing_palm_basis

        bridge = self._bridge()
        self._feed(bridge, self.FRAMES)
        warned = set()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _warn_missing_palm_basis(bridge, ("right",), warned)
        self.assertEqual(buffer.getvalue().count("[WARN]"), 1)
        self.assertIn("three_point", buffer.getvalue())    # 说清「暂时」会怎样
        self._feed(bridge, PALM_LATE_FRAMES, hand=open_hand())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _warn_missing_palm_basis(bridge, ("right",), warned)
        text = buffer.getvalue()
        self.assertEqual(text.count("[WARN]"), 0, text)    # 已经有基了，不再喊「整场没有」
        self.assertEqual(text.count("[INFO]"), 1, text)
        self.assertIn("palm reference acquired", text)

    def test_a_late_reference_survives_save_and_load(self):
        """补采到的基准单独存盘再读回来：来源照样是 ``late``，旋前照样跟得上。"""
        bridge = self._bridge()
        self._feed(bridge, self.FRAMES)
        self._feed(bridge, PALM_LATE_FRAMES, hand=open_hand())
        with tempfile.TemporaryDirectory() as folder:
            path = save_neutral(Path(folder) / "neutral.json", bridge.neutral_record())
            reloaded = RealtimeArmBridge(PROFILES, sides=("right",),
                                         calibration_frames=self.FRAMES,
                                         neutral=load_neutral(path), frame_dt=0.1)
        entry = reloaded.summary()["neutral"]["right"]
        self.assertTrue(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_reference_source"], "late")
        self.assertIsNotNone(entry["palm_reference_pose_deg"])
        target = reloaded.update(arms_world={"right": bent_arm("right")},
                                 hands={"world_right": twisted_hand("right", 40.0)},
                                 frame_dt=0.1)["h1_2"]["right"]
        self.assertEqual(target.info["orientation_source"], "palm_normal")


class PalmWaitTests(unittest.TestCase):
    """``track`` 模式下「先等一次手点再计帧」（``arm_realtime.PALM_WAIT_FRAMES``）。

    现场那场（``outputs/tmp_realtime/live_run_1552.log``）的日志就是这几条用例要钉的东西：
    ``frame=20 hand valid=left=0 right=0 | cal=track left=20/30 right=0/30`` —— 手臂在、
    手不在，窗口照样在烧。等，但不能无限等（到上限就照旧计数，绝不卡在``cal=``）。
    """

    FRAMES = 6

    def _bridge(self, calibrate_mode="track", wait=20, frames=None):
        return RealtimeArmBridge(PROFILES, sides=("right",),
                                 calibration_frames=(self.FRAMES if frames is None else frames),
                                 calibrate_mode=calibrate_mode, palm_wait_frames=wait,
                                 frame_dt=0.1)

    def _feed(self, bridge, count, hands=False):
        last = None
        for _ in range(count):
            last = bridge.update(arms_world={"right": bent_arm("right")},
                                 hands=({"world_right": open_hand()} if hands else None),
                                 frame_dt=0.1)
        return last

    def test_track_mode_waits_for_the_hand_points_before_counting_frames(self):
        bridge = self._bridge()
        self._feed(bridge, 12)
        # 12 帧手臂、0 帧手：一帧都不算进度，但能看出「在等」。
        self.assertTrue(bridge.calibrating)
        self.assertEqual(bridge.calibration_progress()["right"], (0, self.FRAMES))
        self.assertIn("w=12", bridge.status())
        # 手一出现：计数开始，攒够 6 帧就有掌面基，而且散度是干净的。
        self._feed(bridge, self.FRAMES, hands=True)
        entry = bridge.summary()["neutral"]["right"]
        self.assertFalse(bridge.calibrating)
        self.assertEqual(entry["palm_samples"], self.FRAMES)
        self.assertTrue(entry["orientation_palm_recorded"])
        self.assertEqual(entry["palm_note"], "")

    def test_waiting_still_publishes_a_temporary_neutral_so_the_arm_follows(self):
        bridge = self._bridge()
        update = self._feed(bridge, 3)
        self.assertTrue(bridge.calibrating)
        self.assertIsNotNone(update["h1_2"]["right"].position)

    def test_the_wait_is_bounded_so_a_run_without_hands_still_finishes(self):
        # 等 3 帧就放弃：12 帧手臂里 3 帧放过、9 帧计数 -> 6 帧的标定照样完成。
        bridge = self._bridge(wait=3)
        update = self._feed(bridge, 12)
        self.assertFalse(bridge.calibrating)
        entry = bridge.summary()["neutral"]["right"]
        self.assertEqual(entry["palm_samples"], 0)
        self.assertFalse(entry["orientation_palm_recorded"])
        # 没有样本就没有 note（那是 _warn_missing_palm_basis 按 palm_samples=0 报的事）。
        self.assertEqual(entry["palm_note"], "")
        self.assertEqual(update["h1_2"]["right"].info["orientation_source"], "three_point")

    def test_hold_mode_never_waits(self):
        # 历史行为逐字不变：hold 模式 6 帧就完成，不管手在不在。
        bridge = self._bridge(calibrate_mode="hold", wait=90)
        self._feed(bridge, self.FRAMES)
        self.assertFalse(bridge.calibrating)
        self.assertEqual(bridge.summary()["neutral"]["right"]["palm_samples"], 0)

    def test_zero_disables_the_wait(self):
        bridge = self._bridge(wait=0)
        self._feed(bridge, self.FRAMES)
        self.assertFalse(bridge.calibrating)
        self.assertNotIn("w=", bridge.status())

    def test_recalibration_clears_the_wait_counters(self):
        bridge = self._bridge()
        self._feed(bridge, 4)
        self.assertIn("w=4", bridge.status())
        bridge.reset(recalibrate=True)
        self.assertNotIn("w=4", bridge.status())
        self.assertEqual(bridge.calibration_progress()["right"], (0, self.FRAMES))


class PalmBasisChiralityTests(unittest.TestCase):
    """左右手的掌面基必须**同手性** —— 2026-11-16 深夜：「右手跟得上、左手反着拧，三台一起」。

    机制（离线取证 ``outputs/tmp_probe/left_right_palm_sign.py``）：掌面法向用
    ``cross(index - wrist, pinky - wrist)`` 算，而关键点的绕序左右**镜像** —— 同一个叉乘在右手
    给出「掌心朝外」、在左手给出「掌心朝内」。又因为 ``y = z x x`` 是推算出来的，那个基形式上
    仍是正交旋转（det 恒为 +1），差别只藏在「z 指反」里，等价于**整帧绕前臂轴多转 180 度**。

    后果分两种维：绕前臂的自转在 ``R_neutral^T @ R_now`` 里被共轭约掉（两侧都给 +40 度，
    所以「拧」这一维本来就对），而 **pitch/yaw 两维在左侧整体反号**。现场的表现就是左手目标
    系统性偏 180 度、关节一路被推进行程两端：同一跑报告 ``arm.wrist_usage`` 里 h1_2 左
    ``wrist_pitch`` usage 1.000（顶满）/ ``wrist_yaw`` 0.000（顶死另一端），右 0.283/0.248 全在
    中段；g1 左 ``wrist_roll`` 1.000（顶满）、右 0.184；``direction_glitch`` 左 77 / 右 39。

    这里的判据是三个数：掌面法向指回掌心外侧、镜像姿势下两侧的基逐位相同、同一份世界旋转在
    两侧给出**同号**的转角。
    """

    WRIST = np.array((0.20, 0.30, 0.0))
    ELBOW = WRIST - np.array((0.0, 0.25, 0.0))

    def _arm(self):
        """(肩, 肘, 腕)：前臂沿 world +y（MediaPipe 口径 y 向下，即小臂朝下）。"""
        return np.stack((self.ELBOW - np.array((0.0, 0.22, 0.0)), self.ELBOW, self.WRIST))

    def _hand(self, side, rotation=None):
        """21 点手：掌面朝相机（-z）、手指朝上（-y）、食指 MCP 在拇指侧；左右手在 x 上镜像。"""
        across = 0.025 if side == "right" else -0.025
        points = np.zeros((21, 3))
        points[5] = (across, -0.09, 0.0)      # 食指 MCP
        points[17] = (-across, -0.09, 0.0)    # 小指 MCP
        for slot in range(1, 21):
            if slot not in (5, 17):
                points[slot] = (0.0, -0.02 - 0.005 * (slot % 5), 0.0)
        if rotation is not None:
            points = (rotation @ points.T).T
        return points + self.WRIST

    @staticmethod
    def _about(axis, angle_deg):
        """绕给定世界轴的旋转（罗德里格斯，只用来造测试数据）。"""
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        angle = np.radians(angle_deg)
        skew = np.array(((0.0, -axis[2], axis[1]),
                        (axis[2], 0.0, -axis[0]),
                        (-axis[1], axis[0], 0.0)))
        return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)

    def _frame(self, side, rotation=None):
        return palm_orientation_frame(self._hand(side, rotation), self._arm(), side)

    def _relative(self, side, rotation):
        return self._frame(side).T @ self._frame(side, rotation)

    def _twist_deg(self, side, rotation):
        """绕前臂轴（基的 x）那个带符号角：``R_rel @ y`` 落在 (y, z) 平面上。"""
        column = self._relative(side, rotation) @ np.array((0.0, 1.0, 0.0))
        return float(np.degrees(np.arctan2(column[2], column[1])))

    def _swing_relative(self, side, rotation):
        """整条手臂（肩肘腕 + 21 点手）一起摆：绕着腕点转，返回相对中性的旋转。

        为什么要让手臂一起动：掌面基的两个轴里 x 是**前臂方向**、z 是**掌面法向**，所以腕
        「在自己平面里折/偏」对这套基是不可见的（只有旋前那一维看得见）。而现场那个「左手
        反了」是全姿态的偏差，只有手臂真的动起来、前臂方向变了，相对旋转里才出现绕另外
        两个轴的分量 —— 那两维正是被 180 度共轭翻掉的。
        """
        shift = self.WRIST
        arm = (rotation @ (self._arm() - shift).T).T + shift
        hand = (rotation @ (self._hand(side) - shift).T).T + shift
        return (palm_orientation_frame(self._hand(side), self._arm(), side).T
                @ palm_orientation_frame(hand, arm, side))

    def test_the_palm_normal_points_out_of_the_palm_on_both_sides(self):
        """掌面都朝相机（-z）=> 基的 z 轴 z 分量必须是 -1；左手不取反时这里是 +1。"""
        for side in ("right", "left"):
            self.assertAlmostEqual(float(self._frame(side)[2, 2]), -1.0, places=6, msg=side)

    def test_a_mirrored_pose_builds_the_same_basis_on_both_sides(self):
        self.assertTrue(np.allclose(self._frame("left"), self._frame("right"), atol=1e-9))

    def test_both_sides_report_the_same_twist(self):
        """双手朝同一方向拧（同一个世界旋转）：两侧的旋前角必须同号同大小。"""
        rotation = self._about((0.0, 1.0, 0.0), 40.0)
        right, left = (self._twist_deg(side, rotation) for side in ("right", "left"))
        self.assertAlmostEqual(abs(right), 40.0, places=6)
        self.assertAlmostEqual(left, right, places=6)

    def test_both_sides_report_the_same_swing(self):
        """手臂一起摆：两侧的相对旋转必须逐位相同（左侧不再被 180 度共轭翻掉）。"""
        rotation = self._about((1.0, 0.0, 0.0), 30.0)
        right = self._swing_relative("right", rotation)
        left = self._swing_relative("left", rotation)
        self.assertGreater(np.degrees(np.arccos(np.clip((np.trace(right) - 1) / 2, -1, 1))),
                           10.0, "测试姿势离中性太近，量不出偏差")
        self.assertTrue(np.allclose(left, right, atol=1e-9), f"\n{left}\n{right}")

    def test_the_bridge_feeds_the_side_into_the_basis(self):
        """**接线**测试：`_raw_orientation` 必须把 side 传进基的构造（漏传 = 把 bug 装回去）。

        上面几条都直接调 ``palm_orientation_frame``，抓不到「某个调用点忘了传 side」；这里走
        整条桥（标定 -> 一帧动过的手），把桥吐出来的四元数跟**带 side 直接算**的结果比。

        姿势必须**同时有摆动和拧**：纯旋前在「翻/不翻」两种构造下给出**同一个**旋转（共轭约掉，
        见 ``test_both_sides_report_the_same_twist``），只用拧是验不出接线漏没漏的。所以这里拿
        ``side="right"``（= 不取反的那一套）再算一遍当反例，确认这条测试真的能分辨。
        """
        from teleoperation.retargeting.arm.wrist_targets import rotation_matrix_to_quaternion

        frames = 4
        bridge = RealtimeArmBridge(PROFILES, sides=("left",),
                                   calibration_frames=frames, frame_dt=0.1)
        arm = bent_arm("left")
        still = self._normalized_left_hand()
        for _ in range(frames):
            bridge.update(arms_world={"left": arm},
                          hands={"world_left": still}, frame_dt=0.1)
        neutral_palm = bridge.neutral["left"]["orientation_palm"]
        self.assertIsNotNone(neutral_palm, "标定没有拿到掌面基，这条测试就没验到东西")

        # 摆 12 度 + 拧 8 度：合起来在步长上限之内（不带限速），且一定含非前臂轴分量。
        swing = self._about((1.0, 0.0, 0.0), 12.0)
        arm_moved = (swing @ (arm - arm[2]).T).T + arm[2]
        hand_moved = self._twist_about_forearm(
            (swing @ (still - still[0]).T).T + still[0], arm_moved, 8.0)
        bridge.update(arms_world={"left": arm_moved},
                      hands={"world_left": hand_moved}, frame_dt=0.1)

        emitted = np.asarray(bridge.orientation_target()["left"], dtype=np.float64)
        expected = np.asarray(rotation_matrix_to_quaternion(
            relative_palm_rotation(hand_moved, arm_moved, neutral_palm, "left")))
        # 反例 = 不取反的那套（同一个左手、同一个姿势）。
        unflipped = np.asarray(rotation_matrix_to_quaternion(
            relative_palm_rotation(hand_moved, arm_moved, neutral_palm, "right")))
        self.assertLess(abs(float(np.dot(expected, unflipped))), 0.5,
                        "反例与期望几乎重合，这条测试分辨不出接线")
        self.assertAlmostEqual(abs(float(np.dot(emitted, expected))), 1.0, places=6)
        self.assertGreater(abs(float(np.dot(emitted, expected))),
                           abs(float(np.dot(emitted, unflipped))) + 1e-3)

    @staticmethod
    def _normalized_left_hand():
        """一只**左手**的 21 点（``open_hand`` 是右手口径，按 x 镜像就是左手）。"""
        hand = open_hand()
        hand[:, 0] = 1.0 - hand[:, 0]
        return hand

    @classmethod
    def _twist_about_forearm(cls, hand, arm, angle_deg):
        """把手的 21 点绕**前臂轴**（腕 - 肘）拧 ``angle_deg``，腕点不动。"""
        axis = arm[2] - arm[1]
        axis = axis / np.linalg.norm(axis)
        rotation = cls._about(axis, angle_deg)
        wrist = hand[0]
        return (rotation @ (hand - wrist).T).T + wrist


class OperatorSymmetryTests(unittest.TestCase):
    """「你的两条手臂是不是镜像」——把现场那句「左臂跟着右臂」变成一个数字。

    背景（2026-11-17 现场）："屈臂、双手放胸前、把肘向外打开，结果左臂跟着右臂的方向动"。
    桥的这一层**不可能**把镜像手势算成同向：``AxisMap`` 是两侧共用的一个线性映射（置换 +
    符号），而按侧的那三个量（肩锚点 / 上臂长 / 前臂长）实测左右镜像到 0.07 mm
    （``outputs/tmp_probe/probe_arm_anchor_mirror.py``：h1_2 0.07 / g1 0.01 / gr1_t2 5.4 mm）。
    线性映射 + 镜像输入 -> 输出必然镜像。

    于是「同向」只可能来自**输入本身**：一侧手点被挡/跑偏，或手势本来就不对称。这两种情况
    的处理完全不同，所以必须能从报告里分开看 —— 这组用例把两种输入各自钉住。
    """

    FRAMES = 4

    def _bridge(self, **kwargs):
        kwargs.setdefault("calibrate_mode", "hold")
        kwargs.setdefault("palm_wait_frames", 0)
        return RealtimeArmBridge(PROFILES, sides=SIDES, calibration_frames=self.FRAMES,
                                 frame_dt=0.1, **kwargs)

    @staticmethod
    def _arm(side, upper):
        """一侧 (肩, 肘, 腕)：上臂方向**直接给**（前臂固定朝上前方，肘是弯的）。"""
        sign = 1.0 if side == "left" else -1.0
        shoulder = np.array([0.17 * sign, -0.35, 0.0])
        direction = np.asarray(upper, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        elbow = shoulder + 0.30 * direction
        fore = np.array([0.0, -0.30, 0.30])
        wrist = elbow + 0.30 * fore / np.linalg.norm(fore)
        return np.stack((shoulder, elbow, wrist))

    def _feed(self, bridge, left_upper, right_upper, frames=None):
        arms = {"left": self._arm("left", left_upper),
                "right": self._arm("right", right_upper)}
        last = None
        for _ in range(int(frames or self.FRAMES)):
            last = bridge.update(arms_world=arms, hands=None, frame_dt=0.1)
        return last

    def test_mirrored_arms_report_no_same_side_frames(self):
        """把肘向外打开（左手朝我左、右手朝我右）= 镜像手势 -> 偏差 0 度。"""
        bridge = self._bridge()
        self._feed(bridge, (0.5, 0.85, 0.0), (-0.5, 0.85, 0.0))
        symmetry = bridge.summary()["operator_symmetry"]
        self.assertEqual(symmetry["frames"], self.FRAMES)
        self.assertLess(symmetry["median_deg"], 1.0)
        self.assertEqual(symmetry["same_side_frames"], 0)
        self.assertIn("mirror images", symmetry["note"])

    def test_two_arms_swinging_the_same_way_are_reported(self):
        """两条上臂朝**同一边**摆（现场看到的那种输入）-> 偏差远大于阈值。"""
        bridge = self._bridge()
        self._feed(bridge, (0.5, 0.85, 0.0), (0.5, 0.85, 0.0))
        symmetry = bridge.summary()["operator_symmetry"]
        self.assertEqual(symmetry["frames"], self.FRAMES)
        self.assertGreater(symmetry["median_deg"], SYMMETRY_SAME_SIDE_DEG)
        self.assertEqual(symmetry["same_side_frames"], self.FRAMES)
        self.assertIn("the same way", symmetry["note"])

    def test_one_side_alone_is_not_counted(self):
        """只有一侧可用时不记：拿单侧数据凑出来的「对称」是假的。"""
        bridge = self._bridge()
        bridge.update(arms_world={"right": self._arm("right", (0.5, 0.85, 0.0))},
                      hands=None, frame_dt=0.1)
        symmetry = bridge.summary()["operator_symmetry"]
        self.assertEqual(symmetry["frames"], 0)
        self.assertEqual(symmetry["note"], "no frame had both arms usable")

    def test_sample_buffer_is_capped_but_counting_continues(self):
        """封顶只影响内存：超出的帧记进 ``frames_not_sampled``，不静默丢。"""
        bridge = self._bridge()
        bridge._symmetry = [0.0] * SYMMETRY_MAX_SAMPLES
        self._feed(bridge, (0.5, 0.85, 0.0), (-0.5, 0.85, 0.0), frames=2)
        symmetry = bridge.summary()["operator_symmetry"]
        self.assertEqual(symmetry["frames"], SYMMETRY_MAX_SAMPLES)
        self.assertEqual(symmetry["frames_not_sampled"], 2)




# --------------------------------------------------------------------------- #
# 2026-11-17 15:40 场：「手有时卡住不动」—— 卡住的是哪只手、多久、为什么都要说出来
# --------------------------------------------------------------------------- #
class HandStallWarningTests(unittest.TestCase):
    """``realtime_teleop._warn_stalled_hand``：``angles is None`` 时保持上一帧是**故意**的
    （绝不让关节跳），所以现场看到的「手卡住不动」必须由日志解释，而不是只有一个
    ``invalid=`` 在涨。

    实测（``outputs/tmp_realtime/live_run_1547.log`` + ``arm_run_20261117_154001.json``）：
    右手 486 帧里 172 帧没有角度（35%），左只有 37；单次丢帧被
    ``TemporalBuffer(reset_on_missing=True)`` 放大成 3 帧，成串丢失就成片「不动」。
    """

    def _warn(self, **kwargs):
        from teleoperation.simulation.realtime_teleop import _warn_stalled_hand

        warned = kwargs.pop("warned", set())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _warn_stalled_hand(kwargs.pop("side", "right"), kwargs.pop("streak", 20),
                               kwargs.pop("raw_frame", None), warned, **kwargs)
        return output.getvalue(), warned

    def test_a_brief_dropout_never_warns(self):
        # 单次丢帧 = 3 帧没有角度（三帧缓冲），那是常态，不值得刷一条 WARN。
        text, warned = self._warn(streak=3)
        self.assertEqual(text, "")
        self.assertEqual(warned, set())

    def test_a_long_stall_names_the_side_the_frames_and_the_reason(self):
        window = SimpleNamespace(metadata={"invalid_reasons": {"right": "missing"}})
        text, warned = self._warn(streak=172, raw_frame={"metadata": {}}, window=window,
                                  interval=0.1)
        self.assertIn("[WARN] right hand: no angles for 172 frames (~17.2 s)", text)
        self.assertIn("holds its last pose (it never jumps)", text)
        self.assertIn("the arm keeps following", text)
        self.assertIn("MediaPipe reported no hand points for that hand", text)
        self.assertIn("one arm length away", text)
        self.assertEqual(warned, {"right"})
        # 同一侧第二次不再喊：一场只留一条，复发看状态行的 invalid=。
        again, _ = self._warn(streak=400, warned=warned)
        self.assertEqual(again, "")

    def test_the_camera_layer_reason_is_used_when_the_pipeline_has_none(self):
        raw_frame = {"metadata": {"invalid_reasons": {
            "right": "Duplicate handedness: Right; refusing to overwrite a hand"}}}
        text, _ = self._warn(streak=20, raw_frame=raw_frame)
        self.assertIn("Duplicate handedness: Right", text)

    def test_without_any_reason_it_blames_the_three_frame_buffer(self):
        window = SimpleNamespace(metadata={"invalid_reasons": {"right": None}})
        text, _ = self._warn(streak=20, raw_frame={"metadata": {}}, window=window)
        self.assertIn("3 clean frames in a row", text)
        self.assertNotIn("MediaPipe reported no hand points", text)

    def test_the_two_sides_are_tracked_independently(self):
        warned = set()
        left, warned = self._warn(side="left", streak=20, raw_frame={}, warned=warned)
        right, warned = self._warn(side="right", streak=20, raw_frame={}, warned=warned)
        self.assertIn("left hand", left)
        self.assertIn("right hand", right)
        self.assertEqual(warned, {"left", "right"})

