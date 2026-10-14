"""unit tests: simulation/arm_realtime.py (relative arm retargeting) + the replay source.

这些用例**不碰 PyBullet、不开摄像头**：桥本身是纯 numpy，所以能在这里把相对映射、
中性标定、丢帧保持、兜底来源、每侧行程都算清楚。仿真侧"手臂真的跟上了"的判据在
``devtools/verify_realtime_arm_bridge.py``（那里才有残差、行程、漂移的实测数字）。

运行::

    $env:PYTHONPATH = "$pwd\\src"
    E:\\python3.11.7\\python.exe -m unittest tests.test_arm_realtime_bridge -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_profiles import PAPER_ROBOTS, profile_for
from teleoperation.simulation.arm_realtime import (
    CALIBRATE_MODES,
    DEFAULT_LATERAL_GUARD_M,
    GLITCH_HOLD_FRAMES,
    HOLD_REASONS,
    MAX_DIRECTION_RATE_DEG_S,
    MIN_ELBOW_PLANE_SINE,
    AxisMap,
    RealtimeArmBridge,
    arm_reach,
    elbow_bend_degrees,
    load_neutral,
    relative_wrist_rotation,
    save_neutral,
    wrist_orientation_frame,
)
from teleoperation.simulation.realtime_teleop import ReplayFrameSource, build_parser

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
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=30,
                                   calibrate_mode="track")
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
                                     calibrate_mode="track")
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
                                   calibrate_mode="track")
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

    def _calibrated_on_a_bent_arm(self, frames: int = 4) -> RealtimeArmBridge:
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=frames)
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

        bridge = self._calibrated_on_a_bent_arm()
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

    def test_the_hand_fallback_carries_no_relative_orientation(self):
        bridge = self._calibrated_on_a_bent_arm()
        entry = bridge.update(arms_world={"right": None},
                              hands={"right": open_hand()})["h1_2"]["right"]
        self.assertEqual(entry.source, "hand")
        self.assertIsNone(entry.orientation)
        self.assertIn("no elbow", entry.info["orientation_source"])


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



if __name__ == "__main__":
    unittest.main()





