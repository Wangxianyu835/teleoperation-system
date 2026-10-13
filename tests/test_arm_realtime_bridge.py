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
    AxisMap,
    RealtimeArmBridge,
    arm_reach,
    elbow_bend_degrees,
    load_neutral,
    save_neutral,
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
        bridge = RealtimeArmBridge(PROFILES, sides=("right",), calibration_frames=1)
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


if __name__ == "__main__":
    unittest.main()





