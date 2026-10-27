"""``simulation/arm_preview.py`` 的回归用例：采集窗口的手臂层（只画，不碰算法）。

不需要相机、不需要显示器：用一个假 sink（就是 ``{"frame": ndarray}``）+ ``show_window=False``，
直接看 ``annotate`` 出来的像素有没有被改过，以及 ``build_arm_preview`` 的开关语义。

判据都是"画没画"而不是"画得像不像"：像素位置依赖 MediaPipe 坐标与机器分辨率，
断言具体坐标会变成脆测；这里只锁四件真会回归的事 ——
(1) 合法点能画出来、非法点被当"没有"；(2) ``--no-arm-overlay`` 下逐像素等于基线；
(3) 没相机 / 不需要窗口时返回 ``None`` 而不是抛异常；
(4) ``devtools/run_realtime_windows.py`` 跑完回的那句"窗口开没开"，必须按报告里的
``window_rect`` 判断（曾经把"按 q 正常退出"误报成"窗口没开"，见文末那个类）。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from teleoperation.simulation.arm_preview import (
    ARM_POINT_COUNT,
    ARM_PREVIEW_WINDOW,
    ArmCameraPreview,
    build_arm_preview,
    finite_arm_points,
    outside_frame,
)

SIDES = ("left", "right")
HEIGHT, WIDTH = 240, 320


def blank_frame():
    """全 0 画布：任何被画上去的东西都能用非零像素抓出来。"""
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def arm_at(x=0.20):
    """一条左手：肩 -> 肘 -> 腕，归一化坐标（与 inputs/pose_mediapipe.py 同口径）。"""
    return np.asarray([[x, 0.20, 0.0], [x + 0.10, 0.35, 0.0], [x + 0.20, 0.50, 0.0]],
                      dtype=np.float64)


def make_preview(sides=SIDES, frame=None, **kwargs):
    kwargs.setdefault("show_window", False)
    return ArmCameraPreview({"frame": blank_frame() if frame is None else frame}, sides, **kwargs)


def arm_layer(preview, arms=None, lines=None):
    """同一预览画两次（先"没有手臂"，再给数据）相减，只留下手臂层改过的像素。"""
    raw = {"left": None, "right": None}
    before = preview.annotate(raw, {}, "")
    preview.update(raw, {}, "", arms=arms, lines=lines)
    after = preview.annotate(raw, {}, "")
    return np.abs(after.astype(int) - before.astype(int)).sum(axis=2)


class FiniteArmPointsTests(unittest.TestCase):
    def test_a_valid_side_is_kept_and_the_missing_one_is_blanked(self):
        points = finite_arm_points({"left": arm_at(), "right": None}, SIDES)

        self.assertEqual(np.shape(points["left"]), (3, 3))
        self.assertIsNone(points["right"])

    def test_wrong_shapes_are_treated_as_no_data(self):
        for bad in (np.zeros((2, 3)), np.zeros((3, 2)), np.zeros((3, 3, 1)), [1.0, 2.0], "abc"):
            with self.subTest(value=str(bad)):
                self.assertIsNone(finite_arm_points({"left": bad}, SIDES)["left"])

    def test_a_single_non_finite_coordinate_blanks_that_side(self):
        broken = arm_at()
        broken[1, 1] = np.nan
        other = np.full((3, 3), np.inf)

        points = finite_arm_points({"left": broken, "right": other}, SIDES)

        self.assertIsNone(points["left"])
        self.assertIsNone(points["right"])

    def test_no_arm_dict_at_all_gives_none_for_every_side(self):
        points = finite_arm_points(None, SIDES)

        self.assertEqual(sorted(points), ["left", "right"])
        self.assertTrue(all(points[side] is None for side in SIDES))


class OverlayDrawingTests(unittest.TestCase):
    def test_the_arm_chain_is_painted_on_the_canvas(self):
        diff = arm_layer(make_preview(), arms={"left": arm_at(), "right": None})

        self.assertGreater(int(diff.sum()), 0)

    def test_no_arm_data_changes_nothing(self):
        diff = arm_layer(make_preview(), arms=None, lines=None)

        self.assertEqual(int(diff.sum()), 0)

    def test_without_the_overlay_the_canvas_stays_the_baseline(self):
        preview = make_preview(overlay=False)
        diff = arm_layer(preview, arms={"left": arm_at(), "right": arm_at(0.55)},
                         lines={"left": "left P", "right": "right P"})

        self.assertEqual(int(diff.sum()), 0)
        self.assertFalse(preview.overlay)

    def test_non_finite_arm_points_never_reach_the_drawing_code(self):
        nan = np.full((3, 3), np.nan)
        diff = arm_layer(make_preview(), arms={"left": nan, "right": nan})

        self.assertEqual(int(diff.sum()), 0)

    def test_the_number_line_is_drawn_even_without_arm_points(self):
        diff = arm_layer(make_preview(), arms=None, lines={"left": "left - d(cm)--"})

        self.assertGreater(int(diff.sum()), 0)

    def test_mirroring_flips_the_arm_layer_to_the_other_half(self):
        arms = {"left": arm_at(), "right": None}
        plain = arm_layer(make_preview(mirror=False), arms=arms)
        mirrored = arm_layer(make_preview(mirror=True), arms=arms)

        # 肩在画面左侧（x=0.2）：关掉镜像时整层在左半边，开了镜像就翻到右半边
        # （窗口默认就是镜像的，像照镜子）。
        self.assertLess(float(np.nonzero(plain)[1].mean()), WIDTH / 2)
        self.assertGreater(float(np.nonzero(mirrored)[1].mean()), WIDTH / 2)

    def test_a_camera_without_a_frame_returns_no_canvas_instead_of_raising(self):
        preview = make_preview()
        preview.sink["frame"] = None

        self.assertIsNone(preview.annotate({"left": None, "right": None}, {}, ""))

    def test_the_canvas_keeps_the_frame_geometry_and_dtype(self):
        preview = make_preview()
        preview.update({"left": None, "right": None}, {}, "", arms={"left": arm_at()})

        canvas = preview.annotate({"left": None, "right": None}, {}, "")

        self.assertEqual(canvas.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(canvas.dtype, np.uint8)


class OutsideFrameTests(unittest.TestCase):
    """真机实测到的现象：人离相机太近时肩/肘/腕会落到画面外，这时必须给提示而不是"看不见线"。"""

    def test_all_points_inside_are_counted_as_zero(self):
        arms = np.asarray([[0.5, 0.5], [0.55, 0.55], [0.6, 0.6]], dtype=np.float64)
        pixels = make_preview().pixels(arms, WIDTH, HEIGHT)

        self.assertEqual(outside_frame(pixels, WIDTH, HEIGHT), 0)

    def test_points_beyond_the_canvas_are_counted_one_by_one(self):
        arms = np.asarray([[0.5, 0.5], [0.5, 1.4], [0.5, 1.9]], dtype=np.float64)
        pixels = make_preview().pixels(arms, WIDTH, HEIGHT)

        self.assertEqual(outside_frame(pixels, WIDTH, HEIGHT), 2)
        self.assertEqual(ARM_POINT_COUNT, len(arms))

    def test_an_arm_that_left_the_frame_still_gets_the_hint(self):
        # 三个点全在画面外：折线与"left arm"标签都画不出像素，能改动画面的只有提示那一行。
        far_away = np.asarray([[3.0, 3.0, 0.0], [3.1, 3.1, 0.0], [3.2, 3.2, 0.0]])
        diff = arm_layer(make_preview(), arms={"left": far_away, "right": None})

        self.assertGreater(int(diff.sum()), 0)


def preview_args(**overrides):
    """``build_arm_preview`` 需要的那些 ``realtime_teleop`` 参数（默认不开窗口）。"""
    values = dict(headless=True, no_preview=False, preview_dump_dir=None,
                  preview_dump_frames=5, preview_no_mirror=False,
                  preview_window_pos=(20, 60), preview_no_topmost=False,
                  no_arm_overlay=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_camera():
    """只要有 ``_input._cap``：``attach_preview_capture`` 就是靠它拿到采集对象的。"""
    return SimpleNamespace(_input=SimpleNamespace(_cap=object()))


class BuildArmPreviewTests(unittest.TestCase):
    def test_headless_without_a_dump_dir_needs_no_preview(self):
        self.assertIsNone(build_arm_preview(preview_args(), fake_camera(), SIDES))

    def test_no_preview_flag_needs_no_preview(self):
        args = preview_args(headless=False, no_preview=True)

        self.assertIsNone(build_arm_preview(args, fake_camera(), SIDES))

    def test_headless_with_a_dump_dir_only_writes_pngs(self):
        with TemporaryDirectory() as folder:
            args = preview_args(preview_dump_dir=folder, preview_dump_frames=2)
            preview = build_arm_preview(args, fake_camera(), SIDES)

            self.assertIsNotNone(preview)
            self.assertFalse(preview.show)
            self.assertTrue(preview.overlay)
            self.assertEqual(preview.dump_frames, 2)

    def test_window_mode_uses_the_hands_plus_arms_title(self):
        preview = build_arm_preview(preview_args(headless=False), fake_camera(), SIDES)

        self.assertIsNotNone(preview)
        self.assertTrue(preview.show)
        self.assertEqual(preview.window_name, ARM_PREVIEW_WINDOW)
        self.assertTrue(preview.topmost)
        self.assertTrue(preview.mirror)

    def test_the_overlay_flag_reaches_the_preview(self):
        args = preview_args(headless=False, no_arm_overlay=True)
        preview = build_arm_preview(args, fake_camera(), SIDES)

        self.assertIsNotNone(preview)
        self.assertFalse(preview.overlay)

    def test_a_camera_without_a_capture_object_skips_the_preview(self):
        workflow = SimpleNamespace(_input=None)

        self.assertIsNone(build_arm_preview(preview_args(headless=False), workflow, SIDES))


LAUNCHER = Path(__file__).resolve().parents[1] / "devtools" / "run_realtime_windows.py"


def load_launcher():
    """按路径加载 ``devtools/run_realtime_windows.py``：它不在安装包里，不靠 sys.path。"""
    spec = importlib.util.spec_from_file_location("run_realtime_windows_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preview_report(directory, **preview):
    """写一份只有 ``preview`` 块的报告：``summarise_camera_window`` 只读这一块。"""
    path = Path(directory) / "arm_run_bullet_report.json"
    path.write_text(json.dumps({"preview": preview}, ensure_ascii=False), encoding="utf-8")
    return path


def summarise(module, report, want_window=True):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        module.summarise_camera_window(report, want_window)
    return stream.getvalue()


class CameraWindowSummaryTests(unittest.TestCase):
    """``devtools/run_realtime_windows.summarise_camera_window``：跑完回的那一句话。

    真实事故（2026-10-14）：按 q 退出后末尾打的是 ``[WARN] the camera window did NOT
    stay open: unknown reason``，而日志里明明有 ``camera window: on (mirrored)``、报告里
    也有 ``window_rect``。原因是队友入口收尾时把 ``CameraPreview.show`` 置 ``False``
    （``show`` 与写报告是竞态），所以判据只能是 ``window_rect``。
    """

    def setUp(self):
        self.launcher = load_launcher()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_a_window_that_is_still_open_reports_its_rectangle(self):
        report = preview_report(self.tmp.name, window=True, window_rect=[28, 91, 640, 480],
                                mirrored=True, arm_overlay=True, note="")

        text = summarise(self.launcher, report)

        self.assertIn("[OK", text)
        self.assertIn("[28, 91, 640, 480]", text)
        self.assertNotIn("did NOT stay open", text)

    def test_quitting_with_q_is_not_reported_as_a_missing_window(self):
        report = preview_report(self.tmp.name, window=False, window_rect=[28, 91, 640, 480],
                                mirrored=True, arm_overlay=True, note="")

        text = summarise(self.launcher, report)

        self.assertIn("[OK", text)
        self.assertIn("was open at [28, 91, 640, 480]", text)
        self.assertNotIn("did NOT stay open", text)

    def test_a_window_that_never_opened_keeps_the_note_and_a_hint(self):
        note = "window unavailable here (no HighGUI); preview kept off"
        report = preview_report(self.tmp.name, window=False, window_rect=None, note=note)

        text = summarise(self.launcher, report)

        self.assertIn("[WARN", text)
        self.assertIn("no HighGUI", text)
        self.assertIn("--preview-window-pos 20,60", text)

    def test_no_window_requested_stays_quiet(self):
        report = preview_report(self.tmp.name, window=False, window_rect=None, note="")

        self.assertEqual(summarise(self.launcher, report, want_window=False), "")


if __name__ == "__main__":
    unittest.main()
