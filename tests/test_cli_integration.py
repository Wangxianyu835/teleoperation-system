"""Exercise the public CLI and moved realtime implementation without hardware."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleoperation.cli import build_parser, main
from teleoperation.paths import DEFAULT_REALTIME_SNAPSHOT, PROJECT_ROOT
from teleoperation.retargeting.hand.predictor import TwoHandRetargeter
from tests.test_coordinate_pipeline import CountingModel
from tests.test_mediapipe_realtime import raw_frame


class CliIntegrationTests(unittest.TestCase):
    def test_root_help_works_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-m", "teleoperation", "hand", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("train", "export", "inspect", "align", "realtime"):
            self.assertIn(name, result.stdout)

    def test_selected_command_only_is_configured_and_parser_can_be_reused(self):
        import teleoperation.cli as cli
        with mock.patch.object(cli, "import_module", wraps=cli.import_module) as imports:
            parser = build_parser()
            imports.assert_not_called()
            parser.parse_args(["hand", "align", "--input", "raw.h5", "--output", "aligned.h5"])
            parser.parse_args(["hand", "align", "--input", "raw2.h5", "--output", "aligned2.h5"])
            imports.assert_not_called()
            parser.parse_args(["hand", "inspect", "--angle-h5", "angles.h5"])
            imports.assert_not_called()

    def test_realtime_invalid_options_fail_before_loading_or_capture(self):
        for options in (
            ["--frames", "0"], ["--frames", "-1"], ["--headless"],
            ["--visualize", "--headless", "--frames", "0"],
            ["--visualize", "--palm-radius", "nan"],
            ["--visualize", "--robot-radius", "0"],
            ["--visualize", "--snapshot", "bad.jpg"],
        ):
            with self.subTest(options=options), mock.patch("teleoperation.apps.hand_realtime.MediaPipeCameraInput") as capture, \
                    mock.patch("teleoperation.apps.hand_realtime.load_palm_local_retargeter") as load:
                with self.assertRaises(ValueError):
                    main(["hand", "realtime", *options])
                capture.assert_not_called()
                load.assert_not_called()

    def _capture(self):
        capture = mock.Mock()
        sequence = [raw_frame(index, left=index != 3) for index in range(7)]
        for index, frame in enumerate(sequence):
            frame["timestamp"] = 1_700_000_000_000 + index * 33
        capture.next_observation.side_effect = [RawHandFrame({side: frame[side] for side in ("left", "right")}, frame["timestamp"], "mediapipe_approx", frame.get("metadata", {})) for frame in sequence]
        return capture

    def test_console_realtime_executes_windows_resets_and_releases_capture(self):
        capture = self._capture()
        model = CountingModel()
        with mock.patch("teleoperation.apps.hand_realtime.MediaPipeCameraInput", return_value=capture), \
                mock.patch("teleoperation.apps.hand_realtime.load_palm_local_retargeter", return_value=TwoHandRetargeter(model)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["hand", "realtime", "--frames", "7"]), 0)
        capture.close.assert_called_once()
        self.assertEqual(capture.next_observation.call_count, 7)
        self.assertEqual(model.forward_calls, 7)  # left: 2; right: 5
        self.assertIn("recovery_after_3_valid_frames", output.getvalue())
        self.assertIn("window=(3, 25, 3)", output.getvalue())
        self.assertIn("angles=(18,)", output.getvalue())

    def test_headless_visualization_uses_shared_plotting_and_saves_png(self):
        try:
            import cv2
            import teleoperation.apps.hand_visualization as visualization
        except ModuleNotFoundError as error:
            self.skipTest(f"Optional visualization dependency unavailable: {error}")
        capture = self._capture()
        model = CountingModel()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "snapshot.png"
            with mock.patch("teleoperation.apps.hand_realtime.MediaPipeCameraInput", return_value=capture), \
                    mock.patch.object(visualization, "load_palm_local_retargeter", return_value=TwoHandRetargeter(model)), \
                    mock.patch.object(cv2, "namedWindow", side_effect=AssertionError("headless opened GUI")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([
                    "hand", "realtime", "--visualize", "--headless", "--frames", "7", "--snapshot", str(image),
                ]), 0)
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            decoded = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (740, 1260, 3))
        capture.close.assert_called_once()
        self.assertEqual(model.forward_calls, 7)
        self.assertIn("'left': 2", output.getvalue())
        self.assertIn("'right': 5", output.getvalue())
        self.assertEqual(__import__("teleoperation.tools.plotting", fromlist=["project"]).project.__module__, "teleoperation.tools.plotting")

    def test_realtime_capture_failure_returns_failure_and_releases(self):
        capture = mock.Mock()
        capture.next_observation.side_effect = RuntimeError("synthetic capture failure")
        with mock.patch("teleoperation.apps.hand_realtime.MediaPipeCameraInput", return_value=capture), \
                mock.patch("teleoperation.apps.hand_realtime.load_palm_local_retargeter", return_value=TwoHandRetargeter(CountingModel())), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["hand", "realtime", "--frames", "1"]), 1)
        self.assertTrue(capture.close.called)
        self.assertIn("synthetic capture failure", output.getvalue())

    def test_visualization_s_shortcut_keeps_default_snapshot_after_move(self):
        try:
            import cv2
            import teleoperation.apps.hand_visualization as visualization
        except ModuleNotFoundError as error:
            self.skipTest(f"Optional visualization dependency unavailable: {error}")
        capture = self._capture()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("teleoperation.apps.hand_realtime.MediaPipeCameraInput", return_value=capture))
            stack.enter_context(mock.patch.object(visualization, "load_palm_local_retargeter", return_value=TwoHandRetargeter(CountingModel())))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(mock.patch.object(cv2, name))
            stack.enter_context(mock.patch.object(cv2, "waitKey", return_value=ord("s")))
            stack.enter_context(mock.patch.object(cv2, "getWindowProperty", return_value=1))
            save = stack.enter_context(mock.patch.object(visualization, "save_png"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(main(["hand", "realtime", "--visualize", "--frames", "1"]), 0)
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.args[0], DEFAULT_REALTIME_SNAPSHOT)
            cv2.destroyAllWindows.assert_called_once()
        capture.close.assert_called_once()

from teleoperation.contracts.observations import RawHandFrame
