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
from teleoperation.paths import PROJECT_ROOT


class CliIntegrationTests(unittest.TestCase):
    def test_dual_realtime_keeps_the_typed_native_observation_entry(self):
        args = build_parser().parse_args([
            'dual', 'realtime', '--adapter', 'module:factory',
            '--calibration', 'calibration.json', '--checkpoint', 'shared.pth',
        ])
        self.assertEqual(args._handler, 'teleoperation.applications.dual_realtime:main')
        self.assertEqual(args.checkpoint, Path('shared.pth'))
        self.assertFalse(hasattr(args, 'hand_preprocessing'))

    def test_dual_camera_exposes_isolated_smplerx_worker_options(self):
        args = build_parser().parse_args([
            "dual", "camera", "--arms-only", "--calibration", "calibration.json",
            "--smplerx-worker-python", "worker-python",
            "--smplerx-root", "smplerx", "--smplerx-checkpoint", "s32.pth",
            "--smplx-model-dir", "body-models",
        ])
        self.assertEqual(args.smplerx_model, "s32")
        self.assertEqual(args.smplerx_worker_python.name, "worker-python")
        self.assertEqual(args.smplerx_checkpoint.name, "s32.pth")
        self.assertEqual(args.smplx_model_dir.name, "body-models")

    def test_root_help_works_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-m", "teleoperation", "hand", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("train", "export", "inspect", "align", "control", "vr"):
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

    def test_removed_commands_fail_during_argument_parsing(self):
        for route in (("hand", "realtime"), ("replay", "canonical"), ("replay", "npy")):
            with self.subTest(route=route), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(list(route))
                self.assertEqual(error.exception.code, 2)

    def test_removed_sample_tool_and_normalized_record_route_are_rejected(self):
        for route in (['tools','make-sample-data'], ['hand','record','--landmark-space','normalized']):
            with self.subTest(route=route), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(route)
                self.assertEqual(error.exception.code,2)

    def test_native_routes_require_explicit_target_and_backend(self):
        parser = build_parser()
        for command in ("control", "export"):
            extra = ["--input", "input.h5", "--output", "result.h5"] if command == "export" else []
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["hand", command, *extra])
            args = parser.parse_args(["hand", command, "--hand-model", "l21", "--backend", "nn", *extra])
            self.assertEqual(args.hand_model, "l21")
            self.assertEqual(args.backend, "nn")

if __name__ == "__main__":
    unittest.main()
