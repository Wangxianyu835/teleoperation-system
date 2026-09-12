import tempfile
import unittest
from pathlib import Path
from unittest import mock

from retargeting.__main__ import build_parser
from retargeting.config import DEFAULT_CHECKPOINT
from retargeting.inference import _resolve_device
from retargeting.inspect import inspect_angle_h5


class CliTests(unittest.TestCase):
    def test_export_uses_the_retained_checkpoint_by_default(self):
        args = build_parser().parse_args(["export"])
        self.assertEqual(args.checkpoint, DEFAULT_CHECKPOINT)

    def test_all_public_commands_are_registered(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["export"]).command, "export")
        self.assertEqual(
            parser.parse_args(
                ["train", "--input", "input.h5", "--run-name", "test"]
            ).command,
            "train",
        )
        self.assertEqual(
            parser.parse_args(["inspect", "--angle-h5", "angles.h5"]).command,
            "inspect",
        )

    def test_cpu_and_unavailable_cuda_device_selection(self):
        self.assertEqual(_resolve_device("cpu").type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(_resolve_device("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                _resolve_device("cuda")

    def test_inspect_reports_a_missing_file_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.h5"
            with self.assertRaisesRegex(FileNotFoundError, str(path.name)):
                inspect_angle_h5(path)


if __name__ == "__main__":
    unittest.main()
