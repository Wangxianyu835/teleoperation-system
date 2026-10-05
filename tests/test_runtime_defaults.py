import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import h5py

from retargeting.__main__ import build_parser
from retargeting.config import PROJECT_ROOT
from retargeting.data import load_twohand_h5
from retargeting.inference import run
from retargeting.inspect import inspect_angle_h5


EXPECTED_INPUT = PROJECT_ROOT / "input/aligned_visual_hand_data_20260912_153542.h5"
EXPECTED_CHECKPOINT = PROJECT_ROOT / "checkpoint/models/twohand_h5/linker/my_run/model_best.pth"


class RuntimeDefaultTests(unittest.TestCase):
    def test_export_defaults_to_repo_based_aligned_input(self):
        args = build_parser().parse_args(["export"])
        self.assertEqual(args.input, EXPECTED_INPUT)
        self.assertEqual(args.input.relative_to(PROJECT_ROOT).parts[0], "input")

    def test_export_defaults_to_existing_retained_run_name(self):
        args = build_parser().parse_args(["export"])
        self.assertEqual(args.checkpoint, EXPECTED_CHECKPOINT)
        self.assertEqual(args.checkpoint.relative_to(PROJECT_ROOT).parts[0], "checkpoint")

    def test_explicit_input_and_checkpoint_still_override_defaults(self):
        args = build_parser().parse_args(["export", "--input", "custom.h5", "--checkpoint", "custom.pth"])
        self.assertEqual(args.input, Path("custom.h5"))
        self.assertEqual(args.checkpoint, Path("custom.pth"))

    def test_default_export_schema_with_local_recording_and_stub_model(self):
        # Local recordings/weights are ignored by Git. Portable clones can run
        # the path-contract tests without these optional verification artifacts.
        if not EXPECTED_INPUT.is_file() or not EXPECTED_CHECKPOINT.is_file():
            self.skipTest("Local retained recording/checkpoint not present")
        ids, _, _ = load_twohand_h5(EXPECTED_INPUT)

        class StubModel(torch.nn.Module):
            def forward(self, points):
                return torch.zeros((points.shape[0], 18), device=points.device)

        from retargeting.model import TwoHandRetargeter

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "angles.h5"
            args = build_parser().parse_args(["export", "--device", "cpu", "--output", str(output)])
            with mock.patch("retargeting.inference.create_twohand_retargeter", return_value=TwoHandRetargeter(StubModel())) as create:
                self.assertEqual(run(args), 0)
            self.assertEqual(create.call_args.kwargs["checkpoint_path"], str(EXPECTED_CHECKPOINT))
            self.assertEqual(create.call_args.kwargs["expected_coordinate_alignment"], "source_to_l21_xyz")
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["coordinate_alignment"], "source_to_l21_xyz")
                self.assertEqual(handle.attrs["coordinate_frame"], "l21")
            summary = inspect_angle_h5(output)
        self.assertEqual(summary["frames"], len(ids))
        for side in ("left", "right"):
            self.assertEqual(summary[side]["shape"], (len(ids), 18))
            self.assertGreater(summary[side]["valid"], 0)
            self.assertEqual(summary[side]["nonfinite"], 0)
            self.assertEqual(summary[side]["out_of_limits"], 0)


if __name__ == "__main__":
    unittest.main()
