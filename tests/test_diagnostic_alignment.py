"""Useful offline diagnostics obey the same input/checkpoint contract as export."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from retargeting.coordinates import COORDINATE_ALIGNMENT as LEGACY, PALM_LOCAL_COORDINATE_ALIGNMENT as PALM
from retargeting.tracking import ensure_hand25
from scripts import compare_training_hand_pose as comparison
from scripts.verify_p0 import verify
from tests.test_coordinate_contracts import synthetic_hand_pair
from tests.test_coordinate_pipeline import CountingModel


class DiagnosticAlignmentTests(unittest.TestCase):
    def test_pose_comparison_accepts_only_matching_alignment(self):
        sample = {"left_valid": True, "right_valid": False,
                  "left_input": np.zeros((3, 25, 3), dtype=np.float32)}
        fk = mock.Mock()
        fk.forward.return_value = (None, None, torch.zeros((1, 23, 3)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            for expected in (LEGACY, PALM):
                for actual in (LEGACY, PALM):
                    model = CountingModel()
                    torch.save({"model_pos": model.state_dict(), "coordinate_alignment": actual}, checkpoint)
                    with self.subTest(expected=expected, actual=actual), \
                            mock.patch.object(comparison, "build_hand_model", return_value=model):
                        if expected == actual:
                            result = comparison._predict_checkpoint(checkpoint, sample, {"left": fk}, torch.device("cpu"), expected)
                            self.assertEqual(result["left"].shape, (23, 3))
                            self.assertIsNone(result["right"])
                        else:
                            with self.assertRaisesRegex(ValueError, "coordinate alignment mismatch"):
                                comparison._predict_checkpoint(checkpoint, sample, {"left": fk}, torch.device("cpu"), expected)
                            self.assertEqual(model.forward_calls, 0)

    def test_p0_verification_checks_actual_input_alignment_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.h5"
            checkpoint = Path(directory) / "model.pth"
            for expected, actual in ((LEGACY, PALM), (PALM, LEGACY)):
                with h5py.File(source, "w") as handle:
                    handle.attrs.update(coordinate_frame="l21", coordinate_alignment=expected)
                    for side, raw in zip(("left", "right"), synthetic_hand_pair()):
                        handle.create_dataset(f"{side}_hand_keypoints", data=np.repeat(ensure_hand25(raw)[None], 8, axis=0))
                model = CountingModel()
                torch.save({"model_pos": model.state_dict(), "coordinate_alignment": actual}, checkpoint)
                with self.subTest(expected=expected, actual=actual), \
                        mock.patch("retargeting.model.build_hand_model", return_value=model), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "coordinate alignment mismatch"):
                        verify(source, checkpoint)
                self.assertEqual(model.forward_calls, 0)
