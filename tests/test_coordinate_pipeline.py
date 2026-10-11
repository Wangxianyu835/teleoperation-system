"""Coordinate contract propagation and fail-closed training/export regressions."""
import contextlib
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from teleoperation.cli import build_parser
from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT as LEGACY, PALM_LOCAL_COORDINATE_ALIGNMENT as PALM
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.data.hand_h5 import load_twohand_h5, read_coordinate_alignment
from teleoperation.applications.hand_export import run as export
from teleoperation.retargeting.hand.predictor import create_twohand_retargeter, load_hand_checkpoint
from teleoperation.learning.trainer import LOSS_NAMES
from teleoperation.applications.training import _load_checkpoint, _save_checkpoint, run as train
from teleoperation.retargeting.hand.topology import ensure_hand25
from tests.test_coordinate_contracts import synthetic_hand_pair


class CountingModel(torch.nn.Module):
    """Tiny compatible model; tests observe whether inference started."""
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward(self, points):
        self.forward_calls += 1
        return torch.zeros((len(points), 18), device=points.device) + self.value


class CoordinatePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "input.h5"
        self.checkpoint = self.root / "checkpoint.pth"
        self.output = self.root / "angles.h5"

    def write_h5(self, alignment=PALM, frame="l21", count=8):
        with h5py.File(self.input, "w") as handle:
            if frame is not None:
                handle.attrs["coordinate_frame"] = frame
            if alignment is not None:
                handle.attrs["coordinate_alignment"] = alignment
            handle.attrs["source_landmark_space"] = "mediapipe_normalized"
            handle.create_dataset("frame_ids", data=np.arange(count))
            handle.create_dataset("timestamps", data=np.arange(count) / 30)
            for side, raw in zip(("left", "right"), synthetic_hand_pair()):
                handle.create_dataset(f"{side}_hand_keypoints", data=np.repeat(ensure_hand25(raw)[None], count, axis=0))

    def write_checkpoint(self, alignment=PALM):
        checkpoint = {"model_pos": CountingModel().state_dict()}
        if alignment is not None:
            checkpoint["coordinate_alignment"] = alignment
        torch.save(checkpoint, self.checkpoint)

    def export_args(self):
        return build_parser().parse_args([
            "hand", "export", "--input", str(self.input), "--checkpoint", str(self.checkpoint),
            "--output", str(self.output), "--device", "cpu", "--disable-identity-tracking",
        ])

    def test_both_h5_contracts_decode_str_and_bytes_and_are_exposed(self):
        for alignment in (LEGACY, PALM):
            for declaration in (alignment, np.bytes_(alignment)):
                with self.subTest(alignment=alignment, declaration=declaration):
                    self.write_h5(declaration, frame=np.bytes_("l21"))
                    self.assertEqual(read_coordinate_alignment(self.input), alignment)
                    dataset = TwoHandH5Dataset(self.input, track_identity=False)
                    self.assertEqual(dataset.coordinate_alignment, alignment)
                    self.assertEqual(dataset.coordinate_frame, "l21")
                    self.assertEqual(dataset.source_landmark_space, "mediapipe_normalized")

    def test_h5_missing_empty_unknown_and_nonscalar_alignment_fail_closed(self):
        for value in (None, "", "  ", "unknown_mode", "palm_local_to_l21_v2", 123, [PALM, PALM]):
            with self.subTest(value=value):
                self.write_h5(value)
                for reader in (read_coordinate_alignment, TwoHandH5Dataset, load_twohand_h5):
                    with self.assertRaisesRegex(ValueError, "coordinate_alignment"):
                        reader(self.input)
                # Raw inspection and the existing preprocessing path still work.
                self.assertEqual(load_twohand_h5(self.input, require_aligned=False)[0].size, 8)

    def test_wrong_missing_and_nonscalar_coordinate_frame_fail_closed(self):
        for frame in (None, "source", "", 123, ["l21", "l21"]):
            with self.subTest(frame=frame):
                self.write_h5(frame=frame)
                with self.assertRaisesRegex(ValueError, "coordinate_frame"):
                    read_coordinate_alignment(self.input)

    def test_export_four_way_matrix_checks_before_inference_and_preserves_metadata(self):
        for input_alignment, checkpoint_alignment, accepted in (
            (LEGACY, LEGACY, True), (PALM, PALM, True),
            (PALM, LEGACY, False), (LEGACY, PALM, False),
        ):
            with self.subTest(input=input_alignment, checkpoint=checkpoint_alignment):
                self.write_h5(input_alignment)
                self.write_checkpoint(checkpoint_alignment)
                self.output.write_bytes(b"existing output must survive mismatch")
                model = CountingModel()
                with mock.patch("teleoperation.retargeting.hand.predictor.build_hand_model", return_value=model):
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        if accepted:
                            self.assertEqual(export(self.export_args()), 0)
                        else:
                            with self.assertRaisesRegex(ValueError, "coordinate alignment mismatch") as error:
                                export(self.export_args())
                if accepted:
                    self.assertGreater(model.forward_calls, 0)
                    self.assertIn(f"coordinate_alignment={input_alignment}", stdout.getvalue())
                    with h5py.File(self.output, "r") as handle:
                        self.assertEqual(handle.attrs["coordinate_alignment"], input_alignment)
                        self.assertEqual(handle.attrs["coordinate_frame"], "l21")
                        self.assertEqual(handle.attrs["source_landmark_space"], "mediapipe_normalized")
                        self.assertEqual(handle["left_angles"].shape, (8, 18))
                        self.assertEqual(handle["left_valid"][:].sum(), 6)
                else:
                    self.assertEqual(model.forward_calls, 0)
                    self.assertEqual(self.output.read_bytes(), b"existing output must survive mismatch")
                    self.assertIn(input_alignment, str(error.exception))
                    self.assertIn(checkpoint_alignment, str(error.exception))

    def test_missing_or_unknown_checkpoint_alignment_stops_before_state_loading(self):
        for value in (None, "", "unknown_mode", [PALM, PALM]):
            self.write_checkpoint(value)
            model = CountingModel()
            with mock.patch.object(model, "load_state_dict") as load:
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "coordinate_alignment"):
                    load_hand_checkpoint(model, str(self.checkpoint), "cpu", expected_coordinate_alignment=PALM)
                load.assert_not_called()

    def test_bad_expected_alignment_is_rejected_instead_of_disabling_guard(self):
        self.write_checkpoint(PALM)
        for expected in ("", "unknown_mode"):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "coordinate_alignment"):
                load_hand_checkpoint(CountingModel(), str(self.checkpoint), "cpu", expected_coordinate_alignment=expected)

    def test_old_checkpoint_calls_still_require_legacy(self):
        self.write_checkpoint(LEGACY)
        model = CountingModel()
        self.assertIs(load_hand_checkpoint(model, str(self.checkpoint), "cpu"), model)
        self.write_checkpoint(PALM)
        with self.assertRaisesRegex(ValueError, "coordinate alignment mismatch"):
            load_hand_checkpoint(model, str(self.checkpoint), "cpu")

    def test_shared_and_side_checkpoint_factory_paths_forward_expected_alignment(self):
        self.write_checkpoint(PALM)
        selections = (
            {"checkpoint_path": str(self.checkpoint)},
            {"left_checkpoint": str(self.checkpoint), "right_checkpoint": str(self.checkpoint)},
        )
        for selection in selections:
            with self.subTest(selection=selection):
                with mock.patch("teleoperation.retargeting.hand.predictor.build_hand_model", side_effect=lambda _: CountingModel()):
                    retargeter = create_twohand_retargeter({}, "cpu", **selection, expected_coordinate_alignment=PALM)
                self.assertEqual(retargeter.coordinate_alignment, PALM)
                for side in ("left", "right"):
                    model = retargeter.model_for_side(side)
                    self.assertIsInstance(model, CountingModel)
                    self.assertEqual(model.coordinate_alignment, PALM)

    def test_incomplete_checkpoint_pair_is_rejected_before_building_model(self):
        for keyword in ("left_checkpoint", "right_checkpoint"):
            with self.subTest(keyword=keyword), mock.patch("teleoperation.retargeting.hand.predictor.build_hand_model") as build:
                with self.assertRaisesRegex(ValueError, "required together"):
                    create_twohand_retargeter({}, "cpu", **{keyword: str(self.checkpoint)}, expected_coordinate_alignment=PALM)
                build.assert_not_called()

    def test_warmstart_four_way_matrix_matches_training_input(self):
        for expected in (LEGACY, PALM):
            for actual in (LEGACY, PALM):
                self.write_checkpoint(actual)
                with self.subTest(expected=expected, checkpoint=actual):
                    if expected == actual:
                        _load_checkpoint(CountingModel(), self.checkpoint, "cpu", expected)
                    else:
                        with self.assertRaisesRegex(ValueError, "coordinate alignment mismatch"):
                            _load_checkpoint(CountingModel(), self.checkpoint, "cpu", expected)

    def test_checkpoint_save_uses_explicit_contract_and_records_source_space(self):
        model = CountingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for alignment in (LEGACY, PALM):
            with self.subTest(alignment=alignment):
                _save_checkpoint(
                    self.checkpoint, model, optimizer, scheduler,
                    epoch=1, best_val=0., h5_path=self.input, train_end=6, val_start=6,
                    best_epoch=1, run_name="test", init_checkpoint=None,
                    coordinate_alignment=alignment, source_landmark_space="mediapipe_normalized",
                )
                saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(saved["coordinate_alignment"], alignment)
                self.assertEqual(saved["coordinate_frame"], "l21")
                self.assertEqual(saved["source_landmark_space"], "mediapipe_normalized")
                self.assertIsNone(saved["init_checkpoint"])

    def test_training_path_inherits_h5_contract_and_logs_it(self):
        # This covers the actual run/save/load/export plumbing without a long
        # neural-network optimization test. Mathematical losses are tested elsewhere.
        for alignment in (LEGACY, PALM):
            with self.subTest(alignment=alignment):
                self.write_h5(alignment, count=40)
                run_name = "test_" + alignment
                args = build_parser().parse_args([
                    "hand", "train", "--input", str(self.input), "--run-name", run_name,
                    "--checkpoint-root", str(self.root / "runs"), "--epochs", "1", "--device", "cpu",
                ])
                stats = dict(total=1., vec=.1, pos=.1, collision=.1, thumb=.1, tip_distance=.1, thumb2=.1,
                             left_total=1., right_total=1., left_valid=1, right_valid=1, gradient_norm=0.)
                for side in ("left", "right"):
                    for name in LOSS_NAMES:
                        stats[f"{side}_{name}"] = stats[name]
                with mock.patch("teleoperation.applications.training._create_pose_model", return_value=CountingModel()), \
                     mock.patch("teleoperation.applications.training._create_hand_fks", return_value={}), \
                     mock.patch("teleoperation.applications.training._create_collision_loss", return_value=None), \
                     mock.patch("teleoperation.applications.training._run_epoch", return_value=(stats, 0)), \
                     mock.patch("teleoperation.applications.training.SummaryWriter"), \
                     contextlib.redirect_stdout(io.StringIO()) as stdout:
                    try:
                        self.assertEqual(train(args), 0)
                    finally:
                        # Windows cannot remove a directory with an open log.
                        logger = logging.getLogger("teleoperation.learning.trainer")
                        for handler in list(logger.handlers):
                            handler.close()
                            logger.removeHandler(handler)
                self.assertIn(f"coordinate_alignment={alignment}", stdout.getvalue())
                saved_path = self.root / "runs/models/twohand_h5/linker" / run_name / "model_best.pth"
                saved = torch.load(saved_path, map_location="cpu", weights_only=True)
                self.assertEqual(saved["coordinate_alignment"], alignment)
                log = self.root / "runs/logs/twohand_h5/linker" / run_name / "training.log"
                self.assertIn(f"coordinate_alignment={alignment}", log.read_text(encoding="utf-8"))
                self.checkpoint = saved_path
                with mock.patch("teleoperation.retargeting.hand.predictor.build_hand_model", return_value=CountingModel()), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(export(self.export_args()), 0)
                with h5py.File(self.output, "r") as handle:
                    self.assertEqual(handle.attrs["coordinate_alignment"], alignment)


if __name__ == "__main__":
    unittest.main()
