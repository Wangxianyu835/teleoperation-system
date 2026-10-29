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
from teleoperation.applications.training import run as train
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
        for value in (None, "", "  ", "unknown_mode", "palm_local_to_l21_v999", 123, [PALM, PALM]):
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

    def test_legacy_checkpoint_four_way_matrix_checks_before_inference(self):
        for expected in (LEGACY,PALM):
            for actual in (LEGACY,PALM):
                self.write_h5(expected)
                self.write_checkpoint(actual)
                model=CountingModel()
                if expected == actual:
                    load_hand_checkpoint(model,str(self.checkpoint),'cpu',expected_coordinate_alignment=expected)
                    model(torch.zeros(1,3,25,3))
                    self.assertEqual(model.forward_calls,1)
                else:
                    with self.assertRaisesRegex(ValueError,'coordinate alignment mismatch'):
                        load_hand_checkpoint(model,str(self.checkpoint),'cpu',expected_coordinate_alignment=expected)
                    self.assertEqual(model.forward_calls,0)

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

    def test_new_training_rejects_legacy_h5_before_creating_run(self):
        for alignment in (LEGACY,PALM):
            self.write_h5(alignment,count=40)
            args=build_parser().parse_args(['hand','train','--input',str(self.input),
                '--run-name','reject','--hand-model','l21','--checkpoint-root',str(self.root/'models'),
                '--epochs','1','--device','cpu'])
            with self.assertRaisesRegex(ValueError,'canonical world'):
                train(args)
            self.assertFalse((self.root/'models').exists())

    def test_new_checkpoint_records_actual_dimensionless_input_and_native_output(self):
        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import joint_checkpoint_payload
        from tests.test_joint_checkpoints import SMALL
        spec=load_hand_spec('g1','left')
        payload=joint_checkpoint_payload(build_joint_model(spec,SMALL),spec)
        self.assertEqual(payload['units'],'1')
        self.assertEqual(payload['output_units'],'rad')
        self.assertEqual(payload['side'],'left')
        self.assertEqual(payload['model_kwargs']['out_num_joint'],7)
        self.assertNotIn('source_scale',payload)


if __name__ == "__main__":
    unittest.main()
