"""Coordinate contract propagation and fail-closed training/export regressions."""
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
from teleoperation.applications.training import run as train
from teleoperation.retargeting.hand.topology import ensure_hand25
from tests.test_mediapipe_realtime import synthetic_hand_pair


class CountingModel(torch.nn.Module):
    """Tiny compatible model; tests observe whether inference started."""
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.forward_calls = 0

    def forward(self, points):
        self.forward_calls += 1
        return torch.zeros((len(points), 17), device=points.device) + self.value


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

    def test_current_normalized_h5_contract_decodes_str_and_bytes(self):
        for alignment in (PALM,):
            for declaration in (alignment, np.bytes_(alignment)):
                with self.subTest(alignment=alignment, declaration=declaration):
                    self.write_h5(declaration, frame=np.bytes_("l21"))
                    self.assertEqual(read_coordinate_alignment(self.input), alignment)
                    with self.assertRaisesRegex(ValueError,"canonical world"):
                        TwoHandH5Dataset(self.input, track_identity=False)

    def test_recordings_require_root_datasets_and_explicit_time_vectors(self):
        for missing in ('frame_ids','timestamps','left_hand_keypoints','right_hand_keypoints'):
            self.write_h5()
            with h5py.File(self.input,'r+') as source:
                del source[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                load_twohand_h5(self.input)
        self.write_h5()
        with h5py.File(self.input,'r+') as source:
            for side in ('left','right'):
                points=source[side+'_hand_keypoints'][:]
                del source[side+'_hand_keypoints']
                source.create_group(side)['hand_keypoints']=points
        with self.assertRaisesRegex(ValueError,'root datasets'):
            load_twohand_h5(self.input)

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


    def test_native_checkpoint_rejects_old_schema_and_bad_input_before_build(self):
        from teleoperation.retargeting.hand.checkpoint import (
            shared_joint_checkpoint_payload, save_joint_checkpoint, load_shared_joint_checkpoint,
        )
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.robots.hand_spec import load_hand_spec
        from tests.test_joint_checkpoints import SMALL
        spec = load_hand_spec('l21', 'right')
        payload = shared_joint_checkpoint_payload(build_joint_model(spec, SMALL), spec)
        mutations = [{'model_pos': {}}, {**payload, 'units': 'm'},
                     {**payload, 'coordinate_alignment': PALM},
                     {**payload, 'temporal_window': 4},
                     {**payload, 'source_landmark_space': 'mediapipe_normalized'},
                     {**payload, 'spec_hash': 'incorrect'},
                     {**payload, 'joint_names': list(reversed(spec.independent_joint_names))}]
        for bad in mutations:
            save_joint_checkpoint(self.checkpoint, bad)
            with self.subTest(keys=set(bad)), mock.patch(
                    'teleoperation.retargeting.hand.predictor.build_joint_model') as build:
                with self.assertRaises(ValueError):
                    load_shared_joint_checkpoint(self.checkpoint, spec, 'cpu')
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
