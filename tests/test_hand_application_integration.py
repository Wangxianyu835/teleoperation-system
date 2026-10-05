"""Canonical hand data reaches existing destination consumers without PR2 redesign."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import retarget_io as application
from input_adapters.npy_replay_adapter import NpyReplayAdapter
from retargeting import contracts
from retargeting.config import HAND_ANGLE_DIM, HAND_DOF, L21, PROJECT_ROOT
from retargeting.hand_core import CanonicalHandProcessor
from retargeting.simulation import angle18_to_dofs
from retargeting.visionpro import VisionProAdapter
from teleop.native_hand import DIM_NAMES, NATIVE_MAP, PREFIX, build_mapping, map_frame
from tests.test_coordinate_contracts import synthetic_hand_pair


class HandApplicationIntegrationTests(unittest.TestCase):
    def test_application_hand_validation_delegates_to_canonical_contract(self):
        self.assertIs(application._validate_hand_input, contracts.validate_hand_input)
        self.assertIs(application.legacy_visionpro_to_window, contracts.legacy_visionpro_to_window)
        self.assertEqual(application.HAND_KEYPOINTS, 25)
        self.assertEqual(application.RECEPTIVE_FIELD, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            application.validate_hand_input(np.zeros((3, 21, 3)), "left")
        with self.assertRaisesRegex(ValueError, "NaN"):
            application.validate_hand_input(np.full((3, 25, 3), np.nan), "right")

    def test_npy_replay_uses_canonical_processor_and_resets_missing_side(self):
        points = synthetic_hand_pair()[0]
        frames = [{"left_hand": points if index != 3 else None, "right_hand": None,
                   "timestamp": index / 30., "frame_id": index} for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npy"
            np.save(path, np.asarray(frames, dtype=object))
            adapter = NpyReplayAdapter(path)
            self.assertIsInstance(adapter._processor, CanonicalHandProcessor)
            payloads = list(adapter)
        self.assertEqual([item["metadata"]["frame_id"] for item in payloads], [2, 6])
        for payload in payloads:
            self.assertTrue(application.validate_retarget_input(payload))
            self.assertEqual(payload["hands"]["left"].shape, (3, 25, 3))
            self.assertIsNone(payload["hands"]["right"])
            np.testing.assert_array_equal(payload["hands"]["left"][:, 0], 0)

    def test_destination_visionpro_adapter_still_emits_three_frame_windows(self):
        transforms = np.repeat(np.eye(4)[None], 25, axis=0)
        transforms[:, :3, 3] = np.arange(75).reshape(25, 3) / 100.

        class Stream:
            def get_latest(self):
                return {"left_fingers": transforms}

        adapter = VisionProAdapter("unused", streamer=Stream())
        self.assertIsNone(adapter.next_input())
        self.assertIsNone(adapter.next_input())
        payload = adapter.next_input()
        self.assertEqual(payload["source"], "visionpro")
        self.assertEqual(payload["hands"]["left"].shape, (3, 25, 3))
        self.assertIsNone(payload["hands"]["right"])

    def test_existing_native_hand_mapping_consumes_original_18d_indices(self):
        angles = np.linspace(0, 0.17, HAND_ANGLE_DIM, dtype=np.float32)
        self.assertEqual(angle18_to_dofs(angles).shape, (HAND_DOF,))
        np.testing.assert_array_equal(angle18_to_dofs(angles), angles[1:])
        for robot, count in (("h1_2", 12), ("gr1_t2", 11), ("g1", 7)):
            for side in ("left", "right"):
                prefix = PREFIX[robot][side]
                ranges = {template.format(P=prefix): (0., 2.) for template in NATIVE_MAP[robot].values()}
                mapping = build_mapping(robot, side, ranges)
                values, clipped = map_frame(angles, mapping)
                self.assertEqual(len(values), count)
                self.assertEqual(clipped, 0)
                for index, name, *_ in mapping:
                    self.assertEqual(values[name], float(angles[index]))
                    self.assertEqual(DIM_NAMES[index], L21.hand_kinematics_config()["joints_name"][index])

    def test_destination_checkpoint_default_and_application_packing_are_retained(self):
        from retargeting.cli import build_parser
        args = build_parser().parse_args(["export"])
        self.assertEqual(args.checkpoint, PROJECT_ROOT / "checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth")
        self.assertEqual(application.ACTION_ORDER, contracts.ACTION_ORDER)
        # Existing app helper accepts partial commands; fixed destination helper
        # still rejects absent arms. PR1 deliberately preserves both PR2 concerns.
        hand = np.zeros(17, dtype=np.float32)
        self.assertEqual(application.build_action(None, hand, None, hand).shape, (34,))
        with self.assertRaisesRegex(ValueError, "left_arm"):
            contracts.build_action(None, hand, None, hand)
