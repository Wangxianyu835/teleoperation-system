"""Canonical hand data reaches existing destination consumers without PR2 redesign."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from teleoperation.contracts import validation as application
from teleoperation.apps.hand_processing import NpyReplayWorkflow
from teleoperation.contracts import validation as contracts
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM, HAND_DOF, L21
from teleoperation.paths import PROJECT_ROOT
from teleoperation.apps.hand_processing import CanonicalHandPipeline
from teleoperation.retargeting.hand.angles import angle18_to_dofs
from teleoperation.apps.visionpro import VisionProWorkflow
from teleoperation.robots.native_hand import DIM_NAMES, NATIVE_MAP, PREFIX, build_mapping
from teleoperation.apps.replay.hand_support import map_frame
from tests.test_coordinate_contracts import synthetic_hand_pair


class HandApplicationIntegrationTests(unittest.TestCase):
    def test_application_uses_observation_contract_validation(self):
        self.assertEqual(contracts.HAND_KEYPOINTS, 25)
        with self.assertRaisesRegex(ValueError, "shape"):
            contracts.validate_hand_input(np.zeros((3, 21, 3)), "left")
        with self.assertRaisesRegex(ValueError, "NaN"):
            contracts.validate_hand_input(np.full((3, 25, 3), np.nan), "right")

    def test_npy_replay_uses_canonical_processor_and_resets_missing_side(self):
        points = synthetic_hand_pair()[0]
        frames = [{"left_hand": points if index != 3 else None, "right_hand": None,
                   "timestamp": index / 30., "frame_id": index} for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npy"
            np.save(path, np.asarray(frames, dtype=object))
            adapter = NpyReplayWorkflow(path)
            self.assertIsInstance(adapter.pipeline, CanonicalHandPipeline)
            payloads = list(adapter)
        self.assertEqual([item["metadata"]["frame_id"] for item in payloads], [2, 6])
        for payload in payloads:
            self.assertTrue(contracts.validate_retarget_input(payload))
            self.assertEqual(payload["hands"]["left"].shape, (3, 25, 3))
            self.assertIsNone(payload["hands"]["right"])
            np.testing.assert_array_equal(payload["hands"]["left"][:, 0], 0)

    def test_destination_visionpro_adapter_still_emits_three_frame_windows(self):
        transforms = np.repeat(np.eye(4)[None], 25, axis=0)
        transforms[:, :3, 3] = np.arange(75).reshape(25, 3) / 100.

        class Stream:
            def get_latest(self):
                return {"left_fingers": transforms}

        adapter = VisionProWorkflow("unused", streamer=Stream())
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
        from teleoperation.cli import build_parser
        args = build_parser().parse_args(["hand", "export"])
        self.assertEqual(args.checkpoint, PROJECT_ROOT / "checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth")
        self.assertEqual(contracts.ACTION_ORDER, ("left_arm", "left_hand", "right_arm", "right_hand"))
        # Existing app helper accepts partial commands; fixed destination helper
        # still rejects absent arms. PR1 deliberately preserves both PR2 concerns.
        hand = np.zeros(17, dtype=np.float32)
        self.assertEqual(pack_partial_limb_vector(None, hand, None, hand).values.shape, (34,))
        with self.assertRaisesRegex(ValueError, "left_arm"):
            pack_tron2a_l21_action(None, hand, None, hand)

from teleoperation.robots.partial import pack_partial_limb_vector
from teleoperation.robots.action_mapping import pack_tron2a_l21_action
