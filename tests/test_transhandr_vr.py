"""Upstream VR tensor semantics and compatibility without a live headset."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline, transhandr_vr_points
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
from teleoperation.retargeting.hand.transhandr_vr import TransHandRVRRetargeter, load_transhandr_vr_model


def transforms(frame=0):
    matrices = np.tile(np.eye(4), (25, 1, 1))
    joints = np.arange(25)
    matrices[:, 0, 3] = 2 + frame * .3 + joints * .003 * (frame + 1)
    matrices[:, 1, 3] = 3 + frame * .2 + joints * .005
    matrices[:, 2, 3] = 4 + frame * .1 + joints * .007
    return matrices


def observation(frame=0, left=True, right=True):
    return RawHandFrame({"left": transforms(frame) if left else None,
                         "right": transforms(frame) if right else None},
                        frame / 30, "visionpro", {}, {"left": "transforms25", "right": "transforms25"})


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.last_input = None

    def forward(self, data):
        self.last_input = data.detach().clone()
        return torch.zeros((len(data), 18), device=data.device) + self.weight


class VRProcessingTests(unittest.TestCase):
    def test_axis_remap_and_wrist_origin_match_known_upstream_coordinates(self):
        joints = np.arange(25)
        left = transhandr_vr_points(transforms(), "left")
        right = transhandr_vr_points(transforms(), "right")
        np.testing.assert_allclose(left, np.stack((joints * .005, -joints * .007, joints * .003), axis=1))
        np.testing.assert_allclose(right, -left)
        # Wrist translation changes, but wrist-relative geometry does not.
        moved = transforms(); moved[:, :3, 3] += [100, -50, 20]
        np.testing.assert_allclose(transhandr_vr_points(moved, "left"), left, atol=1e-12)

    def test_three_frame_stack_transpose_and_scale_keep_chronological_order(self):
        pipeline = TransHandRVisionProPipeline(scale_factor=1.7)
        self.assertIsNone(pipeline.process_window(observation(0)))
        self.assertIsNone(pipeline.process_window(observation(1)))
        window = pipeline.process_window(observation(2))
        # Original producer stacks (joint,time,xyz); original inference transposes
        # that tensor and applies scale before torch's float32 conversion.
        expected = np.stack([transhandr_vr_points(transforms(i), "left") for i in range(3)], axis=1)
        expected = np.expand_dims(expected.transpose(1, 0, 2), 0)
        expected *= 1.7
        np.testing.assert_array_equal(window.hands["left"], expected[0].astype(np.float32))
        self.assertEqual(window.source, "visionpro")
        self.assertEqual(window.timestamp, 2 / 30)
        fourth = pipeline.process_window(observation(3))
        np.testing.assert_array_equal(fourth.hands["left"][0], window.hands["left"][1])

    def test_missing_side_clears_only_its_window_and_needs_three_new_frames(self):
        pipeline = TransHandRVisionProPipeline()
        for index in range(3):
            pipeline.process_window(observation(index))
        window = pipeline.process_window(observation(3, left=False))
        self.assertIsNone(window.hands["left"])
        self.assertIsNotNone(window.hands["right"])
        self.assertIsNone(pipeline.process_window(observation(4)).hands["left"])
        self.assertIsNone(pipeline.process_window(observation(5)).hands["left"])
        self.assertIsNotNone(pipeline.process_window(observation(6)).hands["left"])

    def test_raw_geometry_provenance_and_finite_values_are_checked(self):
        pipeline = TransHandRVisionProPipeline()
        for index in range(3):
            pipeline.process_window(observation(index))
        bad = transforms(); bad[5, 0, 3] = np.nan
        frame = RawHandFrame({"left": bad, "right": None}, source="visionpro",
                             geometry_encoding={"left": "transforms25"})
        self.assertIsNone(pipeline.process_window(frame))
        already = RawHandFrame(observation().hands, source="visionpro",
                               metadata={"coordinate_alignment": "source_to_l21_xyz"},
                               geometry_encoding=observation().geometry_encoding)
        with self.assertRaisesRegex(ValueError, "repeated alignment"):
            pipeline.process_window(already)
        with self.assertRaisesRegex(ValueError, "source='visionpro'"):
            pipeline.process_window(RawHandFrame({"left": None, "right": None}, source="mediapipe_approx"))


class VRInterfaceTests(unittest.TestCase):
    def test_existing_single_side_input_and_optional_both_side_input(self):
        stream = Mock()
        stream.get_latest.return_value = {"left_fingers": transforms(), "right_fingers": transforms(1)}
        source = VisionProInput("unused", streamer=stream)
        frame = source.next_observation()
        self.assertEqual(frame.metadata["side"], "left")
        self.assertIsNone(frame.hands["right"])
        np.testing.assert_array_equal(frame.hands["left"], transforms())
        both = VisionProInput("unused", "both", streamer=stream).next_observation()
        self.assertEqual(both.geometry_encoding, {"left": "transforms25", "right": "transforms25"})
        np.testing.assert_array_equal(both.hands["right"], transforms(1))
        source.close()
        stream.close.assert_called_once()

    def test_retargeter_keeps_upstream_source_to_destination_routing(self):
        pipeline = TransHandRVisionProPipeline()
        for index in range(3):
            window = pipeline.process_window(observation(index))
        model = RecordingModel()
        retargeter = TransHandRVRRetargeter(model)
        result = retargeter.retarget(window)
        self.assertFalse(result.left_valid)
        self.assertTrue(result.right_valid)
        self.assertEqual(result.right_angles.values.shape, (18,))
        np.testing.assert_array_equal(model.last_input.numpy()[0], window.hands["left"])
        self.assertEqual(tuple(model.last_input.shape), (1, 3, 25, 3))
        self.assertEqual(result.timestamp, window.timestamp)
        missing = pipeline.process_window(observation(3, left=False))
        self.assertFalse(retargeter.retarget(missing).right_valid)

    def test_strict_upstream_loader_rejects_camera_contract_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"model_pos": RecordingModel().state_dict()}, path)
            with patch("teleoperation.retargeting.hand.transhandr_vr.PoseTransformer", side_effect=lambda **_: RecordingModel()):
                self.assertIsInstance(load_transhandr_vr_model(path), RecordingModel)
                torch.save({"model_pos": {}}, path)
                with self.assertRaises(RuntimeError):
                    load_transhandr_vr_model(path)
                torch.save({"model_pos": {}, "coordinate_alignment": "mediapipe_world_wrist_index_middle_canonical_right_v1"}, path)
                with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                    load_transhandr_vr_model(path)

    def test_recording_replay_preserves_raw_transforms_validity_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.h5"
            self.write_recording(path, valid=[True, False, True, True])
            source = VisionProReplayInput(path)
            np.testing.assert_array_equal(source.next_observation().hands["left"], transforms(0))
            self.assertIsNone(source.next_observation().hands["left"])
            source.next_observation(); source.next_observation()
            with self.assertRaises(StopIteration):
                source.next_observation()
            source.close()
            with self.assertRaises(RuntimeError):
                source.next_observation()

    @staticmethod
    def write_recording(path, valid=None):
        with h5py.File(path, "w") as handle:
            handle.attrs["source"] = "visionpro"
            handle.create_dataset("timestamps", data=np.arange(4) / 30)
            for side in ("left", "right"):
                handle.create_dataset(side + "_hand_transforms", data=np.stack([transforms(i) for i in range(4)]))
                handle.create_dataset(side + "_valid", data=np.ones(4, dtype=bool) if valid is None else valid)

    def test_registered_cli_replay_runs_through_actual_l21_scene(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.visionpro_hand import main
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, report = folder / "raw.h5", folder / "report.json"
            self.write_recording(source)
            args = build_parser().parse_args(["hand", "vr", "--input-h5", str(source),
                "--checkpoint", str(folder / "checkpoint.pth"), "--headless", "--fps", "0", "--report", str(report)])
            with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model", return_value=RecordingModel()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["processed_frames"], 4)
            self.assertEqual(result["angle_frames"], 2)
            self.assertEqual(result["source_hand"], "left")
            self.assertEqual(result["target_hand"], "right")
            self.assertIn("right", result["hand_commands"]["collision_guard"])
