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


class DistinctHandModel(RecordingModel):
    """Different inputs produce different bounded angles, including nonzero roll."""

    def forward(self, data):
        self.last_input = data.detach().clone()
        values = torch.full((len(data), 18), .4, device=data.device)
        values[:, [1, 4, 7, 10]] = .06
        values += data[:, -1, 1, 0, None] * 2
        values[:, 0] = 0
        return values


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
        self.assertFalse(result[("l21", "left")].valid)
        self.assertTrue(result[("l21", "right")].valid)
        self.assertEqual(result[("l21", "right")].positions.shape, (17,))
        np.testing.assert_array_equal(model.last_input.numpy()[0], window.hands["left"])
        self.assertEqual(tuple(model.last_input.shape), (1, 3, 25, 3))
        self.assertEqual(result[("l21", "right")].timestamp, window.timestamp)
        missing = pipeline.process_window(observation(3, left=False))
        self.assertFalse(retargeter.retarget(missing)[("l21", "right")].valid)

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

    def test_single_left_target_converts_native_angles_and_keeps_raw_diagnostic(self):
        from teleoperation.retargeting.hand.angles import canonical_to_native_angles
        pipeline=TransHandRVisionProPipeline()
        for i in range(3): window=pipeline.process_window(observation(i))
        retargeter=TransHandRVRRetargeter(DistinctHandModel(),target_side='left')
        results=retargeter.retarget(window)
        self.assertTrue(results[('l21','left')].valid)
        self.assertFalse(results[('l21','right')].valid)
        raw=retargeter.last_raw_angles.copy()
        self.assertNotEqual(raw[1],0.)
        np.testing.assert_array_equal(results[('l21','left')].positions,canonical_to_native_angles(raw,'left')[1:])
        retargeter.reset()
        self.assertIsNone(retargeter.last_raw_angles)

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
        self.assertIn("l21/right", result["hand_commands"]["collision_guard"])


class VRSameSideTests(unittest.TestCase):
    def arguments(self, source, report, *extra):
        from teleoperation.cli import build_parser
        return build_parser().parse_args([
            "hand", "vr", "--input-h5", str(source), "--checkpoint", "unused.pth",
            "--hand", "both", "--hand-mapping", "same", "--headless", "--fps", "0",
            "--report", str(report), *extra])

    def write_recording(self, path):
        with h5py.File(path, "w") as handle:
            handle.attrs["source"] = "visionpro"
            handle["timestamps"] = np.arange(8) / 30
            for side in ("left", "right"):
                handle[side + "_hand_transforms"] = np.stack([transforms(i) for i in range(8)])
                valid = np.ones(8, dtype=bool)
                if side == "left":
                    valid[3] = False
                handle[side + "_valid"] = valid

    def test_dual_actual_scene_exports_independent_raw_native_and_held_commands(self):
        from teleoperation.applications.visionpro_hand import main
        from teleoperation.retargeting.hand.angles import canonical_to_native_angles
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, report = folder / "input.h5", folder / "result.json"
            self.write_recording(source)
            args = self.arguments(source, report)
            with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model", return_value=DistinctHandModel()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            info = json.loads(report.read_text())
            self.assertEqual(info["routes"], {"left": "left", "right": "right"})
            self.assertEqual(info["angle_frames"], 6)  # Frames with at least one output, not hand count.
            self.assertEqual(info["sides"]["left"]["angle_frames"], 3)
            self.assertEqual(info["sides"]["right"]["angle_frames"], 6)
            self.assertEqual(set(info["hand_commands"]["collision_guard"]), {"l21/left", "l21/right"})
            with h5py.File(report.with_suffix(".h5"), "r+") as result:
                np.testing.assert_array_equal(result["left_model_valid"][:],
                                              [False, False, True, False, False, False, True, True])
                np.testing.assert_array_equal(result["right_model_valid"][:],
                                              [False, False, True, True, True, True, True, True])
                self.assertFalse(result["left_input_valid"][3])
                self.assertTrue(result["left_input_valid"][4])  # Visible but still warming up.
                for side in ("left", "right"):
                    valid = result[side + "_model_valid"][:]
                    raw = result[side + "_raw_angles"][:]
                    native = result[side + "_native_angles"][:]
                    np.testing.assert_array_equal(native[valid], canonical_to_native_angles(raw[valid], side))
                    self.assertTrue(np.isnan(raw[~valid]).all())
                    self.assertTrue(np.isfinite(result[side + "_command_angles"][:]).all())
                # Independently compute both routes through the original single-side predictor.
                pipeline = TransHandRVisionProPipeline()
                replay = VisionProReplayInput(source, "both")
                try:
                    for index in range(8):
                        window = pipeline.process_window(replay.next_observation())
                        if window is not None:
                            for side in ("left", "right"):
                                adapter = TransHandRVRRetargeter(DistinctHandModel(), source_side=side, target_side=side)
                                expected = adapter.retarget(window)[("l21", side)]
                                if expected.valid:
                                    np.testing.assert_array_equal(result[side + "_raw_angles"][index], adapter.last_raw_angles)
                finally:
                    replay.close()
                self.assertFalse(np.array_equal(result["left_raw_angles"][2], result["right_raw_angles"][2]))
                held = result["left_command_angles"][2]
                np.testing.assert_array_equal(result["left_command_angles"][3:6], np.tile(held, (3, 1)))

    def test_recording_clock_makes_both_sides_independent_of_playback_rate(self):
        from teleoperation.applications.visionpro_hand import main
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "input.h5"
            self.write_recording(source)
            for fps in (0, 30):
                args = self.arguments(source, folder / f"result{fps}.json", "--fps", str(fps), "--no-collision-guard")
                with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model", return_value=DistinctHandModel()), \
                        patch("teleoperation.applications.visionpro_hand.time.sleep"), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
            with h5py.File(folder / "result0.h5") as fast, h5py.File(folder / "result30.h5") as slow:
                for side in ("left", "right"):
                    for stage in ("raw", "native", "command"):
                        name = f"{side}_{stage}_angles"
                        np.testing.assert_array_equal(fast[name][:], slow[name][:])
                self.assertGreater(np.max(np.abs(np.diff(fast["right_command_angles"][:], axis=0))), 0)
                self.assertLessEqual(np.max(np.abs(np.diff(fast["right_command_angles"][:], axis=0))), .05 + 1e-12)

    def test_invalid_routes_and_output_paths_fail_before_model_or_scene(self):
        from teleoperation.applications.visionpro_hand import main
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, report = folder / "input.h5", folder / "result.json"
            self.write_recording(source)
            variants = (["--hand-mapping", "same", "--hand", "left"],
                        ["--source-hand", "right"], ["--output-h5", str(source)],
                        ["--report", str(source)])
            for options in variants:
                args = self.arguments(source, report, *options)
                with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model") as load, \
                        patch("teleoperation.applications.visionpro_hand.build_scene") as scene, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 1)
                    load.assert_not_called()
                    scene.assert_not_called()
            args = self.arguments(source, report)
            args.hand_mapping = None
            with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model") as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
                load.assert_not_called()
            report.with_suffix(".h5").write_bytes(b"keep existing result")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(self.arguments(source, report)), 1)
            self.assertEqual(report.with_suffix(".h5").read_bytes(), b"keep existing result")

    def test_nonmonotonic_timestamp_failure_closes_input_scene_and_export(self):
        from teleoperation.applications.visionpro_hand import main
        from teleoperation.applications.realtime_hand_sim import RealtimeL21Scene
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, report = folder / "input.h5", folder / "result.json"
            self.write_recording(source)
            with h5py.File(source, "r+") as handle:
                handle["timestamps"][3] = handle["timestamps"][2]
            replay = VisionProReplayInput(source, "both")
            scene = RealtimeL21Scene(("left", "right"), gui=False)
            with patch("teleoperation.applications.visionpro_hand.load_transhandr_vr_model", return_value=DistinctHandModel()), \
                    patch("teleoperation.applications.visionpro_hand.VisionProReplayInput", return_value=replay), \
                    patch("teleoperation.applications.visionpro_hand.build_scene", return_value=scene), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(self.arguments(source, report)), 1)
            self.assertIsNone(replay._file)
            self.assertFalse(scene.alive())
            with h5py.File(report.with_suffix(".h5"), "r+") as result:
                self.assertEqual(len(result["timestamps"]), 3)
