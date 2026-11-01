"""Metre-preserving world hand pipeline, without cameras or long training."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

from teleoperation.applications.hand_align import align_h5
from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.applications.hand_realtime import MediaPipeCameraWorkflow
from teleoperation.cli import build_parser, main
from teleoperation.contracts.coordinates import (
    PALM_LOCAL_COORDINATE_ALIGNMENT, WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT,
    WORLD_LANDMARK_SPACE, WORLD_PALM_LOCAL_METADATA, palm_local_metadata,
)
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.data.hand_recording import RawHandH5Writer
from teleoperation.inputs.mediapipe import parse_result
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.retargeting.hand.config import L21
from tests.test_mediapipe_realtime import raw_frame

SIDES = ("left", "right")


def world_frame(index, left=True, right=True):
    frame = raw_frame(index, left, right)
    # Synthetic lengths in metres, with a hand-centred origin. Preview points
    # deliberately differ so accidental use of normalized data is detectable.
    for side in SIDES:
        points = frame[side]
        frame["world_" + side] = None if points is None else (points - points.mean(0)).astype(np.float32)
        frame[side] = None if points is None else (points * 4 + .4).astype(np.float32)
    frame["timestamp"] = 1_700_000_000_000 + index * 33
    frame["metadata"].update({
        "source_landmark_space": "mediapipe_normalized",
        "world_landmark_space": WORLD_LANDMARK_SPACE,
        "world_length_unit": "m", "world_origin": "hand_geometric_center",
    })
    return frame


def observation(frame):
    return RawHandFrame({side: frame[side] for side in SIDES}, frame["timestamp"],
                        "mediapipe_approx", frame["metadata"],
                        world_hands={side: frame["world_" + side] for side in SIDES})


def save_raw(path, frames, mode="world"):
    with RawHandH5Writer(path, landmark_space=mode) as writer:
        for frame in frames:
            writer.append({side: frame[side] for side in SIDES}, frame["timestamp"] / 1000,
                          world_hands={side: frame["world_" + side] for side in SIDES})
        writer.finish("frame_limit")


def small_model_kwargs():
    kwargs = L21.model_kwargs()
    kwargs.update(embed_dim_ratio=8, spatial_depth=1, temporal_depth=1,
                  num_heads=2, drop_path_rate=0.)
    return kwargs


class WorldHandPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)


    def test_canonical_world_processing_never_loads_robot_reference(self):
        with patch("teleoperation.retargeting.hand.processing.build_l21_reference_basis",
                   side_effect=AssertionError("human normalization must be robot-independent")):
            pipeline = MediaPipePalmLocalPipeline("world")
            for index in range(3): window = pipeline.process_window(world_frame(index))
            self.assertIsNotNone(window)


    def test_single_detection_keeps_both_spaces_and_labels_in_either_order(self):
        for labels in (("Left", "Right"), ("Right", "Left")):
            expected = world_frame(0)
            detector_result = NS(
                handedness=[[NS(category_name=label)] for label in labels],
                hand_landmarks=[[NS(x=x, y=y, z=z) for x, y, z in expected[label.lower()]] for label in labels],
                hand_world_landmarks=[[NS(x=x, y=y, z=z) for x, y, z in expected["world_" + label.lower()]] for label in labels],
            )
            parsed = parse_result(detector_result, expected["timestamp"], include_world=True)
            path = self.root / (labels[0] + ".h5")
            save_raw(path, [parsed])
            with h5py.File(path, "r") as source:
                self.assertEqual(source.attrs["hand_processing_landmark_space"], WORLD_LANDMARK_SPACE)
                self.assertEqual(source.attrs["source_landmark_space"], "mediapipe_normalized")
                self.assertEqual(source.attrs["timestamp_unit"], "unix_seconds")
                for side in SIDES:
                    np.testing.assert_array_equal(source[side + "_hand_keypoints"][0], expected[side])
                    np.testing.assert_array_equal(source[side + "_world_landmarks"][0], expected["world_" + side])
                    self.assertEqual(source[side + "_world_landmarks"].attrs["length_unit"], "m")

    def test_invalid_world_keeps_normalized_preview_and_other_hand(self):
        frame = world_frame(0)
        from tests.test_mediapipe_input import result
        detected = result(["Left", "Right"],
                          [[NS(x=x, y=y, z=z) for x, y, z in frame[side]] for side in SIDES],
                          [[NS(x=x, y=y, z=z) for x, y, z in frame["world_" + side]] for side in SIDES])
        detected.hand_world_landmarks[0][5].z = np.nan
        parsed = parse_result(detected, frame["timestamp"], include_world=True, invalid_hand_as_missing=True)
        self.assertIsNotNone(parsed["left"])
        self.assertIsNone(parsed["world_left"])
        self.assertIsNotNone(parsed["world_right"])
        path = self.root / "invalid.h5"
        save_raw(path, [parsed])
        with h5py.File(path, "r") as source:
            self.assertTrue(np.any(source["left_hand_keypoints"][0]))
            np.testing.assert_array_equal(source["left_world_landmarks"][0], np.zeros((21, 3)))
        detected.hand_world_landmarks = None
        with self.assertRaisesRegex(ValueError, "world landmarks"):
            parse_result(detected, 0, include_world=True, invalid_hand_as_missing=True)

    def test_record_single_side_dual_storage_auto_alignment_and_protection(self):
        from teleoperation.applications import offline_h5_record
        for mode in ("world",):
            frames = [world_frame(i) for i in range(4)]
            capture = Mock()
            capture.next_frame.side_effect = frames
            raw = self.root / (mode + ".h5")
            aligned = self.root / (mode + "_aligned.h5")
            with patch.object(offline_h5_record, "MediaPipeCameraInput", return_value=capture), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["hand", "record", "--landmark-space", mode, "--hand-side", "left",
                                       "--no-preview", "--frames", "4", "--output", str(raw),
                                       "--aligned-output", str(aligned)]), 0)
            capture.close.assert_called_once()
            for call in capture.next_frame.call_args_list:
                self.assertEqual(call.kwargs, {"include_world": True})
            with h5py.File(raw, "r") as source:
                self.assertEqual(source["left_world_landmarks"].shape, (4, 21, 3))
                self.assertFalse(np.any(source["right_world_landmarks"][:]))
                self.assertFalse(np.any(source["right_hand_keypoints"][:]))
            dataset = TwoHandH5Dataset(aligned)
            self.assertEqual(dataset.coordinate_alignment, palm_local_metadata(mode)["coordinate_alignment"])
            self.assertEqual(dataset.side_counts()["left"], 2)
            before = raw.read_bytes()
            with self.assertRaises(FileExistsError):
                RawHandH5Writer(raw)
            self.assertEqual(raw.read_bytes(), before)

    def test_capture_failure_retains_complete_raw_frames(self):
        from teleoperation.applications import offline_h5_record
        raw, aligned = self.root / "partial.h5", self.root / "aligned.h5"
        capture = Mock()
        capture.next_frame.side_effect = [world_frame(0), world_frame(1), RuntimeError("camera failed")]
        with patch.object(offline_h5_record, "MediaPipeCameraInput", return_value=capture), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["hand", "record", "--no-preview", "--frames", "8", "--output", str(raw),
                                   "--aligned-output", str(aligned)]), 1)
        capture.close.assert_called_once()
        self.assertFalse(aligned.exists())
        with h5py.File(raw, "r") as source:
            self.assertEqual(source.attrs["recording_status"], "error")
            self.assertEqual(source.attrs["recorded_frames"], 2)
            self.assertTrue(all(value.shape[0] == 2 for value in source.values()))
        align_h5(raw, aligned)
        self.assertTrue(aligned.exists())

    def test_world_lengths_and_wrist_preserved_and_extra_scaling_rejected(self):
        frames = [world_frame(i) for i in range(5)]
        raw, aligned = self.root / "raw.h5", self.root / "aligned.h5"
        save_raw(raw, frames)
        before = raw.read_bytes()
        align_h5(raw, aligned)
        self.assertEqual(raw.read_bytes(), before)
        _, _, hands = load_twohand_h5(aligned)
        indices25 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
        for index, frame in enumerate(frames):
            for side in SIDES:
                original = frame["world_" + side]
                output = hands[side][index, indices25]
                np.testing.assert_array_equal(output[0], np.zeros(3))
                original_distances = np.linalg.norm(original[:, None] - original[None, :], axis=-1)
                aligned_distances = np.linalg.norm(output[:, None] - output[None, :], axis=-1)
                np.testing.assert_allclose(aligned_distances, original_distances, rtol=0, atol=1e-6)
        with h5py.File(aligned, "r") as source:
            for key, value in WORLD_PALM_LOCAL_METADATA.items():
                self.assertEqual(source.attrs[key], value)
        self.assertFalse(TwoHandH5Dataset(aligned, track_identity=True).track_identity)
        for scale in (2, 0, np.nan):
            with self.assertRaisesRegex(ValueError, "scale_factor=1.0"):
                TwoHandH5Dataset(aligned, scale_factor=scale)
        with patch("teleoperation.retargeting.hand.config.L21", NS(training=NS(robot_scale=2))):
            with self.assertRaisesRegex(ValueError, "robot_scale=1.0"):
                align_h5(raw, self.root / "scaled.h5")

    def test_offline_dataset_and_realtime_match_with_side_failures_and_recovery(self):
        max_error = 0.
        for invalid in ("missing", "nonfinite", "degenerate", "zero", "label_swap"):
            with self.subTest(invalid=invalid):
                frames = [world_frame(i) for i in range(9)]
                if invalid == "missing":
                    frames[3]["world_left"] = None
                elif invalid == "nonfinite":
                    frames[3]["world_left"][7, 0] = np.inf
                elif invalid == "degenerate":
                    frames[3]["world_left"] = np.arange(63, dtype=np.float32).reshape(21, 3) / 1000
                elif invalid == "zero":
                    frames[3]["world_left"] = np.zeros((21, 3), dtype=np.float32)
                else:
                    for frame in frames[3:]:
                        frame["world_left"], frame["world_right"] = frame["world_right"], frame["world_left"]
                raw, aligned = self.root / (invalid + ".h5"), self.root / (invalid + "_aligned.h5")
                save_raw(raw, frames)
                align_h5(raw, aligned)
                dataset = TwoHandH5Dataset(aligned)
                samples = {sample["frame_index"]: sample for sample in dataset.iter_samples()}
                pipeline = MediaPipePalmLocalPipeline("world")
                windows = []
                for index, frame in enumerate(frames):
                    window = pipeline.process_window(frame)
                    windows.append(window)
                    if index < 2:
                        self.assertIsNone(window)
                        continue
                    for side in SIDES:
                        actual = None if window is None else window.hands[side]
                        self.assertEqual(actual is not None, samples[index][side + "_valid"])
                        if actual is not None:
                            expected = samples[index][side + "_input"]
                            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
                            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                            np.testing.assert_array_equal(samples[index][side + "_target"][0], actual[-1])
                    self.assertEqual(window.units, "m")
                if invalid != "label_swap":
                    for index in (3, 4, 5):
                        self.assertIsNone(windows[index].hands["left"])
                        self.assertIsNotNone(windows[index].hands["right"])
                    self.assertIsNotNone(windows[6].hands["left"])
        self.assertLessEqual(max_error, 1e-6)
        print(f"world offline/Dataset/realtime max error={max_error:.9g} m")

    def test_missing_world_never_falls_back_and_clears_previous_window(self):
        pipeline = MediaPipePalmLocalPipeline("world")
        for index in range(3):
            pipeline.process_window(world_frame(index))
        with self.assertRaisesRegex(ValueError, "world_hands"):
            pipeline.process_window(raw_frame(3))
        self.assertEqual(pipeline.valid_streak, dict.fromkeys(SIDES, 0))
        self.assertIsNone(pipeline.process_window(world_frame(4)))
        self.assertIsNone(pipeline.process_window(world_frame(5)))
        self.assertIsNotNone(pipeline.process_window(world_frame(6)))
        raw = self.root / "old.h5"
        save_raw(raw, [world_frame(i) for i in range(4)], "normalized")
        with h5py.File(raw, "r+") as source:
            del source["left_world_landmarks"]
            del source["right_world_landmarks"]
            del source.attrs["hand_processing_landmark_space"]
        with self.assertRaisesRegex(ValueError, "World alignment requires"):
            align_h5(raw, self.root / "world.h5", "world")
        with self.assertRaises(ValueError):
            align_h5(raw,self.root / "legacy.h5","world")

    def test_existing_record_world_file_explicit_alignment_retains_body_fields(self):
        from tests.smplerx_fixtures import write
        from teleoperation.data.body_recording import load_body_recording
        raw, aligned = self.root / 'combined.h5', self.root / 'combined_aligned.h5'
        write(raw, 4)
        align_h5(raw, aligned, 'world')
        before = load_body_recording(raw)
        after = load_body_recording(aligned)
        for a,b in zip(before.samples,after.samples):
            np.testing.assert_array_equal(a['worker_packet'],b['worker_packet'])
            np.testing.assert_array_equal(a['rgb'],b['rgb'])
        with h5py.File(aligned,'r') as target:
            self.assertEqual(target.attrs['coordinate_alignment'],WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT)

    def test_invalid_file_contracts_are_rejected(self):
        raw = self.root / "raw.h5"
        save_raw(raw, [world_frame(i) for i in range(4)])
        for key, value in (("source_landmark_space", "mediapipe_normalized"), ("length_unit", "mm"),
                           ("source_scale", 2.), ("source_scale", "1.0"), ("length_unit", None)):
            aligned = self.root / "bad.h5"
            align_h5(raw, aligned)
            with h5py.File(aligned, "r+") as source:
                if value is None:
                    del source.attrs[key]
                else:
                    source.attrs[key] = value
            with self.assertRaises(ValueError):
                TwoHandH5Dataset(aligned)


    def test_world_workflow_requests_same_detection_and_keeps_preview_normalized(self):
        frames = [world_frame(i) for i in range(3)]
        capture = Mock()
        capture.next_observation.side_effect = [observation(frame) for frame in frames]
        with patch("teleoperation.applications.hand_realtime.MediaPipeCameraInput", return_value=capture):
            with MediaPipeCameraWorkflow("model.task") as workflow:
                for _ in frames:
                    window = workflow.next_window()
                self.assertEqual(window.metadata["source_landmark_space"], WORLD_LANDMARK_SPACE)
                for side in SIDES:
                    np.testing.assert_array_equal(workflow.last_raw_frame[side], frames[-1][side])
                for call in capture.next_observation.call_args_list:
                    self.assertEqual(call.kwargs, {"include_world": True})
        capture.close.assert_called_once()


    def test_cpu_shared_training_reload_and_left_export(self):
        from teleoperation.learning import trainer
        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import load_shared_joint_checkpoint
        from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
        from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
        from tests.test_joint_checkpoints import SMALL
        raw, aligned = self.root/'raw.h5', self.root/'aligned.h5'
        save_raw(raw,[world_frame(i) for i in range(30)])
        align_h5(raw,aligned)
        spec=load_hand_spec('l21','right')
        model=build_joint_model(spec,SMALL)
        initial={name:value.detach().clone() for name,value in model.named_parameters()}
        torch.set_num_threads(1)
        with patch.object(trainer,'build_joint_model',return_value=model):
            self.assertEqual(main(['hand','train','--input',str(aligned),'--run-name','native',
                '--hand-model','l21','--checkpoint-root',str(self.root/'models'),
                '--device','cpu','--epochs','1','--batch-size','4']),0)
        self.assertTrue(any(not torch.equal(initial[n],v) for n,v in model.named_parameters()))
        checkpoint=self.root/'models/l21/shared/nn'/PALM_LENGTH_PROFILE/'native/model_best.pth'
        loaded,payload=load_shared_joint_checkpoint(checkpoint,spec,'cpu')
        self.assertEqual(payload['side'],'shared')
        self.assertEqual(payload['units'],'1')
        self.assertTrue(np.isfinite(payload['best_loss']))
        pipeline=MediaPipePalmLocalPipeline('world',input_profile=PALM_LENGTH_PROFILE)
        for i in range(3): window=pipeline.process_window(world_frame(i))
        left_spec=load_hand_spec('l21','left')
        result=SharedJointNNRetargeter(
            {left_spec.key:left_spec, spec.key:spec}, loaded,
        ).retarget(window)[left_spec.key]
        self.assertEqual(result.positions.shape,(17,))
        output=self.root/'results.h5'
        self.assertEqual(main(['hand','export','--input',str(aligned),'--output',str(output),
            '--hand-model','l21','--backend','nn','--hand','left','--checkpoint',str(checkpoint),'--device','cpu','--no-collision-guard']),0)
        with h5py.File(output,'r') as source:
            self.assertEqual(source.attrs['schema'],'hand_joint_results_v1')
            self.assertEqual(source['l21/left/raw_positions'].shape,(30,17))
            self.assertEqual(int(source['l21/left/valid'][:].sum()),28)
            self.assertTrue(np.isfinite(source['l21/left/command_positions'][:]).all())


if __name__ == "__main__":
    unittest.main()
