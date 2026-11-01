"""Normalized human data is immutable and shared across target hand models."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from teleoperation.applications.hand_align import align_h5
from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE, PALM_LENGTH_ALIGNMENT
from teleoperation.learning.dataset import TwoHandH5Dataset
from tests.test_world_hand_pipeline import world_frame, save_raw


class SharedHumanDatasetTests(unittest.TestCase):
    def test_offline_and_streaming_windows_targets_and_masks_agree_after_missing_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, aligned = Path(directory)/"raw.h5", Path(directory)/"aligned.h5"
            frames = [world_frame(i) for i in range(10)]
            frames[4]["world_left"] = None
            save_raw(raw, frames)
            align_h5(raw, aligned)
            before = hashlib.sha256(aligned.read_bytes()).hexdigest()
            dataset = TwoHandH5Dataset(aligned, input_profile=PALM_LENGTH_PROFILE)
            samples = {sample["frame_index"]: sample for sample in dataset.iter_samples()}
            pipeline = MediaPipePalmLocalPipeline("world", input_profile=PALM_LENGTH_PROFILE)
            for index, frame in enumerate(frames):
                actual = pipeline.process_window(frame)
                if index < 2:
                    self.assertIsNone(actual)
                    continue
                for side in ("left", "right"):
                    points = None if actual is None else actual.hands[side]
                    self.assertEqual(samples[index][side+"_valid"], points is not None)
                    if points is not None:
                        np.testing.assert_allclose(samples[index][side+"_input"], points, atol=1e-6, rtol=0)
                        np.testing.assert_array_equal(samples[index][side+"_target"][0], points[-1])
                if index in (4,5,6):
                    self.assertFalse(samples[index]['left_valid'])
                if index >= 7:
                    self.assertTrue(samples[index]['left_valid'], 'lost side must recover after three new valid frames')
            self.assertEqual(dataset.coordinate_alignment, PALM_LENGTH_ALIGNMENT)
            self.assertEqual(dataset.units, "1")
            other_target_dataset = TwoHandH5Dataset(aligned, input_profile=PALM_LENGTH_PROFILE)
            np.testing.assert_array_equal(dataset.samples[-1]["right_input"], other_target_dataset.samples[-1]["right_input"])
            self.assertEqual(hashlib.sha256(aligned.read_bytes()).hexdigest(), before)

    def test_split_boundary_never_reuses_prior_partition_history(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, aligned = Path(directory)/"raw.h5", Path(directory)/"aligned.h5"
            save_raw(raw, [world_frame(i) for i in range(12)])
            align_h5(raw, aligned)
            dataset = TwoHandH5Dataset(aligned, frame_start=6, frame_end=12, input_profile=PALM_LENGTH_PROFILE)
            self.assertEqual(dataset.samples[0]["frame_index"], 8)
            self.assertTrue(dataset.samples[0]["left_valid"])

    def test_both_sides_missing_then_recover_in_three_frames(self):
        pipeline=MediaPipePalmLocalPipeline('world',input_profile=PALM_LENGTH_PROFILE)
        windows=[]
        for i in range(8):
            frame=world_frame(i)
            if i==3:
                frame['world_left']=frame['world_right']=None
            windows.append(pipeline.process_window(frame))
        self.assertIsNotNone(windows[2])
        self.assertTrue(all(windows[i] is None for i in (3,4,5)))
        self.assertTrue(all(windows[6].hands[s] is not None for s in ('left','right')))

    def test_invalid_palm_during_warmup_does_not_discard_later_valid_frames(self):
        frames=[world_frame(i) for i in range(8)]
        for side in ('left','right'):
            frames[0]['world_'+side] *= 1e-6
        pipeline=MediaPipePalmLocalPipeline('world',input_profile=PALM_LENGTH_PROFILE)
        windows=[pipeline.process_window(frame) for frame in frames]
        self.assertIsNone(windows[2])
        self.assertIsNotNone(windows[3], 'frames1,2,3 are three valid frames after bad frame0')
        with tempfile.TemporaryDirectory() as directory:
            raw,aligned=Path(directory)/'raw.h5',Path(directory)/'aligned.h5'
            save_raw(raw,frames); align_h5(raw,aligned)
            dataset=TwoHandH5Dataset(aligned,input_profile=PALM_LENGTH_PROFILE)
            rows={s['frame_index']:s for s in dataset.samples}
            for side in ('left','right'):
                self.assertFalse(rows[2][side+'_valid'])
                self.assertTrue(rows[3][side+'_valid'])


    def test_profile_rejects_normalized_image_data_and_bad_selection(self):
        with self.assertRaises(ValueError):
            MediaPipePalmLocalPipeline("normalized", input_profile=PALM_LENGTH_PROFILE)
        with tempfile.TemporaryDirectory() as directory:
            raw, aligned = Path(directory)/"raw.h5", Path(directory)/"aligned.h5"
            save_raw(raw, [world_frame(i) for i in range(5)])
            align_h5(raw, aligned)
            with self.assertRaises(ValueError):
                TwoHandH5Dataset(aligned, input_profile="unknown")


if __name__ == "__main__":
    unittest.main()
