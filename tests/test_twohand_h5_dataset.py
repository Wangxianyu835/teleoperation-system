import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from teleoperation.learning.dataset import TwoHandH5ChunkedGenerator, TwoHandH5Dataset
from teleoperation.data.hand_h5 import load_twohand_h5


class TwoHandH5DatasetTests(unittest.TestCase):
    def _write_h5(self, path: Path, frame_count: int = 8) -> None:
        points = np.arange(21 * 3, dtype=np.float32).reshape(21, 3)
        left = np.stack(
            [points + [0.70 + 0.01 * index, 0.0, 0.0] for index in range(frame_count)]
        )
        right = np.stack(
            [points + [0.30 - 0.01 * index, 0.0, 0.0] for index in range(frame_count)]
        )
        left[4] = 0
        right[6] = 0
        with h5py.File(path, "w") as h5_file:
            h5_file.attrs["coordinate_frame"] = "l21"
            h5_file.attrs["coordinate_alignment"] = "source_to_l21_xyz"
            h5_file.create_dataset("frame_ids", data=np.arange(frame_count))
            h5_file.create_dataset(
                "timestamps",
                data=np.arange(frame_count, dtype=np.float64) * 0.02,
            )
            h5_file.create_dataset("left_hand_keypoints", data=left)
            h5_file.create_dataset("right_hand_keypoints", data=right)

    def test_loads_root_schema_and_converts_to_25_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hands.h5"
            self._write_h5(path)
            frame_ids, timestamps, hands = load_twohand_h5(path)
            dataset = TwoHandH5Dataset(path, receptive_field=3)

        self.assertEqual(frame_ids.shape, (8,))
        self.assertEqual(timestamps.shape, (8,))
        self.assertEqual(hands["left"].shape, (8, 21, 3))
        self.assertEqual(len(dataset), 6)
        sample = dataset.samples[0]
        self.assertEqual(sample["left_input"].shape, (3, 25, 3))
        self.assertEqual(sample["right_input"].shape, (3, 25, 3))
        np.testing.assert_array_equal(
            sample["left_input"][:, 0, :],
            np.zeros((3, 3), dtype=np.float32),
        )

    def test_missing_hand_resets_only_that_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hands.h5"
            self._write_h5(path)
            dataset = TwoHandH5Dataset(path, receptive_field=3)

        by_frame = {sample["frame_index"]: sample for sample in dataset.samples}
        self.assertFalse(by_frame[6]["left_valid"])
        self.assertFalse(by_frame[6]["right_valid"])
        self.assertTrue(by_frame[7]["left_valid"])
        self.assertFalse(by_frame[7]["right_valid"])

    def test_generator_returns_masks_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hands.h5"
            self._write_h5(path)
            dataset = TwoHandH5Dataset(path, receptive_field=3)
            batch = next(
                TwoHandH5ChunkedGenerator(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                ).next_epoch()
            )

        self.assertEqual(batch["left_input"].shape, (2, 3, 25, 3))
        self.assertEqual(batch["right_target"].shape, (2, 1, 25, 3))
        self.assertEqual(batch["left_valid"].dtype, np.bool_)
        self.assertEqual(batch["frame_index"].dtype, np.int64)

    def test_identity_swap_is_reassigned_and_resets_original_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "identity_swap.h5"
            shape = np.linspace(0.0, 0.2, 21 * 3, dtype=np.float32).reshape(21, 3)
            left = np.stack([shape + [0.70 + 0.005 * i, 0.0, 0.0] for i in range(8)])
            right = np.stack([shape + [0.30 - 0.005 * i, 0.0, 0.0] for i in range(8)])
            left[5:] = 0
            right[5] = shape + [0.725, 0.0, 0.0]
            right[6] = shape + [0.730, 0.0, 0.0]
            right[7] = 0
            with h5py.File(path, "w") as h5_file:
                h5_file.attrs["coordinate_frame"] = "l21"
                h5_file.attrs["coordinate_alignment"] = "source_to_l21_xyz"
                h5_file.create_dataset("frame_ids", data=np.arange(8))
                h5_file.create_dataset("timestamps", data=np.arange(8) / 30.0)
                h5_file.create_dataset("left_hand_keypoints", data=left)
                h5_file.create_dataset("right_hand_keypoints", data=right)

            dataset = TwoHandH5Dataset(path, receptive_field=3)

        by_frame = {sample["frame_index"]: sample for sample in dataset.samples}
        self.assertTrue(by_frame[5]["left_valid"])
        self.assertTrue(by_frame[6]["left_valid"])
        self.assertFalse(by_frame[5]["right_valid"])
        self.assertFalse(by_frame[6]["right_valid"])
        self.assertFalse(by_frame[7]["left_valid"])
        self.assertFalse(by_frame[7]["right_valid"])


if __name__ == "__main__":
    unittest.main()
