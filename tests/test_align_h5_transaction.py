import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from retargeting.coordinates import align_source_hand_coordinates
from retargeting.tracking import ensure_hand25
from scripts.align_h5_coordinates import align_h5
from tests.test_coordinate_contracts import synthetic_hand_pair


class AlignmentTransactionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "raw.h5"
        self.destination = Path(self.directory.name) / "aligned.h5"
        self.previous = b"previous destination must survive failure"
        self.destination.write_bytes(self.previous)
        self._write_source()

    def _write_source(self, grouped=False):
        with h5py.File(self.source, "w") as handle:
            handle.attrs["data_source"] = "synthetic"
            container = handle.create_group("recording") if grouped else handle
            container.attrs["recording_note"] = "preserve"
            for side, hand in zip(("left", "right"), synthetic_hand_pair()):
                key = f"{side}_hand_keypoints" if not grouped else ("l_glove_pos" if side == "left" else "r_glove_pos")
                dataset = container.create_dataset(key, data=np.repeat(hand[None], 3, axis=0))
                dataset.attrs["units"] = "synthetic_unit"
            container.create_dataset("frame_ids", data=np.arange(3))
            container.create_dataset("timestamps", data=np.arange(3) / 30)
            handle.create_dataset("extra", data=np.array([42]))

    def _assert_preserved(self):
        self.assertEqual(self.destination.read_bytes(), self.previous)
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.*.tmp")), [])

    def test_already_aligned_input_does_not_truncate_destination(self):
        with h5py.File(self.source, "a") as handle:
            handle.attrs["coordinate_frame"] = "l21"
        with self.assertRaisesRegex(ValueError, "already"):
            align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_invalid_metadata_does_not_truncate_destination(self):
        for name, value in (("coordinate_frame", 123), ("coordinate_alignment", "unknown_rotation")):
            with self.subTest(name=name):
                self._write_source()
                with h5py.File(self.source, "a") as handle:
                    handle.attrs[name] = value
                with self.assertRaises(ValueError):
                    align_h5(self.source, self.destination)
                self._assert_preserved()

    def test_nonscalar_coordinate_metadata_is_rejected(self):
        with h5py.File(self.source, "a") as handle:
            handle.attrs["coordinate_frame"] = np.array([b"source", b"source"])
        with self.assertRaises(ValueError):
            align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_nonfinite_geometry_fails_before_replacement(self):
        with h5py.File(self.source, "a") as handle:
            handle["right_hand_keypoints"][0, 4, 0] = np.nan
        with mock.patch("os.replace") as replace:
            with self.assertRaisesRegex(ValueError, "finite"):
                align_h5(self.source, self.destination)
            replace.assert_not_called()
        self._assert_preserved()

    def test_mismatched_frame_counts_are_rejected_before_output(self):
        with h5py.File(self.source, "a") as handle:
            del handle["right_hand_keypoints"]
            handle.create_dataset("right_hand_keypoints", data=np.ones((2, 21, 3)))
        with self.assertRaisesRegex(ValueError, "frame counts"):
            align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_invalid_optional_vectors_are_rejected_before_output(self):
        with h5py.File(self.source, "a") as handle:
            del handle["timestamps"]
            handle.create_dataset("timestamps", data=np.array([0., np.inf, 2.]))
        with self.assertRaisesRegex(ValueError, "finite"):
            align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_success_replaces_closed_destination_and_preserves_input_and_metadata(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                self._write_source(grouped)
                before = hashlib.sha256(self.source.read_bytes()).hexdigest()
                align_h5(self.source, self.destination)
                self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
                with h5py.File(self.destination, "r") as handle:
                    self.assertEqual(handle.attrs["coordinate_frame"], "l21")
                    self.assertEqual(handle.attrs["coordinate_alignment"], "source_to_l21_xyz")
                    self.assertEqual(handle.attrs["data_source"], "synthetic")
                    self.assertEqual(handle["extra"][0], 42)
                    container = handle["recording"] if grouped else handle
                    self.assertEqual(container.attrs["recording_note"], "preserve")
                    for side, hand in zip(("left", "right"), synthetic_hand_pair()):
                        key = f"{side}_hand_keypoints" if not grouped else ("l_glove_pos" if side == "left" else "r_glove_pos")
                        self.assertEqual(container[key].attrs["units"], "synthetic_unit")
                        expected = align_source_hand_coordinates(ensure_hand25(hand))
                        np.testing.assert_allclose(container[key][:], np.repeat(expected[None], 3, axis=0))
                self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.*.tmp")), [])

    def test_write_failure_preserves_destination_and_removes_own_temporary_file(self):
        original = h5py.Group.create_dataset

        def fail_right(group, name, *args, **kwargs):
            if name == "right_hand_keypoints":
                raise OSError("injected write failure")
            return original(group, name, *args, **kwargs)

        with mock.patch.object(h5py.Group, "create_dataset", new=fail_right):
            with self.assertRaisesRegex(OSError, "injected"):
                align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_failure_before_replace_preserves_destination(self):
        with mock.patch("os.replace", side_effect=PermissionError("injected replacement failure")):
            with self.assertRaisesRegex(PermissionError, "injected"):
                align_h5(self.source, self.destination)
        self._assert_preserved()

    def test_same_path_is_rejected_without_modifying_input(self):
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different"):
            align_h5(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_same_file_hardlink_is_rejected_without_modifying_input(self):
        alias = Path(self.directory.name) / "alias.h5"
        os.link(self.source, alias)
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different"):
            align_h5(self.source, alias)
        self.assertEqual(self.source.read_bytes(), before)

    @unittest.skipUnless(os.name == "nt", "Windows open-file replacement semantics")
    def test_windows_open_destination_replacement_failure_preserves_bytes(self):
        with self.destination.open("rb"):
            with self.assertRaises(OSError):
                align_h5(self.source, self.destination)
        self._assert_preserved()


if __name__ == "__main__":
    unittest.main()
