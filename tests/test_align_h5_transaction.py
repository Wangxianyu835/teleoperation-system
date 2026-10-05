import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from model.kinematics import create_hand_kinematics

from retargeting.coordinates import (
    PALM_LOCAL_METADATA, PalmBasisError,
    align_palm_local_coordinates, build_l21_reference_basis,
)
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

    def test_nonfinite_geometry_becomes_zero_without_aborting_recording(self):
        with h5py.File(self.source, "a") as handle:
            handle["right_hand_keypoints"][0, 4, 0] = np.nan
        before = self.source.read_bytes()
        report = align_h5(self.source, self.destination)
        with h5py.File(self.destination, "r") as handle:
            np.testing.assert_array_equal(handle["right_hand_keypoints"][0], np.zeros((25, 3)))
            self.assertTrue(np.isfinite(handle["right_hand_keypoints"][:]).all())
            self.assertTrue(np.any(handle["right_hand_keypoints"][1]))
            self.assertTrue(np.any(handle["left_hand_keypoints"][0]))
        self.assertEqual(report["sides"]["right"]["degenerate_frames"], 1)
        self.assertEqual(self.source.read_bytes(), before)

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
                    for key, value in PALM_LOCAL_METADATA.items():
                        self.assertEqual(handle.attrs[key], value)
                    self.assertEqual(handle.attrs["data_source"], "synthetic")
                    self.assertEqual(handle["extra"][0], 42)
                    container = handle["recording"] if grouped else handle
                    self.assertEqual(container.attrs["recording_note"], "preserve")
                    np.testing.assert_array_equal(container["frame_ids"][:], np.arange(3))
                    np.testing.assert_array_equal(container["timestamps"][:], np.arange(3) / 30)
                    for side, hand in zip(("left", "right"), synthetic_hand_pair()):
                        key = f"{side}_hand_keypoints" if not grouped else ("l_glove_pos" if side == "left" else "r_glove_pos")
                        self.assertEqual(container[key].attrs["units"], "synthetic_unit")
                        expected = align_palm_local_coordinates(ensure_hand25(hand), build_l21_reference_basis(side))
                        np.testing.assert_allclose(container[key][:], np.repeat(expected[None], 3, axis=0))
                self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.*.tmp")), [])

    def test_mixed_frames_keep_timeline_masks_and_compute_reference_once_per_side(self):
        for point_count in (21, 25):
            with self.subTest(point_count=point_count):
                self._write_source()
                with h5py.File(self.source, "a") as handle:
                    handle.attrs["source_file"] = "preserve_existing_provenance"
                    for side, raw in zip(("left", "right"), synthetic_hand_pair()):
                        normal = ensure_hand25(raw) if point_count == 25 else raw
                        corrupt = normal.copy()
                        corrupt[3, 2] = np.inf
                        data = np.stack((normal, np.zeros_like(normal), np.ones_like(normal), corrupt))
                        del handle[f"{side}_hand_keypoints"]
                        handle.create_dataset(f"{side}_hand_keypoints", data=data)
                    for key, value in (("frame_ids", [7, 9, 10, 15]), ("timestamps", [.1, .2, .22, .4])):
                        del handle[key]
                        handle.create_dataset(key, data=value)
                before = self.source.read_bytes()
                with mock.patch("model.kinematics.create_hand_kinematics", wraps=create_hand_kinematics) as factory:
                    with mock.patch("retargeting.coordinates.align_source_hand_coordinates", side_effect=AssertionError("legacy matrix must not be called")):
                        report = align_h5(self.source, self.destination)
                self.assertEqual(factory.call_count, 2)
                self.assertEqual(report["frames"], 4)
                self.assertEqual(self.source.read_bytes(), before)
                with h5py.File(self.destination, "r") as handle:
                    self.assertEqual(handle.attrs["source_file"], "preserve_existing_provenance")
                    np.testing.assert_array_equal(handle["frame_ids"][:], [7, 9, 10, 15])
                    np.testing.assert_array_equal(handle["timestamps"][:], [.1, .2, .22, .4])
                    for side in ("left", "right"):
                        data = handle[f"{side}_hand_keypoints"][:]
                        self.assertEqual(data.shape, (4, 25, 3))
                        self.assertTrue(np.any(data[0]))
                        np.testing.assert_array_equal(data[1:], np.zeros((3, 25, 3)))
                        self.assertEqual(report["sides"][side]["nonzero_frames"], 1)
                        self.assertEqual(report["sides"][side]["zero_source_frames"], 1)
                        self.assertEqual(report["sides"][side]["degenerate_frames"], 2)
                        self.assertEqual(report["sides"][side]["nonfinite_values"], 0)

    def test_declared_source_space_conflicts_preserve_destination(self):
        for grouped in (False, True):
            for space in ("mediapipe_world", "visionpro", "calibrated_3d", 123):
                with self.subTest(grouped=grouped, space=space):
                    self._write_source(grouped)
                    with h5py.File(self.source, "a") as handle:
                        container = handle["recording"] if grouped else handle
                        container.attrs["source_landmark_space"] = space
                    with self.assertRaisesRegex(ValueError, "source_landmark_space"):
                        align_h5(self.source, self.destination)
                    self._assert_preserved()

    def test_normalized_source_space_is_accepted_and_second_alignment_is_rejected(self):
        with h5py.File(self.source, "a") as handle:
            handle.attrs["source_landmark_space"] = "mediapipe_normalized"
        align_h5(self.source, self.destination)
        second = self.destination.with_name("twice.h5")
        with self.assertRaisesRegex(ValueError, "already"):
            align_h5(self.destination, second)
        self.assertFalse(second.exists())

    def test_invalid_robot_reference_fails_with_geometry_before_replacement(self):
        fake_fk = mock.Mock()
        fake_fk.forward.return_value = (None, None, torch.zeros((1, 23, 3)))
        for side in ("left", "right"):
            with self.subTest(side=side):
                with mock.patch("model.kinematics.create_hand_kinematics", return_value=fake_fk):
                    with self.assertRaises(PalmBasisError) as context:
                        build_l21_reference_basis(side)
                message = str(context.exception)
                for expected in (f"side={side}", "(0, 1, 4, 7, 10)", "positions=", "longitudinal"):
                    self.assertIn(expected, message)
        with mock.patch("model.kinematics.create_hand_kinematics", return_value=fake_fk):
            with mock.patch("os.replace") as replace:
                with self.assertRaises(PalmBasisError):
                    align_h5(self.source, self.destination)
                replace.assert_not_called()
        self._assert_preserved()

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
