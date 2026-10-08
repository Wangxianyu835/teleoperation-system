import unittest

import numpy as np
from teleoperation.contracts.coordinates import COORDINATE_FRAME
from teleoperation.retargeting.hand.coordinates import SOURCE_TO_L21_MATRIX, align_source_hand_coordinates, PALM_BASIS_EPS, PalmBasisError, align_palm_local_coordinates, build_l21_reference_basis, build_palm_basis
from teleoperation.retargeting.hand.topology import ensure_hand25
from tests.test_coordinate_contracts import synthetic_hand_pair
from scipy.spatial.transform import Rotation


class CoordinateAlignmentTests(unittest.TestCase):
    def test_fixed_alignment_applies_to_both_hands_identically(self):
        points = np.asarray(
            [[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            align_source_hand_coordinates(points),
            [[[-2.0, 3.0, -1.0], [-5.0, -6.0, 4.0]]],
        )

    def test_alignment_preserves_shape_and_input(self):
        points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
        original = points.copy()
        aligned = align_source_hand_coordinates(points)
        self.assertEqual(aligned.shape, points.shape)
        np.testing.assert_array_equal(points, original)
        self.assertTrue(np.isfinite(aligned).all())

    def test_alignment_matrix_is_a_proper_rotation(self):
        matrix = SOURCE_TO_L21_MATRIX
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)
        self.assertEqual(COORDINATE_FRAME, "l21")

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            align_source_hand_coordinates(np.zeros((3, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            align_source_hand_coordinates(
                np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32)
            )


class PalmLocalAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.references = {side: build_l21_reference_basis(side) for side in ("left", "right")}

    def test_source_and_side_specific_robot_bases_are_proper_rotations(self):
        for side, raw in zip(("left", "right"), synthetic_hand_pair()):
            for basis in (build_palm_basis(ensure_hand25(raw)), self.references[side]):
                with self.subTest(side=side, basis=basis):
                    self.assertEqual(basis.shape, (3, 3))
                    self.assertTrue(np.isfinite(basis).all())
                    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-5, rtol=0)
                    self.assertAlmostEqual(float(np.linalg.det(basis)), 1, places=5)
        self.assertFalse(np.allclose(self.references["left"], self.references["right"]))

    def test_axes_are_columns_and_have_the_declared_anatomical_directions(self):
        points = ensure_hand25(synthetic_hand_pair()[0])
        basis = build_palm_basis(points)
        longitudinal = points[[6, 11, 16, 21]].mean(axis=0) - points[0]
        longitudinal /= np.linalg.norm(longitudinal)
        np.testing.assert_allclose(basis[:, 1], longitudinal, atol=1e-6)
        self.assertGreater(np.dot(basis[:, 0], points[6] - points[21]), 0)
        np.testing.assert_allclose(basis[:, 2], np.cross(basis[:, 0], basis[:, 1]), atol=1e-6)
        # Inserted palm-root nodes have no role in defining the frame.
        points[[5, 10, 15, 20]] = [[.1, -.3, .7]]
        np.testing.assert_array_equal(build_palm_basis(points), basis)

    def test_translation_and_x_y_z_arbitrary_rotation_invariance(self):
        rotations = [Rotation.from_euler(axis, angle, degrees=True).as_matrix()
                     for axis, angle in (("x", 67), ("y", -43), ("z", 119), ("x", 180))]
        rotations.append(Rotation.from_euler("xyz", [37, -61, 83], degrees=True).as_matrix())
        translations = [np.zeros(3), np.array([.37, -.19, .23])]
        for side, hand in zip(("left", "right"), synthetic_hand_pair()):
            reference = self.references[side]
            canonical = align_palm_local_coordinates(ensure_hand25(hand), reference)
            translated = align_palm_local_coordinates(ensure_hand25(hand + translations[1]), reference)
            np.testing.assert_allclose(translated, canonical, atol=1e-5, rtol=0)
            for rotation in rotations:
                for translation in translations:
                    with self.subTest(side=side, rotation=rotation, translation=translation):
                        transformed = ensure_hand25(hand @ rotation.T + translation)
                        aligned = align_palm_local_coordinates(transformed, reference)
                        np.testing.assert_allclose(aligned, canonical, atol=1e-5, rtol=0)
                        # Basis covariance establishes rotation, rather than reflection.
                        np.testing.assert_allclose(build_palm_basis(transformed), rotation @ build_palm_basis(ensure_hand25(hand)), atol=1e-5, rtol=0)

    def test_all_pairwise_distances_are_preserved(self):
        for side, raw in zip(("left", "right"), synthetic_hand_pair()):
            points = ensure_hand25(raw)
            aligned = align_palm_local_coordinates(points, self.references[side])
            before = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            after = np.linalg.norm(aligned[:, None] - aligned[None, :], axis=-1)
            np.testing.assert_allclose(after, before, atol=1e-6, rtol=1e-6)

    def test_only_finger_motion_remains_visible_and_has_no_history(self):
        points = ensure_hand25(synthetic_hand_pair()[0])
        moved = points.copy()
        moved[[7, 8, 9]] += [.005, -.01, .02]
        reference = self.references["left"]
        aligned = align_palm_local_coordinates(points, reference)
        changed = align_palm_local_coordinates(moved, reference)
        np.testing.assert_allclose(build_palm_basis(points), build_palm_basis(moved))
        self.assertGreater(np.max(np.abs(changed - aligned)), .005)
        align_palm_local_coordinates(np.zeros((25, 3)), reference)
        np.testing.assert_array_equal(align_palm_local_coordinates(points, reference), aligned)

    def test_zero_degenerate_and_nonfinite_frames_return_finite_zeros(self):
        normal = ensure_hand25(synthetic_hand_pair()[0])
        collinear = np.zeros((25, 3), dtype=np.float32)
        collinear[[6, 11, 16, 21], 1] = [1, 2, 3, 4]
        coincident = np.ones((21, 3), dtype=np.float32)
        tiny_longitudinal = normal.copy()
        tiny_longitudinal[[6, 11, 16, 21]] = [0, PALM_BASIS_EPS / 2, 0]
        tiny_lateral = normal.copy()
        tiny_lateral[6] = tiny_lateral[21] + [PALM_BASIS_EPS / 4, 0, 0]
        cases = [ensure_hand25(np.zeros((21, 3))), ensure_hand25(coincident), collinear,
                 tiny_longitudinal, tiny_lateral]
        for invalid in (np.nan, np.inf, -np.inf):
            # A non-palm point invalidates the entire source side frame too.
            frame = normal.copy()
            frame[4, 0] = invalid
            cases.append(frame)
        for frame in cases:
            with self.subTest(frame=frame):
                result = align_palm_local_coordinates(frame, self.references["left"])
                self.assertEqual(result.shape, (25, 3))
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_array_equal(result, np.zeros((25, 3)))

    def test_robot_basis_errors_are_not_silently_treated_as_missing_source(self):
        for reference in (np.zeros((3, 3)), np.diag([-1, 1, 1]), np.eye(3) * 2, np.full((3, 3), np.nan)):
            with self.subTest(reference=reference), self.assertRaises(PalmBasisError):
                align_palm_local_coordinates(np.zeros((25, 3)), reference)

    def test_input_points_and_robot_basis_are_not_modified(self):
        points = ensure_hand25(synthetic_hand_pair()[0])
        original = points.copy()
        reference = self.references["left"].copy()
        saved_reference = reference.copy()
        align_palm_local_coordinates(points, reference)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_array_equal(reference, saved_reference)

    def test_bad_shapes_or_side_are_programming_errors(self):
        with self.assertRaises(ValueError):
            align_palm_local_coordinates(np.zeros((21, 3)), self.references["left"])
        with self.assertRaises(ValueError):
            build_l21_reference_basis("other")


if __name__ == "__main__":
    unittest.main()
