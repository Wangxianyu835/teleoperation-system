import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.learning.losses import CollisionLoss, tip_distance_loss
from teleoperation.retargeting.hand.config import L21, ROBOT_JOINTS, SOURCE_JOINTS
from teleoperation.retargeting.hand.coordinates import (
    SOURCE_TO_L21_MATRIX, align_source_hand_coordinates,
    align_palm_local_coordinates, build_l21_reference_basis,
    build_palm_basis, align_mediapipe_hand, MEDIAPIPE_PALM_INDICES,
    OPERATOR_TO_CANONICAL, OPERATOR_TO_DEX_LEFT, OPERATOR_TO_DEX_RIGHT,
    LEFT_DEX_TO_CANONICAL, PalmBasisError,
)
from teleoperation.retargeting.hand.angles import canonical_to_native_angles
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.applications.hand_processing import CanonicalHandPipeline
from teleoperation.retargeting.hand.topology import ensure_hand25
from teleoperation.applications.diagnostics.diagnose_hand_coordinates import signed_hand_volume, transform_diagnostics


def synthetic_hand_pair():
    left = np.zeros((21, 3), dtype=np.float32)
    bases = np.array([[.018, .008, .010], [.015, .030, 0], [0, .035, 0],
                      [-.012, .030, 0], [-.022, .023, .005]], dtype=np.float32)
    for finger, base in enumerate(bases):
        left[1 + 4 * finger:5 + 4 * finger] = np.array([1, 1.35, 1.65, 2])[:, None] * base
    right = left * np.array([-1, 1, 1], dtype=np.float32)
    return left + np.array([-.2, .4, .1]), right + np.array([.2, .4, .1])


class CoordinateContractTests(unittest.TestCase):
    def test_three_point_basis_matches_dex_axes_and_is_proper(self):
        _, raw = synthetic_hand_pair()
        relative = raw - raw[0:1]
        basis = build_palm_basis(relative, MEDIAPIPE_PALM_INDICES)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(basis), 1., places=6)
        np.testing.assert_allclose(basis[:, 0], -relative[9] / np.linalg.norm(relative[9]), atol=1e-6)
        self.assertGreater(np.dot(basis[:, 2], relative[5] - relative[9]), 0)
        for matrix in (OPERATOR_TO_DEX_LEFT, OPERATOR_TO_DEX_RIGHT):
            self.assertAlmostEqual(np.linalg.det(matrix), 1.)
        self.assertAlmostEqual(np.linalg.det(OPERATOR_TO_CANONICAL["left"]), -1.)
        np.testing.assert_array_equal(np.array([[2., 3., 4.]]) @ OPERATOR_TO_DEX_RIGHT, [[-3., 4., -2.]])
        np.testing.assert_array_equal(np.array([[2., 3., 4.]]) @ OPERATOR_TO_DEX_LEFT, [[3., -4., -2.]])
        np.testing.assert_array_equal(LEFT_DEX_TO_CANONICAL, np.diag([1, -1, 1]))

    def test_canonical_coordinates_are_rigid_motion_invariant_and_preserve_distances(self):
        rng = np.random.default_rng(123)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
        for side, raw in zip(("left", "right"), synthetic_hand_pair()):
            output = align_mediapipe_hand(raw, side=side)
            transformed = align_mediapipe_hand(raw @ rotation + [.3, -.2, .5], side=side)
            np.testing.assert_allclose(output, transformed, rtol=0, atol=2e-7)
            np.testing.assert_array_equal(output[0], np.zeros(3))
            before = ensure_hand25(raw)
            np.testing.assert_allclose(np.linalg.norm(output[:, None] - output[None], axis=-1),
                                       np.linalg.norm(before[:, None] - before[None], axis=-1), atol=2e-7)
            # The declared final +Y faces index, +Z faces middle MCP.
            self.assertGreater(output[6, 1] - output[11, 1], 0)
            self.assertGreater(output[11, 2], 0)
            self.assertAlmostEqual(float(output[11, 1]), 0., places=6)

    def test_mirrored_bent_hands_share_canonical_input_without_ring_or_pinky_basis_dependency(self):
        left, right = synthetic_hand_pair()
        left -= left[0:1]
        right -= right[0:1]
        # Nonplanar distal points make an omitted chirality correction visible.
        left[[7, 8, 11, 12], 2] += [.01, .02, .015, .025]
        right = left * [-1, 1, 1]
        expected = align_mediapipe_hand(left, side="left")
        np.testing.assert_allclose(expected, align_mediapipe_hand(right, side="right"), atol=1e-7)
        perturbed = left.copy()
        perturbed[13:] += np.array([.07, -.04, .02])
        np.testing.assert_array_equal(build_palm_basis(left, MEDIAPIPE_PALM_INDICES),
                                      build_palm_basis(perturbed, MEDIAPIPE_PALM_INDICES))
        np.testing.assert_array_equal(expected[:15], align_mediapipe_hand(perturbed, side="left")[:15])

    def test_three_point_degeneracy_is_rejected_even_when_other_fingers_define_a_plane(self):
        hand = synthetic_hand_pair()[0]
        hand[5] = hand[0] + [0., .02, 0.]
        hand[9] = hand[0] + [1e-7, .04, 0.]
        with self.assertRaisesRegex(PalmBasisError, "collinear"):
            build_palm_basis(hand, MEDIAPIPE_PALM_INDICES)
        np.testing.assert_array_equal(align_mediapipe_hand(hand, side="left"), np.zeros((25, 3)))

    def test_native_angle_adapter_matches_both_urdfs_axial_directions(self):
        axes = {}
        for side in ("left", "right"):
            fk = create_hand_kinematics(getattr(L21, side + "_urdf"), L21.hand_kinematics_config(), "cpu")
            _, rotations, _ = fk.forward(torch.zeros((1, 23)))
            matrix = build_l21_reference_basis(side) @ OPERATOR_TO_CANONICAL[side]
            # Under a reflection, a rotation axis is an axial vector.
            native_axes = np.einsum("nij,nj->ni", rotations[0].numpy(), fk.joint_axes.numpy())
            signs = canonical_to_native_angles(np.ones(23), side)
            axes[side] = np.linalg.det(matrix) * (native_axes @ matrix) * signs[:, None]
        dot = np.sum(axes["left"][1:18] * axes["right"][1:18], axis=-1)
        self.assertTrue(np.all(dot > .995), dot)
        values = torch.linspace(-.1, .1, 18, requires_grad=True)
        native = canonical_to_native_angles(values, "left")
        native.sum().backward()
        np.testing.assert_array_equal(values.grad.numpy(), canonical_to_native_angles(np.ones(18), "left"))

    def test_training_fk_and_human_targets_use_the_same_fixed_axes(self):
        from teleoperation.applications.training import _create_hand_fks
        native = _create_hand_fks("cpu", ("right",))["right"]
        canonical = _create_hand_fks("cpu", ("right",), canonical=True)["right"]
        pose = torch.zeros((1, 23))
        raw_positions = native.forward(pose)[2][0].numpy()
        human = np.zeros((21, 3), dtype=np.float32)
        human[[0, 5, 9]] = raw_positions[[0, 1, 4]]
        targets = align_mediapipe_hand(human, side="right")
        actual = canonical.forward(pose)[2][0].numpy()
        np.testing.assert_allclose(actual[[0, 1, 4]], targets[[0, 6, 11]], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(actual[:, None] - actual[None], axis=-1),
                                   np.linalg.norm(raw_positions[:, None] - raw_positions[None], axis=-1), atol=1e-7)
        rotations = canonical.forward(pose)[1][0].numpy()
        np.testing.assert_allclose(np.linalg.det(rotations), np.ones(23), atol=1e-6)

    def test_basis_directions_match_declared_transform(self):
        np.testing.assert_array_equal(
            align_source_hand_coordinates(np.eye(3)),
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
        )

    def test_reflection_is_detected_even_when_matrix_is_orthogonal(self):
        reflected = np.diag([-1, 1, 1]) @ SOURCE_TO_L21_MATRIX
        diagnostic = transform_diagnostics(reflected)
        self.assertTrue(diagnostic["orthogonal"])
        self.assertTrue(diagnostic["reflection_detected"])
        self.assertFalse(diagnostic["proper_rotation"])
        hand = ensure_hand25(synthetic_hand_pair()[0])
        self.assertAlmostEqual(signed_hand_volume(hand @ reflected.T), -signed_hand_volume(hand))

    def test_palm_local_keeps_each_sides_chirality_and_fixed_wrist(self):
        volumes = []
        for side, raw in zip(("left", "right"), synthetic_hand_pair()):
            points = ensure_hand25(raw)
            aligned = align_palm_local_coordinates(points, build_l21_reference_basis(side))
            self.assertEqual(aligned.shape, (25, 3))
            self.assertTrue(np.isfinite(aligned).all())
            np.testing.assert_array_equal(aligned[0], np.zeros(3))
            before, after = signed_hand_volume(points), signed_hand_volume(aligned)
            self.assertGreater(before * after, 0)
            np.testing.assert_allclose(after, before, atol=1e-10, rtol=1e-5)
            volumes.append(after)
        self.assertLess(volumes[0] * volumes[1], 0)

    def test_both_chiralities_survive_conversion_alignment_and_identity_processing(self):
        raw = dict(zip(("left", "right"), synthetic_hand_pair()))
        canonical = {side: ensure_hand25(points) for side, points in raw.items()}
        self.assertLess(signed_hand_volume(canonical["left"]) * signed_hand_volume(canonical["right"]), 0)
        aligned = {side: align_source_hand_coordinates(points) for side, points in canonical.items()}
        processor = CanonicalHandPipeline()
        for _ in range(3):
            payload = processor.update(aligned["left"], aligned["right"])
        for side in ("left", "right"):
            self.assertAlmostEqual(signed_hand_volume(aligned[side]), signed_hand_volume(canonical[side]))
            np.testing.assert_allclose(payload["hands"][side][-1], aligned[side], atol=1e-7)
        # Absolute source positions disambiguate a label exchange. Wrist-relative
        # H5 points discard that evidence; do not claim an ambiguous swap is solved.
        raw_processor = CanonicalHandPipeline()
        for _ in range(3):
            raw_processor.update(raw["left"], raw["right"])
        payload = raw_processor.update(raw["right"], raw["left"])
        for side in ("left", "right"):
            np.testing.assert_allclose(payload["hands"][side][-1], canonical[side], atol=1e-7)

    def test_rotation_and_wrist_subtraction_preserve_measured_lengths(self):
        for hand in synthetic_hand_pair():
            original = np.linalg.norm(hand[[4, 8, 12, 16, 20]] - hand[0], axis=-1)
            canonical = align_source_hand_coordinates(ensure_hand25(hand))
            np.testing.assert_allclose(np.linalg.norm(canonical[SOURCE_JOINTS["TIP_dic"]], axis=-1), original, rtol=1e-6)
            np.testing.assert_array_equal(canonical[0], np.zeros(3))

    def test_configured_source_scale_reaches_h5_model_window_and_target(self):
        left, right = synthetic_hand_pair()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["coordinate_frame"] = "l21"
                handle.attrs["coordinate_alignment"] = "source_to_l21_xyz"
                for side, points in (("left", left), ("right", right)):
                    aligned = align_source_hand_coordinates(ensure_hand25(points))
                    handle.create_dataset(f"{side}_hand_keypoints", data=np.repeat(aligned[None], 3, axis=0))
            unscaled = TwoHandH5Dataset(path).samples[0]
            scaled = TwoHandH5Dataset(path, scale_factor=2.5).samples[0]
            for side in ("left", "right"):
                self.assertTrue(scaled[f"{side}_valid"])
                self.assertEqual(scaled[f"{side}_input"].shape, (3, 25, 3))
                np.testing.assert_allclose(scaled[f"{side}_input"], unscaled[f"{side}_input"] * 2.5)
                np.testing.assert_allclose(scaled[f"{side}_target"], unscaled[f"{side}_target"] * 2.5)

    def test_fk_robot_scale_is_explicit_for_both_sides(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                urdf = getattr(L21, f"{side}_urdf")
                fk = create_hand_kinematics(urdf, L21.hand_kinematics_config(), "cpu")
                scaled_fk = create_hand_kinematics(urdf, L21.hand_kinematics_config(), "cpu", scale_factor=2.5)
                angles = torch.zeros((1, 23))
                torch.testing.assert_close(scaled_fk.forward(angles)[2], fk.forward(angles)[2] * 2.5)

    def test_absolute_distance_loss_scales_quadratically_without_unit_assumptions(self):
        left, right = synthetic_hand_pair()
        source = torch.tensor(ensure_hand25(left)[None])
        robot = torch.zeros((1, 23, 3))
        robot[:, ROBOT_JOINTS["TIP_dic"]] = torch.tensor(ensure_hand25(right)[SOURCE_JOINTS["TIP_dic"]]) * 1.2
        loss = tip_distance_loss(robot, source, torch.nn.MSELoss(), ROBOT_JOINTS, SOURCE_JOINTS)
        scaled = tip_distance_loss(robot * 2.5, source * 2.5, torch.nn.MSELoss(), ROBOT_JOINTS, SOURCE_JOINTS)
        self.assertGreater(loss.item(), 0)
        torch.testing.assert_close(scaled, loss * 2.5 ** 2)

    def test_collision_threshold_remains_in_scaled_fk_units(self):
        points = torch.tensor([[[0., 0., 0.], [.01, 0., 0.], [.015, 0., 0.]]])
        criterion = CollisionLoss(.01, ROBOT_JOINTS)
        self.assertGreater(criterion(points).item(), .1)
        torch.testing.assert_close(criterion(points * 3), torch.tensor(1e-6))


if __name__ == "__main__":
    unittest.main()
