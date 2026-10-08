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
)
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.apps.hand_processing import CanonicalHandPipeline
from teleoperation.retargeting.hand.topology import ensure_hand25
from teleoperation.apps.diagnostics.diagnose_hand_coordinates import signed_hand_volume, transform_diagnostics


def synthetic_hand_pair():
    left = np.zeros((21, 3), dtype=np.float32)
    bases = np.array([[.018, .008, .010], [.015, .030, 0], [0, .035, 0],
                      [-.012, .030, 0], [-.022, .023, .005]], dtype=np.float32)
    for finger, base in enumerate(bases):
        left[1 + 4 * finger:5 + 4 * finger] = np.array([1, 1.35, 1.65, 2])[:, None] * base
    right = left * np.array([-1, 1, 1], dtype=np.float32)
    return left + np.array([-.2, .4, .1]), right + np.array([.2, .4, .1])


class CoordinateContractTests(unittest.TestCase):
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
