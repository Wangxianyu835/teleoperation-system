import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.inputs.smplerx_geometry import (
    fk_global_rotations,
    smplerx_observations,
)


class SmplerXGeometryTests(unittest.TestCase):
    def test_fk_composes_parent_rotations(self):
        local = np.stack([
            np.eye(3),
            Rotation.from_euler("z", 90, degrees=True).as_matrix(),
            Rotation.from_euler("y", 90, degrees=True).as_matrix(),
        ])
        result = fk_global_rotations(local, [-1, 0, 1])
        np.testing.assert_allclose(result[0], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(result[1], local[1], atol=1e-7)
        np.testing.assert_allclose(result[2], result[1] @ local[2], atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(result[2]), 1.0, places=7)

    def test_smplerx_result_produces_body_positions_and_wrist_orientation(self):
        joints = np.zeros((6, 3), dtype=np.float64)
        joints[0:3] = [[0.0, 0.2, 0.0], [0.2, 0.1, 0.0], [0.4, 0.0, 0.0]]
        joints[3:6] = [[0.0, -0.2, 0.0], [-0.2, -0.1, 0.0], [-0.4, 0.0, 0.0]]
        rotations = np.tile(np.eye(3), (6, 1, 1))
        rotations[2] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        result = SmplerXResult(
            BodyFrameId("session", 1, 1.0), 1.01, "s32", "b" * 64,
            {"joints_root_m": joints, "global_rotations": rotations,
             "image_joints": joints.copy()},
            {"left": True, "right": True},
            metadata={"joint_indices": {
                "left": {"shoulder": 0, "elbow": 1, "wrist": 2},
                "right": {"shoulder": 3, "elbow": 4, "wrist": 5},
            }},
        )
        body, wrists = smplerx_observations(result)
        np.testing.assert_allclose(body.world_arms["left"], joints[:3])
        np.testing.assert_allclose(body.arms["right"], joints[3:6])
        self.assertEqual(body.coordinate_frame, "smplx_image_normalized")
        self.assertEqual(body.units, "normalized")
        self.assertTrue(wrists["left"].orientation_valid)
        np.testing.assert_allclose(
            Rotation.from_quat(wrists["left"].orientation_xyzw).as_matrix(), rotations[2], atol=1e-7
        )

    def test_invalid_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            fk_global_rotations(np.zeros((1, 3, 3)), [-1])


if __name__ == "__main__":
    unittest.main()
