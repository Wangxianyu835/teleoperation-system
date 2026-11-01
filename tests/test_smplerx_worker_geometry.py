import unittest
import numpy as np
from external.smplerx_worker.geometry import derive_body_arrays


class WorkerGeometryTests(unittest.TestCase):
    def test_projection_undoes_crop_and_positions_remain_root_relative(self):
        # Camera joints (1,2,3) and (2,2,3), camera translation z=1.
        # Virtual pixels (2.5,5) and (5,5); inverse crop doubles + (10,20).
        arrays = derive_body_arrays(
            np.array([[1., 2., 3.], [2., 2., 3.]]), np.zeros((2, 3)), [-1, 0],
            [0., 0., 1.], [10., 10.], [0., 0.], (100, 100), (100, 100),
            np.array([[2., 0., 10.], [0., 2., 20.]]), (200, 100))
        np.testing.assert_allclose(arrays['joints_root_m'], [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(arrays['image_joints'][:, :2], [[.15, .15], [.2, .15]], atol=1e-5)

    def test_root_rotation_is_composed_into_descendant_wrist(self):
        full_pose = np.zeros((3, 3))
        full_pose[0, 2] = np.pi / 2
        arrays = derive_body_arrays(
            np.zeros((3, 3)), full_pose, [-1, 0, 1], [0, 0, 1], [10, 10],
            [0, 0], (100, 100), (100, 100), np.eye(2, 3), (100, 100))
        np.testing.assert_allclose(arrays['global_rotations'][2] @ [1, 0, 0], [0, 1, 0], atol=1e-7)
