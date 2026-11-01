import unittest
from types import SimpleNamespace

import numpy as np

from teleoperation.contracts.observations import UpperBodyObservation
from teleoperation.retargeting.arm.retargeter import ArmRetargeter


def calibration():
    return SimpleNamespace(
        metadata={
            "coordinate_source": "smplx_camera_root_relative_m",
            "units": "m",
            "human_reference": "each_side_own_shoulder",
            "robot_reference_frame": "base_Link",
            "point_order": ["shoulder", "elbow", "wrist"],
        },
        rotation=np.eye(3),
        translation=np.zeros(3),
        scale=np.ones(3),
        human_neutral_wrist={"left": np.zeros(3), "right": np.zeros(3)},
        robot_neutral_pose={"left": np.array([0., 0., 0., 0., 0., 0., 1.]),
                            "right": np.array([0., 0., 0., 0., 0., 0., 1.])},
        tool_offset_quat_xyzw={"left": np.array([0., 0., 0., 1.]),
                               "right": np.array([0., 0., 0., 1.])},
        workspace_min=np.array([-1., -1., -1.]),
        workspace_max=np.array([1., 1., 1.]),
    )


class SmplerXRetargeterTests(unittest.TestCase):
    def test_smplerx_source_is_explicit_and_uses_root_relative_wrist(self):
        retargeter = ArmRetargeter(calibration(), input_mode="smplerx", orientation_mode="fixed")
        points = np.array([[0., 0., 0.], [0., 0.2, 0.], [0.1, 0.3, 0.]])
        observation = UpperBodyObservation(
            {"left": points, "right": None}, {"left": True, "right": False},
            timestamp=1.0, coordinate_frame="smplx_image_normalized", units="normalized",
            world_arms={"left": points, "right": None},
        )
        target = retargeter.update(observation, "left")
        self.assertTrue(target.valid)
        np.testing.assert_allclose(target.pose[:3], points[2])

    def test_smplerx_calibration_source_is_required(self):
        bad = calibration()
        bad.metadata = {"coordinate_source": "other", "units": "m"}
        with self.assertRaises(ValueError):
            ArmRetargeter(bad, input_mode="smplerx")


if __name__ == "__main__":
    unittest.main()
