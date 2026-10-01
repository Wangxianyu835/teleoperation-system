import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from retargeting.arm import (
    RoboticsToolboxArmIK,
    _forearm_rotation,
    load_arm_specification,
)
from retargeting.dual_teleop import export_robot_commands


URDF = Path("third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf")

# TRON2A 描述包是第三方资产，没有随仓库分发；缺失时跳过机械臂用例，
# 避免整个测试套件变红掩盖其它真实失败。补齐方式见 docs/RETARGETING_PIPELINE.md
URDF_SKIP_REASON = f"TRON2A URDF asset is missing: {URDF}"


@unittest.skipUnless(URDF.is_file(), URDF_SKIP_REASON)
class DualArmTests(unittest.TestCase):
    def test_dach_urdf_has_two_complete_seven_dof_chains(self):
        for side in ("left", "right"):
            specification = load_arm_specification(URDF, side)
            self.assertEqual(len(specification.joint_names), 7)
            self.assertEqual(specification.lower.shape, (7,))
            self.assertEqual(specification.velocity.shape, (7,))
            self.assertTrue(np.all(specification.lower < specification.upper))

    def test_ik_accepts_previous_solution_for_its_current_pose(self):
        specification = load_arm_specification(URDF, "left")
        zero = np.zeros(7)
        result = RoboticsToolboxArmIK(specification).solve(specification.robot.fkine(zero).A, zero, zero)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.q, zero)

    def test_offline_export_has_fixed_command_and_holds_invalid_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            observation = directory / "observation.h5"
            angles = directory / "angles.h5"
            calibration = directory / "calibration.json"
            output = directory / "commands.h5"
            frame_ids = np.arange(2)
            timestamps = np.array([0.0, 1.0 / 30.0])
            points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.1]], dtype=np.float32)
            with h5py.File(observation, "w") as h5_file:
                h5_file.create_dataset("frame_ids", data=frame_ids)
                h5_file.create_dataset("timestamps", data=timestamps)
                for side in ("left", "right"):
                    h5_file.create_dataset(f"{side}_arm_keypoints", data=np.stack((points, points)))
                    h5_file.create_dataset(f"{side}_arm_valid", data=np.array([True, False]))
            with h5py.File(angles, "w") as h5_file:
                h5_file.create_dataset("frame_ids", data=frame_ids)
                h5_file.create_dataset("timestamps", data=timestamps)
                for side in ("left", "right"):
                    h5_file.create_dataset(f"{side}_angles", data=np.zeros((2, 18), dtype=np.float32))
                    h5_file.create_dataset(f"{side}_valid", data=np.array([True, True]))
            human_rotation = _forearm_rotation(points)
            calibration_data = {
                "rotation": np.eye(3).tolist(), "translation": [0, 0, 0], "scale": 1.0,
                "human_neutral_wrist": {}, "robot_neutral_pose": {}, "tool_offset_quat_xyzw": {}, "safe_q": {},
                "workspace_min": [-2, -2, -2], "workspace_max": [2, 2, 2],
            }
            for side in ("left", "right"):
                specification = load_arm_specification(URDF, side)
                matrix = specification.robot.fkine(np.zeros(7)).A
                calibration_data["human_neutral_wrist"][side] = points[2].tolist()
                calibration_data["robot_neutral_pose"][side] = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()].tolist()
                calibration_data["tool_offset_quat_xyzw"][side] = Rotation.from_matrix(human_rotation.T @ matrix[:3, :3]).as_quat().tolist()
                calibration_data["safe_q"][side] = [0] * 7
            calibration.write_text(json.dumps(calibration_data), encoding="utf-8")
            export_robot_commands(observation, angles, calibration, output, URDF)
            with h5py.File(output, "r") as h5_file:
                self.assertEqual(h5_file["robot_command"].shape, (2, 48))
                self.assertEqual(h5_file["left_arm_q"].shape, (2, 7))
                self.assertEqual(h5_file["left_hand_q"].shape, (2, 17))
                self.assertFalse(h5_file["left_arm_valid"][1])
                np.testing.assert_allclose(h5_file["left_arm_q"][1], h5_file["left_arm_q"][0])


if __name__ == "__main__":
    unittest.main()
