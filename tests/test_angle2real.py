import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retargeting.config import L21
from model.kinematics import create_hand_kinematics, parse_urdf_to_joints

hand_cfg = L21.hand_kinematics_config()


REPO_ROOT = Path(__file__).resolve().parents[1]
LEFT_URDF = REPO_ROOT / "dataset" / "robot" / "l21_left" / "linkerhand_l21_left.urdf"
RIGHT_URDF = REPO_ROOT / "dataset" / "robot" / "l21_right" / "linkerhand_l21_right.urdf"

MINIMAL_URDF = """\
<?xml version="1.0"?>
<robot name="root-link-test">
  <link name="base"/>
  <link name="finger"/>
  <joint name="finger_joint" type="revolute">
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="finger"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


class UrdfParsingTests(unittest.TestCase):
    def _minimal_urdf_path(self, directory):
        path = Path(directory) / "root_link.urdf"
        path.write_text(MINIMAL_URDF, encoding="utf-8")
        return path

    def test_root_link_is_supported_as_logical_fk_root(self):
        cfg = {
            "joints_name": ["base", "finger_joint"],
            "edges": [["base", "finger_joint"]],
            "root_name": "base",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            names, parents, offsets, axes = parse_urdf_to_joints(
                self._minimal_urdf_path(temp_dir), cfg
            )

        self.assertEqual(names, cfg["joints_name"])
        self.assertEqual(parents, [-1, 0])
        np.testing.assert_allclose(offsets[0], np.zeros(6))
        np.testing.assert_allclose(axes[0], np.zeros(3))
        np.testing.assert_allclose(offsets[1][:3], [0.0, 0.0, 0.1])
        np.testing.assert_allclose(axes[1], [0.0, 1.0, 0.0])

    def test_missing_non_root_joint_is_not_silently_synthesized(self):
        cfg = {
            "joints_name": ["base", "finger_joint", "finger_tip"],
            "edges": [
                ["base", "finger_joint"],
                ["finger_joint", "finger_tip"],
            ],
            "root_name": "base",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "finger_tip"):
                parse_urdf_to_joints(self._minimal_urdf_path(temp_dir), cfg)

    def test_left_and_right_linker_fk_have_complete_finite_topology(self):
        left_fk = create_hand_kinematics(LEFT_URDF, hand_cfg, device="cpu")
        right_fk = create_hand_kinematics(RIGHT_URDF, hand_cfg, device="cpu")

        expected_names = hand_cfg["joints_name"]
        self.assertEqual(left_fk.joint_names, expected_names)
        self.assertEqual(right_fk.joint_names, expected_names)

        angles = torch.zeros((2, len(expected_names)), dtype=torch.float32)
        for hand_fk in (left_fk, right_fk):
            _, _, positions = hand_fk.forward(angles)
            self.assertEqual(positions.shape, (2, len(expected_names), 3))
            self.assertTrue(torch.isfinite(positions).all())

            index_length = torch.linalg.vector_norm(positions[:, 18] - positions[:, 3], dim=-1)
            thumb_length = torch.linalg.vector_norm(positions[:, 22] - positions[:, 17], dim=-1)
            torch.testing.assert_close(index_length, torch.full((2,), 0.044))
            torch.testing.assert_close(thumb_length, torch.full((2,), 0.026))


if __name__ == "__main__":
    unittest.main()
