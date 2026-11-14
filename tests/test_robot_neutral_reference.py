"""Reference geometry must agree across the real scene and PINK model."""
import unittest
from pathlib import Path
import numpy as np
from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
from teleoperation.robots.arm_spec import load_arm_spec


class RobotNeutralReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = ThreeRobotHandScene(sides=('left', 'right'), gui=False)
        cls.driver = ArmSceneDriver(cls.scene, sides=('left', 'right'))

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def test_forearms_face_forward_without_changing_elbow_bend(self):
        for robot, adapter in self.driver.ik.items():
            for side in ('left', 'right'):
                with self.subTest(robot=robot, side=side):
                    points = adapter.arm_points(side)
                    fore = np.asarray(adapter.to_base(points['wrist'])) - adapter.to_base(points['elbow'])
                    self.assertGreater(fore[0] / np.linalg.norm(fore), .3)
                    if robot == 'gr1_t2':
                        self.assertAlmostEqual(adapter._bend_deg(side), 111.205, delta=5.)

    def test_pinocchio_and_loaded_scene_have_same_reference(self):
        for robot, adapter in self.driver.ik.items():
            spec = load_arm_spec(robot)
            for side, names in spec.arm_joint_names.items():
                for name in names:
                    actual = self.scene.p.getJointState(adapter.body, adapter.index_of[name],
                                                       physicsClientId=self.scene.cid)[0]
                    self.assertAlmostEqual(actual, spec.reference[name], places=6)

    def test_second_ik_adapter_preserves_all_joint_positions_and_velocities(self):
        from teleoperation.simulation.arm_ik import PyBulletArmIK
        for robot, adapter in self.driver.ik.items():
            def state():
                return [self.scene.p.getJointState(adapter.body, i, physicsClientId=self.scene.cid)[:2]
                        for i in adapter.indices]
            before = state()
            PyBulletArmIK(adapter.body, adapter.cid, robot, sides=('left', 'right'))
            np.testing.assert_array_equal(state(), before)

    def test_old_gr1_fitting_is_rejected_after_reference_change(self):
        from teleoperation.data.arm_fitting import load_arm_fitting
        old = Path('D:/2026/code/teleoperation-system-retargeting/outputs/acceptance/unified-smpl-pink/fittings-v2/gr1_t2.json')
        if not old.is_file():
            self.skipTest('existing local GR1 fitting unavailable')
        with self.assertRaisesRegex(ValueError, 'robot specification identity'):
            load_arm_fitting(old, robot_spec_hash=load_arm_spec('gr1_t2').spec_hash)

    def test_unverified_gr1_asset_is_rejected_instead_of_guessing_direction(self):
        import tempfile
        from teleoperation.paths import ROBOTS_ROOT
        with tempfile.TemporaryDirectory() as folder:
            custom = Path(folder)/'robot.urdf'
            custom.write_bytes((ROBOTS_ROOT/'gr1/urdf/robot.urdf').read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'verified GR1 URDF'):
                load_arm_spec('gr1_t2', urdf_path=custom)
