"""The controlled arm experiment must measure the same physical frames."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from devtools.gr1_arm_diagnosis import ControlledArmModel, generate_cases, pose_error, BulletSolver, PinkSolver


UPSTREAM = Path('D:/2026/code/TeleOpBench-reference')


class PoseMetricTests(unittest.TestCase):
    def test_reports_translation_and_rotation_in_physical_units(self):
        target = np.eye(4)
        actual = np.eye(4)
        actual[0, 3] = .012
        actual[:3, :3] = Rotation.from_euler('z', 7, degrees=True).as_matrix()
        position_mm, angle_deg = pose_error(actual, target)
        self.assertAlmostEqual(position_mm, 12.)
        self.assertAlmostEqual(angle_deg, 7.)

    def test_rejects_nonorthogonal_or_nonfinite_targets(self):
        for bad in (np.diag([2., 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                pose_error(np.eye(4), bad)


class ControlledKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.model = ControlledArmModel(UPSTREAM, Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.directory.cleanup()

    def test_same_link_fk_in_both_engines_for_every_generated_case(self):
        cases = generate_cases(self.model)
        self.assertEqual(len(cases), 29)
        for name, q in cases:
            for side in ('left', 'right'):
                with self.subTest(case=name, side=side):
                    pin_pose = self.model.pin_fk(q, side)
                    bullet_pose = self.model.bullet_fk(q, side)
                    mm, deg = pose_error(bullet_pose, pin_pose)
                    self.assertLessEqual(mm, .01)
                    self.assertLessEqual(np.deg2rad(deg), 1e-4)

    def test_fixture_states_stay_inside_common_limits(self):
        self.assertEqual(len(self.model.names), 14)
        for _, q in generate_cases(self.model):
            self.assertTrue(np.all(q > self.model.lower))
            self.assertTrue(np.all(q < self.model.upper))

    def test_measuring_fk_restores_position_and_velocity(self):
        before = self.model.bullet_state()
        q = generate_cases(self.model)[3][1]
        self.model.bullet_fk(q, 'left')
        np.testing.assert_array_equal(self.model.bullet_state(), before)

    def test_position_solver_tracing_preserves_every_joint_state(self):
        solver = BulletSolver(self.model, pose=False)
        self.model.set_bullet(self.model.reference)
        for name in self.model.names:
            k = self.model.indices[name]
            q = self.model.p.getJointState(self.model.body, k, physicsClientId=self.model.cid)[0]
            self.model.p.resetJointState(self.model.body, k, q, targetVelocity=.017, physicsClientId=self.model.cid)
        before = self.model.bullet_state()
        target = self.model.pin_fk(generate_cases(self.model)[2][1], 'left')
        q, details = solver.solve(self.model.reference, target, 'left')
        np.testing.assert_array_equal(self.model.bullet_state(), before)
        self.assertTrue(details['candidates'])
        mm, _ = pose_error(self.model.pin_fk(q, 'left'), target)
        self.assertLess(mm, 10)

    def test_both_solvers_keep_reached_pose_within_declared_diagnostic_gate(self):
        for solver in (BulletSolver(self.model, pose=True), PinkSolver(self.model, pose=True)):
            for side in ('left', 'right'):
                with self.subTest(solver=type(solver).__name__, side=side):
                    q = self.model.reference.copy()
                    target = self.model.pin_fk(q, side)
                    command, _ = solver.solve(q, target, side)
                    mm, deg = pose_error(self.model.pin_fk(command, side), target)
                    # This tests the plan's fixed screening gate, not exact solver identity.
                    # Bullet's null-space/pose pass can move even an already reached target.
                    self.assertLess(mm, 10.)
                    self.assertLess(deg, 5.)
                    inactive = [i for i,n in enumerate(self.model.names) if not n.startswith(side+'_')]
                    np.testing.assert_array_equal(command[inactive], q[inactive])


if __name__ == '__main__':
    unittest.main()
