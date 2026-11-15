"""The controlled arm experiment must measure the same physical frames."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from devtools.gr1_arm_diagnosis import (ControlledArmModel, generate_cases, pose_error, DEPENDENCIES,
                                       BulletSolver, PinkSolver, measure_candidate, candidate_status)


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

    def test_distinguishes_invalid_and_out_of_limit_candidates(self):
        low, high = np.array([-1., -1.]), np.array([1., 1.])
        self.assertEqual(candidate_status(np.array([0., .5]), low, high), 'within_limits')
        self.assertEqual(candidate_status(np.array([0., 1.1]), low, high), 'out_of_limits')
        self.assertEqual(candidate_status(np.array([np.nan, 0.]), low, high), 'nonfinite')


@unittest.skipUnless((UPSTREAM / 'assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf').is_file()
                     and importlib.util.find_spec('pinocchio') is not None
                     and importlib.util.find_spec('pybullet') is not None,
                     'Controlled FK tests require the external pinned GR1 model, Pinocchio and PyBullet')
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

    @unittest.skipUnless((DEPENDENCIES / 'pink/__init__.py').is_file(), 'Isolated PINK diagnostic dependencies required')
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

    def test_native_candidate_measurement_preserves_all_positions_and_velocities(self):
        solver = BulletSolver(self.model, pose=True)
        a = solver.adapter
        self.model.set_bullet(self.model.reference)
        before = self.model.bullet_state()
        candidate = [before[self.model.names.index(a.name_of[k]), 0] + .02 for k in a.indices]
        t = measure_candidate(a, 'left', candidate)
        self.assertEqual(t.shape, (4, 4))
        np.testing.assert_array_equal(self.model.bullet_state(), before)

    def test_far_target_is_reported_as_not_reached_not_proven_unreachable(self):
        solver = BulletSolver(self.model, pose=True)
        target = self.model.pin_fk(self.model.reference, 'left')
        target[:3, 3] = [10., 10., 10.]
        q, details = solver.solve(self.model.reference, target, 'left')
        mm, _ = pose_error(self.model.pin_fk(q, 'left'), target)
        self.assertGreater(mm, 1000.)
        self.assertEqual(candidate_status(q,self.model.lower,self.model.upper),'within_limits')

    def test_known_reachable_stall_makes_progress_without_counterfactual_override(self):
        # Fixed-seed case from the wide-range diagnosis, reduced to its right-arm target.
        goal=self.model.reference.copy()
        goal[7:]=[-.794696868144565,-.9830452799871301,-1.0808504028971584,
                  1.342007358718132,1.7881337663923755,-.06845737307204447,.19986256320259999]
        target=self.model.pin_fk(goal,'right')
        q=self.model.reference.copy()
        solver=BulletSolver(self.model,pose=True)
        before=self.model.bullet_state()
        initial_mm,_=pose_error(self.model.pin_fk(q,'right'),target)
        self.assertGreater(initial_mm,160.)
        accepted_fallbacks=0
        for _ in range(10):
            next_q,detail=solver.solve(q,target,'right')
            d=detail['production']
            np.testing.assert_array_equal(self.model.bullet_state(),before)
            self.assertLessEqual(np.max(np.abs(next_q-q)),.35+1e-9)
            if d['position_fallback'] and d['wrist_rejection'] is None:
                accepted_fallbacks+=1
                self.assertLess(d['fallback_position_after_mm'],d['fallback_position_before_mm'])
            q=next_q
        self.assertGreater(accepted_fallbacks,0)
        final_mm,_=pose_error(self.model.pin_fk(q,'right'),target)
        self.assertLess(final_mm,10.)


if __name__ == '__main__':
    unittest.main()
