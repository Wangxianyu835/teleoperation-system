"""Candidate acceptance checks the actual bounded output and restores simulator state."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from teleoperation.simulation.arm_ik import PyBulletArmIK


def fixture(candidate, fk=lambda q: q, start=0., fallback=True):
    state = [start, .17]
    p = SimpleNamespace(GUI=1)
    p.getJointState = lambda *a, **kw: tuple(state)
    p.getConnectionInfo = lambda **kw: {'connectionMethod':2}
    def reset(body, index, q, targetVelocity=0., **kw):
        state[:] = [q, targetVelocity]
    p.resetJointState = reset
    a = PyBulletArmIK.__new__(PyBulletArmIK)
    a.p, a.body, a.cid = p, 1, 0
    a.arm = {'right':[(0,'joint')]}
    a.indices = [0]; a.position_of = {0:0}
    a.limits = {'lower':[-1.], 'upper':[1.], 'range':[2.], 'rest':[0.]}
    a.ik_iterations = 200; a.residual_threshold = 1e-5; a.ik_lock_others = False
    a._rest_poses = lambda posture: [0.]
    a.target_in_world = lambda position, source: tuple(position)
    a.measure = Mock(side_effect=lambda side: ((fk(state[0]),0.,0.),(0.,0.,0.,1.)))
    def solve_sequence(*args):
        a._sequence_details = {'position_fallback':fallback}
        return [candidate], (0.,0.,0.,1.)
    a._ik_sequence = solve_sequence
    a.last_target_world = {}; a.last_orientation = {}; a.last_solve = {}
    a.last_posture = {}; a.last_elbow_world = {}
    return a, state


class FallbackProgressTests(unittest.TestCase):
    def test_valid_fallback_advances_with_existing_step_bound(self):
        a,state=fixture(1.)
        command=a.solve('right',(1.,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertAlmostEqual(command['joint'],.35)
        self.assertIsNone(a.last_solve['right']['wrist_rejection'])
        self.assertTrue(a.last_solve['right']['position_fallback'])
        self.assertAlmostEqual(a.last_solve['right']['fallback_position_before_mm'],1000.)
        self.assertAlmostEqual(a.last_solve['right']['fallback_position_after_mm'],650.)
        self.assertEqual(state,[0.,.17])

    def test_good_endpoint_with_worse_bounded_step_is_rejected(self):
        a,state=fixture(1.,fk=lambda q: 1.+4*q-5*q*q)
        command=a.solve('right',(0.,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertEqual(command['joint'],0.)
        self.assertEqual(a.last_solve['right']['wrist_rejection'],'position_fallback_no_progress')
        self.assertGreater(a.last_solve['right']['fallback_position_after_mm'],1000.)
        self.assertEqual(state,[0.,.17])

    def test_identical_position_fallback_does_not_report_progress(self):
        a,_=fixture(0.)
        a.solve('right',(1.,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertEqual(a.last_solve['right']['wrist_rejection'],'position_fallback_no_progress')

    def test_invalid_or_out_of_limit_candidate_never_enters_fk_probe(self):
        for candidate in (float('nan'),float('inf'),1.1):
            with self.subTest(candidate=candidate):
                a,state=fixture(candidate)
                command=a.solve('right',(1.,0.,0.),orientation=(0.,0.,0.,1.),source='world')
                self.assertEqual(a.last_solve['right']['wrist_rejection'],'nonfinite_or_limit')
                a.measure.assert_not_called()
                self.assertEqual(command['joint'],0.)
                self.assertEqual(state,[0.,.17])

    def test_bounded_output_is_checked_again_when_start_is_outside_limits(self):
        a,state=fixture(.5,start=1.5)
        a.solve('right',(.5,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertEqual(a.last_solve['right']['wrist_rejection'],'nonfinite_or_limit')
        a.measure.assert_not_called()
        self.assertEqual(state,[1.5,.17])

    def test_failed_fk_probe_restores_position_velocity_and_published_target(self):
        a,state=fixture(1.)
        a.last_target_world['right']=(3.,0.,0.)
        a.measure.side_effect=[((0.,0.,0.),(0.,0.,0.,1.)),RuntimeError('FK failed')]
        with self.assertRaisesRegex(RuntimeError,'FK failed'):
            a.solve('right',(1.,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertEqual(state,[0.,.17])
        self.assertEqual(a.last_target_world['right'],(3.,0.,0.))

    def test_normal_pose_candidate_keeps_existing_acceptance(self):
        a,state=fixture(.2,fallback=False)
        command=a.solve('right',(.2,0.,0.),orientation=(0.,0.,0.,1.),source='world')
        self.assertEqual(command['joint'],.2)
        self.assertIsNone(a.last_solve['right']['wrist_rejection'])
        a.measure.assert_not_called()
        self.assertEqual(state,[0.,.17])


if __name__=='__main__':unittest.main()
