"""Output filtering and real L21 mesh collision regressions."""
import unittest

import numpy as np

from teleoperation.simulation.hand_commands import L21CollisionGuard
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline


def specs():
    from dataclasses import replace
    from teleoperation.robots.hand_spec import load_hand_spec
    return {('l21',side):replace(load_hand_spec('l21',side), joint_limits=((-2.,2.),)*17,
                                joint_velocities=(100.,)*17, palm_frames=()) for side in ('left','right')}


def result(pipeline, side, values):
    from teleoperation.contracts.hand_control import HandJointResult
    spec = pipeline.specs[('l21',side)]
    return HandJointResult(*spec.key,spec.spec_hash,spec.independent_joint_names,values,0.)


class HandCommandFilterTests(unittest.TestCase):
    def test_lowpass_matches_dex_recurrence_and_has_independent_sides(self):
        pipeline = JointCommandPipeline(specs(), alpha=.2, max_velocity=100)
        raw = result(pipeline,'left',np.ones(17))
        first = pipeline.update(raw,timestamp=0.)
        second = pipeline.update(raw,timestamp=1/30)
        np.testing.assert_allclose(first.positions,.2)
        np.testing.assert_allclose(second.positions,.36)
        np.testing.assert_allclose(pipeline.previous[('l21','right')],0.)
        self.assertFalse(first.positions.flags.writeable)

    def test_variable_fps_lowpass_has_same_time_constant(self):
        fast = JointCommandPipeline(specs(),max_velocity=100)
        slow = JointCommandPipeline(specs(),max_velocity=100)
        raw = result(fast,'right',np.ones(17))
        fast.update(raw,timestamp=0.); slow.update(raw,timestamp=0.)
        for i in range(1,4): fast.update(raw,timestamp=i/30)
        np.testing.assert_allclose(fast.previous[('l21','right')],slow.update(raw,timestamp=.1).positions)

    def test_startup_spikes_missing_hand_and_long_gap_are_rate_limited(self):
        for mode in ('none','lowpass','kalman'):
            pipeline = JointCommandPipeline(specs(),filter_mode=mode,alpha=1.,max_velocity=1.5)
            before = np.zeros(17)
            for i in range(20):
                current = pipeline.update(result(pipeline,'right',np.full(17,2. if i%2 else -2.)),timestamp=i/30).positions
                self.assertLessEqual(np.max(np.abs(current-before)),.05+1e-9)
                before=current
            pipeline.missing(('l21','right'),timestamp=20/30)
            current=pipeline.update(result(pipeline,'right',np.full(17,2.)),timestamp=100.).positions
            self.assertLessEqual(np.max(np.abs(current-before)),.05+1e-9)
            after=pipeline.update(result(pipeline,'right',np.full(17,-2.)),timestamp=200.).positions
            self.assertLessEqual(np.max(np.abs(after-current)),.15+1e-9)

    def test_limits_are_enforced_after_filtering(self):
        pipeline=JointCommandPipeline(specs(),filter_mode='kalman',max_velocity=100.)
        for i in range(30):
            current=pipeline.update(result(pipeline,'right',np.full(17,100.)),timestamp=i/30).positions
            self.assertTrue(np.all(current <= 2.))
        with self.assertRaises(ValueError): result(pipeline,'right',np.full(17,np.nan))

    def test_nonincreasing_timestamp_holds_the_last_command(self):
        pipeline=JointCommandPipeline(specs())
        first=pipeline.update(result(pipeline,'right',np.ones(17)),timestamp=1.).positions
        for timestamp in (1.,0.):
            np.testing.assert_array_equal(pipeline.update(result(pipeline,'right',-np.ones(17)),timestamp=timestamp).positions,first)


class L21CollisionGuardTests(unittest.TestCase):
    def setUp(self):
        from teleoperation.applications.realtime_hand_sim import RealtimeL21Scene
        self.scene = RealtimeL21Scene(("right",), gui=False)
        self.addCleanup(self.scene.close)
        self.guard = L21CollisionGuard(self.scene, "right")

    def set_pose(self, positions):
        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.contracts.hand_control import HandJointCommand
        spec = load_hand_spec('l21', 'right')
        self.scene.apply_native(HandJointCommand(*spec.key, spec.spec_hash, spec.joint_names,
                                                positions, 0.))

    def test_baseline_metacarpal_contacts_do_not_disable_the_guard(self):
        self.assertGreater(len(self.guard.pairs), 100)
        self.assertEqual(len(self.guard.baseline_contacts), 3)
        self.assertIsNone(self.guard.violations())

    def test_colliding_target_is_projected_and_query_restores_live_state(self):
        # Seeded in-limit pose known to deepen neighboring-finger mesh contacts.
        candidate = np.array([0., -.1491663, .3717925, 1.258001, .0295783, .147782,
                              .680009, -.0075415, .2507901, 1.1532861, -.1390781,
                              .6142283, .8112821, -.0832464, .9388777, .7378378,
                              1.5013396, .4461958])
        # thumb_cmc_roll is fixed in this URDF.
        candidate[13] = 0.
        candidate = candidate[1:]
        p, cid, body = self.scene.p, self.scene.cid, self.scene.bodies["right"]
        self.set_pose(candidate)
        self.assertIsNotNone(self.guard.violations())
        self.set_pose(np.zeros(17))
        before = p.getJointStates(body, self.guard.indices, physicsClientId=cid)
        projected = self.guard.project(np.zeros(17), candidate)
        self.assertEqual(before, p.getJointStates(body, self.guard.indices, physicsClientId=cid))
        self.assertFalse(np.allclose(projected, candidate))
        self.assertGreater(projected[2], 0.1)  # feasible flexion is retained
        self.set_pose(projected)
        self.assertIsNone(self.guard.violations())
        for fraction in np.linspace(0, 1, 60):
            self.set_pose(projected * fraction)
            self.assertIsNone(self.guard.violations())

    def test_clear_endpoints_do_not_skip_an_unsafe_intermediate_pose(self):
        # Controlled collision interval on a real Bullet joint demonstrates that
        # endpoint-only checks would wrongly accept this segment.
        index = self.scene.joint_index["right"]["index_mcp_pitch"]
        def contact():
            angle = self.scene.p.getJointState(self.scene.bodies["right"], index,
                                               physicsClientId=self.scene.cid)[0]
            return {"distance_m": -0.001} if 0.20 < angle < 0.40 else None
        self.guard.violations = contact
        candidate = np.zeros(17); candidate[1] = 0.6
        projected = self.guard.project(np.zeros(17), candidate)
        self.assertLessEqual(projected[1], 0.20)
        self.assertGreater(projected[1], 0.19)


if __name__ == "__main__":
    unittest.main()
