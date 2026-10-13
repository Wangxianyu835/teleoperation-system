"""Output filtering and real L21 mesh collision regressions."""
import unittest

import numpy as np

from teleoperation.simulation.hand_commands import HandCommandPipeline, L21CollisionGuard


def bounds():
    limits = np.tile((-2., 2.), (18, 1))
    limits[0] = 0
    return {side: limits.copy() for side in ("left", "right")}


class HandCommandFilterTests(unittest.TestCase):
    def test_lowpass_matches_dex_recurrence_and_has_independent_sides(self):
        pipeline = HandCommandPipeline(("left", "right"), bounds(), alpha=0.2, max_velocity=100)
        target = np.ones(18); target[0] = 0
        first = pipeline.update("left", target, timestamp=0.)
        second = pipeline.update("left", target, timestamp=1 / 30)
        np.testing.assert_allclose(first[1:], 0.2)
        np.testing.assert_allclose(second[1:], 0.36)
        np.testing.assert_allclose(pipeline.previous["right"], 0.)
        first[:] = 99
        np.testing.assert_allclose(pipeline.previous["left"][1:], 0.36)

    def test_variable_fps_lowpass_has_same_time_constant(self):
        target = np.ones(18); target[0] = 0
        fast = HandCommandPipeline(("right",), bounds(), max_velocity=100)
        slow = HandCommandPipeline(("right",), bounds(), max_velocity=100)
        fast.update("right", target, timestamp=0.)
        slow.update("right", target, timestamp=0.)
        for index in range(1, 4):
            fast.update("right", target, timestamp=index / 30)
        np.testing.assert_allclose(fast.previous["right"], slow.update("right", target, timestamp=0.1))

    def test_startup_spikes_missing_hand_and_long_gap_are_rate_limited(self):
        for mode in ("none", "lowpass", "kalman"):
            with self.subTest(mode=mode):
                pipeline = HandCommandPipeline(("right",), bounds(), filter_mode=mode,
                                               alpha=1., max_velocity=1.5)
                target = np.full(18, 2.); target[0] = 0
                previous = np.zeros(18)
                for index in range(20):
                    raw = target if index % 2 else -target
                    current = pipeline.update("right", raw, timestamp=index / 30)
                    self.assertLessEqual(np.max(np.abs(current - previous)), 0.05 + 1e-9)
                    previous = current
                pipeline.missing("right")
                reacquired = pipeline.update("right", target, timestamp=100.)
                self.assertLessEqual(np.max(np.abs(reacquired - previous)), 0.05 + 1e-9)
                after_gap = pipeline.update("right", -target, timestamp=200.)
                self.assertLessEqual(np.max(np.abs(after_gap - reacquired)), 0.15 + 1e-9)

    def test_limits_and_fixed_placeholder_are_enforced_after_filtering(self):
        limits = bounds(); limits["right"][2] = (0., 0.3)
        pipeline = HandCommandPipeline(("right",), limits, filter_mode="kalman", max_velocity=100.)
        for index in range(30):
            value = pipeline.update("right", np.full(18, 100.), timestamp=index / 30)
            self.assertEqual(value[0], 0.)
            self.assertGreaterEqual(value[2], 0.)
            self.assertLessEqual(value[2], 0.3)
        with self.assertRaises(ValueError):
            pipeline.update("right", np.full(18, np.nan), timestamp=2.)

    def test_nonincreasing_timestamp_holds_the_last_command(self):
        pipeline = HandCommandPipeline(("right",), bounds())
        value = pipeline.update("right", np.ones(18), timestamp=1.)
        np.testing.assert_array_equal(pipeline.update("right", -np.ones(18), timestamp=1.), value)
        np.testing.assert_array_equal(pipeline.update("right", -np.ones(18), timestamp=0.), value)


class L21CollisionGuardTests(unittest.TestCase):
    def setUp(self):
        from teleoperation.applications.realtime_hand_sim import RealtimeL21Scene
        self.scene = RealtimeL21Scene(("right",), gui=False)
        self.addCleanup(self.scene.close)
        self.guard = L21CollisionGuard(self.scene, "right")

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
        p, cid, body = self.scene.p, self.scene.cid, self.scene.bodies["right"]
        self.scene.apply("right", candidate)
        self.assertIsNotNone(self.guard.violations())
        self.scene.apply("right", np.zeros(18))
        before = p.getJointStates(body, self.guard.indices, physicsClientId=cid)
        projected = self.guard.project(np.zeros(18), candidate)
        self.assertEqual(before, p.getJointStates(body, self.guard.indices, physicsClientId=cid))
        self.assertFalse(np.allclose(projected, candidate))
        self.assertGreater(projected[3], 0.1)  # feasible flexion is retained
        self.scene.apply("right", projected)
        self.assertIsNone(self.guard.violations())
        for fraction in np.linspace(0, 1, 60):
            self.scene.apply("right", projected * fraction)
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
        candidate = np.zeros(18); candidate[2] = 0.6
        projected = self.guard.project(np.zeros(18), candidate)
        self.assertLessEqual(projected[2], 0.20)
        self.assertGreater(projected[2], 0.19)


if __name__ == "__main__":
    unittest.main()
