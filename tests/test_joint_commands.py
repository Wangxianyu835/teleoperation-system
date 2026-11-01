"""Method-independent native commands enforce effective URDF constraints."""
from dataclasses import replace
import unittest

import numpy as np

from teleoperation.contracts.hand_control import HandJointResult, HandJointCommand
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline


def solution(spec, q, timestamp=0., valid=True):
    return HandJointResult(spec.hand_model, spec.side, spec.spec_hash,
                           spec.independent_joint_names, q, timestamp, valid=valid)


class JointCommandTests(unittest.TestCase):
    def test_recovery_uses_actual_short_interval_since_last_held_command(self):
        for name in ('l21','g1'):
            for mode in ('none','lowpass','kalman'):
                spec=load_hand_spec(name,'right')
                pipeline=JointCommandPipeline({spec.key:spec},filter_mode=mode,alpha=1.)
                missing=solution(spec,None,valid=False)
                pipeline.update(missing,timestamp=0.)
                held=pipeline.update(missing,timestamp=.02)
                accepted=pipeline.update(solution(spec,spec.bounds[:,1]),timestamp=.046)
                speed=np.abs(accepted.positions-held.positions)/.026
                expanded_speed=np.array([dict(zip(spec.independent_joint_names,pipeline.speeds[spec.key]))[relation[0]]*abs(relation[1])
                                         for relation in spec._relations])
                self.assertTrue(np.all(speed <= expanded_speed+1e-9),f'{name}/{mode}: {speed.max()}')

    def test_urdf_speed_limits_apply_before_coupling_expansion(self):
        for name in ("l21", "h1_2", "gr1_t2", "g1"):
            spec = load_hand_spec(name, "right")
            pipeline = JointCommandPipeline({spec.key: spec}, filter_mode="none", max_velocity=1.5)
            previous = pipeline.previous[spec.key].copy()
            command = pipeline.update(solution(spec, spec.bounds[:, 1]), timestamp=0.)
            self.assertIsInstance(command, HandJointCommand)
            self.assertEqual(command.joint_names, spec.joint_names)
            np.testing.assert_allclose(command.positions, list(spec.expand(pipeline.previous[spec.key]).values()))
            self.assertTrue(np.all(np.abs(pipeline.previous[spec.key]-previous) <= spec.velocity_limits/30+1e-12))
            if name == "l21":
                self.assertLessEqual(np.max(np.abs(pipeline.previous[spec.key]-previous)), 1/30+1e-12)

    def test_missing_recovery_and_nonincreasing_timestamp_hold_safely(self):
        spec = load_hand_spec("g1", "left")
        pipeline = JointCommandPipeline({spec.key: spec}, filter_mode="none")
        raw = solution(spec, spec.bounds[:, 1])
        first = pipeline.update(raw, timestamp=1.)
        np.testing.assert_array_equal(pipeline.update(raw, timestamp=1.).positions, first.positions)
        held = pipeline.update(solution(spec, None, valid=False), timestamp=2.)
        np.testing.assert_array_equal(held.positions, first.positions)
        before = pipeline.previous[spec.key].copy()
        pipeline.update(raw, timestamp=100.)
        self.assertTrue(np.all(np.abs(pipeline.previous[spec.key]-before) <= spec.velocity_limits/30+1e-12))

    def test_same_dimension_wrong_target_or_joint_order_is_rejected(self):
        spec = load_hand_spec("g1", "right")
        pipeline = JointCommandPipeline({spec.key: spec})
        raw = solution(spec, spec.q_reference)
        for changes in (dict(spec_hash="b"*64), dict(joint_names=tuple(reversed(spec.independent_joint_names)))):
            with self.assertRaises(ValueError):
                pipeline.update(replace(raw, **changes))


if __name__ == "__main__":
    unittest.main()
