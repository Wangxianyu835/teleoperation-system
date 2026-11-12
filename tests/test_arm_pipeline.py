"""Application contracts: rebuild on input, integrate on control, commit on receipt."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from teleoperation.applications.arm_pipeline import ArmPipeline
from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
from teleoperation.robots.arm_spec import load_arm_spec


class SyntheticBody:
    """A deterministic skeleton with counted expensive forward passes."""
    spec = SimpleNamespace(num_betas=2, hash='synthetic')

    def __init__(self, points):
        self.points = points
        self.calls = 0

    def reconstruct(self, pose_params, betas):
        self.calls += 1
        points = np.zeros((len(betas), 22, 3))
        indices = {'pelvis': 0, 'head': 15, 'left_shoulder': 16,
                   'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
                   'left_wrist': 20, 'right_wrist': 21}
        for name, index in indices.items():
            points[:, index] = self.points.get(name, np.zeros(3))
        return points, np.tile(np.eye(3), (len(betas), 22, 1, 1))


def result(sequence=0, stamp=10., valid=True):
    return SmplerXResult(BodyFrameId('test', sequence, stamp), stamp+.01,
                        'synthetic', 'synthetic',
                        {'smplx_root_pose': np.zeros(3), 'smplx_body_pose': np.zeros(63)},
                        {'left': valid, 'right': valid})


class ArmPipelineTests(unittest.TestCase):
    def test_shared_joints_require_a_supported_output_command_contract(self):
        from dataclasses import replace
        pipeline,_=self.make_pipeline()
        original=pipeline.specs['h1_2']
        shared=next(name for name in original._model.names[1:]
                    if name not in {n for names in original.arm_joint_names.values() for n in names})
        spec=replace(original,task_config=replace(original.task_config,shared_joint_names=(shared,)))
        with self.assertRaisesRegex(ValueError,'[Ss]hared.*output'):
            ArmPipeline({'h1_2':spec},pipeline.profiles,pipeline.body_model)

    def test_source_clock_age_maps_to_control_expiry_without_duplicate_refresh(self):
        pipeline,body=self.make_pipeline()
        self.assertTrue(pipeline.ingest(result(5,stamp=1000.),now_s=1000.05,control_time=0.))
        self.assertAlmostEqual(pipeline._expires,.25)
        self.assertFalse(pipeline.ingest(result(5,stamp=1000.),now_s=1000.1,control_time=.2))
        self.assertAlmostEqual(pipeline._expires,.25)
        proposal=pipeline.prepare(.24,dt=.01)
        self.assertIn('h1_2',proposal)
        self.assertEqual(proposal['h1_2'].source_time,1000.)
        self.assertAlmostEqual(proposal['h1_2'].expires_time,.25)
        pipeline.discard()
        self.assertFalse(pipeline.prepare(.26,dt=.01))
        self.assertEqual(body.calls,1)

    def make_pipeline(self):
        spec = load_arm_spec('h1_2')
        points = spec.landmark_positions(tpose=True)
        body = SyntheticBody(points)
        profile = SimpleNamespace(betas=np.zeros(2), body_to_robot=np.eye(3),
            segment_scaling=SegmentScaling.from_tpose(points, points),
            robot_spec_hash=spec.spec_hash)
        return ArmPipeline({'h1_2': spec}, {'h1_2': profile}, body), body

    def test_one_observation_rebuilds_once_across_repeated_control_ticks(self):
        pipeline, body = self.make_pipeline()
        self.assertTrue(pipeline.ingest(result(), now_s=10.01))
        for index in range(3):
            pipeline.prepare(10.02+index*.01, dt=.01)
            pipeline.commit()
        self.assertEqual(body.calls, 1)
        self.assertFalse(pipeline.ingest(result(), now_s=10.05))
        self.assertEqual(body.calls, 1)

    def test_expired_target_holds_and_uncommitted_proposal_does_not_advance(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest(result(), now_s=10.01)
        before = pipeline.solvers['h1_2'].q.copy()
        pipeline.prepare(10.02, dt=.01)
        np.testing.assert_array_equal(pipeline.solvers['h1_2'].q, before)
        pipeline.discard()
        held = pipeline.prepare(11., dt=.01)
        self.assertFalse(held)
        np.testing.assert_array_equal(pipeline.solvers['h1_2'].q, before)

    def test_failed_reconstruction_cannot_refresh_previous_target_metadata(self):
        pipeline, body = self.make_pipeline()
        pipeline.ingest(result(), now_s=10.01)
        old_targets=pipeline.targets
        snapshot=(pipeline._session,pipeline._sequence,pipeline._source_time,pipeline._expires,dict(pipeline._valid))
        with patch.object(body,'reconstruct',side_effect=RuntimeError('invalid geometry')):
            with self.assertRaises(RuntimeError):
                pipeline.ingest(result(1,stamp=10.1),now_s=10.11)
        self.assertEqual(snapshot,(pipeline._session,pipeline._sequence,pipeline._source_time,pipeline._expires,dict(pipeline._valid)))
        self.assertIs(pipeline.targets,old_targets)
        self.assertFalse(pipeline.prepare(10.35,dt=.01))
        self.assertTrue(pipeline.ingest(result(1,stamp=10.1),now_s=10.12))

    def test_enabled_sides_hold_other_arm_with_bilateral_fitting_identity(self):
        initial,_=self.make_pipeline()
        pipeline=ArmPipeline(initial.specs,initial.profiles,initial.body_model,
                             enabled_sides={'h1_2':('left',)})
        right=pipeline.solvers['h1_2'].joint_values()['right']
        pipeline.ingest(result(),now_s=10.01)
        for index in range(4):
            updates=pipeline.prepare(10.02+index*.01,dt=.01)
            self.assertEqual(updates['h1_2'].active_sides,('left',))
            pipeline.commit()
        self.assertEqual(right,pipeline.solvers['h1_2'].joint_values()['right'])

    def test_profile_body_identity_must_match_reconstruction_model(self):
        pipeline,_=self.make_pipeline()
        profile=pipeline.profiles['h1_2']
        profile.body_spec_hash='wrong-model'
        with self.assertRaises(ValueError):
            ArmPipeline(pipeline.specs,pipeline.profiles,pipeline.body_model)

    def test_older_or_future_sample_cannot_replace_active_target(self):
        pipeline, body = self.make_pipeline()
        pipeline.ingest(result(2), now_s=10.01)
        self.assertFalse(pipeline.ingest(result(1), now_s=10.02))
        self.assertFalse(pipeline.ingest(result(3, stamp=20.), now_s=10.02))
        self.assertEqual(body.calls, 1)


if __name__ == '__main__':
    unittest.main()
