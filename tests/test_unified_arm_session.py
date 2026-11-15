"""Accepted output is the only event that advances a PINK session state."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from teleoperation.applications.robot_session import RobotControlSession
from teleoperation.robots.control_spec import load_robot_control_spec
from tests import test_arm_pipeline as fixtures


class Group:
    group_id='test'
    time_step=1/240
    def advance(self, steps): pass
    def close(self): pass


class Backend:
    def __init__(self):
        self.spec=load_robot_control_spec('h1_2',resource_overrides={'parts':'arms'})
        self.physics_group=Group()
        self.driver=None
        self.fail=False
    def read_state(self): return {k:np.asarray(p.reference) for k,p in self.spec.parts.items()}
    def validate(self, frame): self.spec.validate_frame(frame)
    def write(self, frame):
        if self.fail: raise RuntimeError('motor write failed')
    def close(self): pass


class UnifiedArmSessionTests(unittest.TestCase):
    def session(self):
        pipeline,_=fixtures.ArmPipelineTests().make_pipeline()
        pipeline.ingest(fixtures.result(),now_s=10.01)
        backend=Backend()
        return RobotControlSession({'h1_2':backend},arm_pipeline=pipeline),backend,pipeline

    def test_prepare_and_failed_write_do_not_commit_solver_or_assembler(self):
        session,backend,pipeline=self.session()
        q=pipeline.solvers['h1_2'].q.copy()
        old=session.assembler.previous['h1_2']
        batch=session.next_batch(None,0,10.02,new_observation=False)
        np.testing.assert_array_equal(pipeline.solvers['h1_2'].q,q)
        self.assertIs(session.assembler.previous['h1_2'],old)
        backend.fail=True
        with self.assertRaisesRegex(RuntimeError,'motor write'):
            session.submit(batch,steps=2)
        np.testing.assert_array_equal(pipeline.solvers['h1_2'].q,q)
        self.assertIs(session.assembler.previous['h1_2'],old)

    def test_successful_output_commits_both_states_and_same_target_continues(self):
        session,backend,pipeline=self.session()
        batch=session.next_batch(None,0,10.02,new_observation=False)
        session.submit(batch,steps=2)
        self.assertIs(session.assembler.previous['h1_2'],batch['h1_2'])
        batch2=session.next_batch(None,1,10.03,new_observation=False)
        self.assertTrue(batch2['h1_2'].parts['left_arm'].observed)
        self.assertFalse(batch2['h1_2'].parts['left_arm'].source_is_new)
        session.submit(batch2,steps=2)
        self.assertEqual(session.assembler.frames['h1_2'],1)

    def test_prepared_batch_cannot_be_replaced_before_solver_commit(self):
        session,backend,pipeline=self.session()
        batch=session.next_batch(None,0,10.02,new_observation=False)
        with self.assertRaises(TypeError):
            batch['h1_2']=replace(batch['h1_2'],frame_id=20)
