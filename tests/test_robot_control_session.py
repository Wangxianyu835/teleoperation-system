import tempfile
from pathlib import Path
import unittest
import numpy as np
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.applications.robot_session import RobotControlSession


class RobotControlSessionTests(unittest.TestCase):
    def fitted_camera_pipeline(self,spec,backend):
        from tests.test_arm_pipeline import SyntheticBody
        from teleoperation.applications.arm_pipeline import ArmPipeline
        from teleoperation.robots.arm_spec import load_arm_spec
        from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
        arms=load_arm_spec(spec.robot_id,spec.urdf_path)
        points=arms.landmark_positions(tpose=True)
        body=SyntheticBody(points)
        profile=SimpleNamespace(betas=np.zeros(2),body_to_robot=np.eye(3),
            segment_scaling=SegmentScaling.from_tpose(points,points),robot_spec_hash=arms.spec_hash)
        actual=backend.read_state()
        named={name:value for side,names in arms.arm_joint_names.items()
               for name,value in zip(names,actual.get(side+'_arm',[arms.reference[n] for n in names]))}
        pipeline=ArmPipeline({spec.robot_id:arms},{spec.robot_id:profile},body,
            initial={spec.robot_id:named},enabled_sides={spec.robot_id:tuple(
                key.split('_')[0] for key in spec.parts if key.endswith('_arm'))})
        return pipeline,body

    def test_camera_requires_fitted_pipeline_and_rejects_body_without_raw_pose(self):
        from tests.smplerx_fixtures import observation
        group=SimulationGroup(gui=False)
        try:
            spec=load_robot_control_spec('g1',resource_overrides={'parts':'arms','side':'right'})
            backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False)
            with self.assertRaisesRegex(ValueError,'fitted'):
                RobotControlSession({'g1':backend},source_kind='camera')
            pipeline,_=self.fitted_camera_pipeline(spec,backend)
            session=RobotControlSession({'g1':backend},source_kind='camera',arm_pipeline=pipeline)
            raw,result=observation(0)
            with self.assertRaisesRegex(ValueError,'raw pose'):
                session.next_batch(replace(raw,body_result=result),0,0.,source_now_s=raw.hands.timestamp)
        finally:group.close()

    def test_camera_reuses_valid_target_then_holds_accepted_command_after_expiry(self):
        from tests.smplerx_fixtures import observation
        from tests.test_arm_pipeline import result
        group=SimulationGroup(gui=False)
        try:
            spec=load_robot_control_spec('gr1_t2',resource_overrides={'parts':'arms','side':'both'})
            backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False)
            pipeline,body=self.fitted_camera_pipeline(spec,backend)
            session=RobotControlSession({'gr1_t2':backend},source_kind='camera',arm_pipeline=pipeline)
            raw,_=observation(0)
            raw=replace(raw,body_result=result(0,stamp=10.))
            first=session.next_batch(raw,0,0.,source_now_s=10.)
            self.assertTrue(first['gr1_t2'].parts['left_arm'].observed)
            self.assertEqual(session.assembler.previous['gr1_t2'].parts['left_arm'].status,'initialized')
            session.submit(first,steps=2)
            for i in range(1,33):
                accepted=session.assembler.previous['gr1_t2'].parts['left_arm']
                joint=spec.parts['left_arm'].joint_names.index('left_wrist_roll_joint')
                group.scene.p.resetJointState(backend.robot['body'],backend.names['left_wrist_roll_joint'],
                                             .624,physicsClientId=group.scene.cid)
                batch=session.next_batch(raw,i,i*.01,new_observation=False,source_now_s=10.+i*.01)
                current=batch['gr1_t2'].parts['left_arm']
                self.assertEqual(current.source_timestamp,10.)
                self.assertFalse(current.source_is_new)
                if i<29:
                    self.assertTrue(current.observed)
                if i>30:
                    self.assertEqual(current.status,'holding')
                    np.testing.assert_array_equal(current.positions,accepted.positions)
                    self.assertNotEqual(current.positions[joint],.624)
                session.submit(batch,steps=2)
            self.assertEqual(body.calls,1)
        finally:group.close()

    def test_rejected_arm_update_holds_last_legal_command_not_measured_overshoot(self):
        from teleoperation.contracts.observations import RawHandFrame
        from teleoperation.inputs.wrist import vr_wrist
        from teleoperation.applications.accepted_robot_output import AcceptedRobotOutput
        for source in ('visionpro', 'adapter'):
            with self.subTest(source=source):
                group = SimulationGroup(gui=False)
                try:
                    spec = load_robot_control_spec('gr1_t2', resource_overrides={'parts':'arms','side':'both'})
                    backend = RobotPartSceneBuilder().build(spec, group=group)
                    if source == 'adapter':
                        owner = AcceptedRobotOutput({'gr1_t2':backend})
                    else:
                        owner = RobotControlSession({'gr1_t2':backend}, source_kind=source)
                    names = spec.parts['left_arm'].joint_names
                    index = names.index('left_wrist_roll_joint')
                    safe = list(owner.assembler.previous['gr1_t2'].parts['left_arm'].positions)
                    safe[index] = .585
                    right = list(owner.assembler.previous['gr1_t2'].parts['right_arm'].positions)
                    if source == 'visionpro':
                        matrix = np.eye(4); matrix[:3,3] = [.1,.2,.3]
                        wrists = {s:vr_wrist(matrix, 1.) for s in ('left','right')}
                        raw = RawHandFrame({'left':None,'right':None},1.,'visionpro',wrists=wrists)
                        owner.wrists['gr1_t2'].trackers = {s:SimpleNamespace(status='tracking') for s in wrists}
                    def reports(left, accepted=True):
                        return {'left':{'gr1_t2':{'joints':dict(zip(names,left)), 'accepted':accepted,
                                                 'rejection_reason':None if accepted else 'pose_position_conflict'}},
                                'right':{'gr1_t2':{'joints':dict(zip(spec.parts['right_arm'].joint_names,right)),
                                                  'accepted':True,'rejection_reason':None}}}
                    def tick(i, produced):
                        if source == 'adapter':
                            owner.submit(i,i/30,arms=produced,source_timestamp=1.+i/30)
                            return owner.assembler.previous['gr1_t2']
                        if source == 'visionpro':
                            fresh=replace(raw,timestamp=1.+i/30,wrists={
                                s:replace(w,timestamp=1.+i/30) for s,w in raw.wrists.items()})
                            with patch.object(owner.wrists['gr1_t2'],'solve_vr',return_value=produced):
                                measured=backend.read_state()['left_arm'].copy()
                                batch=owner.next_batch(fresh,i,i/30)
                                np.testing.assert_array_equal(backend.read_state()['left_arm'],measured)
                                owner.submit(batch,steps=8)
                                return batch['gr1_t2']
                    initial=owner.assembler.previous['gr1_t2'].parts['left_arm']
                    overshoot=safe.copy();overshoot[index]=.624
                    uninitialized=tick(0,reports(overshoot,False)).parts['left_arm']
                    self.assertEqual(uninitialized.status,'initialized')
                    self.assertFalse(uninitialized.observed)
                    np.testing.assert_array_equal(uninitialized.positions,initial.positions)
                    first = tick(1,reports(safe))
                    for i in (2,3):
                        overshoot = safe.copy(); overshoot[index] = .624
                        group.scene.p.resetJointState(backend.robot['body'],backend.names[names[index]],
                                                      .624,physicsClientId=group.scene.cid)
                        held = tick(i,reports(overshoot,False))
                        part = held.parts['left_arm']
                        np.testing.assert_array_equal(part.positions,first.parts['left_arm'].positions)
                        self.assertFalse(part.observed)
                        self.assertEqual(part.status,'holding')
                        self.assertEqual(part.source_timestamp,first.parts['left_arm'].source_timestamp)
                        self.assertTrue(held.parts['right_arm'].observed)
                    restored=safe.copy();restored[index]=.4
                    final=tick(4,reports(restored))
                    self.assertTrue(final.parts['left_arm'].observed)
                    self.assertAlmostEqual(final.parts['left_arm'].positions[index],.4)
                    with self.assertRaisesRegex(ValueError,'Part exceeds joint limits'):
                        tick(5,reports(overshoot,True))
                finally: group.close()

    def test_scene_report_distinguishes_rejected_solution(self):
        group=SimulationGroup(gui=False)
        try:
            spec=load_robot_control_spec('gr1_t2',resource_overrides={'parts':'arms','side':'right'})
            backend=RobotPartSceneBuilder().build(spec,group=group)
            adapter=backend.driver.ik['gr1_t2']
            q=dict(zip(spec.parts['right_arm'].joint_names,spec.parts['right_arm'].reference))
            for reason in ('pose_position_conflict','nonfinite_or_limit',None):
                adapter.last_solve['right']={'wrist_rejection':reason}
                with patch.object(adapter,'solve',return_value=q):
                    report=backend.driver.solve_each('right',{'gr1_t2':[.2,-.2,.3]})['gr1_t2']
                self.assertEqual(report['accepted'],reason is None)
                self.assertEqual(report['rejection_reason'],reason)
        finally: group.close()

    def test_visionpro_ik_matches_existing_wrist_commands(self):
        from teleoperation.simulation.wrist_control import vr_arm_controller
        for source in ('visionpro',):
            with self.subTest(source=source):
                reference, current = SimulationGroup(gui=False), SimulationGroup(gui=False)
                try:
                    spec = load_robot_control_spec('g1', resource_overrides={'parts':'arms', 'side':'right'})
                    new = RobotPartSceneBuilder().build(spec, group=current)
                    reference.scene.load_native(spec)
                    driver = vr_arm_controller(reference.scene, {'right':'right'}).driver
                    session = RobotControlSession({'g1':new}, source_kind=source)
                    adapter = driver.ik['g1']
                    position, orientation = adapter.measure('right')
                    target = np.asarray(adapter.to_base(position)) + [.1, -.1, .05]
                    expected = driver.solve_each('right', {'g1':target}, source='base', orientation=orientation)
                    actual = session.wrists['g1'].driver.solve_each('right', {'g1':target},
                        source='base', orientation=orientation)
                    names = spec.parts['right_arm'].joint_names
                    np.testing.assert_allclose([actual['g1']['joints'][n] for n in names],
                        [expected['g1']['joints'][n] for n in names], atol=1e-8, rtol=0)
                finally:
                    current.close()
                    reference.close()

    def test_camera_cli_requires_explicit_fitting_resources(self):
        from teleoperation.cli import main
        from tests.smplerx_fixtures import write
        with tempfile.TemporaryDirectory() as d:
            body=write(Path(d)/'body.h5',count=4)
            report=Path(d)/'report.json'
            with self.assertRaisesRegex(ValueError,'fitt|fit-profile|fitting'):
                main(['robot','control','--robot','g1','--parts','arms','--input-body',str(body),
                      '--headless','--report',str(report)])

    def test_camera_observations_produce_complete_g1_frames_with_real_nn(self):
        from tests.smplerx_fixtures import observation
        from tests.test_joint_checkpoints import SMALL
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload,save_joint_checkpoint
        from teleoperation.robots.hand_spec import load_hand_spec
        spec=load_robot_control_spec('g1')
        group=SimulationGroup(gui=False)
        try:
            backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False)
            arms,body=self.fitted_camera_pipeline(spec,backend)
            with tempfile.TemporaryDirectory() as d:
                weight=Path(d)/'nn.pth'; hand=load_hand_spec('g1','right')
                save_joint_checkpoint(weight,shared_joint_checkpoint_payload(build_joint_model(hand,SMALL),hand))
                session=RobotControlSession({'g1':backend},backend='nn',checkpoints={'g1':weight},device='cpu',arm_pipeline=arms)
                before=session.output.read_state()
                for i in range(7):
                    raw,_=observation(i)
                    from tests.test_arm_pipeline import result
                    raw=replace(raw,body_result=result(i,stamp=raw.hands.timestamp))
                    batch=session.next_batch(raw,i,i*.01,source_now_s=raw.hands.timestamp)
                    self.assertEqual(set(batch['g1'].parts),set(spec.parts))
                    receipt=session.submit(batch,steps=8)
                    self.assertEqual(list(receipt.steps.values()),[8])
                self.assertEqual(group.scene.frames,7)
                self.assertTrue(batch['g1'].parts['right_hand'].observed)
                self.assertEqual(batch['g1'].parts['right_hand'].source_timestamp,raw.hands.timestamp)
                self.assertGreater(np.linalg.norm(session.output.read_state()['g1']['right_hand']-before['g1']['right_hand']),1e-5)
        finally:group.close()
