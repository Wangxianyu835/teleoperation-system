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
    def test_rejected_arm_update_holds_last_legal_command_not_measured_overshoot(self):
        from tests.smplerx_fixtures import observation
        from teleoperation.contracts.observations import RawHandFrame
        from teleoperation.inputs.wrist import vr_wrist
        from teleoperation.applications.accepted_robot_output import AcceptedRobotOutput
        for source in ('visionpro', 'camera', 'adapter'):
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
                    elif source == 'camera': raw,_ = observation(0)
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
                                return owner.next_batch(fresh,i,i/30)['gr1_t2']
                        fresh,_=observation(i)
                        with patch('teleoperation.applications.camera_scene_arm.solve_camera_arm_frame',
                                   return_value=({},produced)):
                            return owner.next_batch(fresh,i,i/30)['gr1_t2']
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
                        if source != 'adapter':  # submit() advances physics; next_batch() does not.
                            self.assertAlmostEqual(group.scene.p.getJointState(backend.robot['body'],
                                backend.names[names[index]],physicsClientId=group.scene.cid)[0],.624)
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

    def test_source_specific_ik_matches_existing_wrist_and_camera_commands(self):
        from teleoperation.simulation.wrist_control import vr_arm_controller
        for source in ('visionpro', 'camera'):
            with self.subTest(source=source):
                reference, current = SimulationGroup(gui=False), SimulationGroup(gui=False)
                try:
                    spec = load_robot_control_spec('g1', resource_overrides={'parts':'arms', 'side':'right'})
                    new = RobotPartSceneBuilder().build(spec, group=current)
                    if source == 'visionpro':
                        reference.scene.load_native(spec)
                        driver = vr_arm_controller(reference.scene, {'right':'right'}).driver
                    else:
                        driver = RobotPartSceneBuilder().build(spec, group=reference).driver
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

    def test_registered_robot_cli_replays_body_observations_without_camera(self):
        from teleoperation.cli import main
        from tests.smplerx_fixtures import write
        import json
        with tempfile.TemporaryDirectory() as d:
            body=write(Path(d)/'body.h5',count=4)
            report=Path(d)/'report.json'
            self.assertEqual(main(['robot','control','--robot','g1','--parts','arms','--input-body',str(body),
                                   '--headless','--report',str(report),'--arm-calibrate-frames','1']),0)
            result=json.loads(report.read_text())
            self.assertEqual(result['frames'],4)
            self.assertEqual(result['physics_frames'],4)
            self.assertEqual(result['targets'],['g1'])

    def test_camera_observations_produce_complete_g1_frames_with_real_nn(self):
        from tests.smplerx_fixtures import observation
        from tests.test_joint_checkpoints import SMALL
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload,save_joint_checkpoint
        from teleoperation.robots.hand_spec import load_hand_spec
        spec=load_robot_control_spec('g1')
        group=SimulationGroup(gui=False)
        try:
            backend=RobotPartSceneBuilder().build(spec,group=group)
            with tempfile.TemporaryDirectory() as d:
                weight=Path(d)/'nn.pth'; hand=load_hand_spec('g1','right')
                save_joint_checkpoint(weight,shared_joint_checkpoint_payload(build_joint_model(hand,SMALL),hand))
                session=RobotControlSession({'g1':backend},backend='nn',checkpoints={'g1':weight},device='cpu',arm_calibrate_frames=1)
                before=session.output.read_state()
                for i in range(7):
                    raw,_=observation(i)
                    batch=session.next_batch(raw,i,i/30)
                    self.assertEqual(set(batch['g1'].parts),set(spec.parts))
                    receipt=session.output.submit(batch,steps=8)
                    self.assertEqual(list(receipt.steps.values()),[8])
                self.assertEqual(group.scene.frames,7)
                self.assertTrue(batch['g1'].parts['right_hand'].observed)
                self.assertEqual(batch['g1'].parts['right_hand'].source_timestamp,raw.hands.timestamp)
                self.assertGreater(np.linalg.norm(session.output.read_state()['g1']['right_hand']-before['g1']['right_hand']),1e-5)
        finally:group.close()
