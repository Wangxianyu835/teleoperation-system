from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np

from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder


class RobotFinalAcceptanceTests(unittest.TestCase):
    def test_all_targets_parts_and_sides_render_only_selected_geometry(self):
        from tests.test_robot_comparison import CALIBRATION
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        calibration=ArmCalibration.load(CALIBRATION)
        for name in ('g1','h1_2','gr1_t2','tron2a_l21'):
            for parts in ('both','arms','hands'):
                for side in ('left','right','both'):
                    with self.subTest(robot=name,parts=parts,side=side):
                        group=SimulationGroup(gui=False)
                        try:
                            spec=load_robot_control_spec(name,resource_overrides={'parts':parts,'side':side})
                            b=RobotPartSceneBuilder().build(spec,group=group,calibration=calibration,geometry='simplified')
                            self.assertEqual([r['type'] for r in group.scene.robots],[name])
                            if name=='tron2a_l21':
                                self.assertEqual(set(b.part_bodies),set(spec.parts))
                            if parts!='both':
                                for body,visible in group.scene.visible_links.items():
                                    shapes=group.scene.p.getVisualShapeData(body,physicsClientId=group.scene.cid)
                                    actual={item[1] for item in shapes}
                                    expected=group.scene.visible_indices[body]
                                    self.assertTrue(actual<=expected)
                                    for index in range(group.scene.p.getNumJoints(body,physicsClientId=group.scene.cid)):
                                        if index not in expected:
                                            self.assertFalse(group.scene.p.getCollisionShapeData(body,index,physicsClientId=group.scene.cid))
                            image=group.scene.render(96,96)
                            self.assertGreater(image[:,:,:3].std(),2.)
                        finally:group.close()

    def test_explicit_two_target_hands_only_scene(self):
        group=SimulationGroup(gui=False)
        try:
            for i,name in enumerate(('g1','h1_2')):
                spec=load_robot_control_spec(name,resource_overrides={'parts':'hands','side':'both'})
                RobotPartSceneBuilder().build(spec,group=group,base_pose=(i*.6,0,.6))
            self.assertEqual([r['type'] for r in group.scene.robots],['g1','h1_2'])
            self.assertEqual(len(group.scene.visible_links),2)
            self.assertGreater(group.scene.render(160,120)[:,:,:3].std(),2.)
        finally:group.close()

    def test_vr_arm_uses_wrist_source_time_and_does_not_track_a_repeated_sample(self):
        from teleoperation.applications.robot_session import RobotControlSession
        from teleoperation.contracts.observations import RawHandFrame
        from teleoperation.inputs.wrist import vr_wrist
        group=SimulationGroup(gui=False)
        try:
            spec=load_robot_control_spec('g1',resource_overrides={'parts':'arms','side':'right'})
            b=RobotPartSceneBuilder().build(spec,group=group)
            session=RobotControlSession({'g1':b},source_kind='visionpro')
            matrix=np.eye(4);matrix[:3,3]=(.1,.2,.3)
            for i in range(12):
                stamp=10.+min(i,10)/30
                raw=RawHandFrame({'left':None,'right':None},20.+i/30,'visionpro',
                    wrists={'right':vr_wrist(matrix,stamp)})
                batch=session.next_batch(raw,i,i/30)
                session.output.submit(batch,steps=8)
            part=batch['g1'].parts['right_arm']
            self.assertFalse(part.observed)
            self.assertEqual(part.status,'holding')
            self.assertAlmostEqual(part.source_timestamp,10.+10/30)
        finally:group.close()

    def test_whole_robot_vr_h5_nn_and_optional_dex_move_all_native_targets(self):
        from teleoperation.inputs.visionpro import VisionProReplayInput
        from teleoperation.applications.robot_session import RobotControlSession
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload,save_joint_checkpoint
        from teleoperation.robots.hand_spec import load_hand_spec
        from tests.test_joint_checkpoints import SMALL
        from tests.test_unified_hand_inputs import paired_observations
        backends=('nn','dex') if importlib.util.find_spec('dex_retargeting') else ('nn',)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'vr.h5';names=('g1','h1_2','gr1_t2');weights={}
            with h5py.File(source,'w') as f:
                f.attrs['source']='visionpro';f['timestamps']=1000.+np.arange(20)/30
                for side in ('left','right'):
                    f[side+'_hand_transforms']=np.stack([paired_observations(i)[1].hands[side] for i in range(20)])
                    wrists=np.repeat(np.eye(4)[None],20,axis=0)
                    wrists[:,:3,3]=(.1,.2,.3)
                    wrists[12:,:3,3]+=np.arange(1,9)[:,None]*np.array([.001,.001,0.])
                    f[side+'_wrist_transform']=wrists
                    f[side+'_valid']=np.array([not(side=='left' and i==15) for i in range(20)])
            for name in names:
                hand=load_hand_spec(name,'right');weights[name]=root/(name+'.pth')
                save_joint_checkpoint(weights[name],shared_joint_checkpoint_payload(build_joint_model(hand,SMALL),hand))
            for method in backends:
                with self.subTest(backend=method):
                    group=SimulationGroup(gui=False);capture=VisionProReplayInput(source,'both')
                    try:
                        bindings={name:RobotPartSceneBuilder().build(load_robot_control_spec(name),group=group,base_pose=(i*2.2,0,.8)) for i,name in enumerate(names)}
                        session=RobotControlSession(bindings,backend=method,checkpoints=weights if method=='nn' else {},device='cpu',source_kind='visionpro')
                        before=session.output.read_state();last=None
                        for i in range(20):
                            raw=capture.next_observation()
                            with patch.object(group.scene.p,'setJointMotorControl2',wraps=group.scene.p.setJointMotorControl2) as motors:
                                batch=session.next_batch(raw,i,i/30)
                                self.assertEqual(motors.call_count,0)
                            if i==15:
                                for frame in batch.values():
                                    self.assertFalse(frame.parts['left_arm'].observed)
                                    self.assertFalse(frame.parts['left_hand'].observed)
                                    self.assertTrue(frame.parts['right_arm'].observed)
                            session.output.submit(batch,steps=8);last=batch
                        after=session.output.read_state()
                        for name in names:
                            for part in ('right_hand','right_arm'):
                                self.assertGreater(np.linalg.norm(after[name][part]-before[name][part]),1e-6,(method,name,part))
                            self.assertEqual(last[name].parts['right_arm'].source_timestamp,raw.wrists['right'].timestamp)
                        self.assertEqual(group.scene.frames,20)
                    finally:capture.close();group.close()
