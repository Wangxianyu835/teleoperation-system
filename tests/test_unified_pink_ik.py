from pathlib import Path
import unittest
import importlib.util
import os
from unittest.mock import patch
import numpy as np

ASSETS = Path(os.environ.get('TELEOPERATION_ROBOTS_ROOT', 'D:/2026/code/teleoperation-system-retargeting/assets/robots'))
PATHS = {'h1_2':'h1_2/h1_2.urdf','gr1_t2':'gr1/urdf/robot.urdf', 'g1':'g1/g1_29dof_with_hand_lock_waist.urdf','tron2a_l21':'tron2a/robot.urdf'}

@unittest.skipUnless(ASSETS.is_dir() and importlib.util.find_spec('pinocchio') and importlib.util.find_spec('pink'),
                     'Requires real robot assets and pinned Pinocchio/PINK runtime')
class UnifiedPinkTests(unittest.TestCase):
    def make(self, robot):
        from teleoperation.robots.arm_spec import load_arm_spec
        from teleoperation.retargeting.arm.pink_ik import PinkArmIK
        return PinkArmIK(load_arm_spec(robot, ASSETS / PATHS[robot]))

    def test_all_models_track_real_fk_and_preserve_transaction(self):
        import pinocchio as pin
        for robot in PATHS:
            with self.subTest(robot=robot):
                ik = self.make(robot)
                original = ik.q.copy()
                goal = original.copy()
                for side in ik.spec.arm_joint_names:
                    for name in ik.spec.arm_joint_names[side][:4]:
                        j=ik.model.joints[ik.model.getJointId(name)]
                        goal[j.idx_q] += .09
                targets=ik.measure(goal)
                first=ik.prepare(targets,.02)
                np.testing.assert_array_equal(original,ik.q)
                self.assertGreater(np.linalg.norm(first.q-original),0)
                self.assertTrue(np.all(np.abs(first.q-original)<=ik.model.velocityLimit*.02+1e-9))
                ik.commit(first)
                for _ in range(100): ik.commit(ik.prepare(targets,.02))
                for side, actual in ik.measure().items():
                    self.assertLess(np.linalg.norm(actual[:3,3]-targets[side][:3,3]),.005)
                    self.assertLess(np.linalg.norm(pin.log3(actual[:3,:3].T@targets[side][:3,:3])),np.deg2rad(3))
                landmarks=ik.spec.landmark_positions()
                for side in ('left','right'):
                    self.assertGreater(np.linalg.norm(landmarks[side+'_elbow']-landmarks[side+'_shoulder']),.1)
                    self.assertGreater(np.linalg.norm(landmarks[side+'_wrist']-landmarks[side+'_elbow']),.1)

    def test_tpose_uses_actual_collinear_arm_geometry(self):
        for robot in PATHS:
            ik=self.make(robot)
            landmarks=ik.spec.landmark_positions()
            for side in ('left','right'):
                upper=landmarks[side+'_elbow']-landmarks[side+'_shoulder']
                lower=landmarks[side+'_wrist']-landmarks[side+'_elbow']
                self.assertGreater(np.dot(upper,lower)/np.linalg.norm(upper)/np.linalg.norm(lower),.8,robot)

    def test_anatomical_anchor_distinguishes_tabletop_and_pelvis_models(self):
        for robot in PATHS:
            spec=self.make(robot).spec
            self.assertEqual(spec.anatomical_anchor,'shoulder_center' if robot=='tron2a_l21' else 'pelvis')
            self.assertEqual(spec.reference_up_axis,(0.,0.,1.) if robot=='tron2a_l21' else None)

    def test_task_config_rejects_invalid_and_unimplemented_options(self):
        from teleoperation.robots.arm_spec import ArmTaskConfig
        for option in ({'position_cost':-1}, {'orientation_cost':float('nan')},
                       {'posture_cost':-1}, {'lm_damping':float('inf')},
                       {'gain':0}, {'gain':1.1}, {'elbow_task':True},
                       {'position_cost':0,'orientation_cost':0},
                       {'shared_joint_names':('torso','torso')}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                ArmTaskConfig(**option)

    def test_wrist_tpose_rotations_are_independent_actual_full_model_fk(self):
        import pinocchio as pin
        for robot in PATHS:
            ik=self.make(robot); before=ik.q.copy(); spec=ik.spec
            rotations=spec.wrist_tpose_rotations()
            q=pin.neutral(spec._model)
            for name,value in spec.tpose.items():
                q[spec._model.joints[spec._model.getJointId(name)].idx_q]=value
            data=spec._model.createData()
            pin.framesForwardKinematics(spec._model,data,q)
            for side,frame in spec.wrist_frames.items():
                expected=(data.oMf[spec._model.getFrameId(frame)].homogeneous@spec.wrist_offsets[side])[:3,:3]
                np.testing.assert_allclose(rotations[side],expected,atol=1e-12)
                np.testing.assert_allclose(rotations[side].T@rotations[side],np.eye(3),atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotations[side]),1.)
            np.testing.assert_array_equal(ik.q,before)

    def test_single_side_and_missing_side_are_exactly_held(self):
        for robot in PATHS:
            ik=self.make(robot); initial=ik.joint_values()
            goal=ik.q.copy(); goal[0]+=.15
            target=ik.measure(goal)['left']
            for _ in range(8): ik.commit(ik.prepare({'left':target},.02))
            self.assertEqual(initial['right'],ik.joint_values()['right'])
            np.testing.assert_array_equal(ik.q,ik.prepare({},.02).q)

    def test_reset_validation_failure_and_stale_candidate_preserve_state(self):
        from teleoperation.retargeting.arm.pink_ik import PinkIKError
        ik=self.make('gr1_t2'); name=ik.spec.arm_joint_names['left'][0]
        ik.reset({name: -.5}); self.assertAlmostEqual(ik.joint_values()['left'][name],-.5)
        before=ik.q.copy(); candidate=ik.prepare(ik.measure(),.02)
        for dt in (0,-1,.021,float('nan')):
            with self.assertRaises(ValueError): ik.prepare(ik.measure(),dt)
        bad=ik.measure(); bad['left'][0,0]=3
        with self.assertRaises(ValueError): ik.prepare(bad,.02)
        with patch('pink.solve_ik',side_effect=RuntimeError('injected QP failure')):
            with self.assertRaises(PinkIKError): ik.prepare(ik.measure(),.02)
        np.testing.assert_array_equal(before,ik.q)
        ik.commit(candidate)
        with self.assertRaises(ValueError): ik.commit(candidate)
        with self.assertRaises(ValueError): ik.reset({name:100})
        with self.assertRaises(ValueError): ik.reset({'nonexistent':0})

if __name__=='__main__': unittest.main()
