import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
import numpy as np

ASSETS=Path('D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models')
MODULE=Path(__file__).resolve().parents[1]/'outputs/runtime'

class ArmFitTests(unittest.TestCase):
    def test_torso_alignment_is_proper_rotation_and_rejects_degenerate_basis(self):
        from teleoperation.applications.arm_fit import torso_alignment
        source={'pelvis':np.zeros(3),'left_shoulder':np.array([1.,2,0]),'right_shoulder':np.array([-1.,2,0])}
        target={'pelvis':np.zeros(3),'left_shoulder':np.array([0.,1,2]),'right_shoulder':np.array([0.,-1,2])}
        rotation=torso_alignment(source,target)
        np.testing.assert_allclose(rotation@source['left_shoulder'],target['left_shoulder'],atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation),1.)
        shifted={**target,'pelvis':np.array([4.,3,2])}
        configured=torso_alignment(source,shifted,robot_up_axis=(0.,0.,1.))
        np.testing.assert_allclose(configured,rotation,atol=1e-10)
        target['right_shoulder']=target['left_shoulder']
        with self.assertRaises(ValueError): torso_alignment(source,target)

    def test_existing_any_output_refuses_before_loading_model(self):
        from teleoperation.applications.arm_fit import run
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'g1.json').write_text('original')
            with self.assertRaises(FileExistsError):
                run(Namespace(robots=['h1_2','g1'],output_dir=path,body_model_path=path/'missing'))
            self.assertFalse((path/'h1_2.json').exists())
            self.assertEqual((path/'g1.json').read_text(),'original')

    @unittest.skipUnless(ASSETS.exists() and MODULE.exists(),'licensed assets and SMPL-X package unavailable')
    def test_actual_robot_fit_persists_native_pose_and_wrist_frames(self):
        from teleoperation.applications.arm_fit import run,JOINT_INDICES
        from teleoperation.data.arm_fitting import load_arm_fitting
        from teleoperation.robots.arm_spec import load_arm_spec
        from teleoperation.retargeting.arm.body_model import BodyModel
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            summary=run(Namespace(robots=['g1'],output_dir=path,body_model_path=ASSETS,
                        smplx_module_dir=MODULE,num_betas=10,fit_steps=2,report=path/'report.json'))
            robot=load_arm_spec('g1')
            profile=load_arm_fitting(path/'g1.json',robot_spec_hash=robot.spec_hash)
            self.assertEqual(profile.pose_params,{})
            self.assertEqual(profile.betas.shape,(10,))
            self.assertTrue((path/'report.json').exists())
            self.assertEqual(summary['robots']['g1']['iterations'],2)
            _,rotations=BodyModel(profile.body_spec).reconstruct({},profile.betas[None])
            for side,expected in robot.wrist_tpose_rotations().items():
                actual=profile.body_to_robot@rotations[0,JOINT_INDICES[side+'_wrist']]@profile.wrist_rotation_offsets[side]
                np.testing.assert_allclose(actual,expected,atol=1e-6)

    @unittest.skipUnless(ASSETS.exists() and MODULE.exists(),'licensed assets and SMPL-X package unavailable')
    def test_actual_tabletop_robot_uses_shoulder_anchor_without_pelvis_constraint(self):
        from teleoperation.applications.arm_fit import run
        from teleoperation.data.arm_fitting import load_arm_fitting
        from teleoperation.robots.arm_spec import load_arm_spec
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            summary=run(Namespace(robots=['tron2a_l21'],output_dir=path,body_model_path=ASSETS,
                        smplx_module_dir=MODULE,num_betas=10,fit_steps=2))
            profile=load_arm_fitting(path/'tron2a_l21.json')
            self.assertEqual(profile.anchor,'shoulder_center')
            self.assertNotIn('pelvis',summary['robots']['tron2a_l21']['fit_point_names'])
            self.assertNotIn('shoulder_center',profile.segment_scaling.scales)
            points=load_arm_spec('tron2a_l21').landmark_positions()
            np.testing.assert_allclose(profile.robot_anchor,(points['left_shoulder']+points['right_shoulder'])/2)
