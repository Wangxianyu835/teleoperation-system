import os, unittest, tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS=Path(os.environ.get('SMPL_MODEL_PATH', 'D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models'))
MODULE=os.environ.get('SMPLX_MODULE_DIR',str(Path(__file__).resolve().parents[1]/'outputs'/'runtime'))

class GeometryTests(unittest.TestCase):
    def test_asymmetric_scaling_preserves_articulation_and_roundtrips(self):
        from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
        human=dict(pelvis=np.zeros(3),left_shoulder=np.array([0,1,1.]),right_shoulder=np.array([0,-1,1.]),left_elbow=np.array([0,2,1.]),right_elbow=np.array([0,-2,1.]),left_wrist=np.array([0,3,1.]),right_wrist=np.array([0,-3,1.]))
        robot={k:v.copy() for k,v in human.items()}
        robot['left_elbow'][1]=3; robot['left_wrist'][1]=6
        scaling=SegmentScaling.from_tpose(human,robot)
        result=SegmentScaling.from_dict(scaling.to_dict()).apply(human)
        for name in human: np.testing.assert_allclose(result[name],robot[name])
        human['left_wrist']=human['left_elbow']+np.array([1.,0,0])
        result=scaling.apply(human)
        np.testing.assert_allclose(result['left_wrist']-result['left_elbow'],[3,0,0])

    def test_shoulder_anchored_scaling_needs_no_artificial_pelvis_segment(self):
        from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
        points=dict(left_shoulder=np.array([0.,1,0]),right_shoulder=np.array([0.,-1,0]),
                    left_elbow=np.array([0.,2,0]),right_elbow=np.array([0.,-2,0]),
                    left_wrist=np.array([0.,3,0]),right_wrist=np.array([0.,-3,0]))
        target={k:v*2 for k,v in points.items()}
        scales=SegmentScaling.from_tpose(points,target,anchor='shoulder_center')
        self.assertNotIn('shoulder_center',scales.scales)
        self.assertEqual(scales.anchor,'shoulder_center')
        for key,point in SegmentScaling.from_dict(scales.to_dict()).apply(points).items():
            np.testing.assert_allclose(point,target[key])

    @unittest.skipUnless(ASSETS.exists(), 'licensed local SMPL-X assets unavailable')
    def test_real_model_dimensions_root_pose_and_gradient_fit(self):
        from teleoperation.retargeting.arm.body_model import BodyModel,BodyModelSpec,reference_pose
        from teleoperation.retargeting.arm.shape_fitting import fit_shape
        import torch
        for count in (10,20):
            model=BodyModel(BodyModelSpec(str(ASSETS), num_betas=count,module_dir=MODULE))
            joints,rot=model.reconstruct({},np.zeros((1,count)))
            self.assertEqual(joints.shape,(1,55,3)); self.assertEqual(rot.shape,(1,55,3,3))
            np.testing.assert_allclose(joints[:,0],0,atol=1e-7)
            np.testing.assert_allclose(rot[0,0],np.eye(3),atol=1e-7)
            turned,rotated=model.reconstruct(reference_pose(),np.zeros((1,count)))
            expected=Rotation.from_euler('xyz',[np.pi/2,0,np.pi/2]).as_matrix()
            np.testing.assert_allclose(turned[0],joints[0]@expected.T,atol=1e-6)
            np.testing.assert_allclose(rotated[0,0],expected,atol=1e-6)
            native=model.layer(betas=torch.zeros((1,count)),global_orient=torch.tensor(reference_pose()['global_orient'][None],dtype=torch.float32),return_verts=False)
            native_joints=native.joints[:,:55].detach().cpu().numpy()
            np.testing.assert_allclose(turned,native_joints-native_joints[:,:1],atol=2e-6)
            params={'smplx_body_pose':np.linspace(-.2,.2,63),'smplx_lhand_pose':np.linspace(-.1,.1,45)}
            articulated,_=model.reconstruct(params,np.zeros((1,count)))
            native=model.layer(betas=torch.zeros((1,count)),body_pose=torch.tensor(params['smplx_body_pose'][None],dtype=torch.float32),left_hand_pose=torch.tensor(params['smplx_lhand_pose'][None],dtype=torch.float32),return_verts=False)
            native_joints=native.joints[:,:55].detach().cpu().numpy()
            np.testing.assert_allclose(articulated,native_joints-native_joints[:,:1],atol=2e-6)
            with self.assertRaises(ValueError): model.reconstruct({},np.zeros((1,count+1)))
            shape_batch=np.stack([np.linspace(-3+i,4+i,count) for i in range(4)])
            batch_joints,batch_rotations=model.reconstruct(params,shape_batch)
            for index in range(4):
                single_joints,single_rotations=model.reconstruct(params,shape_batch[index:index+1])
                np.testing.assert_array_equal(batch_joints[index],single_joints[0])
                np.testing.assert_array_equal(batch_rotations[index],single_rotations[0])
            beta=torch.zeros((1,count),requires_grad=True)
            model.joints_tensor(beta).square().sum().backward()
            self.assertGreater(float(beta.grad.abs().sum()),0)
            if count==10:
                truth=np.zeros((1,count));truth[0,0]=1
                indices=(0,12,16,17,18,19,20,21)
                target=model.reconstruct(reference_pose(),truth)[0][0,list(indices)]
                fit=fit_shape(model,target,indices,steps=80,learning_rate=.08)
                self.assertLess(fit.final_rms,fit.initial_rms*.15)
                self.assertEqual(fit.betas.shape,(count,))
                relative_target=target-target[[2,3]].mean(0)
                anchored=fit_shape(model,relative_target,indices,steps=80,learning_rate=.08,anchor_indices=(16,17))
                self.assertLess(anchored.final_rms,anchored.initial_rms*.15)
                from teleoperation.data.arm_fitting import save_arm_fitting,load_arm_fitting
                from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'fit.json'
                    save_arm_fitting(path,body_spec=model.spec,robot_spec_hash='a'*64,fit_result=fit,segment_scaling=SegmentScaling({'shoulder_center':1.}),body_to_robot=expected,pose_params=reference_pose())
                    data=load_arm_fitting(path,body_spec=model.spec,robot_spec_hash='a'*64)
                    self.assertEqual(len(data.betas),count)
                    np.testing.assert_allclose(data.body_to_robot,expected)
                    np.testing.assert_allclose(data.pose_params['global_orient'],reference_pose()['global_orient'])
                    self.assertEqual(data.body_spec.num_betas,count)
                    np.testing.assert_allclose(data.wrist_rotation_offsets['left'],np.eye(3))
                    with self.assertRaises(FileExistsError): save_arm_fitting(path,body_spec=model.spec,robot_spec_hash='a'*64,fit_result=fit,segment_scaling=SegmentScaling({'shoulder_center':1.}))
                    with self.assertRaises(ValueError): load_arm_fitting(path,robot_spec_hash='b'*64)
        with self.assertRaises(ValueError): BodyModel(BodyModelSpec(str(ASSETS),num_betas=301,module_dir=MODULE)).layer
