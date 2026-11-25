"""Regress the actual demo assembly, not just successful URDF loading."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from teleoperation.applications import tron2a_l21_demo as demo
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.urdf_loader import prepare_tron2a_urdf

URDF = Path(__file__).parent / 'fixtures' / 'tron2a.urdf'


class Tron2AL21MountTests(unittest.TestCase):
    def test_demo_replaces_grippers_and_points_both_hands_out_of_wrists(self):
        original = RobotPartSceneBuilder.build
        checked = []

        def check(builder, spec, **kwargs):
            backend = original(builder, spec, **kwargs)
            p, cid = backend.scene.p, backend.scene.cid
            main = backend.robot['body']
            lowest = min(p.getAABB(main, i, physicsClientId=cid)[0][2]
                         for i in range(-1, p.getNumJoints(main, physicsClientId=cid)))
            self.assertGreaterEqual(lowest, .029)
            for side in ('left', 'right'):
                index = backend.robot['ee_indices'][side]
                visuals = [v for v in p.getVisualShapeData(main, physicsClientId=cid) if v[1] == index]
                self.assertEqual(visuals, [], 'Original gripper visual is still loaded')
                self.assertFalse(p.getCollisionShapeData(main, index, physicsClientId=cid))
                self.assertEqual(p.getDynamicsInfo(main, index, physicsClientId=cid)[0], 0.)
                ee = p.getLinkState(main, index, computeForwardKinematics=True, physicsClientId=cid)
                body = backend.part_bodies[side + '_hand']
                for visual in p.getVisualShapeData(body, physicsClientId=cid):
                    if visual[2] == p.GEOM_MESH:
                        mesh = Path(visual[4].decode())
                        self.assertTrue(mesh.is_absolute(), 'Relative mesh names alias left/right in Bullet cache')
                        self.assertEqual(mesh.parent.parent.name, side)
                inertia = p.getDynamicsInfo(body, -1, physicsClientId=cid)
                root = p.multiplyTransforms(*p.getBasePositionAndOrientation(body, physicsClientId=cid),
                                            *p.invertTransform(inertia[3], inertia[4]))
                np.testing.assert_allclose(root[0], ee[4], atol=1e-6)
                distal = p.rotateVector(ee[5], [0, 0, -1])
                fingers = p.rotateVector(root[1], [0, 0, 1])
                self.assertGreater(np.dot(distal, fingers), .9999, side)
                checked.append(side)
            return backend

        with patch.object(RobotPartSceneBuilder, 'build', check):
            self.assertEqual(demo.main(['--urdf', str(URDF), '--headless', '--static', '--seconds', '.02']), 0)
        self.assertEqual(checked, ['left', 'right'])

    def test_replacement_keeps_kinematic_frames_and_original_asset_untouched(self):
        source = URDF.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_tron2a_urdf(URDF, Path(directory), display='simplified', replace_grippers=True)
            before, after = ET.fromstring(source), ET.parse(prepared).getroot()
            self.assertEqual([ET.tostring(j) for j in before.findall('joint')],
                             [ET.tostring(j) for j in after.findall('joint')])
            for side in ('L', 'R'):
                link = after.find(f"link[@name='grasper_{side}_Link']")
                self.assertIsNotNone(link)
                self.assertFalse(link.findall('visual') + link.findall('collision'))
                self.assertEqual(float(link.find('inertial/mass').get('value')), 0.)
        self.assertEqual(URDF.read_bytes(), source)

    def test_new_neutral_calibration_uses_model_mount_instead_of_identity(self):
        from tests.test_mediapipe_tron2a import synthetic_calibration
        from scipy.spatial.transform import Rotation
        calibration = synthetic_calibration()
        for side in ('left', 'right'):
            mount = calibration.hand_mount_pose[side]
            np.testing.assert_allclose(mount[:3], 0.)
            np.testing.assert_allclose(Rotation.from_quat(mount[3:]).apply([0, 0, 1]), [0, 0, -1], atol=1e-8)

    def test_marked_legacy_placeholder_migrates_without_rewriting_custom_mounts(self):
        from tests.test_mediapipe_tron2a import synthetic_calibration
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        data = synthetic_calibration().to_dict()
        data['metadata']['hand_mount_source'] = 'identity_placeholder_not_calibrated'
        data['hand_mount_pose']['left'] = [0, 0, 0, 0, 0, 0, 1]
        data['hand_mount_pose']['right'] = [.01, .02, .03, 0, 0, 0, 1]
        loaded = ArmCalibration.from_dict(data)
        np.testing.assert_allclose(loaded.hand_mount_pose['left'], [0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(loaded.hand_mount_pose['right'], data['hand_mount_pose']['right'])
        self.assertEqual(data['hand_mount_pose']['left'], [0, 0, 0, 0, 0, 0, 1])
        self.assertIn('hand_mount_migrated_from', loaded.metadata)
        data['metadata']['hand_mount_source'] = 'measured'
        np.testing.assert_allclose(ArmCalibration.from_dict(data).hand_mount_pose['left'], data['hand_mount_pose']['left'])

    def test_full_mesh_motion_keeps_flange_constraint_and_ground_clearance(self):
        from teleoperation.paths import DEFAULT_TRON2A_URDF
        from teleoperation.simulation.robot_output import RobotOutput
        source = next((p for p in (URDF.parents[2] / 'assets/Tron2a/urdf/robot.urdf',
                                   DEFAULT_TRON2A_URDF) if p.is_file()), None)
        if source is None:
            self.skipTest('Full TRON2A meshes are not installed')
        original = RobotOutput.submit
        maxima = {'left': 0., 'right': 0.}

        def submit(output, batch, **kwargs):
            receipt = original(output, batch, **kwargs)
            b = output.backends['tron2a_l21']
            p, cid = b.scene.p, b.scene.cid
            for side in maxima:
                index, body = b.robot['ee_indices'][side], b.part_bodies[side + '_hand']
                main = b.robot['body']
                self.assertFalse([v for v in p.getVisualShapeData(main, physicsClientId=cid) if v[1] == index])
                self.assertFalse(p.getCollisionShapeData(main, index, physicsClientId=cid))
                ee = p.getLinkState(main, index, computeForwardKinematics=True, physicsClientId=cid)
                inertia = p.getDynamicsInfo(body, -1, physicsClientId=cid)
                root = p.multiplyTransforms(*p.getBasePositionAndOrientation(body, physicsClientId=cid),
                                            *p.invertTransform(inertia[3], inertia[4]))
                maxima[side] = max(maxima[side], np.linalg.norm(np.asarray(root[0]) - ee[4]))
                self.assertGreater(abs(np.dot(root[1], p.multiplyTransforms(
                    ee[4], ee[5], [0, 0, 0], [1, 0, 0, 0])[1])), .999)
                self.assertGreater(min(p.getAABB(body, i, physicsClientId=cid)[0][2]
                                       for i in range(-1, p.getNumJoints(body, physicsClientId=cid))), .02)
                self.assertFalse(p.getContactPoints(main, body, physicsClientId=cid))
            return receipt

        with patch.object(RobotOutput, 'submit', submit):
            self.assertEqual(demo.main(['--urdf', str(source), '--display', 'full',
                                       '--headless', '--seconds', '4']), 0)
        for side, error in maxima.items():
            self.assertLess(error, .002, (side, error))


if __name__ == '__main__':
    unittest.main()
