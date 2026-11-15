"""Replay target-native commands through real whole-body URDFs."""
from dataclasses import replace
import contextlib
import io
import unittest

import numpy as np

from teleoperation.contracts.hand_control import HandJointResult
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from tests.test_arm_ik_multi_robot import ASSETS_AVAILABLE, PYBULLET_AVAILABLE


def stub_camera_backend(hand_models, sides, backend, checkpoints=None, device='cpu'):
    """Deterministic native hand outputs isolate arm integration from NN/dex deps."""
    specs = {(model, side): load_hand_spec(model, side)
             for model in hand_models for side in sides}

    class Retargeter:
        def retarget(self, window):
            assert window.metadata['input_profile'] == PALM_LENGTH_PROFILE
            return {key: HandJointResult(*key, spec.spec_hash, spec.independent_joint_names,
                        spec.q_reference, window.timestamp,
                        valid=window.hands[key[1]] is not None)
                    for key, spec in specs.items()}

    retargeter = Retargeter()
    retargeter.specs = specs
    return specs, retargeter


@unittest.skipUnless(ASSETS_AVAILABLE and PYBULLET_AVAILABLE, 'robot assets and PyBullet required')
class NativeHandReplayTests(unittest.TestCase):
    def test_missing_selected_robot_fails_initialization_and_closes_client(self):
        from unittest.mock import patch
        import pybullet as p
        from teleoperation.simulation.robot_loader import RobotLoader
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        original = RobotLoader.load_robot
        connected = []
        connect = p.connect
        def tracked_connect(*args, **kwargs):
            cid = connect(*args, **kwargs)
            connected.append(cid)
            return cid
        def load(loader, name, *args, **kwargs):
            if name == 'gr1_t2':
                raise FileNotFoundError('missing selected robot asset')
            return original(loader, name, *args, **kwargs)
        with patch.object(p, 'connect', side_effect=tracked_connect), \
                patch.object(RobotLoader, 'load_robot', new=load), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'gr1_t2.*missing selected robot asset'):
                ThreeRobotHandScene(gui=False, robot_types=('h1_2', 'gr1_t2'))
        self.assertTrue(connected)
        self.assertFalse(any(p.isConnected(cid) for cid in connected))

    def setUp(self):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        with contextlib.redirect_stdout(io.StringIO()):
            self.scene = ThreeRobotHandScene(gui=False)
        self.addCleanup(self.scene.close)
        self.specs = {spec.key: spec for robot in self.scene.robots
                      for spec, _ in robot['plans'].values()}

    def test_native_replay_moves_every_target_and_holds_non_hand_joints(self):
        scene = self.scene
        pipeline = JointCommandPipeline(self.specs, filter_mode='none', max_velocity=10.)
        held = {r['type']: list(r['hold_targets']) for r in scene.robots}
        start = {side: scene.joint_values(side) for side in scene.sides}
        for frame in range(30):
            for key, spec in self.specs.items():
                bounds = spec.bounds
                endpoint = np.where(abs(bounds[:, 1]) >= abs(bounds[:, 0]), bounds[:, 1], bounds[:, 0])
                q = spec.q_reference + .6 * frame / 29 * (endpoint - spec.q_reference)
                result = HandJointResult(*key, spec.spec_hash, spec.independent_joint_names, q, frame / 30)
                command = pipeline.update(result, timestamp=frame / 30)
                scene.apply_native(command)
                self.assertEqual(scene.native_commands[key], command.named_positions())
            scene.step()
        for robot in scene.robots:
            for side in scene.sides:
                spec, _ = robot['plans'][side]
                self.assertEqual(len(spec.joint_names), {'h1_2': 12, 'gr1_t2': 11, 'g1': 7}[robot['type']])
                actual = scene.joint_values(side)
                self.assertGreater(max(abs(actual[f"{robot['type']}.{n}"] - start[side][f"{robot['type']}.{n}"])
                                       for n in spec.joint_names), .05)
            actual = [scene.p.getJointState(robot['body'], i, physicsClientId=scene.cid)[0]
                      for i in robot['hold_ids']]
            self.assertLess(max(abs(np.array(actual) - held[robot['type']])), .05)
        self.assertFalse(hasattr(scene, 'apply'))

    def test_bad_identity_coupling_and_limits_rejected_before_motors(self):
        from unittest.mock import patch
        spec = self.specs['h1_2', 'left']
        pipeline = JointCommandPipeline({spec.key: spec})
        command = pipeline.missing(spec.key, timestamp=0.)
        self.scene.apply_native(command)
        bad = command.positions.copy()
        dependent = next(n for n in spec.joint_names if n not in spec.independent_joint_names)
        bad[spec.joint_names.index(dependent)] += .2
        for wrong in (replace(command, spec_hash='0' * 64), replace(command, positions=bad),
                      replace(command, positions=np.full(len(spec.joint_names), 100.))):
            with self.subTest(wrong=wrong), patch.object(self.scene.p, 'setJointMotorControl2') as motor:
                with self.assertRaises(ValueError):
                    self.scene.apply_native(wrong)
                motor.assert_not_called()

    def test_nn_and_dex_share_window_commands_and_native_reports(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from teleoperation.simulation import realtime_teleop as app
        from tests.smplerx_fixtures import write
        reports = []
        with tempfile.TemporaryDirectory() as folder:
            capture = Path(folder) / 'body.h5'
            write(capture, 5)
            for backend in ('nn', 'dex'):
                path = Path(folder) / f'{backend}.json'
                args = app.build_parser().parse_args([
                    '--input-body', str(capture), '--report', str(path), '--no-arms',
                    '--backend', backend, '--headless', '--no-preview', '--robot', 'g1',
                    '--side', 'left', '--print-every', '0'])
                with patch.object(app, 'build_camera_retargeter', side_effect=stub_camera_backend), \
                     patch.object(app, '_drain_in_background', side_effect=lambda action, label: action()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(app.run(args), 0)
                reports.append(json.loads(path.read_text(encoding='utf-8')))
        for report in reports:
            self.assertEqual(set(report['hand_targets']), {'g1/left'})
            target = report['hand_targets']['g1/left']
            self.assertEqual(target['dof'], 7)
            self.assertEqual(len(target['angles']['mapping']), 7)
            self.assertEqual(target['valid_frames'], 3)
            self.assertEqual(target['invalid_frames'], 2)
            self.assertEqual(report['hand_commands']['sides']['g1/left']['held_frames'], 2)
            self.assertEqual(report['scene_summary']['native_applied_frames'], {'g1/left': 5})
            self.assertEqual([r['robot'] for r in report['scene_summary']['robots']], ['g1'])
        self.assertEqual(reports[0]['hand_targets'], reports[1]['hand_targets'])


if __name__ == '__main__':
    unittest.main()
