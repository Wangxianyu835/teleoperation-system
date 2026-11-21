"""Native entry points retain the existing camera arm algorithm and recordings."""
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from teleoperation.cli import build_parser, main
from teleoperation.contracts.observations import RawTeleopObservation, UpperBodyObservation
from tests.test_world_hand_pipeline import observation, world_frame


class _Camera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def next_teleop_observation(self):
        return next(self.frames)

    def close(self):
        self.closed = True


def camera_frame(index):
    hands = observation(world_frame(index))
    left = np.array([[-.18, 0, 0], [-.28, .22, -.05], [-.20, .32, -.20]])
    right = left * [-1, 1, 1]
    if index >= 4:
        left[1:, 0] -= .01
        right[1:, 0] += .01
    body = UpperBodyObservation(
        {'left': left, 'right': right}, {'left': True, 'right': True},
        hands.timestamp, 'mediapipe_pose_normalized', 'normalized',
        world_arms={'left': left, 'right': right},
    )
    return RawTeleopObservation(hands, body)


class NativeControlAlignmentTests(unittest.TestCase):
    def test_camera_timestamp_conversion_preserves_same_frame_hand_and_arm_data(self):
        from teleoperation.applications.robot_control import _CameraSource
        raw = camera_frame(0)
        camera = _Camera([raw])
        source = _CameraSource(camera, True)
        result = source.next_teleop_observation()
        self.assertEqual(result.hands.timestamp, 1700000000.)
        self.assertEqual(result.upper_body.timestamp, 1700000000.)
        np.testing.assert_array_equal(result.hands.world_hands['left'], raw.hands.world_hands['left'])
        np.testing.assert_array_equal(result.upper_body.world_arms['right'], raw.upper_body.world_arms['right'])
        source.close()
        self.assertTrue(camera.closed)

    def test_camera_both_sides_use_paper_solver_and_replay_accepted_commands(self):
        from teleoperation.applications.mediapipe_robot_arms import MediaPipeRobotArms
        from teleoperation.data.robot_command_recording import load_robot_recording
        camera = _Camera([camera_frame(index) for index in range(9)])
        observed = []
        original = MediaPipeRobotArms.solve

        def solve(controller, raw, dt):
            observed.extend(adapter.ik_solver for backend in controller.backends.values()
                            for adapter in backend.driver.ik.values())
            return original(controller, raw, dt)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / 'pose.task'
            pose.write_bytes(b'camera replaced at the device boundary')
            recording, report = root / 'commands.h5', root / 'control.json'
            arguments = ['robot', 'control', '--robot', 'h1_2', '--parts', 'both',
                         '--side', 'both', '--backend', 'dex', '--headless', '--fps', '0',
                         '--frames', '9', '--arm-calibrate-frames', '3',
                         '--pose-model-asset-path', str(pose),
                         '--record-results', str(recording), '--report', str(report)]
            with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput', return_value=camera), \
                    patch.object(MediaPipeRobotArms, 'solve', solve), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertTrue(camera.closed)
            self.assertEqual(set(observed), {'paper'})
            result = json.loads(report.read_text())
            self.assertEqual(result['frames'], 9)
            saved = load_robot_recording(recording, ('h1_2',))
            frames = list(saved.iter_frames())
            self.assertEqual(len(frames), 9)
            tracked = set()
            recorded_steps = 0
            for batch, receipt in frames:
                steps = sum(receipt.steps.values())
                self.assertGreaterEqual(steps, 1)
                self.assertLessEqual(steps, 24)
                recorded_steps += steps
                frame = batch['h1_2']
                saved.specs['h1_2'].validate_frame(frame)
                tracked.update(key for key, part in frame.parts.items() if part.observed)
            self.assertEqual(tracked, {'left_arm', 'right_arm', 'left_hand', 'right_hand'})
            replay = root / 'replay.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['robot', 'replay', '--robot', 'h1_2', '--input', str(recording),
                                       '--headless', '--report', str(replay)]), 0)
            self.assertEqual(json.loads(replay.read_text())['physics_steps'], recorded_steps)

    def test_native_training_export_and_hand_replay_use_matching_target_contract(self):
        import torch
        from teleoperation.applications.native_hand_align import align_h5
        from teleoperation.paths import training_run_directory
        from teleoperation.contracts.native_coordinates import PALM_LENGTH_PROFILE
        from teleoperation.retargeting.hand.checkpoint import save_joint_checkpoint, shared_joint_checkpoint_payload
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.robots.hand_spec import load_hand_spec
        from tests.test_joint_checkpoints import SMALL
        from tests.test_world_hand_pipeline import save_raw

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, aligned, weights = root / 'raw.h5', root / 'aligned.h5', root / 'init.pth'
            save_raw(raw, [world_frame(index) for index in range(40)])
            align_h5(raw, aligned)
            spec = load_hand_spec('g1', 'right')
            save_joint_checkpoint(weights, shared_joint_checkpoint_payload(build_joint_model(spec, SMALL), spec))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['hand', 'train', '--input', str(aligned), '--hand-model', 'g1',
                    '--run-name', 'smoke', '--checkpoint-root', str(root / 'models'),
                    '--init-checkpoint', str(weights), '--epochs', '1', '--batch-size', '8', '--device', 'cpu']), 0)
                checkpoint = training_run_directory(root / 'models', 'g1', 'shared', PALM_LENGTH_PROFILE, 'smoke') / 'model_best.pth'
                result = root / 'joints.h5'
                self.assertEqual(main(['hand', 'export', '--input', str(aligned), '--output', str(result),
                    '--hand-model', 'g1', '--backend', 'nn', '--checkpoint', str(checkpoint),
                    '--hand', 'both', '--device', 'cpu']), 0)
                self.assertEqual(main(['hand', 'inspect', '--angle-h5', str(result)]), 0)
                self.assertEqual(main(['replay', 'hand-joints', '--input', str(result),
                    '--hand-model', 'g1', '--hand', 'both', '--headless']), 0)
            with h5py.File(result) as saved:
                self.assertEqual(saved.attrs['schema'], 'hand_joint_results_v1')

    def test_non_tron_calibration_is_rejected_before_devices(self):
        from teleoperation.applications.robot_control import preflight
        args = build_parser().parse_args(['robot', 'control', '--robot', 'g1', '--parts', 'hands',
                                         '--tron2a-calibration', 'unused.json'])
        with self.assertRaises(ValueError):
            preflight(args)


if __name__ == '__main__':
    unittest.main()
