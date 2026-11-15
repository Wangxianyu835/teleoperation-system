"""Regressions from the unified camera-arm code review."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from tests import test_arm_pipeline as fixtures

result = fixtures.result


class ArmReviewRegressions(unittest.TestCase):
    def test_camera_loss_arrival_revokes_old_target_without_reconstruction(self):
        from teleoperation.applications.camera_teleop_source import CameraTeleopSource
        from tests.test_smplerx_camera_source import FakeCamera, FakeHands, FakeClient
        pipeline, body = fixtures.ArmPipelineTests().make_pipeline()
        pipeline.ingest(result(0, stamp=.9), now_s=.91)
        pipeline.prepare(.92, dt=.01)
        pipeline.commit()
        loss = replace(result(1, stamp=1., valid=False), arrays={})
        source = CameraTeleopSource(FakeCamera(), FakeHands(), FakeClient(loss), session_id='test')
        raw = source.next_teleop_observation()
        pipeline.ingest(raw.body_result, now_s=1.)
        self.assertFalse(pipeline.prepare(1.01, dt=.01))
        self.assertEqual(body.calls, 1)
        self.assertIsNotNone(raw.body_result)
        self.assertEqual(raw.body_result.frame_id.sequence, 1)
        # The same revocation must survive raw-body recording/replay.
        from teleoperation.data.body_recording import capture_sample, save_body_recording
        from teleoperation.applications.body_replay_source import BodyReplaySource
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'loss.h5'
            save_body_recording(path, [capture_sample(source, raw)], metadata={
                'source':'smplerx', 'model_id':'synthetic', 'model_hash':'synthetic', 'time_unit':'monotonic_s'})
            replay = BodyReplaySource(path)
            restored = replay.next_teleop_observation()
            replay.release()
        previous, replay_body = fixtures.ArmPipelineTests().make_pipeline()
        previous.ingest(result(0, stamp=.9), now_s=.91)
        previous.ingest(restored.body_result, now_s=1.)
        self.assertFalse(previous.prepare(1.01, dt=.01))
        self.assertEqual(replay_body.calls, 1)
        # A waiting poll must not invent another loss arrival.
        self.assertIsNone(source.next_teleop_observation().body_result)

    def test_fit_rejects_non_smplx_before_loading_or_publishing(self):
        from teleoperation.applications.arm_fit import run
        for family in ('smpl', 'smplh'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                with self.assertRaisesRegex(ValueError, 'SMPL-X|smplx'):
                    run(SimpleNamespace(robots=['g1'], output_dir=directory,
                        body_model_path=directory/'missing', body_model_type=family))
                self.assertEqual(list(directory.iterdir()), [])

    def test_loaded_non_smplx_profile_fails_before_control(self):
        from teleoperation.applications.arm_pipeline import ArmPipeline
        pipeline, body = fixtures.ArmPipelineTests().make_pipeline()
        body.spec = SimpleNamespace(num_betas=2, hash='synthetic', model_type='smpl')
        with self.assertRaisesRegex(ValueError, 'SMPL-X|smplx'):
            ArmPipeline(pipeline.specs, pipeline.profiles, body)

    def test_legacy_recording_preserves_rejected_worker_arrival(self):
        from teleoperation.data.body_recording import replay_observation
        from teleoperation.inputs.smplerx import encode_packet
        row = {'observation_packet':np.frombuffer(encode_packet('observation', {
            'hands':{'hands':{'left':None,'right':None},'timestamp':1.,'source':'camera'},
            'upper_body':{'arms':{'left':None,'right':None},'valid':{'left':False,'right':False},
                          'metadata':{'source':'smplerx'}}, 'wrists':None}), dtype=np.uint8),
            'worker_packet':np.frombuffer(encode_packet('result', {'result':result(1,stamp=1.,valid=False)}),
                                         dtype=np.uint8)}
        raw = replay_observation(row)
        self.assertIsNotNone(raw.body_result)
        self.assertFalse(any(raw.body_result.valid.values()))

    def test_default_asset_is_resolved_before_unified_camera_dispatch(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.robot_control import run
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'g1.json').write_text('{}')
            asset = directory/'hand_landmarker.task'
            asset.write_bytes(b'fixture')
            args = build_parser().parse_args(['robot', 'control', '--robot', 'g1',
                '--parts', 'arms', '--arm-fitting-dir', str(directory), '--headless'])
            self.assertIsNone(args.model_asset_path)
            for key in ('smplerx_worker_python', 'smplerx_root', 'smplerx_checkpoint', 'smplx_model_dir'):
                setattr(args, key, directory)
            # Resource lookup and device loop are external; dispatch and preflight are real.
            with patch('teleoperation.applications.realtime_hand_sim.resolve_model_asset', return_value=asset), \
                 patch('teleoperation.applications.unified_camera_control.run_camera_control',
                       side_effect=lambda options, targets, checkpoints: options.model_asset_path):
                self.assertEqual(run(args), asset)

    def test_waiting_control_ticks_preserve_hand_filter_and_source_interval(self):
        from teleoperation.applications.robot_session import RobotControlSession
        from teleoperation.contracts.hand_control import HandJointResult
        from teleoperation.contracts.observations import RawHandFrame
        from teleoperation.robots.control_spec import load_robot_control_spec
        from tests.test_unified_arm_session import Group
        spec = load_robot_control_spec('g1', resource_overrides={'parts':'hands', 'side':'right'})
        hand = spec.hand_specs['right']
        solution = HandJointResult(*hand.key, hand.spec_hash, hand.independent_joint_names,
            hand.bounds[:, 1], 0.)
        class Backend:
            physics_group = Group()
            driver = None
            def __init__(self): self.spec = spec
            def read_state(self): return {key:np.array(part.reference) for key,part in spec.parts.items()}
            def validate(self, frame): spec.validate_frame(frame)
            def write(self, frame): pass
            def close(self): pass
        def session():
            return RobotControlSession({'g1':Backend()},
                hand_selection=({hand.key:hand}, SimpleNamespace(retarget=lambda window:{hand.key:solution})),
                command_options={'filter_mode':'lowpass', 'alpha':.2, 'max_velocity':1.})
        slow, fast = session(), session()
        for index in range(10):
            stamp = index/30
            raw = RawHandFrame({'left':None, 'right':None}, stamp, 'camera')
            for owner in (slow, fast):
                with patch.object(owner.pipeline, 'process_window', return_value=object()):
                    batch = owner.next_batch(raw, index*3, stamp)
                owner.submit(batch, steps=1)
            for offset in (1, 2):
                batch = fast.next_batch(None, index*3+offset, stamp+offset/100, new_observation=False)
                fast.submit(batch, steps=1)
            np.testing.assert_allclose(fast.commands.previous[hand.key], slow.commands.previous[hand.key], atol=1e-12)
            self.assertEqual(fast.commands.times[hand.key], stamp)
            np.testing.assert_allclose(fast.commands.filtered[hand.key], slow.commands.filtered[hand.key], atol=1e-12)

    @unittest.skipUnless(Path('D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models').exists(),
                         'licensed SMPL-X assets unavailable')
    def test_camera_root_positions_and_rotations_share_native_calibration_frame(self):
        from teleoperation.applications.arm_pipeline import ArmPipeline
        from teleoperation.retargeting.arm.body_model import BodyModel, BodyModelSpec
        from teleoperation.retargeting.arm.segment_scaling import SegmentScaling
        from teleoperation.robots.arm_spec import load_arm_spec
        model = BodyModel(BodyModelSpec('D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models'))
        pose = np.zeros(63)
        pose[15*3+2], pose[16*3+2] = .25, -.15
        native, rotations = model.reconstruct({'body_pose':pose}, np.zeros((1,20)))
        body_to_robot = np.array([[0.,0.,1.], [1.,0.,0.], [0.,1.,0.]])
        specs = {name:load_arm_spec(name) for name in ('h1_2','gr1_t2','g1','tron2a_l21')}
        offsets = {'left':Rotation.from_euler('x',.3).as_matrix(),
                   'right':Rotation.from_euler('z',-.2).as_matrix()}
        profiles = {name:SimpleNamespace(betas=np.zeros(20), body_to_robot=body_to_robot,
            segment_scaling=SegmentScaling(dict(shoulder_center=1.,left_shoulder=1.,right_shoulder=1.,
                left_elbow=1.,right_elbow=1.,left_wrist=1.,right_wrist=1.)),
            wrist_rotation_offsets=offsets, robot_spec_hash=spec.spec_hash) for name,spec in specs.items()}
        pipeline = ArmPipeline(specs, profiles, model)
        # Front-facing upright SMPLer-X has camera root Rx(pi), not identity.
        # A subsequent 30-degree native yaw must survive the camera-axis conversion.
        camera_reference = np.diag([1.,-1.,-1.])
        for index, yaw in enumerate((0., np.pi/6)):
            motion = Rotation.from_euler('y', yaw).as_matrix()
            root = Rotation.from_matrix(camera_reference@motion).as_rotvec()
            sample = replace(result(index, stamp=10.+index*.1), arrays={
                'smplx_root_pose':root.copy(), 'smplx_body_pose':pose.copy()})
            pipeline.ingest(sample, now_s=10.+index*.1)
            for name in specs:
                for side, joint in (('left',20), ('right',21)):
                    target = pipeline.targets[name][side]
                    np.testing.assert_allclose(target[:3,3], body_to_robot@motion@native[0,joint], atol=2e-6)
                    np.testing.assert_allclose(target[:3,:3], body_to_robot@motion@rotations[0,joint]@offsets[side], atol=2e-6)
            np.testing.assert_array_equal(sample.arrays['smplx_root_pose'], root)


if __name__ == '__main__': unittest.main()
