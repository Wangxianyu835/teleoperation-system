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

    def next_observation(self, **kwargs):
        return self.next_teleop_observation().hands


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


class Tron2AUnifiedControlTests(unittest.TestCase):
    """Keep the real CLI, kinematics, native hands, output and HDF5; replace cameras only."""

    def setUp(self):
        from tests.test_mediapipe_tron2a import synthetic_calibration, DEFAULT_TRON2A_URDF
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.urdf = DEFAULT_TRON2A_URDF.resolve()
        self.calibration = synthetic_calibration()
        # An injected device supplies deterministic observations; the CLI's live gate stays enabled.
        self.calibration.metadata['capture_kind'] = 'injected_camera_test'
        self.calibration_path = self.root / 'neutral.json'
        self.calibration_path.write_text(json.dumps(self.calibration.to_dict()))
        self.pose = self.root / 'pose.task'; self.pose.touch()
        self.hands = self.root / 'hands.task'; self.hands.touch()

    def frame(self, index, *, invalid=None, hand_missing=False):
        from tests.test_mediapipe_tron2a import observation as body_observation
        raw = body_observation(index, invalid=invalid)
        hands = observation(world_frame(index))
        if hand_missing:
            hands = replace(hands,hands={'left':None,'right':None},world_hands={'left':None,'right':None})
        hands = replace(hands,timestamp=raw.hands.timestamp,
            metadata={**hands.metadata,**raw.hands.metadata})
        return replace(raw,hands=hands)

    def arguments(self, *extra):
        return ['robot','control','--robot','tron2a_l21','--headless','--fps','0',
                '--tron2a-urdf',str(self.urdf),'--tron2a-calibration',str(self.calibration_path),
                '--model-asset-path',str(self.hands),'--pose-model-asset-path',str(self.pose),
                '--geometry','simplified',*extra]

    def test_real_cli_dex_both_sides_records_and_replays_without_inference(self):
        from teleoperation.data.robot_command_recording import load_robot_recording
        import pybullet as p
        camera = _Camera([self.frame(i, invalid='left' if i==5 else None) for i in range(9)])
        recording, report = self.root/'commands.h5', self.root/'report.json'
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera) as create, \
                patch.object(p,'stepSimulation',wraps=p.stepSimulation) as step, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.arguments('--record-results',str(recording),'--report',str(report))),0)
        self.assertTrue(camera.closed); create.assert_called_once()
        self.assertEqual(create.call_args.kwargs['pose_min_visibility'],self.calibration.metadata['min_visibility'])
        saved = load_robot_recording(recording,('tron2a_l21',))
        rows = list(saved.iter_frames())
        self.assertEqual(len(rows),9)
        self.assertEqual(step.call_count,sum(sum(r.steps.values()) for _,r in rows))
        frame = rows[-1][0]['tron2a_l21']
        self.assertEqual({k:len(v.positions) for k,v in frame.parts.items()},
                         {'left_arm':7,'right_arm':7,'left_hand':17,'right_hand':17})
        self.assertTrue(frame.parts['left_hand'].observed)
        self.assertFalse(rows[5][0]['tron2a_l21'].parts['left_arm'].observed)
        self.assertTrue(rows[5][0]['tron2a_l21'].parts['right_arm'].observed)
        self.assertEqual(saved.metadata['backend'],'dex')
        self.assertIn('base_pose',saved.metadata['execution_config']['tron2a_l21'])
        summary=json.loads(report.read_text())['arms']['robots']['tron2a_l21']
        self.assertEqual(summary['ik_solver'],'roboticstoolbox-lm')
        self.assertIn('actual_position_error_m',summary['sides']['left']['metrics'])
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',side_effect=AssertionError('camera during replay')), \
                patch('teleoperation.retargeting.arm.ik.RoboticsToolboxArmIK',side_effect=AssertionError('IK during replay')), \
                patch('teleoperation.applications.hand_backend.build_target_hand_retargeters',side_effect=AssertionError('hand inference during replay')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['robot','replay','--robot','tron2a_l21','--input',str(recording),
                                  '--geometry','simplified','--headless']),0)

    def test_shared_scene_and_one_capture_with_another_robot(self):
        from teleoperation.data.robot_command_recording import load_robot_recording
        import pybullet as p
        args = self.arguments('--parts','arms','--arm-calibrate-frames','3')
        args[2:4] = ['--robots','g1','tron2a_l21']
        camera = _Camera([self.frame(i) for i in range(6)])
        recording = self.root/'mixed.h5'
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera) as create, \
                patch.object(p,'stepSimulation',wraps=p.stepSimulation) as step, \
                contextlib.redirect_stdout(io.StringIO()):
            main([*args,'--record-results',str(recording)])
        self.assertTrue(camera.closed); create.assert_called_once()
        rows=list(load_robot_recording(recording,('g1','tron2a_l21')).iter_frames())
        self.assertEqual(step.call_count,sum(next(iter(r.steps.values())) for _,r in rows))
        self.assertTrue(all(len(r.steps)==1 for _,r in rows))
        for batch,_ in rows:
            self.assertEqual(batch['g1'].timestamp,batch['tron2a_l21'].timestamp)

    def test_all_parts_and_sides_share_one_group_and_preserve_disabled_joints(self):
        from teleoperation.robots.control_spec import load_robot_control_spec
        from teleoperation.simulation.robot_backends import SimulationGroup
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        from teleoperation.simulation.robot_output import RobotOutput
        for parts in ('arms','hands','both'):
            for side in ('left','right','both'):
                with self.subTest(parts=parts,side=side):
                    spec=load_robot_control_spec('tron2a_l21',resource_overrides={
                        'urdf_path':self.urdf,'parts':parts,'side':side})
                    group=SimulationGroup()
                    try:
                        backend=RobotPartSceneBuilder().build(spec,group=group,
                            calibration=self.calibration,geometry='simplified',base_pose=(2.,0.,.8))
                        output=RobotOutput({'tron2a_l21':backend})
                        initial=backend.read_state()
                        frame=replace(spec.reference_frame(0,0.),parts={
                            key:replace(part,positions=initial[key])
                            for key,part in spec.reference_frame(0,0.).parts.items()})
                        for i in range(8):output.submit({'tron2a_l21':replace(frame,frame_id=i,timestamp=i/30)},steps=8)
                        self.assertEqual(group.frames,8)
                        if parts!='hands':
                            actual=backend.read_arm_state()
                            for arm_side in ('left','right'):
                                np.testing.assert_allclose(actual[arm_side]['q'],self.calibration.safe_q[arm_side],atol=.02)
                                np.testing.assert_allclose(actual[arm_side]['pose'][:3],
                                    self.calibration.robot_neutral_pose[arm_side][:3],atol=.02)
                        if side!='both':self.assertTrue(all(key.startswith(side) for key in backend.part_bodies))
                    finally:group.close()

    def test_nn_uses_shared_l21_weights_and_rejects_legacy_before_devices(self):
        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.retargeting.hand.checkpoint import save_joint_checkpoint, shared_joint_checkpoint_payload
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from tests.test_joint_checkpoints import SMALL
        spec=load_hand_spec('l21','right')
        checkpoint=self.root/'shared.pth'
        save_joint_checkpoint(checkpoint,shared_joint_checkpoint_payload(build_joint_model(spec,SMALL),spec))
        camera=_Camera([self.frame(i) for i in range(6)])
        result=self.root/'nn.h5'
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera), contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments('--backend','nn','--hand-checkpoint','l21',str(checkpoint),'--record-results',str(result)))
        with h5py.File(result) as stream:
            self.assertTrue(stream['robots/tron2a_l21/parts/right_hand/observed'][-1])
        checkpoint.write_bytes(b'legacy checkpoint is not a target-shared model')
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput') as create, \
                patch('teleoperation.simulation.robot_backends.SimulationGroup') as scene:
            with self.assertRaises(Exception):
                main(self.arguments('--backend','nn','--hand-checkpoint','l21',str(checkpoint)))
        create.assert_not_called(); scene.assert_not_called()

    def test_preflight_rejects_invalid_calibration_before_devices(self):
        for kind in ('synthetic_validation_only','bad_fk','bad_units','unsafe_reference'):
            data=json.loads(json.dumps(self.calibration.to_dict()))
            if kind=='synthetic_validation_only':data['metadata']['capture_kind']=kind
            elif kind=='bad_fk':data['robot_neutral_pose']['left'][0]+=.01
            elif kind=='bad_units':data['metadata']['units']='cm'
            else:data['safe_q']['right'][0]=99
            self.calibration_path.write_text(json.dumps(data))
            with self.subTest(kind=kind), patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput') as create, \
                    patch('teleoperation.simulation.robot_backends.SimulationGroup') as scene:
                with self.assertRaises(ValueError):main(self.arguments('--parts','arms'))
                create.assert_not_called(); scene.assert_not_called()

    def test_camera_failure_closes_scene_and_can_restart(self):
        import pybullet as p
        for attempt in range(3):
            camera=_Camera([self.frame(0)])
            camera.next_teleop_observation=lambda: (_ for _ in ()).throw(RuntimeError('capture lost'))
            with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera), \
                    patch.object(p,'disconnect',wraps=p.disconnect) as disconnect, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'capture lost'):main(self.arguments('--parts','arms'))
            self.assertTrue(camera.closed); disconnect.assert_called_once()

    def test_no_frame_reuse_and_explicit_braking_return_are_recordable(self):
        from teleoperation.robots.control_spec import load_robot_control_spec
        from teleoperation.simulation.robot_backends import SimulationGroup
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        from teleoperation.applications.robot_session import RobotControlSession
        from teleoperation.applications.mediapipe_robot_arms import CameraArmControllers
        from teleoperation.retargeting.arm.retargeter import Tron2ADualArmController
        from teleoperation.data.robot_command_recording import RobotCommandWriter,load_robot_recording
        spec=load_robot_control_spec('tron2a_l21',resource_overrides={'parts':'arms','urdf_path':self.urdf})
        controller=Tron2ADualArmController(self.calibration,self.urdf,input_mode='camera_world')
        group=SimulationGroup()
        try:
            backend=RobotPartSceneBuilder().build(spec,group=group,calibration=self.calibration,geometry='simplified')
            args=build_parser().parse_args(self.arguments('--parts','arms'))
            arms=CameraArmControllers({'tron2a_l21':backend},args,controller)
            session=RobotControlSession({'tron2a_l21':backend},camera_arms=arms)
            from teleoperation.applications.robot_control import _CameraSource
            source=_CameraSource(_Camera([self.frame(i) for i in range(10)]),True)
            path=self.root/'safety.h5'
            with RobotCommandWriter(path,{'tron2a_l21':spec},{}) as writer:
                states=[]
                raw = None
                for i in range(80):
                    raw=source.next_teleop_observation() if i<10 else raw if i==10 else None
                    if raw is not None and i<10:
                        raw.upper_body.world_arms['left'][2]+=[.03,0,0]
                    batch=session.next_batch(raw,i,i*.01,new_observation=raw is not None)
                    receipt=session.submit(batch,steps=2)
                    writer.append(batch,receipt)
                    states.append(batch['tron2a_l21'].parts['left_arm'])
                before=controller.controllers['left'].q.copy()
                with self.assertRaises(ValueError):session.next_batch(None,80,float('nan'))
                np.testing.assert_array_equal(controller.controllers['left'].q,before)
            self.assertIn('braking',[p.status for p in states])
            self.assertIn('returning',[p.status for p in states])
            self.assertTrue(all(not p.observed for p in states[10:]))
            self.assertGreater(np.max(np.abs(states[10].positions-states[9].positions)),0)
            saved=list(load_robot_recording(path,('tron2a_l21',)).iter_frames())
            self.assertEqual(saved[-1][0]['tron2a_l21'].parts['left_arm'].status,'returning')
        finally:group.close()

    def test_cli_parts_sides_and_native_hand_backends(self):
        from teleoperation.retargeting.hand.checkpoint import save_joint_checkpoint,shared_joint_checkpoint_payload
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.robots.hand_spec import load_hand_spec
        from teleoperation.data.robot_command_recording import load_robot_recording
        from tests.test_joint_checkpoints import SMALL
        hand=load_hand_spec('l21','right');weights=self.root/'l21.pth'
        save_joint_checkpoint(weights,shared_joint_checkpoint_payload(build_joint_model(hand,SMALL),hand))
        for parts in ('arms','hands','both'):
            for side in ('left','right','both'):
                for backend in (('dex',) if parts=='arms' else ('dex','nn')):
                    with self.subTest(parts=parts,side=side,backend=backend):
                        args=self.arguments('--parts',parts,'--side',side,'--backend',backend)
                        if parts=='hands':
                            for flag in ('--tron2a-urdf','--tron2a-calibration'):
                                index=args.index(flag);del args[index:index+2]
                        if backend=='nn':args.extend(['--hand-checkpoint','l21',str(weights)])
                        recording=self.root/f'{parts}-{side}-{backend}.h5'
                        camera=_Camera([self.frame(i) for i in range(4)])
                        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera), \
                                contextlib.redirect_stdout(io.StringIO()):
                            main([*args,'--record-results',str(recording)])
                        frame=list(load_robot_recording(recording,('tron2a_l21',)).iter_frames())[-1][0]['tron2a_l21']
                        selected=('left','right') if side=='both' else (side,)
                        kinds=('arm','hand') if parts=='both' else ('arm',) if parts=='arms' else ('hand',)
                        self.assertEqual(set(frame.parts),{s+'_'+k for s in selected for k in kinds})
                        self.assertTrue(all(part.observed for part in frame.parts.values()))
                        self.assertTrue(camera.closed)

    def test_interrupt_keeps_failed_recording_and_releases_camera(self):
        camera=_Camera([])
        camera.next_teleop_observation=lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        path=self.root/'interrupted.h5'
        with patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera), \
                contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments('--parts','arms','--record-results',str(path)))
        self.assertTrue(camera.closed);self.assertFalse(path.exists())
        pending=list(self.root.glob('interrupted.h5.pending-*.h5'))
        self.assertEqual(len(pending),1)
        with h5py.File(pending[0]) as stream:self.assertEqual(stream.attrs['recording_status'],'failed')

    def test_legacy_h5_replay_uses_shared_output(self):
        from teleoperation.data.command_h5 import write_robot_commands
        values={}
        for side in ('left','right'):
            values[side+'_arm_q']=np.stack([self.calibration.safe_q[side]]*2)
            values[side+'_hand_q']=np.zeros((2,17))
            values[side+'_arm_valid']=values[side+'_hand_valid']=np.ones(2,dtype=bool)
        values['robot_command']=np.concatenate([values[k] for k in
            ('left_arm_q','left_hand_q','right_arm_q','right_hand_q')],axis=1)
        path=self.root/'legacy.h5'
        write_robot_commands(path,np.arange(2),np.arange(2)/30,values,self.urdf,self.calibration_path)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['dual','replay','--command-h5',str(path),'--urdf',str(self.urdf),
                '--calibration',str(self.calibration_path),'--headless','--display','simplified']),0)

    def test_hands_only_record_replay_preserves_actual_mounting_height(self):
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        args=self.arguments('--parts','hands')
        for flag in ('--tron2a-urdf','--tron2a-calibration'):
            index=args.index(flag);del args[index:index+2]
        camera=_Camera([self.frame(i) for i in range(4)])
        recording=self.root/'hands.h5';poses=[]
        original=RobotPartSceneBuilder.build
        def build(builder,spec,**kwargs):
            backend=original(builder,spec,**kwargs)
            poses.append({key:backend.scene.p.getBasePositionAndOrientation(body,physicsClientId=backend.scene.cid)
                          for key,body in backend.part_bodies.items()})
            return backend
        with patch.object(RobotPartSceneBuilder,'build',build), \
                patch('teleoperation.inputs.mediapipe.MediaPipeCameraInput',return_value=camera), \
                contextlib.redirect_stdout(io.StringIO()):
            main([*args,'--record-results',str(recording)])
            main(['robot','replay','--robot','tron2a_l21','--input',str(recording),'--headless'])
        self.assertEqual(len(poses),2)
        for key in poses[0]:
            for before,after in zip(poses[0][key],poses[1][key]):
                np.testing.assert_allclose(before,after,atol=1e-7,rtol=0)
