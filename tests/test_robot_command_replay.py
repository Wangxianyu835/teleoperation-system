from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class RobotReplayTests(unittest.TestCase):
    def test_registered_record_replay_uses_exact_commands_and_physics_without_solvers(self):
        from dataclasses import replace
        from teleoperation.cli import main
        from teleoperation.data.robot_command_recording import RobotCommandWriter,load_robot_recording
        from teleoperation.robots.control_spec import load_robot_control_spec
        from teleoperation.simulation.robot_backends import SimulationGroup
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        from teleoperation.applications.robot_session import RobotControlSession
        from tests.test_robot_control_session import RobotControlSessionTests
        from tests.test_arm_pipeline import result
        from tests.smplerx_fixtures import observation
        before=[];before_states=[];steps=[];states=[]
        with tempfile.TemporaryDirectory() as d:
            commands=Path(d)/'commands.h5'
            spec=load_robot_control_spec('g1',resource_overrides={'parts':'arms'})
            group=SimulationGroup(gui=False)
            try:
                backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False)
                pipeline,_=RobotControlSessionTests().fitted_camera_pipeline(spec,backend)
                session=RobotControlSession({'g1':backend},arm_pipeline=pipeline)
                with RobotCommandWriter(commands,{'g1':spec},{'geometry':'full'}) as writer:
                    for i in range(5):
                        raw,_=observation(i)
                        raw=replace(raw,body_result=result(i,stamp=raw.hands.timestamp))
                        batch=session.next_batch(raw,i,i*.01,source_now_s=raw.hands.timestamp)
                        receipt=session.submit(batch,steps=7+i)
                        writer.append(batch,receipt)
                        before.append((batch,7+i));before_states.append(session.output.read_state())
            finally:group.close()
            recording=load_robot_recording(commands,('g1',))
            self.assertEqual(recording.specs['g1'].display,'arms')
            for (expected,_),(stored,_) in zip(before,recording.iter_frames()):
                for key,part in expected['g1'].parts.items():
                    np.testing.assert_array_equal(part.positions,stored['g1'].parts[key].positions)
            replay_group=SimulationGroup(gui=False)
            advance=replay_group.advance
            def replay_advance(count):
                advance(count)
                steps.append(count)
                p=replay_group.scene.p;cid=replay_group.scene.cid
                body=replay_group.scene.robots[0]['body']
                names={p.getJointInfo(body,i,physicsClientId=cid)[1].decode():i
                       for i in range(p.getNumJoints(body,physicsClientId=cid))}
                states.append({'g1':{key:np.array([p.getJointState(body,names[n],physicsClientId=cid)[0]
                                                  for n in part.joint_names]) for key,part in spec.parts.items()}})
            with patch('teleoperation.simulation.robot_backends.SimulationGroup',return_value=replay_group), \
                 patch.object(replay_group,'advance',side_effect=replay_advance), \
                 patch('teleoperation.simulation.arm_scene_driver.ArmSceneDriver',side_effect=AssertionError('IK constructed')), \
                 patch('teleoperation.retargeting.arm.pink_ik.PinkArmIK',side_effect=AssertionError('PINK constructed')), \
                 patch('teleoperation.applications.robot_session.RobotControlSession',side_effect=AssertionError('solver session')), \
                 patch('teleoperation.retargeting.hand.joint_commands.JointCommandPipeline',side_effect=AssertionError('filter')):
                self.assertEqual(main(['robot','replay','--robot','g1','--input',str(commands),'--headless']),0)
            self.assertEqual([s for b,s in before],steps)
            for a,b in zip(before_states,states):
                for key in a['g1']:np.testing.assert_allclose(a['g1'][key],b['g1'][key],atol=1e-12,rtol=0)

    def test_four_target_recording_replays_tron_without_calibration_solver_and_subset_without_other_targets(self):
        from dataclasses import replace
        from teleoperation.robots.control_spec import load_robot_control_spec
        from teleoperation.data.robot_command_recording import RobotCommandWriter
        from teleoperation.simulation.robot_backends import SimulationGroup
        from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
        from teleoperation.simulation.robot_output import RobotOutput
        from teleoperation.retargeting.arm.calibration import ArmCalibration
        from teleoperation.cli import main
        from tests.test_robot_comparison import CALIBRATION
        calibration=ArmCalibration.load(CALIBRATION)
        config={key:{s:v.tolist() for s,v in getattr(calibration,key).items()} for key in ('safe_q','hand_mount_pose')}
        specs={n:load_robot_control_spec(n) for n in ('g1','h1_2','gr1_t2','tron2a_l21')}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'four.h5';report=Path(d)/'subset.json'
            group=SimulationGroup(gui=False)
            try:
                backends={n:RobotPartSceneBuilder().build(s,group=group,execution_only=True,
                    execution_config=config if n=='tron2a_l21' else None,geometry='simplified',base_pose=(i*2.2,0,.8))
                    for i,(n,s) in enumerate(specs.items())}
                output=RobotOutput(backends)
                with RobotCommandWriter(path,specs,{'geometry':'simplified','execution_config':{'tron2a_l21':config}}) as writer:
                    for i in range(3):
                        batch={n:s.reference_frame(i,i/30) for n,s in specs.items()}
                        writer.append(batch,output.submit(batch,steps=7+i))
            finally:group.close()
            with patch('devtools.legacy_dual_baseline.Tron2ADualArmController',side_effect=AssertionError('arm solver')), \
                 patch('teleoperation.simulation.arm_scene_driver.ArmSceneDriver',side_effect=AssertionError('native IK')):
                self.assertEqual(main(['robot','replay','--robots',*specs,'--input',str(path),'--headless','--geometry','simplified']),0)
                self.assertEqual(main(['robot','replay','--robot','g1','--input',str(path),'--headless','--report',str(report)]),0)
            import json
            result=json.loads(report.read_text())
            self.assertEqual(result['targets'],['g1']);self.assertEqual(result['physics_steps'],24)

    def test_corrupt_recording_fails_before_scene_creation(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.robot_control import replay
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.h5';path.write_bytes(b'not a recording')
            args=build_parser().parse_args(['robot','replay','--robot','g1','--input',str(path)])
            with patch('teleoperation.simulation.robot_backends.SimulationGroup') as group:
                with self.assertRaises((ValueError,OSError)):replay(args)
                group.assert_not_called()
