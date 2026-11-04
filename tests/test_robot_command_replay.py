from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class RobotReplayTests(unittest.TestCase):
    def test_registered_record_replay_uses_exact_commands_and_physics_without_solvers(self):
        from teleoperation.cli import main
        from teleoperation.data.robot_command_recording import load_robot_recording
        from teleoperation.simulation.robot_output import RobotOutput
        from tests.smplerx_fixtures import write
        records=[];states=[]
        original=RobotOutput.submit
        def submit(output,batch,*,steps):
            result=original(output,batch,steps=steps)
            records.append((batch,steps));states.append(output.read_state())
            return result
        with tempfile.TemporaryDirectory() as d:
            body=write(Path(d)/'body.h5',count=5)
            commands=Path(d)/'commands.h5'
            with patch.object(RobotOutput,'submit',submit):
                self.assertEqual(main(['robot','control','--robot','g1','--parts','arms',
                    '--input-body',str(body),'--headless','--arm-calibrate-frames','1',
                    '--record-results',str(commands)]),0)
            before=list(records);before_states=list(states);records.clear();states.clear()
            self.assertEqual(load_robot_recording(commands,('g1',)).specs['g1'].display,'arms')
            with patch.object(RobotOutput,'submit',submit), \
                 patch('teleoperation.simulation.arm_scene_driver.ArmSceneDriver',side_effect=AssertionError('IK constructed')), \
                 patch('teleoperation.applications.robot_session.RobotControlSession',side_effect=AssertionError('solver session')), \
                 patch('teleoperation.retargeting.hand.joint_commands.JointCommandPipeline',side_effect=AssertionError('filter')):
                self.assertEqual(main(['robot','replay','--robot','g1','--input',str(commands),'--headless']),0)
            self.assertEqual([s for b,s in before],[s for b,s in records])
            for (a,_),(b,_) in zip(before,records):
                self.assertEqual(a['g1'].frame_id,b['g1'].frame_id)
                for key in a['g1'].parts:
                    np.testing.assert_array_equal(a['g1'].parts[key].positions,b['g1'].parts[key].positions)
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
            with patch('teleoperation.applications.dual.Tron2ADualArmController',side_effect=AssertionError('arm solver')), \
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
