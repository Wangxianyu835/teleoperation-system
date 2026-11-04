import tempfile
from pathlib import Path
import unittest
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.applications.robot_session import RobotControlSession


class RobotControlSessionTests(unittest.TestCase):
    def test_registered_robot_cli_replays_body_observations_without_camera(self):
        from teleoperation.cli import main
        from tests.smplerx_fixtures import write
        import json
        with tempfile.TemporaryDirectory() as d:
            body=write(Path(d)/'body.h5',count=4)
            report=Path(d)/'report.json'
            self.assertEqual(main(['robot','control','--robot','g1','--parts','arms','--input-body',str(body),
                                   '--headless','--report',str(report),'--arm-calibrate-frames','1']),0)
            result=json.loads(report.read_text())
            self.assertEqual(result['frames'],4)
            self.assertEqual(result['physics_frames'],4)
            self.assertEqual(result['targets'],['g1'])

    def test_camera_observations_produce_complete_g1_frames_with_real_nn(self):
        from tests.smplerx_fixtures import observation
        from tests.test_joint_checkpoints import SMALL
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload,save_joint_checkpoint
        from teleoperation.robots.hand_spec import load_hand_spec
        spec=load_robot_control_spec('g1')
        group=SimulationGroup(gui=False)
        try:
            backend=RobotPartSceneBuilder().build(spec,group=group)
            with tempfile.TemporaryDirectory() as d:
                weight=Path(d)/'nn.pth'; hand=load_hand_spec('g1','right')
                save_joint_checkpoint(weight,shared_joint_checkpoint_payload(build_joint_model(hand,SMALL),hand))
                session=RobotControlSession({'g1':backend},backend='nn',checkpoints={'g1':weight},device='cpu',arm_calibrate_frames=1)
                before=session.output.read_state()
                for i in range(7):
                    raw,_=observation(i)
                    batch=session.next_batch(raw,i,i/30)
                    self.assertEqual(set(batch['g1'].parts),set(spec.parts))
                    receipt=session.output.submit(batch,steps=8)
                    self.assertEqual(list(receipt.steps.values()),[8])
                self.assertEqual(group.scene.frames,7)
                self.assertTrue(batch['g1'].parts['right_hand'].observed)
                self.assertEqual(batch['g1'].parts['right_hand'].source_timestamp,raw.hands.timestamp)
                self.assertGreater(np.linalg.norm(session.output.read_state()['g1']['right_hand']-before['g1']['right_hand']),1e-5)
        finally:group.close()
