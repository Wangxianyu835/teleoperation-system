import contextlib
import io
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
from teleoperation.simulation.robot_backends import NativeRobotBackend,SimulationGroup
from teleoperation.simulation.robot_output import RobotOutput


class RobotOutputTests(unittest.TestCase):
    def test_shared_scene_steps_once_and_commands_move_selected_joints(self):
        with contextlib.redirect_stdout(io.StringIO()):
            scene=ThreeRobotHandScene(sides=('right',),gui=False,robot_types=('g1','h1_2'))
        group=SimulationGroup(scene)
        specs={n:load_robot_control_spec(n,resource_overrides={'side':'right'}) for n in ('g1','h1_2')}
        output=RobotOutput({n:NativeRobotBackend(s,scene,group) for n,s in specs.items()})
        try:
            before=output.read_state()
            batch={n:s.reference_frame(0,0.) for n,s in specs.items()}
            with patch.object(scene.p,'stepSimulation',wraps=scene.p.stepSimulation) as step:
                receipt=output.submit(batch,steps=8)
                self.assertEqual(step.call_count,8)
            self.assertEqual(list(receipt.steps.values()),[8])
            after=output.read_state()
            self.assertGreater(np.linalg.norm(after['g1']['right_arm']-before['g1']['right_arm']),1e-5)
            self.assertEqual(scene.frames,1)
            with patch.object(scene.p,'setJointMotorControl2',wraps=scene.p.setJointMotorControl2) as motors:
                bad={n:replace(f,frame_id=1,timestamp=.1) for n,f in batch.items()}
                bad['h1_2']=replace(bad['h1_2'],spec_hash='0'*64)
                with self.assertRaises(ValueError):output.submit(bad,steps=8)
                self.assertEqual(motors.call_count,0)
            with self.assertRaises(ValueError):output.submit(batch,steps=8)
        finally:output.close()
        self.assertFalse(scene.p.isConnected(scene.cid))
