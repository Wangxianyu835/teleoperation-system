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
from types import SimpleNamespace


class RobotOutputTests(unittest.TestCase):
    def failure_output(self,*,write_failure=False,step_failure=False):
        from teleoperation.contracts.robot_control import RobotCommandFrame
        writes=[]
        class Group:
            group_id='world';time_step=1/240
            def advance(self,steps):
                if step_failure:raise RuntimeError('advance interrupted')
            def close(self):pass
        group=Group()
        class Backend:
            physics_group=group
            def __init__(self,name):self.spec=SimpleNamespace(robot_id=name)
            def validate(self,frame):pass
            def write(self,frame):
                writes.append(frame.robot_id)
                if write_failure and frame.robot_id=='h1_2':raise RuntimeError('motor interrupted')
            def close(self):pass
        names=('g1','h1_2','gr1_t2')
        from teleoperation.contracts.robot_control import RobotPartCommand
        batch={name:RobotCommandFrame(name,'a'*64,0,0.,{'left_arm':RobotPartCommand(('j',),[0.],None,'none',False,'initialized')}) for name in names}
        return RobotOutput({name:Backend(name) for name in names}),batch,writes

    def test_partial_motor_failure_reports_failed_attempt_and_unattempted_targets(self):
        output,batch,writes=self.failure_output(write_failure=True)
        with self.assertRaisesRegex(RuntimeError,'motor interrupted'):output.submit(batch,steps=8)
        self.assertEqual(writes,['g1','h1_2'])
        self.assertEqual(dict(output.last_receipt.statuses),{'g1':'submitted','h1_2':'failed','gr1_t2':'not_submitted'})
        self.assertEqual(dict(output.last_receipt.steps),{'world':0})
        self.assertTrue(output.closed)

    def test_interrupted_physics_does_not_claim_zero_completed_steps(self):
        output,batch,writes=self.failure_output(step_failure=True)
        with self.assertRaisesRegex(RuntimeError,'advance interrupted'):output.submit(batch,steps=8)
        self.assertEqual(dict(output.last_receipt.steps),{'world':None})
        self.assertTrue(all(s=='submitted' for s in output.last_receipt.statuses.values()))

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
