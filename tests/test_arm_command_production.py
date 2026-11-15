import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
from teleoperation.simulation.arm_scene_driver import ArmSceneDriver


class ArmCommandProductionTests(unittest.TestCase):
    def test_solve_returns_same_values_without_actuation_and_restores_state(self):
        with contextlib.redirect_stdout(io.StringIO()):
            scene=ThreeRobotHandScene(sides=('right',),gui=False,robot_types=('g1',))
        try:
            driver=ArmSceneDriver(scene,sides=('right',),track_posture=False)
            adapter=driver.ik['g1']
            target={'g1':adapter.profile.realtime_neutral('right')}
            before=np.array([scene.p.getJointState(adapter.body,i,physicsClientId=scene.cid)[:2] for i in adapter.indices])
            with patch.object(scene.p,'setJointMotorControl2',wraps=scene.p.setJointMotorControl2) as motors, \
                 patch.object(scene.p,'stepSimulation',wraps=scene.p.stepSimulation) as steps:
                solved=driver.solve_each('right',target)
                self.assertEqual(motors.call_count,0)
                self.assertEqual(steps.call_count,0)
                after=np.array([scene.p.getJointState(adapter.body,i,physicsClientId=scene.cid)[:2] for i in adapter.indices])
                np.testing.assert_array_equal(before,after)
                applied=driver.apply_each('right',target)
                self.assertEqual(motors.call_count,7)
                self.assertEqual(steps.call_count,0)
            np.testing.assert_allclose(list(solved['g1']['joints'].values()),list(applied['g1']['joints'].values()),atol=1e-10)
            self.assertEqual(driver.applied['right'],1)
        finally: scene.close()
