import unittest
from unittest.mock import patch
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.applications.robot_session import RobotControlSession
from tests.smplerx_fixtures import observation


class ArmDisplayStabilityTests(unittest.TestCase):
    def setUp(self):
        self.group=SimulationGroup(gui=False)
        self.addCleanup(self.group.close)
        self.backend=RobotPartSceneBuilder().build(load_robot_control_spec('g1',
            resource_overrides={'parts':'arms'}),group=self.group)

    def test_missing_body_never_solves_or_temporarily_moves_arms(self):
        session=RobotControlSession({'g1':self.backend},arm_calibrate_frames=1)
        scene=self.group.scene;adapter=self.backend.driver.ik['g1']
        before=self.backend.read_state()
        with patch.object(adapter,'solve',wraps=adapter.solve) as solve, \
             patch.object(scene.p,'resetJointState',wraps=scene.p.resetJointState) as resets:
            for i in range(8):
                raw,_=observation(i)
                raw.upper_body.valid.update(left=False,right=False)
                raw.upper_body.arms.update(left=None,right=None)
                raw.upper_body.world_arms.update(left=None,right=None)
                batch=session.next_batch(raw,i,i/30)
                for key,part in batch['g1'].parts.items():
                    self.assertFalse(part.observed)
                    np.testing.assert_array_equal(part.positions,before[key])
                session.output.submit(batch,steps=8)
            self.assertEqual(solve.call_count,0)
            self.assertEqual(resets.call_count,0)

    def test_gui_solve_hides_temporary_states_and_restores_rendering_on_error(self):
        scene=self.group.scene;adapter=self.backend.driver.ik['g1']
        target=adapter.to_base(adapter.measure('right')[0])
        for failing in (False,True):
            before=np.array([scene.p.getJointState(adapter.body,i,physicsClientId=scene.cid)[:2] for i in adapter.indices])
            events=[]
            reset=scene.p.resetJointState
            def resetting(*args,**kwargs):
                events.append('reset');return reset(*args,**kwargs)
            def rendering(flag,value,**kwargs):events.append(('render',value))
            original=adapter._ik_sequence
            def sequence(*args,**kwargs):
                if failing:
                    scene.p.resetJointState(adapter.body,adapter.indices[0],.3,physicsClientId=scene.cid)
                    raise RuntimeError('IK failed')
                return original(*args,**kwargs)
            with patch.object(scene.p,'getConnectionInfo',return_value={'connectionMethod':scene.p.GUI}), \
                 patch.object(scene.p,'configureDebugVisualizer',side_effect=rendering), \
                 patch.object(scene.p,'resetJointState',side_effect=resetting), \
                 patch.object(adapter,'_ik_sequence',side_effect=sequence):
                if failing:
                    with self.assertRaisesRegex(RuntimeError,'IK failed'):adapter.solve('right',target)
                else:adapter.solve('right',target)
            self.assertEqual(events[0],('render',0))
            self.assertEqual(events[-1],('render',1))
            after=np.array([scene.p.getJointState(adapter.body,i,physicsClientId=scene.cid)[:2] for i in adapter.indices])
            np.testing.assert_array_equal(before,after)

    def test_missing_left_body_does_not_stop_right_arm(self):
        session=RobotControlSession({'g1':self.backend},arm_calibrate_frames=1)
        adapter=self.backend.driver.ik['g1']
        with patch.object(adapter,'solve',wraps=adapter.solve) as solve:
            for i in range(3):
                raw,_=observation(i)
                raw.upper_body.valid['left']=False
                raw.upper_body.arms['left']=None
                raw.upper_body.world_arms['left']=None
                batch=session.next_batch(raw,i,i/30)
                self.assertFalse(batch['g1'].parts['left_arm'].observed)
                self.assertTrue(batch['g1'].parts['right_arm'].observed)
                session.output.submit(batch,steps=8)
            self.assertEqual([call.args[0] for call in solve.call_args_list],['right']*3)
