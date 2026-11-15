import unittest
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.robot_backends import SimulationGroup


class RobotSceneSelectionTests(unittest.TestCase):
    def test_output_only_all_robots_apply_commands_without_legacy_solvers(self):
        from unittest.mock import patch
        from dataclasses import replace
        for name in ('h1_2','gr1_t2','g1','tron2a_l21'):
            with self.subTest(robot=name):
                group=SimulationGroup(gui=False)
                try:
                    spec=load_robot_control_spec(name,resource_overrides={'parts':'both','side':'right'})
                    with patch('teleoperation.simulation.arm_scene_driver.ArmSceneDriver',side_effect=AssertionError('legacy IK created')), \
                         patch('devtools.legacy_dual_baseline.Tron2ADualArmController',side_effect=AssertionError('legacy TRON IK created')):
                        backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False,geometry='simplified')
                    self.assertIsNone(backend.driver)
                    self.assertIsNone(getattr(backend,'arm_controller',None))
                    frame=spec.reference_frame(0,0.)
                    before=backend.read_state()
                    arm=frame.parts['right_arm']
                    q=before['right_arm'].copy();q[0]+=.05
                    frame=replace(frame,parts={**frame.parts,'right_arm':replace(arm,positions=q)})
                    backend.validate(frame)
                    with patch.object(group.scene.p,'setJointMotorControl2',wraps=group.scene.p.setJointMotorControl2) as motors:
                        backend.write(frame)
                    hand=frame.parts['right_hand']
                    if name=='tron2a_l21':
                        hand_body=backend.part_bodies['right_hand'];joint_map=backend.names[hand_body]
                    else:
                        hand_body=backend.robot['body'];joint_map=backend.names
                    emitted={(call.args[0],call.args[1]):call.kwargs['targetPosition'] for call in motors.call_args_list}
                    for joint,value in zip(hand.joint_names,hand.positions):
                        self.assertEqual(emitted[hand_body,joint_map[joint]],value)
                    group.advance(24)
                    self.assertGreater(np.linalg.norm(backend.read_state()['right_arm']-before['right_arm']),1e-5)
                    if name=='tron2a_l21':
                        self.assertEqual(set(backend.execution_config),{'safe_q','hand_mount_pose'})
                        self.assertEqual(set(backend.execution_config['safe_q']),{'left','right'})
                        self.assertEqual(backend.execution_config['hand_mount_pose']['right'],[0.,0.,0.,0.,0.,0.,1.])
                    else:
                        active={backend.names[n] for n in arm.joint_names}
                        self.assertFalse(active.intersection(backend.robot['hold_ids']))
                finally:group.close()

    def test_output_only_tron_execution_configuration_replays_the_actual_initial_state(self):
        spec=load_robot_control_spec('tron2a_l21',resource_overrides={'parts':'arms','side':'right'})
        group=SimulationGroup(gui=False)
        try:
            backend=RobotPartSceneBuilder().build(spec,group=group,solve_arms=False,geometry='simplified')
            config=backend.execution_config
            initial=backend.read_state()['right_arm']
        finally:group.close()
        replay=SimulationGroup(gui=False)
        try:
            backend=RobotPartSceneBuilder().build(spec,group=replay,execution_only=True,
                execution_config=config,geometry='simplified')
            self.assertIsNone(backend.arm_controller)
            np.testing.assert_array_equal(backend.read_state()['right_arm'],initial)
            self.assertEqual(backend.execution_config,config)
        finally:replay.close()

    def test_full_robots_clear_ground_and_keep_uncontrolled_joints_held(self):
        from dataclasses import replace
        for name in ('h1_2', 'gr1_t2'):
            for base_pose in (None, (2.2, 0., .8)):
                for replay in (False, True):
                    with self.subTest(robot=name, base=base_pose, replay=replay):
                        group = SimulationGroup(gui=False)
                        try:
                            backend = RobotPartSceneBuilder().build(load_robot_control_spec(name),
                                group=group, base_pose=base_pose, execution_only=replay)
                            p, cid, robot = group.scene.p, group.scene.cid, backend.robot
                            body = robot['body']
                            lowest = min(p.getAABB(body, i, physicsClientId=cid)[0][2]
                                for i in range(-1, p.getNumJoints(body, physicsClientId=cid)))
                            self.assertGreaterEqual(lowest, .03 - 1e-6)
                            indices = robot['hold_ids']
                            before = np.array([p.getJointState(body, i, physicsClientId=cid)[0] for i in indices])
                            frame = backend.spec.reference_frame(0, 0.)
                            actual = backend.read_state()
                            backend.write(replace(frame, parts={k:replace(v, positions=actual[k])
                                for k,v in frame.parts.items()}))
                            for _ in range(45):
                                group.advance(8)
                            after = np.array([p.getJointState(body, i, physicsClientId=cid)[0] for i in indices])
                            self.assertLess(float(np.max(np.abs(after-before))), .01)
                        finally:
                            group.close()

    def test_g1_only_requested_geometry_is_loaded_and_rendered(self):
        for parts,side in (('both','both'),('arms','right'),('hands','right'),('hands','both')):
            with self.subTest(parts=parts,side=side):
                group=SimulationGroup(gui=False)
                try:
                    spec=load_robot_control_spec('g1',resource_overrides={'parts':parts,'side':side})
                    backend=RobotPartSceneBuilder().build(spec,group=group)
                    self.assertEqual([r['type'] for r in group.scene.robots],['g1'])
                    visible=group.scene.visible_links[backend.robot['body']]
                    if parts=='hands':
                        self.assertTrue(visible)
                        self.assertTrue(all('hand' in n for n in visible))
                        if side=='right':self.assertFalse(any(n.startswith('left') for n in visible))
                    if parts=='arms':
                        self.assertTrue(visible)
                        self.assertFalse(any('hand' in n or 'hip' in n or 'torso' in n for n in visible))
                    image=group.scene.render(256,256)
                    self.assertEqual(image.shape,(256,256,4))
                    self.assertGreater(np.std(image[:,:,:3]),2.)
                    self.assertTrue(np.isfinite(group.scene.camera_target).all())
                finally:group.close()
