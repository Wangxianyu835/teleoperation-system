import unittest
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.robot_backends import SimulationGroup


class RobotSceneSelectionTests(unittest.TestCase):
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
