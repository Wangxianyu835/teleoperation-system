import unittest
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.robot_backends import SimulationGroup


class RobotSceneSelectionTests(unittest.TestCase):
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
