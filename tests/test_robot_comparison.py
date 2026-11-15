from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_part_scene import RobotPartSceneBuilder
from teleoperation.simulation.robot_backends import SimulationGroup
from teleoperation.simulation.robot_output import RobotOutput
from teleoperation.retargeting.arm.calibration import ArmCalibration

CALIBRATION=Path(__file__).parent/'fixtures/tron2a_camera_neutral.synthetic.json'


class RobotComparisonTests(unittest.TestCase):
    def test_borrowed_tron_scene_does_not_disconnect_group(self):
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        from teleoperation.paths import DEFAULT_TRON2A_URDF
        group=SimulationGroup(gui=False)
        try:
            world=Tron2ADualArmWorld(DEFAULT_TRON2A_URDF,gui=False,display='simplified',
                attach_hands=False,client_id=group.scene.cid,owns_client=False)
            world.close()
            self.assertTrue(group.scene.alive())
        finally:group.close()

    def test_four_targets_share_connection_and_only_one_physics_advance(self):
        names=('h1_2','gr1_t2','g1','tron2a_l21')
        group=SimulationGroup(gui=False)
        try:
            specs={n:load_robot_control_spec(n) for n in names}
            calibration=ArmCalibration.load(CALIBRATION)
            backends={n:RobotPartSceneBuilder().build(s,group=group,base_pose=(i*2.2,0.,.8),
                         calibration=calibration if n=='tron2a_l21' else None,geometry='simplified')
                      for i,(n,s) in enumerate(specs.items())}
            output=RobotOutput(backends)
            self.assertEqual({b.scene.cid for b in backends.values()},{group.scene.cid})
            self.assertEqual([r['type'] for r in group.scene.robots],list(names))
            with patch.object(group.scene.p,'stepSimulation',wraps=group.scene.p.stepSimulation) as steps:
                output.submit({n:s.reference_frame(0,0.) for n,s in specs.items()},steps=8)
                self.assertEqual(steps.call_count,8)
            backends['g1'].close()
            self.assertTrue(group.scene.alive())
            state=output.read_state()
            for n,s in specs.items():
                self.assertEqual(set(state[n]),set(s.parts))
                self.assertTrue(all(np.isfinite(v).all() for v in state[n].values()))
        finally:group.close()

    def test_tron_hands_require_no_body_or_calibration(self):
        group=SimulationGroup(gui=False)
        try:
            spec=load_robot_control_spec('tron2a_l21',resource_overrides={'parts':'hands','side':'right','urdf_path':'missing'})
            b=RobotPartSceneBuilder().build(spec,group=group)
            self.assertIsNone(b.arm_controller)
            self.assertEqual(len(b.part_bodies),1)
            self.assertEqual(set(b.part_bodies),{'right_hand'})
            self.assertEqual([r['type'] for r in group.scene.robots],['tron2a_l21'])
            b.write(spec.reference_frame(0,0.))
            group.advance(8)
        finally:group.close()

    def test_native_parts_display_matrix(self):
        for name in ('h1_2','gr1_t2','g1'):
            for parts in ('arms','hands'):
                group=SimulationGroup(gui=False)
                try:
                    spec=load_robot_control_spec(name,resource_overrides={'parts':parts,'side':'right'})
                    b=RobotPartSceneBuilder().build(spec,group=group)
                    visible=group.scene.visible_links[b.robot['body']]
                    self.assertFalse(any('hip' in n or n.startswith('left_') for n in visible))
                    self.assertTrue(visible)
                    image=group.scene.render(128,128)
                    self.assertGreater(image[:,:,:3].std(),2.)
                finally:group.close()
