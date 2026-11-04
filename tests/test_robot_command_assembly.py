import unittest
from dataclasses import replace
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.contracts.robot_control import RobotPartCommand
from teleoperation.applications.robot_commands import RobotCommandAssembler


class RobotCommandAssemblyTests(unittest.TestCase):
    def test_independent_part_holding_preserves_origin_time(self):
        spec=load_robot_control_spec('g1')
        initial=spec.reference_frame(0,0.)
        assembler=RobotCommandAssembler({'g1':spec},{'g1':initial})
        first=assembler.assemble('g1',0,0.,{})
        q=first.parts['left_arm'].positions.copy(); q[0]=.1
        update=RobotPartCommand(first.parts['left_arm'].joint_names,q,101.,'camera_monotonic_s',True,'tracking')
        second=assembler.assemble('g1',1,.033,{'left_arm':update})
        self.assertAlmostEqual(second.parts['left_arm'].positions[0],.1)
        np.testing.assert_array_equal(second.parts['left_hand'].positions,first.parts['left_hand'].positions)
        self.assertFalse(second.parts['left_hand'].observed)
        third=assembler.assemble('g1',2,.066,{})
        self.assertEqual(third.parts['left_arm'].source_timestamp,101.)
        self.assertEqual(third.parts['left_arm'].status,'holding')
        self.assertFalse(third.parts['left_arm'].observed)

    def test_bad_update_does_not_advance_state_and_targets_are_isolated(self):
        specs={n:load_robot_control_spec(n,resource_overrides={'parts':'hands','side':'right'}) for n in ('g1','h1_2')}
        assembler=RobotCommandAssembler(specs,{n:s.reference_frame(0,0.) for n,s in specs.items()})
        a=assembler.assemble('g1',0,0.,{})
        wrong=specs['h1_2'].reference_frame(0,0.).parts['right_hand']
        with self.assertRaises(ValueError): assembler.assemble('g1',1,.1,{'right_hand':wrong})
        b=assembler.assemble('g1',1,.1,{})
        np.testing.assert_array_equal(a.parts['right_hand'].positions,b.parts['right_hand'].positions)
        with self.assertRaises(ValueError): assembler.assemble('g1',1,.1,{})
        other=assembler.assemble('h1_2',0,0.,{})
        self.assertEqual(other.frame_id,0)
