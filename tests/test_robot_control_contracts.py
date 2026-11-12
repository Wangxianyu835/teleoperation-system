import unittest
import numpy as np

from teleoperation.contracts.robot_control import RobotPartCommand, RobotCommandFrame
from teleoperation.robots.control_spec import load_robot_control_spec, resolve_robot_targets


class RobotControlContractTests(unittest.TestCase):
    def test_control_spec_accepts_actual_arm_joint_count_in_custom_urdf(self):
        import tempfile
        from pathlib import Path
        import xml.etree.ElementTree as ET
        source=load_robot_control_spec('g1',resource_overrides={'parts':'arms','side':'right'})
        root=ET.parse(source.urdf_path).getroot()
        root.find("joint[@name='right_wrist_yaw_joint']").set('type','fixed')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'six_joint_arm.urdf'
            ET.ElementTree(root).write(path)
            spec=load_robot_control_spec('g1',resource_overrides={'parts':'arms','side':'right','urdf_path':path})
            self.assertEqual(len(spec.parts['right_arm'].joint_names),6)
            spec.validate_frame(spec.reference_frame(0,0.))

    def test_target_selection_is_explicit_and_unique(self):
        self.assertEqual(resolve_robot_targets(robot='g1'), ('g1',))
        self.assertEqual(resolve_robot_targets(robots=['g1', 'tron2a_l21']), ('g1', 'tron2a_l21'))
        for options in ({}, {'robots': ['g1']}, {'robots': ['g1','g1']},
                        {'robot':'g1','robots':['g1','h1_2']}, {'robot':'unknown'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_robot_targets(**options)

    def test_actual_robot_joint_sets_and_coupling(self):
        for name, size in (('h1_2',38),('gr1_t2',36),('g1',28),('tron2a_l21',48)):
            spec = load_robot_control_spec(name)
            self.assertEqual(sum(len(p.joint_names) for p in spec.parts.values()),size)
            self.assertEqual(len(spec.parts['left_arm'].joint_names),7)
            spec.validate_frame(spec.reference_frame(0,0.))

    def test_hands_only_does_not_require_tron_arm_asset(self):
        spec=load_robot_control_spec('tron2a_l21',resource_overrides={'parts':'hands','side':'right','urdf_path':'absent.urdf'})
        self.assertEqual(set(spec.parts),{'right_hand'})

    def test_command_copies_and_freezes_inputs_and_checks_identity(self):
        spec=load_robot_control_spec('g1',resource_overrides={'parts':'hands','side':'right'})
        original=spec.reference_frame(0,0.)
        part=original.parts['right_hand']
        values=part.positions.copy()
        cmd=RobotPartCommand(part.joint_names,values,1.,'sensor_seconds',True,'tracking')
        values[:]=99
        self.assertFalse(np.any(cmd.positions==99))
        with self.assertRaises(ValueError): cmd.positions[0]=4
        parts={'right_hand':cmd}
        frame=RobotCommandFrame('g1',spec.spec_hash,1,1.,parts)
        parts.clear()
        self.assertEqual(len(frame.parts),1)
        with self.assertRaises(TypeError): frame.parts['left_hand']=cmd
        spec.validate_frame(frame)
        with self.assertRaises(ValueError):
            spec.validate_frame(RobotCommandFrame('h1_2',spec.spec_hash,1,1.,frame.parts))
        with self.assertRaises(ValueError):
            RobotPartCommand(('j',),np.array([np.nan]),0.,'seconds',True,'tracking')
        with self.assertRaises(ValueError):
            RobotCommandFrame('g1',spec.spec_hash,True,1.,frame.parts)
