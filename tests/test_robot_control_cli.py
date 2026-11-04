import contextlib
import io
import unittest
from teleoperation.cli import build_parser


class RobotControlCliTests(unittest.TestCase):
    def parse(self,*args):
        return build_parser().parse_args(['robot','control',*args])

    def test_single_target_and_parts_are_explicit(self):
        args=self.parse('--robot','g1','--parts','hands','--side','right','--backend','dex')
        self.assertEqual(args.robot,'g1')
        self.assertIsNone(args.robots)
        self.assertEqual(args.parts,'hands')
        self.assertEqual(args._handler,'teleoperation.applications.robot_control:run')

    def test_missing_conflicting_duplicate_targets_fail(self):
        from teleoperation.applications.robot_control import preflight
        for argv in (['--backend','dex'], ['--robot','g1','--robots','h1_2','g1','--backend','dex']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): self.parse(*argv)
        for targets in (['g1'],['g1','g1']):
            with self.assertRaises(ValueError): preflight(self.parse('--robots',*targets,'--backend','dex'))

    def test_incompatible_source_or_weight_rejected_before_devices(self):
        from teleoperation.applications.robot_control import preflight
        cases=[['--robot','tron2a_l21','--source','visionpro','--ip','unused','--backend','dex'],
               ['--robot','g1','--backend','dex','--hand-checkpoint','g1','unused.pth'],
               ['--robot','g1','--backend','nn'],
               ['--robot','g1','--backend','dex','--parts','hands','--tron2a-calibration','unused.json']]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ValueError): preflight(self.parse(*args))
