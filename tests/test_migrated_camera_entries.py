"""Production camera aliases must use the same fitted SMPL/PINK entry."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

class CameraEntryMigrationTests(unittest.TestCase):
    def test_realtime_arms_delegate_with_explicit_robot_selection_and_fit(self):
        from teleoperation.simulation import realtime_teleop
        args=realtime_teleop.build_parser().parse_args(['--robot','g1','--robot','h1_2',
            '--headless','--input-body','raw.h5','--arm-fitting-dir','fits','--control-hz','50'])
        with patch('teleoperation.applications.robot_control.run',return_value=17) as common:
            self.assertEqual(realtime_teleop.run(args),17)
        target=common.call_args.args[0]
        self.assertEqual(target.robots,('g1','h1_2'))
        self.assertEqual(target.parts,'both')
        self.assertEqual(target.arm_fitting_dir,Path('fits'))
        self.assertEqual(target.control_hz,50)
        self.assertEqual(target.input_body,Path('raw.h5'))

    def test_tron_camera_delegates_with_one_common_robot_identity(self):
        from teleoperation.cli import build_parser
        from teleoperation.applications.mediapipe_tron2a import run_camera
        args=build_parser().parse_args(['dual','camera','--arms-only','--arm-fitting-dir','fits','--no-gui'])
        with patch('teleoperation.applications.robot_control.run',return_value=19) as common:
            self.assertEqual(run_camera(args),19)
        target=common.call_args.args[0]
        self.assertEqual(target.robot,'tron2a_l21')
        self.assertEqual(target.parts,'arms')
        self.assertTrue(target.headless)
        self.assertEqual(target.arm_fitting_dir,Path('fits'))

    def test_missing_fit_and_derived_point_npz_rejected_before_resources(self):
        from teleoperation.simulation import realtime_teleop
        for options,reason in (([], 'arm-fitting-dir'),(['--input-npz','old.npz','--arm-fitting-dir','fits'],'raw SMPL')):
            args=realtime_teleop.build_parser().parse_args(['--robot','g1',*options])
            with patch('teleoperation.applications.robot_control.run') as common:
                with self.assertRaisesRegex(ValueError,reason):realtime_teleop.run(args)
            common.assert_not_called()

    def test_hands_only_npz_still_runs_without_an_arm_solver(self):
        import contextlib
        import io
        import json
        import tempfile
        import numpy as np
        from teleoperation.simulation import realtime_teleop as app
        from tests.smplerx_fixtures import row
        from teleoperation.data.body_recording import replay_observation
        from tests.test_native_hand_replay import stub_camera_backend
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); capture=folder/'hands.npz'; report=folder/'report.json'
            raw=[replay_observation(row(i)).hands for i in range(5)]
            arrays={'timestamps':np.arange(5)/30}
            for side in ('left','right'):
                arrays[side+'_hand']=np.stack([r.hands[side] for r in raw])
                arrays['world_'+side+'_hand']=np.stack([r.world_hands[side] for r in raw])
            np.savez(capture,**arrays)
            args=app.build_parser().parse_args(['--robot','g1','--side','left','--no-arms',
                '--input-npz',str(capture),'--report',str(report),'--headless','--no-preview','--print-every','0'])
            with patch.object(app,'build_camera_retargeter',side_effect=stub_camera_backend), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run(args),0)
            result=json.loads(report.read_text(encoding='utf-8'))
            self.assertFalse(result['arms_enabled'])
            self.assertEqual(result['processed_frames'],5)
            self.assertEqual(result['simulation_timing']['physics_steps'],40)

    def test_legacy_dual_export_and_custom_adapter_fail_explicitly(self):
        from teleoperation.applications import dual_export, dual_realtime
        with self.assertRaisesRegex(ValueError,'robot control'):
            dual_export.main(SimpleNamespace())
        with self.assertRaisesRegex(ValueError,'robot control'):
            dual_realtime.main(SimpleNamespace())

if __name__=='__main__':unittest.main()
