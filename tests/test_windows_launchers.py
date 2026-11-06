import contextlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch


class WindowsLaunchersTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        path=patch.object(sys,'path',[str(self.root/'devtools'),*sys.path]);path.start();self.addCleanup(path.stop)
        self.runtime=importlib.import_module('windows_runtime')

    def invoke(self,name,flags):
        module=importlib.import_module(name)
        from teleoperation.simulation import realtime_teleop
        with patch.object(self.runtime,'prepare_environment'),patch.object(self.runtime,'redirect_caches'), \
             patch.object(self.runtime,'relaunch_with_canonical_python',return_value=None), \
             patch.object(self.runtime,'gui_support',return_value=(True,'test')), \
             patch.object(realtime_teleop,'main',return_value=0) as run,contextlib.redirect_stdout(io.StringIO()):
            status=module.main(flags)
        return status,run

    def test_presets_keep_target_backend_checkpoint_worker_and_report_arguments(self):
        for name,arms in (('run_realtime_windows',False),('run_realtime_arm_windows',True),('run_realtime_arm_elbow_follow_windows',True)):
            with self.subTest(launcher=name),tempfile.TemporaryDirectory() as folder:
                report=Path(folder)/'report.json'
                status,run=self.invoke(name,['--robot','g1','--backend','nn','--hand-checkpoint','g1','shared.pth',
                    '--smplerx-checkpoint','custom-body.pth','--report',str(report),'--side','left','--no-preview','--headless'])
                self.assertEqual(status,0)
                args=run.call_args.args[0]
                self.assertEqual(args.robot,['g1']);self.assertEqual(args.backend,'nn')
                self.assertEqual(args.hand_checkpoint,[['g1','shared.pth']])
                self.assertEqual(args.smplerx_checkpoint,Path('custom-body.pth'))
                self.assertEqual(args.report,report);self.assertEqual(args.side,'left')
                self.assertEqual(args.no_arms,not arms)

    def test_elbow_explicit_mode_wins_and_off_switch_is_not_forwarded_to_parser(self):
        for extra,mode in (([],'operator'),(['--off'],'off'),(['--off','--arm-elbow-follow','operator'],'operator')):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as folder:
                status,run=self.invoke('run_realtime_arm_elbow_follow_windows',[
                    '--robot','g1','--headless','--report',str(Path(folder)/'report.json'),*extra])
                self.assertEqual(status,0)
                self.assertEqual(run.call_args.args[0].arm_elbow_follow,mode)

    def test_no_target_fails_before_the_camera_or_simulation_entry(self):
        status,run=self.invoke('run_realtime_arm_windows',['--headless'])
        self.assertEqual(status,2)
        run.assert_not_called()

    def test_relaunch_preserves_the_original_launcher_and_exit_code(self):
        canonical=Path(sys.executable)
        script=self.root/'devtools/run_realtime_arm_elbow_follow_windows.py'
        with patch.object(self.runtime,'missing_modules',return_value=['mediapipe']), \
             patch.object(self.runtime,'CANONICAL_PYTHON',canonical), \
             patch.object(self.runtime.sys,'executable',str(self.root/'wrong-python.exe')), \
             patch.dict(self.runtime.os.environ,{self.runtime.REEXEC_FLAG:'0'}), \
             patch.object(self.runtime.subprocess,'run',return_value=Mock(returncode=7)) as process, \
             contextlib.redirect_stdout(io.StringIO()):
            result=self.runtime.relaunch_with_canonical_python(['--backend','nn','--robot','g1'],script_path=script)
        self.assertEqual(result,7)
        self.assertEqual(process.call_args.args[0],[str(canonical),str(script),'--backend','nn','--robot','g1'])
        self.assertEqual(process.call_args.kwargs['env'][self.runtime.REEXEC_FLAG],'1')

    def test_interpreter_selection_follows_the_last_backend_option(self):
        self.assertEqual(self.runtime._selected_backend(['--backend=dex','--backend','nn']),'nn')
        self.assertEqual(self.runtime._selected_backend(['--backend','nn','--backend=dex']),'dex')

    def test_check_mode_uses_smplerx_paths_without_starting_control(self):
        from teleoperation.simulation import realtime_teleop
        module=importlib.import_module('run_realtime_arm_windows')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('python.exe','s32.pth'):(root/name).touch()
            for name in ('source','models'):(root/name).mkdir()
            flags=['--check','--headless','--smplerx-worker-python',str(root/'python.exe'),
                   '--smplerx-root',str(root/'source'),'--smplerx-checkpoint',str(root/'s32.pth'),'--smplx-model-dir',str(root/'models')]
            with patch.object(self.runtime,'prepare_environment'),patch.object(self.runtime,'redirect_caches'), \
                 patch.object(self.runtime,'relaunch_with_canonical_python',return_value=None), \
                 patch.object(self.runtime,'run_check',return_value=0) as check, \
                 patch.object(realtime_teleop,'main') as run,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(flags),0)
                (root/'s32.pth').unlink()
                self.assertEqual(module.main(flags),3)
                run.assert_not_called()
                self.assertEqual(check.call_count,2)
