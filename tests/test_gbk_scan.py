import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch


class GbkScanTests(unittest.TestCase):
    def test_owned_robot_sources_are_scanned_and_external_artifacts_are_skipped(self):
        from teleoperation.applications.diagnostics.check_gbk_safe import scan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for rel in ('src/teleoperation/robots/owned.py','assets/ignored.py','outputs/ignored.py','third_party/ignored.py','.venv/ignored.py'):
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('print('+repr(chr(0x1f642))+')\n',encoding='utf-8')
            rows=scan(root)
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0][0].endswith('owned.py'))
            self.assertEqual(rows[0][1:3],(1,chr(0x1f642)))

    def test_both_entries_share_output_and_strict_exit_code(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.py';path.write_text('print('+repr(chr(0x1f642))+')\n',encoding='utf-8')
            commands=([sys.executable,'-B','-m','teleoperation','tools','check-gbk-safe'],
                      [sys.executable,'-B',str(root/'devtools/check_gbk_safe.py')])
            outputs=[]
            for command in commands:
                completed=subprocess.run([*command,'--root',folder,'--strict'],cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=30)
                self.assertEqual(completed.returncode,1,completed.stderr)
                outputs.append(completed.stdout)
            self.assertEqual(*outputs)

    def test_import_has_no_stdout_side_effect_and_read_failure_is_reported(self):
        from teleoperation.applications.diagnostics import check_gbk_safe
        with patch('sys.stdout',new=Mock()) as stream:
            importlib.reload(check_gbk_safe)
            stream.reconfigure.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'blocked.py').touch()
            output=io.StringIO()
            with patch('builtins.open',side_effect=PermissionError('locked')),contextlib.redirect_stdout(output):
                self.assertEqual(check_gbk_safe.scan(folder),[])
            self.assertIn('READ_FAIL',output.getvalue())
