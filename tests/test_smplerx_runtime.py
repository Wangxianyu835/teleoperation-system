import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SmplerXRuntimeTests(unittest.TestCase):
    def test_missing_assets_fail_check_and_report_each_path(self):
        # Child processes in the managed Windows runner cannot write to the
        # per-user sandbox temp directory. Keep this test's scratch path in
        # the repository workspace instead.
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            report = Path(directory) / "report.json"
            command = [
                sys.executable,
                str(ROOT / "devtools" / "verify_smplerx_runtime.py"),
                "--check",
                "--worker-python",
                str(Path(directory) / "python.exe"),
                "--smplerx-root",
                str(Path(directory) / "source"),
                "--checkpoint",
                str(Path(directory) / "model.pth"),
                "--body-model-dir",
                str(Path(directory) / "body"),
                "--report",
                str(report),
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "missing")
            self.assertTrue(any("worker-python" in item for item in payload["missing"]))
            self.assertTrue(any("checkpoint" in item for item in payload["missing"]))

    def test_runtime_checker_import_is_lightweight(self):
        command = [
            sys.executable,
            "-c",
            "import sys; import devtools.verify_smplerx_runtime; "
            "print('torch' in sys.modules); print('mmcv' in sys.modules)",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.splitlines(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
