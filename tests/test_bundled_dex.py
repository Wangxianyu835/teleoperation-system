"""The backend must ship with source and built wheels, independently of pip dex."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


class BundledDexTests(unittest.TestCase):
    def test_runtime_loads_the_repository_backend_and_reports_its_actual_version(self):
        spec = importlib.util.find_spec('dex_retargeting')
        self.assertIsNotNone(spec)
        self.assertEqual(Path(spec.origin).resolve(), ROOT / 'src/dex_retargeting/__init__.py')
        import dex_retargeting
        from teleoperation.retargeting.hand.dex_native import VECTOR_PARAMETERS, NormalizedDexRetargeter
        self.assertEqual(dex_retargeting.__version__, '0.4.6')
        self.assertEqual(VECTOR_PARAMETERS['installed_version'], '0.4.6')
        if not all(importlib.util.find_spec(name) for name in ('nlopt', 'pinocchio')):
            self.skipTest('NLopt and Pinocchio runtime dependencies are unavailable')
        backend = NormalizedDexRetargeter(('g1',), ('left', 'right'))
        self.assertEqual(set(backend.solvers), {('g1', 'left'), ('g1', 'right')})

    def test_built_wheel_contains_backend_configuration_and_license(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                '--wheel-dir', directory, str(ROOT),
            ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheel = next(Path(directory).glob('*.whl'))
            with ZipFile(wheel) as archive:
                names = set(archive.namelist())
                self.assertIn('dex_retargeting/optimizer.py', names)
                self.assertIn('dex_retargeting/kinematics_adaptor.py', names)
                self.assertIn('dex_retargeting/configs/teleop/allegro_hand_right.yml', names)
                self.assertIn('dex_retargeting/LICENSE', names)
                self.assertIn('dex_retargeting/UPSTREAM.json', names)
                provenance = json.loads(archive.read('dex_retargeting/UPSTREAM.json'))
                self.assertEqual(provenance['version'], '0.4.6')


if __name__ == '__main__':
    unittest.main()
