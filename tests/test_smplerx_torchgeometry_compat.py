import subprocess
import unittest
from pathlib import Path


class TorchgeometryCompatibilityTests(unittest.TestCase):
    def test_legacy_mask_fix_preserves_identity_and_right_angle_rotations(self):
        root = Path(__file__).resolve().parents[1]
        python = root / '.venv/smplerx-worker/Scripts/python.exe'
        if not python.is_file():
            self.skipTest('isolated worker runtime is not installed')
        program = '''
import torch
import torchgeometry as tgm
from external.smplerx_worker.torchgeometry_compat import configure_torchgeometry
configure_torchgeometry()
r = torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],
                  [[1.,0.,0.,0.],[0.,0.,-1.,0.],[0.,1.,0.,0.]]])
angles = tgm.rotation_matrix_to_angle_axis(r)
assert torch.allclose(angles, torch.tensor([[0.,0.,0.],[1.57079632679,0.,0.]]), atol=1e-6), angles
'''
        result = subprocess.run([str(python), '-c', program], cwd=root, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
