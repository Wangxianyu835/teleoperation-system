from pathlib import Path
import tempfile
import unittest
import numpy as np


class ArrayRecordingTests(unittest.TestCase):
    def test_npy_frames_and_npz_arrays_preserve_values_and_release_files(self):
        from teleoperation.data.array_recording import read_recording,read_arrays
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            hand=np.arange(75,dtype=float).reshape(25,3)
            np.save(root/'frames.npy',{'left':hand})
            np.savez(root/'frames.npz',timestamps=np.array([0.,1.]),left=hand)
            np.testing.assert_array_equal(read_recording(root/'frames.npy').item()['left'],hand)
            arrays=read_arrays(root/'frames.npz')
            (root/'frames.npz').unlink()
            np.testing.assert_array_equal(arrays['left'],hand)
            np.testing.assert_array_equal(arrays['timestamps'],[0.,1.])
