# Development Environment

The tested environment for the current test suite is:

```text
Python                 3.12.4
numpy                  1.26.4
scipy                  1.13.1
h5py                   3.11.0
trimesh                5.1.0
torch                  2.11.0+cu130
einops                 0.8.2
timm                   1.0.30
urchin                 0.0.30
roboticstoolbox-python 1.4.2
spatialmath-python     1.1.18
```

The minimal ABI fix is the NumPy pin in `requirements.txt`:

```text
numpy==1.26.4
```

The existing SciPy 1.13.1, h5py 3.11.0, and trimesh 5.1.0 combination imports and runs successfully with NumPy 1.26.4 on Python 3.12. No model, retargeting, coordinate, contract, or test changes are required.

The repository does not currently contain a fully locked dependency file. Until one is added, install the declared requirements into a clean Python 3.12 environment, then install the teleoperation requirements when running the arm tests:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-teleop.txt
python -m unittest discover -s tests -v
```

The current machine also has `opencv-python 4.13.0.92`, whose package metadata requires NumPy>=2. It is not declared in this repository and is not imported by the current tests because the MediaPipe adapter imports OpenCV lazily. Consequently `pip check` reports that unrelated optional-environment conflict even though the complete test suite passes. If camera/MediaPipe execution is added to the supported environment, use an OpenCV release compatible with NumPy 1.26 (or maintain that adapter in a separate environment); that is outside this test-environment repair.

## Verification record

On 2026-09-29, after installing the missing declared packages and applying the NumPy pin, the command below completed with `Ran 31 tests ... OK`:

```text
python -m unittest discover -s tests -v
```

The initial failure was NumPy 2.2.6 against binary extensions built for the NumPy 1.x ABI. After changing only NumPy to 1.26.4, the ABI errors disappeared. The other installed packages were retained at the versions listed above; `einops`, `timm`, and `roboticstoolbox-python==1.4.2` were installed because they were declared by the repository but missing from the environment.
