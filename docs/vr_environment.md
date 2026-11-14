# Windows VR + retargeting environment

Use a separate `teleoperation-vr` Conda environment. Keep the existing
`teleoperation` environment available for ongoing training and camera work.
Commands below run in PowerShell from this repository root.

## Why the AVP wheel has a local version

Upstream `avp-stream==2.51` requires `opencv-python`, while this repository and
MediaPipe require `opencv-contrib-python`. Both distributions write `cv2` and
must not be installed together. New OpenCV versions also require NumPy 2 on
Python 3.10; this repository requires NumPy below 2.

`devtools/build_vr_wheel.py` verifies the upstream PyPI wheel's SHA-256 and creates
`avp-stream==2.51+opencvcontrib1`. It changes only the distribution version and
the OpenCV requirement to `opencv-contrib-python==4.11.0.86`, regenerates RECORD,
and checks that all AVP Python code is byte-for-byte identical. This is a local
packaging customization, not an official upstream release. It retains all
other AVP dependencies, including WebRTC, codecs, gRPC, and MuJoCo.

Sources:
- https://github.com/Improbable-AI/VisionProTeleop/blob/main/setup.py
- https://pypi.org/project/avp-stream/2.51/
- https://pypi.org/project/opencv-contrib-python/ (select only one OpenCV package)

## Create and install

Clone an existing environment to reuse its CUDA PyTorch installation:

```powershell
conda create -n teleoperation-vr --clone teleoperation -y
conda activate teleoperation-vr
python devtools/build_vr_wheel.py
python -m pip install --dry-run -r requirements-vr.txt
python -m pip install -r requirements-vr.txt
python -m pip install -e . --no-deps --no-build-isolation
```

For a fresh installation instead of cloning, create Python 3.10.20 and install
the build tools before following the wheel and requirements steps above:

```powershell
conda create -n teleoperation-vr python=3.10.20 pip -y
conda activate teleoperation-vr
python -m pip install "setuptools>=77.0.1" wheel
```

The CUDA wheel index is included in `requirements-vr.txt`. Cloning may preserve
unrelated packages; use the fresh route if those packages have conflicting
requirements. For either route, if the environment already has several OpenCV
distributions, close its running Python processes, uninstall all four OpenCV
flavours in this new environment, then reinstall only the pinned contrib wheel.

## Verify and connect

```powershell
python -m pip check
python -c "import numpy, cv2, torch, mediapipe, aiortc, av, grpc, mujoco; from avp_stream import VisionProStreamer; print('NumPy', numpy.__version__); print('OpenCV', cv2.__version__); print('Torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('VR imports OK')"
python -m unittest discover -s tests
python devtools/visionpro_record.py --self-test

# Use the IP shown by Tracking Streamer on the Vision Pro, on the same LAN.
$vrIp = '192.168.1.100'
python devtools/visionpro_record.py --list-keys --ip $vrIp
python devtools/visionpro_check.py --ip $vrIp --frames 120
```

The environment verification does not prove a live headset connection. A live
connection needs the user's headset and its address. This setup also does not
add the missing VR-to-L21 realtime application or its coordinate adapter.

Use this interpreter in PyCharm:
`D:\Anaconda\envs\teleoperation-vr\python.exe`.

For subsequent package additions, keep the constraints active:

```powershell
python -m pip install -c constraints-vr.txt <package>
```

Do not replace the local wheel with an unrestricted `pip install --upgrade
avp_stream`, which would reintroduce the standard OpenCV dependency. Downloaded
wheels, resolution reports, and a snapshot of the tested installed packages
belong in ignored `outputs/vr_environment/`.
