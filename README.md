# Teleoperation and Retargeting

This project detects human hand landmarks with a camera.
It converts the landmarks to L21 joint angles.
PyBullet shows the hand motion during replay or real-time operation.

For normal operation, use this sequence:

**Record hand motion → Align landmarks automatically → Export joint angles with an existing model → Replay L21 motion.**

If you have a compatible model, you do not need to train a new model.

- [1. Prepare the environment](#1-prepare-the-environment)
- [2. Record hand motion](#2-record-hand-motion)
- [3. Export and replay joint angles](#3-export-and-replay-joint-angles)
- [4. Train a model or use real-time operation](#4-train-a-model-or-use-real-time-operation)
- [5. Find other functions and source files](#5-find-other-functions-and-source-files)

## 1. Prepare the environment

The commands in this document use **Windows PowerShell**.
The examples use local paths on the development computer.

If the `teleoperation` environment is already installed, go to Section 2.
For the first installation, do these steps:

1. Open the project directory.
2. Create the Conda environment.
3. Activate the environment.
4. Install the project.

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda env create -f environment.yml
conda activate teleoperation
python -m pip install -e . --no-deps --no-build-isolation
```

Put the MediaPipe model for hand detection at `datasets/hand_landmarker.task`.
Section 3 also requires a shared model checkpoint in `.pth` format for `world` coordinates.
A checkpoint contains the trained model weights.
The program does not download these models automatically.

The `$python` variable below selects the local Conda interpreter.
If your environment has a different installation path, change this variable.

## 2. Record hand motion

1. Copy the complete command block below into PowerShell.
2. Execute the command block.
3. Keep your hands in the camera image.
4. Open your hands.
5. Close your hands.
6. Bend individual fingers.
7. To stop and save the data, use one of these actions:
   - Press `q` or `Esc` in the preview window.
   - Close the preview window.
   - Press `Ctrl+C` in the terminal.

The example records both hands with camera `0`.

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'

$session = Get-Date -Format 'yyyyMMdd_HHmmss'
$camera = 0
$handAsset = 'datasets/hand_landmarker.task'
$raw = "datasets/raw/camera_$session.h5"
$aligned = "datasets/raw/aligned_camera_$session.h5"

if (-not (Test-Path $handAsset)) { throw "Missing MediaPipe model: $handAsset" }

& $python -m teleoperation hand record `
  --landmark-space world --hand-side both `
  --camera-index $camera --model-asset-path $handAsset `
  --width 640 --height 480 --fps 30 --frames 0 `
  --output $raw --aligned-output $aligned
if ($LASTEXITCODE -ne 0) { throw 'Recording failed; check the output above.' }
```

To change the recording settings, use this table:

| Task | Instruction |
|---|---|
| Select a different camera | Set `$camera` to the camera index, for example, `1` or `2`. |
| Record one hand | Replace `--hand-side both` with `--hand-side left` or `--hand-side right`. |
| Stop at a specified frame count | Replace `--frames 0` with `--frames 900`. The recording time depends on the actual capture rate. |
| Record without a preview | Add `--no-preview`. To stop, press `Ctrl+C`. |

The H5 files contain **landmarks and timestamps**.
They do not contain camera video.
When recording stops normally, the program aligns the landmarks automatically.
You do not need a separate `hand align` command.

| Variable | File contents | Use |
|---|---|---|
| `$raw` | 21 raw landmarks per hand in `normalized` and `world` coordinates, from the same frames | Inspect the input or align the landmarks again. |
| `$aligned` | 25 landmarks per hand, aligned from `world` coordinates to a palm-local frame | Train a model or calculate joint angles offline. |
| `$angles` (Section 3) | 18 L21 joint angles per hand, with validity flags for the left and right hands | Replay motion in the simulation. |

A palm-local frame is a coordinate system relative to the palm.
The filenames contain the recording time to prevent replacement of data from earlier sessions.

**Use the same terminal for Sections 3 and 4.**
If you open a new terminal, set `$python` and `PYTHONPATH` again.
Also set the variables for the files and settings that the selected command uses.

## 3. Export and replay joint angles

### 3.1 Export joint angles with an existing model

The example checkpoint is a local shared model for `world` coordinates.
If this file is not available, select your own compatible checkpoint.
If you do not have a compatible checkpoint, first train a model as given in Section 4.1.

After recording stops, execute this command block:

```powershell
$checkpoint = 'outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_shared_20261014_000112/model_best.pth'
$angles = "outputs/hand_retargeting/camera_angles_$session.h5"

if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }

& $python -m teleoperation hand export `
  --input $aligned --checkpoint $checkpoint --output $angles --device auto
if ($LASTEXITCODE -ne 0) { throw 'Angle export failed; check the output above.' }
```

Export and replay do not use the camera.

**Use a checkpoint for `world` coordinates with `world` data.**
Do not use an old checkpoint for `normalized` coordinates or a Vision Pro / VR checkpoint with this data.
Do not change file metadata to make incompatible files appear compatible.

### 3.2 Replay both L21 hands in one window

To show both palms from the front, execute this command:

```powershell
& $python devtools/replay_capture_to_l21.py `
  --angles-h5 $angles --scene hands --hand both --view front `
  --gui --fps 30 --render-every 0 --hold-seconds 30
```

The window shows only the two L21 hands.
Both palms face the window.
After replay stops, the window shows the final pose for 30 seconds.

| Task | Instruction |
|---|---|
| Show only the right hand | Replace `--hand both` with `--hand right`. |
| Replay only the first 900 frames | Add `--max-frames 900`. |
| Set the maximum replay frame rate | Change the value of `--fps`. |

This command shows the exported native joint angles directly.
It does not apply the output filter or collision checks from the real-time application.

To replay one right L21 hand with the standard command, execute this command:

```powershell
& $python -m teleoperation replay l21-hand `
  --file $angles --hand right --render --speed 1.0
```

The standard command uses timestamps to control replay speed.
To show both hands, replace `--hand right` with `--hand both`.
To replay at half speed, replace `--speed 1.0` with `--speed 0.5`.

**Both commands require the `$angles` file.**
Do not supply `$raw` or `$aligned` directly.
For a later replay, set `$angles` to the path of the saved joint-angle file.

### 3.3 Show hand motion on a robot (optional)

Select one of these commands:

- `native-hand` uses the robot's original hand.
  The mapping omits L21 degrees of freedom that this hand cannot reproduce.
- `mounted-hand` attaches an L21 hand to the robot model.

To replay the original right hand of H1-2 continuously, execute this command:

```powershell
& $python -m teleoperation replay native-hand `
  --file $angles --robot h1_2 --hand right --view right-hand --render --loop 0
```

The view shows a close-up of the right hand.
The other joints stay in their neutral positions.

To replay an L21 right hand attached to H1-2, execute this command:

```powershell
& $python -m teleoperation replay mounted-hand `
  --file $angles --robot h1_2 --hand right --render --speed 1.0
```

The `--robot` option also accepts `gr1_t2` and `g1`.
All these replay commands use PyBullet simulation.
They do not connect to a physical robot.

## 4. Train a model or use real-time operation

### 4.1 Train a shared model (optional)

If you do not have compatible model weights, use `$aligned` from Section 2 to train a shared model.
The same shared model can use data from the left hand, the right hand, or both hands.
Training does not require robot joint-angle labels.

To train the model, execute this command block:

```powershell
$run = "world_$session"
$checkpointRoot = 'outputs/hand_retargeting/checkpoint'

& $python -m teleoperation hand train `
  --input $aligned --run-name $run --hand-side shared `
  --checkpoint-root $checkpointRoot --device auto `
  --epochs 100 --batch-size 64 --learning-rate 0.0001
if ($LASTEXITCODE -ne 0) { throw 'Training failed; check the output above.' }

$checkpoint = "$checkpointRoot/models/twohand_h5/linker/$run/model_best.pth"
```

After training completes successfully, use this sequence:

1. Keep `$checkpoint` set to the new model path.
2. Set `$angles` to the required output path.
3. Execute the `hand export` command from Section 3.1 with the new checkpoint.
4. Execute a replay command from Section 3.2.

Do not replace the new checkpoint path with the example checkpoint path from Section 3.1.
The [workflow reference](docs/workflow-reference.md) gives the data split, loss functions, and model contract.

### 4.2 Control L21 hands in real time (optional)

This command uses the selected `$checkpoint`, `$camera`, and `$handAsset` values.
It does not require a recording step.

To start real-time operation, execute this command:

```powershell
& $python -m teleoperation.applications.realtime_hand_sim `
  --backend checkpoint --checkpoint $checkpoint --landmark-space world --device cpu `
  --camera-index $camera --model-asset-path $handAsset `
  --scene hands --hand both --view front --frames 0
```

To stop, use one of these actions:

- Press `q` or `Esc` in the camera window.
- Close the camera window.
- Press `Ctrl+C` in the terminal.

This application does not save an H5 file of joint angles.
The [hand output reference](docs/hand-command-safety.md) describes the filter, angular velocity limits, and collision checks.

## 5. Find other functions and source files

| Task or subject | Reference |
|---|---|
| Manual alignment, training parameters, geometry backend, original robot hands, and task simulation | [Workflow reference](docs/workflow-reference.md) |
| Synchronized capture of hand, shoulder, elbow, and wrist landmarks in metric coordinates | [World landmark capture](docs/world-landmark-capture.md) |
| TRON2A arm calibration, real-time control of both arms and hands, and combined offline commands | [Body and arm workflows](docs/workflow-reference.md#4-扩展身体采集双臂和任务仿真) |
| Vision Pro recording and offline data | [Headset recording](docs/visionpro-offline-capture.md) |
| Vision Pro / VR inference and L21 replay | [TransHandR VR workflow](docs/transhandr-vr.md) |
| H5 formats, coordinates, and model compatibility | [Data contracts](docs/contracts.md) |
| Module responsibilities and dependencies | [Architecture](docs/architecture.md) |
| Progress presentations | [Presentation material](present/README.md) |

For temporary changes, use command-line parameters.
For permanent defaults, use these files:

| Defaults | Source file |
|---|---|
| Training and models | [`config.py`](src/teleoperation/retargeting/hand/config.py) |
| Unified command-line interface (CLI) | [`schemas.py`](src/teleoperation/cli/schemas.py) |
| Resource paths | [`paths.py`](src/teleoperation/paths.py) |

To see all parameters for a command, add `--help`.
For example, execute one of these commands:

```powershell
& $python -m teleoperation hand record --help
& $python -m teleoperation hand export --help
& $python -m teleoperation replay l21-hand --help
```
