# Unified SMPL morphology and PINK arm retargeting

The fitting stage uses one anatomical model and one geometry algorithm for every
robot. Robot differences come from `ArmSpec`: URDF geometry, named arm joints,
landmark frames, T-pose joint values, end-effector frames, wrist offsets,
anatomical fitting anchor, and optional declared reference up axis.
No fit or segment-scaling code branches on robot identity.

## Model assets and dependency

SMPL-family model assets are licensed separately and remain outside this source
repository. `BodyModelSpec` names the selected asset, gender, model type, and
exact beta dimension. Its identity combines the actual asset bytes with model
type, gender, and beta count. Requesting more shape coefficients than the actual
asset/loader supports fails; coefficients are never padded or truncated.

The local acceptance run uses the actual neutral SMPL-X NPZ asset under
`D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models`, 20 betas,
and official `smplx==0.1.28`. The official package was installed with
`pip install smplx --no-deps --target outputs/runtime`, using the existing
`dex-native` Python environment's PyTorch. Set `--smplx-module-dir` to that
package directory when it is not installed into the selected Python environment.
The SMPLer-X vendored package imports its separate inference configuration;
this acceptance run uses the official package without modifying vendor source.

`BodyModel.reconstruct` returns pelvis-relative native articulated joints and
matching global rotations. SMPL-X has 55 native kinematic joints here; facial,
vertex, and extra regressed landmarks are not included. Differentiable shape
reconstruction uses the actual model template, shape directions, joint
regressor, and parent tree. Tests compare its positions with the official full
model forward function, including body and hand articulation.

## Reference fitting

`applications.arm_fit.run(args)` performs the following operations:

1. Reject every existing destination profile or report before loading models.
   Load both arm sides for each selected robot and reuse one `BodyModel`.
2. Reconstruct the native zero-pose body with an explicit empty pose dictionary.
   Runtime root pose is not substituted with a fitting pose. The separately
   exported `reference_pose()` helper offers the requested Euler-derived upright
   reference for callers that explicitly choose it.
3. Define the lateral torso axis from right to left shoulder, orthogonalize the
   pelvis-to-shoulder-center axis against it, and form forward as lateral cross
   up. Compute a proper body-to-robot rotation between these orthonormal bases.
   Humanoids use the actual pelvis-to-shoulder-center up direction. Tabletop
   robots use a declared reference up axis, so their arbitrary URDF base offset
   never becomes an anatomical torso direction. Degenerate axes fail explicitly.
4. Transform robot targets into native body axes and optimize betas against
   pelvis, left/right shoulder, and left/right wrist for pelvis-anchored
   humanoids. Shoulder-anchored tabletop models fit only bilateral shoulder/wrist
   points relative to their shoulder center; the differentiable human anchor is
   recomputed as betas change. No tabletop pelvis/head landmark is invented.
   Include head only when the robot declares an actual head landmark. Adam minimizes mean squared
   Euclidean point distance, and retains the best observed shape. Elbows are
   measured for later scaling but are not additional shape-fit constraints.
5. Reconstruct the fitted body, apply the saved body-to-robot rotation, and
   derive independent segment length ratios along the declared root chain:
   pelvis to shoulder center for humanoids, then shoulder center to each
   shoulder, shoulder to elbow, and elbow to wrist. Shoulder-anchored tabletop
   models start directly at shoulder center, with no pelvis segment.
   Scaling preserves the current direction of each segment.
6. Derive wrist orientation offsets from the robot's physical T-pose wrist
   frames: `(body_to_robot @ human_wrist_rotation).T @ robot_wrist_rotation`.
7. Prepare every fit before publishing profiles. Save each profile exclusively
   as `<output_dir>/<robot_id>.json`; existing profiles are never overwritten.
   Each individual file uses exclusive creation. A filesystem failure during
   publication can still leave a subset; this is not a filesystem transaction.

The persisted profile stores body and robot identities, betas, reference pose,
body-to-robot rotation, anchor type and robot base-local anchor position,
independent segment ratios, wrist rotation offsets,
initial/final RMS, point errors, and optimizer iteration count. Loading can
require the expected body identity and robot specification hash. The report
also records the fitted point names and residual errors after segment scaling.

At runtime, reconstruct the observed root/body/hand pose with the fitted betas,
subtract its declared human anchor, rotate it into robot coordinates once,
scale the articulated segments, add the saved robot anchor, then
provide positions and calibrated wrist orientations to the shared PINK solver.
The saved reference pose describes calibration; it does not replace camera
root or local joint rotations.

## Measured fitting quality

A real run using all four bundled robot URDFs, the actual neutral SMPL-X asset,
20 betas, 300 Adam steps, learning rate 0.05, and CPU produced:

| Robot | Initial shape RMS | Final shape RMS | Shape RMS <= 20 mm | Maximum scaled landmark error |
| --- | ---: | ---: | :---: | ---: |
| H1-2 | 71.93 mm | 12.79 mm | Yes | 95.32 mm |
| GR1-T2 | 56.32 mm | 16.27 mm | Yes | 78.69 mm |
| G1 | 155.47 mm | 11.89 mm | Yes | 50.27 mm |
| TRON2A-L21 (shoulder anchor) | 136.55 mm | 17.96 mm | Yes | 99.03 mm |

These are morphology measurements, not robot tracking or hardware acceptance.
RMS is the square root of the mean squared Euclidean distance over the selected
points. Pelvis is included for humanoids and has zero error in the shared
relative frame. Tabletop RMS uses four shoulder/wrist points and excludes pelvis.
The maximum scaled error includes elbows and can exceed shape-fit RMS because
elbows are outside the shape objective and segment scaling preserves direction.
Correct segment lengths alone do not make the fitted skeleton coincide with
all robot T-pose landmarks.

TRON2A is a tabletop robot without an anatomical pelvis or head. Its declared
`shoulder_center` anchor fits its real bilateral shoulders/wrists and uses the
URDF base +Z reference up direction. This is a configuration of the shared
algorithm, rather than a synthetic pelvis offset. An earlier base-root fit
incorrectly treated its approximately 58 mm diagonal base-to-shoulder-center
vector as a human torso; that run failed the shape criterion and is superseded
by the anchor-aware artifacts below.
The optimizer is unconstrained and has no statistical beta prior: low geometric
error can require extreme coefficients. A small fit residual does not establish
a plausible human mesh, collision freedom, reachability, or accurate live
retargeting. Those require separate replay and robot-control checks.

The actual unconstrained coefficients are large:

| Robot | Beta L2 norm | Maximum absolute beta | Coefficients with absolute value > 3 |
| --- | ---: | ---: | ---: |
| H1-2 | 16.47 | 6.77 | 8 / 20 |
| GR1-T2 | 30.32 | 10.48 | 16 / 20 |
| G1 | 23.21 | 11.34 | 12 / 20 |
| TRON2A-L21 | 34.73 | 13.21 | 16 / 20 |

This table is a diagnostic of the geometric optimization, not a claim of
statistically plausible shape coefficients. The application reports beta norm,
maximum absolute value, and count above 3 on subsequent runs. There is no beta
regularizer in this implementation. Adding one would change the fit objective
and should be assessed against actual residuals rather than implied by the
20 mm point threshold.


Acceptance artifacts are generated locally under
`outputs/acceptance/unified-smpl-pink/fittings-v2/` and
`outputs/acceptance/unified-smpl-pink/fitting-report-v2.json`. The outputs directory
and licensed assets are not source-controlled fixtures. Re-run into a fresh
output directory to compare another iteration budget or model configuration.

## Reproducing the local fitting run

Run from the repository root in PowerShell. The selected Python needs NumPy,
SciPy, PyTorch, and Pinocchio, in addition to the separately installed SMPL-X
package and licensed model asset. The local environment below already supplies
the numerical and URDF dependencies. Other machines can substitute their Python
executable and asset path; the optional module directory identifies an explicit
SMPL-X installation.

```powershell
$fitPython = 'D:/2026/code/teleoperation-system-retargeting/.venv/dex-native/Scripts/python.exe'
$env:PYTHONPATH = 'src'
& $fitPython -m pip install smplx==0.1.28 --no-deps --target outputs/runtime
& $fitPython -m teleoperation arm fit `
  --robots h1_2 gr1_t2 g1 tron2a_l21 `
  --body-model-path 'D:/2026/code/teleoperation-system-retargeting/datasets/smpl_models' `
  --smplx-module-dir outputs/runtime `
  --body-model-type smplx --gender neutral --num-betas 20 `
  --fit-steps 300 --fit-learning-rate 0.05 --device cpu `
  --output-dir outputs/acceptance/unified-smpl-pink/fittings-new `
  --report outputs/acceptance/unified-smpl-pink/fitting-report-new.json
```

Use a fresh output destination. `run_cli` prints the summary as JSON and returns
zero after successful publication; a shape RMS over 20 mm is recorded as a failed
quality indicator, not converted into a successful geometric acceptance claim.
The report distinguishes shape fit from segment-scaled reference errors.

## Verification

`tests/test_smpl_arm_geometry.py` uses actual licensed assets when present and
checks 10/20 coefficient models, native body/hand forward equivalence, root
rotation, differentiable shape gradients, synthetic-target convergence,
asymmetric scaling, and persistence identity/overwrite behavior. Configure
`SMPL_MODEL_PATH` and `SMPLX_MODULE_DIR` for another environment.

`tests/test_arm_fit.py` checks torso basis construction, early overwrite refusal,
and real G1/TRON fits that preserve native zero poses and declared anatomical
anchors. The G1 check reconstructs the
physical wrist frames using the saved calibration offsets. Licensed-asset tests
skip when their local prerequisites are absent; a skip is not evidence of a
successful real-model run.

## Real-model CLI integration replay

`devtools/verify_unified_smpl_pipeline.py` generates synthetic raw SMPL-X root,
body, and hand parameters, reconstructs their derived observations with the
actual licensed model, stores RGB and worker packets through `capture_sample`,
and invokes the public robot control CLI. It runs each robot alone and all four
in one shared session, then loads the accepted HDF5 command recordings through
`load_robot_recording` to compare every control timestamp and arm command.
It does not replace model reconstruction with a test double and does not patch
resource loaders.

The verified run uses the v2 profiles, 30 new source poses at 30 Hz, identity
root rotations explicitly recorded, and small shoulder sinusoids. It produces
98 control frames at a nominal 100 Hz timeline per run. Actual reconstruction
count is 30 in every run, including the shared four-robot session. After fixing
the shape reduction order, every recorded joint command matches its standalone
counterpart exactly: maximum difference is 0.0 radians for all four robots.
Native H1/GR1/G1 scenes use original full meshes; TRON uses an explicitly selected
simplified proxy. TRON full-mesh and hardware acceptance remain excluded.

The regression behind this check matters: a batched float32 shape GEMM changed
native points by approximately 6e-8 meters when selecting one versus four
morphologies. Iterative redundant-arm IK amplified these tiny changes into
different trajectories. Geometry now reduces each shape's coefficients in the
same order regardless of batch count, while sharing cached projections and
batched articulated FK. Real 10/20-beta tests require exact single/batch joint
positions and rotations in addition to official forward equivalence.

CPU timings measured from the CLI's actual control loop in this replay:

| Selection | Median control time | 95th percentile | 99th percentile | Missed 10 ms budgets |
| --- | ---: | ---: | ---: | ---: |
| All four | 18.78 ms | 27.16 ms | 31.79 ms | 98 / 98 |
| H1-2 | 5.22 ms | 10.79 ms | 12.95 ms | 9 / 98 |
| GR1-T2 | 5.50 ms | 10.46 ms | 12.40 ms | 7 / 98 |
| G1 | 8.05 ms | 14.68 ms | 15.36 ms | 21 / 98 |
| TRON2A-L21 | 4.09 ms | 8.75 ms | 9.29 ms | 1 / 98 |

The nominal replay timeline is deterministic; this CPU run does not achieve a
100 Hz wall-clock budget for the four-robot selection. The software integration
pass records command equivalence and reconstruction cadence, and is independent
of timing, tracking accuracy, collisions, motor convergence, and hardware
acceptance. All source poses were generated, not inferred from a real camera.

The complete final report and HDF5 recordings are under
`outputs/acceptance/unified-smpl-pink/pipeline-v3/`. Earlier pipeline directories
record failed harness/numerical checks and are superseded by the final report.
To reproduce in the supplied environment:

```powershell
$env:PYTHONPATH = 'src;outputs/runtime;D:/2026/code/teleoperation-system-retargeting/outputs/acceptance/teleopbench-gr1-20261111/dependencies'
& $fitPython devtools/verify_unified_smpl_pipeline.py `
  --fitting-dir outputs/acceptance/unified-smpl-pink/fittings-v2 `
  --output-dir outputs/acceptance/unified-smpl-pink/pipeline-new
```

The extra dependency directory supplies the pinned PINK/QP packages for this
local runtime. Other installations should provide the corresponding installed
packages. The verifier refuses a nonempty output directory.

## Runtime installation without changing the base environment

`requirements-arm.txt` pins the optional shared arm runtime. The existing main
Python already provides compatible PyTorch and Pinocchio; install the optional
packages into a local target directory without upgrading those base libraries:

```powershell
& $fitPython -m pip install --no-deps --target outputs/runtime -r requirements-arm.txt
$env:PYTHONPATH = 'src;outputs/runtime'
```

This command intentionally assumes the main environment already has the
remaining numerical dependencies, including the QP package's Python
requirements. For the verified machine, the extra dependency directory shown
above supplies them. Avoid replacing the separate SMPLer-X worker environment
with the main control environment.

## Live camera launch and replay

These commands describe the live entry point; the end-to-end evidence above
uses generated raw pose recordings, not a successful live-camera test. The
worktree does not contain the main repository's ignored `.venv` assets, so pass
all worker paths explicitly. Choose fresh result/report filenames.

```powershell
$mainRepository = 'D:/2026/code/teleoperation-system-retargeting'
$env:PYTHONPATH = 'src;outputs/runtime;D:/2026/code/teleoperation-system-retargeting/outputs/acceptance/teleopbench-gr1-20261111/dependencies'
& $fitPython -m teleoperation robot control `
  --robots h1_2 gr1_t2 g1 tron2a_l21 `
  --source camera --parts arms --side both --backend dex --headless `
  --arm-fitting-dir outputs/acceptance/unified-smpl-pink/fittings-v2 `
  --smplx-module-dir outputs/runtime --body-device cpu `
  --smplerx-worker-python "$mainRepository/.venv/smplerx-worker/Scripts/python.exe" `
  --smplerx-root "$mainRepository/.venv/smplerx-assets/SMPLer-X" `
  --smplerx-checkpoint "$mainRepository/.venv/smplerx-assets/smpler_x_s32.pth.tar" `
  --smplx-model-dir "$mainRepository/datasets/smpl_models" `
  --camera-index 0 --control-hz 100 --target-max-age 0.3 `
  --geometry simplified `
  --record-results outputs/live-arm-control.h5 --report outputs/live-arm-control.json
```

The explicit simplified geometry applies to TRON; the three native robots keep
original meshes. Full geometry is the normal default and must fail when TRON's
mesh resources are unavailable. Remove TRON from the selection to exercise the
three full-mesh scenes. The all-four CPU run missed every measured 10 ms budget;
this launch example's 100 Hz setting is a requested cadence, not evidence that
this machine meets it.

For the recorded integration source, replace the worker/camera options with
`--input-body outputs/acceptance/unified-smpl-pink/pipeline-v3/synthetic-raw-real-smpl.h5`.
This bypasses camera and inference and retains the source's recorded timeline.
To replay accepted joint commands without reconstruction or solving:

```powershell
& $fitPython -m teleoperation robot replay `
  --robots h1_2 gr1_t2 g1 tron2a_l21 --headless --geometry simplified `
  --input outputs/acceptance/unified-smpl-pink/pipeline-v3/all.h5
```

## Separate full-mesh physical simulation evidence

[The PINK acceptance report](../outputs/acceptance/unified-smpl-pink/pink-acceptance.json)
measures actual PyBullet link poses after driving reachable FK-generated static
targets and holding the last command for two additional physical seconds. This
is a different test from the raw-body CLI integration replay. With original
full meshes, fixed roots, gravity, and position motors, it reports:

| Robot | Maximum wrist position error | Maximum wrist orientation error | Physical simulation threshold |
| --- | ---: | ---: | --- |
| H1-2 | 0.088 mm | 0.011 degrees | Pass: <5 mm and <3 degrees |
| GR1-T2 | 0.410 mm | 0.195 degrees | Pass: <5 mm and <3 degrees |
| G1 | 0.271 mm | 0.117 degrees | Pass: <5 mm and <3 degrees |
| TRON2A-L21 | Unavailable | Unavailable | Full meshes missing |

TRON's required mesh set includes missing `base_Link.STL`,
`proximal_pitch_L_Link.STL`, `elbow_L_Link.STL`, and other declared resources.
The report retains the complete missing-file list. Its full-mesh physical check
is unavailable and the overall full-four physical acceptance is false. The
explicit proxy in the CLI integration test does not replace this missing
full-mesh evidence. No hardware execution is claimed by either report.

## Reproducibility manifest

[The generated local manifest](../outputs/acceptance/unified-smpl-pink/reproducibility-manifest.json)
records the Python executable/version, imported numerical package versions and
module entry paths/hashes, selected model asset SHA256/spec identity, current
URDF and fitting hashes, relevant implementation-file hashes, command-recording
hashes, and the explicit worker-resource locations. These are local code and
asset paths; the manifest contains no credentials or copied model weights.

The verified runtime imports NumPy 2.2.6, SciPy 1.13.1, PyTorch 2.14.1+cu130,
Pinocchio 3.3.1, PINK 4.4.0, qpsolvers 4.13.0, quadprog 0.1.13, and SMPL-X 0.1.28.
CPU was explicitly selected for body reconstruction. GPU-enabled PyTorch's
presence does not mean the measured control loop ran on the GPU. Model weights
and URDFs are referenced by their actual hashes, rather than inferred from
filenames or a generic package version.

## Real S32 inference startup check

A separate isolated-worker image smoke uses the supplied upstream example image
`assets/smpler_x_vis1.jpg`, the actual S32 checkpoint, and licensed body models.
It does not use a camera or establish live end-to-end latency. The original
cold-start run spent 4438 ms on its first inference. Because person selection
had already recorded that frame's capture timestamp, the next capture exceeded
the existing two-second operator absence window and correctly locked identity.
The model had been reported ready before its cold inference work had completed.

Worker initialization now warms detector, S32 forward, and the same derived
SMPL-X body pass on dummy inputs, then synchronizes CUDA before accepting any
person frames. Warmup neither selects a person nor samples/freezes operator
shape. Operator absence, IoU, session and five-sample shape rules are unchanged.
The helper for derived body output is shared by startup and runtime, so its
parameters remain identical to the original inference path.

[The real warm-start smoke report](../outputs/acceptance/unified-smpl-pink/smplerx-real-smoke-warm.json)
passed on RTX 4050 with isolated-worker PyTorch 1.13.0+cu117. Startup warmup took
1.578 seconds. Seven real-image inference times were approximately
219, 203, 157, 187, 188, 218, and 250 ms. Samples 0-3 remained in normal shape
calibration; samples 4-6 produced valid bilateral output. The checkpoint hash
was unchanged and the output includes 55 native joint positions/rotations plus
raw root/body/hand parameters. This is successful real-model image inference,
not proof of camera capture, live operator identity behavior, or hardware control.
