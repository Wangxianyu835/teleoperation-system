# SMPLer-X camera arm integration: implementation status

Status: incomplete. The real S32 backend passes single-image inference and a
real subprocess round trip on this RTX 4050. Camera command migration, recording
and full runtime acceptance are still incomplete. Do not use the current branch
as a completed camera arm release.

## Upstream and model assets

Official source: https://github.com/MotrixLab/SMPLer-X

The source inspected on 2026-10-30 is commit
`064baef0e4ab5277a3297691bc1d46ea5412586f`. The working reference checkout is
`.superpowers/sdd/2026-10-30-smplerx-camera-arm/upstream` (ignored scratch).
The reference files are `README.md`, `main/inference.py`,
`main/config/config_smpler_x_s32.py`, and `common/utils/human_models.py`.

Public S32 weights are linked by the official README:
https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_s32.pth.tar

Downloaded to `.superpowers/sdd/2026-10-30-smplerx-camera-arm/smpler_x_s32.pth.tar`.
Size: 384315279 bytes. SHA-256:
`1c78569f4e7b6e22e7166b55f322e5d5e019073b8d1a430a6926f93b4c86aa62`.
This hash records the downloaded artifact; a real model load is still pending.

SMPL-X model access requires registration and agreement to the model license:
https://smpl-x.is.tue.mpg.de/ . The user must complete this account step.
No account password is needed in the repository or chat. Download the original
body model assets, not a Blender/Unity add-on. The inspected upstream loads
neutral, male and female SMPL-X models as well as SMPL models and correspondence
files at import time. Its expected layout is:

```text
human_model_files/
  smpl/
    SMPL_NEUTRAL.pkl
    SMPL_MALE.pkl
    SMPL_FEMALE.pkl
  smplx/
    SMPLX_NEUTRAL.npz
    SMPLX_MALE.npz
    SMPLX_FEMALE.npz
    MANO_SMPLX_vertex_ids.pkl
    SMPL-X__FLAME_vertex_ids.npy
    SMPLX_to_J14.pkl
```

Keep these licensed assets outside Git. Additional detector weights and
ViTPose initialization requirements must be checked against the actual S32
loader before the smoke test. No training datasets are required for the
camera inference milestone.

Local model root: `datasets/smpl_models`. All nine required SMPL-X, SMPL and
correspondence files were located there. The original SMPL v1.1.0 names
`basicmodel_{neutral,m,f}_lbs_10_207_0_v1.1.0.pkl` are accepted at the asset
boundary. A private temporary view supplies upstream `SMPL_{NEUTRAL,MALE,FEMALE}.pkl`
names through hard links (copy fallback). User originals are never renamed.
The directory is ignored by Git.

A real neutral SMPL-X forward pass using the vendored upstream body model and
the existing main-environment torch succeeded: 10475 vertices, 55 skeletal
joints and valid derived wrist rotations. Four poses also match upstream LBS
FK: maximum position error 0 m and rotation matrix component error 1.79e-7.

## Inference boundary adaptations

`backend.py` calls upstream `SMPLer_X.get_model('test')`, preserves the
`Demoer._make_model` checkpoint-key conversion, and fails if learned parameters
are missing. It avoids `common.base`, whose imports eagerly load training
datasets. `cfg.human_model_path` and the encoder config are explicit absolute
paths; the upstream network itself is not edited.

The official detector, crop, RGB normalization and full S32 network are used.
Multiple qualifying people produce an invalid result rather than selecting a
larger person. Continuous identity verification after a loss remains pending.
Derived positions and full-pose FK rotations come from a second forward of the
same neutral SMPL-X layer. Projection follows `Model.get_coord` and then inverts
`generate_patch_image`'s affine transform to original image coordinates.

Shape is currently frozen from the first accepted sample, with the raw shape
retained in each result. Stable-window shape fitting remains unfinished.
Model logs are redirected to stderr so stdout stays a framed protocol stream.
Pipe writes use a bounded writer thread so upstream initialization cannot block
the hand capture loop.

Two runtime compatibility adaptations are explicit: import MMDetection before
custom MMPose registrations, as official inference.py does; and replace obsolete
`1 - boolean_mask` expressions in torchgeometry 0.1.2 with logically identical
`~boolean_mask` expressions through a checked in-memory shim. The installed
third-party files and upstream network source are unchanged. Identity and
90-degree rotation tests cover the shim.

Reference file SHA-256 values at upstream commit `064baef0`:

| Upstream file | SHA-256 |
| --- | --- |
| `main/SMPLer_X.py` | `ad3215d700e917a9a6a818500bca853d1bd4d647778f24e78625a9554b5ebcf8` |
| `common/utils/preprocessing.py` | `a6aedd2a8fef22f8c0e3d12ed65b0af6b4d1aff30e187659e0e3a9fae5b60ebb` |
| `common/utils/human_models.py` | `343cf180b90e65e272817057ea04d0941d9d54eb6ec27450b96088384ad0cd3e` |

The upstream installation example uses Python 3.8, torch 1.12 and mmcv 1.7.1.
The application protocol requires Python 3.10+. A separate Windows candidate
runtime is installed in `.venv/smplerx-worker` using torch 1.13.0/cu117,
torchvision 0.14.0/cu117 and the official mmcv-full 1.7.1 Windows wheel compiled
for torch 1.13/cu117. `environment.lock.yml` captures the installed versions after
CUDA, MMCV ROIAlign and real S32 image inference passed. This verifies this
Windows machine, not all platforms or long-running stability. Neither the
teleoperation nor dex environment was upgraded for these dependencies.

## Real smoke evidence

Input: PyTorch vision gallery `person1.jpg` (480 x 640 RGB), downloaded from
https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/person1.jpg .
Three sequential detections/forwards returned valid bilateral observations.
The measured times were 1469 ms (first/warmup), 156 ms and 156 ms. CUDA peak
allocated memory was 694183424 bytes (about 662 MiB); peak reserved memory was
763363328 bytes (about 728 MiB). This is a single-image smoke test, not 300-frame
latency statistics or evidence that the 10 Hz / p95 150 ms acceptance target is
met. Simulation, finger inference, live capture and RGB serialization are not
included in those frame times.

The actual length-prefixed worker round trip also passed and returned a
55 x 3 skeleton with the expected checkpoint hash. Startup plus the first
request took about 14.44 seconds. Applications must hold while the worker loads;
this time is not steady-state inference latency.

## Verified software behavior

- One RGB capture is distributed to Hands and the body client.
- Body and wrist observations carry independent sample time and source.
- Expired or cached body samples retain their hold geometry but are invalid for
  fresh control or neutral-sample accumulation.
- Duplicate worker results cannot release an active request for another frame.
- Pipe reads reassemble fragments and reject excessive packet lengths.
- Calibration permits skipped camera frames and counts each body sequence once.
- Calibration and retargeting use the same shoulder-relative wrist displacement.
- MediaPipe Hands parser and camera lifecycle tests continue to pass.

The focused verification before the compatibility shim ran 74 tests successfully.
The shim also passed its isolated runtime test. The latest full suite ran
546 tests with 10 failures, 72 errors and 3 skips.
The full suite contains unmigrated Pose fixtures and removed-mode expectations,
as well as IK and VR failures that require separate baseline comparison. These
results do not establish VR regression safety or full integration success.
Raw test logs are in the plan's ignored `.superpowers/sdd` workspace.
Failure names are preserved in `docs/smplerx-regression-status.md`. The real
smoke report is tracked in `docs/smplerx-s32-smoke.json`. Direct versus worker
maximum absolute differences were 2.98e-8 m for body joints, 7.90e-7 for rotation
matrix components and 5.96e-8 for normalized image coordinates, below the 1e-5
protocol equivalence tolerance.

## Remaining implementation and acceptance

1. Extend the verified S32/environment smoke to short-video and endurance tests.
2. Derive joints, wrist FK rotations and image projection from the real model;
   validate projected joints with a rendered overlay. Four-pose body FK is
   verified; real image projection overlay still needs inspection.
3. Complete body recording/replay and capture wiring. `hand record-world`
   currently writes Hands only; new body persistence exists separately and is
   not yet the integrated replacement for combined capture.
4. Remove remaining generic hand-fallback implementation and migrate the old
   camera tests to new body fixtures. Reconcile shared-controller and VR tests.
5. Enforce one smoothing/rate-limit owner, recovery rate bounds, session-neutral
   reset and operator identity policy through the full runtime.
6. Run full regression, deterministic replay and real GPU/video acceptance.
   The smoke measured latency and GPU memory above; jitter improvement has not
   been measured.

Historical launcher files do not establish the runtime on this machine. Use
the teleoperation environment for the application and a separately verified
interpreter for inference. The present branch is not ready for a live command
example with guaranteed successful output.
