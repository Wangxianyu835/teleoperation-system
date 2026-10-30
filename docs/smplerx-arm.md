# SMPLer-X camera arm integration: implementation status

Status: incomplete. The camera commands have been migrated in part, but they
cannot yet run real inference. `external/smplerx_worker/adapter.py::load` still
raises an explicit backend-not-installed error. Do not use the current branch
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

The upstream installation example uses Python 3.8, torch 1.12 and mmcv 1.7.1.
The application protocol currently requires Python 3.10+. This is an unresolved
compatibility constraint; the worker environment file is a template, not a
verified lock. Neither the teleoperation nor dex environment has been upgraded
to accommodate these dependencies.

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

The focused verification on this follow-up ran 67 tests successfully. The
latest full suite ran 538 tests with 10 failures, 72 errors and 3 skips.
The full suite contains unmigrated Pose fixtures and removed-mode expectations,
as well as IK and VR failures that require separate baseline comparison. These
results do not establish VR regression safety or full integration success.
Raw test logs are in the plan's ignored `.superpowers/sdd` workspace.

## Remaining implementation and acceptance

1. Implement and verify the real pinned S32 backend and its environment.
2. Derive joints, wrist FK rotations and image projection from the real model;
   validate them against upstream and a rendered overlay. The geometry adapter's
   current synthetic tests do not establish upstream numerical equivalence.
3. Complete body recording/replay and capture wiring. `hand record-world`
   currently writes Hands only; new body persistence exists separately and is
   not yet the integrated replacement for combined capture.
4. Remove remaining generic hand-fallback implementation and migrate the old
   camera tests to new body fixtures. Reconcile shared-controller and VR tests.
5. Enforce one smoothing/rate-limit owner, recovery rate bounds, session-neutral
   reset and operator identity policy through the full runtime.
6. Run full regression, deterministic replay and real GPU/video acceptance.
   No inference latency, GPU memory or jitter improvement has been measured.

Historical launcher files do not establish the runtime on this machine. Use
the teleoperation environment for the application and a separately verified
interpreter for inference. The present branch is not ready for a live command
example with guaranteed successful output.
