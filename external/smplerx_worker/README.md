# SMPLer-X worker

This directory contains the isolated worker boundary for the official
SMPLer-X S32 model.  The worker environment is separate from `teleoperation`
and `.venv/dex-native`.

The worker must receive one RGB frame and return one versioned result.  It must
write protocol data to stdout and diagnostic text to stderr.  Model weights
and SMPL-X body-model files stay outside Git.

The runtime is not ready until `devtools/verify_smplerx_runtime.py --check`
reports all paths as present and a real single-frame model smoke test records
the output shape, elapsed time, and peak CUDA memory.

The local Windows runtime in `.venv/smplerx-worker` has passed CUDA, MMCV
ROIAlign, S32 single-image inference and a real worker round trip on the
RTX 4050. `environment.lock.yml` captures that environment. Full camera and
endurance acceptance are still pending; see `docs/smplerx-arm.md`.

The local licensed body root is `datasets/smpl_models`. Both canonical SMPL
filenames and the original v1.1.0 download filenames are accepted. The adapter
builds a temporary normalized view without renaming the original files.

To perform a real image smoke test, run the checker from the main environment
with `--input-image PATH`, the four explicit worker/source/checkpoint/body paths,
and `--report PATH`. It launches `smoke.py` with the isolated interpreter.
An assets-only check reports `assets_present`; it never reports inference
success merely because a directory exists.
