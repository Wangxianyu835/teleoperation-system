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
