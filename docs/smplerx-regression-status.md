# SMPLer-X migration regression status

Date: 2026-10-31. Software migration and real-camera acceptance are separate.

## Final full regression

Interpreter: `D:/Anaconda/envs/teleoperation/python.exe`.

Command: `python -m unittest discover -s tests`.

Result: **560 tests, 0 failures, 0 errors, 3 skips**, 120.699 seconds.
The previous incomplete 546-test result is superseded by this verified run.
No skip was added to hide a migration failure. Optional dex checks use their separate environment.

## Independent review and fixes

One fresh reviewer found three Important issues, no Critical issues and no deferred minors:

1. Pending asynchronous polls cleared wrist neutral samples. A 5 Hz body / 30 Hz
   camera test now proves that ten distinct samples align the wrist.
2. A returning pose could recover in one frame after loss. Recovery now requires
   three distinct consistent samples; invalid/stale results break that window.
3. Live and replay used different control intervals. Both adapters now expose
   adjacent capture-time differences, tested with irregular timestamps.

The author's real public-command checks found two additional issues and fixed them:

- A body recording without Hands failed metric-hand preprocessing. Missing hands
  now reset the finger window while the body continues; metric fallback stays forbidden.
- Posture mapping was checked against a chain displacement box. Offline acceptance
  now checks measured robot reach in posture mode and retains the original travel
  box for displacement modes. A deliberately unreachable target is rejected.

Each issue was reproduced by a failing test before its fix. The complete suite
above passed after the final changes. VR numerical implementation was unchanged;
its source/interface and shared-controller tests are included in the full run.

## Real model evidence

- Five-minute S32 single-model run: 1,604 new static-image results; 5.34 Hz;
  p50 187 ms, p95 234 ms; allocated GPU memory growth 0.
  See `smplerx-s32-endurance.json`. **The 10 Hz / p95 150 ms target was not met.**
- Real subprocess -> tracker -> combined H5 -> reader: 40 samples, body values
  preserved exactly, RGB and raw model packets retained.
- Formal three-robot headless replay: 40 frames, no arm execution errors,
  both wrist alignments reached tracking.
- Corresponding offline diagnostic: 15 checks passed. Projection was inspected.
- These runs use a repeated static image. They do not verify live hands, occlusion,
  motion accuracy or the combined camera/hand/robot endurance load.

The licensed assets and model runtime remain local and ignored by Git.
Actual camera validation remains outstanding; the software tests do not replace it.

