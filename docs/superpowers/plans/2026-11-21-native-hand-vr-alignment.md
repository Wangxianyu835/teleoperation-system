# Native hand and Vision Pro alignment implementation plan

Goal is to align native NN/dex backends, hand outputs, palm-length input processing, target-specific training, Vision Pro and unified control with teleoperation-system-retargeting at fe04415be19fcf1d0100bffc91bbe8cf01d8605c.

The user requires the current restore/arm-1aaabb2 branch and the existing MediaPipe plus paper IK camera arm algorithm. Execution proceeds in the current session. Existing public hand windows, L21 results, camera commands and task evaluation callers continue to work.

The pre-change commit is 9b2b055ce075451fa3010ea996e35e002c219c8e. Its annotated checkpoint is checkpoint-20261121-1948-align-native-hands-vr.

## Hand processing and native models

- [x] Run imported native hand specification, normalization, NN/dex and input tests against the baseline and record the missing behavior.
- [x] Port HandSpec, HandJointResult, HandJointCommand, normalized windows, target-native model construction and command execution from the fixed source.
- [x] Add native functionality to shared modules while retaining the legacy L21 interfaces used by camera and task callers.
- [x] Verify both sides and all four hand models with the source behavior tests.

## Training and Vision Pro

- [x] Port target-specific datasets, six-term training losses, shared checkpoints and resume. Route explicit --hand-model training and --backend exports to native workflows and retain existing legacy invocations.
- [x] Port raw live/replay Vision Pro input, common NN/dex processing and explicit TransHandR compatibility.
- [x] Read the actual local TransHandR numerical source and verify axis remapping, wrist centering and three-frame scaling equivalence.
- [x] Record upstream paths, hashes and boundary adaptations in implementation documentation.

## Unified control with existing camera arms

- [x] Port robot command contracts, assembly, output receipts and command recording/replay.
- [x] Expose hand control/list-models/vr, replay hand-joints and robot control/replay alongside existing commands.
- [x] Compose MediaPipe observations with the existing RealtimeArmBridge and paper solver for camera arms. Reuse the source wrist controller for Vision Pro arms.
- [x] Verify joint commands, tracking loss, both sides, cleanup and a shared physical world.

## Verification and delivery

- [x] Run focused hand, training, VR, unified control and camera arm tests.
- [x] Run python -m unittest discover -s tests and verify legacy camera and task regressions.
- [x] Exercise CLI help, short headless command recording/replay and training/export workflows.
- [x] Review the diff, record source hashes and verification limits, and commit on restore/arm-1aaabb2.

## Execution evidence

The fixed source was ported with explicit native coordinate modules so existing L21 camera, checkpoint and task callers retain their public contracts. The user confirmed MediaPipe/Paper camera arms during implementation.

Full unittest discovery completed with 639 tests, 636 passed and 3 skipped in 139.988 seconds. Skips require local historical model weights. All 39 registered handlers imported and all 39 help parsers returned successfully. Both-side camera control, native training/export/inspection/replay, deterministic training resume, output receipts and loss handling passed behavior tests.

Same-input source comparisons produced zero absolute error for both normalized windows and all eight dex target/side outputs. Numerical statements extracted from the actual TransHandR source matched rolling-window inputs, and its actual model returned exactly identical output for identical weights.

The three-robot Vision Pro CLI smoke replayed 22 raw frames, accepted observations for all 12 arm/hand parts, recorded commands and replayed the result. Software checks used the existing source repository's unified interpreter with this target repository's source path. Live camera, online headset and desktop GUI remain field acceptance items.
