# Known Issues and Audit Findings

## README versus implementation

1. README says the repository maintains one formal chain, but the implemented realtime and offline branches differ: offline H5 requires `coordinate_frame=l21`; MediaPipe realtime never calls `align_source_hand_coordinates`; Vision Pro uses a separate side-dependent transform.
2. README says identity anomalies clear affected buffers and recover after three valid frames. That is true for `MediaPipeCameraAdapter` and `TwoHandH5Dataset`, but `HandWindowBuffer` itself does not clear on missing input; clearing is caller-specific. `VisionProAdapter` has no identity tracker and intentionally feeds only one side.
3. README says training/inference/visualization accept aligned files. The loader enforces only `coordinate_frame`, not `coordinate_alignment`; the shown visualization scripts are not part of the loader contract.
4. README describes invalid H5 frames as “hold previous.” Inference and `iter_angle_h5` do hold previous values, but `*_valid` remains false and leading invalid frames remain zeros. Realtime `TwoHandRetargeter.predict` returns `None` for a missing side; holding is implemented only by `DualTeleoperator.hand_q`, which belongs to the arm/teleop extension.
5. README's input-adapter description omits the orphan `input_adapters/npy_replay_adapter.py`, which imports the missing `input_adapters.hand_keypoints` module and is not runnable as checked out.
6. README lists output angles as “rad”; the model and limits use radian-looking values, but source coordinate units and the hardware's exact angle convention are not established in code.

## Formal, experimental, and legacy boundaries

**Hand-only formal core (currently exercised by the CLI/tests):**

- `retargeting/contracts.py`, `coordinates.py`, `tracking.py`, `data.py`, `model.py`, `inference.py`, `training.py`, `simulation.py`;
- `model/pose_transformer.py`, `model/kinematics.py`, `model/losses.py`;
- `scripts/align_h5_coordinates.py` as the required offline preprocessing command;
- `tests/test_input_adapters.py`, `test_coordinate_modes.py`, `test_twohand_h5_dataset.py`, `test_twohand_retarget.py`, `test_offline_twohand.py`, `test_simulation.py` (subject to environment imports).

**Experimental/adapter code:**

- `retargeting/mediapipe.py` is explicitly described as a low-cost tracking experiment and is not wired into `python -m retargeting`.
- `retargeting/visionpro.py` depends on an external streamer and has no identity continuity stage.
- `input_adapters/npy_replay_adapter.py` is currently broken because its imported `input_adapters.hand_keypoints` module is absent.

**Arm/TRON2 extension mixed into the package:**

- `retargeting/arm.py`, `command.py`, `dual_teleop.py`, `realtime_dual_teleop.py`;
- arm fields and `ARM_SIDES`, `ACTION_ORDER`, `build_action`, `validate_arm_input`, `validate_retarget_input(...require_arms=...)` in `contracts.py`;
- README's TRON2A section and arm H5 schema;
- `tests/test_dual_arm.py`.

These are not needed to produce hand-only angle H5, but their contracts are imported by the shared package and can obscure the hand-only boundary.

## Input-adapter responsibility overlap

`retargeting/tracking.py` owns the actual conversion (`ensure_hand25`, `mediapipe21_to_hand25`, `wrist_relative`) and buffering. `retargeting/mediapipe.py` and `visionpro.py` are source adapters that should only acquire/label points and feed that buffer. `input_adapters/npy_replay_adapter.py` attempts to import a parallel `input_adapters.hand_keypoints` API, but that module does not exist. This is both duplication in naming and a broken import boundary, not a second working pipeline.

## Behavior not established or not tested

- No test instantiates `MediaPipeCameraAdapter` with a mocked MediaPipe result, so camera timestamping, handedness parsing, side reset, and source metadata are untested.
- No test covers Vision Pro transform extraction, its left/right sign asymmetry, a wrong number of transform matrices, or its missing-side behavior.
- No test proves offline and realtime payloads are numerically equivalent for the same 21-point data.
- No test exercises `scripts/align_h5_coordinates.py` end to end, including preservation of H5 groups/attrs or rejection of already aligned input.
- No test verifies `coordinate_alignment` is required (it is not) or that source units are consistent with URDF units.
- No test verifies a real `PoseTransformer` checkpoint's output semantics, angle clamp behavior, or checkpoint compatibility; the model tests use a dummy model.
- No test verifies the complete FK-to-loss path with real L21 URDFs and finite gradients. The current environment cannot import the relevant binary packages.
- No test verifies the exact left/right mirror rule or hardware command sign/order.
- No test covers invalid/non-finite frames in realtime after a complete window, or leading invalid output behavior in the realtime service.
- No test covers `track_identity=False` behavior against swapped labels beyond the option existing.
- No test verifies the declared H5 root/group schemas against representative files with missing optional vectors.

## Environment verification

`python -m unittest discover -s tests -v` was attempted. 13 tests ran successfully, but six test modules failed during import because the installed NumPy 2.2.6 is ABI-incompatible with compiled SciPy/h5py/trimesh dependencies (`numpy.dtype size changed` / modules built for NumPy 1.x). This is an environment blocker, not a behavioral conclusion.

## Arm logic that has already leaked into hand contracts

- `contracts.py` defines arm sides, arm validation, arm-inclusive payloads, fixed 48D action ordering, and optional/required arm containers.
- `build_retarget_output` cannot construct a hand-only output because `build_action` requires both 7D arms and both 17D hands.
- `retargeting.__main__` remains hand CLI-only, but the shared contract and README expose arm payloads alongside hand payloads.
- `dual_teleop.py` consumes exported hand angles and converts dim 0 away for a 17D robot command; this is an arm/robot integration consumer, not part of hand inference.

## Minimal next three code changes for a stable hand-only V1 (proposal only)

1. **Define one canonical hand adapter boundary.** Add a small hand-only canonical entry point that owns source conversion, coordinate alignment declaration, identity handling, and the `(3,25,3)` window. Make MediaPipe, Vision Pro, and replay adapters call it; remove/fix the missing `input_adapters.hand_keypoints` import rather than maintaining two names.
2. **Make the contract hand-only and explicit.** Split hand payload validation/output from arm-inclusive `contracts.py`, require and validate both `coordinate_frame` and `coordinate_alignment` for offline inputs/checkpoints, and document/validate source units plus the left/right mirror convention once confirmed.
3. **Add end-to-end contract tests before refactoring behavior.** Use synthetic 21-point frames to assert identical offline/realtime conversion, identity jump reset/recovery, invalid-frame masks and hold policy, Vision Pro mirror rules, `(B,3,25,3)->(B,18)` model shape, and the exact 18-to-17 order. Keep arm tests separate and leave arm code untouched.

These are recommendations only; no runtime code was changed in this audit.

