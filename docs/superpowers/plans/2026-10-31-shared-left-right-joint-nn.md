# Shared Left-Right Joint NN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `--hand-model` as the target hand selection while training one shared camera NN checkpoint per target model that serves both left and right hands.

**Architecture:** The trainer will use the canonical right-hand coordinate contract and the right-hand target specification as the shared output convention. It will combine valid left and right windows into one training stream, save checkpoints under a `shared` target directory with explicit shared metadata, and convert canonical angles to the requested native side at inference. Existing side-specific checkpoint loading remains available for compatibility, while new training and dual-side inference use the shared checkpoint path.

**Tech Stack:** Python 3.10+, PyTorch, HDF5, existing PoseTransformer, differentiable L21 FK, `unittest`.

**Spec:** `docs/superpowers/plans/2026-10-31-shared-left-right-joint-nn.md`

## Global Constraints

- Preserve the current public hand window and result interfaces.
- Keep `--hand-model` because different target hand models have different output spaces.
- Remove the training CLI's left/right target selection; training always consumes both sides and produces one shared checkpoint per target model.
- Keep camera preprocessing explicit as `camera_palm_length_window3_v1`.
- Shared output uses canonical right-hand angle semantics and converts left output at the inference boundary.
- Do not rewrite existing side-specific checkpoint metadata or claim old checkpoints are shared.
- Preserve existing unrelated working-tree changes in the pre-change checkpoint `checkpoint-20261031-2128-shared-hand-model`.

---

### Task 1: Define shared checkpoint identity and path

**Files:**
- Modify: `src/teleoperation/paths.py`
- Modify: `src/teleoperation/retargeting/hand/checkpoint.py`
- Test: `tests/test_joint_checkpoints.py`

**Interfaces:**
- `training_run_directory(..., side="shared", ...)` returns `<root>/<hand_model>/shared/nn/<profile>/<run>`.
- `shared_joint_checkpoint_payload(model, spec)` records `hand_side="shared"`, `hand_sides=["left", "right"]`, `training_robot="right"`, canonical angle semantics, and the right target spec identity.
- `load_shared_joint_checkpoint(path, spec, device)` validates and loads a shared checkpoint against the right-hand target specification.

- [ ] **Step 1: Write failing tests** for the shared path, metadata, strict rejection of a right-only checkpoint, and shared checkpoint loading.
- [ ] **Step 2: Run `python -m unittest tests.test_joint_checkpoints -v` and verify the new tests fail for missing shared support.
- [ ] **Step 3: Implement the shared path and checkpoint payload/loader without changing side-specific validation.
- [ ] **Step 4: Run the focused checkpoint tests and verify they pass.

### Task 2: Train one target model from both hand sides

**Files:**
- Modify: `src/teleoperation/learning/trainer.py`
- Modify: `src/teleoperation/applications/training.py`
- Modify: `src/teleoperation/cli/schemas.py`
- Test: `tests/test_joint_training.py`
- Test: `tests/test_hand_control_cli.py`

**Interfaces:**
- `train_joint_models(..., hand_models=[...], run_name=..., ...)` trains each target model once with both sides and no `hand_side` argument.
- `_epoch(..., sides=("left", "right"), ...)` combines valid side windows into one shuffled batch stream.
- Each run stores `target_side="shared"`, both source sides, shared checkpoint metadata, identical dataset identity, and the same best/last artifacts.

- [ ] **Step 1: Replace side-specific training tests with a failing shared-run test that supplies left and right synthetic frames, asserts one shared output directory, and checks both side counts contribute to training.
- [ ] **Step 2: Remove `--hand-side` from the training parser and run CLI tests to observe the expected failure before implementation.
- [ ] **Step 3: Implement shared side pooling, right-spec canonical target construction, shared resume/init validation, and shared output paths.
- [ ] **Step 4: Run `python -m unittest tests.test_joint_training tests.test_hand_control_cli -v` and verify the shared training flow passes.

### Task 3: Load one shared model for left and right runtime output

**Files:**
- Modify: `src/teleoperation/retargeting/hand/interface.py`
- Modify: `src/teleoperation/applications/hand_control.py`
- Test: `tests/test_normalized_retargeters.py`
- Test: `tests/test_hand_control_cli.py`

**Interfaces:**
- Add `SharedJointNNRetargeter(specs, model, device="cpu")` that evaluates one canonical model and emits side-specific `HandJointResult` values.
- `hand control --backend nn --hand both --input-profile camera_palm_length_window3_v1 --checkpoint <shared.pth>` loads one shared checkpoint.
- Keep explicit left/right checkpoint selection working for legacy or existing side-specific callers.

- [ ] **Step 1: Add failing tests proving one model object serves both sides and left MCP roll indices are sign-converted while right values are unchanged.
- [ ] **Step 2: Run the focused retargeter tests and verify failure because shared inference is not implemented.
- [ ] **Step 3: Implement the shared retargeter and new-profile checkpoint selection with strict shared metadata validation.
- [ ] **Step 4: Run focused runtime tests, including missing-hand behavior and pre-camera checkpoint validation.

### Task 4: Update user-facing training and runtime documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/hand-retargeting-unified.md`
- Modify: `docs/contracts.md`
- Modify: `docs/workflow-reference.md`

- [ ] **Step 1: Update commands to remove training `--hand-side`, retain `--hand-model`, and show one shared checkpoint path.
- [ ] **Step 2: Document canonical right-angle training, left-side conversion, shared checkpoint metadata, resume rules, and the incompatibility of old side-specific new-profile checkpoints with shared loading.
- [ ] **Step 3: Search for stale new-training commands containing `--hand-side` or side-specific new-profile output paths.

### Task 5: Full verification and commit

**Files:**
- No additional files.

- [ ] **Step 1: Run focused tests for checkpoints, training, normalized retargeting, CLI integration, and coordinate contracts.
- [ ] **Step 2: Run `python -m unittest discover -s tests`.
- [ ] **Step 3: Inspect `git diff --check`, `git status --short`, and the final diff for unrelated changes.
- [ ] **Step 4: Commit the shared model implementation with a descriptive message and report the post-change hash plus pre-change checkpoint tag.
