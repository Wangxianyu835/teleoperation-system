# Unified SMPL shape fitting and PINK arm control implementation ledger

**Goal:** All four robot targets use one camera pose/reconstruction/scaling/PINK pipeline, with independent per-robot committed solver state.

**Architecture:** A single SMPLer-X observation supplies pose parameters. A shared body-model implementation reconstructs each robot's fitted morphology only on new observations. A fixed-step controller prepares candidates; output receipts commit solver and assembler state together.

**Tech Stack:** Python, NumPy, PyTorch, the existing vendored SMPL-X implementation, Pinocchio, PINK, quadprog, PyBullet, unittest.

**Spec:** User-approved revised plan in this chat (2026-11-12). This file records its binding constraints and execution progress.

## Baseline

- Source: `04edd4083c797a9bd759e556ae34c26e42b95be4`.
- Checkpoint: `checkpoint-20261112-unified-smpl-pink`.
- Main project: `D:/2026/code/teleoperation-system-retargeting`.
- Implementation worktree: `D:/2026/code/PyCharmMiscProject/teleoperation-smpl-pink`.
- Licensed model files and external runtimes are reused read-only from the main project.

## Binding constraints

- Fitting and reconstruction must use the same model identity and shape dimension; reject padding/truncation.
- Robot differences are model/landmark/task/control configuration, not alternative camera algorithms.
- Network-inferred human shape is provenance, distinct from robot-fitted shape.
- Targets declare position/orientation/elbow weights, damping, gain and posture regularization.
- Default unilateral/lost-side policy locks inactive arms and shared torso joints.
- Prepare never mutates committed solver or assembler state. Receipt acceptance precedes commit.
- Actual-state synchronization occurs at initialization/reset/recovery; normal integration starts at the last accepted command.
- New observations rebuild targets once. Control updates reuse unexpired targets with bounded integration steps.
- Replays reconstruct from recorded pose parameters through the same body model, without image inference.
- New recording fields are versioned; old recordings are never relabelled.
- PINK engineering implementation and numerical equivalence to published CasADi paths are distinct claims.
- No automatic fallback to old camera IK.
- Delete old algorithms only after their actual consumers migrate. Preserve VR-specific consumers.

## Tasks and verification

- [x] Body model/fitting/scaling and JSON persistence: red/green tests, actual 10/20-dimensional model loading and reconstruction, mismatch rejection, rigid segment invariants.
- [x] Robot specifications and transactional PINK: all four URDFs, known reachable FK targets, single-side locking, candidate rejection, finite limit-bounded outputs.
- [x] Command transactions: same unexpired target may advance at a new control time; stale/out-of-order sources may not; recording v1/v2 and partial output failure receipts.
- [x] Shared observation-to-command application and fit CLI: multi-target instances, new-sample reconstruction caching, timeouts, bounded dt and reset.
- [x] Worker/source/recording wiring: retain source identity; pose replay bypasses the network; camera controls use the shared producer.
- [x] Unified robot backend/output integration, migrating TRON2A and native camera routes.
- [x] Real-model and PyBullet acceptance artifacts: single/dual/multi-target runs, tracking errors, joint velocity, expiration/hold behavior and timing.
- [x] Old camera entry migration, deletion of dead camera logic, VR regression.
- [ ] Full regression, diff review, final commit and integration into the original project.

## Rulings

- Use an isolated worktree based on the latest main-project commit, not the older VR worktree. The implementation must preserve the newer model/command contracts.
- Use existing independent Python environments and PINK experiment dependencies first; do not upgrade the base environments.
- Every robot begins with shared torso joints locked. Additional torso/head tasks must be explicit and must have supported command parts before activation.
- H1-2 and TRON2A are the first end-to-end structure checks; generic configuration and four-model solver tests are required from the start.

## Completion evidence

Updated as implementation and real validation finish. No unexecuted test is counted as passed.

## Implementation evidence, 2026-11-13

- All four real URDFs use the same `PinkArmIK`; no camera-specific solver fallback.
- Official SMPL-X 0.1.28 supports actual 10/20-beta reconstruction. Native-joint positions match the official full forward function. Single/batch geometry uses the same reduction order.
- Four 20-beta morphology fits completed; RMS is 12.79 / 16.27 / 11.89 / 17.96 mm for H1 / GR1 / G1 / TRON. These unconstrained betas are extreme and post-scaled landmark directions still differ; no claim of statistically plausible human shape.
- Tabletop TRON uses declared shoulder-center anchor and base +Z up; no fabricated pelvis.
- All four genuine raw-pose CLI runs and the combined run produce 98 control frames from 30 reconstructions. Isolated/combined command differences are exactly 0 radians.
- Full-mesh physical known-target tests pass H1/GR1/G1. TRON meshes are absent: mathematical tests and explicitly simplified integration only. Full physical acceptance remains unavailable.
- Four-robot CPU median 18.78 ms, p95 27.16 ms, p99 31.79 ms. The 100 Hz wall-clock goal is not met. Overruns do not accumulate unbounded integration time.
- Prepared commands and solver candidates commit only after successful output; batches are immutable. Failed/partial writes stop output with an honest receipt.
- Source timestamps and target expiry use an explicit source/control clock conversion. Repeated valid targets continue integration; duplicate observations do not refresh expiry.
- Shared torso/head actuation remains explicitly locked. Nonempty shared-joint configuration is rejected at application construction until the output contract supports it.
- Historical camera numerics moved to `devtools/legacy_*`; active camera aliases use the common producer. VR and hands-only NPZ paths retain their original consumers and regression coverage.
- Licensed model/resource paths are read-only from the original project; numerical packages installed only under ignored `outputs/runtime`.

### Final verification before commit

- Full unittest discovery: 682 tests in 161.425 seconds, OK, 2 skipped. Log `outputs/acceptance/unified-smpl-pink/full-regression-final.log`.
- Real S32 image inference: passed after initialization warmup. Actual first/steady real frames 157..250 ms on RTX4050. No operator identity policy was weakened. This is image inference, not a live-camera or hardware acceptance claim.
- `git diff --check`: clean.
- Runtime retains the known limits above; the software path passes without claiming all performance and physical-quality targets pass.
