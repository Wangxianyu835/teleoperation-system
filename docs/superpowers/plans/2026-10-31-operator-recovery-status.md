# Operator Recovery and Status Implementation Plan

> **For agentic workers:** Execute each task with failing regression tests, minimal changes, and verification before committing.

**Goal:** Use the user's requested permissive person selection and expose current holds accurately.

**Architecture:** Filter person scores at 0.3. Select the highest IoU against the previous operator, breaking ties by confidence; select highest confidence at startup. Remove ambiguity holds and identity recovery windows. Retain the two-second absence timeout and abrupt unrelated-box rejection. Keep display state separate from the bridge's last valid numerical state.

**Tech Stack:** Existing NumPy, unittest, isolated S32 worker, existing simulation HUD.

**Spec:** User request to resolve identity locking and stale status, supported by `outputs/tmp_realtime/arm_diagnose_findings.md` and recorded sample 547.

## Global Constraints

- Preserve existing camera, window, result and VR interfaces; do not change upstream S32 numerical inference.
- Choose startup candidates by confidence and later candidates by IoU then confidence, as explicitly requested by the user.
- Competing matches never trigger an ambiguity hold or identity recovery window.
- Keep the 0.5-second body expiry and two-second identity timeout.
- Create an annotated checkpoint before edits; commit completed changes without pushing.

## Tasks

1. Add regression tests for recorded boxes 545→547→549, immediate selection among competing matches, confidence tie-breaking, the 0.3 score filter, identity timeouts, and unrelated-person rejection. Verify failures, implement selection and shared filtering in `external/smplerx_worker/operator_state.py`, and wire `backend.py` and the probe to the same threshold. Preserve selection provenance in result metadata.
2. Add tests for stale reason preservation and current held trace/HUD. Verify failures, preserve current input failure reasons in `body_tracking.py`, and add a display snapshot in `arm_realtime.py` without overwriting its last valid control state. Display validity, hold, freshness and age in `realtime_teleop.py`.
3. Verify targeted tests, re-run real inference on extracted fault frames, run the full unittest suite, document policy and limits, inspect the diff, and commit. Keep recorded artifacts under ignored outputs.
