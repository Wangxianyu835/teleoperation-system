# Operator Recovery and Status Implementation Plan

> **For agentic workers:** Execute each task with failing regression tests, minimal changes, and verification before committing.

**Goal:** Prevent an isolated duplicate person detection from permanently stopping the current operator, and expose current holds accurately.

**Architecture:** Associate filtered detector boxes with the previously accepted box at the worker boundary. Preserve conservative holds when association is ambiguous, require three consistent recovery frames, and retain the two-second identity timeout. Keep display state separate from the bridge's last valid numerical state.

**Tech Stack:** Existing NumPy, unittest, isolated S32 worker, existing simulation HUD.

**Spec:** User request to resolve identity locking and stale status, supported by `outputs/tmp_realtime/arm_diagnose_findings.md` and recorded sample 547.

## Global Constraints

- Preserve existing camera, window, result and VR interfaces; do not change upstream S32 numerical inference.
- Do not choose an operator by box size or detector confidence alone.
- Ambiguous associations produce no usable body command; no automatic adoption of a spatially unrelated person.
- Keep the 0.5-second body expiry and two-second identity timeout.
- Create an annotated checkpoint before edits; commit completed changes without pushing.

## Tasks

1. Add regression tests for recorded boxes 545→547→549, ambiguous recovery, competing associations, identity timeouts, and unrelated-person rejection. Verify failures, implement selection/recovery in `external/smplerx_worker/operator_state.py`, and wire `backend.py` to use it. Add selection provenance to result metadata.
2. Add tests for stale reason preservation and current held trace/HUD. Verify failures, preserve current input failure reasons in `body_tracking.py`, and add a display snapshot in `arm_realtime.py` without overwriting its last valid control state. Display validity, hold, freshness and age in `realtime_teleop.py`.
3. Verify targeted tests, re-run real inference on extracted fault frames, run the full unittest suite, document policy and limits, inspect the diff, and commit. Keep recorded artifacts under ignored outputs.
