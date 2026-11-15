# Historical camera diagnostics

Production camera arm control uses `applications/arm_entry.py` (compatibility
arguments only), `applications/unified_camera_control.py`, `ArmPipeline`, and one
`PinkArmIK` implementation for all four robot models. `dual camera` and the old
`simulation.realtime_teleop` arm command delegate to that path. Both require an
explicit fitted profile directory. Derived-point NPZ files cannot drive arms.
The realtime command retains a hands-only live/H5/NPZ loop without arm mapping,
arm IK, arm bridge, or arm driver construction.

The following historical source snapshots are intentionally outside the runtime
package. They exist for numerical A/B diagnosis and regression coverage, not as
alternative production commands:

| Diagnostic source | Explicit consumers |
| --- | --- |
| `legacy_camera_baseline.py` | `tests/test_arm_following.py`, `tests/test_arm_diagnostics.py` |
| `legacy_tron_camera_baseline.py` | `tests/test_mediapipe_tron2a.py` historical orchestration and calibration tests |
| `legacy_dual_baseline.py` | Historical TRON numerical tests and the above diagnostic snapshot |
| `legacy_dual_export_baseline.py` | Historical command export fixture comparison in `tests/test_mediapipe_tron2a.py` |

No production module imports these snapshots. Production `dual.py` retains only
the shared native L21 hand factory. `dual realtime` and `dual export` reject the
old adapter/derived-point contract and explain the replacement `robot control`
command. Existing `ArmCalibration` data contracts and calibration capture remain
because calibration validation and recorded VR/hand mounting data still use them.
VR arm paths for the supported native robots remain in their existing controller.

New behavior is covered independently by `tests/test_migrated_camera_entries.py`,
`tests/test_unified_pink_ik.py`, `tests/test_arm_pipeline.py`, and root session
transaction tests. Historical fixture success does not establish correctness of
the new numerical algorithm; the acceptance tool `verify_unified_pink.py` records
its real-model mathematical and physical checks separately.
