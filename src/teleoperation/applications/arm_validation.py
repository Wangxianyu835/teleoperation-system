"""Offline six-arm acceptance using existing bridge, IK, motors and physics."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np

from teleoperation.data.arm_validation import load_validation_recording, save_validation_recording, validate_recording
from teleoperation.simulation.arm_diagnostics import ArmDiagnostics
from teleoperation.simulation.arm_profiles import PAPER_ROBOTS, profile_for
from teleoperation.simulation.arm_realtime import RealtimeArmBridge
from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
from teleoperation.simulation.arm_scenarios import synthetic_recording
from teleoperation.simulation.arm_validation_metrics import analyze_rows, read_trace
from teleoperation.simulation.realtime_teleop import SimulationStepClock


def _points(data, key, index, shape):
    if key not in data:
        return None
    points = np.asarray(data[key][index], dtype=float)
    return points if points.shape == shape and np.isfinite(points).all() else None


def run_case(recording, trace_path, calibration_frames=30):
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

    profiles = {name: profile_for(name) for name in PAPER_ROBOTS}
    data, seconds = recording["arrays"], recording["seconds"]
    scene = None
    diag = ArmDiagnostics(trace_path)
    logical_time = [0.]
    clock = SimulationStepClock(clock=lambda: logical_time[0])
    drift = 0.
    started = time.perf_counter()
    try:
        scene = ThreeRobotHandScene(sides=("left", "right"), gui=False)
        driver = ArmSceneDriver(scene, sides=("left", "right"), ik_iterations=60,
                                residual_threshold=1e-4, diagnostics=diag)
        bridge = RealtimeArmBridge(list(profiles.values()), calibration_frames=calibration_frames,
                                   orientation="auto", fallback_hand=False)
        for index, sample_time in enumerate(seconds):
            diag.begin_frame()
            arms = {side: _points(data, side + "_arm", index, (3, 3)) for side in ("left", "right")}
            # Stored keys retain the established dump-npz spelling.
            hands = {side: _points(data, side + "_hand", index, (21, 3)) for side in ("left", "right")}
            hands.update({"world_" + side: _points(data, "world_" + side + "_hand", index, (21, 3))
                          for side in ("left", "right")})
            with diag.stage("arm_mapping"):
                targets = bridge.update(arms_world=arms, hands=hands)
            for side in ("left", "right"):
                driver.apply_each(side, {name: targets[name][side].position for name in profiles},
                    orientation={name: targets[name][side].orientation for name in profiles})
            dt = seconds[1] - seconds[0] if index == 0 else sample_time - seconds[index - 1]
            logical_time[0] += dt
            scene.steps_per_frame = clock.next_steps()
            with diag.stage("physics"):
                driver.step()
            diag.capture(index, driver, bridge, targets, arms=arms,
                         raw_timestamp=float(data["timestamps"][index]),
                         timestamp_unit=recording["timestamp_unit"], sample_time_s=float(sample_time),
                         scenario=str(recording["scenarios"][index]))
            with diag.stage("diagnostic_overhead"):
                drift = max(drift, max(item["max_drift_rad"] for item in driver.hold_drift().values()))
        calibrated = not bridge.calibrating
    finally:
        try:
            diag.close()
        finally:
            if scene is not None:
                scene.close()
    rows = read_trace(trace_path)
    summary = analyze_rows(rows, profiles, synthetic=recording["metadata"].get("source") == "synthetic",
                           max_hold_drift_rad=drift, calibrated=calibrated)
    summary.update(source=recording["metadata"].get("source", "imported_npz"),
                   timestamp_unit=recording["timestamp_unit"], metadata=recording["metadata"],
                   frames=len(seconds), input_duration_s=float(seconds[-1] - seconds[0]),
                   simulation_timing={"physics_steps": clock.total_steps, "capped_frames": clock.capped_frames,
                                      "simulated_seconds": clock.total_steps / 240},
                   diagnostic_summary=diag.summary(), wall_processing_seconds=time.perf_counter() - started,
                   trace=str(trace_path), real_camera_verified=False, hand_inference_exercised=False)
    return summary


def recommendations(cases):
    failures = [case.get("name", "case") + ":" + item["name"] for case in cases
                for item in case["checks"] if item["status"] == "fail"]
    jumps = [entry["recovery_first_target_jump_mm"]["max"] for case in cases
             for sides in case["transitions"].values() for entry in sides.values()
             if entry["recovery_first_target_jump_mm"]["max"] is not None]
    return {"failed_checks": failures,
            "next_actions": (["Investigate failed spatial/hold checks before tuning gains or filters."] if failures else
                             ["Preserve this offline baseline and collect a real-camera diagnostic trace next."])
                + ([f"Recovery target jump peaks at {max(jumps):.3f} mm; evaluate recovery continuity before adding smoothing."] if jumps else [])
                + ["Use live capture_detection and preview timings to assess camera performance; headless processing cannot establish camera latency."]}


def build_parser():
    parser = argparse.ArgumentParser(description="Offline H1/GR1/G1 six-arm diagnostics and acceptance")
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--timestamp-unit", choices=("seconds", "milliseconds"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.output_dir or Path("outputs/arm_validation") / datetime.now(
        timezone(timedelta(hours=8))).strftime("run_%Y%m%d_%H%M%S")
    cases = []
    created = False
    try:
        # Reject invalid or unlabelled imported time before creating artifacts.
        imported = load_validation_recording(args.input_npz, args.timestamp_unit) if args.input_npz else None
        root.mkdir(parents=True, exist_ok=False)
        created = True
        for rate in ((None,) if imported is not None else (30, 13)):
            label = "imported" if rate is None else f"synthetic_{rate}hz"
            if imported is None:
                data = synthetic_recording(rate, args.seed)
                save_validation_recording(root / (label + ".npz"), data)
                recording = validate_recording(data)
            else:
                recording = imported
            case = run_case(recording, root / (label + ".csv"))
            case["name"] = label
            cases.append(case)
            (root / (label + ".json")).write_text(json.dumps(case, indent=2, allow_nan=False), encoding="utf-8")
            failed = sum(item["status"] == "fail" for item in case["checks"])
            print(f"{label}: frames={case['frames']} checks={len(case['checks'])} failures={failed}", flush=True)
        result = {"schema": "three_robot_arm_acceptance_v1", "real_camera_verified": False,
                  "cases": cases, "passed": all(case["passed"] for case in cases),
                  "recommendations": recommendations(cases)}
        (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"summary={root / 'summary.json'}; real camera not verified", flush=True)
        return 0 if result["passed"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        if created:
            (root / "failure.json").write_text(json.dumps({"error": str(error), "completed_cases": len(cases)}), encoding="utf-8")
        print(f"[FAIL] arm validation: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
