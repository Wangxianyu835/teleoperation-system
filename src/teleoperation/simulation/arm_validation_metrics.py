"""Offline spatial acceptance and observational motion metrics from arm CSV."""
import csv
import json

import numpy as np


STEADY_SCENARIOS = {"static", "static_noise", "move_left", "move_right", "move_both",
                    "body_translation", "settle", "pose_fallback", "imported"}


def distribution(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic metric")
    return {"count": len(values), "mean": float(values.mean()), "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def read_trace(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["frame_id"] = int(row["frame_id"])
        row["sample_time_s"] = None if row["sample_time_s"] == "" else float(row["sample_time_s"])
        row["position_error_mm"] = None if row["position_error_mm"] == "" else float(row["position_error_mm"])
        for key in ("input_valid", "valid", "held", "clamped", "position_fallback", "command_accepted", "calibration_complete"):
            row[key] = row[key] == "True"
        for key in ("target_base", "target_world", "actual_world", "command_q", "actual_q", "operator_wrist_relative", "timings_ms"):
            row[key] = json.loads(row[key])
        for key in ("ik_target_base", "ik_target_world"):
            if key in row:
                row[key] = json.loads(row[key])
        if "ik_position_error_mm" in row:
            row["ik_position_error_mm"] = (None if row["ik_position_error_mm"] == ""
                                            else float(row["ik_position_error_mm"]))
    return rows


def _jitter(rows, field, scale):
    values = np.asarray([row[field] for row in rows if row[field] is not None], dtype=float)
    if not len(values):
        return None
    return float(np.sqrt(np.mean(np.sum((values - values.mean(axis=0)) ** 2, axis=1))) * scale)


def analyze_rows(rows, profiles, *, synthetic=False, max_hold_drift_rad=0., calibrated=True,
                 map_mode='chain', robot_geometry=None):
    groups, pairs, checks = {}, {}, []
    def check(name, passed, **details):
        checks.append({"name": name, "status": "observation" if passed is None else "pass" if passed else "fail", **details})

    for row in rows:
        groups.setdefault((row["robot"], row["side"], row["scenario"]), []).append(row)
        pairs.setdefault((row["robot"], row["side"]), []).append(row)
    expected = {(name, side) for name in profiles for side in ("left", "right")}
    check("six_arm_coverage", set(pairs) == expected)
    check("calibration_complete", calibrated)
    check("held_joint_drift", max_hold_drift_rad <= .02, actual_rad=max_hold_drift_rad, limit_rad=.02)
    summaries = {}
    for (name, side, scenario), entries in groups.items():
        start = entries[0]["sample_time_s"]
        steady = [row for row in entries if row["sample_time_s"] is not None and start is not None
                  and row["sample_time_s"] - start >= .75 and row["calibration_complete"]]
        eligible = [row for row in steady if row["input_valid"] and row["valid"] and not row["clamped"]]
        errors = distribution([row["position_error_mm"] for row in eligible])
        joint_steps = [max(abs(row["actual_q"][key] - before["actual_q"][key]) for key in row["actual_q"])
                       for before, row in zip(entries, entries[1:]) if row["actual_q"]]
        stats = {"frames": len(entries), "input_valid_fraction": sum(row["input_valid"] for row in entries) / len(entries),
                 "valid_fraction": sum(row["valid"] for row in entries) / len(entries),
                 "clamped_fraction": sum(row["clamped"] for row in entries) / len(entries),
                 "position_fallback_count": sum(row["position_fallback"] for row in entries),
                 "position_error_mm": errors, "all_frame_error_mm": distribution([row["position_error_mm"] for row in entries]),
                 "joint_step_rad": distribution(joint_steps), "classification": "observation"}
        if scenario in ("static", "static_noise"):
            stats.update(target_jitter_rms_mm=_jitter(steady, "target_world", 1000),
                         actual_jitter_rms_mm=_jitter(steady, "actual_world", 1000))
        stage_values = {}
        for row in entries:
            for key, value in row["timings_ms"].items():
                stage_values.setdefault(key, []).append(value)
        stats["host_processing_ms"] = {key: distribution(values) for key, values in stage_values.items()}
        summaries.setdefault(name, {}).setdefault(side, {})[scenario] = stats
        if scenario in STEADY_SCENARIOS:
            check(f"steady_error.{name}.{side}.{scenario}", errors["count"] > 0 and errors["mean"] <= profiles[name].max_residual_mm,
                  actual_mean_mm=errors["mean"], limit_mm=profiles[name].max_residual_mm,
                  definition="Valid, calibrated, unclamped frames after 0.75 s segment settling")

    transitions = {}
    for (name, side), entries in pairs.items():
        hold_deltas, recovery_jumps, motion = [], [], []
        limit = profiles[name].travel_limit(side)
        neutral = np.asarray(profiles[name].realtime_neutral(side))
        inside = True
        reachable = True
        geometry = (robot_geometry or {}).get(name, {}).get(side)
        for i, row in enumerate(entries):
            if row["target_base"] is not None:
                inside &= bool(np.all(np.abs(np.asarray(row["target_base"]) - neutral) <= limit + 1e-9))
                if map_mode == 'posture' and geometry is not None and row['valid']:
                    distance = np.linalg.norm(np.asarray(row['target_base']) - geometry['shoulder'])
                    reachable &= bool(np.isfinite(distance) and distance <= geometry['upper'] + geometry['fore'] + 1e-6)
            if row["scenario"].startswith("move_"):
                motion.append(row["actual_world"])
            if i == 0:
                continue
            previous = entries[i - 1]
            delta = float(np.linalg.norm(np.asarray(row["target_base"]) - previous["target_base"]) * 1000)
            if row["held"] and previous["calibration_complete"]:
                hold_deltas.append(delta)
            if previous["held"] and previous["calibration_complete"] and row["valid"]:
                recovery_jumps.append(delta)
        span = 0. if not motion else float(np.ptp(motion, axis=0).max() * 1000)
        transitions.setdefault(name, {})[side] = {"held_target_delta_mm": distribution(hold_deltas),
            "recovery_first_target_jump_mm": distribution(recovery_jumps), "actual_motion_span_mm": span,
            "step_response_90_s": None, "classification": "observation"}
        if map_mode == 'posture':
            check(f'posture_reach.{name}.{side}', reachable if geometry is not None else None,
                  definition='Valid posture targets within the measured robot arm reach; chain travel box does not apply')
        else:
            check(f"travel_clamp.{name}.{side}", inside, limit_m=limit)
        check(f"dropout_hold.{name}.{side}", all(value <= 1e-6 for value in hold_deltas) if hold_deltas else None,
              samples=len(hold_deltas))
        if synthetic:
            check(f"motion_travel.{name}.{side}", span >= profiles[name].min_travel_mm,
                  actual_mm=span, minimum_mm=profiles[name].min_travel_mm)
            check(f"dropout_exercised.{name}.{side}", bool(hold_deltas))
            clamp_rows = [row for row in entries if row["scenario"] == "clamp"]
            check(f"clamp_exercised.{name}.{side}", any(row["clamped"] for row in clamp_rows))
            body = [row["target_base"] for row in entries if row["scenario"] == "body_translation"]
            check(f"body_translation_invariance.{name}.{side}", bool(body) and np.ptp(body, axis=0).max() <= 1e-9)
            if name == "h1_2":
                check(f"pose_fallback_exercised.{side}", any(row["position_fallback"] for row in entries if row["scenario"] == "pose_fallback"))
        step = next((i for i, row in enumerate(entries) if row["scenario"] == "step_both"), None)
        if step is not None and step > 0:
            previous, first = entries[step - 1], entries[step]
            direction = np.asarray(first["target_world"]) - previous["target_world"]
            squared = float(direction @ direction)
            if squared > 1e-12:
                for row in entries[step:]:
                    if row["scenario"] != "step_both":
                        break
                    progress = float((np.asarray(row["actual_world"]) - previous["actual_world"]) @ direction / squared)
                    if progress >= .9:
                        transitions[name][side]["step_response_90_s"] = row["sample_time_s"] - first["sample_time_s"]
                        break
    unique = {row["frame_id"]: row for row in rows}
    timings = {}
    for row in unique.values():
        for key, value in row["timings_ms"].items():
            timings.setdefault(key, []).append(value)
    overview = {}
    for (name, side), entries in pairs.items():
        sides = summaries[name][side]
        measured = [(scenario, stats["position_error_mm"]["mean"]) for scenario, stats in sides.items()
                    if scenario in STEADY_SCENARIOS and stats["position_error_mm"]["mean"] is not None]
        worst = max(measured, key=lambda item: item[1]) if measured else (None, None)
        overview.setdefault(name, {})[side] = {
            "worst_steady_scenario": worst[0], "worst_steady_mean_error_mm": worst[1],
            "error_limit_mm": profiles[name].max_residual_mm,
            "recovery_jump_max_mm": transitions[name][side]["recovery_first_target_jump_mm"]["max"],
            "static_noise_actual_jitter_mm": sides.get("static_noise", {}).get("actual_jitter_rms_mm"),
            "step_response_90_s": transitions[name][side]["step_response_90_s"],
            "spatial_checks_passed": all(item["status"] != "fail" for item in checks
                if f".{name}.{side}" in item["name"]),
        }
    return {"checks": checks, "scenarios": summaries, "transitions": transitions,
            "arm_overview": overview,
            "host_processing_ms": {key: distribution(values) for key, values in timings.items()},
            "passed": all(item["status"] != "fail" for item in checks),
            "observations": ["Jitter, motion response and recovery jumps have no validated pass threshold.",
                "Response uses sample time and projected displacement; null means 90% was not reached or the scenario is absent.",
                "Host timings are not camera end-to-end latency. No real-camera verification was performed."]}
