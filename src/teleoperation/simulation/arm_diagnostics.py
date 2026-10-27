"""Read-only arm observations and bounded timing statistics."""
from contextlib import contextmanager
import json
import time

import numpy as np

from teleoperation.data.arm_trace import ArmTraceWriter
from teleoperation.data.arm_validation import sample_seconds


class ArmDiagnostics:
    def __init__(self, trace_path=None, clock=time.perf_counter):
        self.clock = clock
        self.trace = None if trace_path is None else ArmTraceWriter(trace_path)
        self.timings = {}
        self.frame_timings = {}
        self.commands = {}
        self.accepted = set()
        self.rows = self.frames = 0
        self.counters = {}
        self.closed = False

    def begin_frame(self):
        self.frame_timings = {}
        self.accepted = set()

    def record_ms(self, name, value):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Timing must be finite and nonnegative")
        item = self.timings.setdefault(name, {"count": 0, "total_ms": 0., "max_ms": 0.})
        item["count"] += 1
        item["total_ms"] += value
        item["max_ms"] = max(item["max_ms"], value)
        self.frame_timings[name] = self.frame_timings.get(name, 0.) + value

    @contextmanager
    def stage(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.record_ms(name, (self.clock() - start) * 1000)

    def command(self, robot, side, values):
        start = self.clock()
        self.commands[(robot, side)] = dict(values)
        self.accepted.add((robot, side))
        self.record_ms("diagnostic_overhead", (self.clock() - start) * 1000)

    def capture(self, frame_id, driver, bridge, targets, *, arms=None,
                raw_timestamp=None, timestamp_unit="unspecified_legacy", sample_time_s=None,
                scenario="unspecified"):
        start = self.clock()
        self.frames += 1
        world_arms = (getattr(arms, "world_arms", None) if not isinstance(arms, dict) else arms) or {}
        timings = json.dumps(self.frame_timings, allow_nan=False, sort_keys=True)
        for robot, adapter in (() if driver is None else driver.ik.items()):
            for side in driver.sides:
                target = targets.get(robot, {}).get(side)
                if target is None:
                    continue
                points = world_arms.get(side)
                points = None if points is None else np.asarray(points, dtype=float)
                input_valid = points is not None and points.shape == (3, 3) and np.isfinite(points).all()
                actual, _ = adapter.measure(side)
                target_world = None if target.position is None else adapter.to_world(target.position)
                error = (None if target_world is None else
                         float(np.linalg.norm(np.asarray(actual) - target_world) * 1000))
                # Keep requested-target fields compatible; expose the actual IK
                # target separately after slew limiting and shoulder following.
                ik_target_world = adapter.last_target_world.get(side)
                ik_target_base = (None if ik_target_world is None else
                                  adapter.to_base(ik_target_world))
                ik_error = (None if ik_target_world is None else float(np.linalg.norm(
                    np.asarray(actual) - ik_target_world) * 1000))
                neutral = bridge.neutral[side]
                accepted = (robot, side) in self.accepted
                details = adapter.last_solve.get(side, {}) if accepted else {}
                command = self.commands.get((robot, side), {})
                actual_q = adapter.joint_values(side)
                row = {
                    "frame_id": frame_id, "scenario": scenario, "sample_time_s": sample_time_s,
                    "raw_timestamp": raw_timestamp, "timestamp_unit": timestamp_unit,
                    "robot": robot, "side": side, "input_valid": bool(input_valid),
                    "calibration_samples": (neutral["samples"] if neutral["complete"]
                        else bridge.calibration_progress()[side][0]),
                    "calibration_complete": bool(neutral["complete"]),
                    "valid": target.valid, "held": target.held, "source": target.source,
                    "clamped": bool(target.info.get("clamped", False)),
                    "position_fallback": bool(details.get("position_fallback", False)),
                    "command_accepted": accepted,
                    "target_base": target.position, "target_world": target_world,
                    "actual_world": actual, "position_error_mm": error,
                    "ik_target_base": ik_target_base, "ik_target_world": ik_target_world,
                    "ik_position_error_mm": ik_error,
                    "command_q": command, "actual_q": actual_q,
                    "operator_wrist_relative": (points[2] - points[0]).tolist() if input_valid else None,
                    "timings_ms": timings,
                }
                counter = self.counters.setdefault(robot, {}).setdefault(side, {
                    "frames": 0, "valid": 0, "held": 0, "clamped": 0,
                    "position_fallback": 0, "command_accepted": 0,
                })
                counter["frames"] += 1
                for name in ("valid", "held", "clamped", "position_fallback", "command_accepted"):
                    counter[name] += int(row[name])
                if self.trace is not None:
                    for key in ("target_base", "target_world", "actual_world", "command_q",
                                "actual_q", "operator_wrist_relative", "ik_target_base", "ik_target_world"):
                        row[key] = json.dumps(row[key], allow_nan=False, sort_keys=True)
                    self.trace.write(row)
                self.rows += 1
        self.record_ms("diagnostic_overhead", (self.clock() - start) * 1000)

    def summary(self):
        return {
            "schema": "arm_diagnostics_v1", "frames": self.frames, "rows": self.rows,
            "trace": None if self.trace is None else str(self.trace.path),
            "timing_definition": "Host processing duration; not camera latency or motion response. CSV write/read overhead is separate.",
            "timings_ms": {name: {"count": item["count"],
                "mean_ms": item["total_ms"] / item["count"], "max_ms": item["max_ms"]}
                for name, item in self.timings.items()},
            "sides": self.counters,
        }

    def close(self):
        if not self.closed:
            if self.trace is not None:
                self.trace.close()
            self.closed = True
