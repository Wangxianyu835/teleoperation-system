"""Streaming, explicitly diagnostic CSV; never a robot command format."""
import csv
from pathlib import Path


TRACE_FIELDS = (
    "frame_id", "scenario", "sample_time_s", "raw_timestamp", "timestamp_unit",
    "robot", "side", "input_valid", "calibration_samples", "calibration_complete",
    "valid", "held", "source", "clamped", "position_fallback", "command_accepted",
    "target_base", "target_world", "actual_world", "position_error_mm",
    "command_q", "actual_q", "operator_wrist_relative", "timings_ms",
    "ik_target_base", "ik_target_world", "ik_position_error_mm",
)


class ArmTraceWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", newline="", encoding="utf-8")
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=TRACE_FIELDS)
            self.writer.writeheader()
        except BaseException:
            self.file.close()
            raise

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()
