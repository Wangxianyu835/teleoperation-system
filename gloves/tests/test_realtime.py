import numpy as np

from ffg_capture.api import FFGSample
from ffg_capture.realtime import RealtimeStats


def _sample(sequence: int, time_ns: int, *, valid: bool = True) -> FFGSample:
    return FFGSample(
        host_time_ns=time_ns,
        sequence=sequence,
        side="right",
        positions_rad=np.zeros(21),
        valid=valid,
    )


def test_realtime_stats_reports_rate_gaps_and_invalid_frames():
    stats = RealtimeStats(max_gap_threshold_ms=50.0)
    stats.update(_sample(1, 1_000_000_000))
    stats.update(_sample(2, 1_020_000_000, valid=False))
    stats.update(_sample(3, 1_140_000_000))
    snapshot = stats.snapshot()
    assert snapshot["frames"] == 3
    assert snapshot["invalid_frames"] == 1
    assert snapshot["max_gap_ms"] == 120.0
    assert snapshot["long_gap_count"] == 1
    assert abs(float(snapshot["rate_hz"]) - 2 / 0.14) < 1e-12
