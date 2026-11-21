from __future__ import annotations

import numpy as np

from ffg_capture.api import FFGSample
from ffg_capture.config import QualityConfig
from ffg_capture.quality import analyze_capture, grade_capture
from ffg_capture.recorder import CaptureRecorder
from ffg_capture.replay import iter_samples, load_capture


def _sample(index: int, side: str = "right") -> FFGSample:
    positions = np.full(21, 0.2 + 0.01 * index, dtype=np.float64)
    positions[0] += 0.02 * np.sin(index / 3.0)
    return FFGSample(
        host_time_ns=1_000_000_000 + index * 20_000_000,
        sequence=index + 1,
        side=side,
        positions_rad=positions,
        force_response=np.zeros(5),
        valid=True,
        status="tracking",
        metadata={"device_kind": "synthetic_ffg"},
    )


def test_h5_roundtrip_preserves_ffg_values(tmp_path):
    path = tmp_path / "capture.h5"
    with CaptureRecorder(
        path,
        stream_id="ffg_right",
        device_info={"model": "synthetic", "side": "right"},
    ) as writer:
        for index in range(25):
            writer.append(_sample(index))
        writer.mark_event("segment_start", metadata={"step": "neutral"})

    data = load_capture(path)
    assert data.sample_count == 25
    assert data.positions_rad.shape == (25, 21)
    assert data.stream_id == "ffg_right"
    assert data.events[0]["label"] == "segment_start"
    assert data.events[0]["metadata"]["step"] == "neutral"
    replayed = list(iter_samples(data))
    assert len(replayed) == 25
    np.testing.assert_allclose(replayed[-1].positions_rad, _sample(24).positions_rad)

    analysis = analyze_capture(data, neutral_seconds=0.2)
    assert analysis["sample_count"] == 25
    assert analysis["channel_count"] == 21
    assert analysis["all_finite"] is True
    assert analysis["has_motion"] is True


def test_quality_grade_returns_concise_summary(tmp_path):
    path = tmp_path / "graded.h5"
    with CaptureRecorder(
        path,
        stream_id="ffg_right",
        device_info={"model": "synthetic", "side": "right"},
    ) as writer:
        for index in range(100):
            positions = np.linspace(0.1, 1.0, 21) + index * 0.002
            sample = FFGSample(
                host_time_ns=1_000_000_000 + index * 20_000_000,
                sequence=index + 1,
                side="right",
                positions_rad=positions,
                valid=True,
            )
            writer.append(sample)
    result = grade_capture(load_capture(path), QualityConfig(neutral_seconds=0.2))
    assert result["grade"] in {"PASS", "WARN"}
    assert set(result["summary"]) == {
        "rate_hz",
        "invalid_percent",
        "max_gap_ms",
        "motion_range_rad",
        "static_jitter_rad",
        "saturation_percent",
    }


def test_recorder_close_repairs_partial_tail_after_interrupt(tmp_path):
    path = tmp_path / "interrupted.h5"
    writer = CaptureRecorder(
        path,
        stream_id="ffg_right",
        device_info={"model": "synthetic", "side": "right"},
    )
    writer.append(_sample(0))
    writer.append(_sample(1))
    writer._host_time.resize((3,))
    writer._sequence.resize((3,))
    writer.close()
    data = load_capture(path)
    assert data.sample_count == 2


def test_reader_trims_existing_partial_tail(tmp_path):
    path = tmp_path / "legacy_partial.h5"
    with CaptureRecorder(
        path,
        stream_id="ffg_right",
        device_info={"model": "synthetic", "side": "right"},
    ) as writer:
        writer.append(_sample(0))
        writer.append(_sample(1))
    import h5py

    with h5py.File(path, "r+") as handle:
        group = handle["streams"]["ffg_right"]
        group["host_time_ns"].resize((3,))
        group["sequence"].resize((3,))
    data = load_capture(path)
    assert data.sample_count == 2
    assert data.session_metadata["load_trimmed_trailing_rows"] == 1
