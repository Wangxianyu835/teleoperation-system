import time

import numpy as np

from ffg_capture.api import FFGSample
from ffg_capture.calibration import capture_pose_anchor, run_calibration
from ffg_capture.config import CalibrationConfig
from ffg_capture.device import SyntheticSource


def test_synthetic_pose_capture_returns_21_channel_anchor():
    source = SyntheticSource(
        fps=100.0,
        side="right",
        duration_s=2.0,
        still_seconds=2.0,
    )
    config = CalibrationConfig(
        hold_seconds=0.08,
        settle_seconds=0.0,
        stability_window_seconds=0.08,
        max_mean_variance=0.03,
        max_wait_seconds=1.0,
    )
    source.start()
    try:
        anchor = capture_pose_anchor(source, "open", config)
    finally:
        source.stop()
    assert anchor.positions_rad.shape == (21,)
    assert anchor.sample_count >= 5
    assert anchor.stable is True


def test_calibration_file_is_written_and_reloaded(tmp_path):
    source = SyntheticSource(
        fps=100.0,
        side="right",
        duration_s=2.0,
        still_seconds=2.0,
    )
    config = CalibrationConfig(
        output_dir=tmp_path,
        show_fist=False,
        hold_seconds=0.06,
        settle_seconds=0.0,
        stability_window_seconds=0.08,
        max_mean_variance=0.03,
        max_wait_seconds=1.0,
    )
    source.start()
    try:
        result = run_calibration(
            source,
            config,
            device_info={"model": "synthetic_ffg", "side": "right"},
            input_fn=lambda _: "",
        )
    finally:
        source.stop()
    assert result.path.is_file()
    assert set(result.anchors) == {"opose", "open"}
    assert result.stable is True


class _ThumbJitterSource:
    stream_id = "ffg_right"

    def __init__(self) -> None:
        self.sequence = 0
        self.running = False
        self.latest_sample = None
        self.rng = np.random.default_rng(3)

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def read(self, timeout_s: float = 0.2):
        time.sleep(0.003)
        positions = np.full(21, 1.0)
        positions[0] += self.rng.normal(0.0, 0.30)
        self.sequence += 1
        self.latest_sample = FFGSample(
            host_time_ns=time.monotonic_ns(),
            sequence=self.sequence,
            side="right",
            positions_rad=positions,
            valid=True,
        )
        return self.latest_sample

    def latest(self):
        return self.latest_sample

    def stop(self) -> None:
        self.running = False


def test_thumb_jitter_is_warning_not_calibration_failure():
    source = _ThumbJitterSource()
    config = CalibrationConfig(
        hold_seconds=0.08,
        settle_seconds=0.0,
        stability_window_seconds=0.04,
        max_mean_variance=0.03,
        channel_std_warning_rad=0.20,
        max_wait_seconds=1.0,
    )
    source.start()
    try:
        anchor = capture_pose_anchor(source, "open", config)
    finally:
        source.stop()
    assert anchor.stable is True
    assert "thumb_side_0" in anchor.warning_channels
