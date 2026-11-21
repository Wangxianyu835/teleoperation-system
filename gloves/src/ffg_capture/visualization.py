"""Live and offline FFG plots using a consistent five-finger layout."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib
import numpy as np

from .api import FFGSample
from .frames import CHANNEL_GROUPS, CHANNEL_NAMES, FLEXION_GROUPS, SIDE_CHANNELS
from .replay import CaptureData


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


class LivePlot:
    def __init__(
        self,
        *,
        max_points: int = 3000,
        refresh_hz: float = 15.0,
        side: str = "right",
    ) -> None:
        self.max_points = max_points
        self.refresh_hz = refresh_hz
        self.side = side
        self._times: deque[float] = deque(maxlen=max_points)
        self._history: deque[np.ndarray] = deque(maxlen=max_points)
        self._start_ns: int | None = None
        self._last_draw = 0.0
        self._status_text = ""
        self.figure = None
        self.axes = []
        self.finger_lines = []
        self.heatmap = None

    def open(self) -> None:
        plt = _pyplot()
        plt.ion()
        self.figure = plt.figure(figsize=(11, 6), layout="constrained")
        grid = self.figure.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.35)
        activity_axis = self.figure.add_subplot(grid[0, 0])
        heat_axis = self.figure.add_subplot(grid[1, 0], sharex=activity_axis)
        self.axes = [activity_axis, heat_axis]
        for finger in FLEXION_GROUPS:
            (line,) = activity_axis.plot([], [], linewidth=1.3, label=finger)
            self.finger_lines.append(line)
        activity_axis.set_ylabel("flexion mean (rad)")
        activity_axis.grid(True, alpha=0.25)
        activity_axis.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)
        heat_axis.set_ylabel("21 channels")
        heat_axis.set_xlabel("capture time (s)")
        self.heatmap = heat_axis.imshow(
            np.zeros((21, 2)),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=2.0 * np.pi,
            extent=(0.0, 0.1, -0.5, 20.5),
        )
        self.figure.colorbar(self.heatmap, ax=heat_axis, fraction=0.025, pad=0.01)
        self.figure.suptitle(f"Linker FFG live - {self.side}")
        self._status_text = activity_axis.set_title(
            "Waiting for samples...", fontsize=9
        )
        plt.show(block=False)

    @property
    def window_open(self) -> bool:
        if self.figure is None:
            return True
        plt = _pyplot()
        return bool(plt.fignum_exists(self.figure.number))

    def update(self, sample: FFGSample, *, force: bool = False) -> None:
        import time

        now = time.monotonic()
        if not force and now - self._last_draw < 1.0 / self.refresh_hz:
            return
        self._last_draw = now
        if self._start_ns is None:
            self._start_ns = sample.host_time_ns
        elapsed = (sample.host_time_ns - self._start_ns) / 1e9
        self._times.append(elapsed)
        self._history.append(sample.positions_rad.copy())
        times = np.asarray(self._times)
        history = np.asarray(self._history)
        finger_means = np.column_stack(
            [
                history[:, list(indices)].mean(axis=1)
                for indices in FLEXION_GROUPS.values()
            ]
        )
        for index, line in enumerate(self.finger_lines):
            line.set_data(times, finger_means[:, index])
        self.axes[0].set_xlim(max(0.0, times[-1] - 8.0), max(2.0, times[-1]))
        self.axes[0].set_ylim(
            float(np.min(finger_means)) - 0.05,
            float(np.max(finger_means)) + 0.05,
        )
        self.heatmap.set_data(history.T)
        self.heatmap.set_extent((times[0], max(times[-1], times[0] + 1e-3), -0.5, 20.5))
        self._status_text.set_text(
            f"seq={sample.sequence} | {elapsed:.1f}s | "
            f"{sample.status} | valid={sample.valid}"
        )
        plt = _pyplot()
        plt.pause(0.001)

    def save(self, path: str | Path, dpi: int = 140) -> Path:
        if self.figure is None:
            raise RuntimeError("Live plot has not been opened")
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(target, dpi=dpi, bbox_inches="tight")
        return target

    def close(self) -> None:
        if self.figure is None:
            return
        plt = _pyplot()
        plt.close(self.figure)
        self.figure = None


def save_capture_plot(
    data: CaptureData,
    path: str | Path,
    *,
    dpi: int = 140,
    max_points: int = 5000,
) -> Path:
    plt = _pyplot()
    count = data.sample_count
    if count == 0:
        raise ValueError("Cannot plot an empty capture")
    stride = max(1, int(np.ceil(count / max_points)))
    positions = data.positions_rad[::stride]
    times = (data.host_time_ns[::stride] - data.host_time_ns[0]) / 1e9
    figure, axes = plt.subplots(5, 1, figsize=(13, 8), sharex=True)
    for axis, (finger, indices) in zip(axes, CHANNEL_GROUPS.items()):
        for index in indices:
            axis.plot(
                times, positions[:, index], linewidth=1.1, label=CHANNEL_NAMES[index]
            )
        axis.set_ylabel(f"{finger}\nrad")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left", ncol=2, fontsize=7, frameon=False)
    axes[-1].set_xlabel("capture time (s)")
    figure.suptitle(f"Linker FFG capture - {data.stream_id}")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return target


def save_summary_plot(
    data: CaptureData,
    path: str | Path,
    *,
    dpi: int = 140,
    max_points: int = 3000,
) -> Path:
    """Create a compact two-panel quality overview."""

    plt = _pyplot()
    if data.sample_count == 0:
        raise ValueError("Cannot plot an empty capture")
    stride = max(1, int(np.ceil(data.sample_count / max_points)))
    positions = data.positions_rad[::stride]
    times = (data.host_time_ns[::stride] - data.host_time_ns[0]) / 1e9
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": (3, 2)},
    )
    heat = axes[0].imshow(
        positions.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=2.0 * np.pi,
        extent=(times[0], max(times[-1], times[0] + 1e-3), -0.5, 20.5),
    )
    axes[0].set_ylabel("21 FFG channels")
    axes[0].set_xlabel("capture time (s)")
    figure.colorbar(heat, ax=axes[0], fraction=0.025, pad=0.01)

    flexion_spans = [
        float(np.ptp(positions[:, list(indices)], axis=0).max())
        for indices in FLEXION_GROUPS.values()
    ]
    side_spans = [
        float(np.ptp(positions[:, list(indices)], axis=0).max())
        for indices in SIDE_CHANNELS.values()
    ]
    x = np.arange(len(FLEXION_GROUPS))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        flexion_spans,
        width,
        color="#2f7bbf",
        label="flexion",
    )
    axes[1].bar(
        x + width / 2,
        side_spans,
        width,
        color="#d8892b",
        label="side sway",
    )
    axes[1].set_xticks(x, list(FLEXION_GROUPS))
    axes[1].set_ylabel("max peak-to-peak (rad)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("flexion and side-sway range")
    axes[1].legend(frameon=False)
    figure.suptitle(f"Linker FFG quality overview - {data.stream_id}")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return target


def set_headless_backend() -> None:
    matplotlib.use("Agg", force=True)
