from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HAND21_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
HAND25_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)
SIDE_COLORS = {
    "left": "#2563eb",
    "right": "#dc2626",
}

def _plot_hands(hands: dict, output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    plotted_points = []

    for side in HAND_SIDES:
        points = hands.get(side)
        if points is None:
            continue
        points = np.asarray(points, dtype=np.float32)
        if not _is_plottable(points):
            continue
        plotted_points.append(points)
        _plot_one_hand(ax, points, side)

    if not plotted_points:
        raise ValueError("No valid hand keypoints found for the requested frame")

    all_points = np.concatenate(plotted_points, axis=0)
    _set_equal_axes(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_one_hand(ax, points: np.ndarray, side: str) -> None:
    color = SIDE_COLORS[side]
    connections = HAND25_CONNECTIONS if points.shape[0] == 25 else HAND21_CONNECTIONS
    label = f"{side} hand ({points.shape[0]} points)"
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        s=34,
        alpha=0.9,
        label=label,
    )
    for start, end in connections:
        if start >= points.shape[0] or end >= points.shape[0]:
            continue
        segment = points[[start, end]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.65,
        )
    for index, point in enumerate(points):
        ax.text(
            point[0],
            point[1],
            point[2],
            str(index),
            fontsize=7,
            color="black",
        )


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _is_plottable(points: np.ndarray) -> bool:
    return bool(
        points.ndim == 2
        and points.shape[1] == 3
        and points.shape[0] in (21, 25)
        and np.isfinite(points).all()
        and not np.all(points == 0)
    )


from teleoperation.contracts.constants import HAND_SIDES
