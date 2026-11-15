from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .plotting import set_equal_axes
from teleoperation.retargeting.hand.config import JOINT_EDGES, JOINT_NAMES
TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
SIDE_COLORS = {
    "left": "#2563eb",
    "right": "#dc2626",
}

def _plot_side(positions, side: str, pose: str, label_names: bool, output_path: Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_hand(ax, positions, side=side, label_names=label_names)
    set_equal_axes(ax, positions, padding=0.6, zero_radius=0.1)
    ax.set_title(f"L21 {side} FK - {pose} pose")
    _finish_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_both(hands, pose: str, label_names: bool, output_path: Path) -> None:
    all_points = np.concatenate(tuple(hands.values()), axis=0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for side, positions in hands.items():
        _draw_hand(ax, positions, side=side, label_names=label_names)
    set_equal_axes(ax, all_points, padding=0.6, zero_radius=0.1)
    ax.set_title(f"L21 left/right FK - {pose} pose")
    _finish_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_hand(ax, positions: np.ndarray, side: str, label_names: bool) -> None:
    color = SIDE_COLORS[side]
    name_to_index = {name: index for index, name in enumerate(JOINT_NAMES)}
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=color,
        s=28,
        alpha=0.9,
        label=f"{side} FK nodes",
    )
    tip_indices = [name_to_index[name] for name in TIP_NAMES]
    ax.scatter(
        positions[tip_indices, 0],
        positions[tip_indices, 1],
        positions[tip_indices, 2],
        c="#f59e0b",
        s=70,
        alpha=0.95,
        label=f"{side} tips",
    )
    for parent, child in JOINT_EDGES:
        parent_index = name_to_index[parent]
        child_index = name_to_index[child]
        segment = positions[[parent_index, child_index]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.8,
            alpha=0.7,
        )
    for index, point in enumerate(positions):
        label = JOINT_NAMES[index] if label_names else str(index)
        ax.text(point[0], point[1], point[2], label, fontsize=7, color="black")


def _finish_axes(ax) -> None:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
