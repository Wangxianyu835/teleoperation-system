"""Visualize LinkerHand L21 FK nodes from the configured URDF assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from retargeting.config import JOINT_EDGES, JOINT_NAMES, L21


TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
SIDE_COLORS = {
    "left": "#2563eb",
    "right": "#dc2626",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save 3D FK visualizations for the L21 left/right URDFs."
    )
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("picture"))
    parser.add_argument(
        "--pose",
        choices=("zero", "curl"),
        default="zero",
        help="Use zero angles or a small curled pose.",
    )
    parser.add_argument(
        "--label-names",
        action="store_true",
        help="Label each point with its joint name instead of its index.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sides = ("left", "right") if args.side == "both" else (args.side,)
    outputs = []
    for side in sides:
        path = args.output_dir / f"l21_{side}_fk_{args.pose}.png"
        _plot_side(side, args.pose, args.label_names, path)
        outputs.append(path)

    if len(sides) == 2:
        combined = args.output_dir / f"l21_both_fk_{args.pose}.png"
        _plot_both(args.pose, args.label_names, combined)
        outputs.append(combined)

    for path in outputs:
        print(path)
    return 0


def _plot_side(side: str, pose: str, label_names: bool, output_path: Path) -> None:
    positions = _fk_positions(side, pose)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_hand(ax, positions, side=side, label_names=label_names)
    _set_equal_axes(ax, positions)
    ax.set_title(f"L21 {side} FK - {pose} pose")
    _finish_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_both(pose: str, label_names: bool, output_path: Path) -> None:
    hands = {side: _fk_positions(side, pose) for side in ("left", "right")}
    all_points = np.concatenate(tuple(hands.values()), axis=0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for side, positions in hands.items():
        _draw_hand(ax, positions, side=side, label_names=label_names)
    _set_equal_axes(ax, all_points)
    ax.set_title(f"L21 left/right FK - {pose} pose")
    _finish_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _fk_positions(side: str, pose: str) -> np.ndarray:
    urdf = L21.left_urdf if side == "left" else L21.right_urdf
    fk = create_hand_kinematics(
        urdf,
        L21.hand_kinematics_config(),
        device="cpu",
        scale_factor=L21.training.robot_scale,
    )
    angles = _angles_for_pose(pose)
    _, _, positions = fk.forward(angles)
    return positions[0].detach().cpu().numpy()


def _angles_for_pose(pose: str) -> torch.Tensor:
    angles = torch.zeros((1, len(JOINT_NAMES)), dtype=torch.float32)
    if pose == "curl":
        for name in (
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
            "index_pip",
            "middle_pip",
            "ring_pip",
            "pinky_pip",
            "thumb_cmc_pitch",
            "thumb_mcp",
            "thumb_ip",
        ):
            angles[0, JOINT_NAMES.index(name)] = 0.55
    return angles


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
    ax.view_init(elev=0, azim=0)
    ax.set_proj_type("ortho")


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6)
    if radius <= 0:
        radius = 0.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    raise SystemExit(main())
