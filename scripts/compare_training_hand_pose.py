"""Compare before/after checkpoint FK outputs against hand keypoint targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from retargeting.config import (
    JOINT_EDGES,
    JOINT_NAMES,
    L21,
)
from retargeting.data import HAND_SIDES, TwoHandH5Dataset
from retargeting.model import build_hand_model, load_hand_checkpoint


HAND25_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)
TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
SIDE_COLORS = {"left": "#2563eb", "right": "#dc2626"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a side-by-side before/after training comparison image."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--before-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--after-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("picture/training_compare.png"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dataset = TwoHandH5Dataset(
        args.input,
        receptive_field=L21.model.receptive_field,
        scale_factor=L21.training.source_scale,
    )
    sample = _sample_for_frame(dataset, args.frame)

    fks = _create_fks(device)
    checkpoints = (
        ("Before", args.before_checkpoint),
        ("After", args.after_checkpoint),
    )
    predictions = {
        title: _predict_checkpoint(path, sample, fks, device)
        for title, path in checkpoints
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_comparison(sample, predictions, args.frame, args.output)
    print(args.output)
    return 0


def _sample_for_frame(dataset: TwoHandH5Dataset, frame: int) -> dict:
    by_frame = {sample["frame_index"]: sample for sample in dataset.samples}
    if frame not in by_frame:
        raise ValueError(
            f"Frame {frame} is not available as a training sample. "
            f"Try frame >= {L21.model.receptive_field - 1} or a valid hand frame."
        )
    sample = by_frame[frame]
    if not any(sample[f"{side}_valid"] for side in HAND_SIDES):
        raise ValueError(f"Frame {frame} has no valid hand target")
    return sample


def _create_fks(device: torch.device) -> dict:
    cfg = L21.hand_kinematics_config()
    return {
        "left": create_hand_kinematics(
            L21.left_urdf,
            cfg,
            device=device,
            scale_factor=L21.training.robot_scale,
        ),
        "right": create_hand_kinematics(
            L21.right_urdf,
            cfg,
            device=device,
            scale_factor=L21.training.robot_scale,
        ),
    }


def _predict_checkpoint(
    checkpoint: Path,
    sample: dict,
    fks: dict,
    device: torch.device,
) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = build_hand_model(L21.model_kwargs()).to(device)
    load_hand_checkpoint(model, str(checkpoint), device=device)
    model.eval()

    result = {}
    with torch.no_grad():
        for side in HAND_SIDES:
            if not sample[f"{side}_valid"]:
                result[side] = None
                continue
            hand_input = torch.from_numpy(sample[f"{side}_input"][None].astype(np.float32)).to(device)
            angles18 = model(hand_input)
            angles23 = F.pad(angles18, (0, 5, 0, 0), "constant", 0)
            _, _, robot = fks[side].forward(angles23)
            result[side] = robot[0].detach().cpu().numpy()
    return result


def _plot_comparison(sample: dict, predictions: dict, frame: int, output: Path) -> None:
    fig = plt.figure(figsize=(14, 10))
    axes = [
        [fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d") for col in range(2)]
        for row in range(2)
    ]
    titles = tuple(predictions.keys())

    all_points = []
    for side in HAND_SIDES:
        if sample[f"{side}_valid"]:
            all_points.append(sample[f"{side}_target"][0])
        for title in titles:
            points = predictions[title][side]
            if points is not None:
                all_points.append(points)
    limits = _equal_limits(np.concatenate(all_points, axis=0))

    for row, side in enumerate(HAND_SIDES):
        target = sample[f"{side}_target"][0] if sample[f"{side}_valid"] else None
        for col, title in enumerate(titles):
            ax = axes[row][col]
            robot = predictions[title][side]
            if target is not None:
                _draw_target(ax, target)
            if robot is not None:
                _draw_robot(ax, robot, side)
            ax.set_title(f"{title} training - {side} hand")
            _apply_limits(ax, limits)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Frame {frame}: target keypoints vs model FK output", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _draw_target(ax, points: np.ndarray) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="#6b7280",
        s=22,
        alpha=0.55,
        label="target hand keypoints",
    )
    for first, second in HAND25_CONNECTIONS:
        segment = points[[first, second]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color="#9ca3af",
            linewidth=1.2,
            alpha=0.45,
        )


def _draw_robot(ax, points: np.ndarray, side: str) -> None:
    color = SIDE_COLORS[side]
    name_to_index = {name: index for index, name in enumerate(JOINT_NAMES)}
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        s=28,
        alpha=0.9,
        label="model FK output",
    )
    tip_indices = [name_to_index[name] for name in TIP_NAMES]
    ax.scatter(
        points[tip_indices, 0],
        points[tip_indices, 1],
        points[tip_indices, 2],
        c="#f59e0b",
        s=62,
        alpha=0.95,
        label="model tips",
    )
    for parent, child in JOINT_EDGES:
        segment = points[[name_to_index[parent], name_to_index[child]]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=1.7,
            alpha=0.72,
        )


def _equal_limits(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.58)
    if radius <= 0:
        radius = 0.1
    return center, radius


def _apply_limits(ax, limits: tuple[np.ndarray, float]) -> None:
    center, radius = limits
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
