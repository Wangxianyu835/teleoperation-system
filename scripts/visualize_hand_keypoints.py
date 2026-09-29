"""Visualize hand keypoints used by the two-hand training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retargeting.data import HAND_SIDES, TwoHandH5Dataset, load_twohand_h5
from retargeting.tracking import ensure_hand25


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save 3D hand-keypoint marker images into picture/."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--frame-start",
        type=int,
        help="First frame in a continuous inclusive range.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Last frame in a continuous inclusive range.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("picture"))
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot stored keypoints directly instead of training-processed 25-point keypoints.",
    )
    parser.add_argument(
        "--receptive-field",
        type=int,
        default=1,
        help="History window for processed H5 visualization; use 3 to match training windows.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip frames with no valid hand keypoints instead of stopping.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_sides = HAND_SIDES if args.side == "both" else (args.side,)
    output_paths = []
    for frame_index in _resolve_frames(args):
        try:
            hands, title_bits = _load_frame(args, frame_index)
            hands = {side: hands.get(side) for side in selected_sides}
            output_path = args.output_dir / (
                f"{args.input.stem}_frame_{frame_index:06d}_{args.side}.png"
            )
            _plot_hands(hands, output_path, " | ".join(title_bits))
            output_paths.append(output_path)
        except ValueError as exc:
            if not args.skip_invalid:
                raise
            print(f"skipped frame {frame_index}: {exc}")

    for path in output_paths:
        print(path)
    return 0


def _resolve_frames(args: argparse.Namespace) -> list[int]:
    if args.frame_step < 1:
        raise ValueError("--frame-step must be positive")
    if args.frame_start is None and args.frame_end is None:
        return list(args.frames)
    if args.frame_start is None or args.frame_end is None:
        raise ValueError("--frame-start and --frame-end must be used together")
    if args.frame_end < args.frame_start:
        raise ValueError("--frame-end must be greater than or equal to --frame-start")
    return list(range(args.frame_start, args.frame_end + 1, args.frame_step))


def _load_frame(args: argparse.Namespace, frame_index: int) -> tuple[dict, list[str]]:
    suffix = args.input.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        if args.raw:
            return _load_raw_h5_frame(args.input, frame_index)
        return _load_processed_h5_frame(args, frame_index)
    if suffix == ".npy":
        return _load_npy_frame(args.input, frame_index, raw=args.raw)
    if suffix == ".npz":
        return _load_npz_frame(args.input, frame_index, raw=args.raw)
    raise ValueError(f"Unsupported input type: {args.input.suffix}")


def _load_raw_h5_frame(path: Path, frame_index: int) -> tuple[dict, list[str]]:
    frame_ids, _, raw_hands = load_twohand_h5(path, require_aligned=False)
    _check_frame_index(frame_index, len(frame_ids), path)
    hands = {
        side: np.asarray(raw_hands[side][frame_index], dtype=np.float32)
        for side in HAND_SIDES
    }
    return hands, [path.name, f"raw frame={frame_index}"]


def _load_processed_h5_frame(
    args: argparse.Namespace,
    frame_index: int,
) -> tuple[dict, list[str]]:
    dataset = TwoHandH5Dataset(
        args.input,
        receptive_field=args.receptive_field,
        scale_factor=args.scale_factor,
    )
    by_frame = {sample["frame_index"]: sample for sample in dataset.samples}
    if frame_index not in by_frame:
        raise ValueError(
            f"Frame {frame_index} is not available as a processed sample. "
            f"Try a later frame or use --raw. "
            f"First processed frame is usually receptive_field - 1."
        )
    sample = by_frame[frame_index]
    hands = {}
    for side in HAND_SIDES:
        if sample[f"{side}_valid"]:
            hands[side] = sample[f"{side}_target"][0]
        else:
            hands[side] = None
    return hands, [
        args.input.name,
        f"processed frame={frame_index}",
        f"rf={args.receptive_field}",
    ]


def _load_npy_frame(path: Path, frame_index: int, raw: bool) -> tuple[dict, list[str]]:
    data = np.load(path, allow_pickle=True)
    _check_frame_index(frame_index, len(data), path)
    frame = data[frame_index]
    hands = _frame_to_hands(frame)
    if not raw:
        hands = _ensure_hands25(hands)
    return hands, [path.name, f"{'raw' if raw else 'processed'} frame={frame_index}"]


def _load_npz_frame(path: Path, frame_index: int, raw: bool) -> tuple[dict, list[str]]:
    with np.load(path, allow_pickle=True) as data:
        hands = {}
        for side in HAND_SIDES:
            for key in (f"{side}_hand_keypoints", f"{side}_hand", side):
                if key in data:
                    values = np.asarray(data[key])
                    _check_frame_index(frame_index, values.shape[0], path)
                    hands[side] = values[frame_index]
                    break
    if not hands:
        raise ValueError(
            "NPZ input must contain left/right hand arrays, for example "
            "'left_hand_keypoints' and 'right_hand_keypoints'."
        )
    if not raw:
        hands = _ensure_hands25(hands)
    return hands, [path.name, f"{'raw' if raw else 'processed'} frame={frame_index}"]


def _frame_to_hands(frame: object) -> dict:
    if isinstance(frame, np.ndarray) and frame.shape in ((21, 3), (25, 3)):
        return {"right": frame}
    if hasattr(frame, "item"):
        try:
            frame = frame.item()
        except ValueError:
            pass
    if not isinstance(frame, dict):
        raise TypeError(f"Expected a frame dict or hand array, got {type(frame)!r}")
    return {
        "left": _first_present(frame, ("left_hand", "left_hand_keypoints")),
        "right": _first_present(frame, ("right_hand", "right_hand_keypoints")),
    }


def _first_present(frame: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in frame:
            return frame[key]
    return None


def _ensure_hands25(hands: dict) -> dict:
    converted = {}
    for side, points in hands.items():
        if points is None:
            converted[side] = None
            continue
        converted[side] = ensure_hand25(points)
    return converted


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


def _check_frame_index(frame_index: int, frame_count: int, path: Path) -> None:
    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(
            f"Frame {frame_index} is out of range for {path} "
            f"with {frame_count} frames"
        )


if __name__ == "__main__":
    raise SystemExit(main())
