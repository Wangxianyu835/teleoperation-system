"""Visualize hand keypoints used by the two-hand training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from teleoperation.paths import PROJECT_ROOT

from teleoperation.learning.dataset import HAND_SIDES, TwoHandH5Dataset
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.retargeting.hand.topology import ensure_hand25


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


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

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
    data = read_recording(path)
    _check_frame_index(frame_index, len(data), path)
    frame = data[frame_index]
    hands = _frame_to_hands(frame)
    if not raw:
        hands = _ensure_hands25(hands)
    return hands, [path.name, f"{'raw' if raw else 'processed'} frame={frame_index}"]


def _load_npz_frame(path: Path, frame_index: int, raw: bool) -> tuple[dict, list[str]]:
    data = read_arrays(path)
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


def _check_frame_index(frame_index: int, frame_count: int, path: Path) -> None:
    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(
            f"Frame {frame_index} is out of range for {path} "
            f"with {frame_count} frames"
        )


from teleoperation.data.npy import read_recording
from teleoperation.data.npz import read_arrays

from teleoperation.tools.hand_keypoints import _is_plottable, _plot_hands, _plot_one_hand, _set_equal_axes
