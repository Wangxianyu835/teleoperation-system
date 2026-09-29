"""Create an explicit L21-aligned H5 file without modifying the source file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retargeting.coordinates import (
    COORDINATE_ALIGNMENT,
    COORDINATE_FRAME,
    align_source_hand_coordinates,
)
from retargeting.tracking import ensure_hand25


def align_h5(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output H5 paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input H5 was not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        existing_frame = source.attrs.get("coordinate_frame")
        if isinstance(existing_frame, bytes):
            existing_frame = existing_frame.decode("utf-8")
        if existing_frame == COORDINATE_FRAME:
            raise ValueError(f"Input is already marked as {COORDINATE_FRAME!r}")

        for key in source.keys():
            source.copy(key, target)
        for key, value in source.attrs.items():
            target.attrs[key] = value

        if "left_hand_keypoints" in target and "right_hand_keypoints" in target:
            container = target
            left_name = "left_hand_keypoints"
            right_name = "right_hand_keypoints"
        else:
            groups = [
                key for key in target.keys()
                if isinstance(target[key], h5py.Group)
                and "l_glove_pos" in target[key]
                and "r_glove_pos" in target[key]
            ]
            if len(groups) != 1:
                raise ValueError(
                    "Expected root left/right datasets or one group containing "
                    "l_glove_pos and r_glove_pos"
                )
            container = target[groups[0]]
            left_name = "l_glove_pos"
            right_name = "r_glove_pos"

        for name in (left_name, right_name):
            raw = np.asarray(container[name][:])
            if raw.ndim != 3 or raw.shape[-1] != 3:
                raise ValueError(f"{name} must have shape (frames, joints, 3)")
            aligned = np.stack(
                [align_source_hand_coordinates(ensure_hand25(frame)) for frame in raw],
                axis=0,
            )
            del container[name]
            container.create_dataset(name, data=aligned, compression="gzip")

        target.attrs["coordinate_frame"] = COORDINATE_FRAME
        target.attrs["coordinate_alignment"] = COORDINATE_ALIGNMENT
        target.attrs["source_file"] = str(input_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a raw two-hand H5 file to the fixed L21 frame"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    align_h5(args.input, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
