"""Create an explicit L21-aligned H5 file without modifying the source file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

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
from retargeting.data import load_twohand_h5


def _validate_coordinate_metadata(source) -> None:
    values = {}
    for name in ("coordinate_frame", "coordinate_alignment"):
        value = source.attrs.get(name)
        if value is not None and not isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a scalar string")
        values[name] = value.decode("utf-8") if isinstance(value, bytes) else value
    if values["coordinate_frame"] == COORDINATE_FRAME:
        raise ValueError(f"Input is already marked as {COORDINATE_FRAME!r}")
    if values["coordinate_alignment"] is not None:
        raise ValueError(
            "Input already declares coordinate_alignment; refusing to apply "
            "alignment twice or interpret an unknown alignment"
        )


def align_h5(input_path: Path, output_path: Path) -> None:
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output H5 paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input H5 was not found: {input_path}")
    if output_path.exists() and input_path.samefile(output_path):
        raise ValueError("Input and output H5 files must be different")

    temporary_path = None
    try:
        with h5py.File(input_path, "r") as source:
            _validate_coordinate_metadata(source)
            frame_ids, _, raw_hands = load_twohand_h5(input_path, require_aligned=False)
            if frame_ids.size == 0:
                raise ValueError("Input H5 contains no frames")
            # Complete shape/vector/geometry validation before creating output.
            aligned = {
                side: np.stack([
                    align_source_hand_coordinates(ensure_hand25(frame))
                    for frame in raw
                ]) for side, raw in raw_hands.items()
            }
            if "left_hand_keypoints" in source and "right_hand_keypoints" in source:
                group_name = None
                names = ("left_hand_keypoints", "right_hand_keypoints")
            else:
                group_name = next(key for key in source if isinstance(source[key], h5py.Group)
                                  and "l_glove_pos" in source[key] and "r_glove_pos" in source[key])
                names = ("l_glove_pos", "r_glove_pos")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Close the native temporary handle before HDF5 opens it on Windows.
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent, prefix=f".{output_path.name}.",
                suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            with h5py.File(temporary_path, "w") as target:
                for key in source:
                    source.copy(key, target)
                target.attrs.update(source.attrs)
                container = target if group_name is None else target[group_name]
                for side, name in zip(("left", "right"), names):
                    attributes = dict(container[name].attrs)
                    del container[name]
                    dataset = container.create_dataset(name, data=aligned[side], compression="gzip")
                    dataset.attrs.update(attributes)
                target.attrs["coordinate_frame"] = COORDINATE_FRAME
                target.attrs["coordinate_alignment"] = COORDINATE_ALIGNMENT
                target.attrs["source_file"] = str(input_path)
                target.flush()
        # Both HDF5 handles are closed. Same-directory replacement avoids a
        # cross-volume move; any failure leaves the old destination untouched.
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


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
