"""Create an explicit L21-aligned H5 file without modifying the source file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from retargeting.coordinates import (
    COORDINATE_FRAME,
    PALM_LOCAL_METADATA,
    align_palm_local_coordinates,
    build_l21_reference_basis,
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
    _validate_source_landmark_space(source)


def _validate_source_landmark_space(container) -> None:
    value = container.attrs.get("source_landmark_space")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is not None and (not isinstance(value, str) or value != "mediapipe_normalized"):
        raise ValueError(
            "source_landmark_space must be 'mediapipe_normalized' or undeclared; "
            f"got {value!r}"
        )


def align_h5(input_path: Path, output_path: Path) -> dict:
    """Write palm-local H5 atomically and return frame/reference diagnostics."""
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
            if "left_hand_keypoints" in source and "right_hand_keypoints" in source:
                group_name = None
                names = ("left_hand_keypoints", "right_hand_keypoints")
            else:
                group_name = next(key for key in source if isinstance(source[key], h5py.Group)
                                  and "l_glove_pos" in source[key] and "r_glove_pos" in source[key])
                names = ("l_glove_pos", "r_glove_pos")
                _validate_source_landmark_space(source[group_name])

            # Fatal robot-reference validation happens before creating output.
            # Exactly one zero-pose FK per side for the entire recording.
            references = {side: build_l21_reference_basis(side) for side in ("left", "right")}
            aligned, statistics = {}, {}
            for side, raw in raw_hands.items():
                result = np.zeros((len(raw), 25, 3), dtype=np.float32)
                missing, degenerate = 0, 0
                for index, frame in enumerate(raw):
                    # ensure_hand25 already performs wrist-relative conversion.
                    with np.errstate(invalid="ignore", over="ignore"):
                        points25 = ensure_hand25(frame)
                    result[index] = align_palm_local_coordinates(points25, references[side])
                    if not np.any(frame):
                        missing += 1
                    elif not np.any(result[index]):
                        degenerate += 1
                aligned[side] = result
                basis = references[side]
                statistics[side] = {
                    "nonzero_frames": int(np.any(result != 0, axis=(1, 2)).sum()),
                    "zero_source_frames": missing,
                    "degenerate_frames": degenerate,
                    "nonfinite_values": int((~np.isfinite(result)).sum()),
                    "robot_basis": basis.tolist(),
                    "determinant": float(np.linalg.det(basis.astype(np.float64))),
                    "orthogonality_error": float(np.max(np.abs(basis.astype(np.float64).T @ basis - np.eye(3)))),
                }

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
                target.attrs.update(PALM_LOCAL_METADATA)
                if "source_file" not in target.attrs:
                    target.attrs["source_file"] = str(input_path)
                target.flush()
        # Both HDF5 handles are closed. Same-directory replacement avoids a
        # cross-volume move; any failure leaves the old destination untouched.
        os.replace(temporary_path, output_path)
        return {
            "input": str(input_path), "output": str(output_path),
            "frames": int(frame_ids.size), "sides": statistics,
            "coordinate_metadata": dict(PALM_LOCAL_METADATA),
        }
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    report = align_h5(args.input, args.output)
    print(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
