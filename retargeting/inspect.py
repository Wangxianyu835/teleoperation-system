"""Validate and summarize an exported two-hand angle H5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from retargeting.config import ANGLE_LIMITS
from retargeting.contracts import HAND_SIDES


def inspect_angle_h5(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Angle H5 was not found: {path}")

    required = ("frame_ids", "timestamps") + tuple(
        f"{side}_{suffix}"
        for side in HAND_SIDES
        for suffix in ("angles", "valid")
    )
    with h5py.File(path, "r") as h5_file:
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError(f"Angle H5 is missing datasets: {', '.join(missing)}")
        arrays = {name: np.asarray(h5_file[name][:]) for name in required}
        attrs = {name: h5_file.attrs[name] for name in h5_file.attrs}

    frame_count = arrays["frame_ids"].reshape(-1).shape[0]
    if arrays["timestamps"].reshape(-1).shape[0] != frame_count:
        raise ValueError("timestamps length does not match frame_ids")

    limits = np.asarray(ANGLE_LIMITS, dtype=np.float32)
    summary: dict[str, object] = {
        "path": str(path.resolve()),
        "frames": frame_count,
        "attrs": attrs,
    }
    for side in HAND_SIDES:
        angles = np.asarray(arrays[f"{side}_angles"], dtype=np.float32)
        valid = np.asarray(arrays[f"{side}_valid"], dtype=bool).reshape(-1)
        if angles.shape != (frame_count, 18):
            raise ValueError(f"{side}_angles must have shape ({frame_count}, 18)")
        if valid.shape != (frame_count,):
            raise ValueError(f"{side}_valid must have shape ({frame_count},)")
        finite = np.isfinite(angles).all(axis=1)
        in_limits = (
            (angles >= limits[:, 0] - 1e-4).all(axis=1)
            & (angles <= limits[:, 1] + 1e-4).all(axis=1)
        )
        summary[side] = {
            "shape": angles.shape,
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "nonfinite": int((~finite).sum()),
            "out_of_limits": int((finite & ~in_limits).sum()),
        }
    return summary


def run(args: argparse.Namespace) -> int:
    summary = inspect_angle_h5(args.angle_h5)
    print(f"path={summary['path']}")
    print(f"frames={summary['frames']}")
    for side in HAND_SIDES:
        values = summary[side]
        print(f"{side}_shape={values['shape']}")
        print(f"{side}_valid={values['valid']}")
        print(f"{side}_invalid={values['invalid']}")
        print(f"{side}_nonfinite={values['nonfinite']}")
        print(f"{side}_out_of_limits={values['out_of_limits']}")
    for name, value in sorted(summary["attrs"].items()):
        print(f"attr.{name}={value}")
    return 0


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--angle-h5", type=Path, required=True)
    parser.set_defaults(handler=run)

