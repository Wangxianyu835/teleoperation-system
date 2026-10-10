"""Read-only coordinate, chirality and scale evidence for a hand H5 file.

Signed volumes are software orientation probes, not a physical handedness
classifier. Missing units remain undeclared; no calibration is inferred.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

from teleoperation.paths import PROJECT_ROOT

from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.retargeting.hand.config import L21, ROBOT_JOINTS, SOURCE_JOINTS
from teleoperation.contracts.coordinates import SUPPORTED_COORDINATE_ALIGNMENTS, COORDINATE_ALIGNMENT, COORDINATE_FRAME
from teleoperation.retargeting.hand.coordinates import SOURCE_TO_L21_MATRIX
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.retargeting.hand.topology import ensure_hand25


def _text_attr(attrs, name):
    value = attrs.get(name)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "UNDECLARED" if value is None else str(value)


def diagnose_h5(path: str | Path, sample_count: int = 5) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    path = Path(path)
    frame_ids, _, hands = load_twohand_h5(path, require_aligned=False)
    attrs = read_attributes(path)
    frame = _text_attr(attrs, "coordinate_frame")
    alignment = _text_attr(attrs, "coordinate_alignment")
    report = {
        "input": str(path),
        "read_only": True,
        "frames": len(frame_ids),
        "source_coordinate_convention": _text_attr(attrs, "source_coordinate_convention"),
        "source_coordinate_frame": _text_attr(attrs, "source_coordinate_frame"),
        "data_source": _text_attr(attrs, "data_source"),
        "coordinate_frame": frame,
        "coordinate_alignment": alignment,
        "aligned_metadata_consistent": frame == COORDINATE_FRAME and alignment in SUPPORTED_COORDINATE_ALIGNMENTS,
        "loader_checks": "Dataset requires l21 plus a supported nonempty alignment; checkpoint loading requires exact alignment equality",
        "declared_units": _text_attr(attrs, "units"),
        "declared_source_units": _text_attr(attrs, "source_units"),
        "source_assumption": (
            "Runtime MediaPipe adapter copies hand_landmarks x/y/z; this H5 alone "
            "does not establish the capture convention or a metric unit."
        ),
        "transform": transform_diagnostics(),
        "chirality_definition": (
            "signed triple product of wrist->thumb/index/pinky MCP; preserves "
            "orientation under proper rotations; not a physical left/right classifier"
        ),
        "source_scale": L21.training.source_scale,
        "robot_scale": L21.training.robot_scale,
        "physical_coordinate_status": "MANUAL VERIFICATION REQUIRED",
        "physical_scale_status": "PHYSICAL SCALE NOT VERIFIED",
        "hands": {},
        "fk": {},
        "distance_loss_assumptions": {
            "tip_distance": "source and FK distances both multiplied by 1000, then MSE; factor does not prove mm",
            "thumb_plane": "absolute point-to-plane distance; FK distance multiplied by 0.9",
            "collision": f"fixed threshold {L21.training.collision_threshold} in scaled FK coordinate units",
        },
        "unresolved": [
            "physical meaning of source and L21 axes and handedness labels",
            "source and URDF length units and their scale correspondence",
            "camera mirror mode, source aspect/depth scaling and perspective",
            "recording-independent accuracy and measured known-pose agreement",
        ],
    }
    tip_names = ("thumb", "index", "middle", "ring", "pinky")
    for side, raw in hands.items():
        finite = np.isfinite(raw).all(axis=(1, 2))
        nonzero = np.any(raw != 0, axis=(1, 2))
        indices = np.flatnonzero(finite & nonzero)
        selected = indices[np.linspace(0, len(indices) - 1, min(sample_count, len(indices)), dtype=int)] if len(indices) else []
        samples = []
        for index in selected:
            canonical = ensure_hand25(raw[index], scale_factor=L21.training.source_scale)
            distances = np.linalg.norm(canonical[SOURCE_JOINTS["TIP_dic"]] - canonical[0], axis=-1)
            samples.append({
                "frame_index": int(index),
                "frame_id": int(frame_ids[index]),
                "wrist_to_tip_distances": dict(zip(tip_names, distances.tolist())),
                "index_to_pinky_mcp_distance": float(np.linalg.norm(canonical[6] - canonical[21])),
                "signed_volume": signed_hand_volume(canonical),
            })
        report["hands"][side] = {
            "stored_shape": list(raw.shape),
            "finite_nonzero_frames": len(indices),
            "nonfinite_frames": int((~finite).sum()),
            "zero_frames": int((finite & ~nonzero).sum()),
            "distance_units": "declared/source coordinate units multiplied by source_scale; no metric conversion",
            "samples": samples,
        }
        urdf = getattr(L21, f"{side}_urdf")
        xml = ET.parse(urdf).getroot()
        # Existing FK constructor prints its path; keep stdout valid JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            fk = create_hand_kinematics(
                urdf, L21.hand_kinematics_config(), "cpu",
                scale_factor=L21.training.robot_scale,
            )
        with torch.no_grad():
            positions = fk.forward(torch.zeros((1, 23)))[2][0].numpy()
        report["fk"][side] = {
            "urdf": str(urdf.relative_to(PROJECT_ROOT)),
            "declared_length_units": xml.attrib.get("length_unit", xml.attrib.get("units", "UNDECLARED")),
            "expected_scale": "URDF origin lengths multiplied by robot_scale; physical match unresolved",
            "zero_pose_wrist_to_tip_distances": dict(zip(
                tip_names, np.linalg.norm(positions[ROBOT_JOINTS["TIP_dic"]] - positions[0], axis=-1).tolist(),
            )),
        }
    return report


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    print(json.dumps(diagnose_h5(args.input, args.sample_count), indent=2, ensure_ascii=False))
    return 0


from teleoperation.retargeting.hand.coordinate_diagnostics import signed_hand_volume, transform_diagnostics

from teleoperation.data.hand_h5 import read_attributes
