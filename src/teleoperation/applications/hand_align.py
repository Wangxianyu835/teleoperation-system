"""Read → align → persist, assembled only at the application layer."""
import argparse
import json
from pathlib import Path
from teleoperation.data.hand_h5 import read_alignment_input, write_aligned_copy
from teleoperation.data.transaction import atomic_output
from teleoperation.retargeting.hand.alignment import align_recording


def align_h5(input_path, output_path, landmark_space=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve(): raise ValueError("Input and output H5 paths must be different")
    if not input_path.is_file(): raise FileNotFoundError(f"Input H5 was not found: {input_path}")
    if output_path.exists() and input_path.samefile(output_path): raise ValueError("Input and output H5 files must be different")
    frame_ids, raw, metadata = read_alignment_input(input_path, landmark_space, return_metadata=True)
    from teleoperation.contracts.coordinates import WORLD_LANDMARK_SPACE
    from teleoperation.retargeting.hand.config import L21
    if metadata["source_landmark_space"] == WORLD_LANDMARK_SPACE and L21.training.robot_scale != 1.0:
        raise ValueError("World alignment requires robot_scale=1.0 (URDF metres)")
    aligned, statistics = align_recording(raw)
    with atomic_output(output_path) as temporary:
        write_aligned_copy(input_path, temporary, aligned, metadata)
    return {"input": str(input_path), "output": str(output_path), "frames": int(frame_ids.size), "sides": statistics, "coordinate_metadata": metadata}


def run(args):
    report = align_h5(args.input, args.output, getattr(args, "landmark_space", None))
    print(args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
