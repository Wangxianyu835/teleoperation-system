"""Compare both VR branches against actual local upstream source, without SDK/CAN.

The right worker is constructed from the upstream's commented assignments; it
must not be described as a live upstream right-hand or hardware validation.
"""
import argparse
import ast
import contextlib
import copy
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

from einops import rearrange, repeat
import h5py
import numpy as np
from timm.layers import DropPath
import torch

from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.retargeting.hand.angles import canonical_to_native_angles
from teleoperation.retargeting.hand.config import L21
from teleoperation.retargeting.hand.coordinates import build_l21_reference_basis, OPERATOR_TO_CANONICAL
from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.retargeting.hand.transhandr_vr import load_transhandr_vr_model, TransHandRVRRetargeter


def upstream_window(source, samples, side, scale):
    tree = ast.parse(source)
    worker = copy.deepcopy(next(node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name == "vision_pro_data_process"))
    if side == "right":
        # Activate only the actual adjacent right-hand comment, not a new formula.
        assignments = {}
        for line in ast.get_source_segment(source, worker).splitlines():
            line = line.strip()
            if line.startswith(("# x =", "# y =", "# z =")):
                assignment = ast.parse(line[1:].strip()).body[0]
                assignments[assignment.targets[0].id] = assignment.value
        if set(assignments) != {"x", "y", "z"}:
            raise ValueError("Cannot locate upstream commented right-hand coordinates")
        for node in ast.walk(worker):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in assignments:
                    node.value = assignments[name]
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value == "left_fingers":
                    node.slice.value = "right_fingers"
    class Stream:
        index = 0

        def get_latest(self):
            value = {side + "_fingers": samples[self.index]}
            self.index += 1
            return value
    stream = Stream()
    namespace = {"np": np, "time": SimpleNamespace(time=time.time, sleep=lambda _: None),
                 "VisionProStreamer": lambda **_: stream}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[])),
                 "upstream_vr_worker", "exec"), namespace)
    shared = {}
    with contextlib.redirect_stdout(io.StringIO()):
        namespace[worker.name](shared, SimpleNamespace(is_set=lambda: stream.index >= 3))
    value = shared["vision_pro_data"].transpose(1, 0, 2)[None, ...]
    value *= scale
    return value.astype(np.float32)


def verify(args):
    root = args.upstream_root
    source_path = root / "main_visionpro_realtime.py"
    model_path = root / "model/model_poseformer.py"
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    linker = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "hand_brand == 'linker'")
    upstream_cfg = next(ast.literal_eval(node.value) for node in linker.body
                        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "hand_cfg")
    assert upstream_cfg["joints_name"] == L21.hand_kinematics_config()["joints_name"]
    namespace = {"nn": torch.nn, "torch": torch, "math": math, "partial": partial,
                 "rearrange": rearrange, "repeat": repeat, "DropPath": DropPath}
    nodes = [node for node in ast.parse(model_path.read_text(encoding="utf-8-sig")).body
             if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(model_path), "exec"), namespace)
    torch.set_num_threads(1)
    reference = namespace["PoseTransformer"](**L21.model_kwargs()).eval()
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_pos"], strict=True)
    model = load_transhandr_vr_model(args.checkpoint)
    report = {"branches": {}, "sources": {}}
    with h5py.File(args.input_h5) as handle:
        for side in ("left", "right"):
            valid = handle[side + "_valid"][:]
            candidates = np.flatnonzero(valid[:-2] & valid[1:-1] & valid[2:])
            if not len(candidates):
                raise ValueError(f"No valid three-frame windows for {side}")
            starts = candidates[np.linspace(0, len(candidates) - 1, min(5, len(candidates)), dtype=int)]
            input_error = output_error = 0.
            for start in starts:
                samples = handle[side + "_hand_transforms"][start:start + 3]
                for scale in (1., 1.7):
                    expected = upstream_window(source, samples, side, scale)
                    pipeline = TransHandRVisionProPipeline(scale)
                    for index, sample in enumerate(samples):
                        hands = {"left": None, "right": None, side: sample}
                        window = pipeline.process_window(RawHandFrame(
                            hands, index / 30, "visionpro", {}, {side: "transforms25"}))
                    actual = window.hands[side][None, ...]
                    np.testing.assert_array_equal(actual, expected)
                    with torch.no_grad():
                        raw = reference(torch.from_numpy(expected)).numpy()[0]
                    adapter = TransHandRVRRetargeter(model, source_side=side, target_side=side)
                    native_result = adapter.retarget(window)[("l21", side)]
                    np.testing.assert_array_equal(
                        native_result.positions, canonical_to_native_angles(raw, side)[1:])
                    assert native_result.positions.shape == (17,)
                    ported = adapter.last_raw_angles
                    np.testing.assert_array_equal(ported, raw)
                    input_error = max(input_error, float(np.max(np.abs(actual - expected))))
                    output_error = max(output_error, float(np.max(np.abs(ported - raw))))
            report["branches"][side] = {"windows": len(starts) * 2, "scales": [1., 1.7],
                "input_max_difference": input_error, "output_max_difference": output_error, "raw_output_shape": [18], "native_result_shape": [17],
                "reference": "executed worker" if side == "left" else "activated upstream commented right branch"}
    upstream_urdf = root / "dataset/robot/l21_right/linkerhand_l21_right.urdf"
    reference_fk = create_hand_kinematics(str(upstream_urdf), upstream_cfg, "cpu")
    axes = {}
    for side in ("left", "right"):
        fk = create_hand_kinematics(getattr(L21, side + "_urdf"), L21.hand_kinematics_config(), "cpu")
        if side == "right":
            torch.testing.assert_close(fk.joint_axes, reference_fk.joint_axes, rtol=0, atol=0)
            torch.testing.assert_close(fk.joint_offsets, reference_fk.joint_offsets, rtol=0, atol=0)
        _, rotations, _ = fk.forward(torch.zeros((1, 23)))
        basis = build_l21_reference_basis(side) @ OPERATOR_TO_CANONICAL[side]
        world_axes = np.einsum("nij,nj->ni", rotations[0].numpy(), fk.joint_axes.numpy())
        axes[side] = np.linalg.det(basis) * (world_axes @ basis) * canonical_to_native_angles(np.ones(23), side)[:, None]
    dot = np.sum(axes["left"][1:18] * axes["right"][1:18], axis=-1)
    assert np.all(dot > .995), dot
    report["target_adaptation"] = {"right_fk_equals_upstream": True,
                                    "minimum_axis_dot_after_left_adaptation": float(dot.min()),
                                    "left_sign_reversed_dimensions": [1, 4, 7, 10]}
    for path in (source_path, model_path, upstream_urdf, root / "dataset/data_collect/data_collect_visionpro.py",
                 Path(L21.left_urdf), Path(L21.right_urdf)):
        report["sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["scope"] = "numerical source/FK equivalence; not live headset or motion-quality certification"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("D:/2026/code/TransHandR/TransHandR"))
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    verify(parser.parse_args())
