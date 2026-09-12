"""Compare two-hand coordinate and retargeting experiments offline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from config.variables_define import (
    angle_limit_rob,
    correction_matrix,
    drop_path_rate,
    embed_dim_ratio,
    hand_cfg,
    hand_brand,
    in_chans,
    num_heads,
    num_joints,
    out_num_joint,
    qk_scale,
    qkv_bias,
    receptive_field,
    rb_dic,
    scaling_factor,
    scaling_factor_rb,
    source_dic,
    spatial_depth,
    spatial_mlp_ratio,
    temporal_depth,
    temporal_mlp_ratio,
    left_urdf_file,
    right_urdf_file,
)
from dataset.twohand_h5_dataset import (
    HAND_SIDES,
    TwoHandH5ChunkedGenerator,
    TwoHandH5Dataset,
    load_twohand_h5,
)
from input_adapters.coordinate_modes import (
    transform_fk_positions,
    validate_left_coordinate_mode,
)
from model.angle2real import create_hand_kinematics
from model.model_poseformer import PoseTransformer


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_RUNS = (
    "none_warmstart",
    "mirror_x_warmstart",
    "none_random",
    "mirror_x_random",
)


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids, timestamps, raw_hands = load_twohand_h5(args.input)
    coordinate_rows, coordinate_summary = _coordinate_report(
        args.input,
        raw_hands,
    )
    _write_csv(
        output_dir / "coordinate_stats.csv",
        coordinate_rows,
        [
            "mode",
            "side",
            "valid_frames",
            "mean_x",
            "mean_y",
            "mean_z",
            "std_x",
            "std_y",
            "std_z",
            "min_x",
            "min_y",
            "min_z",
            "max_x",
            "max_y",
            "max_z",
            "left_right_mean_abs_diff",
        ],
    )

    urdf_summary = _urdf_report()
    historical_path = (
        Path(args.historical_checkpoint)
        if args.historical_checkpoint is not None
        else Path(args.checkpoint_root)
        / "models"
        / "twohand_h5"
        / hand_brand
        / "model_best.pth"
    )
    run_specs = [
        {
            "run_name": "historical_baseline",
            "checkpoint": historical_path,
            "historical": True,
        }
    ]
    for run_name in args.runs:
        run_specs.append(
            {
                "run_name": run_name,
                "checkpoint": (
                    Path(args.checkpoint_root)
                    / "models"
                    / "twohand_h5"
                    / hand_brand
                    / run_name
                    / "model_best.pth"
                ),
                "historical": False,
            }
        )

    experiment_rows = []
    tip_rows = []
    for spec in run_specs:
        checkpoint_path = spec["checkpoint"]
        if not checkpoint_path.is_file():
            experiment_rows.append(
                {
                    "run_name": spec["run_name"],
                    "status": "missing_checkpoint",
                    "checkpoint": str(checkpoint_path),
                }
            )
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        mode = _checkpoint_mode(checkpoint, historical=spec["historical"])
        metrics_path = checkpoint_path.parent / "epoch_metrics.csv"
        metrics = _read_metrics(metrics_path)
        best_metrics = _best_metrics(metrics)
        tip_summary, rows = _tip_errors(
            args.input,
            checkpoint_path,
            mode,
            device,
        )
        tip_rows.extend(
            {
                "run_name": spec["run_name"],
                "init_mode": (
                    "historical"
                    if spec["historical"]
                    else (
                        "warmstart"
                        if checkpoint.get("init_checkpoint")
                        else "random"
                    )
                ),
                "coordinate_mode": mode,
                **row,
            }
            for row in rows
        )
        experiment_rows.append(
            {
                "run_name": spec["run_name"],
                "status": "ok",
                "checkpoint": str(checkpoint_path),
                "init_mode": (
                    "historical"
                    if spec["historical"]
                    else (
                        "warmstart"
                        if checkpoint.get("init_checkpoint")
                        else "random"
                    )
                ),
                "coordinate_mode": mode,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_best_epoch": checkpoint.get("best_epoch"),
                "checkpoint_best_val": checkpoint.get("best_val"),
                "metrics_best_epoch": best_metrics.get("epoch"),
                "metrics_best_val": best_metrics.get("val_total"),
                "val_left_total": best_metrics.get("val_left_total"),
                "val_right_total": best_metrics.get("val_right_total"),
                "left_valid": tip_summary["left_valid"],
                "right_valid": tip_summary["right_valid"],
                "left_tip_mean": tip_summary["left_mean"],
                "right_tip_mean": tip_summary["right_mean"],
                "left_tip_rmse": tip_summary["left_rmse"],
                "right_tip_rmse": tip_summary["right_rmse"],
                "left_tip_max": tip_summary["left_max"],
                "right_tip_max": tip_summary["right_max"],
                "tip_by_finger": tip_summary["by_finger"],
            }
        )

    _write_csv(
        output_dir / "tip_errors.csv",
        tip_rows,
        [
            "run_name",
            "init_mode",
            "coordinate_mode",
            "side",
            "frame_index",
            "finger",
            "error",
        ],
    )
    summary = {
        "input": str(args.input),
        "frames": int(frame_ids.shape[0]),
        "timestamps": int(timestamps.shape[0]),
        "missing_frames": {
            side: int(
                np.sum(
                    ~(
                        np.isfinite(raw_hands[side]).all(axis=(1, 2))
                        & ~np.all(raw_hands[side] == 0, axis=(1, 2))
                    )
                )
            )
            for side in HAND_SIDES
        },
        "coordinate_summary": coordinate_summary,
        "urdf": urdf_summary,
        "experiments": experiment_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"summary={output_dir / 'summary.json'}")
    print(f"coordinate_stats={output_dir / 'coordinate_stats.csv'}")
    print(f"tip_errors={output_dir / 'tip_errors.csv'}")
    return 0


def _coordinate_report(input_path, raw_hands):
    converted = {}
    rows = []
    for mode in ("none", "mirror_x"):
        for side in HAND_SIDES:
            dataset = TwoHandH5Dataset(
                input_path,
                receptive_field=receptive_field,
                scale_factor=scaling_factor,
                left_coordinate_mode=mode,
            )
            values = []
            for sample in dataset.samples:
                if sample[f"{side}_valid"]:
                    values.append(sample[f"{side}_target"][0])
            array = np.asarray(values, dtype=np.float32)
            converted[(mode, side)] = array
            flat = array.reshape(-1, 3)
            rows.append(
                {
                    "mode": mode,
                    "side": side,
                    "valid_frames": len(array),
                    **_vector_stats(flat),
                    "left_right_mean_abs_diff": "",
                }
            )
        left_mean = converted[(mode, "left")].mean(axis=0)
        right_mean = converted[(mode, "right")].mean(axis=0)
        diff = float(np.abs(left_mean - right_mean).mean())
        for row in rows:
            if row["mode"] == mode:
                row["left_right_mean_abs_diff"] = diff
    return rows, {
        mode: {
            "left_right_mean_abs_diff": next(
                float(row["left_right_mean_abs_diff"])
                for row in rows
                if row["mode"] == mode
            )
        }
        for mode in ("none", "mirror_x")
    }


def _vector_stats(values):
    return {
        "mean_x": float(values[:, 0].mean()),
        "mean_y": float(values[:, 1].mean()),
        "mean_z": float(values[:, 2].mean()),
        "std_x": float(values[:, 0].std()),
        "std_y": float(values[:, 1].std()),
        "std_z": float(values[:, 2].std()),
        "min_x": float(values[:, 0].min()),
        "min_y": float(values[:, 1].min()),
        "min_z": float(values[:, 2].min()),
        "max_x": float(values[:, 0].max()),
        "max_y": float(values[:, 1].max()),
        "max_z": float(values[:, 2].max()),
    }


def _urdf_report():
    report = {}
    expected = list(hand_cfg["joints_name"])
    for side, path in (("left", left_urdf_file), ("right", right_urdf_file)):
        root = ElementTree.parse(path).getroot()
        joints = [node.attrib["name"] for node in root.findall("joint")]
        links = [node.attrib["name"] for node in root.findall("link")]
        report[side] = {
            "urdf": str(path),
            "explicit_joint_count": len(joints),
            "link_count": len(links),
            "missing_configured_joints": [
                name for name in expected if name not in joints
            ],
            "tip_joints": [
                name for name in expected[-5:] if name in joints
            ],
            "fk_node_count": len(expected),
            "topology_complete": all(
                name in joints or name == hand_cfg.get("root_name")
                and name in links
                for name in expected
            ),
        }
    return report


def _tip_errors(input_path, checkpoint_path, mode, device):
    model = _create_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_pos"]
        if isinstance(checkpoint, dict) and "model_pos" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    dataset = TwoHandH5Dataset(
        input_path,
        receptive_field=receptive_field,
        scale_factor=scaling_factor,
        left_coordinate_mode=mode,
    )
    generator = TwoHandH5ChunkedGenerator(
        dataset,
        batch_size=256,
        shuffle=False,
    )
    fks = {
        "left": create_hand_kinematics(
            left_urdf_file,
            hand_cfg,
            device=device,
            scale_factor=scaling_factor_rb,
            axis_correction_matrix=correction_matrix,
        ),
        "right": create_hand_kinematics(
            right_urdf_file,
            hand_cfg,
            device=device,
            scale_factor=scaling_factor_rb,
            axis_correction_matrix=correction_matrix,
        ),
    }
    errors = {side: [] for side in HAND_SIDES}
    rows = []
    with torch.no_grad():
        for batch in generator.next_epoch():
            for side in HAND_SIDES:
                mask = batch[f"{side}_valid"]
                if not mask.any():
                    continue
                prediction = model(
                    torch.from_numpy(
                        batch[f"{side}_input"][mask].astype(np.float32)
                    ).to(device)
                )
                prediction = F.pad(prediction, (0, 5, 0, 0))
                positions = fks[side].forward(prediction)[2]
                positions = transform_fk_positions(
                    positions,
                    side=side,
                    mode=mode,
                )
                target = torch.from_numpy(
                    batch[f"{side}_target"][mask, 0].astype(np.float32)
                ).to(device)
                tip_error = torch.linalg.vector_norm(
                    positions[:, rb_dic["TIP_dic"], :]
                    - target[:, source_dic["TIP_dic"], :],
                    dim=-1,
                ).cpu().numpy()
                frame_indices = batch["frame_index"][mask]
                errors[side].append(tip_error)
                for frame_index, values in zip(frame_indices, tip_error):
                    for finger, value in zip(FINGER_NAMES, values):
                        rows.append(
                            {
                                "side": side,
                                "frame_index": int(frame_index),
                                "finger": finger,
                                "error": float(value),
                            }
                        )
    summary = {}
    by_finger = {}
    for side in HAND_SIDES:
        if not errors[side]:
            summary[side] = {
                "valid": 0,
                "mean": None,
                "rmse": None,
                "max": None,
            }
            by_finger[side] = {}
            continue
        array = np.concatenate(errors[side], axis=0)
        summary[side] = {
            "valid": int(array.shape[0]),
            "mean": float(array.mean()),
            "rmse": float(np.sqrt(np.mean(array * array))),
            "max": float(array.max()),
        }
        by_finger[side] = {
            finger: {
                "mean": float(array[:, index].mean()),
                "median": float(np.median(array[:, index])),
                "rmse": float(
                    np.sqrt(np.mean(array[:, index] * array[:, index]))
                ),
                "max": float(array[:, index].max()),
            }
            for index, finger in enumerate(FINGER_NAMES)
        }
    return {
        "left_valid": summary["left"]["valid"],
        "right_valid": summary["right"]["valid"],
        "left_mean": summary["left"]["mean"],
        "right_mean": summary["right"]["mean"],
        "left_rmse": summary["left"]["rmse"],
        "right_rmse": summary["right"]["rmse"],
        "left_max": summary["left"]["max"],
        "right_max": summary["right"]["max"],
        "by_finger": by_finger,
    }, rows


def _create_model():
    return PoseTransformer(
        num_frame=receptive_field,
        in_num_joints=num_joints,
        in_chans=in_chans,
        out_num_joint=out_num_joint,
        out_chans=1,
        embed_dim_ratio=embed_dim_ratio,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        spatial_mlp_ratio=spatial_mlp_ratio,
        temporal_mlp_ratio=temporal_mlp_ratio,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_path_rate=drop_path_rate,
        angle_limit_rad=angle_limit_rob,
    )


def _checkpoint_mode(checkpoint, historical):
    if historical:
        return validate_left_coordinate_mode(
            checkpoint.get("left_coordinate_mode", "none")
        )
    return validate_left_coordinate_mode(
        checkpoint.get("left_coordinate_mode", "none")
    )


def _read_metrics(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_metrics(rows):
    if not rows:
        return {}
    row = min(rows, key=lambda value: float(value["val_total"]))
    return {
        "epoch": int(row["epoch"]),
        "val_total": float(row["val_total"]),
        "val_left_total": float(row["val_left_total"]),
        "val_right_total": float(row["val_right_total"]),
    }


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose two-hand coordinate and model experiments"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/visual_hand_data_20260912_112108.h5"),
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--historical-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/twohand_diagnostics"),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def _resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
