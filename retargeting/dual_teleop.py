"""Offline assembly of TRON2A arm IK and existing L21 hand angles."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from retargeting.arm import ARM_DOF, ArmCalibration, ArmRetargeter, ArmSafetyController, RoboticsToolboxArmIK, load_arm_specification
from retargeting.command import RobotCommand
from retargeting.simulation import angle18_to_dofs


def export_robot_commands(observation_h5: str | Path, angle_h5: str | Path, calibration_path: str | Path, output_path: str | Path, urdf_path: str | Path) -> int:
    frame_ids, timestamps, arms, arm_valid = _load_arm_observations(observation_h5)
    angle_data = _load_hand_angles(angle_h5, frame_ids, timestamps)
    calibration = ArmCalibration.load(calibration_path)
    retargeter = ArmRetargeter(calibration)
    solvers = {side: RoboticsToolboxArmIK(load_arm_specification(urdf_path, side)) for side in ("left", "right")}
    controllers = {
        side: ArmSafetyController(solvers[side].specification, calibration.safe_q[side], calibration.max_acceleration, calibration.tracking_timeout)
        for side in ("left", "right")
    }
    output = _empty_output(frame_ids.shape[0])

    for index, timestamp in enumerate(timestamps):
        values = {}
        for side in ("left", "right"):
            ee_pose, observation_valid = retargeter.update(side, arms[side][index], bool(arm_valid[side][index]))
            result = solvers[side].solve(ee_pose, controllers[side].q, calibration.safe_q[side]) if observation_valid else None
            solved = result is not None and result.success
            q, command_valid = controllers[side].update(result.q if solved else None, solved, float(timestamp))
            values[side] = (ee_pose, q, command_valid)
            output[f"{side}_arm_ee_pose"][index] = ee_pose
            output[f"{side}_arm_q"][index] = q
            output[f"{side}_arm_valid"][index] = solved

        command = RobotCommand(
            timestamp=float(timestamp),
            left_arm=values["left"][1], left_hand=angle_data["left_q"][index],
            right_arm=values["right"][1], right_hand=angle_data["right_q"][index],
            left_arm_valid=values["left"][2], left_hand_valid=bool(angle_data["left_valid"][index]),
            right_arm_valid=values["right"][2], right_hand_valid=bool(angle_data["right_valid"][index]),
        )
        output["robot_command"][index] = command.flatten()
        for side in ("left", "right"):
            output[f"{side}_hand_q"][index] = getattr(command, f"{side}_hand")
            output[f"{side}_hand_valid"][index] = getattr(command, f"{side}_hand_valid")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("frame_ids", data=frame_ids)
        h5_file.create_dataset("timestamps", data=timestamps)
        for name, value in output.items():
            h5_file.create_dataset(name, data=value)
        h5_file.attrs["robot_variant"] = "TRON2A_DACH"
        h5_file.attrs["urdf"] = str(Path(urdf_path))
        h5_file.attrs["calibration"] = str(Path(calibration_path))
        h5_file.attrs["command_order"] = "left_arm,left_hand,right_arm,right_hand"
        h5_file.attrs["invalid_arm_policy"] = "hold_previous"
    return frame_ids.shape[0]


def _load_arm_observations(path: str | Path):
    with h5py.File(Path(path), "r") as h5_file:
        required = ["frame_ids", "timestamps"] + [f"{side}_arm_{suffix}" for side in ("left", "right") for suffix in ("keypoints", "valid")]
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError(f"Observation H5 is missing datasets: {', '.join(missing)}")
        frame_ids = np.asarray(h5_file["frame_ids"][:]).reshape(-1)
        timestamps = np.asarray(h5_file["timestamps"][:], dtype=np.float64).reshape(-1)
        arms = {side: np.asarray(h5_file[f"{side}_arm_keypoints"][:], dtype=np.float32) for side in ("left", "right")}
        valid = {side: np.asarray(h5_file[f"{side}_arm_valid"][:], dtype=bool).reshape(-1) for side in ("left", "right")}
    count = frame_ids.shape[0]
    if timestamps.shape != (count,):
        raise ValueError("Observation timestamps do not match frame_ids")
    for side in ("left", "right"):
        if arms[side].shape != (count, 3, 3) or valid[side].shape != (count,):
            raise ValueError(f"{side} arm observations must be ({count}, 3, 3) with ({count},) valid")
    return frame_ids, timestamps, arms, valid


def _load_hand_angles(path: str | Path, expected_ids: np.ndarray, expected_timestamps: np.ndarray):
    with h5py.File(Path(path), "r") as h5_file:
        required = ["frame_ids", "timestamps"] + [f"{side}_{suffix}" for side in ("left", "right") for suffix in ("angles", "valid")]
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError(f"Angle H5 is missing datasets: {', '.join(missing)}")
        if not np.array_equal(np.asarray(h5_file["frame_ids"][:]).reshape(-1), expected_ids):
            raise ValueError("Angle H5 frame_ids do not match observation H5")
        timestamps = np.asarray(h5_file["timestamps"][:], dtype=np.float64).reshape(-1)
        if not np.allclose(timestamps, expected_timestamps, atol=1e-6):
            raise ValueError("Angle H5 timestamps do not match observation H5")
        data = {}
        for side in ("left", "right"):
            angles = np.asarray(h5_file[f"{side}_angles"][:], dtype=np.float32)
            if angles.shape != (expected_ids.shape[0], 18):
                raise ValueError(f"{side}_angles has unexpected shape: {angles.shape}")
            data[f"{side}_q"] = np.stack([angle18_to_dofs(angle) for angle in angles])
            data[f"{side}_valid"] = np.asarray(h5_file[f"{side}_valid"][:], dtype=bool).reshape(-1)
    return data


def _empty_output(frame_count: int) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {"robot_command": np.zeros((frame_count, 48), dtype=np.float32)}
    for side in ("left", "right"):
        data[f"{side}_arm_ee_pose"] = np.zeros((frame_count, 7), dtype=np.float32)
        data[f"{side}_arm_q"] = np.zeros((frame_count, ARM_DOF), dtype=np.float32)
        data[f"{side}_hand_q"] = np.zeros((frame_count, 17), dtype=np.float32)
        data[f"{side}_arm_valid"] = np.zeros(frame_count, dtype=bool)
        data[f"{side}_hand_valid"] = np.zeros(frame_count, dtype=bool)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fixed 48D TRON2A DACH + L21 robot commands")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--angle-h5", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=Path("third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf"))
    args = parser.parse_args()
    count = export_robot_commands(args.observations, args.angle_h5, args.calibration, args.output, args.urdf)
    print(f"output={args.output}")
    print(f"frames={count}")
    return 0
