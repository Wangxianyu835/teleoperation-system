"""Offline assembly of TRON2A arm IK and existing L21 hand angles."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from teleoperation.retargeting.arm.ik import ARM_DOF, RoboticsToolboxArmIK, load_arm_specification
from teleoperation.retargeting.arm.calibration import ArmCalibration
from teleoperation.retargeting.arm.retargeter import ArmRetargeter
from teleoperation.retargeting.arm.safety import ArmSafetyController
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.retargeting.hand.angles import angle18_to_dofs


def export_robot_commands(observation_h5: str | Path, angle_h5: str | Path, calibration_path: str | Path, output_path: str | Path, urdf_path: str | Path) -> int:
    frame_ids, timestamps, arms, arm_valid = _load_arm_observations(observation_h5)
    angle_data = _load_hand_angles(angle_h5, frame_ids, timestamps)
    for side in ("left", "right"):
        angle_data[side + "_q"] = np.stack([angle18_to_dofs(angle) for angle in angle_data[side + "_angles"]])
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
            observation = UpperBodyObservation({s: arms[s][index] for s in ("left", "right")}, {s: bool(arm_valid[s][index]) for s in ("left", "right")}, float(timestamp))
            target = retargeter.update(observation, side)
            ee_pose, observation_valid = target.pose, target.valid
            result = solvers[side].solve(target, controllers[side].q, calibration.safe_q[side]) if observation_valid else None
            solved = result is not None and result.success
            q, command_valid = controllers[side].update(result.q if solved else None, solved, float(timestamp))
            values[side] = (ee_pose, q, command_valid)
            output[f"{side}_arm_ee_pose"][index] = ee_pose
            output[f"{side}_arm_q"][index] = q
            output[f"{side}_arm_valid"][index] = solved

        command = Tron2AL21Command(
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

    write_robot_commands(output_path, frame_ids, timestamps, output, urdf_path, calibration_path)
    return frame_ids.shape[0]


def _empty_output(frame_count: int) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {"robot_command": np.zeros((frame_count, 48), dtype=np.float32)}
    for side in ("left", "right"):
        data[f"{side}_arm_ee_pose"] = np.zeros((frame_count, 7), dtype=np.float32)
        data[f"{side}_arm_q"] = np.zeros((frame_count, ARM_DOF), dtype=np.float32)
        data[f"{side}_hand_q"] = np.zeros((frame_count, 17), dtype=np.float32)
        data[f"{side}_arm_valid"] = np.zeros(frame_count, dtype=bool)
        data[f"{side}_hand_valid"] = np.zeros(frame_count, dtype=bool)
    return data


from teleoperation.contracts.observations import UpperBodyObservation
from teleoperation.data.command_h5 import _load_arm_observations, _load_hand_angles, write_robot_commands
