"""Offline assembly of TRON2A arm IK and accepted native L21 hand commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from teleoperation.retargeting.arm.ik import ARM_DOF
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.applications.dual import Tron2ADualArmController
from teleoperation.data.command_h5 import _load_arm_observations, _load_hand_commands, write_robot_commands


def export_robot_commands(observation_h5: str | Path, hand_joint_h5: str | Path, calibration_path: str | Path, output_path: str | Path, urdf_path: str | Path) -> int:
    frame_ids, timestamps, _, _ = _load_arm_observations(observation_h5)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Observation timestamps must be finite strictly increasing seconds")
    from teleoperation.data.body_recording import load_body_recording, replay_observation
    observations = [replay_observation(row) for row in load_body_recording(observation_h5).samples]
    hands = _load_hand_commands(hand_joint_h5, frame_ids, timestamps)
    controller = Tron2ADualArmController(calibration_path, urdf_path, input_mode="smplerx")
    output = _empty_output(len(frame_ids))
    for index, timestamp in enumerate(timestamps):
        raw = observations[index]
        valid = controller.update(raw.upper_body, float(timestamp), raw.wrists)
        command = Tron2AL21Command(
            timestamp=float(timestamp),
            left_arm=controller.controllers["left"].q.copy(), left_hand=hands["left_q"][index],
            right_arm=controller.controllers["right"].q.copy(), right_hand=hands["right_q"][index],
            left_arm_valid=valid["left"], left_hand_valid=bool(hands["left_valid"][index]),
            right_arm_valid=valid["right"], right_hand_valid=bool(hands["right_valid"][index]),
        )
        output["robot_command"][index] = command.flatten()
        for side in ("left", "right"):
            output[f"{side}_arm_ee_pose"][index] = controller.diagnostics[side]["target_pose"]
            for kind in ("arm", "hand"):
                output[f"{side}_{kind}_q"][index] = getattr(command, f"{side}_{kind}")
                output[f"{side}_{kind}_valid"][index] = getattr(command, f"{side}_{kind}_valid")
    write_robot_commands(output_path, frame_ids, timestamps, output, urdf_path, calibration_path)
    return len(frame_ids)


def _empty_output(frame_count: int) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {"robot_command": np.zeros((frame_count, 48), dtype=np.float32)}
    for side in ("left", "right"):
        data[f"{side}_arm_ee_pose"] = np.zeros((frame_count, 7), dtype=np.float32)
        data[f"{side}_arm_q"] = np.zeros((frame_count, ARM_DOF), dtype=np.float32)
        data[f"{side}_hand_q"] = np.zeros((frame_count, 17), dtype=np.float32)
        data[f"{side}_arm_valid"] = np.zeros(frame_count, dtype=bool)
        data[f"{side}_hand_valid"] = np.zeros(frame_count, dtype=bool)
    return data


def main(args):
    count = export_robot_commands(args.observations, args.angle_h5, args.calibration, args.output, args.urdf)
    print(f"output={args.output}")
    print(f"frames={count}")
    return 0
