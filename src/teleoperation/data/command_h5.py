from pathlib import Path
import h5py
import numpy as np
def _load_arm_observations(path: str | Path):
    from .body_recording import load_body_recording, replay_observation
    recording = load_body_recording(path)
    bodies = [replay_observation(row).upper_body for row in recording.samples]
    with h5py.File(path, "r") as stream:
        frame_ids = np.asarray(stream["frame_ids"]) if "frame_ids" in stream else np.arange(len(bodies), dtype=np.int64)
    if frame_ids.shape != (len(bodies),) or frame_ids.dtype.kind not in "iu" or np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Observation frame_ids must be ordered unique integers matching body frames")
    timestamps = np.asarray([row['capture_time_s'] for row in recording.samples])
    arms = {side: np.asarray([(body.world_arms or {}).get(side) if (body.world_arms or {}).get(side) is not None
                             else np.full((3, 3), np.nan) for body in bodies]) for side in ('left', 'right')}
    valid = {side: np.asarray([body.valid[side] for body in bodies]) for side in ('left', 'right')}
    count = frame_ids.shape[0]
    if timestamps.shape != (count,):
        raise ValueError("Observation timestamps do not match frame_ids")
    for side in ("left", "right"):
        if arms[side].shape != (count, 3, 3) or valid[side].shape != (count,):
            raise ValueError(f"{side} arm observations must be ({count}, 3, 3) with ({count},) valid")
    return frame_ids, timestamps, arms, valid


def _load_hand_commands(path: str | Path, expected_ids: np.ndarray, expected_timestamps: np.ndarray):
    """Read accepted native L21 commands with strict frame and asset identity."""
    from .hand_joint_recording import iter_joint_commands
    from teleoperation.robots.hand_spec import load_hand_spec
    specs = {("l21", side): load_hand_spec("l21", side) for side in ("left", "right")}
    count = len(expected_ids)
    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != "hand_joint_results_v1":
            raise ValueError("Expected hand_joint_results_v1 accepted commands")
        ids, times = np.asarray(stream["frame_ids"]), np.asarray(stream["timestamps"])
        if ids.shape != (count,) or not np.array_equal(ids, expected_ids):
            raise ValueError("Hand command frame_ids do not match observation H5")
        if (times.shape != (count,) or not np.isfinite(times).all()
                or not np.isfinite(expected_timestamps).all()
                or not np.allclose(times, expected_timestamps, atol=1e-6, rtol=0)):
            raise ValueError("Hand command timestamps do not match observation H5")
        data = {}
        for side in ("left", "right"):
            valid = np.asarray(stream[f"l21/{side}/valid"])
            if valid.shape != (count,) or valid.dtype.kind != "b":
                raise ValueError(f"{side} validity must be a boolean per frame")
            data[f"{side}_valid"] = valid
            data[f"{side}_q"] = np.empty((count, 17), dtype=np.float32)
    for index, (_, commands) in enumerate(iter_joint_commands(path, specs)):
        for key, command in commands.items():
            named = command.named_positions()
            data[f"{key[1]}_q"][index] = [named[n] for n in specs[key].independent_joint_names]
    return data


def write_robot_commands(output_path, frame_ids, timestamps, output, urdf_path, calibration_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("frame_ids", data=frame_ids)
        h5_file.create_dataset("timestamps", data=timestamps)
        for name, value in output.items(): h5_file.create_dataset(name, data=value)
        h5_file.attrs["robot_variant"] = "TRON2A_DACH"
        h5_file.attrs["urdf"] = str(Path(urdf_path))
        h5_file.attrs["calibration"] = str(Path(calibration_path))
        h5_file.attrs["command_order"] = "left_arm,left_hand,right_arm,right_hand"
        h5_file.attrs["invalid_arm_policy"] = "hold_previous"


def read_robot_commands(path):
    with h5py.File(path, "r") as h5_file:
        timestamps = np.asarray(h5_file["timestamps"][:], dtype=np.float64)
        arrays = {name: np.asarray(h5_file[name][:]) for name in ("left_arm_q", "left_hand_q", "right_arm_q", "right_hand_q", "left_arm_valid", "left_hand_valid", "right_arm_valid", "right_hand_valid")}
    return timestamps, arrays
