from pathlib import Path
import h5py
import numpy as np
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
            data[f"{side}_angles"] = angles
            data[f"{side}_valid"] = np.asarray(h5_file[f"{side}_valid"][:], dtype=bool).reshape(-1)
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
