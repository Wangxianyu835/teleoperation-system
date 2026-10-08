from pathlib import Path
import numpy as np
from .config import ANGLE_LIMITS
from teleoperation.contracts.constants import HAND_SIDES
def summarize_angles(path, arrays, attrs) -> dict[str, object]:
    path = Path(path)

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
