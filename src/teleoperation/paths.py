"""Repository paths for the supported editable installation."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_ROOT = PROJECT_ROOT / "assets" / "robots"
L21_ROOT = ROBOTS_ROOT / "l21"
DEFAULT_MEDIAPIPE_ASSET = PROJECT_ROOT / "datasets" / "hand_landmarker.task"
DEFAULT_WORLD_CAPTURE_DIR = PROJECT_ROOT / "outputs" / "hand_body_capture"
DEFAULT_HAND_MODEL_ROOT = PROJECT_ROOT / "outputs" / "hand_retargeting" / "models"


def training_run_directory(root, hand_model, side, input_profile, run_name):
    """An explicit target/profile run; never search for an implicit latest model."""
    import re
    for value in (hand_model, side, input_profile, run_name):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"Invalid model path component: {value!r}")
    if side not in ("left", "right", "shared"):
        raise ValueError(f"Invalid target side: {side!r}")
    return Path(root) / hand_model / side / "nn" / input_profile / run_name

DEFAULT_TRON2A_URDF = ROBOTS_ROOT / "tron2a" / "robot.urdf"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
