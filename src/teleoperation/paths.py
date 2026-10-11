"""Repository paths for the supported editable installation."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_ROOT = PROJECT_ROOT / "assets" / "robots" / "from_teleopbench"
L21_ROOT = PROJECT_ROOT / "assets" / "robots" / "l21"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "hand_retargeting" / "checkpoint"
DEFAULT_OUTPUT_H5 = PROJECT_ROOT / "outputs" / "hand_retargeting" / "twohand_angles.h5"
DEFAULT_REALTIME_SNAPSHOT = PROJECT_ROOT / "outputs" / "hand_retargeting" / "mediapipe_realtime.png"
DEFAULT_REALTIME_CHECKPOINT = DEFAULT_CHECKPOINT_ROOT / "models/twohand_h5/linker/palm_local_v2/model_best.pth"
DEFAULT_MEDIAPIPE_ASSET = PROJECT_ROOT / "datasets" / "hand_landmarker.task"
DEFAULT_POSE_ASSET = PROJECT_ROOT / "datasets" / "pose_landmarker_lite.task"
DEFAULT_INPUT_H5 = PROJECT_ROOT / "input" / "visual_hand_data_20260912_112108.h5"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoint/models/twohand_h5/linker/coord_aligned_100ep/model_best.pth"
DEFAULT_WARMSTART_CHECKPOINT = None

DEFAULT_TRON2A_URDF = PROJECT_ROOT / "third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
