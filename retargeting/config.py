"""Single source of truth for the LinkerHand L21 retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAND25_POINTS = 25
TEMPORAL_WINDOW = 3
HAND_ANGLE_DIM = 18
HAND_DOF = 17
FK_FIXED_TIPS = 5
HAND_COORDINATE_DIM = 3
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "hand_retargeting" / "checkpoint"
DEFAULT_OUTPUT_H5 = PROJECT_ROOT / "outputs" / "hand_retargeting" / "twohand_angles.h5"
DEFAULT_REALTIME_SNAPSHOT = PROJECT_ROOT / "outputs" / "hand_retargeting" / "mediapipe_realtime.png"
DEFAULT_REALTIME_CHECKPOINT = (
    DEFAULT_CHECKPOINT_ROOT / "models" / "twohand_h5" / "linker" / "palm_local_v2" / "model_best.pth"
)
DEFAULT_MEDIAPIPE_ASSET = PROJECT_ROOT / "hand_landmarker.task"
DEFAULT_INPUT_H5 = PROJECT_ROOT / "input" / "visual_hand_data_20260912_112108.h5"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoint"
    / "models"
    / "twohand_h5"
    / "linker"
    / "coord_aligned_100ep"
    / "model_best.pth"
)
DEFAULT_WARMSTART_CHECKPOINT = None


@dataclass(frozen=True)
class ModelConfig:
    receptive_field: int = TEMPORAL_WINDOW
    input_joints: int = HAND25_POINTS
    input_channels: int = HAND_COORDINATE_DIM
    output_joints: int = HAND_ANGLE_DIM
    embed_dim_ratio: int = 32
    spatial_depth: int = 6
    temporal_depth: int = 4
    spatial_mlp_ratio: float = 4.0
    temporal_mlp_ratio: float = 1.0
    num_heads: int = 8
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_path_rate: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0001
    val_ratio: float = 0.2
    early_stopping_patience: int = 20
    seed: int = 1234
    weight_decay: float = 0.01
    optimizer_eps: float = 1e-6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    minimum_learning_rate: float = 1e-6
    gradient_clip_norm: float = 10.0
    loss_weights: tuple[float, ...] = (500, 500, 10, 500, 10, 500)
    collision_threshold: float = 0.010
    source_scale: float = 1.0
    robot_scale: float = 1.0


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    export_batch_size: int = 256
    camera_device: str = "cpu"
    camera_index: int = 0
    camera_frames: int = 300
    view: str = "iso"
    palm_radius: float = 0.22
    robot_radius: float = 0.16


RUNTIME = RuntimeConfig()


JOINT_NAMES = (
    "hand_base_link",
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
    "thumb_mcp", "thumb_ip",
    "index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip",
)

JOINT_EDGES = (
    ("hand_base_link", "index_mcp_roll"),
    ("index_mcp_roll", "index_mcp_pitch"),
    ("index_mcp_pitch", "index_pip"),
    ("hand_base_link", "middle_mcp_roll"),
    ("middle_mcp_roll", "middle_mcp_pitch"),
    ("middle_mcp_pitch", "middle_pip"),
    ("hand_base_link", "ring_mcp_roll"),
    ("ring_mcp_roll", "ring_mcp_pitch"),
    ("ring_mcp_pitch", "ring_pip"),
    ("hand_base_link", "pinky_mcp_roll"),
    ("pinky_mcp_roll", "pinky_mcp_pitch"),
    ("pinky_mcp_pitch", "pinky_pip"),
    ("hand_base_link", "thumb_cmc_roll"),
    ("thumb_cmc_roll", "thumb_cmc_yaw"),
    ("thumb_cmc_yaw", "thumb_cmc_pitch"),
    ("thumb_cmc_pitch", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"),
    ("index_pip", "index_tip"),
    ("middle_pip", "middle_tip"),
    ("ring_pip", "ring_tip"),
    ("pinky_pip", "pinky_tip"),
    ("thumb_ip", "thumb_tip"),
)

ANGLE_LIMITS = (
    (0.0, 0.0),
    (-0.18, 0.18), (0.0, 1.57), (0.0, 1.57),
    (-0.18, 0.18), (0.0, 1.57), (0.0, 1.57),
    (-0.18, 0.18), (0.0, 1.57), (0.0, 1.57),
    (-0.18, 0.18), (0.0, 1.57), (0.0, 1.57),
    (-0.6, 0.6), (0.0, 1.6), (0.0, 1.0),
    (0.0, 1.57), (0.0, 1.57),
)

SOURCE_JOINTS = {
    "TIP_dic": [4, 9, 14, 19, 24],
    "DIP_dic": [3, 8, 13, 18, 23],
    "PIP_dic": [2, 7, 12, 17, 22],
    "MCP_dic": [1, 6, 11, 16, 21],
    "PALM_dic": [1, 5, 10, 15, 20],
}

ROBOT_JOINTS = {
    "TIP_dic": [22, 18, 19, 20, 21],
    "DIP_dic": [17, 3, 6, 9, 12],
    "PIP_dic": [15, 2, 5, 8, 11],
    "MCP_dic": [14, 1, 4, 7, 10],
}

EXCLUDED_COLLISION_PAIRS = ((1, 2), (4, 5), (7, 8), (10, 11), (14, 15))


@dataclass(frozen=True)
class L21Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    left_urdf: Path = PROJECT_ROOT / "dataset" / "robot" / "l21_left" / "linkerhand_l21_left.urdf"
    right_urdf: Path = PROJECT_ROOT / "dataset" / "robot" / "l21_right" / "linkerhand_l21_right.urdf"

    def model_kwargs(self) -> dict[str, Any]:
        model = self.model
        return {
            "num_frame": model.receptive_field,
            "in_num_joints": model.input_joints,
            "in_chans": model.input_channels,
            "out_num_joint": model.output_joints,
            "out_chans": 1,
            "embed_dim_ratio": model.embed_dim_ratio,
            "spatial_depth": model.spatial_depth,
            "temporal_depth": model.temporal_depth,
            "spatial_mlp_ratio": model.spatial_mlp_ratio,
            "temporal_mlp_ratio": model.temporal_mlp_ratio,
            "num_heads": model.num_heads,
            "qkv_bias": model.qkv_bias,
            "qk_scale": model.qk_scale,
            "drop_path_rate": model.drop_path_rate,
            "angle_limit_rad": ANGLE_LIMITS,
        }

    def hand_kinematics_config(self) -> dict[str, Any]:
        return {
            "joints_name": list(JOINT_NAMES),
            "edges": [list(edge) for edge in JOINT_EDGES],
            "root_name": "hand_base_link",
            "end_effectors": ["index_pip", "middle_pip", "ring_pip", "pinky_pip", "thumb_ip"],
            "elbows": [
                "index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch",
                "pinky_mcp_pitch", "thumb_mcp",
            ],
        }


L21 = L21Config()
