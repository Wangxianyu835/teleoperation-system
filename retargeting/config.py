"""Single source of truth for the LinkerHand L21 retargeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_H5 = PROJECT_ROOT / "input" / "aligned_visual_hand_data_20260912_153542.h5"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoint"
    / "models"
    / "twohand_h5"
    / "linker"
    / "my_run"
    / "model_best.pth"
)
DEFAULT_WARMSTART_CHECKPOINT = None


@dataclass(frozen=True)
class ModelConfig:
    receptive_field: int = 3
    input_joints: int = 25
    input_channels: int = 3
    output_joints: int = 18
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
    loss_weights: tuple[float, ...] = (500, 500, 10, 500, 10, 500)
    collision_threshold: float = 0.010
    source_scale: float = 1.0
    robot_scale: float = 1.0


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
