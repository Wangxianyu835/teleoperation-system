import torch
from .config import L21
JOINT_NAMES = L21.hand_kinematics_config()["joints_name"]

def _angles_for_pose(pose: str) -> torch.Tensor:
    angles = torch.zeros((1, len(JOINT_NAMES)), dtype=torch.float32)
    if pose == "curl":
        for name in (
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
            "index_pip",
            "middle_pip",
            "ring_pip",
            "pinky_pip",
            "thumb_cmc_pitch",
            "thumb_mcp",
            "thumb_ip",
        ):
            angles[0, JOINT_NAMES.index(name)] = 0.55
    return angles

