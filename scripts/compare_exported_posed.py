from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from retargeting.config import JOINT_NAMES, L21
from scripts.visualize_l21_fk import _draw_hand, _set_equal_axes


ROOT = PROJECT_ROOT
FILES = [
    ROOT / "output/manual_angles_20261004_151644_no_tracking.h5",
    ROOT / "output/manual_identity_angles_20261004_151644.h5",
]
TITLES = ["Original rotation", "Identity rotation"]
OUT = ROOT / "picture/compare_rotations_20261004"
OUT.mkdir(parents=True, exist_ok=True)


def read_angles(path):
    with h5py.File(path, "r") as f:
        return {
            name: f[name][:]
            for name in (
                "frame_ids", "timestamps",
                "left_angles", "right_angles",
                "left_valid", "right_valid",
            )
        }


data = [read_angles(path) for path in FILES]

# 确保比较的是同一段采集数据的同一帧。
for key in ("frame_ids", "timestamps"):
    if not np.array_equal(data[0][key], data[1][key]):
        raise ValueError(f"两份文件的 {key} 不一致，不能直接比较")

for side in ("left", "right"):
    # 只选择两份文件中都有效的帧，避开保持上一帧角度的数据。
    valid = (
        data[0][f"{side}_valid"].astype(bool)
        & data[1][f"{side}_valid"].astype(bool)
    )
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        print(f"{side}: 没有共同有效帧")
        continue

    selected = indices[
        np.linspace(0, len(indices) - 1, min(3, len(indices)), dtype=int)
    ]
    print(f"{side}: 选取数组帧下标 {selected.tolist()}")

    urdf = L21.left_urdf if side == "left" else L21.right_urdf
    fk = create_hand_kinematics(
        urdf,
        L21.hand_kinematics_config(),
        device="cpu",
        scale_factor=L21.training.robot_scale,
    )

    poses = []
    with torch.no_grad():
        for item in data:
            angles = torch.zeros(
                (len(selected), len(JOINT_NAMES)), dtype=torch.float32
            )
            angles[:, :18] = torch.from_numpy(
                item[f"{side}_angles"][selected].astype(np.float32)
            )
            _, _, positions = fk.forward(angles)
            poses.append(positions.detach().cpu().numpy())

    # 使用相同坐标范围和观察角度，方便比较。
    common_points = np.concatenate(poses, axis=0).reshape(-1, 3)

    for row, frame in enumerate(selected):
        fig = plt.figure(figsize=(14, 7))

        for column, title in enumerate(TITLES):
            ax = fig.add_subplot(1, 2, column + 1, projection="3d")
            _draw_hand(
                ax, poses[column][row], side=side, label_names=False
            )
            _set_equal_axes(ax, common_points)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=20, azim=-60)
            ax.set_proj_type("ortho")
            ax.set_title(title)

        fig.suptitle(f"L21 {side} - frame index {frame}")
        fig.tight_layout()
        path = OUT / f"{side}_frame_{frame:06d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(path)