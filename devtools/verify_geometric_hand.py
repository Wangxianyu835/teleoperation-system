"""L21 往返实测：几何后端从 MediaPipe 21 点能多准地读出 L21 角度。

为什么要这个脚本
----------------
`retargeting/hand/geometric.py` 是不依赖检查点的备用重定向后端，它的正确性
无法靠"看演示像不像"来判断。本脚本把问题变成**可复现的数值实验**：

    1. 用仓库自带的 L21 URDF + FK，造出**已知角度**的手（零位 / 各指屈曲 /
       拇指对掌等若干组姿态）；
    2. 把这 23 个节点反向还原成 MediaPipe 21 点原始输入（与队友采集代码
       完全相同的格式）；
    3. 走**与实时循环一模一样**的处理链：
       21 点 -> ensure_hand25 -> 掌面局部对齐 -> 三帧窗口 -> 几何重定向；
    4. 把恢复出的 18 维角度与第 1 步的真值逐维对比。

因此"准不准"是一个由 L21 资产本身产生的数字，不依赖任何主观判断。

用法（先重定向缓存与 PYTHONPATH，见 docs/CONVENTIONS.md 约定 1/2）：

    E:\\python3.11.7\\python.exe devtools\\verify_geometric_hand.py

退出码 0 = 全部判据通过；1 = 有判据失败（会打印失败原因）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teleoperation.apps.hand_realtime import MediaPipePalmLocalPipeline  # noqa: E402
from teleoperation.contracts.observations import RawHandFrame  # noqa: E402
from teleoperation.retargeting.hand.config import L21  # noqa: E402
from teleoperation.retargeting.hand.geometric import (  # noqa: E402
    GeometricHandRetargeter,
    PIP_DIM,
    PITCH_DIM,
    THUMB_IP_DIM,
    THUMB_MCP_DIM,
    THUMB_PITCH_DIM,
    THUMB_YAW_DIM,
)
from teleoperation.retargeting.hand.kinematics import create_hand_kinematics  # noqa: E402

SIDES = ("left", "right")
# 零位标定帧数：第一个姿态（平手）既提供零位，也用来验证"平手读数为 0"。
CALIBRATION_FRAMES = 3

# L21 FK 的节点下标（JOINT_NAMES 顺序）。MCP 取 **屈曲轴** 节点（*_mcp_pitch），
# 因为 MediaPipe 的 MCP 标的是掌指关节中心，而 L21 的 mcp_roll 在它前面 18mm。
FINGER_NODES = {
    "index": (2, 3, 18),    # mcp_pitch(掌指关节), pip, tip
    "middle": (5, 6, 19),
    "ring": (8, 9, 20),
    "pinky": (11, 12, 21),
}
THUMB_NODES = (13, 16, 17, 22)   # cmc, mcp, ip, tip
FINGER_POINT_STARTS = {"index": 5, "middle": 10, "ring": 15, "pinky": 20}
RAW21_DROP = tuple(FINGER_POINT_STARTS.values())   # 25 点里被插入的掌根点
DIP_FRACTION = 0.45   # 25 点布局没有独立的 DIP 节点：沿 PIP->TIP 取点

# 判据：屈曲维必须为正、真值 0 的维必须接近 0、有真值的维误差上限
ZERO_TOLERANCE = 0.05
FLEX_TOLERANCE = 0.25
MIN_FLEX_RESPONSE = 0.10


_FK_CACHE: dict[str, object] = {}


def fk_for(side: str):
    """Build (and cache) the repository's L21 forward kinematics for one side."""
    if side not in _FK_CACHE:
        _FK_CACHE[side] = create_hand_kinematics(
            getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(),
            device="cpu", scale_factor=L21.training.robot_scale,
        )
    return _FK_CACHE[side]


def build_poses() -> dict[str, dict[int, float]]:
    """Named 18D pose targets used as ground truth, flat pose first."""
    poses = {"flat": {}}
    for name in FINGER_NODES:
        poses[f"{name}_mcp"] = {PITCH_DIM[name]: 0.6}
        poses[f"{name}_pip"] = {PIP_DIM[name]: 0.9}
    poses["fingers_all"] = {dim: 0.8 for dim in (*PITCH_DIM.values(), *PIP_DIM.values())}
    poses["grasp"] = {dim: 1.2 for dim in (*PITCH_DIM.values(), *PIP_DIM.values())}
    poses["thumb_oppose"] = {THUMB_YAW_DIM: 0.7, THUMB_PITCH_DIM: 0.4,
                             THUMB_MCP_DIM: 0.8, THUMB_IP_DIM: 0.9}
    return poses


def fk_nodes(side: str, angles: dict[int, float]) -> np.ndarray:
    """Run the L21 FK for one 18D pose; returns the 23 node positions (23,3)."""
    pose = torch.zeros((1, 23), dtype=torch.float32)
    for dim, value in angles.items():
        pose[0, dim] = float(value)
    with torch.no_grad():
        _, _, positions = fk_for(side).forward(pose)
    return positions[0].detach().cpu().numpy().astype(np.float64)


def nodes_to_points25(nodes: np.ndarray) -> np.ndarray:
    """23 FK nodes -> the pipeline's 25-point layout (wrist-relative later)."""
    points = np.zeros((25, 3), dtype=np.float64)
    points[0] = nodes[0]
    for offset, node in enumerate(THUMB_NODES):
        points[1 + offset] = nodes[node]
    for name, (mcp, pip, tip) in FINGER_NODES.items():
        start = FINGER_POINT_STARTS[name]
        points[start] = 0.5 * (nodes[0] + nodes[mcp])   # 与 mediapipe21_to_hand25 一致
        points[start + 1] = nodes[mcp]
        points[start + 2] = nodes[pip]
        points[start + 3] = nodes[pip] + DIP_FRACTION * (nodes[tip] - nodes[pip])
        points[start + 4] = nodes[tip]
    return points


def points25_to_raw21(points25: np.ndarray) -> np.ndarray:
    """Drop the four inserted palm-root points to get MediaPipe's 21-point input."""
    keep = [index for index in range(25) if index not in RAW21_DROP]
    return np.ascontiguousarray(points25[keep])


def measure_side(side: str, poses: dict[str, dict[int, float]],
                 frames_per_pose: int = CALIBRATION_FRAMES + 2) -> list[tuple[str, dict[int, float], np.ndarray]]:
    """Feed every synthetic hand through the real-time chain and collect angles.

    ``build_poses()`` 的第一个姿态就是平手，也是零位标定帧；比标定帧数多喂 2 帧
    是因为 ``MediaPipePalmLocalPipeline`` 要攒满窗口才出第一个 ``HandWindow``。
    """
    pipeline = MediaPipePalmLocalPipeline()
    retargeter = GeometricHandRetargeter(calibration_frames=CALIBRATION_FRAMES)
    other = "right" if side == "left" else "left"
    rows = []
    for name, angles in poses.items():
        raw = points25_to_raw21(nodes_to_points25(fk_nodes(side, angles)))
        values, window = None, None
        for _ in range(frames_per_pose):
            window = pipeline.process_window(
                RawHandFrame({side: raw, other: None}, 0.0, "synthetic_l21_fk", {}))
            if window is None:
                continue
            # 与实时循环一致：每帧都要 retarget，零位标定才可能攒够帧数。
            frame_values = getattr(retargeter.retarget(window), side + "_angles")
            if frame_values is not None:
                values = np.asarray(frame_values.values, dtype=np.float64)
        if window is None:
            raise AssertionError(f"{side}:{name} produced no window; the synthetic frame is unusable")
        if values is None:
            raise AssertionError(f"{side}:{name} produced no angles; invalid_reasons="
                                 f"{pipeline.invalid_reasons}; calibration="
                                 f"{retargeter.calibration_status}")
        rows.append((name, angles, values))
    return rows


def check_rows(side: str, rows: list) -> list[str]:
    """Every dim with a zero target must stay near zero; flexion dims must respond."""
    failures = []
    for name, truth, values in rows:
        for dim in range(1, 18):
            expected = float(truth.get(dim, 0.0))
            got = float(values[dim])
            if expected == 0.0:
                if abs(got) > ZERO_TOLERANCE:
                    failures.append(f"{side}:{name} dim{dim} target 0 -> {got:+.3f}")
            elif got < MIN_FLEX_RESPONSE or abs(got - expected) > FLEX_TOLERANCE:
                failures.append(f"{side}:{name} dim{dim} target {expected:.3f} -> {got:+.3f}")
    return failures


def report_side(side: str, rows: list) -> None:
    print(f"\n---- {side}")
    for name, truth, values in rows:
        pairs = [f"dim{dim} {float(truth[dim]):.3f} -> {float(values[dim]):+.3f}"
                 for dim in sorted(truth) if float(truth[dim]) != 0.0]
        worst = max([abs(float(values[dim]) - float(truth[dim])) for dim in truth] or [0.0])
        print(f"  {name:14s} | " + ("; ".join(pairs) if pairs else "(all dims target 0)")
              + f"  max_err={worst:.3f}")
        leaked = [f"dim{dim}={float(values[dim]):+.3f}" for dim in range(1, 18)
                  if float(truth.get(dim, 0.0)) == 0.0 and abs(float(values[dim])) > ZERO_TOLERANCE]
        if leaked:
            print(f"     泄漏到未驱动的维: {', '.join(leaked)}")


def main() -> int:
    print("L21 往返实测: FK 姿态 -> MediaPipe 21 点 -> 掌面局部对齐 -> 几何重定向")
    print(f"判据: 目标为 0 的维 |err| <= {ZERO_TOLERANCE}; 目标非 0 的维 "
          f"{MIN_FLEX_RESPONSE} <= value 且 |err| <= {FLEX_TOLERANCE}")
    poses = build_poses()
    failures, worst_per_dim = [], {dim: 0.0 for dim in range(1, 18)}
    for side in SIDES:
        rows = measure_side(side, poses)
        report_side(side, rows)
        failures.extend(check_rows(side, rows))
        for _, truth, values in rows:
            for dim in range(1, 18):
                worst_per_dim[dim] = max(
                    worst_per_dim[dim], abs(float(values[dim]) - float(truth.get(dim, 0.0))))
    print("\n---- 各维最大绝对误差（左右手全部姿态）")
    print("  " + " ".join(f"d{dim}:{worst_per_dim[dim]:.3f}" for dim in range(1, 18)))
    if failures:
        print(f"\n[FAIL] {len(failures)} 条判据不通过：")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\n[OK] 全部判据通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
