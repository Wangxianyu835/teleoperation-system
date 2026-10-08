from __future__ import annotations
import numpy as np
from teleoperation.contracts.constants import *
MEDIAPIPE_HAND_KEYPOINTS = 21
VISIONPRO_SOURCE = "visionpro"
MEDIAPIPE_APPROX_SOURCE = "mediapipe_approx"
def wrist_relative(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    将关键点坐标转换为相对于腕部（索引 0）的偏移量，并可选缩放。

    参数:
        points: 形状为 (N, 3) 的 NumPy 数组，N 为任意关键点数
        scale_factor: 缩放系数，用于尺度归一化

    返回:
        形状为 (N, 3) 的数组，每个点减去腕部坐标后乘以 scale_factor

    用途：
        消除手部在图像中的绝对位置影响，使特征与位置无关。
    """
    points = np.asarray(points, dtype=np.float32)
    # 检查维度：必须是 2 维，且第二维为坐标数（3）
    if points.ndim != 2 or points.shape[1] != HAND_COORDS:
        raise ValueError(f"Hand points must have shape (joints, {HAND_COORDS})")
    # points[0:1] 保持形状为 (1,3)，实现广播减法；然后整体缩放
    return (points - points[0:1]) * float(scale_factor)


def ensure_hand25(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    将支持的手部关键点布局转换为规范化的 25x3 布局。

    支持两种输入格式：
    1. 形状为 (25, 3)：已是规范格式，直接进行手腕相对化处理。
    2. 形状为 (21, 3)：MediaPipe 格式，调用专用转换函数。

    参数:
        points: 输入关键点数组
        scale_factor: 缩放系数

    返回:
        形状为 (25, 3) 的规范化数组（相对腕部，缩放）

    抛出:
        ValueError: 如果输入形状不支持
    """
    points = np.asarray(points, dtype=np.float32)
    # 情况1：已经是项目标准 25 点
    if points.shape == (HAND_KEYPOINTS, HAND_COORDS):
        return wrist_relative(points, scale_factor=scale_factor)
    # 情况2：MediaPipe 的 21 点
    if points.shape == (MEDIAPIPE_HAND_KEYPOINTS, HAND_COORDS):
        return mediapipe21_to_hand25(points, scale_factor=scale_factor)
    # 不支持其他形状
    raise ValueError(
        "Hand points must have shape "
        f"({HAND_KEYPOINTS}, {HAND_COORDS}) or "
        f"({MEDIAPIPE_HAND_KEYPOINTS}, {HAND_COORDS}), got {points.shape}"
    )


def mediapipe21_to_hand25(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """
    将 MediaPipe 的 21 个手部关键点映射到项目自定义的 25 点拓扑。

    项目拓扑在每根非拇指手指的根部添加了一个掌根点（palm-root point），
    位于腕部与该手指的 MCP（掌指关节）之间。
    MediaPipe 并未直接提供这些点，因此我们采用中点（腕部与 MCP 的中点）作为近似。

    映射关系（假设 MediaPipe 的关键点索引顺序与标准一致）：
        - 索引 0：腕部
        - 索引 1-4：拇指（从 MCP 到指尖）
        - 索引 5-8：食指（MCP, PIP, DIP, TIP）
        - 索引 9-12：中指
        - 索引 13-16：无名指
        - 索引 17-20：小指

    项目 25 点布局（假设）：
        - 索引 0：腕部
        - 索引 1-4：拇指（同 MediaPipe 的 1-4）
        - 索引 5：食指掌根（新增，腕部与食指 MCP 的中点）
        - 索引 6-9：食指（对应 MediaPipe 的 5-8）
        - 索引 10：中指掌根（新增）
        - 索引 11-14：中指（对应 MediaPipe 的 9-12）
        - 索引 15：无名指掌根（新增）
        - 索引 16-19：无名指（对应 MediaPipe 的 13-16）
        - 索引 20：小指掌根（新增）
        - 索引 21-24：小指（对应 MediaPipe 的 17-20）

    参数:
        points: 形状 (21, 3) 的 MediaPipe 关键点
        scale_factor: 缩放系数，最终应用于手腕相对化

    返回:
        形状 (25, 3) 的数组，坐标已相对腕部并缩放
    """
    mp_points = np.asarray(points, dtype=np.float32)
    if mp_points.shape != (MEDIAPIPE_HAND_KEYPOINTS, HAND_COORDS):
        raise ValueError(
            f"MediaPipe hand must have shape "
            f"({MEDIAPIPE_HAND_KEYPOINTS}, {HAND_COORDS}), got {mp_points.shape}"
        )

    # 初始化输出数组 (25, 3)
    out = np.zeros((HAND_KEYPOINTS, HAND_COORDS), dtype=np.float32)

    # 索引 0：腕部直接复制
    out[0] = mp_points[0]

    # 拇指（索引 1-4）保持不变，因为项目拓扑中拇指没有新增掌根点
    out[1:5] = mp_points[1:5]

    # 对于其他四根手指，每根手指前插入一个掌根点，然后复制该手指的 4 个关节
    # 手指起始索引：食指 (5)，中指 (9)，无名指 (13)，小指 (17)
    # 输出起始索引：食指 (5)，中指 (10)，无名指 (15)，小指 (20)
    _copy_finger_with_palm_root(out, mp_points, out_start=5, mp_start=5)   # 食指
    _copy_finger_with_palm_root(out, mp_points, out_start=10, mp_start=9)  # 中指
    _copy_finger_with_palm_root(out, mp_points, out_start=15, mp_start=13) # 无名指
    _copy_finger_with_palm_root(out, mp_points, out_start=20, mp_start=17) # 小指

    # 最后进行手腕相对化并应用缩放
    return wrist_relative(out, scale_factor=scale_factor)


def _copy_finger_with_palm_root(
    out: np.ndarray,
    mp_points: np.ndarray,
    out_start: int,
    mp_start: int,
) -> None:
    """
    内部辅助函数：将 MediaPipe 的一根手指（4 个点）复制到输出数组中，
    并在其前方插入一个掌根点（腕部与 MCP 的中点）。

    参数:
        out: 输出数组 (25, 3)，会被原地修改
        mp_points: MediaPipe 数组 (21, 3)
        out_start: 输出中该手指的起始索引（即掌根点应放置的位置）
        mp_start: MediaPipe 中该手指 MCP 的索引（即第一个关节）

    效果：
        out[out_start]     = 腕部 (mp_points[0]) 与 MCP (mp_points[mp_start]) 的中点
        out[out_start+1]   = mp_points[mp_start]   (MCP)
        out[out_start+2]   = mp_points[mp_start+1] (PIP)
        out[out_start+3]   = mp_points[mp_start+2] (DIP)
        out[out_start+4]   = mp_points[mp_start+3] (TIP)
    """
    out[out_start] = 0.5 * (mp_points[0] + mp_points[mp_start])   # 中点
    out[out_start + 1 : out_start + 5] = mp_points[mp_start : mp_start + 4]

def visionpro_fingers_to_points(fingers: list | np.ndarray, side: str = "left") -> np.ndarray:
    """Extract 25 hand points from Vision Pro 4x4 transform matrices."""
    coordinates = []
    for transform_matrix in fingers:
        matrix = np.asarray(transform_matrix)
        if side == "right":
            x = -matrix[1][3]
            y = matrix[2][3]
            z = -matrix[0][3]
        else:
            x = matrix[1][3]
            y = -matrix[2][3]
            z = matrix[0][3]
        coordinates.append([x, y, z])

    return np.asarray(coordinates, dtype=np.float32)
