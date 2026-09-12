"""
手部关键点转换辅助模块，用于重定向输入适配器（retargeting input adapters）。
主要功能：
1. 将不同来源的手部关键点（如 MediaPipe 的 21 点）统一转换为项目内部使用的标准 25 点格式。
2. 提供手腕相对坐标转换（使腕部为原点）。
3. 维护一个滑动窗口缓冲区，用于收集连续帧并构建三帧窗口输入，以供后续模型使用。
"""

from __future__ import annotations

from collections import deque
from itertools import product
from typing import Deque

import numpy as np

# 从重定向标准配置中导入常量：
# - HAND_COORDS: 坐标维度数，通常为 3 (x, y, z)
# - HAND_KEYPOINTS: 项目规范的关键点数，通常为 25
# - HAND_SIDES: 手部两侧标识，如 ("left", "right")
# - RECEPTIVE_FIELD: 时间感受野大小，即缓存多少帧
# - build_retarget_input: 构建最终模型输入的函数
from retargeting.contracts import (
    HAND_COORDS,
    HAND_KEYPOINTS,
    HAND_SIDES,
    RECEPTIVE_FIELD,
    build_retarget_input,
)

# MediaPipe 手部检测的标准关键点数为 21（包含腕部及每根手指的多个关节）
MEDIAPIPE_HAND_KEYPOINTS = 21

# 定义几种数据来源标识，用于记录输入来源
VISIONPRO_SOURCE = "visionpro"          # Apple Vision Pro 数据源
MEDIAPIPE_APPROX_SOURCE = "mediapipe_approx"  # MediaPipe 近似数据源

DEFAULT_MAX_CENTER_DISPLACEMENT = 0.08
DEFAULT_MAX_SHAPE_RMSE = 0.05


class HandIdentityTracker:
    """Keep MediaPipe detections attached to continuous left/right tracks."""

    def __init__(
        self,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        self.max_center_displacement = float(max_center_displacement)
        self.max_shape_rmse = float(max_shape_rmse)
        if self.max_center_displacement <= 0 or self.max_shape_rmse <= 0:
            raise ValueError("Hand tracking thresholds must be positive")
        self._previous: dict[str, np.ndarray | None] = {
            side: None for side in HAND_SIDES
        }

    def update(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | None]:
        detections = [
            (side, np.asarray(points, dtype=np.float32))
            for side, points in (("left", left_hand), ("right", right_hand))
            if _is_trackable_hand(points)
        ]
        return self.update_detections(detections)

    def update_detections(
        self,
        detections: list[tuple[str, np.ndarray]],
    ) -> dict[str, np.ndarray | None]:
        assigned = self.assign(detections)
        self._previous = {
            side: None if assigned[side] is None else assigned[side].copy()
            for side in HAND_SIDES
        }
        return assigned

    def assign(
        self,
        detections: list[tuple[str, np.ndarray]],
    ) -> dict[str, np.ndarray | None]:
        """Assign labeled detections using temporal continuity as the authority."""
        candidates: list[tuple[str, np.ndarray]] = []
        for label, points in detections:
            if label not in HAND_SIDES:
                raise ValueError(f"Invalid hand side: {label}")
            array = np.asarray(points, dtype=np.float32)
            if _is_trackable_hand(array):
                candidates.append((label, array))
        if len(candidates) > len(HAND_SIDES):
            raise ValueError("At most two hand detections are supported")

        result = {side: None for side in HAND_SIDES}
        if not candidates:
            return result
        if all(self._previous[side] is None for side in HAND_SIDES):
            for label, points in candidates:
                if result[label] is None:
                    result[label] = points
            return result

        choices = (None,) + HAND_SIDES
        plausible_existing: list[set[str]] = []
        for _, points in candidates:
            matches = set()
            for side in HAND_SIDES:
                previous = self._previous[side]
                if previous is None:
                    continue
                center_displacement, shape_rmse = _continuity_metrics(
                    points,
                    previous,
                )
                if (
                    center_displacement <= self.max_center_displacement
                    and shape_rmse <= self.max_shape_rmse
                ):
                    matches.add(side)
            plausible_existing.append(matches)

        best_score = None
        best_assignment = None
        for targets in product(choices, repeat=len(candidates)):
            assigned_sides = [target for target in targets if target is not None]
            if len(set(assigned_sides)) != len(assigned_sides):
                continue

            accepted = 0
            label_matches = 0
            total_cost = 0.0
            valid = True
            for candidate_index, ((label, points), target) in enumerate(
                zip(candidates, targets)
            ):
                if target is None:
                    continue
                previous = self._previous[target]
                if previous is None:
                    if target != label or plausible_existing[candidate_index]:
                        valid = False
                        break
                    cost = 0.0
                else:
                    metrics = _continuity_metrics(points, previous)
                    if (
                        metrics[0] > self.max_center_displacement
                        or metrics[1] > self.max_shape_rmse
                    ):
                        valid = False
                        break
                    cost = (
                        metrics[0] / self.max_center_displacement
                        + metrics[1] / self.max_shape_rmse
                    )
                accepted += 1
                label_matches += int(target == label)
                total_cost += cost

            if not valid:
                continue
            score = (accepted, label_matches, -total_cost)
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = targets

        if best_assignment is not None:
            for (_, points), target in zip(candidates, best_assignment):
                if target is not None:
                    result[target] = points
        return result

    def reset(self) -> None:
        for side in HAND_SIDES:
            self._previous[side] = None


def _is_trackable_hand(points: np.ndarray | None) -> bool:
    if points is None:
        return False
    array = np.asarray(points)
    return bool(
        array.ndim == 2
        and array.shape[1:] == (HAND_COORDS,)
        and array.shape[0] in (MEDIAPIPE_HAND_KEYPOINTS, HAND_KEYPOINTS)
        and np.isfinite(array).all()
        and not np.all(array == 0)
    )


def _continuity_metrics(
    current: np.ndarray,
    previous: np.ndarray,
) -> tuple[float, float]:
    if current.shape != previous.shape:
        return float("inf"), float("inf")
    center_displacement = float(
        np.linalg.norm(_palm_center(current) - _palm_center(previous))
    )
    current_shape = current - current[0:1]
    previous_shape = previous - previous[0:1]
    shape_rmse = float(np.sqrt(np.mean((current_shape - previous_shape) ** 2)))
    return center_displacement, shape_rmse


def _palm_center(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == MEDIAPIPE_HAND_KEYPOINTS:
        palm_indices = (0, 5, 9, 13, 17)
    elif points.shape[0] == HAND_KEYPOINTS:
        palm_indices = (0, 6, 11, 16, 21)
    else:
        raise ValueError(f"Unsupported hand point count: {points.shape[0]}")
    return points[np.asarray(palm_indices)].mean(axis=0)


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


class HandWindowBuffer:
    """
    手部帧数据缓冲器，用于收集逐帧的手部关键点，并组装成规范的三帧窗口。

    维护左右手各自的双端队列（deque），队列长度固定为 RECEPTIVE_FIELD（感受野）。
    每次调用 update() 时，将新帧的手部数据（如有）转换并压入对应的队列。
    当左右手都积累了足够帧数时，将队列内容堆叠成形状为 (T, N, 3) 的张量，
    然后调用 build_retarget_input 构建最终的模型输入字典。

    用途：
        为时序模型（如 LSTM 或 Transformer）提供连续多帧的上下文信息。
    """

    def __init__(self, scale_factor: float = 1.0):
        """
        初始化缓冲区。

        参数:
            scale_factor: 应用于所有手部关键点的缩放系数
        """
        self.scale_factor = scale_factor
        # 为左右手分别创建固定长度的 deque，存储的是经过 ensure_hand25 转换后的数组
        self._buffers: dict[str, Deque[np.ndarray]] = {
            side: deque(maxlen=RECEPTIVE_FIELD) for side in HAND_SIDES
        }

    def update(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict | None = None,
    ) -> dict | None:
        """
        向缓冲区添加一帧的手部关键点数据，如果累积足够帧数，则返回模型输入字典。

        参数:
            left_hand: 左手的原始关键点数组（形状 (21,3) 或 (25,3)）
            right_hand: 右手的原始关键点数组
            timestamp: 时间戳（可选）
            source: 数据来源标识
            metadata: 附加元数据

        返回:
            如果左右手都达到 RECEPTIVE_FIELD 帧，则返回 build_retarget_input 构建的字典；
            否则返回 None。
        """
        # 对左右手分别处理：如果提供了关键点，则转换并添加到对应队列
        for side, points in (("left", left_hand), ("right", right_hand)):
            if points is not None:
                self._buffers[side].append(
                    ensure_hand25(points, scale_factor=self.scale_factor)
                )

        # 尝试获取左右手的窗口（堆叠后的帧序列），若任一尚未凑够帧数则返回 None
        left_window = self._window_or_none("left")
        right_window = self._window_or_none("right")
        if left_window is None and right_window is None:
            return None

        # 调用外部构建函数生成最终输入字典
        return build_retarget_input(
            source=source,
            timestamp=timestamp,
            left_hand=left_window,
            right_hand=right_window,
            metadata=metadata,
        )

    def reset(self) -> None:
        """Clear all cached hand frames."""
        for buffer in self._buffers.values():
            buffer.clear()

    def reset_side(self, side: str) -> None:
        """Clear the cached frames for one hand side."""
        if side not in self._buffers:
            raise ValueError(f"Invalid hand side: {side}")
        self._buffers[side].clear()

    def _window_or_none(self, side: str) -> np.ndarray | None:
        """
        获取指定侧（"left" / "right"）的帧窗口堆叠数组。
        如果缓冲区尚未填满，则返回 None。

        返回的数组形状为 (RECEPTIVE_FIELD, 25, 3)，即 T 帧，每帧 25 个关键点。
        """
        buffer = self._buffers[side]
        # 若帧数不足，返回 None
        if len(buffer) != RECEPTIVE_FIELD:
            return None
        # 将 deque 中的元素按顺序堆叠成新的数组，并确保类型为 float32
        return np.stack(tuple(buffer), axis=0).astype(np.float32, copy=False)
