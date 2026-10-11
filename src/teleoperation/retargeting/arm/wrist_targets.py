"""21 点手部关键点 -> 手腕（机械臂末端）目标位姿：手臂链路的"近似输入"。

为什么需要它，以及它不是什么
----------------------------
论文 Vision 链路里，机械臂的输入是 **SMPLer-X 估计的手腕 6D 位姿**，再交给 PINK
（``camera_ik_h1_2_media.py``）做全身 IK。本仓库没有 SMPLer-X 权重，也拿不到
TRON2A 的 URDF / 标定文件（队友的 ``retargeting/arm/ik.py`` 是给 TRON2A 写的），
所以 ``dual realtime`` 那条手臂链路在本机跑不起来。

本模块提供的是它的**替代品中的最小一块**：只从 MediaPipe 21 点里读出"手腕在哪、
掌面朝哪"，输出一个末端目标位姿；IK 与仿真由调用方（``devtools/verify_arm_from_capture.py``）
用 PyBullet 自带的 ``calculateInverseKinematics`` 完成。

**必须如实声明**：这里的"图像坐标 -> 机器人坐标"映射是一组**约定常数**，不是标定
结果 —— 真正的映射需要相机内参 + 相机到机器人基座的外参（手眼标定），本机没有。
因此每条数字都可被 ``WristMapping`` 覆盖，默认值只保证"形状与符号自洽、H1-2 的可达
工作空间内 IK 能收敛"（实测数据见 devtools 脚本的 report），**不能**当作论文数值。

坐标系约定（与仓库既有掌面基一致）
----------------------------------
掌面基沿用 ``retargeting/hand/geometric.py`` 与 ``build_l21_reference_basis`` 的定义
（MediaPipe 21 点下标）：:

    X = index MCP(5) - pinky MCP(17)        # 掌侧向，+X 指向食指
    Y = mean(MCP 5,9,13,17) - wrist(0)      # 掌纵向，+Y 指向指尖
    Z = X x Y                               # 掌法向

MediaPipe 归一化坐标：x 向右递增、y 向下递增、z 越靠近相机越负。

到机器人基座坐标的映射（可在 ``WristMapping`` 里改）::

    位置 = base_offset + scale * P @ [x - origin_x, y - origin_y, z]
    P 由 position_order/position_signs 给出，默认 ("Y","Z","X") + (1,-1,-1)，即
        X_robot (前)  <- -z      # 手靠近相机 = 靠近操作者
        Y_robot (左)  <- +(x - origin_x)
        Z_robot (上)  <- -(y - origin_y)

    左右镜像不需要额外符号：正面摄像头下操作者的右手在图像左侧（x 小），
    而机器人 +Y 是"操作者左侧"，同一个公式对两只手都成立。

    姿态 = 掌面基的三根轴按 axis_order/axis_signs 重新排列成机器人末端的
           (X 前, Y 左, Z 上)；例如默认 ("Y","X","Z") 表示
           X_robot <- Y_palm, Y_robot <- -X_palm, Z_robot <- Z_palm。

上面这 3 个轴的去向与 3 个符号都是**约定**（尤其 -z 那一项取决于相机/机器人朝向）。
``scale`` 默认 0.7 m 来自粗算：手距相机约 0.6 m、水平 FOV 约 60 度时，画面宽度约
0.69 m，所以"1 个归一化单位"约等于 0.69 m。它不是标定，只是让默认值在量级上说得通
（实测：scale=1.6 会让 0.10 归一化的扫掠把 H1-2 推到可达边缘，IK 残差 55 mm；
0.7 时见本文件验收脚本的 report）。

退化输入（点重合、NaN、掌面基不可张成）一律返回 ``valid=False`` 并给出原因，调用方
应保持上一帧姿态，而不是拿一个瞎猜的目标去解 IK。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
FOUR_MCPS = (5, 9, 13, 17)
LANDMARK_COUNT = 21
AXIS_NAMES = ("X", "Y", "Z")
# 掌面基的最小长度：低于它就认为是退化输入（归一化坐标下 1e-6 足够小）
MIN_BASIS_LENGTH = 1e-6


def _unit(vector) -> np.ndarray | None:
    """Unit vector, or None when the input is zero-length or non-finite."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= MIN_BASIS_LENGTH:
        return None
    return np.asarray(vector, dtype=np.float64) / norm


@dataclass(frozen=True)
class WristMapping:
    """Convention constants for image -> robot-base coordinates (module doc).

    All of them are tunable; the defaults are self-consistent and inside the
    H1-2 reachable workspace, but they are NOT a hand-eye calibration.
    """

    scale: float = 0.7
    origin_x: float = 0.5
    origin_y: float = 0.5
    position_order: tuple = ("Y", "Z", "X")
    position_signs: tuple = (1.0, -1.0, -1.0)
    base_offset: tuple = (0.24, 0.0, 0.19)
    axis_order: tuple = ("Y", "X", "Z")
    axis_signs: tuple = (1.0, -1.0, 1.0)

    def __post_init__(self):
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if sorted(self.axis_order) != ["X", "Y", "Z"]:
            raise ValueError(
                f"axis_order must be a permutation of X/Y/Z, got {self.axis_order}")
        if sorted(self.position_order) != ["X", "Y", "Z"]:
            raise ValueError(
                "position_order must be a permutation of X/Y/Z, "
                f"got {self.position_order}")
        for name, values in (("axis_signs", self.axis_signs),
                             ("position_signs", self.position_signs),
                             ("base_offset", self.base_offset)):
            if len(values) != 3:
                raise ValueError(f"{name} must have 3 entries, got {values}")


@dataclass
class WristTarget:
    """One frame's end-effector target; ``valid=False`` means "hold the last pose"."""

    position: np.ndarray
    quaternion: np.ndarray
    valid: bool
    reason: str = ""
    palm_frame: np.ndarray | None = None
    normalized_position: np.ndarray | None = None

    def as_tuple(self) -> tuple:
        return (tuple(float(value) for value in self.position),
                tuple(float(value) for value in self.quaternion))


def invalid_target(reason: str) -> WristTarget:
    """Neutral, unusable target (identity orientation) with an explicit reason."""
    return WristTarget(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), False, reason)


def palm_frame_from_landmarks(points) -> np.ndarray | None:
    """Orthonormal palm basis as a (3,3) matrix of columns X/Y/Z, else None.

    Same definition as the repository's L21 palm reference basis, so the hand
    and arm paths agree on what "the palm frame" means. Returns None (instead of
    raising) for missing, non-finite or degenerate input.
    """
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (LANDMARK_COUNT, 3) or not np.isfinite(array).all():
        return None
    lateral = array[INDEX_MCP] - array[PINKY_MCP]
    longitudinal = array[list(FOUR_MCPS)].mean(axis=0) - array[WRIST]
    y_axis = _unit(longitudinal)
    if y_axis is None:
        return None
    x_axis = _unit(lateral - float(np.dot(lateral, y_axis)) * y_axis)
    if x_axis is None:
        return None
    z_axis = _unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_matrix_to_quaternion(matrix) -> np.ndarray:
    """Rotation matrix -> unit quaternion in PyBullet order [x, y, z, w].

    Shepperd's method (numerically stable branch on the largest diagonal term);
    no scipy/tf dependency, and it never returns a non-unit or NaN quaternion.
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= MIN_BASIS_LENGTH:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quaternion / norm


def matrix_from_quaternion(quaternion) -> np.ndarray:
    """Inverse of ``rotation_matrix_to_quaternion`` (used by the tests/harness)."""
    x, y, z, w = (float(value) for value in
                  np.asarray(quaternion, dtype=np.float64).reshape(4))
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def robot_orientation(palm_frame, mapping: WristMapping) -> np.ndarray:
    """Robot-side rotation matrix: columns are the robot's (X, Y, Z) axes.

    The palm basis is mapped by ``mapping.axis_order``/``axis_signs``; a
    left-handed permutation is repaired by rebuilding the third column as
    ``col0 x col1`` so the result is always a proper rotation.
    """
    index = {name: position for position, name in enumerate(AXIS_NAMES)}
    columns = [float(sign) * np.asarray(palm_frame, dtype=np.float64)[:, index[name]]
               for name, sign in zip(mapping.axis_order, mapping.axis_signs)]
    matrix = np.column_stack(columns)
    matrix[:, 2] = np.cross(matrix[:, 0], matrix[:, 1])
    for column in range(3):
        axis = _unit(matrix[:, column])
        if axis is None:
            return np.eye(3)
        matrix[:, column] = axis
    return matrix


def wrist_pose_from_landmarks(points, side: str = "right",
                              mapping: WristMapping | None = None) -> WristTarget:
    """Estimate one frame's wrist/EE target; never raises on bad landmarks.

    ``points`` is the MediaPipe 21x3 array of one hand (or None when the side was
    not detected). ``valid=False`` means the caller should hold the previous
    target: either the landmarks are unusable or the palm basis is degenerate.

    ``side`` is validated but does not change the mapping: with ``Y_robot`` as the
    operator's left, the documented image-x formula already mirrors the two hands
    (see the module docstring). The argument stays in the signature because real
    per-side calibration will need it.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    mapping = mapping or WristMapping()
    frame = palm_frame_from_landmarks(points)
    if frame is None:
        return invalid_target("no usable landmarks (missing, non-finite or degenerate)")
    wrist = np.asarray(points, dtype=np.float64)[WRIST]
    image_offset = np.array([wrist[0] - mapping.origin_x,
                             wrist[1] - mapping.origin_y,
                             wrist[2]], dtype=np.float64)
    axes = {name: index for index, name in enumerate(AXIS_NAMES)}
    robot_offset = np.zeros(3, dtype=np.float64)
    for component, (name, sign) in enumerate(zip(mapping.position_order,
                                                 mapping.position_signs)):
        robot_offset[axes[name]] += float(sign) * image_offset[component]
    position = np.asarray(mapping.base_offset, dtype=np.float64) + mapping.scale * robot_offset
    quaternion = rotation_matrix_to_quaternion(robot_orientation(frame, mapping))
    return WristTarget(position, quaternion, True, "", frame, wrist.copy())
