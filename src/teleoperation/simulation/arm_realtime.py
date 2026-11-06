"""SMPL-X root-relative shoulder/elbow/wrist positions to robot arm targets.

The active camera route supplies tracked body geometry and uses this bridge for
chain/segments/posture mapping. Robot link lengths, limits, IK and command rate
limits remain in their existing owners. Wrist orientation is supplied separately
by the SMPL-X FK chain through WristSceneController. Synthetic numerical probes
may exercise the elbow-plane orientation helper directly.

AxisMap is a configured camera-axis convention, not measured camera extrinsics.
Missing or rejected body sides hold their previous targets. Hands never replace
body input. See docs/smplerx-arm.md for source, timing and calibration contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import acos, degrees
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_profiles import (POSTURE_FRAMES, SIDES,
                                                    ArmProfile)

AXIS_NAMES = ("X", "Y", "Z")
#: 默认轴向映射，见模块 docstring（约定，不是标定）。
DEFAULT_AXIS_ORDER = ("Z", "X", "Y")
DEFAULT_AXIS_SIGNS = (-1.0, 1.0, -1.0)
#: 手臂长度归一化的参考值（米）：高于/矮于这个臂长的操作者由 gain 自动补偿。
REFERENCE_REACH_M = 0.60
#: 归一化系数的允许区间，防止一次坏标定把 gain 推到离谱的值。
ARM_LENGTH_BAND = (0.5, 2.0)
#: Reject degenerate shoulder-to-wrist geometry shorter than this many metres.
MIN_OPERATOR_REACH_M = 0.15
#: 基座系里的默认行程上限（米，每轴），profile 没有 realtime_travel_m 时用它。
DEFAULT_MAX_TRAVEL_M = 0.10
#: 肘平面可信判据：|上臂 x 前臂|（= 上臂与前臂夹角的正弦）小于它时这一帧不给姿态。
MIN_ELBOW_PLANE_SINE = 0.05
TRAVEL_MODES = ("clip", "scale")
#: hold waits for a full neutral window; track uses the available sample median.
CALIBRATE_MODES = ("hold", "track")
#: Stable diagnostic keys for calibration, missing body geometry, short reach and other holds.
HOLD_REASONS = ("calibrating", "no_pose_landmarks", "reach_below_min", "other")
#: chain maps wrist displacement; segments scales each bone; posture maps directions.
MAP_MODES = ("chain", "segments", "posture")
#: Re-export the robot profile direction-frame choices.
POSTURE_FRAMES = POSTURE_FRAMES
#: Operator direction-rate guard in degrees/second; zero disables it.
MAX_DIRECTION_RATE_DEG_S = 600.0
#: 拿不到上一帧间隔时的名义帧长（秒）：与 realtime_teleop.NOMINAL_FRAME_DT 同一口径。
NOMINAL_FRAME_DT_S = 1.0 / 30.0
#: Maximum consecutive direction-guard holds before accepting a persistent change.
GLITCH_HOLD_FRAMES = 3
#: Minimum absolute wrist y in base metres; zero disables the side guard.
DEFAULT_LATERAL_GUARD_M = 0.0
#: Minimum calibrated reach trusted for arm-length normalization, in metres.
MIN_PLAUSIBLE_ARM_REACH_M = 0.30
#: Maximum relative difference between left and right calibrated reach.
ARM_REACH_SIDE_TOLERANCE = 0.25
SOURCE_BODY = "smplerx"
SOURCE_NONE = "none"


def _is_points(value, shape) -> bool:
    """``True`` 表示这是一组给定形状、全有限、可用的点。"""
    if value is None:
        return False
    array = np.asarray(value, dtype=np.float64)
    return array.shape == tuple(shape) and bool(np.isfinite(array).all())


def _unit_direction(vector):
    """单位化一个方向向量；``None`` 表示退化（缺数据 / 非有限 / 长度接近 0）。

    退化不抛异常：调用方据此退回别的口径，并把原因写进那一帧的 ``info``。
    """
    if vector is None:
        return None
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3:
        return None
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return values / norm


def _slerp_direction(start, end, gain: float):
    """两个单位方向之间按夹角插值；``gain=1`` 就是 ``end``（方向口径的 gain 语义）。

    方向口径里 gain 不再是「位移乘多少倍」，而是「机器人离你的方向有多近」：
    1.0 = 完全对齐你的方向（默认），0.5 = 停在机器人中性方向与你的方向正中，0 = 回中性。
    用夹角插值而不是线性插值再单位化，是因为后者在夹角大时角速度不均匀（看起来会先快后慢）。
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    gain = float(min(max(gain, 0.0), 1.0))
    if gain >= 1.0:
        return end
    if gain <= 0.0:
        return start
    angle = float(np.arccos(float(np.clip(np.dot(start, end), -1.0, 1.0))))
    if angle <= 1e-6:
        return end
    sine = float(np.sin(angle))
    if abs(sine) <= 1e-9:
        # 夹角 180 度：最小旋转不唯一，绕一个与自身垂直的固定轴转，至少结果是确定的。
        axis = np.cross(start, np.array([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(start, np.array([1.0, 0.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        cross = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
        rotation = (np.eye(3) + np.sin(gain * angle) * cross
                    + (1.0 - np.cos(gain * angle)) * (cross @ cross))
        return rotation @ start
    return ((np.sin((1.0 - gain) * angle) / sine) * start
            + (np.sin(gain * angle) / sine) * end)


def _rotation_between_directions(start, end):
    """把单位方向 ``start`` 转到 ``end`` 的**最小**旋转（3x3；缺数据返回 ``None``）。

    方向口径的 ``delta`` 档用它：操作者这一帧相对自己中性转了 R，就把**同一个 R** 套在本机
    自己的中性方向上（``R @ rest_u``），而不是把操作者的绝对方向直接搬过来。这样
    「你没动 = 本机停在自己的加载姿势」是恒等式，而不是靠两台手臂的加载姿势恰好一致。

    两条边界：两方向平行（R = I）、反向（旋转不唯一，绕一个与自身垂直的固定轴转 180 度，
    与 ``_slerp_direction`` 的兜底同一口径：结果是确定的，不是随机的）。
    """
    start = _unit_direction(start)
    end = _unit_direction(end)
    if start is None or end is None:
        return None
    cross = np.cross(start, end)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if sine <= 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        axis = np.cross(start, np.array([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(start, np.array([1.0, 0.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    angle = float(np.arctan2(sine, cosine))
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _angle_between_deg(left, right) -> float:
    """两个方向的夹角（度）：报告里用来说明「你的上臂转了多少度」。"""
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(degrees(acos(cosine)))


def elbow_bend_degrees(shoulder, elbow, wrist) -> float:
    """肘关节弯曲角（度）：0 = 完全伸直，180 = 完全折起。

    纯几何诊断量，用来对比「你的肘弯了多少、机器人的肘弯了多少」；NaN/inf 输入返回
    ``float("nan")``，调用方自己决定要不要展示（不参与控制）。
    """
    upper = np.asarray(elbow, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
    lower = np.asarray(elbow, dtype=np.float64) - np.asarray(wrist, dtype=np.float64)
    norms = (float(np.linalg.norm(upper)), float(np.linalg.norm(lower)))
    if not all(np.isfinite(norms)) or min(norms) <= 0.0:
        return float("nan")
    cosine = float(np.dot(upper, lower)) / (norms[0] * norms[1])
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def arm_reach(arm_world) -> float:
    """上臂 + 前臂长度之和（米）；输入不可用时返回 NaN。"""
    if not _is_points(arm_world, (3, 3)):
        return float("nan")
    points = np.asarray(arm_world, dtype=np.float64)
    return float(np.linalg.norm(points[1] - points[0])
                 + np.linalg.norm(points[2] - points[1]))


def wrist_orientation_frame(arm_world):
    """(肩, 肘, 腕) 3x3 点 -> 3x3 腕坐标系（列 = x/y/z 基）；退化时 None。

    与论文的差别写在模块 docstring：论文从 SMPL-X 的旋转链拿手腕 4x4，我们用手上有的
    三点。定义（列向量）：

        x = 前臂方向 (腕 - 肘)        手臂指哪儿（前臂方向，2 个自由度）
        z = 上臂 x 前臂 的单位化       肘的拐向（手臂平面，1 个自由度）
        y = z x x                    补成右手系

    位置 3 + x 2 + z 1 = 6 个自由度，正是论文那条「手腕完整位姿」约束在三点输入下的
    等价物；缺的只有手的轴向扭转（那要用 21 点，见模块 docstring）。

    直臂（三点近共线）时上臂与「肘 -> 腕」的夹角正弦趋零，平面不定：返回 ``None``，
    由调用方保持上一帧，而不是猜一个方向。
    """
    if not _is_points(arm_world, (3, 3)):
        return None
    points = np.asarray(arm_world, dtype=np.float64)
    upper = points[1] - points[0]
    forearm = points[2] - points[1]
    upper_norm = float(np.linalg.norm(upper))
    forearm_norm = float(np.linalg.norm(forearm))
    if upper_norm <= 0.0 or forearm_norm <= 0.0:
        return None
    x_axis = forearm / forearm_norm
    normal = np.cross(upper / upper_norm, x_axis)
    sine = float(np.linalg.norm(normal))
    if not np.isfinite(sine) or sine < MIN_ELBOW_PLANE_SINE:
        return None
    z_axis = normal / sine
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def relative_wrist_rotation(arm_world, neutral_frame):
    """这一帧的腕坐标系相对中性那一帧的旋转 ``R_neutral^T @ R_now``；给不出时 None。

    论文那条「手腕完整位姿」约束的可用部分：轴向约定在相减里约掉，所以不需要相机外参。
    """
    if neutral_frame is None:
        return None
    frame = wrist_orientation_frame(arm_world)
    if frame is None:
        return None
    return np.asarray(neutral_frame, dtype=np.float64).T @ frame


def _average_rotation(matrices):
    """一组 3x3 旋转的均值 -> 最接近的真旋转（SVD 投影）；没有可用值时 None。"""
    usable = [np.asarray(matrix, dtype=np.float64) for matrix in matrices
              if matrix is not None]
    if not usable:
        return None
    mean = np.mean(np.stack(usable, axis=0), axis=0)
    left, _singular, right_t = np.linalg.svd(mean)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


@dataclass(frozen=True)
class AxisMap:
    """操作者系的轴到机器人基座系的轴：每个机器人轴由哪一个操作者轴、带什么符号。

    ``order``/``signs`` 按机器人轴顺序 (X, Y, Z) 给：``order[i]`` 是被取用的操作者轴名，
    ``signs[i]`` 是它的正负号。默认 (``("Z","X","Y")``, ``(-1,1,-1)``) 即模块 docstring 里
    那三行。这里只做置换与符号，不含旋转矩阵拟合 —— 真正的相机外参要标定。
    """

    order: tuple = DEFAULT_AXIS_ORDER
    signs: tuple = DEFAULT_AXIS_SIGNS

    def __post_init__(self) -> None:
        if sorted(self.order) != sorted(AXIS_NAMES):
            raise ValueError(
                f"axis order must be a permutation of X/Y/Z, got {tuple(self.order)!r}")
        if len(self.signs) != 3:
            raise ValueError(f"axis signs must have 3 entries, got {tuple(self.signs)!r}")
        for sign in self.signs:
            if float(sign) not in (-1.0, 1.0):
                raise ValueError(f"axis signs must be +1 or -1, got {tuple(self.signs)!r}")

    def matrix(self) -> np.ndarray:
        """3x3 矩阵 ``M``：``d_base = M @ d_op``。"""
        index = {name: position for position, name in enumerate(AXIS_NAMES)}
        matrix = np.zeros((3, 3), dtype=np.float64)
        for row, (name, sign) in enumerate(zip(self.order, self.signs)):
            matrix[row, index[name]] = float(sign)
        return matrix

    def apply(self, vector) -> np.ndarray:
        """把操作者系的位移向量转到机器人基座系的轴序（只换轴，不缩放）。"""
        values = np.asarray(vector, dtype=np.float64).reshape(3)
        return self.matrix() @ values

    def describe(self) -> dict:
        """JSON 化：给报告和命令行回显用。"""
        return {"order": list(self.order), "signs": [float(sign) for sign in self.signs],
                "robot_axes_from_operator": [
                    f"{name}_robot = {sign:+.0f} * {self.order[row]}_op"
                    for row, (name, sign) in enumerate(
                        zip(AXIS_NAMES, self.signs))]}

    @classmethod
    def parse(cls, order: str, signs: str) -> "AxisMap":
        """命令行写法：``--arm-axis-order ZXY --arm-axis-signs -1,1,-1``（逗号或空格都认）。"""
        letters = str(order).strip().upper().replace(",", "").replace(" ", "")
        values = [part for part in str(signs).replace(" ", ",").split(",") if part]
        if len(values) == 1 and len(values[0]) >= 3:
            values = [values[0][index] for index in range(3)]
        return cls(tuple(letters), tuple(float(value) for value in values))



@dataclass(frozen=True)
class ArmTarget:
    """一台机器人的一侧、某一帧的手臂目标（机器人基座系）。

    ``valid=True`` 表示这一帧真的解算出了新目标；``held=True`` 表示沿用上一帧（该侧
    没数据 / 还在标定 / 目标非法）。``source`` 说明用的是 pose 手臂还是手部兜底映射，
    ``info`` 里是可检查的诊断量（你移动了多少、肘弯了多少、有没有被行程夹住）。

    ``orientation`` 是**相对中性**的四元数（``R_neutral^T @ R_now``）：喂给
    ``PyBulletArmIK.solve(orientation_frame="neutral")`` 才会乘上本机的中性 EE 朝向，
    直接当绝对姿态用是错的。
    """

    side: str
    position: tuple | None = None
    elbow: tuple | None = None
    orientation: tuple | None = None
    valid: bool = False
    held: bool = False
    reason: str = ""
    source: str = SOURCE_NONE
    info: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """JSON 化（报告与 HUD 共用）。"""
        return {
            "side": self.side,
            "position": None if self.position is None else [round(float(v), 5) for v in self.position],
            "elbow": None if self.elbow is None else [round(float(v), 5) for v in self.elbow],
            "orientation": None if self.orientation is None else [round(float(v), 6) for v in self.orientation],
            "valid": bool(self.valid),
            "held": bool(self.held),
            "reason": self.reason,
            "source": self.source,
            "info": self.info,
        }


def _neutral_default() -> dict:
    return {"offset": None, "reach": None, "orientation": None,
            # 上臂段 (肘-肩) 与「前臂+手」段 (腕-肘) 的中性向量：两段式映射要用的基准。
            # 老记录里没有这两项 -> None，那一侧自动退回整段 gain（不会算错，只是不生效）。
            "upper": None, "fore": None,
            "samples": 0, "complete": False}


def _neutral_rotation(orientation):
    """中性记录里的腕坐标系 -> 3x3；缺 / 为空 / 形状不对时 ``None``。

    兼容三种写法：``None``（老记录，那时还不记姿态）、JSON 里的 9 个数、已经是 3x3 的数组。
    """
    if orientation is None:
        return None
    values = np.asarray(orientation, dtype=np.float64)
    if values.size != 9:
        return None
    return values.reshape(3, 3)


def load_neutral(path) -> dict:
    """读一份 ``--save-neutral`` 写出（或 ``--neutral-json`` 指定）的中性姿势记录。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get('format') != 'smplerx_arm_neutral_v1' or payload.get('source') != 'smplerx':
        raise ValueError('Legacy arm neutral calibration is removed; capture a new SMPL-X neutral')
    neutral = {}
    for side in SIDES:
        entry = payload.get("sides", {}).get(side, {})
        offset = entry.get("offset")
        orientation = entry.get("orientation")
        neutral[side] = {
            "offset": None if offset is None else np.asarray(offset, dtype=np.float64),
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
            # 中性那一帧的腕坐标系（相对姿态的基准）。老记录没有这一段 -> None，
            # 于是这一侧退化成位置模式，而不是把缺的姿态当成零。
            "orientation": _neutral_rotation(orientation),
            # 两段式映射（--arm-map segments）的基准：上臂段 / 「前臂+手」段。
            "upper": _neutral_vector(entry.get("upper"), 3),
            "fore": _neutral_vector(entry.get("fore"), 3),
            "samples": int(entry.get("samples", 0)),
            "complete": bool(entry.get("complete", False)) and offset is not None,
        }
    return {"sides": neutral, "axis_map": payload.get("axis_map"),
            "created": payload.get("created"), "path": str(path)}


def _neutral_vector(value, size: int):
    """中性记录里的一段向量（上臂段 / 前臂段）-> ``ndarray``；缺 / 形状不对时 ``None``。

    跟 ``_neutral_rotation`` 一个口径：老记录（或手写错的记录）不给这一段时返回 ``None``，
    调用方据此退回整段 gain，而不是把 ``None`` 当成零向量算出错误位移。
    """
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size:
        return None
    return vector


def save_neutral(path, sides: dict, extra: dict | None = None) -> Path:
    """把中性姿势写进 JSON，方便下次直接复用（不用再站一次）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "smplerx_arm_neutral_v1", 'source': 'smplerx', "extra": extra or {}, "sides": {}}
    for side, entry in sides.items():
        offset = entry.get("offset")
        payload["sides"][side] = {
            "offset": None if offset is None else [float(v) for v in np.asarray(offset).reshape(3)],
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
            # 9 个数（行主序）：中性那一帧的腕坐标系，读回来就是相对姿态的基准。
            "orientation": (None if entry.get("orientation") is None else
                            [float(value) for value in
                             np.asarray(entry["orientation"], dtype=np.float64).reshape(9)]),
            # 两段式映射的基准（上臂段 / 「前臂+手」段，米）。没采到就写 null。
            "upper": None if entry.get("upper") is None else
                     [float(v) for v in np.asarray(entry["upper"]).reshape(3)],
            "fore": None if entry.get("fore") is None else
                    [float(v) for v in np.asarray(entry["fore"]).reshape(3)],
            "samples": int(entry.get("samples", 0)),
            "complete": bool(entry.get("complete", False)),
        }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _side_map(value) -> dict:
    """``{side: ...}`` 的宽松读取：支持 dict / ``UpperBodyObservation`` / ``RawHandFrame``。"""
    if value is None:
        return {}
    if hasattr(value, "world_arms"):
        return {side: points if value.valid.get(side, False) else None
                for side, points in (value.world_arms or {}).items()}
    if isinstance(value, dict):
        return value
    return {}


def _hand_frames(hands) -> tuple:
    """-> ``(normalized, world)`` 两套 21 点字典；缺哪套哪套为 ``{}``。"""
    normalized, world = {}, {}
    if hands is None:
        return normalized, world
    if hasattr(hands, "hands"):
        normalized = dict(getattr(hands, "hands") or {})
        world = dict(getattr(hands, "world_hands") or {})
        return normalized, world
    if isinstance(hands, dict):
        for side in SIDES:
            normalized[side] = hands.get(side)
            world[side] = hands.get(f"world_{side}")
    return normalized, world


class RealtimeArmBridge:
    """一帧摄像头数据 -> 每台机器人每侧一个 ``ArmTarget``（基座系，可直接喂 IK）。

    用法（每个摄像头帧调用一次）::

        targets = bridge.update(arms_world=obs.upper_body, hands=raw_frame)
        for robot_type, per_side in targets.items():
            for side, target in per_side.items():
                if target.position is not None:
                    driver.apply_each(side, {robot_type: target.position},
                                      source="base", orientation=target.orientation)

    这个类只做数学与状态：不 import PyBullet、不开摄像头、不写关节。因此它可以被单元
    测试直接驱动（``tests/test_arm_realtime_bridge.py``），仿真侧只负责把 ``position``
    交给 ``PyBulletArmIK.solve(source="base")``。
    """

    def __init__(self, profiles=(), sides=SIDES, axis_map=None,
                 calibration_frames: int = 30,
                 calibrate_mode: str = "hold",
                 arm_length_reference: float = REFERENCE_REACH_M,
                 normalise_arm_length: bool = True,
                 arm_length_band=ARM_LENGTH_BAND,
                 orientation: str = "auto",
                 gains=None, max_travel_m=None, neutral=None,
                 travel_mode: str = "clip",
                 lateral_guard_m: float = DEFAULT_LATERAL_GUARD_M,
                 gain_quality_guard: bool = True,
                 min_plausible_reach_m: float = MIN_PLAUSIBLE_ARM_REACH_M,
                 arm_reach_side_tolerance: float = ARM_REACH_SIDE_TOLERANCE,
                 map_mode: str = "chain",
                 robot_links=None,
                 elbow_task: bool = False,
                 elbow_gain: float = 1.0,
                 posture_gain: float = 1.0,
                 posture_frame: str | None = None,
                 max_direction_rate_deg_s: float = MAX_DIRECTION_RATE_DEG_S,
                 frame_dt: float | None = None,
                 anchors=None):
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if map_mode not in MAP_MODES:
            raise ValueError(f"map_mode must be one of {MAP_MODES}, got {map_mode!r}")
        if elbow_task and map_mode not in ("segments", "posture"):
            raise ValueError(
                "elbow_task needs map_mode='segments' or map_mode='posture' (in posture the elbow target is built by that recipe itself): the elbow and the wrist target must come from the same length-scaled chain, otherwise the two position tasks fight each other")
        if int(calibration_frames) < 1:
            raise ValueError("calibration_frames must be >= 1")
        if calibrate_mode not in CALIBRATE_MODES:
            raise ValueError(f"calibrate_mode must be one of {CALIBRATE_MODES},"
                             f" got {calibrate_mode!r}")
        if orientation not in ("off", "auto", "on"):
            raise ValueError(
                f"orientation must be 'off', 'auto' or 'on', got {orientation!r}")
        self.profiles = {}
        for profile in profiles:
            if isinstance(profile, ArmProfile):
                self.profiles[profile.robot_type] = profile
            else:
                raise TypeError(f"profiles must contain ArmProfile, got {profile!r}")
        if not self.profiles:
            raise ValueError("at least one ArmProfile is required")
        self.robots = tuple(self.profiles)
        self.axis_map = axis_map or AxisMap()
        self.map_mode = str(map_mode)
        self.robot_links = {}
        self.set_robot_links(robot_links)
        #: 「肘位置任务」：把肘也当目标交给 IK（只在 segments 口径下有意义，见 __init__）。
        self.elbow_task = bool(elbow_task)
        #: 肘目标的力度 0-1：1 = 整段上臂位移（肩最积极、腕残差最大），越小越像只喂腕。
        #: 两个位置目标在 PyBullet 里是等权的（没有优先级），所以这条是唯一的折中旋钮。
        if not 0.0 <= float(elbow_gain) <= 1.0:
            raise ValueError(f"elbow_gain must be within 0..1, got {elbow_gain!r}")
        self.elbow_gain = float(elbow_gain)
        #: 方向口径的力度 0-1（``map_mode="posture"``）：1 = 机器人方向完全对齐你的方向
        #: （默认），0.5 = 停在机器人中性方向与你方向的夹角正中，0 = 回中性。它不是位移
        #: 缩放，见 _slerp_direction。
        if not 0.0 <= float(posture_gain) <= 1.0:
            raise ValueError(f"posture_gain must be within 0..1, got {posture_gain!r}")
        self.posture_gain = float(posture_gain)
        if posture_frame is not None and posture_frame not in POSTURE_FRAMES:
            raise ValueError(
                f"posture_frame must be None (per-robot) or one of {POSTURE_FRAMES},"
                f" got {posture_frame!r}")
        self.posture_frame = None if posture_frame is None else str(posture_frame)
        self.robot_anchors = {}
        self.set_robot_anchors(anchors)
        self.calibration_frames = int(calibration_frames)
        #: 标定没攒够帧时的口径（``hold`` / ``track``，见 ``CALIBRATE_MODES``）。
        self.calibrate_mode = str(calibrate_mode)
        self.arm_length_reference = float(arm_length_reference)
        self.normalise_arm_length = bool(normalise_arm_length)
        self.arm_length_band = (float(arm_length_band[0]), float(arm_length_band[1]))
        self.orientation = orientation
        self.gains = {str(name): float(value) for name, value in (gains or {}).items()}
        self.max_travel_m = (None if max_travel_m is None else float(max_travel_m))
        if travel_mode not in TRAVEL_MODES:
            raise ValueError(f"travel_mode must be one of {TRAVEL_MODES}, got {travel_mode!r}")
        if float(lateral_guard_m) < 0.0:
            raise ValueError("lateral_guard_m must be >= 0 (0 turns the guard off)")
        if float(min_plausible_reach_m) <= 0.0:
            raise ValueError("min_plausible_reach_m must be > 0")
        self.travel_mode = str(travel_mode)
        self.lateral_guard_m = float(lateral_guard_m)
        self.gain_quality_guard = bool(gain_quality_guard)
        self.min_plausible_reach_m = float(min_plausible_reach_m)
        self.arm_reach_side_tolerance = float(arm_reach_side_tolerance)
        self.frames = 0
        self._samples = {side: [] for side in self.sides}
        self.neutral = {side: _neutral_default() for side in self.sides}
        self._last_state = {side: None for side in self.sides}
        #: 关键点跳变护栏的角速度上限（度/秒）：0 = 关（A/B 用）。见 ``_guard_direction``。
        self.max_direction_rate_deg_s = max(0.0, float(max_direction_rate_deg_s))
        #: 关键点跳变护栏用的「上一帧通过的方向」（每侧 ``(upper, fore)`` 米制向量）。
        self._last_direction = {}
        #: 每侧连续被拦的帧数（见 ``GLITCH_HOLD_FRAMES``）。
        self._glitch_streak = {}
        #: 这一帧的时长（秒）：护栏的角速度上限按它换算。见 _guard_direction。
        #: 构造时给一个值 = 这个调用方按固定帧率跑（``update(frame_dt=...)`` 仍可逐帧覆盖）。
        self.frame_dt = NOMINAL_FRAME_DT_S
        if frame_dt is not None:
            value = float(frame_dt)
            if value > 0.0:
                self.frame_dt = value
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},

                       "oriented": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides},
                       "guarded": {side: 0 for side in self.sides},
                       "segments": {side: 0 for side in self.sides},
                       "posture": {side: 0 for side in self.sides},
                       "elbow": {side: 0 for side in self.sides},
                       "direction_glitch": {side: 0 for side in self.sides},
                       "direction_glitch_kept": {side: 0 for side in self.sides},
                       "gain_guarded": {side: 0 for side in self.sides},
                       "held_reason": {side: {kind: 0 for kind in HOLD_REASONS}
                                       for side in self.sides}}
        self.clamped_by_robot = {name: {side: 0 for side in self.sides} for name in self.robots}
        if neutral is not None:
            self.load_neutral(neutral)

    # ------------------------------------------------------------------ #
    # 中性姿势（标定的替代品）：站好一次，记下「腕 - 肩」与臂长
    # ------------------------------------------------------------------ #
    def load_neutral(self, payload) -> None:
        """装入一份中性记录（``load_neutral()`` 的返回值或它的 ``sides`` 段）。"""
        sides = payload.get("sides", payload) if isinstance(payload, dict) else {}
        for side in self.sides:
            entry = sides.get(side) or {}
            offset = entry.get("offset")
            self.neutral[side] = {
                "offset": None if offset is None else np.asarray(offset, dtype=np.float64).reshape(3),
                "reach": None if entry.get("reach") is None else float(entry["reach"]),
                # 姿态基准：老记录里没有 -> None（这一侧退化成位置模式）。
                "orientation": _neutral_rotation(entry.get("orientation")),
                # 两段式映射的基准：老记录里没有 -> None（那一侧退回整段 gain）。
                "upper": _neutral_vector(entry.get("upper"), 3),
                "fore": _neutral_vector(entry.get("fore"), 3),
                "samples": int(entry.get("samples", 0)),
                "complete": bool(entry.get("complete", False)) and offset is not None,
            }

    def neutral_record(self) -> dict:
        """``{side: {offset, reach, upper, fore, samples, complete}}``（可直接 ``save_neutral``）。"""
        return {side: dict(entry) for side, entry in self.neutral.items()}

    # ------------------------------------------------------------------ #
    # 两段式映射（论文 scale_joints）：把「只动大臂」也变成目标位移
    # ------------------------------------------------------------------ #
    def set_robot_links(self, links) -> None:
        """Set per-robot, per-side upper/forearm lengths in metres."""
        self.robot_links = {}
        for robot_type, per_side in dict(links or {}).items():
            entries = {}
            for side, value in dict(per_side or {}).items():
                if not isinstance(value, dict):
                    continue
                upper = float(value.get("upper") or 0.0)
                fore = float(value.get("fore") or 0.0)
                if upper > 0.0 and fore > 0.0:
                    entries[str(side)] = {"upper": upper, "fore": fore}
            if entries:
                self.robot_links[str(robot_type)] = entries

    def set_robot_anchors(self, anchors) -> None:
        """Set per-robot shoulder/elbow/wrist neutral anchors in base-frame metres.

        Use the elbow parent-link origin as the rigid upper-arm shoulder anchor."""
        self.robot_anchors = {}
        for robot_type, per_side in dict(anchors or {}).items():
            entries = {}
            for side, value in dict(per_side or {}).items():
                if not isinstance(value, dict):
                    continue
                elbow = _neutral_vector(value.get("elbow"), 3)
                wrist = _neutral_vector(value.get("wrist"), 3)
                if elbow is not None and wrist is not None:
                    entry = {"elbow": elbow, "wrist": wrist}
                    shoulder = _neutral_vector(value.get("shoulder"), 3)
                    if shoulder is not None:
                        entry["shoulder"] = shoulder
                    entries[str(side)] = entry
            if entries:
                self.robot_anchors[str(robot_type)] = entries

    def _segment_scales(self, profile, side):
        """``(k_上臂, k_前臂)`` = 机器人连杆长 / 你的连杆长；缺数据就 ``None``。"""
        robot = self.robot_links.get(profile.robot_type, {}).get(side)
        entry = self.neutral[side]
        if not robot or entry.get("upper") is None or entry.get("fore") is None:
            return None
        human_upper = float(np.linalg.norm(entry["upper"]))
        human_fore = float(np.linalg.norm(entry["fore"]))
        if human_upper <= 1e-6 or human_fore <= 1e-6:
            return None
        return (float(robot["upper"]) / human_upper, float(robot["fore"]) / human_fore)

    def _segment_delta(self, profile, side, state):
        """Return k_upper * delta(upper) + k_fore * delta(forearm) in mapped metres."""
        terms = self._segment_terms(profile, side, state)
        return None if terms is None else terms[0] + terms[1]

    def _segment_terms(self, profile, side, state):
        """Return scaled upper/forearm vectors, or None if geometry is unavailable."""
        scales = self._segment_scales(profile, side)
        if scales is None:
            return None
        upper, fore = state.get("upper"), state.get("fore")
        entry = self.neutral[side]
        if upper is None or fore is None:
            return None
        k_upper, k_fore = scales
        return (k_upper * (np.asarray(upper, dtype=np.float64) - entry["upper"]),
                k_fore * (np.asarray(fore, dtype=np.float64) - entry["fore"]))

    # ------------------------------------------------------------------ #
    # 方向口径（posture）：搬方向不搬位移，长度用机器人自己的
    # ------------------------------------------------------------------ #
    def _posture_terms(self, profile, side, state):
        """``{你的上臂方向, 你的前臂方向, 机器人上臂方向, 机器人前臂方向, 两根连杆}``。

        缺任何一项返回 ``None``（缺什么由 ``_posture_unavailable`` 说明）。你的方向先过
        ``axis_map``（与位移口径同一条轴向约定），再与机器人自己的中性方向按 ``posture_gain``
        做夹角插值 —— ``gain=1`` 就是完全用你的方向。
        """
        anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
        if anchor is None or "shoulder" not in anchor:
            return None
        links = self.robot_links.get(profile.robot_type, {}).get(side)
        if not links:
            return None
        entry = self.neutral[side]
        u_op = _unit_direction(state.get("upper"))
        v_op = _unit_direction(state.get("fore"))
        u_op_rest = _unit_direction(entry.get("upper"))
        v_op_rest = _unit_direction(entry.get("fore"))
        if u_op is None or v_op is None or u_op_rest is None or v_op_rest is None:
            return None
        # rest_u 是**刚体**上臂方向（肩锚点 -> 肘）：gain<1 时它就是插值的基准方向，
        # 必须与 operator rest 对应同一个姿势（锚点口径见 _posture_target 的说明）。
        rest_u = _unit_direction(anchor["elbow"] - anchor["shoulder"])
        rest_v = _unit_direction(anchor["wrist"] - anchor["elbow"])
        u_base = _unit_direction(self.axis_map.apply(u_op))
        v_base = _unit_direction(self.axis_map.apply(v_op))
        if (rest_u is None or rest_v is None
                or u_base is None or v_base is None):
            return None
        frame = self.posture_frame_of(profile)
        if frame == "delta":
            u_base = self._turned_direction(u_op_rest, u_op, rest_u)
            v_base = self._turned_direction(v_op_rest, v_op, rest_v)
            if u_base is None or v_base is None:
                return None
        upper = float(links["upper"])
        fore = float(links["fore"])
        if not (np.isfinite(upper) and np.isfinite(fore)) or upper <= 0.0 or fore <= 0.0:
            return None
        return {
            "shoulder": np.asarray(anchor["shoulder"], dtype=np.float64),
            # 机器人这一侧最终要指的方向：gain=1 时是绝对档里的「你的方向本身」，或相对档里
            # 「本机自己的中性方向被你的旋转转过去之后」的那个方向。
            "u": _slerp_direction(rest_u, u_base, self.posture_gain),
            "v": _slerp_direction(rest_v, v_base, self.posture_gain),
            "u_rest": rest_u, "v_rest": rest_v,
            "u_op": u_op, "v_op": v_op,
            "u_op_rest": u_op_rest, "v_op_rest": v_op_rest,
            "upper": upper, "fore": fore,
            "frame": frame,
        }

    def posture_frame_of(self, profile) -> str:
        """Resolve the explicit posture frame override, falling back to the robot profile."""
        return str(self.posture_frame or profile.posture_frame)

    def _turned_direction(self, op_rest, op_now, robot_rest):
        """把「操作者从 ``op_rest`` 转到 ``op_now``」的那个旋转套在 ``robot_rest`` 上。

        三个输入都先过 ``axis_map``（前两个）或已经是基座系（第三个）；缺数据返回 ``None``，
        调用方据此退回绝对档（不是静默用半截数据）。
        """
        rotation = _rotation_between_directions(self.axis_map.apply(op_rest),
                                                self.axis_map.apply(op_now))
        if rotation is None:
            return None
        return _unit_direction(rotation @ np.asarray(robot_rest, dtype=np.float64))

    def _posture_unavailable(self, profile, side) -> str:
        """方向口径这一侧缺什么 —— 写进那一帧的 ``info``，别让下一个人猜。"""
        anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
        if anchor is None:
            return "no anchors measured for this robot/side"
        if "shoulder" not in anchor:
            return "no shoulder anchor measured for this robot/side"
        if not self.robot_links.get(profile.robot_type, {}).get(side):
            return "no robot link lengths measured for this robot/side"
        return "no usable operator arm directions in this frame"

    def _posture_target(self, profile, side, state):
        """Build elbow and wrist targets in the robot base frame.

        Elbow = shoulder + upper_length * upper_direction.
        Wrist = elbow + forearm_length * forearm_direction.
        The shoulder anchor must be rigid to the upper arm. Directions follow the
        selected mapped/delta frame; displacement travel clipping does not apply.
        Return None when required geometry is unavailable."""
        terms = self._posture_terms(profile, side, state)
        if terms is None:
            return None
        elbow_point = terms["shoulder"] + terms["upper"] * terms["u"]
        wrist_point = elbow_point + terms["fore"] * terms["v"]
        return elbow_point, wrist_point, terms

    @property
    def calibrating(self) -> bool:
        """还有哪一侧没攒够中性帧。"""
        return any(not self.neutral[side]["complete"] for side in self.sides)

    def calibration_progress(self) -> dict:
        """``{side: (已有帧数, 需要的帧数)}``。"""
        return {side: (len(self._samples[side]), self.calibration_frames)
                for side in self.sides}

    def status(self) -> str:
        """HUD 用的一行中性标定状态（``track`` 模式多一个前缀，表示已经在跟手）。"""
        if not self.calibrating:
            return "cal=ok"
        parts = " ".join(f"{side}={len(self._samples[side])}/{self.calibration_frames}"
                         for side in self.sides)
        if self.calibrate_mode == "track":
            return f"cal=track {parts}"
        return f"cal={parts}"


    # ------------------------------------------------------------------ #
    # 每帧入口
    # ------------------------------------------------------------------ #
    def update(self, arms_world=None, hands=None, frame_dt=None) -> dict:
        """Map one tracked body frame to {robot_type: {side: ArmTarget}}.

        Missing or rejected sides hold the previous target. Hands do not supply arm data."""
        if frame_dt is not None:
            try:
                value = float(frame_dt)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                self.frame_dt = value
        normalized, world = _hand_frames(hands)
        arms = _side_map(arms_world)
        body_metadata = getattr(arms_world, 'metadata', {}) or {}
        inputs = {}
        for side in self.sides:
            inputs[side] = self._frame_inputs(side, arms.get(side), normalized.get(side),
                                              world.get(side))
            if inputs[side]['arm'] is None:
                inputs[side]['reason'] = body_metadata.get('body_reason', {}).get(side) or inputs[side]['reason']
            self._collect_calibration(side, inputs[side])
        self.frames += 1
        self.counts["frames"] = self.frames
        states = {side: self._operator_state(side, inputs[side]) for side in self.sides}
        # Display the current hold without replacing the last valid control pose.
        self._trace_state = {side: {
            **state,
            'tracking_state': body_metadata.get('tracking_state', {}).get(side,
                'tracking' if state['valid'] else 'hold'),
            'fresh': bool(body_metadata.get('body_fresh', {}).get(side, state['valid'])),
            'age_s': body_metadata.get('body_age_s', {}).get(side),
            'sample_id': body_metadata.get('body_sample_ids', {}).get(side),
        } for side, state in states.items()}
        for side in self.sides:
            if states[side].get("orientation") is not None:
                self.counts["oriented"][side] += 1
        targets = {}
        for robot_type, profile in self.profiles.items():
            targets[robot_type] = {
                side: self._target_from_state(profile, side, states[side])
                for side in self.sides}
        return targets

    def _frame_inputs(self, side, arm, hand, world_hand) -> dict:
        """一帧一侧的原始输入：可用的 pose 手臂点 / 手部点，以及不可用时的原因。"""
        reason = ""
        ok = _is_points(arm, (3, 3))
        if ok:
            reach = arm_reach(arm)
            if reach < MIN_OPERATOR_REACH_M:
                ok = False
                reason = (f"SMPL-X arm reach {reach * 1000.0:.0f} mm is below "
                          f"{MIN_OPERATOR_REACH_M * 1000.0:.0f} mm; frame rejected")
        if not ok and not reason:
            reason = "no usable SMPL-X arm landmarks for this side"
        return {
            "side": side,
            "arm": np.asarray(arm, dtype=np.float64) if ok else None,
            "hand": np.asarray(hand, dtype=np.float64) if _is_points(hand, (21, 3)) else None,
            "world_hand": (np.asarray(world_hand, dtype=np.float64)
                           if _is_points(world_hand, (21, 3)) else None),
            "reason": reason,
        }

    def _collect_calibration(self, side, data) -> None:
        """Collect valid neutral samples until calibration_frames is reached.

        In track mode, use the available sample median while calibration converges."""
        entry = self.neutral[side]
        if entry["complete"] or data["arm"] is None:
            return
        points = data["arm"]
        self._samples[side].append((points[2] - points[0], arm_reach(points),
                                    wrist_orientation_frame(points),
                                    points[1] - points[0], points[2] - points[1]))
        self._refresh_neutral(side)

    def _refresh_neutral(self, side) -> None:
        """Update neutral geometry from sample medians and finalize a complete window."""
        samples = self._samples[side]
        if not samples:
            return
        enough = len(samples) >= self.calibration_frames
        if not enough and self.calibrate_mode != "track":
            return
        entry = self.neutral[side]
        offsets = np.stack([sample[0] for sample in samples], axis=0)
        reaches = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        uppers = np.stack([sample[3] for sample in samples], axis=0)
        fores = np.stack([sample[4] for sample in samples], axis=0)
        entry["offset"] = np.median(offsets, axis=0)
        entry["reach"] = float(np.median(reaches))
        entry["orientation"] = _average_rotation([sample[2] for sample in samples])
        entry["upper"] = np.median(uppers, axis=0)
        entry["fore"] = np.median(fores, axis=0)
        entry["samples"] = int(len(samples))
        entry["complete"] = bool(enough)

    def _guard_direction(self, side, upper, fore):
        """Hold implausibly fast direction changes, bounded by GLITCH_HOLD_FRAMES."""
        upper = np.asarray(upper, dtype=np.float64)
        fore = np.asarray(fore, dtype=np.float64)
        if self.max_direction_rate_deg_s <= 0.0:
            return upper, fore, 0.0
        previous = self._last_direction.get(side)
        if previous is None:
            self._last_direction[side] = (upper, fore)
            return upper, fore, 0.0
        step = 0.0
        for old, new in zip(previous, (upper, fore)):
            old_unit = _unit_direction(old)
            new_unit = _unit_direction(new)
            if old_unit is None or new_unit is None:
                continue
            step = max(step, _angle_between_deg(old_unit, new_unit))
        limit = self.max_direction_rate_deg_s * self.frame_dt
        if step > limit:
            streak = self._glitch_streak.get(side, 0) + 1
            if streak <= GLITCH_HOLD_FRAMES:
                self._glitch_streak[side] = streak
                self.counts["direction_glitch"][side] += 1
                return previous[0], previous[1], step
            # 连续拦满 -> 放过去（见 GLITCH_HOLD_FRAMES）
            self.counts["direction_glitch_kept"][side] += 1
        self._glitch_streak[side] = 0
        self._last_direction[side] = (upper, fore)
        return upper, fore, 0.0

    def _operator_state(self, side, data) -> dict:
        """一侧这一帧的「操作者层」结论：用哪条来源、位移多少、姿态、还是保持上一帧。"""
        entry = self.neutral[side]
        if data["arm"] is None:
            self._last_direction.pop(side, None)
            self._glitch_streak.pop(side, None)
        if self._has_neutral(side) and data["arm"] is not None:
            points = data["arm"]
            offset = points[2] - points[0]
            state = {
                "source": SOURCE_BODY,
                "valid": True,
                "held": False,
                "reason": "",
                "offset": offset,
                "delta_op": offset - entry["offset"],
                "upper": points[1] - points[0],
                "fore": points[2] - points[1],
                "reach": arm_reach(points),
                "elbow_bend_deg": elbow_bend_degrees(points[0], points[1], points[2]),
                "orientation": self._orientation(data),
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
            state["upper"], state["fore"], state["direction_glitch_deg"] = \
                self._guard_direction(side, state["upper"], state["fore"])
            self._last_state[side] = state
            self.counts["valid"][side] += 1
            return state
        previous = self._last_state[side]
        if previous is None:
            self.counts["held"][side] += 1
            self._count_hold(side, data)
            return {
                "source": SOURCE_NONE,
                "valid": False,
                "held": True,
                "reason": self._hold_reason(side, data),
                "offset": None,
                "delta_op": None,
                "upper": None,
                "fore": None,
                "reach": float("nan"),
                "elbow_bend_deg": float("nan"),
                "orientation": None,
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
        held = dict(previous)
        held.update({"valid": False, "held": True,
                     "reason": self._hold_reason(side, data)})
        self.counts["held"][side] += 1
        self._count_hold(side, data)
        return held

    def _has_neutral(self, side) -> bool:
        """这一侧现在有没有可用的中性基准。

        ``hold``：必须有攒够帧的完整中性（``complete``）—— 与改前逐字一致。
        ``track``：``_refresh_neutral`` 一发布临时中性（第 1 帧有效数据）就算有，
        所以「只要看见一次就开始跟手」。
        """
        entry = self.neutral[side]
        if entry["offset"] is None:
            return False
        return bool(entry["complete"]) or self.calibrate_mode == "track"

    def _count_hold(self, side, data) -> None:
        """Record the reason for holding this side in counts["held_reason"]."""
        reason = data.get("reason") or ""
        if not self._has_neutral(side):
            kind = "calibrating"
        elif data.get("arm") is not None:
            kind = "other"
        elif "SMPL-X arm reach" in reason:
            kind = "reach_below_min"
        elif reason:
            kind = "no_pose_landmarks"
        else:
            kind = "other"
        self.counts["held_reason"][side][kind] += 1

    def _hold_reason(self, side, data) -> str:
        """保持上一帧时解释为什么（标定中 / 没检到 / 帧被拒）。"""
        if not self._has_neutral(side):
            prefix = "tracking" if self.calibrate_mode == "track" else "calibrating"
            return (f"{prefix} {len(self._samples[side])}/{self.calibration_frames}"
                    f" ({data['side']})")
        return data["reason"] or "no arm data in this frame"


    # ------------------------------------------------------------------ #
    # 操作者层 -> 机器人层
    # ------------------------------------------------------------------ #
    def _target_from_state(self, profile, side, state) -> ArmTarget:
        """把「操作者层」结论换算成某一台机器人基座系里的目标。

        ``position`` 是基座系的点；``orientation`` 是**相对中性**的四元数（同一份目标给
        所有机器人，各自乘自己的中性 EE 朝向 —— 见 ``PyBulletArmIK.solve`` 的
        ``orientation_frame``）。"""
        gain = self._gain(profile, side, state)
        gain_reach, gain_source = self._gain_reference(side, state)
        trusted, trust_reason = self._arm_length_trust(side)
        if not trusted:
            # 这一帧用的是 metric_scale（臂长归一化被护栏关掉），如实写进报告：
            # 否则 gain_source 会说 "neutral"，看报告的人会以为归一化生效了。
            gain_source = "metric_scale"
            if self.neutral[side]["complete"]:
                # 标定没攒够帧的那几帧不计数：那时目标是中性工作点，归一化本来就不参与。
                self.counts["gain_guarded"][side] += 1
        neutral = self._neutral(profile, side)
        limit = self._travel_limit(profile, side)
        info = {
            "source": state["source"],
            "gain": round(gain, 4),
            "gain_reach_m": _round_or_none(gain_reach, 4),
            "gain_source": gain_source,
            "gain_trusted": bool(trusted),
            "gain_guard_reason": "" if trusted else trust_reason,
            "travel_limit_m": limit,
            "travel_mode": self.travel_mode,
            "lateral_guard_m": self.lateral_guard_m,
            "neutral_m": [round(float(value), 5) for value in neutral],
            "operator_reach_m": _round_or_none(state.get("reach"), 4),
            "elbow_bend_deg": _round_or_none(state.get("elbow_bend_deg"), 1),
        }
        if state["source"] == SOURCE_BODY and state["delta_op"] is not None:
            delta_op = state["delta_op"]
            info["map"] = self.map_mode
            if self.map_mode == "posture":
                # 方向口径：腕/肘目标都从机器人自己的肩锚点 + 自己的连杆长推出来，
                # 与「你移动了多少」无关（所以不夹行程、不需要 gain）。
                built = self._posture_target(profile, side, state)
                if built is None:
                    info["map"] = ("chain (posture unavailable: "
                                   + self._posture_unavailable(profile, side) + ")")
                else:
                    elbow_point, position, terms = built
                    self.counts["posture"][side] += 1
                    self.counts["elbow"][side] += 1
                    guarded = self._apply_lateral_guard(side, position)
                    if guarded:
                        self.counts["guarded"][side] += 1
                    info.update({
                        "posture_gain": self.posture_gain,
                        "posture_frame": terms["frame"],
                        "operator_direction_glitch_deg": _round_or_none(
                            state.get("direction_glitch_deg"), 1),
                        "operator_upper_deg": round(_angle_between_deg(
                            terms["u_op_rest"], terms["u_op"]), 2),
                        "operator_fore_deg": round(_angle_between_deg(
                            terms["v_op_rest"], terms["v_op"]), 2),
                        "robot_upper_deg": round(_angle_between_deg(
                            terms["u_rest"], terms["u"]), 2),
                        "robot_fore_deg": round(_angle_between_deg(
                            terms["v_rest"], terms["v"]), 2),
                        "posture_elbow_bend_deg": round(elbow_bend_degrees(
                            terms["shoulder"], elbow_point, position), 1),
                        "posture_reach_m": round(float(np.linalg.norm(
                            position - terms["shoulder"])), 4),
                        "posture_upper_m": round(float(terms["upper"]), 4),
                        "posture_fore_m": round(float(terms["fore"]), 4),
                        # 目标由机器人自己的肩锚点 + 自己的连杆长推出来，天然落在它自己的
                        # 可达球面上，所以逐轴行程上限不参与（也不该参与：g1 的每轴 40 mm
                        # 正是把大臂幅度压没的那个夹子）。这一帧如实留下"没夹"的痕迹。
                        "travel_limit_m": None,
                        "travel_limit_applied": False,
                        "clamped": False,
                        "elbow_target": "posture",
                    })
                    return ArmTarget(
                        side=side, position=tuple(float(value) for value in position),
                        elbow=tuple(float(value) for value in elbow_point),
                        orientation=self._orientation_for(profile, state),
                        valid=bool(state["valid"]), held=bool(state["held"]),
                        reason=state["reason"], source=SOURCE_BODY, info=info)
            if self.map_mode == "segments":
                mapped = self._segment_delta(profile, side, state)
                if mapped is not None:
                    delta_op = mapped
                    self.counts["segments"][side] += 1
                    info["segment_scales"] = [round(float(value), 4)
                                              for value in self._segment_scales(profile, side)]
                else:
                    info["map"] = "chain (segments unavailable)"
            # 肘位置任务：肘目标 = 机器人自己的中性肘点 + M(上臂段位移)。它与腕目标出自
            # 同一个和里的同一项，所以两个目标同时可达（见 _segment_terms）。逐轴夹到与腕
            # 相同的行程上限：不夹的话一次方向反转可以要出 2*上臂长 的位移。
            elbow_point = None
            if self.elbow_task:
                terms = self._segment_terms(profile, side, state)
                anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
                if terms is None:
                    info["elbow_task"] = "off (no link lengths / old neutral record)"
                elif anchor is None:
                    info["elbow_task"] = "off (no elbow anchor measured for this robot)"
                else:
                    elbow_delta = np.clip(
                        np.asarray(self.axis_map.apply(terms[0]), dtype=np.float64)
                        * self.elbow_gain,
                        -limit, limit)
                    elbow_point = tuple(float(value) for value in (
                        np.asarray(anchor["elbow"], dtype=np.float64) + elbow_delta))
                    info["elbow_task"] = "on"
                    info["elbow_gain"] = self.elbow_gain
                    info["elbow_delta_base_m"] = [round(float(value), 4)
                                                  for value in elbow_delta]
                    self.counts["elbow"][side] += 1
            info["delta_map_m"] = [round(float(value), 4) for value in delta_op]
            delta = np.asarray(gain * self.axis_map.apply(delta_op),
                               dtype=np.float64)
            request = float(np.max(np.abs(delta))) if delta.size else 0.0
            clamped = request > limit
            if clamped:
                self.counts["clamped"][side] += 1
                self.clamped_by_robot[profile.robot_type][side] += 1
            if clamped and self.travel_mode == "scale" and request > 0.0:
                delta = delta * (limit / request)
            else:
                delta = np.clip(delta, -limit, limit)
            position = np.asarray(neutral, dtype=np.float64) + delta
            guarded = self._apply_lateral_guard(side, position)
            if guarded:
                self.counts["guarded"][side] += 1
            info.update({
                "delta_op_m": [round(float(value), 4) for value in state["delta_op"]],
                "delta_base_m": [round(float(value), 4) for value in delta],
                "clamped": clamped,
                "request_m": round(request, 5),
                "guarded": guarded,
                "fallback_reason": state.get("fallback_reason", ""),
            })
            return ArmTarget(
                side=side, position=tuple(float(value) for value in position),
                elbow=elbow_point,
                orientation=self._orientation_for(profile, state),
                valid=bool(state["valid"]), held=bool(state["held"]),
                reason=state["reason"], source=SOURCE_BODY, info=info)
        # 还没有任何目标（标定中 / 一侧一直没数据）：停在 profile 的中性工作点，
        # 这样手臂不会因为没人管而垂下去（held=True 会把它写进报告）。
        info["fallback_reason"] = state.get("fallback_reason", "")
        return ArmTarget(
            side=side,
            position=tuple(float(value) for value in self._neutral(profile, side)),
            orientation=None, valid=False, held=True,
            reason=state["reason"] or "no target yet", source=SOURCE_NONE, info=info)

    def _orientation(self, data) -> tuple | None:
        """相对中性姿势的腕部姿态四元数（机器人无关；IK 那边再乘本机的中性 EE 朝向）。

        论文喂给 IK 的是「手腕 4x4 位姿」（SMPL-X 旋转链）；我们用手上的三点造一个腕
        坐标系再减掉中性那一帧（模块 docstring 有公式）。三个前提缺一不可，缺任何一个
        就返回 ``None`` —— 这一帧退回位置模式，不猜姿态：

        - 这一帧有 pose 手臂（兜底的手 21 点路径没有肘，给不出手臂平面）；
        - 中性记录里有姿态基准（``--neutral-json`` 是记了 ``orientation`` 的那一版）；
        - 三点不近共线（见 ``wrist_orientation_frame``）。

        返回的是 ``R_neutral^T @ R_now``；乘本机的中性 EE 朝向由
        ``PyBulletArmIK.solve(orientation_frame="neutral")`` 做，所以这里不需要知道是
        哪台机器人 —— 同一份目标三台各自乘自己的。
        """
        if self.orientation == "off":
            return None
        entry = self.neutral.get(data.get("side"))
        if entry is None or entry.get("orientation") is None:
            return None
        relative = relative_wrist_rotation(data.get("arm"), entry["orientation"])
        if relative is None:
            return None
        from teleoperation.retargeting.arm.rotations import (
            rotation_matrix_to_quaternion)

        return tuple(float(value) for value in rotation_matrix_to_quaternion(relative))

    def _orientation_wanted(self, profile) -> bool:
        """要不要带姿态约束：``auto`` = 只在 profile 声明 ``ik_mode="pose"`` 时带。"""
        if self.orientation == "off":
            return False
        if self.orientation == "on":
            return True
        return profile.ik_mode == "pose"

    def _orientation_for(self, profile, state):
        return state.get("orientation") if self._orientation_wanted(profile) else None

    def orientation_target(self) -> dict:
        """Return per-side neutral-relative elbow-plane quaternions or None.

        This numerical helper is independent of the active camera wrist route, which
        uses SMPL-X wrist FK through WristSceneController."""
        if self.orientation == "off":
            return {side: None for side in self.sides}
        return {side: (self._last_state.get(side) or {}).get("orientation")
                for side in self.sides}

    @staticmethod
    def _bend_of(upper, fore):
        """``(上臂向量, 前臂向量)`` -> 肘弯（度），口径与 ``elbow_bend_degrees`` 完全一致。

        直接用那个函数而不是另写一个夹角：报告里的 ``elbow_bend_deg`` 就是它算的，
        「你弯了多少」和「报告里写着你弯了多少」必须是同一个数（180 = 伸直，0 = 折起）。
        造三个点喂进去即可：肩 = 原点、肘 = 上臂向量、腕 = 上臂向量 + 前臂向量。
        """
        u = _unit_direction(upper)
        v = _unit_direction(fore)
        if u is None or v is None:
            return None
        origin = np.zeros(3, dtype=np.float64)
        return float(elbow_bend_degrees(origin, u, u + v))

    def elbow_bend_neutral(self) -> dict:
        """Return each side's neutral elbow angle in degrees, or None.

        Delta posture mapping uses this as its null-space reference baseline."""
        out = {}
        for side in self.sides:
            entry = self.neutral[side]
            out[side] = self._bend_of(entry.get("upper"), entry.get("fore"))
        return out

    def elbow_bend_target(self) -> dict:
        """Return each side's current elbow angle in degrees, or None.

        PyBulletArmIK converts the angle to a robot-specific elbow joint reference."""
        out = {}
        for side in self.sides:
            state = self._last_state.get(side) or {}
            out[side] = self._bend_of(state.get("upper"), state.get("fore"))
        return out

    def _gain(self, profile, side, state) -> float:
        """Return robot metric scale with optional bounded operator arm-length normalization."""
        gain = float(self.gains.get(profile.robot_type, profile.metric_scale))
        if not self.normalise_arm_length:
            return gain
        reach = self.neutral[side]["reach"]
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            reach = state.get("reach")
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            return gain
        if self.gain_quality_guard:
            trusted, _reason = self._arm_length_trust(side)
            if not trusted:
                return gain
        scale = self.arm_length_reference / float(reach)
        return gain * min(max(scale, self.arm_length_band[0]), self.arm_length_band[1])

    def _reach_plausible(self, reach) -> bool:
        """标定出来的「肩到腕」是不是一个像样的手臂长度（米）。"""
        if reach is None:
            return False
        number = float(reach)
        return bool(np.isfinite(number)) and number >= self.min_plausible_reach_m

    def _arm_length_trust(self, side) -> tuple:
        """Return whether calibrated arm lengths pass plausibility and side-symmetry checks."""
        reach = self.neutral[side]["reach"]
        if not self._reach_plausible(reach):
            shown = _round_or_none(reach, 4)
            if not self.neutral[side]["complete"]:
                # 标定还没攒够帧：这时目标就是中性工作点本身，归一化用不上，也没必要报警。
                return False, f"{side} calibrated arm reach {shown} m is not usable yet (calibrating)"
            return False, (f"{side} calibrated arm reach {shown} m is "
                           f"below {self.min_plausible_reach_m:.2f} m (pose landmarks bad)")
        others = [self.neutral[other]["reach"] for other in self.sides if other != side]
        others = [value for value in others if self._reach_plausible(value)]
        if others:
            values = [float(reach)] + [float(value) for value in others]
            low, high = min(values), max(values)
            if (high - low) > self.arm_reach_side_tolerance * high:
                return False, (f"left/right calibrated reaches differ by "
                               f"{100.0 * (high - low) / high:.0f}% "
                               f"(> {100.0 * self.arm_reach_side_tolerance:.0f}%): "
                               f"{low:.4f} / {high:.4f} m")
        return True, ""

    def _gain_reference(self, side, state) -> tuple:
        """``(用于 gain 的臂长, 来源)``；来源是 ``neutral`` / ``frame`` / ``none``。"""
        reach = self.neutral[side]["reach"]
        if reach is not None and np.isfinite(reach) and float(reach) > 0.0:
            return float(reach), "neutral"
        frame_reach = state.get("reach")
        if frame_reach is not None and np.isfinite(frame_reach) and float(frame_reach) > 0.0:
            return float(frame_reach), "frame"
        return None, "none"

    def _neutral(self, profile, side) -> tuple:
        """Return the side-specific base-frame neutral point, falling back to the profile."""
        method = getattr(profile, "realtime_neutral", None)
        if callable(method):
            return tuple(float(value) for value in method(side))
        return tuple(float(value) for value in profile.metric_offset)

    def _travel_limit(self, profile, side=None) -> float:
        """基座系里每个轴允许的最大位移（米）。"""
        if self.max_travel_m is not None:
            return float(self.max_travel_m)
        method = getattr(profile, "travel_limit", None)
        if side is not None and callable(method):
            return float(method(side))
        override = getattr(profile, "realtime_travel_m", None)
        if isinstance(override, dict):
            if side is not None and side in override:
                return float(override[side])
        elif override is not None:
            return float(override)
        return float(getattr(profile, "sweep_amplitude_m", DEFAULT_MAX_TRAVEL_M))

    def travel_limits(self) -> dict:
        """``{robot: {side: 米}}``：每台每侧的行程上限（HUD 与报告用）。"""
        return {name: {side: self._travel_limit(profile, side) for side in self.sides}
                for name, profile in self.profiles.items()}

    def _apply_lateral_guard(self, side, position) -> bool:
        """Clamp wrist target y to its own side in place; return whether it changed."""
        guard = self.lateral_guard_m
        if guard <= 0.0:
            return False
        sign = 1.0 if side == "left" else -1.0
        if float(position[1]) * sign >= guard:
            return False
        position[1] = sign * guard
        return True

    # ------------------------------------------------------------------ #
    # 报告 / HUD
    # ------------------------------------------------------------------ #
    def trace(self) -> dict:
        """Return operator-side diagnostics for the most recent update."""
        out = {}
        for side in self.sides:
            state = getattr(self, '_trace_state', self._last_state).get(side) or {}
            delta = state.get("delta_op")
            out[side] = {
                "source": state.get("source", SOURCE_NONE),
                "valid": bool(state.get("valid", False)),
                "held": bool(state.get("held", True)),
                "tracking_state": state.get('tracking_state', 'calibrating'),
                "fresh": bool(state.get('fresh', False)),
                "age_s": state.get('age_s'),
                "sample_id": state.get('sample_id'),
                "reason": state.get("reason", "no frame yet"),
                "delta_op_cm": None if delta is None else [round(float(v) * 100.0, 2)
                                                           for v in delta],
                "operator_reach_m": _round_or_none(state.get("reach"), 4),
                "elbow_bend_deg": _round_or_none(state.get("elbow_bend_deg"), 1),
            }
        return out

    def summary(self) -> dict:
        """报告段：轴向约定、每台的 gain/行程、中性记录、各来源的帧数。"""
        return {
            "axis_map": self.axis_map.describe(),
            "orientation": self.orientation,
            # 姿态目标的口径：相对中性（每台各自乘自己的中性 EE 朝向）。
            "orientation_frame": "neutral_relative",
            "normalise_arm_length": self.normalise_arm_length,
            "arm_length_reference_m": self.arm_length_reference,
            "calibrate_mode": self.calibrate_mode,
            "calibration_frames": self.calibration_frames,
            "calibration_progress": {side: list(value) for side, value
                                     in self.calibration_progress().items()},
            "travel_mode": self.travel_mode,
            "lateral_guard_m": self.lateral_guard_m,
            "gain_quality_guard": self.gain_quality_guard,
            "min_plausible_reach_m": self.min_plausible_reach_m,
            "arm_reach_side_tolerance": self.arm_reach_side_tolerance,
            "arm_length_trust": {side: {"trusted": trusted, "reason": reason}
                                 for side, (trusted, reason) in
                                 ((side, self._arm_length_trust(side))
                                  for side in self.sides)},
            "gain_by_side": {
                side: {name: round(self._gain(profile, side, {"reach": None}), 4)
                       for name, profile in self.profiles.items()}
                for side in self.sides},
            "travel_limit_m": self.travel_limits(),
            # 位移口径与两段式映射的两段比（k = 机器人连杆长 / 你的连杆长）。缺数据的那一侧
            # 会在每一帧的 info["map"] 里显示 "chain (segments unavailable)"。
            "map_mode": self.map_mode,
            # 方向口径的力度（map_mode="posture" 时生效）：1 = 机器人方向完全对齐你的方向。
            "posture_gain": self.posture_gain,
            "posture_frame": self.posture_frame or "profile",
            # 每台这一场真正生效的档 —— 读报告的人不必再去 profile 表里对照。
            "posture_frame_by_robot": {name: self.posture_frame_of(profile)
                                       for name, profile in self.profiles.items()},
            # 方向口径这一侧能不能跑（要肩锚点 + 机器人自己的连杆长）：跑不了时那一帧的
            # info["map"] 会写明 "chain (posture unavailable: ...)"。
            "posture_available": {
                name: {side: bool(
                    "shoulder" in self.robot_anchors.get(name, {}).get(side, {})
                    and self.robot_links.get(name, {}).get(side))
                    for side in self.sides}
                for name in self.profiles},
            "shoulder_anchors_m": {name: {side: [round(float(value), 5)
                                                 for value in entry["shoulder"]]
                                          for side, entry in per_side.items()
                                          if "shoulder" in entry}
                                   for name, per_side in self.robot_anchors.items()},
            "elbow_task": self.elbow_task,
            "elbow_gain": self.elbow_gain,
            "elbow_anchors_m": {name: {side: {"elbow": [round(float(value), 5)
                                                        for value in entry["elbow"]],
                                              "wrist": [round(float(value), 5)
                                                        for value in entry["wrist"]]}
                                     for side, entry in per_side.items()}
                                for name, per_side in self.robot_anchors.items()},
            "robot_links_m": {name: {side: {"upper": round(float(value["upper"]), 4),
                                            "fore": round(float(value["fore"]), 4)}
                                     for side, value in per_side.items()}
                              for name, per_side in self.robot_links.items()},
            "segment_scales": {name: {side: (None if self._segment_scales(profile, side) is None
                                             else [round(float(value), 4) for value in
                                                   self._segment_scales(profile, side)])
                                      for side in self.sides}
                               for name, profile in self.profiles.items()},
            "neutral_base_m": {name: {side: list(self._neutral(profile, side))
                                      for side in self.sides}
                               for name, profile in self.profiles.items()},
            "neutral": {side: {"offset": None if self.neutral[side]["offset"] is None
                               else [round(float(v), 5)
                                     for v in self.neutral[side]["offset"]],
                               "reach_m": _round_or_none(self.neutral[side]["reach"], 4),
                               # 记了姿态基准的侧面才能做姿态约束（老记录没有这一段）。
                               "orientation_recorded": self.neutral[side]["orientation"] is not None,
                               "orientation": (None if self.neutral[side]["orientation"] is None
                                               else [[round(float(v), 4) for v in row]
                                                     for row in
                                                     np.asarray(self.neutral[side]["orientation"],
                                                                dtype=np.float64)]),
                               "samples": self.neutral[side]["samples"],
                               "complete": self.neutral[side]["complete"]}
                        for side in self.sides},
            "elbow_bend_deg": {side: {
                "neutral": _round_or_none(
                    self._bend_of(self.neutral[side].get("upper"),
                                  self.neutral[side].get("fore")), 2),
                "current": _round_or_none(self.elbow_bend_target()[side], 2)}
                for side in self.sides},
            "clamped_by_robot": {name: dict(values) for name, values in self.clamped_by_robot.items()},
            "counts": {"frames": self.counts["frames"],
                       "valid": dict(self.counts["valid"]),
                       "held": dict(self.counts["held"]),
                       "oriented": dict(self.counts["oriented"]),
                       "clamped": dict(self.counts["clamped"]),
                       "guarded": dict(self.counts["guarded"]),
                       "segments": dict(self.counts["segments"]),
                       "posture": dict(self.counts["posture"]),
                       "elbow": dict(self.counts["elbow"]),
                       "direction_glitch": dict(self.counts["direction_glitch"]),
                       "direction_glitch_kept": dict(self.counts["direction_glitch_kept"]),
                       "gain_guarded": dict(self.counts["gain_guarded"]),
                       "held_reason": {side: dict(kinds) for side, kinds
                                       in self.counts["held_reason"].items()}},
            "last_trace": self.trace(),
        }

    def reset(self, recalibrate: bool = False) -> None:
        """清状态。``recalibrate=True`` 时连中性姿势一起清掉（要重新站一次）。"""
        self.frames = 0
        self._last_state = {side: None for side in self.sides}
        self._trace_state = {side: None for side in self.sides}
        self._last_direction = {}
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},

                       "oriented": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides},
                       "guarded": {side: 0 for side in self.sides},
                       "segments": {side: 0 for side in self.sides},
                       "posture": {side: 0 for side in self.sides},
                       "elbow": {side: 0 for side in self.sides},
                       "direction_glitch": {side: 0 for side in self.sides},
                       "direction_glitch_kept": {side: 0 for side in self.sides},
                       "gain_guarded": {side: 0 for side in self.sides},
                       "held_reason": {side: {kind: 0 for kind in HOLD_REASONS}
                                       for side in self.sides}}
        self.clamped_by_robot = {name: {side: 0 for side in self.sides} for name in self.robots}
        if recalibrate:
            self._samples = {side: [] for side in self.sides}
            self.neutral = {side: _neutral_default() for side in self.sides}


def _round_or_none(value, digits: int):
    """四舍五入；``None``/NaN/inf 一律返回 ``None``（JSON 里不出现 NaN）。"""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)
