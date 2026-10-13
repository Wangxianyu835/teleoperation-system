"""Camera arm landmarks (metres) -> per-robot arm targets: relative retargeting.

为什么是「相对」而不是标定
--------------------------
论文链路要 SMPLer-X 的手腕 6D 位姿 + 手眼标定 + PINK 全身 IK；本机没有相机内参/外参，
所以这里做的是**相对重定向**：以一次「中性姿势」为原点，把操作者手腕相对肩的位移按
比例搬进机器人的基座系。

    d_op   = (W - S) - (W0 - S0)            # 你移动了多少（操作者系，米）
    d_base = gain * M(d_op)                 # M = 轴向置换 + 符号（约定，可覆盖）
    target = profile.metric_offset + d_base # 机器人基座系，夹在 profile 的行程内

好处：不需要手眼标定 —— 人整体前后左右挪动时肩和腕一起动，d_op 不变，机器人不会
被你的站位带跑。坏处同样如实说：**这不是论文数值**，gain / 轴向 / 行程都是本机调参，
``docs/multi-robot-arm.md`` 第 7 节记了它们是怎么量出来的。

数据从哪来（同一帧，来自 ``inputs/mediapipe.py`` 的 ``next_teleop_observation``）
-------------------------------------------------------------------------------
``upper_body.world_arms[side]`` = (肩, 肘, 腕) 3x3、米制、原点=髋中点 -> **位置**用它；
``world_hands[side]`` / ``hands[side]`` = 手 21 点 -> **姿态**用它（掌面基只用方向，
与平移原点无关：手部 world 的原点是「手的几何中心」，不能当位置用）。

退化输入（该侧没检到 / 非有限 / 掌面基不可张成）一律不猜：保持上一帧目标，并在
``ArmTarget.reason`` 里写明原因。

轴向约定（默认值，不是标定）
--------------------------
操作者系（MediaPipe pose world 口径）：x 图像向右、y 向下、z 远离相机为正。
机器人基座系（URDF 惯例）：X 前、Y 左、Z 上。

    robot X <- -op_z     手远离相机 = 远离你 = 机器人往前伸
    robot Y <- +op_x     图像向右 = 你的左侧 = 机器人 +Y
    robot Z <- -op_y     图像向下 = 机器人向下

相机在你正前方、机器人也朝着你时这组符号自洽；相机挪到侧面或机器人背对你时必须改
（``--arm-axis-order`` / ``--arm-axis-signs``）。真值要在你坐到位后实测确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import acos, degrees
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_profiles import SIDES, ArmProfile

AXIS_NAMES = ("X", "Y", "Z")
#: 默认轴向映射，见模块 docstring（约定，不是标定）。
DEFAULT_AXIS_ORDER = ("Z", "X", "Y")
DEFAULT_AXIS_SIGNS = (-1.0, 1.0, -1.0)
#: 手臂长度归一化的参考值（米）：高于/矮于这个臂长的操作者由 gain 自动补偿。
REFERENCE_REACH_M = 0.60
#: 归一化系数的允许区间，防止一次坏标定把 gain 推到离谱的值。
ARM_LENGTH_BAND = (0.5, 2.0)
#: 比这更短的「肩到腕」认为 Pose 这帧不可信（正常人手臂 >= 0.3 m）。
MIN_OPERATOR_REACH_M = 0.15
#: 基座系里的默认行程上限（米，每轴），profile 没有 realtime_travel_m 时用它。
DEFAULT_MAX_TRAVEL_M = 0.10
#: 没有 pose 手臂时回落到「手 21 点 -> 手腕目标」的映射比例（沿用 wrist_targets 默认）。
FALLBACK_IMAGE_SCALE = 0.7
SOURCE_POSE = "pose"
SOURCE_HAND = "hand"
SOURCE_NONE = "none"


def _is_points(value, shape) -> bool:
    """``True`` 表示这是一组给定形状、全有限、可用的点。"""
    if value is None:
        return False
    array = np.asarray(value, dtype=np.float64)
    return array.shape == tuple(shape) and bool(np.isfinite(array).all())


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

    def with_defaults(self, order=None, signs=None) -> "AxisMap":
        """用可选覆盖值得到一个新的映射（``None`` = 保持原值）。"""
        return AxisMap(tuple(order) if order else self.order,
                       tuple(signs) if signs else self.signs)


@dataclass(frozen=True)
class ArmTarget:
    """一台机器人的一侧、某一帧的手臂目标（机器人基座系）。

    ``valid=True`` 表示这一帧真的解算出了新目标；``held=True`` 表示沿用上一帧（该侧
    没数据 / 还在标定 / 目标非法）。``source`` 说明用的是 pose 手臂还是手部兜底映射，
    ``info`` 里是可检查的诊断量（你移动了多少、肘弯了多少、有没有被行程夹住）。
    """

    side: str
    position: tuple | None = None
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
            "orientation": None if self.orientation is None else [round(float(v), 6) for v in self.orientation],
            "valid": bool(self.valid),
            "held": bool(self.held),
            "reason": self.reason,
            "source": self.source,
            "info": self.info,
        }


def _neutral_default() -> dict:
    return {"offset": None, "reach": None, "samples": 0, "complete": False}


def load_neutral(path) -> dict:
    """读一份 ``--save-neutral`` 写出（或 ``--neutral-json`` 指定）的中性姿势记录。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    neutral = {}
    for side in SIDES:
        entry = payload.get("sides", {}).get(side, {})
        offset = entry.get("offset")
        neutral[side] = {
            "offset": None if offset is None else np.asarray(offset, dtype=np.float64),
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
            "samples": int(entry.get("samples", 0)),
            "complete": bool(entry.get("complete", False)) and offset is not None,
        }
    return {"sides": neutral, "axis_map": payload.get("axis_map"),
            "created": payload.get("created"), "path": str(path)}


def save_neutral(path, sides: dict, extra: dict | None = None) -> Path:
    """把中性姿势写进 JSON，方便下次直接复用（不用再站一次）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "realtime_arm_neutral_v1", "extra": extra or {}, "sides": {}}
    for side, entry in sides.items():
        offset = entry.get("offset")
        payload["sides"][side] = {
            "offset": None if offset is None else [float(v) for v in np.asarray(offset).reshape(3)],
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
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
        return dict(value.world_arms or {})
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
                 arm_length_reference: float = REFERENCE_REACH_M,
                 normalise_arm_length: bool = True,
                 arm_length_band=ARM_LENGTH_BAND,
                 orientation: str = "auto",
                 gains=None, max_travel_m=None, neutral=None,
                 fallback_hand: bool = True,
                 fallback_scale: float = FALLBACK_IMAGE_SCALE):
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if int(calibration_frames) < 1:
            raise ValueError("calibration_frames must be >= 1")
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
        self.calibration_frames = int(calibration_frames)
        self.arm_length_reference = float(arm_length_reference)
        self.normalise_arm_length = bool(normalise_arm_length)
        self.arm_length_band = (float(arm_length_band[0]), float(arm_length_band[1]))
        self.orientation = orientation
        self.gains = {str(name): float(value) for name, value in (gains or {}).items()}
        self.max_travel_m = (None if max_travel_m is None else float(max_travel_m))
        self.fallback_hand = bool(fallback_hand)
        self.fallback_scale = float(fallback_scale)
        self.frames = 0
        self._samples = {side: [] for side in self.sides}
        self.neutral = {side: _neutral_default() for side in self.sides}
        self._last_state = {side: None for side in self.sides}
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},
                       "fallback": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides}}
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
                "samples": int(entry.get("samples", 0)),
                "complete": bool(entry.get("complete", False)) and offset is not None,
            }

    def neutral_record(self) -> dict:
        """``{side: {offset, reach, samples, complete}}``（可直接 ``save_neutral``）。"""
        return {side: dict(entry) for side, entry in self.neutral.items()}

    @property
    def calibrating(self) -> bool:
        """还有哪一侧没攒够中性帧。"""
        return any(not self.neutral[side]["complete"] for side in self.sides)

    def calibration_progress(self) -> dict:
        """``{side: (已有帧数, 需要的帧数)}``。"""
        return {side: (len(self._samples[side]), self.calibration_frames)
                for side in self.sides}

    def status(self) -> str:
        """HUD 用的一行中性标定状态。"""
        if not self.calibrating:
            return "cal=ok"
        parts = " ".join(f"{side}={len(self._samples[side])}/{self.calibration_frames}"
                         for side in self.sides)
        return f"cal={parts}"


    # ------------------------------------------------------------------ #
    # 每帧入口
    # ------------------------------------------------------------------ #
    def update(self, arms_world=None, hands=None) -> dict:
        """一帧摄像头数据 -> ``{robot_type: {side: ArmTarget}}``。

        ``arms_world`` 是 Pose 的 (肩, 肘, 腕) 米制数据（``UpperBodyObservation`` 或它的
        ``world_arms`` 字典），``hands`` 是同一帧的手部关键点（``RawHandFrame`` 或
        ``{side: normalized, world_side: world}`` 字典）。两者都可以是 ``None``：某一侧
        没数据时该侧保持上一帧目标，不会拿瞎猜的位置去解 IK。
        """
        normalized, world = _hand_frames(hands)
        arms = _side_map(arms_world)
        inputs = {}
        for side in self.sides:
            inputs[side] = self._frame_inputs(side, arms.get(side), normalized.get(side),
                                              world.get(side))
            self._collect_calibration(side, inputs[side])
        self.frames += 1
        self.counts["frames"] = self.frames
        states = {side: self._operator_state(side, inputs[side]) for side in self.sides}
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
                reason = (f"pose arm reach {reach * 1000.0:.0f} mm is below "
                          f"{MIN_OPERATOR_REACH_M * 1000.0:.0f} mm; frame rejected")
        if not ok and not reason:
            reason = "no usable pose arm landmarks for this side"
        return {
            "side": side,
            "arm": np.asarray(arm, dtype=np.float64) if ok else None,
            "hand": np.asarray(hand, dtype=np.float64) if _is_points(hand, (21, 3)) else None,
            "world_hand": (np.asarray(world_hand, dtype=np.float64)
                           if _is_points(world_hand, (21, 3)) else None),
            "reason": reason,
        }

    def _collect_calibration(self, side, data) -> None:
        """中性姿势采样：只取有效帧，攒够 ``calibration_frames`` 帧后取中位数。"""
        entry = self.neutral[side]
        if entry["complete"] or data["arm"] is None:
            return
        points = data["arm"]
        self._samples[side].append((points[2] - points[0], arm_reach(points)))
        if len(self._samples[side]) < self.calibration_frames:
            return
        offsets = np.stack([sample[0] for sample in self._samples[side]], axis=0)
        reaches = np.asarray([sample[1] for sample in self._samples[side]],
                             dtype=np.float64)
        entry["offset"] = np.median(offsets, axis=0)
        entry["reach"] = float(np.median(reaches))
        entry["samples"] = int(len(self._samples[side]))
        entry["complete"] = True

    def _operator_state(self, side, data) -> dict:
        """一侧这一帧的「操作者层」结论：用哪条来源、位移多少、姿态、还是保持上一帧。"""
        entry = self.neutral[side]
        if entry["complete"] and data["arm"] is not None:
            points = data["arm"]
            offset = points[2] - points[0]
            state = {
                "source": SOURCE_POSE,
                "valid": True,
                "held": False,
                "reason": "",
                "offset": offset,
                "delta_op": offset - entry["offset"],
                "reach": arm_reach(points),
                "elbow_bend_deg": elbow_bend_degrees(points[0], points[1], points[2]),
                "orientation": self._orientation(data),
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
            self._last_state[side] = state
            self.counts["valid"][side] += 1
            return state
        if data["arm"] is None and self.fallback_hand and data["hand"] is not None:
            state = {
                "source": SOURCE_HAND,
                "valid": True,
                "held": False,
                "reason": "",
                "offset": None,
                "delta_op": None,
                "reach": float("nan"),
                "elbow_bend_deg": float("nan"),
                "orientation": self._orientation(data),
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
            self._last_state[side] = state
            self.counts["valid"][side] += 1
            self.counts["fallback"][side] += 1
            return state
        previous = self._last_state[side]
        if previous is None:
            self.counts["held"][side] += 1
            return {
                "source": SOURCE_NONE,
                "valid": False,
                "held": True,
                "reason": self._hold_reason(side, data),
                "offset": None,
                "delta_op": None,
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
        return held

    def _hold_reason(self, side, data) -> str:
        """保持上一帧时解释为什么（标定中 / 没检到 / 帧被拒）。"""
        entry = self.neutral[side]
        if not entry["complete"]:
            return (f"calibrating {len(self._samples[side])}/{self.calibration_frames}"
                    f" ({data['side']})")
        return data["reason"] or "no arm data in this frame"


    # ------------------------------------------------------------------ #
    # 操作者层 -> 机器人层
    # ------------------------------------------------------------------ #
    def _target_from_state(self, profile, side, state) -> ArmTarget:
        """把「操作者层」结论换算成某一台机器人基座系里的目标。"""
        gain = self._gain(profile, side, state)
        gain_reach, gain_source = self._gain_reference(side, state)
        neutral = self._neutral(profile, side)
        limit = self._travel_limit(profile, side)
        info = {
            "source": state["source"],
            "gain": round(gain, 4),
            "gain_reach_m": _round_or_none(gain_reach, 4),
            "gain_source": gain_source,
            "travel_limit_m": limit,
            "neutral_m": [round(float(value), 5) for value in neutral],
            "operator_reach_m": _round_or_none(state.get("reach"), 4),
            "elbow_bend_deg": _round_or_none(state.get("elbow_bend_deg"), 1),
        }
        if state["source"] == SOURCE_HAND and state["hand"] is not None:
            return self._fallback_target(profile, side, state, info)
        if state["source"] == SOURCE_POSE and state["delta_op"] is not None:
            delta = gain * self.axis_map.apply(state["delta_op"])
            clamped = bool(np.any(np.abs(delta) > limit))
            if clamped:
                self.counts["clamped"][side] += 1
                self.clamped_by_robot[profile.robot_type][side] += 1
            delta = np.clip(delta, -limit, limit)
            position = np.asarray(neutral, dtype=np.float64) + delta
            info.update({
                "delta_op_m": [round(float(value), 4) for value in state["delta_op"]],
                "delta_base_m": [round(float(value), 4) for value in delta],
                "clamped": clamped,
                "fallback_reason": state.get("fallback_reason", ""),
            })
            return ArmTarget(
                side=side, position=tuple(float(value) for value in position),
                orientation=self._orientation_for(profile, state),
                valid=bool(state["valid"]), held=bool(state["held"]),
                reason=state["reason"], source=SOURCE_POSE, info=info)
        # 还没有任何目标（标定中 / 一侧一直没数据）：停在 profile 的中性工作点，
        # 这样手臂不会因为没人管而垂下去（held=True 会把它写进报告）。
        info["fallback_reason"] = state.get("fallback_reason", "")
        return ArmTarget(
            side=side,
            position=tuple(float(value) for value in self._neutral(profile, side)),
            orientation=None, valid=False, held=True,
            reason=state["reason"] or "no target yet", source=SOURCE_NONE, info=info)

    def _fallback_target(self, profile, side, state, info) -> ArmTarget:
        """pose 手臂这帧不可信时的兜底：沿用既有「21 点 -> 手腕目标」映射。

        它输出**已经在机器人基座系**里的位置（``base_offset`` 直接取 profile 的
        ``metric_offset``），所以调用方要用 ``source="base"``，不能再叠一次 profile 缩放。
        """
        from teleoperation.retargeting.arm.wrist_targets import (
            WristMapping, wrist_pose_from_landmarks)

        mapping = WristMapping(scale=self.fallback_scale,
                               base_offset=self._neutral(profile, side))
        target = wrist_pose_from_landmarks(state["hand"], side, mapping)
        info["fallback_scale"] = self.fallback_scale
        info["fallback_reason"] = state.get("fallback_reason", "")
        if not target.valid:
            info["fallback_reason"] = target.reason or "hand landmarks unusable"
            return ArmTarget(
                side=side,
                position=tuple(float(value) for value in self._neutral(profile, side)),
                orientation=None, valid=False, held=True,
                reason=info["fallback_reason"], source=SOURCE_NONE, info=info)
        orientation = None
        if self._orientation_wanted(profile):
            orientation = tuple(float(value) for value in target.quaternion)
        return ArmTarget(
            side=side, position=tuple(float(value) for value in target.position),
            orientation=orientation, valid=True, held=False, reason="",
            source=SOURCE_HAND, info=info)

    def _orientation(self, data) -> tuple | None:
        """掌面基 -> 末端四元数（只用到方向，因此 world / 归一化两套点都能用）。"""
        if self.orientation == "off":
            return None
        from teleoperation.retargeting.arm.wrist_targets import (
            WristMapping, palm_frame_from_landmarks, robot_orientation,
            rotation_matrix_to_quaternion)

        points = data["world_hand"] if data["world_hand"] is not None else data["hand"]
        if points is None:
            return None
        frame = palm_frame_from_landmarks(points)
        if frame is None:
            return None
        matrix = robot_orientation(frame, WristMapping())
        return tuple(float(value) for value in rotation_matrix_to_quaternion(matrix))

    def _orientation_wanted(self, profile) -> bool:
        """要不要带姿态约束：``auto`` = 只在 profile 声明 ``ik_mode="pose"`` 时带。"""
        if self.orientation == "off":
            return False
        if self.orientation == "on":
            return True
        return profile.ik_mode == "pose"

    def _orientation_for(self, profile, state):
        return state.get("orientation") if self._orientation_wanted(profile) else None

    def _gain(self, profile, side, state) -> float:
        """米/米。``profile.metric_scale`` 是本机调参，再按你标定时的臂长归一化。

        臂长固定取**中性标定**那一侧的值（一个常数），不用当帧的肩-肘-腕长度 —— 否则
        肘一弯 gain 就漂，动作幅度会跟着姿势变。
        """
        gain = float(self.gains.get(profile.robot_type, profile.metric_scale))
        if not self.normalise_arm_length:
            return gain
        reach = self.neutral[side]["reach"]
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            reach = state.get("reach")
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            return gain
        scale = self.arm_length_reference / float(reach)
        return gain * min(max(scale, self.arm_length_band[0]), self.arm_length_band[1])

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
        """这一侧的中性工作点（基座系，米）：优先用 profile 的实测每侧值。"""
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

    # ------------------------------------------------------------------ #
    # 报告 / HUD
    # ------------------------------------------------------------------ #
    def trace(self) -> dict:
        """最近一次更新的操作者侧诊断量（``--arm-trace`` 打印它）。

        ``delta_op_cm`` 是「腕相对肩」相对中性姿势的位移，单位厘米。你往某个方向动手，
        只有对应分量明显变化 —— 这正是用来核对轴向约定（哪个符号对不对）的现场证据。
        """
        out = {}
        for side in self.sides:
            state = self._last_state.get(side) or {}
            delta = state.get("delta_op")
            out[side] = {
                "source": state.get("source", SOURCE_NONE),
                "valid": bool(state.get("valid", False)),
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
            "normalise_arm_length": self.normalise_arm_length,
            "arm_length_reference_m": self.arm_length_reference,
            "gain_by_side": {
                side: {name: round(self._gain(profile, side, {"reach": None}), 4)
                       for name, profile in self.profiles.items()}
                for side in self.sides},
            "travel_limit_m": self.travel_limits(),
            "neutral_base_m": {name: {side: list(self._neutral(profile, side))
                                      for side in self.sides}
                               for name, profile in self.profiles.items()},
            "neutral": {side: {"offset": None if self.neutral[side]["offset"] is None
                               else [round(float(v), 5)
                                     for v in self.neutral[side]["offset"]],
                               "reach_m": _round_or_none(self.neutral[side]["reach"], 4),
                               "samples": self.neutral[side]["samples"],
                               "complete": self.neutral[side]["complete"]}
                        for side in self.sides},
            "counts": {"frames": self.counts["frames"],
                       "valid": dict(self.counts["valid"]),
                       "held": dict(self.counts["held"]),
                       "fallback": dict(self.counts["fallback"]),
                       "clamped": dict(self.counts["clamped"])},
            "clamped_by_robot": {name: dict(counts)
                                 for name, counts in self.clamped_by_robot.items()},
            "last_trace": self.trace(),
        }

    def reset(self, recalibrate: bool = False) -> None:
        """清状态。``recalibrate=True`` 时连中性姿势一起清掉（要重新站一次）。"""
        self.frames = 0
        self._last_state = {side: None for side in self.sides}
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},
                       "fallback": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides}}
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

