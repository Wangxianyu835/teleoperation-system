"""L21 数据 -> 机器人【原装灵巧手】的降维映射（本项目正式采用的方案）

决策背景（2026-09-12）
---------------------
队友的重定向目标手是 **LinkerHand L21**（17 可动关节），
而论文里三台机器人出厂装的是**各自的手**：

    H1-2   -> Inspire          12 关节/手
    GR1-T2 -> Fourier 原生手    11 关节/手
    G1     -> 轻量三指手         7 关节/手

本项目**决定在仿真里使用机器人原装的手**（不换手），
因此需要把 18 维 L21 数据降维映射到原装手上。

映射原则：语义 1:1，不做无依据的「合并」
--------------------------------------
    L21 mcp_pitch  <->  原装手 proximal      （近端指节屈曲）
    L21 pip        <->  原装手 intermediate  （远端指节屈曲）
    L21 *_mcp_roll ->   丢弃（原装手没有侧摆自由度）
    拇指 cmc_yaw/pitch -> proximal_yaw/pitch
    拇指 mcp/ip        -> intermediate/distal（各机器人略有差异）

[注意] 这是**有损**映射，必须在报告里声明丢弃的自由度：
    H1-2   丢 5 个（4 个 *_mcp_roll + 拇指 cmc_roll）
    GR1-T2 丢 6 个（再丢拇指 1 个）
    G1     丢 10 个（另丢无名指/小指整根 + 拇指 1 个）

符号方向自动判定（关键，否则手指会反着弯）
----------------------------------------
    H1-2   [0.000, +1.700]  屈曲为正
    GR1-T2 [-1.570, 0.000]  屈曲为负
    G1     left 负 / right 正（左右手还不一样）
规则：sign = +1 if |hi| >= |lo| else -1
"""
import numpy as np

# 18 维数据的语义名（dim 0 是 hand_base_link 固定占位）
DIM_NAMES = {
    0: 'hand_base_link(占位)',
    1: 'index_mcp_roll', 2: 'index_mcp_pitch', 3: 'index_pip',
    4: 'middle_mcp_roll', 5: 'middle_mcp_pitch', 6: 'middle_pip',
    7: 'ring_mcp_roll', 8: 'ring_mcp_pitch', 9: 'ring_pip',
    10: 'pinky_mcp_roll', 11: 'pinky_mcp_pitch', 12: 'pinky_pip',
    13: 'thumb_cmc_roll', 14: 'thumb_cmc_yaw', 15: 'thumb_cmc_pitch',
    16: 'thumb_mcp', 17: 'thumb_ip',
}

# 原装手的关节名前缀
PREFIX = {
    'h1_2':   {'left': 'L_', 'right': 'R_'},
    'gr1_t2': {'left': 'L_', 'right': 'R_'},
    'g1':     {'left': 'left_hand_', 'right': 'right_hand_'},
}

# 语义名 -> 原装手关节名模板（{P} 替换成上面的前缀）
# 未列出的语义名 = 原装手没有这个自由度，只能丢弃
NATIVE_MAP = {
    'h1_2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_mcp':        '{P}thumb_intermediate_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'gr1_t2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'g1': {
        # G1 只有 食指 / 中指 / 拇指 三根
        'index_mcp_pitch':  '{P}index_0_joint',
        'index_pip':        '{P}index_1_joint',
        'middle_mcp_pitch': '{P}middle_0_joint',
        'middle_pip':       '{P}middle_1_joint',
        'thumb_cmc_yaw':    '{P}thumb_0_joint',
        'thumb_cmc_pitch':  '{P}thumb_1_joint',
        'thumb_mcp':        '{P}thumb_2_joint',
    },
}

# l21 真正可动的维度数（dim 1~17）
N_MOVABLE = 17

# L21 各维的关节限位（来自 l21 URDF，可用 robots/l21.py
# 里的 parse_joint_limits() 复核）。只有 --limit-mode rescale 会用到。
L21_LIMITS = {
    1: (-0.18, 0.18), 2: (0.0, 1.57), 3: (0.0, 1.57),
    4: (-0.18, 0.18), 5: (0.0, 1.57), 6: (0.0, 1.57),
    7: (-0.18, 0.18), 8: (0.0, 1.57), 9: (0.0, 1.57),
    10: (-0.18, 0.18), 11: (0.0, 1.57), 12: (0.0, 1.57),
    13: (-0.60, 0.60), 14: (0.0, 1.60), 15: (0.0, 1.00),
    16: (0.0, 1.57), 17: (0.0, 1.57),
}


def _landmark_array(points, dimensions: int):
    """Return a finite MediaPipe 21-point array, or ``None``."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return array[:, :dimensions]


def _hand_scale(points) -> float:
    """A stable hand-size denominator without trusting one projected palm edge."""
    spans = (
        np.linalg.norm(points[5] - points[17]),
        np.linalg.norm(points[0] - points[9]),
        np.linalg.norm(points[5] - points[9]),
        np.linalg.norm(points[9] - points[17]),
    )
    return float(max(spans))


def _pinch_measurement(points, *, allow_fist: bool):
    """Measure thumb/index/middle closure from one landmark coordinate space."""
    if points is None:
        return None
    scale = _hand_scale(points)
    if scale < 1e-6:
        return None

    # H1-2 and GR1-T2 cannot physically bring the thumb to the ring/pinky tips
    # inside the URDF limits. A ring/pinky pinch therefore falls back to the
    # nearer supported index/middle opposition instead of commanding an
    # unreachable "pinch" that visibly leaves the fingertips far apart.
    tips = {"index": 8, "middle": 12}
    distances = {
        name: float(np.linalg.norm(points[4] - points[slot]))
        for name, slot in tips.items()
    }
    finger = min(distances, key=distances.get)
    # MediaPipe alternates index/middle by a few millimetres during a real
    # pinch. Keep the target on the index unless the middle is clearly nearer.
    if distances["index"] <= distances[finger] + 0.10 * scale:
        finger = "index"
    ratio = distances[finger] / scale
    # An open MediaPipe hand commonly reports a thumb/index centre distance of
    # 0.6-0.8 palm scales. Linear closure from every non-zero distance made the
    # native hand sit half-closed all the time. Require a genuine approach, and
    # reach the exact native pinch endpoint at 0.30 scale or closer.
    pinch_trigger_ratio = 0.58
    strength = float(np.clip(
        (pinch_trigger_ratio - ratio) / 0.28, 0.0, 1.0))

    if allow_fist:
        # A fist closes four fingers even when the thumb rests across their
        # middle phalanges instead of touching the index tip.
        spans = np.array([
            np.linalg.norm(points[tip] - points[mcp]) / scale
            for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))
        ], dtype=np.float64)
        curl = np.clip((1.0 - spans) / 0.55, 0.0, 1.0)
        grasp_strength = float(np.median(curl))
        if grasp_strength >= 0.10 and grasp_strength >= strength:
            return "fist", grasp_strength, ratio
    if strength <= 0.0:
        return None, 0.0, ratio
    return finger, strength, ratio


def pinch_signal_details(world_hand, image_hand=None):
    """Return the selected pinch signal and both coordinate-space measurements.

    World landmarks give the geometrically correct answer while the hand is
    visible. When the thumb and finger touch, however, MediaPipe often occludes
    one tip and its world-space estimate can jump to a distance of half a palm
    width. The image-space normalized landmarks retain the visible contact much
    more reliably. Once valid image landmarks exist, they therefore arbitrate
    index/middle pinch (including "not pinching"); world data is retained as the
    fallback and its measurement is still reported for diagnosis.
    """
    world_points = _landmark_array(world_hand, 3)
    world = _pinch_measurement(world_points, allow_fist=True)
    image_points = _landmark_array(image_hand, 2)
    image = _pinch_measurement(image_points, allow_fist=False)
    if image is not None:
        selected = image
        source = "image" if image[0] is not None else "image-open"
    elif world is not None:
        selected = world
        source = "world" if world[0] is not None else "world-open"
    else:
        selected = (None, 0.0, None)
        source = "none"

    def describe(measurement):
        if measurement is None:
            return None
        return {
            "finger": measurement[0],
            "strength": round(float(measurement[1]), 3),
            "ratio": None if measurement[2] is None
            else round(float(measurement[2]), 3),
        }

    return {
        "selected": selected,
        "source": source,
        "world": describe(world),
        "image": describe(image),
    }


def pinch_signal(world_hand, image_hand=None):
    """MediaPipe landmarks -> ``(finger, strength, ratio)`` for hand closure."""
    return pinch_signal_details(world_hand, image_hand)["selected"]


class PinchSignalStabilizer:
    """Hold a confirmed closure for a few camera frames while landmarks flicker.

    At the 20 fps task-camera cadence, eight frames are roughly 0.4 seconds.
    The peak strength decays during a continuing weaker signal, so this smooths
    occlusion without permanently latching the hand closed.
    """

    def __init__(self, hold_frames: int = 8, decay: float = 0.90):
        self.hold_frames = max(0, int(hold_frames))
        self.decay = min(max(float(decay), 0.0), 1.0)
        self._held = None
        self._remaining = 0

    def update(self, signal):
        """Return a stabilized ``(finger, strength, ratio)`` signal."""
        finger, strength, ratio = signal
        strength = float(strength)
        if finger is not None and strength > 0.0:
            if self._held is not None and self._held[0] == finger:
                strength = max(strength, float(self._held[1]) * self.decay)
            safe_ratio = None if ratio is None else float(ratio)
            self._held = (str(finger), strength, safe_ratio)
            self._remaining = self.hold_frames
            return self._held
        if self._held is not None and self._remaining > 0:
            result = self._held
            self._remaining -= 1
            if self._remaining == 0:
                self._held = None
            return result
        self._held = None
        self._remaining = 0
        return None, 0.0, ratio


def _pinch_endpoint(robot_type, side: str, selected_finger: str, name: str,
                    low: float, high: float) -> float:
    """Native-hand pose that brings a thumb/finger pair together.

    "Largest absolute limit" is not a pinch for all three hands. The URDFs use
    different axes: on GR1-T2 the index proximal joint closes at 0 while its
    intermediate joint closes at the negative limit; on G1 the distal finger joint
    closes at 0 and the thumb opposition sign is mirrored between hands. Driving
    every joint to its farthest limit therefore moved some fingertips apart.
    """
    lower = name.lower()
    side = "left" if "left" in lower else "right" if "right" in lower else side
    if robot_type == "gr1_t2":
        values = {
            ("right", "index"): {
                "thumb_proximal_yaw": -0.87, "thumb_proximal_pitch": 1.22,
                "thumb_distal": 1.04, "finger_proximal": -0.785,
                "finger_intermediate": -1.74},
            ("right", "middle"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 0.61,
                "thumb_distal": 0.0, "finger_proximal": -1.57,
                "finger_intermediate": -0.87},
            ("left", "index"): {
                "thumb_proximal_yaw": -0.87, "thumb_proximal_pitch": 1.22,
                "thumb_distal": 0.52, "finger_proximal": -1.57,
                "finger_intermediate": -0.87},
            ("left", "middle"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 0.61,
                "thumb_distal": 1.04, "finger_proximal": -1.57,
                "finger_intermediate": -0.87},
            ("right", "fist"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 1.0,
                "thumb_distal": 1.04, "finger_proximal": -1.57,
                "finger_intermediate": -1.74},
            ("left", "fist"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 1.0,
                "thumb_distal": 1.04, "finger_proximal": -1.57,
                "finger_intermediate": -1.74},
            ("right", "grasp"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 1.0,
                "thumb_distal": 1.04, "finger_proximal": -1.57,
                "finger_intermediate": -1.74},
            ("left", "grasp"): {
                "thumb_proximal_yaw": -1.74, "thumb_proximal_pitch": 1.0,
                "thumb_distal": 1.04, "finger_proximal": -1.57,
                "finger_intermediate": -1.74},
        }.get((side, selected_finger), {})
        for role in ("thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_distal"):
            if role in lower and role in values:
                return min(max(values[role], low), high)
        if selected_finger in ("fist", "grasp") or selected_finger in lower:
            if "_proximal_joint" in lower:
                return min(max(values["finger_proximal"], low), high)
            if "_intermediate_joint" in lower:
                return min(max(values["finger_intermediate"], low), high)
    if robot_type == "g1":
        values = {
            ("right", "index"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": -1.0472,
                "thumb_2_joint": 0.0, "finger_0_joint": 1.5708,
                "finger_1_joint": 0.0},
            ("right", "middle"): {
                "thumb_0_joint": 1.0472, "thumb_1_joint": -1.0472,
                "thumb_2_joint": -0.873, "finger_0_joint": 1.5708,
                "finger_1_joint": 0.873},
            ("left", "index"): {
                "thumb_0_joint": -1.0472, "thumb_1_joint": 1.0472,
                "thumb_2_joint": 0.873, "finger_0_joint": -1.5708,
                "finger_1_joint": -0.873},
            ("left", "middle"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": 1.0472,
                "thumb_2_joint": 0.873, "finger_0_joint": -1.5708,
                "finger_1_joint": -0.873},
            ("right", "fist"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": -1.0472,
                "thumb_2_joint": 0.0, "finger_0_joint": 1.5708,
                "finger_1_joint": 1.7453},
            ("left", "fist"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": 1.0472,
                "thumb_2_joint": 0.0, "finger_0_joint": -1.5708,
                "finger_1_joint": -1.7453},
            ("right", "grasp"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": -1.0472,
                "thumb_2_joint": 0.0, "finger_0_joint": 1.5708,
                "finger_1_joint": 0.0},
            ("left", "grasp"): {
                "thumb_0_joint": 0.0, "thumb_1_joint": 1.0472,
                "thumb_2_joint": 0.0, "finger_0_joint": -1.5708,
                "finger_1_joint": 0.0},
        }.get((side, selected_finger), {})
        for role in ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint"):
            if role in lower and role in values:
                return min(max(values[role], low), high)
        if selected_finger in ("fist", "grasp") or selected_finger in lower:
            if "_0_joint" in lower:
                return min(max(values["finger_0_joint"], low), high)
            if "_1_joint" in lower:
                return min(max(values["finger_1_joint"], low), high)
    if robot_type == "h1_2":
        values = {
            ("right", "index"): {
                "thumb_proximal_yaw": 0.6, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.0, "thumb_distal": 0.6,
                "finger_proximal": 1.7, "finger_intermediate": 0.0},
            ("right", "middle"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.25,
                "thumb_intermediate": 0.0, "thumb_distal": 0.6,
                "finger_proximal": 1.7, "finger_intermediate": 0.0},
            ("left", "index"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.4, "thumb_distal": 1.2,
                "finger_proximal": 0.85, "finger_intermediate": 0.85},
            ("left", "middle"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.8, "thumb_distal": 1.2,
                "finger_proximal": 1.7, "finger_intermediate": 0.85},
            ("right", "fist"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.8, "thumb_distal": 1.2,
                "finger_proximal": 1.7, "finger_intermediate": 1.7},
            ("left", "fist"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.8, "thumb_distal": 1.2,
                "finger_proximal": 1.7, "finger_intermediate": 1.7},
            ("right", "grasp"): {
                "thumb_proximal_yaw": 0.6, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.0, "thumb_distal": 0.6,
                "finger_proximal": 1.7, "finger_intermediate": 0.0},
            ("left", "grasp"): {
                "thumb_proximal_yaw": 1.3, "thumb_proximal_pitch": 0.6,
                "thumb_intermediate": 0.4, "thumb_distal": 1.2,
                "finger_proximal": 1.7, "finger_intermediate": 0.0},
        }.get((side, selected_finger), {})
        for role in ("thumb_proximal_yaw", "thumb_proximal_pitch",
                     "thumb_intermediate", "thumb_distal"):
            if role in lower and role in values:
                return min(max(values[role], low), high)
        if selected_finger in ("fist", "grasp") or selected_finger in lower:
            if "_proximal_joint" in lower:
                return min(max(values["finger_proximal"], low), high)
            if "_intermediate_joint" in lower:
                return min(max(values["finger_intermediate"], low), high)
    return high if abs(high) >= abs(low) else low


def _make_pinch_closure(joints, mapping, finger: str, strength: float,
                        robot_type=None, side: str = "right") -> None:
    """Blend the native thumb and selected finger toward a model-specific pinch."""
    if not finger or strength <= 0.0:
        return
    keys = (("thumb", "index", "middle", "ring", "pinky")
            if finger in ("fist", "grasp") else ("thumb", finger))
    for _dim, name, _sign, low, high in mapping:
        lower_name = name.lower()
        if not any(key in lower_name for key in keys):
            continue
        endpoint = _pinch_endpoint(
            robot_type, side, finger, name, low, high)
        joints[name] += float(strength) * (endpoint - joints[name])


def build_mapping(robot_type, side, joint_ranges):
    """生成映射表：[(数据维度, 原装手关节名, 符号, 下限, 上限), ...]"""
    pref = PREFIX[robot_type][side]

    def sign_of(lo, hi):
        return 1.0 if abs(hi) >= abs(lo) else -1.0

    table = NATIVE_MAP[robot_type]
    out = []
    for dim, sem in DIM_NAMES.items():
        tpl = table.get(sem)
        if tpl is None:
            continue                      # 原装手没有这个自由度
        jn = tpl.format(P=pref)
        if jn not in joint_ranges:
            continue
        lo, hi = joint_ranges[jn]
        out.append((dim, jn, sign_of(lo, hi), lo, hi))
    return out


def dropped_dims(robot_type, side, mapping):
    """返回被丢弃的数据维度（原装手没有对应自由度）"""
    del side
    used = {d for d, *_ in mapping}
    return [d for d in DIM_NAMES if d != 0 and d not in used]


def map_dofs(row, mapping, limit_mode='clamp', pinch=None,
             robot_type=None, side='right'):
    """把一帧 17DOF 数据映射成 {关节名: 目标角度}（已按符号与限位处理）

    Args:
        row: (17,) 一帧 L21 可动关节角
        mapping: build_mapping() 的输出
        limit_mode:
          'clamp'   （默认，忠实映射）
                    超出原装手限位的值直接【截断】。语义准确，
                    但数据超出原装手行程时手指会"卡"在限位上。
          'rescale' （视觉辅助，不是真实映射）
                    用 L21 的限位把比例线性映射到原装手的限位：
                    保运动形状、幅度等比缩放，不会卡住。
                    代价：改变了角度语义，报告里必须声明。
                    ⚠️ 实测触发原因：L21 拇指 cmc_pitch 行程 0~1.0，
                       而 H1-2 原装手 thumb_proximal_pitch 只有 -0.1~0.6
                       -> 右撇子有 50% 的帧被截断。

    Returns:
        (joints: dict, n_clipped: int)
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (17,): raise ValueError("Native hand mapping requires 17DOF without placeholder")
    joints, clipped = {}, 0
    for d, jn, sg, lo, hi in mapping:
        v = sg * float(row[d - 1])
        if limit_mode == 'rescale' and d in L21_LIMITS:
            l21lo, l21hi = L21_LIMITS[d]
            span = l21hi - l21lo
            v = lo + (v - l21lo) / span * (hi - lo) if span > 1e-9 else lo
            v = min(max(v, lo), hi)
        else:
            if v < lo or v > hi:
                clipped += 1
            v = min(max(v, lo), hi)
        joints[jn] = v
    if pinch is not None:
        finger, strength, _ratio = pinch
        _make_pinch_closure(joints, mapping, finger, strength, robot_type, side)
    return joints, clipped


def coverage(robot_type, side, mapping):
    """覆盖率（可表达自由度 / 17）"""
    del side
    return len(mapping) / float(N_MOVABLE)

from teleoperation.contracts.hand import L21HandDOFs


class NativeHandAdapter:
    def __init__(self, robot_type, side, joint_ranges, limit_mode="clamp"):
        if robot_type not in NATIVE_MAP or side not in ("left", "right"): raise ValueError("Unknown robot or hand side")
        self.robot_type, self.side = robot_type, side
        self.mapping = build_mapping(robot_type, side, joint_ranges)
        self.limit_mode = limit_mode

    def map(self, dofs: L21HandDOFs, pinch=None):
        if not isinstance(dofs, L21HandDOFs): raise TypeError("NativeHandAdapter requires L21HandDOFs[17]")
        return map_dofs(dofs.values, self.mapping, self.limit_mode, pinch=pinch,
                        robot_type=self.robot_type, side=self.side)
