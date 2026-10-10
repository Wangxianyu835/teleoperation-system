"""解析式（零权重）手部重定向：手掌局部 25 点 -> L21 18 维角度。

为什么有这个文件
----------------
本仓库的正式重定向是训练出来的 PoseTransformer，它必须有 ``palm_local``
检查点才能推理。本机（无 GPU、仓库内无任何 .pth）拿不到该检查点，因此
"摄像头 -> 重定向 -> 仿真"的实时闭环需要第二个可选后端先跑起来。

本模块提供的就是这个**可替换后端**：纯几何、零权重、无训练，接口与
``HandRetargeter`` 完全一致（``retarget(HandWindow) -> HandRetargetResult``）。
**注意：它目前还没有接进任何 CLI** —— ``hand realtime`` 只加载检查点模型；
本文件当前的使用者是 ``devtools/verify_geometric_hand.py`` 那条往返实测。
要把"摄像头 -> 重定向 -> 仿真"跑起来，需要新增一个实时入口（见 README 的
待办），入口内以 ``--backend geometric`` 选择本后端、检查点缺失时自动回落。

与论文的关系（必须如实声明）
----------------------------
论文 Vision 接口的手部重定向用 dex-retargeting（向量优化，非学习）。
本模块同样**不含学习模型**，直接读骨节夹角，属于同一"非学习"类别；
但它**不是 dex-retargeting 的复现**：不做指尖距离优化，不做多指联合
优化，没有碰撞项。用它得到的结果只能作为实时闭环的可用后端，不能
当作论文数值对比结果。

坐标系与符号（本模块唯一的"魔法"，全部是实测结果）
--------------------------------------------------
输入是 ``MediaPipePalmLocalProcessor`` 对齐后的 25 点窗口（``HandWindow``），
即"以 L21 零位姿掌面基为轴、相对手腕"的坐标：

    X = pinky MCP -> index MCP    （掌侧向，+X = 食指侧）
    Y = wrist -> 四指 MCP 均值     （掌纵向，+Y = 指向指尖）
    Z = X x Y                     （掌法向，正负由手性决定）

屈曲量一律绕**该关节在 URDF 里的自身轴**测量（见 ``joint_axes``），不靠几何
猜测：URDF 的轴已经自带"正值为屈曲"的方向，所以正常输入不做任何手性缩放。
实测依据：``outputs/tmp_probe/l21_axes_probe.py`` 在左右 URDF 上把 11 个屈曲维
置 0.8 rad，指尖都朝各自掌心移动（左 z=-0.078 / 右 z=+0.084）；而绕掌面侧向
轴测拇指会得到 -30.7 度且符号翻转，故拇指必须用自身轴。

镜像输入（自拍镜像 / 图像翻转）会让上面这套量整体反号，所以保留一个**镜像
自检**：L21 零位姿的拇指掌骨偏掌心一侧（``expected_thumb_lean``，实测约
±0.03），把输入拇指掌骨在法向上的投影取窗口内中位数，与它同号则为正常输入
（系数 1.0），异号则判为镜像输入（系数 -1.0，屈曲维与拇指 yaw 整体取反）；
证据不足时按正常输入处理（URDF 轴已在正常输入下给出正确符号）。

手掌法向/掌面基来自公开函数 ``build_l21_reference_basis(side)``（首次使用时
惰性计算并缓存，走既有 L21 URDF 与 FK，不新增资产）。

角度定义（几何量 -> 18 维）
--------------------------
四指（index/middle/ring/pinky），每指用 5 个点 [掌根, MCP, PIP, DIP, TIP]：

    mcp_pitch = 掌骨方向与近节指骨方向绕**该关节自身轴**（L21 URDF 的
                mcp_pitch 轴，由 FK 读出）的有符号夹角
    pip       = PIP 屈曲 + DIP 屈曲（L21 远端只有 1 个屈曲关节，必须承载
                远端两节的总屈曲；再夹到 1.57）
    mcp_roll  = 0.0（无标定的侧摆估计不可靠；H1-2/GR1-T2/G1 的原装手本来
                就没有侧摆自由度，NATIVE_MAP 会整维丢弃）

拇指，用 4 个点 [CMC, MCP, IP, TIP]。L21 的拇指是**串联链**：cmc_yaw 会带动
cmc_pitch/mcp/ip 及其后面全部骨骼，cmc_pitch 又会带动 mcp/ip，所以拇指掌骨
方向是 yaw 与 pitch 的**联合**结果，不能拿两个互相独立的平面角去凑。实测
（outputs/tmp_probe/geom_thumb_probe.py）：把真值 (0.70, 0.40) 的 L21 姿态
喂回来，旧的两角分解只能读成 (0.35, 0.15)，拇指对掌严重欠拟合；链式反解
能读到 (0.70, 0.42)，误差 0.02 以内。

    thumb_cmc_yaw / thumb_cmc_pitch
        = 以 L21 URDF + FK 建出拇指零位链（CMC->MCP->IP->TIP 的连杆向量与
          各关节轴，见 _thumb_model），把 (yaw, pitch) 当未知量做链式正解，
          反求使**掌骨方向**与观测最接近的那一组（粗网格 + 高斯-牛顿，见
          _solve_yaw_pitch）；两者都相对标定零位（0 -> 1.6 / 0 -> 1.0）
    thumb_mcp = 掌骨与近节夹角，但近节骨**与关节轴**都要先用 yaw/pitch 旋转
        父链再测量（用零位轴测量等于把锥面压扁，远端角度会失真）
    thumb_ip  = 同上，父链再多带一层 mcp
    thumb_cmc_roll  = 0.0（同上：无法从 21 点稳定估计，原装手也丢弃）

零位标定（open-hand calibration）
--------------------------------
L21 的零位姿并不是"平手"（实测拇指 IP 静屈约 83 度，掌骨也各有几度偏斜），
真人放松的手同样不是平手，所以上面**每个屈曲维都取相对"标定帧"的量**：
用前 ``calibration_frames`` 个有效帧的均值作为零位参考并冻结（``reset()``
清除）；``calibrate=False`` 时零位取全 0，退化成绝对几何量。标定未完成时
该手返回"无效"而不是猜一个角度，调用方保持上一帧姿态即可。

已知代价（写报告时必须带上）
---------------------------
1) 4 个 ``*_mcp_roll`` 与 ``thumb_cmc_roll`` 恒为 0，侧摆信息丢失；
2) 所有屈曲维都依赖一次"张开手"零位标定，标定姿势不标准会带来常值偏移
   （拇指的 yaw/pitch 对标定尤其敏感）；
3) 没有 dex-retargeting 的指尖优化，精度低于论文做法（误差量级见
   ``devtools/verify_geometric_hand.py`` 的 L21 往返实测）；
4) 拇指链式反解只在**掌骨方向**上做最小二乘，观测量（2 个自由度）恰好等于
   未知量（2 个），在 yaw/pitch 影响掌骨方向接近同向的位形上病态：右手实测
   最大残差约 0.03 rad（yaw 0.01 / pitch 0.03，见探针的"方案 G"），来源是
   对齐帧与 URDF 零位帧之间约 1 度的朝向差被放大；判据（devtools 脚本里
   0.25）能过，但不是复现；先按标定帧的掌骨方向把模型转正（见 _thumb_model）
   已把误差从 0.06 rad 压到 0.03 rad 以内；
5) 镜像自检只在"拇指掌骨明显偏出掌面"时才有证据（零位附近掌骨几乎躺在掌面内
   时会回落成不取反），且拇指链与屈曲维的镜像处理都依赖该自检正确；判为镜像
   时整条拇指模型按掌面做镜像、关节轴取反（实测与镜像输入自洽）。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from teleoperation.contracts.hand import HandRetargetResult, HandWindow, L21HandAngles
from teleoperation.retargeting.hand.config import ANGLE_LIMITS, HAND_ANGLE_DIM, L21
from teleoperation.retargeting.hand.interface import HandRetargeter

SIDES = ("left", "right")

# 25 点布局（与 retargeting/hand/topology.py 的 mediapipe21_to_hand25 一致）
WRIST = 0
THUMB = {"cmc": 1, "mcp": 2, "ip": 3, "tip": 4}
FINGERS = {
    "index": (5, 6, 7, 8, 9),
    "middle": (10, 11, 12, 13, 14),
    "ring": (15, 16, 17, 18, 19),
    "pinky": (20, 21, 22, 23, 24),
}

# 18 维 L21 向量里的下标（0 是 hand_base_link 固定占位）
ROLL_DIM = {"index": 1, "middle": 4, "ring": 7, "pinky": 10}
PITCH_DIM = {"index": 2, "middle": 5, "ring": 8, "pinky": 11}
PIP_DIM = {"index": 3, "middle": 6, "ring": 9, "pinky": 12}
THUMB_ROLL_DIM, THUMB_YAW_DIM, THUMB_PITCH_DIM = 13, 14, 15
THUMB_MCP_DIM, THUMB_IP_DIM = 16, 17

# 必须"绕关节自身轴"测量的屈曲维（下标同时就是 L21 关节下标）。
# 用掌面侧向轴测四指是等价的，但拇指的屈曲平面相对掌面是斜的：实测绕侧向轴
# 测 thumb_mcp=0.8 rad 只剩 -30.7 度且符号翻转，绕关节自身轴则严格一致。
# 拇指的 cmc_yaw/cmc_pitch 也要取轴：链式反解（_solve_yaw_pitch）用它们做正解，
# 而 mcp/ip 的父链旋转同样要用到这两条轴。
HINGE_DIMS = (2, 3, 5, 6, 8, 9, 11, 12, THUMB_YAW_DIM, THUMB_PITCH_DIM,
              THUMB_MCP_DIM, THUMB_IP_DIM)

# 手性：URDF 的关节轴已自带"正值为屈曲"的方向，正常输入不需要再翻转；
# 只有**镜像输入**（自拍镜像等）需要整体取反，判据见 _mirror_factor。
MIN_AXIS_LENGTH = 1e-6
MIN_SIGN_EVIDENCE = 1e-4
# 取关节轴时的试探角度（rad）；只要落在 (0, pi) 内，轴与正负号就唯一确定。
PROBE_DELTA = 0.3
# L21 关节序号里拇指掌骨的两端（左手；右手 = 23 - 序号）。
THUMB_CMC_NODE, THUMB_MCP_NODE = 13, 16
THUMB_IP_NODE, THUMB_TIP_NODE = 17, 22
L21_JOINT_COUNT = 23
# 拇指链式反解的规模：粗网格 11x9=99 个候选（矢量化的 Rodrigues）先选盆地，
# 再用 12 步 Gauss-Newton 细化到模型能做到的方向精度。
# 实测（E:\\python3.11.7，单线程，outputs/tmp_probe/geom_timing.log）：
# 单次求解 0.92 ms，两只手一帧 1.83 ms，整条 retarget 3.13 ms/帧，
# 相对 30 fps 的 33 ms 预算有余量；要再压成本先降 REFINE_STEPS，
# 降完必须重跑 devtools/verify_geometric_hand.py 确认判据仍全过。
COARSE_YAW_STEPS, COARSE_PITCH_STEPS, REFINE_STEPS = 11, 9, 12


def _unit(vector: np.ndarray) -> np.ndarray | None:
    """Return a unit vector, or None for zero/non-finite input."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= MIN_AXIS_LENGTH:
        return None
    return vector / norm


def bend_angle(parent: np.ndarray, child: np.ndarray, axis: np.ndarray, sign: float) -> float:
    """Signed rotation from ``parent`` to ``child`` about ``axis`` (radians).

    ``sign`` carries the chirality so that bending toward the palm is positive.
    Returns 0.0 when either bone is degenerate.
    """
    parent_unit = _unit(np.asarray(parent, dtype=np.float64))
    child_unit = _unit(np.asarray(child, dtype=np.float64))
    if parent_unit is None or child_unit is None:
        return 0.0
    circle = float(np.dot(parent_unit, child_unit))
    sine = float(np.dot(np.cross(parent_unit, child_unit), np.asarray(axis, dtype=np.float64)))
    return float(sign * np.arctan2(sine, circle))


def _in_plane_angle(direction: np.ndarray, lateral: np.ndarray, longitudinal: np.ndarray) -> float:
    """Angle of ``direction`` inside the palm plane, measured from +Y toward +X.

    Kept for probing the palm-plane reading of the thumb metacarpal (the coarse
    "in-plane yaw" the chain inversion replaced); no longer used by the pipeline.
    """
    return float(np.arctan2(float(np.dot(direction, lateral)), float(np.dot(direction, longitudinal))))


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit ``axis`` by ``angle`` radians."""
    x, y, z = (float(value) for value in axis)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (np.cos(angle) * np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * np.outer(axis, axis))


def _rotation_family(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """``_rodrigues`` for many angles at once; returns ``(len(angles), 3, 3)``."""
    x, y, z = (float(value) for value in axis)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    return cosine * np.eye(3) + sine * cross + (1.0 - cosine) * np.outer(axis, axis)


def _reflect(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mirror ``vector`` through the plane whose unit normal is ``normal``."""
    return vector - 2.0 * float(np.dot(vector, normal)) * normal


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit ``source`` onto unit ``target``."""
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(float(np.dot(source, target)), -1.0, 1.0))
    if sine < MIN_AXIS_LENGTH:
        return np.eye(3) if cosine > 0.0 else -np.eye(3)
    return _rodrigues(cross / sine, float(np.arctan2(sine, cosine)))


def _thumb_chain_offsets(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pose link vectors of the L21 thumb chain: CMC->14, 14->15, 15->MCP.

    L21 links the thumb as hand_base -> cmc_roll -> cmc_yaw -> cmc_pitch -> mcp
    -> ip -> tip, i.e. nodes 13, 14, 15, 16, 17, 22; node 13 is the CMC and node
    16 the MCP, so the metacarpal is the sum of these three vectors after the
    per-joint rotations (cmc_roll is fixed at 0 in L21, hence a plain sum).
    """
    return (nodes[13 + 1] - nodes[13], nodes[14 + 1] - nodes[14], nodes[15 + 1] - nodes[15])


def _thumb_metacarpal(offsets, axes: dict[int, np.ndarray],
                      yaw: float, pitch: float) -> np.ndarray:
    """Forward kinematics of the L21 thumb chain: CMC -> MCP direction."""
    d_roll, d_yaw, d_pitch = offsets
    return d_roll + _rodrigues(axes[THUMB_YAW_DIM], yaw) @ (
        d_yaw + _rodrigues(axes[THUMB_PITCH_DIM], pitch) @ d_pitch)


def _normalise(vector: np.ndarray) -> np.ndarray:
    """Unit vector of a thumb link chain; the chain is never degenerate here."""
    return vector / max(float(np.linalg.norm(vector)), MIN_AXIS_LENGTH)


def _solve_yaw_pitch(offsets, axes: dict[int, np.ndarray], observed: np.ndarray,
                     bounds: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Invert the thumb chain: the (yaw, pitch) whose metacarpal matches ``observed``.

    ``observed`` is the unit CMC -> MCP direction of the input hand. The two CMC
    angles cannot be separated in closed form: both rotate a chain whose later
    links are oblique to their own axes, so the metacarpal direction is their
    joint image. A coarse grid over the declared limits (vectorised Rodrigues)
    picks the basin, then a few Gauss-Newton steps refine it; the returned
    residual is the direction error left over (0 = the model can reproduce it).
    """
    yaw_low, yaw_high, pitch_low, pitch_high = bounds
    yaw_grid = np.linspace(yaw_low, yaw_high, COARSE_YAW_STEPS)
    pitch_grid = np.linspace(pitch_low, pitch_high, COARSE_PITCH_STEPS)
    d_roll, d_yaw, d_pitch = offsets
    yaw_rotations = _rotation_family(axes[THUMB_YAW_DIM], yaw_grid)
    pitch_rotations = _rotation_family(axes[THUMB_PITCH_DIM], pitch_grid)
    tails = d_yaw[None, None, :] + np.einsum("...ij,j->...i", pitch_rotations, d_pitch)[None]
    directions = np.einsum("...ij,...j->...i", yaw_rotations[:, None], tails) + d_roll
    costs = np.linalg.norm(directions / np.linalg.norm(directions, axis=-1, keepdims=True)
                           - observed, axis=-1)
    row, column = np.unravel_index(int(np.argmin(costs)), costs.shape)
    yaw, pitch = float(yaw_grid[row]), float(pitch_grid[column])
    for _ in range(REFINE_STEPS):
        residual = _normalise(_thumb_metacarpal(offsets, axes, yaw, pitch)) - observed
        step = 1e-4
        jacobian = np.column_stack([
            (_normalise(_thumb_metacarpal(offsets, axes, yaw + step, pitch))
             - observed - residual) / step,
            (_normalise(_thumb_metacarpal(offsets, axes, yaw, pitch + step))
             - observed - residual) / step,
        ])
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        yaw = float(np.clip(yaw + float(delta[0]), yaw_low, yaw_high))
        pitch = float(np.clip(pitch + float(delta[1]), pitch_low, pitch_high))
    left = _normalise(_thumb_metacarpal(offsets, axes, yaw, pitch)) - observed
    return yaw, pitch, float(np.linalg.norm(left))




def _clamp(value: float, index: int) -> float:
    """Clamp one L21 dimension into its declared limit (see ANGLE_LIMITS)."""
    low, high = ANGLE_LIMITS[index]
    return float(min(max(value, low), high))


class GeometricHandRetargeter(HandRetargeter):
    """Palm-local landmarks -> L21 angles, with a per-frame chirality self-check.

    ``retarget`` never raises on poor geometry: a side without enough usable
    points is reported invalid instead, so a live loop keeps running while one
    hand is out of frame.
    """

    def __init__(
        self,
        calibration_frames: int = 15,
        sign_window: int = 30,
        smoothing: float = 0.0,
        calibrate: bool = True,
        reference_bases: dict[str, np.ndarray] | None = None,
    ):
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be positive")
        if sign_window < 1:
            raise ValueError("sign_window must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.calibration_frames = int(calibration_frames)
        self.sign_window = int(sign_window)
        self.smoothing = float(smoothing)
        self.calibrate = bool(calibrate)
        self._bases = dict(reference_bases or {})
        self._axes: dict[str, dict[int, np.ndarray]] = {}
        self._nodes: dict[str, np.ndarray] = {}
        self._lean: dict[str, float] = {side: 0.0 for side in SIDES}
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Drop the open-hand calibration, the sign history and the smoothing state."""
        self._rest_samples = {side: [] for side in SIDES}
        self._rest_directions = {side: [] for side in SIDES}
        self._rest_angles = {
            side: None if self.calibrate else np.zeros(HAND_ANGLE_DIM, dtype=np.float64)
            for side in SIDES
        }
        self._rest_direction = {side: None for side in SIDES}
        self._lean_evidence = {side: deque(maxlen=self.sign_window) for side in SIDES}
        self._previous = {side: None for side in SIDES}

    @property
    def calibration_status(self) -> dict[str, bool]:
        """Per-side calibration status; the app prints it while starting up."""
        return {side: self._rest_angles[side] is not None for side in SIDES}

    def basis(self, side: str) -> np.ndarray:
        """Zero-pose L21 palm basis (columns = X, Y, Z); cached per side."""
        if side not in SIDES:
            raise ValueError(f"Unknown hand side: {side!r}")
        if side not in self._bases:
            from teleoperation.retargeting.hand.coordinates import build_l21_reference_basis

            self._bases[side] = build_l21_reference_basis(side)
        return self._bases[side]

    def joint_axes(self, side: str) -> dict[int, np.ndarray]:
        """Each flexion joint's axis in the aligned frame, read from the L21 URDF.

        The axis is the joint's own: rotating that joint by a positive value
        rotates its child chain about this axis, which is exactly the frame the
        aligned landmarks live in (``build_l21_reference_basis`` uses the same
        FK). Guessing the axis from the palm geometry is not equivalent for the
        thumb, whose flexion plane is oblique to the palm lateral axis.
        """
        return self._robot_reference(side)[0]

    def expected_thumb_lean(self, side: str) -> float:
        """How this side's own zero-pose thumb metacarpal leans along the palm normal."""
        return self._robot_reference(side)[1]

    def _robot_reference(self, side: str) -> tuple[dict[int, np.ndarray], float, np.ndarray]:
        """One FK pass over the zero pose: per-joint axes, thumb lean, node cloud.

        The node cloud is the zero-pose skeleton the thumb chain inversion needs
        (``_thumb_chain_offsets``); only its link vectors are used, so a caller
        may rotate the whole cloud freely.
        """
        if side not in self._axes:
            import torch

            from teleoperation.retargeting.hand.kinematics import create_hand_kinematics

            fk = create_hand_kinematics(
                getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(),
                device="cpu", scale_factor=L21.training.robot_scale,
            )
            zero = torch.zeros((1, L21_JOINT_COUNT), dtype=torch.float32)
            axes = {}
            with torch.no_grad():
                _, zero_rotations, global_positions = fk.forward(zero)
            for dim in HINGE_DIMS:
                pose = zero.clone()
                pose[0, dim] = PROBE_DELTA
                with torch.no_grad():
                    _, delta_rotations, _ = fk.forward(pose)
                # R_delta @ R_zero^T 就是该关节在零位姿坐标系里的纯旋转，
                # 其反对称部即轴：与 FK 内部"角度左乘还是右乘"的约定无关。
                relative = (delta_rotations[0, dim] @ zero_rotations[0, dim].T).numpy()
                axis = _unit(np.asarray([
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ], dtype=np.float64))
                if axis is not None:
                    axes[dim] = axis
            nodes = global_positions[0].numpy().astype(np.float64)
            metacarpal = _unit(nodes[THUMB_MCP_NODE] - nodes[THUMB_CMC_NODE])
            lean = 0.0 if metacarpal is None else float(np.dot(metacarpal, self.basis(side)[:, 2]))
            self._axes[side] = axes
            self._lean[side] = lean
            self._nodes[side] = nodes
        return self._axes[side], self._lean[side], self._nodes[side]

    # ------------------------------------------------------------------ #
    def retarget(self, hand_window: HandWindow) -> HandRetargetResult:
        if not isinstance(hand_window, HandWindow):
            raise TypeError("hand_window must be a HandWindow")
        angles = {"left": None, "right": None}
        for side in SIDES:
            window = hand_window.hands[side]
            if window is None:
                continue
            frame = np.asarray(window, dtype=np.float64)[-1]
            values = self._side_angles(side, frame)
            if values is None:
                continue
            angles[side] = values if self.smoothing == 0.0 else self._smooth(side, values)
        return HandRetargetResult(
            left_angles=None if angles["left"] is None else L21HandAngles(angles["left"]),
            right_angles=None if angles["right"] is None else L21HandAngles(angles["right"]),
            left_valid=angles["left"] is not None,
            right_valid=angles["right"] is not None,
            timestamp=hand_window.timestamp,
        )

    # ------------------------------------------------------------------ #
    def _side_angles(self, side: str, points: np.ndarray) -> np.ndarray | None:
        """One aligned (25,3) frame -> 18D L21 angles, or None if unusable."""
        if points.shape != (25, 3) or not np.isfinite(points).all() or not np.any(points):
            return None
        normal = self.basis(side)[:, 2]
        axes = self.joint_axes(side)
        mirror = self._mirror_factor(side, points, normal)
        raw = np.zeros(HAND_ANGLE_DIM, dtype=np.float64)
        measured = 0
        for name, (root, mcp, pip, dip, tip) in FINGERS.items():
            pitch_axis = axes.get(PITCH_DIM[name])
            if pitch_axis is None:
                continue
            bones = [
                points[mcp] - points[root],
                points[pip] - points[mcp],
                points[dip] - points[pip],
                points[tip] - points[dip],
            ]
            if any(_unit(bone) is None for bone in bones):
                continue
            raw[PITCH_DIM[name]] = bend_angle(bones[0], bones[1], pitch_axis, mirror)
            raw[PIP_DIM[name]] = (bend_angle(bones[1], bones[2], pitch_axis, mirror)
                                  + bend_angle(bones[2], bones[3], pitch_axis, mirror))
            measured += 1
        thumb = self._thumb_angles(side, points, mirror, axes, normal)
        raw[THUMB_MCP_DIM] = thumb["mcp"]
        raw[THUMB_IP_DIM] = thumb["ip"]
        raw[THUMB_YAW_DIM] = thumb["yaw"]
        raw[THUMB_PITCH_DIM] = thumb["pitch"]
        if not measured and not thumb["valid"]:
            return None
        reference = self._open_hand_reference(side, raw, points)
        if reference is None:
            return None                     # 标定未完成：本帧不输出角度
        values = raw - reference
        return np.asarray([_clamp(float(values[dim]), dim) for dim in range(HAND_ANGLE_DIM)])

    def _mirror_factor(self, side: str, points: np.ndarray, normal: np.ndarray) -> float:
        """+1 for a normal source, -1 when the input hand looks mirrored.

        The L21 zero pose says which way its own thumb metacarpal leans along the
        palm normal (``expected_thumb_lean``); a source whose thumb leans the
        other way is a mirrored point cloud (selfie mirror, flipped image), and
        then every signed angle must be negated. Weak evidence falls back to +1
        because the URDF joint axes already carry the flexion sign.
        """
        metacarpal = _unit(points[THUMB["mcp"]] - points[THUMB["cmc"]])
        if metacarpal is not None:
            self._lean_evidence[side].append(float(np.dot(metacarpal, normal)))
        evidence = float(np.median(self._lean_evidence[side])) if self._lean_evidence[side] else 0.0
        expected = self._lean[side]
        if abs(evidence) <= MIN_SIGN_EVIDENCE or expected == 0.0:
            return 1.0
        return 1.0 if evidence * expected > 0.0 else -1.0

    def _open_hand_reference(self, side: str, raw: np.ndarray,
                             points: np.ndarray) -> np.ndarray | None:
        """Freeze the open-hand zero pose: all 18 raw readings from one frame.

        Neither the L21 URDF zero pose nor a resting human hand is flat (the L21
        thumb IP rests about 83 deg flexed and its metacarpals lean a few
        degrees), so all dims - the four-finger bends, the thumb chain
        yaw/pitch/mcp/ip and the frozen rolls - are reported relative to one
        calibrated frame. The reference freezes after ``calibration_frames``
        valid frames; until then this side reports no angles instead of a
        guessed curl.
        """
        if self._rest_angles[side] is not None:
            return self._rest_angles[side]
        if not self.calibrate:
            self._rest_angles[side] = np.zeros(HAND_ANGLE_DIM, dtype=np.float64)
            return self._rest_angles[side]
        direction = _unit(points[THUMB["mcp"]] - points[THUMB["cmc"]])
        if direction is None:
            return None
        samples = self._rest_samples[side]
        samples.append(raw.copy())
        self._rest_directions[side].append(direction)
        if len(samples) < self.calibration_frames:
            return None
        reference = np.mean(samples, axis=0)
        # yaw/pitch 现在由链式反解给出（相对 URDF 零位姿的绝对量），标定帧的
        # 读数不再恒为 0，所以照平均值冻结；_thumb_model 还会按该帧的掌骨方向
        # 把模型转正，两项合起来让标定姿势读到 0（见模块文档"已知代价"4）。
        self._rest_direction[side] = _unit(np.mean(self._rest_directions[side], axis=0))
        self._rest_angles[side] = reference
        return reference

    def _thumb_angles(self, side: str, points: np.ndarray, mirror: float,
                      axes: dict[int, np.ndarray], normal: np.ndarray) -> dict:
        """Thumb joints by inverting the L21 thumb chain (see the module docstring).

        ``yaw``/``pitch`` come from the chain forward kinematics solved against
        the observed metacarpal direction; ``mcp``/``ip`` then measure the distal
        bones about axes (and from parent bones) carried along by that chain. All
        four are absolute readings in the model's frame; ``_open_hand_reference``
        subtracts the calibrated rest, so the open hand reads 0.
        """
        result = {"mcp": 0.0, "ip": 0.0, "yaw": 0.0, "pitch": 0.0,
                  "residual": 0.0, "valid": False}
        metacarpal = _unit(points[THUMB["mcp"]] - points[THUMB["cmc"]])
        proximal = _unit(points[THUMB["ip"]] - points[THUMB["mcp"]])
        distal = _unit(points[THUMB["tip"]] - points[THUMB["ip"]])
        if metacarpal is None or proximal is None or distal is None:
            return result
        result["valid"] = True
        nodes, model_axes = self._thumb_model(side, mirror, normal)
        offsets = _thumb_chain_offsets(nodes)
        yaw, pitch, residual = _solve_yaw_pitch(
            offsets, model_axes, metacarpal,
            (ANGLE_LIMITS[THUMB_YAW_DIM][0], ANGLE_LIMITS[THUMB_YAW_DIM][1],
             ANGLE_LIMITS[THUMB_PITCH_DIM][0], ANGLE_LIMITS[THUMB_PITCH_DIM][1]))
        result["yaw"], result["pitch"], result["residual"] = yaw, pitch, residual
        # 父链旋转：yaw 与 pitch 都在 mcp/ip 之前，所以远端读数要带上它
        # （用零位轴去量近节/末节等于把锥面压扁，远端角度会系统性偏小）。
        parent = _rodrigues(model_axes[THUMB_YAW_DIM], yaw) @ _rodrigues(
            model_axes[THUMB_PITCH_DIM], pitch)
        if THUMB_MCP_DIM in model_axes:
            rest_proximal = _normalise(nodes[THUMB_IP_NODE] - nodes[THUMB_MCP_NODE])
            result["mcp"] = bend_angle(parent @ rest_proximal, proximal,
                                       parent @ model_axes[THUMB_MCP_DIM], 1.0)
        if THUMB_IP_DIM in model_axes:
            rest_distal = _normalise(nodes[THUMB_TIP_NODE] - nodes[THUMB_IP_NODE])
            above_mcp = parent @ _rodrigues(model_axes[THUMB_MCP_DIM], result["mcp"])
            result["ip"] = bend_angle(above_mcp @ rest_distal, distal,
                                      above_mcp @ model_axes[THUMB_IP_DIM], 1.0)
        return result

    def _thumb_model(self, side: str, mirror: float,
                     normal: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Zero-pose thumb chain (nodes, axes) brought into this frame.

        Two corrections are applied before the chain is inverted:

        1) ``mirror < 0`` (see ``_mirror_factor``): the whole skeleton is mirrored
           through the palm plane and its axes negated, so a mirrored source is
           compared against a mirrored L21 (measured to be self-consistent);
        2) once calibrated, the model is rotated by the smallest rotation taking
           its own zero-pose metacarpal onto the calibrated rest direction, so
           the inverted angles are relative to the user's open hand instead of
           the URDF zero pose (this is what makes the flat hand read 0).
        """
        axes, _lean, nodes = self._robot_reference(side)
        # 缓存只读：镜像与掌骨对齐都会造新数组，两条都不触发时按引用返回也不会被改写。
        model_axes = dict(axes)
        if mirror < 0.0:
            nodes = np.asarray([_reflect(node, normal) for node in nodes])
            model_axes = {dim: -_reflect(axis, normal) for dim, axis in model_axes.items()}
        rest = self._rest_direction[side]
        zero_metacarpal = _unit(nodes[THUMB_MCP_NODE] - nodes[THUMB_CMC_NODE])
        if rest is not None and zero_metacarpal is not None:
            rotation = _rotation_between(zero_metacarpal, rest)
            nodes = nodes @ rotation.T
            model_axes = {dim: rotation @ axis for dim, axis in model_axes.items()}
        return nodes, model_axes

    def _smooth(self, side: str, values: np.ndarray) -> np.ndarray:
        """First-order low pass; a stand-in for the paper's Kalman filter."""
        previous = self._previous[side]
        smoothed = values.copy() if previous is None else (
            (1.0 - self.smoothing) * values + self.smoothing * previous)
        self._previous[side] = smoothed
        return smoothed
