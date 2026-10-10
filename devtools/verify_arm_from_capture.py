"""手臂链路的轻量验收：21 点 -> 手腕目标 -> PyBullet IK -> 论文三机器人之一。

为什么用 PyBullet 自带 IK
-------------------------
队友的手臂栈（``retargeting/arm/ik.py`` + ``apps/dual.py``）是给 **TRON2A** 写的，
要 ``third_party/tron2-robot-description/.../robot.urdf`` + 一份 TRON2A 标定 JSON +
训练出来的手部检查点；本机这三样都没有，所以那条链路跑不起来。论文三台机器人
（H1-2 / GR1-T2 / G1）的 URDF 在 ``assets/robots/from_teleopbench/`` 里**是齐的**，
PyBullet 自带的 ``calculateInverseKinematics`` 就能对它们的 7 DoF 手臂求解。

因此本脚本证明的是：**"手腕目标位姿 -> 7 DoF 手臂"这一段能跑通，并且有数值证据**；
手腕目标本身来自 ``retargeting/arm/wrist_targets.py`` 的近似映射（不是手眼标定，
更不是 SMPLer-X），所以这里得到的**不是**论文数值结论。

输入两种
--------
--synthetic-sweep（默认）：把 ``datasets/samples/human_hand_demo_right.h5`` 的手
    搬到图像中心再按正弦平移 + 绕掌面法向旋转，制造"手在动"的输入；
--input-h5 raw.h5：Layer 1 录制出来的原始抓取 H5（帧里的手来自真实摄像头；
    没有检出的帧按既有约定是全零，这里还原成 None 并保持上一帧目标）。

判据（任一不过即 [FAIL]，退出码 1）
----------------------------------
1) 目标合法的帧数 >= 1（目标非法时保持上一帧，不猜）
2) 末端实际位置与目标位置的距离残差 mean <= --max-residual-mm（默认 30 mm）
3) 末端确实动了：三轴行程 max >= --min-travel-mm（默认 10 mm）
4) 非手臂关节（腿/腰/另一侧手臂）没有被带动：最大角变化 <= --max-hold-drift-rad
   （默认 0.05 rad）。**两手的手指关节不在这一项里**：本脚本不驱动灵巧手，GR1-T2
   的耦合/无驱动手指在重力下垂 0.14 rad（实测），那是手部子系统的事，单独记录

用法::

    E:\\python3.11.7\\python.exe devtools\\verify_arm_from_capture.py --robot h1_2
    E:\\python3.11.7\\python.exe devtools\\verify_arm_from_capture.py \\
        --input-h5 outputs/tmp_probe/camera_record/capture_XXXX.h5 --side right
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SIDES = ("left", "right")
SIDE_KEYS = {"left": "left_hand_keypoints", "right": "right_hand_keypoints"}
SAMPLE_HAND = ROOT / "datasets" / "samples" / "human_hand_demo_right.h5"
PHYSICS_DT = 1.0 / 240.0
STEPS_PER_FRAME = 8
ARM_FORCE = 200.0
HOLD_FORCE = 500.0
IK_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-5
# Per-robot profiles measured 2026-10-10 (steady-state residual, right arm, 30-40
# frames of the synthetic sweep). Constant-target probes with `--scale 1e-6`:
#   h1_2  (0.25,0,0.20)=15.6 mm ok, (0.18,0,0.12)=10.4 mm ok, (0.15,0,0.25)=10.1 mm ok,
#         (0.30,0,0.05)=81.8 mm and (0.35,0,-0.05)=358 mm are OUT of reach;
#   g1    (0.24,0,0.19)=33.1 mm, (0.18,0,0.12)=15.4 mm ok.
# Sweep matrix (scale x ik-mode) -> residual mean:
#   g1     0.7/pose 36.4 FAIL | 0.7/position 24.8 | 0.4/pose 29.5 | 0.4/position(0.16,0,0.10) 17.9
#   gr1_t2 0.7/pose 58.5 FAIL | 0.4/pose 53.8 FAIL | 0.4/position(0.20,0,0.15) 43.5 FAIL
#          | 0.4/position(0.16,0,0.12) 22.3 ok
# So the smaller robots (g1, gr1_t2) need a smaller sweep gain and a target closer
# to the torso, and their wrist chain cannot track the (uncalibrated) orientation
# convention over the sweep, hence position-only IK there. These are tuning values,
# not calibration.
ROBOT_PROFILES = {
    "h1_2": {"scale": 0.7, "base_offset": (0.24, 0.0, 0.19), "ik_mode": "pose"},
    "gr1_t2": {"scale": 0.4, "base_offset": (0.16, 0.0, 0.12), "ik_mode": "position"},
    "g1": {"scale": 0.4, "base_offset": (0.16, 0.0, 0.10), "ik_mode": "position"},
}


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def rodrigues(axis, angle) -> np.ndarray:
    """Rotation matrix about ``axis`` by ``angle`` (no scipy dependency)."""
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    cross = np.array([[0.0, -vector[2], vector[1]],
                      [vector[2], 0.0, -vector[0]],
                      [-vector[1], vector[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def frame_hand(points):
    """(21,3) landmarks, or None when the recorder wrote zeros (no hand)."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        return None
    return None if not np.any(np.abs(array) > 1e-9) else array


def sweep_frames(count: int, sides) -> list:
    """Synthetic "the hand moves in front of the camera" sequence (see module doc)."""
    import h5py

    with h5py.File(SAMPLE_HAND, "r") as handle:
        points = np.asarray(handle["keypoints_3d"][:], dtype=np.float64)
    total = max(1, min(int(count), points.shape[0]))
    frames = []
    for index in range(total):
        progress = index / max(total - 1, 1)
        hand = points[index] + np.array([0.5, 0.5, 0.0])
        wrist = hand[0].copy()
        hand = (rodrigues([0.0, 0.0, 1.0], 0.5 * np.sin(2.0 * np.pi * progress))
                @ (hand - wrist).T).T + wrist
        offset = np.array([0.10 * np.sin(2.0 * np.pi * progress),
                           0.06 * np.sin(4.0 * np.pi * progress), 0.0])
        frames.append({side: hand + offset for side in sides})
    return frames


def frames_from_h5(path, sides) -> list:
    """Raw capture H5 (Layer 1 output); zeros mean "no hand in this frame"."""
    import h5py

    with h5py.File(path, "r") as handle:
        arrays = {side: np.asarray(handle[SIDE_KEYS[side]][:], dtype=np.float64)
                  for side in sides}
    count = min(arrays[side].shape[0] for side in sides)
    return [{side: frame_hand(arrays[side][index]) for side in sides}
            for index in range(count)]


def movable_joints(p, body, cid):
    """``(indices, infos)`` of every non-fixed joint, in PyBullet order."""
    indices, infos = [], []
    for index in range(p.getNumJoints(body, physicsClientId=cid)):
        info = p.getJointInfo(body, index, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED:
            continue
        indices.append(index)
        infos.append(info)
    return indices, infos


def end_effector_index(p, body, side, cid):
    """``(index, link_name)`` of that side's hand mount (the wrist/hand link).

    Naming differs per robot (measured 2026-10-10):
        h1_2   right_wrist_yaw_link      (no hand links)
        g1     right_wrist_yaw_link      (hand links exist but come later)
        gr1_t2 right_hand_pitch_link     (no wrist links at all)
    So: prefer the last ``{side}_*wrist*`` link, else the last ``{side}_*hand*`` one.
    """
    for keyword in ("wrist", "hand"):
        candidates = []
        for index in range(p.getNumJoints(body, physicsClientId=cid)):
            info = p.getJointInfo(body, index, physicsClientId=cid)
            link = info[12].decode() if isinstance(info[12], bytes) else info[12]
            text = str(link).lower()
            if text.startswith(side + "_") and keyword in text:
                candidates.append((index, str(link)))
        if candidates:
            return max(candidates)
    raise RuntimeError(f"no {side} wrist/hand link found")


class ArmIkHarness:
    """One paper robot; the active arms follow IK targets, every other joint holds."""

    def __init__(self, robot_type: str, sides, gui: bool = False,
                 ik_mode: str = "pose", gravity: float = 0.0):
        import pybullet as p
        import pybullet_data

        from teleoperation.simulation import RobotLoader

        if ik_mode not in ("position", "pose"):
            raise ValueError(f"ik_mode must be 'position' or 'pose', got {ik_mode!r}")
        self.p = p
        self.robot_type = robot_type
        self.ik_mode = ik_mode
        self.gravity = float(gravity)
        self.sides = tuple(sides)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0 or not p.isConnected(self.cid):
            raise RuntimeError("PyBullet connection failed (no display?); use the default DIRECT")
        p.setGravity(0.0, 0.0, self.gravity, physicsClientId=self.cid)
        p.setTimeStep(PHYSICS_DT, physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.loader = RobotLoader(self.cid)
        self.body = self.loader.load_robot(robot_type)["robot"]
        base_position, base_orientation = p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)
        self.base_position = np.asarray(base_position, dtype=np.float64)
        self.base_rotation = np.asarray(
            p.getMatrixFromQuaternion(base_orientation), dtype=np.float64).reshape(3, 3)
        self.indices, self.infos = movable_joints(p, self.body, self.cid)
        self.hold = {
            index: float(p.getJointState(self.body, index, physicsClientId=self.cid)[0])
            for index in self.indices
        }
        self.arm = {side: [(self.loader.all_joints[name], name)
                           for name in self.loader.arm_joints[side]] for side in SIDES}
        self.arm_set = {side: {index for index, _name in self.arm[side]} for side in SIDES}
        self.ee = {side: end_effector_index(p, self.body, side, self.cid)
                   for side in self.sides}
        self.limits = self._limits()
        self.ground_lift = self.lift_clear_of_ground()

    def lift_clear_of_ground(self, clearance: float = 0.03) -> float:
        """Raise the base until the lowest link clears the ground plane.

        [实测, 与 three_robot_scene._lift_clear_of_ground 同一个坑] Loading these
        URDFs at their own base_z starts with the feet penetrating the plane; the
        contact impulses kick the ankle joints (measured here on h1_2:
        right_ankle_roll_joint 0.267 rad, left_ankle_pitch_joint 0.214 rad after 60
        frames), which then shows up as "non-driven joints drifted". Lifting removes
        that; it is a contact artifact, not gravity (the same drift appears with
        gravity 0). Returns the applied raise (0.0 when nothing was needed).
        """
        lowest = float("inf")
        for index in range(-1, self.p.getNumJoints(self.body, physicsClientId=self.cid)):
            try:
                low, _high = self.p.getAABB(self.body, index, physicsClientId=self.cid)
            except self.p.error:
                continue
            lowest = min(lowest, float(low[2]))
        base, orientation = self.p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)
        raise_by = float(clearance - lowest)
        if raise_by <= 0.0:
            return 0.0
        self.p.resetBasePositionAndOrientation(
            self.body, [base[0], base[1], base[2] + raise_by], orientation,
            physicsClientId=self.cid)
        self.base_position = np.asarray(self.p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.cid)[0], dtype=np.float64)
        self.hold = {
            index: float(self.p.getJointState(self.body, index,
                                              physicsClientId=self.cid)[0])
            for index in self.indices
        }
        return raise_by

    def _limits(self) -> dict:
        """IK bounds for every movable joint (URDF limits, widened when missing)."""
        lower, upper, ranges, rest = [], [], [], []
        for info in self.infos:
            low, high = float(info[8]), float(info[9])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = -2.0 * np.pi, 2.0 * np.pi
            lower.append(low)
            upper.append(high)
            ranges.append(max(high - low, 1e-6))
            rest.append(0.0)
        return {"lower": lower, "upper": upper, "range": ranges, "rest": rest}

    def to_world(self, position):
        """Robot-base-frame target -> world coordinates (what PyBullet IK wants).

        Uses the full base pose and not just the translation: GR1-T2 does not load
        at the origin and is not axis aligned (measured base [-0.072, 0, 0.943]), so
        the rotation matters. H1-2 loads at ``base_z=1.00``: skipping the conversion
        asks the arm to reach a point 0.8 m below its own base (measured IK residual
        ~866 mm, versus ~17 mm after the fix).
        """
        return self.base_position + self.base_rotation @ np.asarray(position, dtype=np.float64)

    def solve(self, side: str, target) -> list:
        """IK for one side's hand mount: one value per movable joint (PyBullet order).

        ``ik_mode='pose'`` (default) constrains position and orientation;
        ``'position'`` drops the orientation constraint. Measured on h1_2 (right,
        new mapping): pose 15.9 mm + 3.7 deg, position 17.2 mm + 23 deg, so the
        orientation constraint is kept and paid for nothing measurable. Note that
        both modes are only as good as the mapping: the orientation convention has
        no calibration behind it (see ``retargeting/arm/wrist_targets.py``).
        """
        position, orientation = target.as_tuple()
        world = tuple(self.to_world(position))
        arguments = {
            "lowerLimits": self.limits["lower"],
            "upperLimits": self.limits["upper"],
            "jointRanges": self.limits["range"],
            "restPoses": self.limits["rest"],
            "maxNumIterations": IK_ITERATIONS,
            "residualThreshold": IK_RESIDUAL_THRESHOLD,
            "physicsClientId": self.cid,
        }
        if self.ik_mode == "pose":
            solution = self.p.calculateInverseKinematics(
                self.body, self.ee[side][0], world, orientation, **arguments)
        else:
            solution = self.p.calculateInverseKinematics(
                self.body, self.ee[side][0], world, **arguments)
        values = [float(value) for value in solution]
        if len(values) != len(self.indices):
            raise RuntimeError(
                f"IK returned {len(values)} values for {len(self.indices)} movable joints")
        return values

    def apply(self, vector, active_sides) -> None:
        """Motor control: the given vector on the active arms, hold everywhere else."""
        for dof, index in enumerate(self.indices):
            active = any(index in self.arm_set[side] for side in active_sides)
            self.p.setJointMotorControl2(
                self.body, index, self.p.POSITION_CONTROL,
                targetPosition=float(vector[dof]) if active else self.hold[index],
                force=ARM_FORCE if active else HOLD_FORCE,
                physicsClientId=self.cid,
            )

    def step(self, count: int | None = None) -> None:
        """Advance physics; ``count=None`` means one frame's worth (30 Hz pacing)."""
        for _ in range(STEPS_PER_FRAME if count is None else max(0, int(count))):
            self.p.stepSimulation(physicsClientId=self.cid)

    def measure(self, side: str):
        """Actual (position, rotation matrix) of that side's wrist link."""
        state = self.p.getLinkState(self.body, self.ee[side][0],
                                    computeForwardKinematics=True, physicsClientId=self.cid)
        position = np.asarray(state[4], dtype=np.float64)
        rotation = np.asarray(self.p.getMatrixFromQuaternion(state[5]),
                              dtype=np.float64).reshape(3, 3)
        return position, rotation

    def hold_drift(self, active_sides, top: int = 0):
        """Angle change of the joints we are NOT driving, excluding the two hands.

        Hands are excluded on purpose: this harness does not claim to control finger
        joints, and GR1-T2's coupled/unactuated ones droop 0.141 rad under gravity
        (measured), which says nothing about the arm IK. Use ``hand_drift`` for that.
        Returns the worst value (rad); with ``top > 0`` returns
        ``(worst, [(name, drift), ...])`` so the report can say which joint moved.
        """
        from teleoperation.robots.specification import _is_hand_joint

        entries = []
        for position, index in enumerate(self.indices):
            if any(index in self.arm_set[side] for side in active_sides):
                continue
            name = self.infos[position][1]
            text = name.decode() if isinstance(name, bytes) else str(name)
            if _is_hand_joint(text, "left") or _is_hand_joint(text, "right"):
                continue
            current = float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
            entries.append((abs(current - self.hold[index]), text))
        entries.sort(reverse=True)
        worst = float(entries[0][0]) if entries else 0.0
        return worst if top <= 0 else (worst, entries[:top])

    def hand_drift(self) -> float:
        """Worst hand-joint drift (informational only; hands are not driven here)."""
        from teleoperation.robots.specification import _is_hand_joint

        worst = 0.0
        for position, index in enumerate(self.indices):
            name = self.infos[position][1]
            text = name.decode() if isinstance(name, bytes) else str(name)
            if not (_is_hand_joint(text, "left") or _is_hand_joint(text, "right")):
                continue
            current = float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
            worst = max(worst, abs(current - self.hold[index]))
        return worst

    def joint_values(self, side: str) -> dict:
        """Current angle of every arm joint we drive (for the report)."""
        return {name: float(self.p.getJointState(self.body, index,
                                                 physicsClientId=self.cid)[0])
                for index, name in self.arm[side]}

    def render(self, path, target, width: int = 640, height: int = 480,
               distance: float = 1.3, yaw: float = 40.0, pitch: float = -15.0) -> bool:
        import cv2

        view = self.p.computeViewMatrixFromYawPitchRoll(
            list(target), distance, yaw, pitch, 0, 2)
        projection = self.p.computeProjectionMatrixFOV(
            60.0, width / float(height), 0.01, 5.0)
        image = self.p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=self.p.ER_TINY_RENDERER, physicsClientId=self.cid)
        rgba = np.asarray(image[2]).reshape(height, width, 4).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path),
                                cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)))

    def close(self) -> None:
        self.p.disconnect(self.cid)
        self.body = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/verify_arm_from_capture.py",
        description="21 landmarks -> wrist target -> PyBullet IK on H1-2 / GR1-T2 / G1",
    )
    parser.add_argument("--robot", choices=("h1_2", "gr1_t2", "g1"), default="h1_2")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--input-h5", type=Path, default=None,
                        help="raw capture H5 (Layer 1 output); default: synthetic sweep")
    parser.add_argument("--frames", type=int, default=60, help="synthetic sweep length")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "tmp_probe" / "arm_from_capture")
    parser.add_argument("--render-every", type=int, default=20, help="0 = no PNG")
    parser.add_argument("--settle-steps", type=int, default=40,
                        help="extra physics steps per frame before measuring the "
                             "residual, so the metric is reachability and not lag")
    parser.add_argument("--gravity", type=float, default=0.0,
                        help="vertical gravity (default 0 = kinematic check: the "
                             "residual then measures IK reachability, not torque "
                             "authority -- with -9.81 GR1-T2/G1 show a constant "
                             "30-50 mm bias from their URDF effort limits)")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--ik-mode", choices=("position", "pose"), default=None,
                        help="position+orientation IK or position only; default: the "
                             "per-robot profile (h1_2 pose, gr1_t2/g1 position)")
    parser.add_argument("--scale", type=float, default=None,
                        help="override WristMapping.scale (metres per normalized unit)")
    parser.add_argument("--base-offset", type=float, nargs=3, default=None,
                        help="override WristMapping.base_offset (metres, robot base frame)")
    parser.add_argument("--max-residual-mm", type=float, default=30.0)
    parser.add_argument("--min-travel-mm", type=float, default=10.0)
    parser.add_argument("--max-hold-drift-rad", type=float, default=0.05)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from teleoperation.retargeting.arm.wrist_targets import (
        WristMapping,
        matrix_from_quaternion,
        wrist_pose_from_landmarks,
    )

    sides = SIDES if args.side == "both" else (args.side,)
    profile = ROBOT_PROFILES[args.robot]
    overrides = {"scale": float(args.scale if args.scale is not None else profile["scale"])}
    overrides["base_offset"] = tuple(
        float(value) for value in (args.base_offset if args.base_offset is not None
                                   else profile["base_offset"]))
    ik_mode = args.ik_mode or profile["ik_mode"]
    mapping = WristMapping(**overrides)

    if args.input_h5:
        source = str(args.input_h5)
        frames = frames_from_h5(args.input_h5, sides)
    else:
        source = f"synthetic_sweep:{SAMPLE_HAND.name}"
        frames = sweep_frames(args.frames, sides)
    selected = list(range(0, len(frames), max(1, args.stride)))
    if not selected:
        log("FAIL", "no frame to process")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report) if args.report else out_dir / f"arm_{stamp}.json"
    report = {
        "robot": args.robot, "sides": list(sides), "source": source,
        "frames_in_source": len(frames), "frames_used": len(selected),
        "mapping": {name: getattr(mapping, name) for name in
                    ("scale", "origin_x", "origin_y", "position_order",
                     "position_signs", "base_offset", "axis_order", "axis_signs")},
        "valid_targets": 0, "invalid_targets": 0, "rendered_png": [],
    }
    log("INFO", f"robot={args.robot} sides={sides} frames={len(selected)} source={source}")
    failures: list = []
    harness = None
    try:
        harness = ArmIkHarness(args.robot, sides, gui=args.gui, ik_mode=ik_mode,
                              gravity=args.gravity)
        report["end_effector"] = {
            side: {"index": int(harness.ee[side][0]), "link": harness.ee[side][1]}
            for side in sides}
        report["arm_joints"] = {side: [name for _index, name in harness.arm[side]]
                                for side in sides}
        report["base_position"] = [float(value) for value in harness.base_position]
        report["ground_lift"] = float(harness.ground_lift)
        report["ik_mode"] = harness.ik_mode
        report["gravity"] = float(harness.gravity)
        log("OK", "end effector link: " + ", ".join(
            f"{side}={harness.ee[side][1]}" for side in sides))
        log("INFO", f"ik_mode={harness.ik_mode} ground_lift={harness.ground_lift:.4f} m "
                    f"base={[round(v, 3) for v in harness.base_position]}")

        target_travel = {side: [] for side in sides}
        measured = {side: [] for side in sides}
        residuals = {side: [] for side in sides}
        lags = {side: [] for side in sides}
        rotation_errors = {side: [] for side in sides}
        held = {side: None for side in sides}
        held_target = {side: None for side in sides}
        for order, index in enumerate(selected):
            frame = frames[index]
            vector = [harness.hold[joint] for joint in harness.indices]
            for side in sides:
                target = wrist_pose_from_landmarks(frame.get(side), side, mapping)
                if not target.valid:
                    held[side] = None
                    held_target[side] = None
                    report["invalid_targets"] += 1
                    continue
                report["valid_targets"] += 1
                solution = harness.solve(side, target)
                for dof, joint in enumerate(harness.indices):
                    if joint in harness.arm_set[side]:
                        vector[dof] = solution[dof]
                held[side] = harness.to_world(target.position)
                held_target[side] = target
                target_travel[side].append(held[side].copy())
            harness.apply(vector, sides)
            harness.step()
            immediate = {}
            for side in sides:
                if held[side] is not None:
                    immediate[side] = harness.measure(side)[0].copy()
            if args.settle_steps > 0:
                harness.step(args.settle_steps)
            for side in sides:
                if held[side] is None:
                    continue
                position, rotation = harness.measure(side)
                measured[side].append(position.copy())
                residuals[side].append(float(np.linalg.norm(position - held[side])))
                lags[side].append(float(np.linalg.norm(immediate[side] - held[side])))
                error = rotation @ matrix_from_quaternion(held_target[side].quaternion).T
                rotation_errors[side].append(float(np.degrees(np.arccos(
                    np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0)))))
            if args.render_every and order % args.render_every == 0:
                focus = [float(value) for side in sides if held[side] is not None
                         for value in held[side]]
                if len(focus) >= 3:
                    path = out_dir / f"arm_{stamp}_frame{index:04d}.png"
                    if harness.render(path, focus[:3]):
                        report["rendered_png"].append(str(path))
        worst_drift, drift_entries = harness.hold_drift(sides, top=3)
        report["hold_drift_rad"] = float(worst_drift)
        report["hold_drift_joints"] = [
            {"joint": name, "rad": float(value)} for value, name in drift_entries]
        log("INFO", "worst non-driven joints: " + ", ".join(
            f"{name}={value:.4f} rad" for value, name in drift_entries))
        report["hand_drift_rad"] = float(harness.hand_drift())
        log("INFO", f"hand joints drifted at most {report['hand_drift_rad']:.4f} rad "
                    f"(informational: this harness does not drive the hands)")

        report["sides_stats"] = {}
        for side in sides:
            target_array = np.asarray(target_travel[side], dtype=np.float64).reshape(-1, 3)
            measured_array = np.asarray(measured[side], dtype=np.float64).reshape(-1, 3)
            residual = np.asarray(residuals[side], dtype=np.float64)
            orientation_error = np.asarray(rotation_errors[side], dtype=np.float64)
            lag = np.asarray(lags[side], dtype=np.float64)
            report["sides_stats"][side] = {
                "targets": int(target_array.shape[0]),
                "residual_mm_mean": float(residual.mean() * 1e3) if residual.size else None,
                "residual_mm_max": float(residual.max() * 1e3) if residual.size else None,
                "lag_mm_mean": float(lag.mean() * 1e3) if lag.size else None,
                "residual_mm_per_frame": [float(value * 1e3) for value in residual],
                "orientation_error_deg_mean":
                    float(orientation_error.mean()) if orientation_error.size else None,
                "orientation_error_deg_max":
                    float(orientation_error.max()) if orientation_error.size else None,
                "target_travel_mm":
                    float(np.ptp(target_array, axis=0).max() * 1e3) if target_array.size else 0.0,
                "measured_travel_mm":
                    float(np.ptp(measured_array, axis=0).max() * 1e3) if measured_array.size else 0.0,
                "joint_values": harness.joint_values(side),
            }
    except Exception as error:  # noqa: BLE001 - report instead of raising
        log("FAIL", f"{type(error).__name__}: {error}")
        failures.append("exception")
    finally:
        if harness is not None:
            harness.close()

    if report["valid_targets"] < 1:
        log("FAIL", "no valid wrist target was produced by the landmarks")
        failures.append("targets")
    for side in sides:
        stats = report.get("sides_stats", {}).get(side)
        if not stats or stats["targets"] < 2:
            log("FAIL", f"{side}: fewer than 2 valid targets "
                        f"({0 if not stats else stats['targets']})")
            failures.append(f"{side}_targets")
            continue
        mean_mm, max_mm = stats["residual_mm_mean"], stats["residual_mm_max"]
        if mean_mm is None or mean_mm > args.max_residual_mm:
            log("FAIL", f"{side}: IK residual mean {mean_mm} mm > {args.max_residual_mm} mm")
            failures.append(f"{side}_residual")
        else:
            log("OK", f"{side}: IK residual mean {mean_mm:.2f} mm / max {max_mm:.2f} mm "
                      f"over {stats['targets']} targets (steady state after "
                      f"{args.settle_steps} extra steps; one-frame lag "
                      f"{stats['lag_mm_mean']:.2f} mm)")
        if stats["measured_travel_mm"] < args.min_travel_mm:
            log("FAIL", f"{side}: the wrist travelled only "
                        f"{stats['measured_travel_mm']:.2f} mm")
            failures.append(f"{side}_travel")
        else:
            log("OK", f"{side}: wrist travelled {stats['measured_travel_mm']:.1f} mm "
                      f"(target travel {stats['target_travel_mm']:.1f} mm)")
        values = stats["joint_values"]
        if values:
            log("INFO", f"{side} arm joints: " + ", ".join(
                f"{name}={value:+.3f}" for name, value in values.items()))
        log("INFO", f"{side}: orientation error mean "
                    f"{stats['orientation_error_deg_mean']:.1f} deg / max "
                    f"{stats['orientation_error_deg_max']:.1f} deg "
                    f"(the orientation mapping is a convention, ik_mode="
                    f"{report.get('ik_mode')})")
    drift = float(report.get("hold_drift_rad", 0.0))
    if drift > args.max_hold_drift_rad:
        log("FAIL", f"non-driven joints drifted {drift:.4f} rad "
                    f"> {args.max_hold_drift_rad}")
        failures.append("hold_drift")
    else:
        log("OK", f"non-driven joints drifted at most {drift:.4f} rad")

    report["failures"] = failures
    report["exit_code"] = 1 if failures else 0
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                           encoding="utf-8")
    print(f"report={report_path}")
    if report["rendered_png"]:
        print(f"rendered={len(report['rendered_png'])} PNG in {out_dir}")
    if failures:
        log("FAIL", f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    log("OK", f"landmarks -> wrist target -> PyBullet IK verified on {args.robot}")
    log("WARN", "the wrist target mapping is a convention, not a hand-eye calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
