"""转手腕的时候，**谁**在动？—— ``--arm-orientation-scope`` 的判据。

用法::

    E:\\python3.11.7\\python.exe devtools\\probe_wrist_twist.py [--scope wrist|arm]
        [--robot h1_2] [--side both] [--twist 45] [--json out.json]

为什么（2026-11-16 现场反馈）：

掌面基修好之后，``--arm-orientation on`` 第一次真的把腕的朝向送进 IK，操作者转手腕，
机器人腕子跟着转 —— 但**大臂和小臂也一起动了**，而同一份代码 ``--arm-orientation off``
时臂形很好。这不是姿态目标算错了，是「谁来兑现这个姿态」的问题：PyBullet 的
``calculateInverseKinematics`` 只给了限位 + ``restPoses`` 两样软约束，7 个自由度会一起
摊这个朝向误差，肩/肘分到的那部分在画面上就是「机械臂自己动了」。真机转腕只用腕关节，
所以 ``PyBulletArmIK(orientation_scope="wrist")`` 把腕以前的关节钉在第一遍位置解上 ——
**光钉不住**（实测：7 个关节全钉在 +-1e-6 的窗口里再解一个 45 度朝向，返回的解照样偏离
4.2-27.1 度），所以求解之后还要把那些关节的值**抄回去**（``arm_ik._hold``）。

这个探针就量这件事，**纯运动学**（把解写回关节再读 FK，不走物理、不打电话给电机，
所以数字只反映求解器，与重力/增益无关）：

``baseline``  位置解 + ``orientation`` = 中性（单位旋转）那一帧的关节角与腕坐标系；
每个 ``twist`` 换一个轴向/角度，解一遍，量：

* ``proximal_deg`` —— 肩/肘（腕以前那些关节）里头最大的关节变化 = **不该有的臂动**；
* ``wrist_deg``    —— 腕关节里最大的关节变化 = 真正该有的转动；
* ``elbow_mm`` / ``upper_arm_deg`` / ``forearm_deg`` —— 肘点位移、上臂与前臂方向变化；
* ``achieved_deg`` —— 腕坐标系相对 baseline **实际**转了多少（目标角就写在 ``twist_deg``）；
* ``ori_err_deg``  / ``residual_mm`` —— 姿态误差与腕位置残差（钉住关节有没有把解弄坏）。

判据：``scope=wrist`` 时 ``proximal_deg`` 应当接近 0（肩/肘已经被抄回位置解），
同时 ``achieved_deg`` 不该比 ``scope=arm`` 掉太多 —— 掉太多说明腕只有 3 个自由度，
兑现不了这个朝向，那就得回头谈判据/口径，而不是把整条臂交出去。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: 默认试的三个轴向（世界系）+ 角度：转腕（前臂轴）、抬腕、摆手各来一遍。
DEFAULT_TWIST_DEG = 45.0
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/probe_wrist_twist.py",
        description="which joints absorb a commanded wrist orientation")
    parser.add_argument("--scope", choices=("wrist", "arm"), action="append", default=None,
                        help="orientation_scope to measure (repeatable; default: both)")
    parser.add_argument("--robot", action="append", default=None,
                        help="only these robots (repeatable; default: every loaded one)")
    parser.add_argument("--ik-mode", choices=("pose", "position"), default=None,
                        help="override the profile's ik_mode (A/B: g1 / gr1_t2 ship with "
                             "'position', i.e. their wrist never rotates)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--twist", type=float, default=DEFAULT_TWIST_DEG,
                        help=f"twist angle in degrees (default {DEFAULT_TWIST_DEG:.0f})")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    return parser


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 1e-12 else np.zeros(3)


def _angle_deg(left, right) -> float:
    dot = float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _quat_delta_deg(adapter, first, second) -> float:
    """两个四元数之间的夹角（度）：PyBullet 自己算，省得在下面对符号。"""
    difference = adapter.p.getDifferenceQuaternion(first, second)
    return float(math.degrees(2.0 * math.acos(min(1.0, abs(float(difference[3]))))))


def _pose(adapter, side: str, values) -> dict:
    """把 ``{关节名: 角度}`` 写回关节（纯运动学），读回 FK 三点 + 腕朝向。"""
    for index, name in adapter.arm[side]:
        adapter.p.resetJointState(adapter.body, index, float(values[name]),
                                  physicsClientId=adapter.cid)
    points = adapter.arm_points(side)
    position, orientation = adapter.measure(side)
    return {
        "joints": {name: float(values[name]) for _index, name in adapter.arm[side]},
        "points": {key: np.asarray(point, dtype=float) for key, point in points.items()},
        "orientation": tuple(float(value) for value in orientation),
        "wrist_m": np.asarray(position, dtype=float),
    }


def _delta(start: dict, stop: dict) -> dict:
    return {
        "elbow_mm": round(1000.0 * float(np.linalg.norm(
            stop["points"]["elbow"] - start["points"]["elbow"])), 2),
        "wrist_mm": round(1000.0 * float(np.linalg.norm(
            stop["points"]["wrist"] - start["points"]["wrist"])), 2),
        "upper_arm_deg": round(_angle_deg(
            start["points"]["upper"] - start["points"]["shoulder"],
            stop["points"]["upper"] - stop["points"]["shoulder"]), 2),
        "forearm_deg": round(_angle_deg(
            start["points"]["wrist"] - start["points"]["elbow"],
            stop["points"]["wrist"] - stop["points"]["elbow"]), 2),
    }

#: 「不需要动」的关节（腕以前）最多允许转多少度：``orientation_scope="wrist"`` 会把它们抄回
#: 位置解（``arm_ik._hold``），理论上是 0；判据给 1 度，免得浮点噪声误伤。
PROXIMAL_OK_DEG = 1.0
IDENTITY = (0.0, 0.0, 0.0, 1.0)
#: baseline 用真实路径（solve -> apply -> step）伺候到中性目标的帧数：不够的话量到的
#: 是「冷启动残差」，会把 deltas 淹掉。
SETTLE_FRAMES = 30


def measure_side(adapter, scene, side: str, profile, scope: str,
                 twist_deg: float) -> dict:
    """一个机器人一侧：baseline + 六个 twist。

    baseline 走**真实路径**（``solve`` -> ``apply`` -> ``scene.step()``）站稳到中性目标；
    之后的每个 twist 都从 baseline 的关节角出发（``resetJointState`` 写回），解一遍后
    也只是把解写回关节读 FK —— 这样两次测量之间只有「目标朝向」变了，deltas 才是求解器
    的行为，而不是上一帧姿势的残留。
    """
    wrist_joints = [name for _index, name in adapter.wrist[side]]
    proximal = [name for _index, name in adapter.arm[side] if name not in wrist_joints]
    shoulder = [name for name in proximal if "shoulder" in name.lower()]
    elbow = [name for name in proximal if "elbow" in name.lower()]
    target = tuple(float(value) for value in profile.realtime_neutral(side))
    adapter.orientation_scope = scope

    # baseline = 位置解 + 「与中性同向」的姿态目标：之后每个 twist 都是相对它的旋转。
    for _frame in range(SETTLE_FRAMES):
        adapter.apply(side, adapter.solve(side, target, source="base",
                                          orientation=IDENTITY,
                                          orientation_frame="neutral"))
        scene.step()
    resting = adapter.joint_values(side)
    baseline = _pose(adapter, side, resting)
    baseline["ori_err_deg"] = round(float(adapter.orientation_error_deg(side)), 2)
    baseline["residual_mm"] = round(float(adapter.residual_mm(side)), 2)

    variants = []
    for axis_name, axis in AXES.items():
        for sign in (1, -1):
            deg = sign * twist_deg
            commanded = adapter.p.getQuaternionFromAxisAngle(axis, math.radians(deg))
            _pose(adapter, side, resting)
            values = adapter.solve(side, target, source="base", orientation=commanded,
                                   orientation_frame="neutral")
            pose = _pose(adapter, side, values)
            joint_delta = {name: round(math.degrees(pose["joints"][name]
                                                    - baseline["joints"][name]), 2)
                           for _index, name in adapter.arm[side]}
            entry = {
                "axis": axis_name,
                "twist_deg": round(deg, 1),
                "achieved_deg": round(_quat_delta_deg(adapter, baseline["orientation"],
                                                     pose["orientation"]), 2),
                "proximal_deg": round(max((abs(joint_delta[name]) for name in proximal),
                                          default=0.0), 2),
                "shoulder_deg": round(max((abs(joint_delta[name]) for name in shoulder),
                                          default=0.0), 2),
                "elbow_deg": round(max((abs(joint_delta[name]) for name in elbow),
                                       default=0.0), 2),
                "wrist_deg": round(max((abs(joint_delta[name]) for name in wrist_joints),
                                       default=0.0), 2),
                "joint_delta_deg": joint_delta,
                "ori_err_deg": round(float(adapter.orientation_error_deg(side)), 2),
                "residual_mm": round(float(adapter.residual_mm(side)), 2),
            }
            entry.update(_delta(baseline, pose))
            variants.append(entry)
    _pose(adapter, side, resting)
    limits = {}
    for index, name in adapter.wrist[side]:
        position = adapter.position_of[index]
        limits[name] = {"lower": round(float(adapter.limits["lower"][position]), 4),
                        "upper": round(float(adapter.limits["upper"][position]), 4),
                        "load": round(float(adapter.hold[index]), 4)}
    return {"scope": scope, "wrist_joints": wrist_joints, "proximal_joints": proximal,
            "wrist_limits": limits,
            "baseline": {"ori_err_deg": baseline["ori_err_deg"],
                         "residual_mm": baseline["residual_mm"]},
            "twists": variants}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import SIDES, loadable_robots, profile_for
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    robots = tuple(args.robot) if args.robot else tuple(loadable_robots())
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    scopes = tuple(args.scope) if args.scope else ("wrist", "arm")
    scene = ThreeRobotHandScene(sides=sides, gui=False, steps_per_frame=1)
    report = {"mode": "probe_wrist_twist", "twist_deg": float(args.twist),
              "scopes": list(scopes), "robots": {}}
    worst = {scope: 0.0 for scope in scopes}
    try:
        driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                ik_iterations=200, residual_threshold=1e-5,
                                lock_others=True, ik_mode=args.ik_mode)
        for name in robots:
            adapter = driver.ik[name]
            profile = profile_for(name)
            entry = {"ik_mode": adapter.ik_mode, "sides": {},
                     "orientation_joints": {side: adapter.orientation_joints(side)
                                            for side in sides}}
            report["robots"][name] = entry
            for side in sides:
                measured = {scope: measure_side(adapter, scene, side, profile, scope,
                                                args.twist)
                            for scope in scopes}
                entry["sides"][side] = measured
                for scope, data in measured.items():
                    worst[scope] = max(worst[scope],
                                       max((twist["proximal_deg"]
                                            for twist in data["twists"]), default=0.0))
                    print(f"[{name:<6}] {side:<5} scope={scope:<5} "
                          f"wrist joints: {'/'.join(data['wrist_joints']) or 'none'} | "
                          f"baseline ori err {data['baseline']['ori_err_deg']:.1f} deg, "
                          f"wrist residual {data['baseline']['residual_mm']:.1f} mm",
                          flush=True)
                    print("    limits: " + ", ".join(
                        f"{joint} [{entry_['lower']:.2f}, {entry_['upper']:.2f}] "
                        f"load {entry_['load']:.2f}"
                        for joint, entry_ in data["wrist_limits"].items()), flush=True)
                    for twist in data["twists"]:
                        print(f"    twist {twist['axis']}{twist['twist_deg']:+6.1f} deg -> "
                              f"shoulder {twist['shoulder_deg']:6.2f} / elbow "
                              f"{twist['elbow_deg']:6.2f} / wrist joints "
                              f"{twist['wrist_deg']:6.2f} deg, "
                              f"achieved {twist['achieved_deg']:6.2f} deg | "
                              f"elbow point {twist['elbow_mm']:6.2f} mm, "
                              f"upper arm {twist['upper_arm_deg']:5.2f} deg, "
                              f"forearm {twist['forearm_deg']:5.2f} deg | "
                              f"ori err {twist['ori_err_deg']:5.1f} deg, "
                              f"wrist residual {twist['residual_mm']:5.1f} mm", flush=True)
    finally:
        scene.close()
    report["worst_proximal_deg"] = {scope: round(value, 3)
                                    for scope, value in worst.items()}
    print("", flush=True)
    for scope in scopes:
        verdict = ("ok: the arm does not move" if worst[scope] <= PROXIMAL_OK_DEG
                   else "arm moves: the whole chain is absorbing the twist")
        print(f"scope={scope:<5} worst shoulder/elbow motion {worst[scope]:6.2f} deg -> "
              f"{verdict}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
