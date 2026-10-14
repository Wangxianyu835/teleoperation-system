"""量一件事：只动大臂 / 张开怀抱这两种动作，**目标**到底动了多少（不含 IK）。

为什么需要它
------------
现场（2026-10-13）：「双手握拳贴胸只摆大臂」三台都没反应；「张开怀抱」只能微微张开。
``devtools/probe_elbow_task.py`` 已证明「腕目标不变时 IK 纹丝不动」（control 0.01 度），
所以问题出在**目标这一层**。本脚本把 IK 摘掉，只跑 ``RealtimeArmBridge``（实时链路里真正
决定目标的那一层），用合成手臂流对比两种口径：

``chain``    历史口径：只喂「腕 - 肩」一个向量
``segments`` 论文 scale_joints 口径：上臂段 + 前臂段（方向是你的、长度是机器人的）

两种动作（都在「手握拳贴胸」的中性姿势上做）
--------------------------------------------
``swing``  只把肘绕「肩-腕」轴摆开 90 度，手留在胸前不动 —— 现场那个「没反应」的动作。
``hug``    张开怀抱：两侧上臂外展，手跟着向两侧张开。

每个动作、每台机器人、每侧报告：你肘动了多少、你腕动了多少、机器人目标动了多少
（含横向分量与逐轴上限）、以及有多少帧被 profile 的逐轴行程夹掉（「张不够开」的候选原因）。

判据
----
- ``visible``：``segments`` 在 swing 动作下目标位移 >= 20 mm，而 chain < 5 mm。

用法::

    E:\\python3.11.7\\python.exe devtools\\verify_arm_wing.py
    E:\\python3.11.7\\python.exe devtools\\verify_arm_wing.py --report outputs\\tmp_realtime\\wing.json

退出码：0 = segments 口径在 swing 动作下真的给出非零目标（且 chain 给不出）；1 = 不是。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: 操作者中性的三点基准（米，MediaPipe pose world 口径：x 右、y 下、z 远离相机）。
#: 「双手握拳贴胸」：上臂向下略前，前臂折回胸前，|腕-肩| 约 0.27 m。
NEUTRAL_SHOULDER = {"left": np.array([-0.20, -0.30, 0.00]),
                    "right": np.array([0.20, -0.30, 0.00])}
UPPER_ARM_M = 0.30
FORE_ARM_M = 0.28
#: 中性肘方向（单位向量，操作者系）：向下、略向前外。
NEUTRAL_ELBOW_DIR = {"left": np.array([0.10, 0.94, 0.32]),
                     "right": np.array([-0.10, 0.94, 0.32])}
#: 中性前臂方向（相对肘）：向上、向内收回胸前。
NEUTRAL_WRIST_DIR = {"left": np.array([0.30, -0.80, -0.52]),
                     "right": np.array([-0.30, -0.80, -0.52])}
#: 「张开怀抱」时上臂外展的方向（单位向量，操作者系）。
HUG_ELBOW_DIR = {"left": np.array([0.62, 0.20, 0.76]),
                 "right": np.array([-0.62, 0.20, 0.76])}
STEPS = 7


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def _arm(side: str, elbow_dir=None, wrist_dir=None) -> np.ndarray:
    """合成一帧 (肩, 肘, 腕)：方向给定，长度用操作者自己的（米，操作者系）。"""
    shoulder = np.asarray(NEUTRAL_SHOULDER[side], dtype=float)
    elbow_dir = _unit(NEUTRAL_ELBOW_DIR[side] if elbow_dir is None else elbow_dir)
    wrist_dir = _unit(NEUTRAL_WRIST_DIR[side] if wrist_dir is None else wrist_dir)
    elbow = shoulder + UPPER_ARM_M * elbow_dir
    wrist = elbow + FORE_ARM_M * wrist_dir
    return np.stack((shoulder, elbow, wrist))


def _rotate_about(vector, axis, angle_rad) -> np.ndarray:
    """Rodrigues：把 ``vector`` 绕 ``axis`` 转 ``angle_rad``。"""
    axis = _unit(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rotation = (np.eye(3) + np.sin(angle_rad) * cross
                + (1.0 - np.cos(angle_rad)) * (cross @ cross))
    return rotation @ np.asarray(vector, dtype=float)


def _swing_arm(side: str, angle: float) -> np.ndarray:
    """只摆大臂：把上臂绕「肩->腕」轴转 angle，腕点不动（前臂长度自动守恒）。"""
    arm = _arm(side)
    shoulder, elbow, wrist = arm
    axis = wrist - shoulder
    return np.stack((shoulder,
                     shoulder + _rotate_about(elbow - shoulder, axis, angle),
                     wrist))


def _lean_arm(side: str, step: float) -> np.ndarray:
    """张开怀抱：上臂方向朝外展方向插值 step，前臂方向不变（手跟着出去）。"""
    arm = _arm(side)
    shoulder, elbow, wrist = arm
    direction = _unit((1.0 - step) * _unit(elbow - shoulder)
                      + step * _unit(HUG_ELBOW_DIR[side]))
    new_elbow = shoulder + UPPER_ARM_M * direction
    return np.stack((shoulder, new_elbow, new_elbow + (wrist - elbow)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/verify_arm_wing.py",
        description="operator arm stream -> bridge target, chain vs paper segments")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import (SIDES, loadable_robots,
                                                       profile_for)
    from teleoperation.simulation.arm_realtime import RealtimeArmBridge
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    robots = loadable_robots()
    sides = tuple(SIDES)
    scene = ThreeRobotHandScene(sides=sides, gui=False, steps_per_frame=1)
    report = {"mode": "verify_arm_wing", "robots": {}, "actions": ("swing", "hug")}
    visible = 0
    try:
        driver = ArmSceneDriver(scene, sides=sides, robots=robots)
        # 量每台每侧的实测连杆长（segments 口径的「长度是机器人自己的」）。
        links = {}
        for name in robots:
            adapter = driver.ik[name]
            links[name] = {}
            for side in sides:
                points = adapter.arm_points(side)
                shoulder = np.asarray(points["shoulder"], dtype=float)
                elbow = np.asarray(points["elbow"], dtype=float)
                wrist = np.asarray(points["wrist"], dtype=float)
                links[name][side] = {"upper": float(np.linalg.norm(elbow - shoulder)),
                                     "fore": float(np.linalg.norm(wrist - elbow))}
        report["robot_links_m"] = {
            name: {side: {"upper": round(value["upper"], 4),
                          "fore": round(value["fore"], 4)}
                   for side, value in per.items()}
            for name, per in links.items()}
        log("INFO", "measured link lengths: " + "; ".join(
            f"{name} " + "/".join(f"{side} {per[side]['upper']:.3f}+{per[side]['fore']:.3f}"
                                  for side in sides)
            for name, per in links.items()))
        bridge_links = {name: {side: {"upper": links[name][side]["upper"],
                                      "fore": links[name][side]["fore"]}
                               for side in sides} for name in robots}
        streams = {
            "swing": {side: [_swing_arm(side, angle)
                             for angle in np.linspace(0.0, np.pi / 2.0, STEPS)]
                      for side in sides},
            "hug": {side: [_lean_arm(side, step)
                           for step in np.linspace(0.0, 1.0, STEPS)]
                    for side in sides},
        }

        for mode in ("chain", "segments"):
            bridge = RealtimeArmBridge([profile_for(name) for name in robots],
                                       sides=sides, calibration_frames=4,
                                       map_mode=mode, robot_links=bridge_links)
            for _frame in range(4):
                bridge.update(arms_world={side: _arm(side) for side in sides})
            for action, per_side in streams.items():
                for side in sides:
                    collected = {name: [] for name in robots}
                    for frame in per_side[side]:
                        out = bridge.update(arms_world={side: frame})
                        for name in robots:
                            neutral = np.asarray(
                                profile_for(name).realtime_neutral(side), dtype=float)
                            collected[name].append(
                                np.asarray(out[name][side].position, dtype=float)
                                - neutral)
                    for name in robots:
                        stack = np.stack(collected[name], axis=0)
                        operator = np.stack(per_side[side], axis=0)
                        limit_mm = round(1000.0 * float(
                            profile_for(name).travel_limit(side)), 1)
                        entry = {
                            "operator_elbow_mm": round(1000.0 * float(np.abs(
                                operator[:, 1, :] - operator[0, 1, :]).max()), 2),
                            "operator_wrist_mm": round(1000.0 * float(np.abs(
                                operator[:, 2, :] - operator[0, 2, :]).max()), 2),
                            "target_mm": round(1000.0 * float(np.abs(stack).max()), 2),
                            "target_lateral_mm": round(1000.0 * float(
                                np.abs(stack[:, 1]).max()), 2),
                            "target_per_axis_mm": [round(1000.0 * float(value), 2)
                                                   for value in
                                                   np.abs(stack).max(axis=0)],
                            "travel_limit_mm": limit_mm,
                            # 饱和判据用 98% 判定：逐轴剪断后目标会**正好**贴在行程上限上
                            # （实测 g1 就是 40.0 mm = 它的 realtime_travel_m）。
                            "saturated_frames": int(sum(
                                1 for row in stack
                                if float(np.abs(row).max()) >= 0.98 * limit_mm)),
                        }
                        entry["ratio"] = (None if not entry["operator_wrist_mm"]
                                          else round(entry["target_mm"]
                                                     / entry["operator_wrist_mm"], 3))
                        report["robots"].setdefault(name, {}).setdefault(
                            mode, {}).setdefault(side, {})[action] = entry
                        log("INFO", f"{mode:<8} {name}/{side} {action}: operator elbow "
                                    f"{entry['operator_elbow_mm']:.0f} / wrist "
                                    f"{entry['operator_wrist_mm']:.0f} mm -> target "
                                    f"{entry['target_mm']:.1f} mm (lateral "
                                    f"{entry['target_lateral_mm']:.1f}, per-axis "
                                    f"{entry['target_per_axis_mm']}, limit "
                                    f"{limit_mm:.0f} mm, saturated "
                                    f"{entry['saturated_frames']} frames)")
        for name in robots:
            for side in sides:
                chain_swing = report["robots"][name]["chain"][side]["swing"]["target_mm"]
                seg_swing = report["robots"][name]["segments"][side]["swing"]["target_mm"]
                ok = bool(seg_swing >= 20.0 and chain_swing < 5.0)
                report["robots"][name]["segments"][side]["swing"]["visible"] = ok
                visible += 1 if ok else 0
                log("OK" if ok else "FAIL",
                    f"{name}/{side} swing: chain {chain_swing:.1f} mm / segments "
                    f"{seg_swing:.1f} mm target travel")
    except Exception as error:  # noqa: BLE001 - 验收入口要把原因原样报出来
        log("FAIL", f"{type(error).__name__}: {error}")
        traceback.print_exc()
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        scene.close()

    report["visible_count"] = visible
    report["exit_code"] = 0 if visible else 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    if not visible:
        log("FAIL", "segments mapping did not beat chain on the swing action")
        return 1
    log("OK", f"{visible} (robot, side) pairs: segments gives a target where chain "
              f"gives none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


