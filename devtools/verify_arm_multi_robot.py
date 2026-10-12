"""手臂多机器人验收：robot_type -> 对应手臂 -> 关节真的跟上（不接摄像头）。

这和 ``devtools/verify_arm_from_capture.py`` 的分工
---------------------------------------------------
那个脚本从 21 点出发，验的是「手腕目标 -> 手臂」这一段（单台机器人、重力可关、用
手眼约定给的姿态）。本脚本验的是**仿真侧按 robot_type 选臂**这一段：三台机器人同时
在场、重力 -9.81、其它关节被 hold 住、按各自 profile 的幅度整圈扫掠，看每台手臂是不是
真的跟得上、行程对不对、不该动的关节有没有动。

判据（任一不过即 [FAIL]，退出码 1）
-----------------------------------
1) 每台机器人：该侧手臂解析出 7 个关节（名字按 robots/specification 的规则），
   末端 link 与 profile 的 ee 关键字选中一致
2) 稳态残差 mean <= profile.max_residual_mm（冷启动前几帧另列，见 --settle-skip）
3) 手腕行程 >= profile.min_travel_mm，且与目标行程同量级（差异 <= --max-travel-gap-mm）
4) 未被驱动的关节（腿 / 腰 / 头）漂移 <= --max-hold-drift-rad（默认 0.02 rad）；
   手指里没被映射覆盖的关节单列 hand_drift_rad（重力下垂，属于手部子系统）

用法::

    E:\\python3.11.7\\python.exe devtools\\verify_arm_multi_robot.py
    E:\\python3.11.7\\python.exe devtools\\verify_arm_multi_robot.py --robot g1 --side left
    E:\\python3.11.7\\python.exe devtools\\verify_arm_multi_robot.py --list-profiles
"""

from __future__ import annotations

import argparse
import json
import sys
from math import pi
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FRAMES = 40
DEFAULT_SETTLE_SKIP = 5


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/verify_arm_multi_robot.py",
        description="robot_type -> arm joints -> PyBullet IK, on all three paper robots")
    parser.add_argument("--robot", action="append", default=None,
                        help="only these robots (repeatable; default: all loaded)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="right")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames per sweep (one full loop of the profile's box)")
    parser.add_argument("--settle-skip", type=int, default=DEFAULT_SETTLE_SKIP,
                        help="frames ignored when averaging (the arm is still servoing)")
    parser.add_argument("--max-hold-drift-rad", type=float, default=0.02)
    parser.add_argument("--max-travel-gap-mm", type=float, default=10.0,
                        help="allowed |wrist travel - target travel|")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def sweep_side(adapter, scene, side: str, frames: int, settle_skip: int) -> dict:
    """One full loop of this profile's box for one side -> residual / travel numbers.

    The arm is *not* reset first: this is what the realtime loop sees, so the cold
    frames are reported separately instead of being hidden by a warm start.
    """
    from teleoperation.simulation.arm_scene_driver import sweep_target

    profile = adapter.profile
    frames = max(2, int(frames))
    residuals, measured, targets = [], [], []
    for frame in range(frames):
        target = sweep_target(profile, profile.sweep_amplitude_m,
                              2.0 * pi * frame / frames, "base")
        adapter.apply(side, adapter.solve(side, target, source="base"))
        scene.step()
        residuals.append(adapter.residual_mm(side))
        measured.append(np.asarray(adapter.measure(side)[0]))
        targets.append(np.asarray(adapter.last_target_world[side]))
    skip = max(0, min(int(settle_skip), frames - 2))

    def span_mm(points) -> float:
        array = np.asarray(points[skip:], dtype=float).reshape(-1, 3)
        return 0.0 if array.size == 0 else float(np.ptp(array, axis=0).max() * 1e3)

    steady = np.asarray(residuals[skip:], dtype=float)
    return {
        "residual_mm_cold": round(float(np.max(residuals)), 2),
        "residual_mm_mean": round(float(steady.mean()), 2),
        "residual_mm_max": round(float(steady.max()), 2),
        "travel_mm": round(span_mm(measured), 2),
        "target_travel_mm": round(span_mm(targets), 2),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_profiles:
        from teleoperation.simulation.arm_profiles import table

        for name, entry in table().items():
            state = "loadable" if entry["loadable"] else "registered only"
            print(f"{name:<8} {state:<16} ik_mode={entry['ik_mode']:<9} "
                  f"ceiling={entry['max_residual_mm']} mm  {entry['notes']}")
        return 0

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import SIDES
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    sides = tuple(SIDES) if args.side == "both" else (args.side,)
    scene = ThreeRobotHandScene(sides=sides, gui=args.gui)
    failures = []
    report = {"sides": list(sides), "frames": int(args.frames),
              "settle_skip": int(args.settle_skip), "robots": {}}
    try:
        driver = ArmSceneDriver(scene, sides=sides,
                                robots=tuple(args.robot) if args.robot else None)
        for robot, adapter in sorted(driver.ik.items()):
            profile = adapter.profile
            entry = {"ik_mode": adapter.ik_mode, "sides": {}}
            report["robots"][robot] = entry
            arms = ", ".join(f"{side}:{adapter.ee[side][1]}" for side in sides)
            log("INFO", f"{robot}: ik_mode={adapter.ik_mode}, ee {arms}, "
                        f"{len(adapter.driven)} arm joints driven")
            for side in sides:
                entry["sides"][side] = {
                    "ee_link": adapter.ee[side][1],
                    "joints": [name for _index, name in adapter.arm[side]]}
                stats = sweep_side(adapter, scene, side, args.frames,
                                   args.settle_skip)
                entry["sides"][side].update(stats)
                entry["sides"][side]["joint_values"] = adapter.joint_values(side)
                ceiling = profile.max_residual_mm
                if stats["residual_mm_mean"] > ceiling:
                    log("FAIL", f"{robot}/{side}: steady residual mean "
                                f"{stats['residual_mm_mean']} mm > {ceiling} mm")
                    failures.append(f"{robot}_{side}_residual")
                else:
                    log("OK", f"{robot}/{side}: steady residual mean "
                              f"{stats['residual_mm_mean']} / max "
                              f"{stats['residual_mm_max']} mm (ceiling {ceiling} mm; "
                              f"cold-start peak {stats['residual_mm_cold']} mm)")

                if stats["travel_mm"] < profile.min_travel_mm:
                    log("FAIL", f"{robot}/{side}: wrist travelled only "
                                f"{stats['travel_mm']} mm "
                                f"(< {profile.min_travel_mm} mm)")
                    failures.append(f"{robot}_{side}_travel")
                else:
                    log("OK", f"{robot}/{side}: wrist travelled "
                              f"{stats['travel_mm']} mm "
                              f"(target {stats['target_travel_mm']} mm)")
                gap = abs(stats["travel_mm"] - stats["target_travel_mm"])
                if gap > args.max_travel_gap_mm:
                    log("FAIL", f"{robot}/{side}: travel differs from the target by "
                                f"{gap:.1f} mm (> {args.max_travel_gap_mm} mm)")
                    failures.append(f"{robot}_{side}_travel_gap")
        drift = driver.hold_drift()
        report["hold_drift"] = drift
        for robot, entry in sorted(drift.items()):
            if entry["max_drift_rad"] > args.max_hold_drift_rad:
                log("FAIL", f"{robot}: held joints drifted "
                            f"{entry['max_drift_rad']} rad ({entry['joint']})")
                failures.append(f"{robot}_hold_drift")
            else:
                log("OK", f"{robot}: held joints drifted at most "
                          f"{entry['max_drift_rad']} rad ({entry['held_joints']} "
                          f"joints; unmapped hand joints up to "
                          f"{entry['hand_drift_rad']} rad, informational)")
    except Exception as error:  # noqa: BLE001 - report instead of raising
        log("FAIL", f"{type(error).__name__}: {error}")
        failures.append("exception")
    finally:
        scene.close()

    report["failures"] = failures
    report["exit_code"] = 1 if failures else 0
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    if failures:
        log("FAIL", f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    log("OK", "robot_type -> arm joints -> PyBullet IK verified on every loaded robot")
    log("WARN", "the operator mapping is per-robot tuning, not a hand-eye calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


