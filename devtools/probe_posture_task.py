"""量一件事：把「操作者的肘角」写进 IK 的 restPoses，肘到底会不会跟着动。

为什么需要这个探针
------------------
论文的视觉接口不是「只有腕的位置」：它有 PINK 的 ``FrameTask``（腕）+ ``PostureTask``
（整条手臂的姿势参考），所以论文里人的肘弯多少、机器人的肘就弯多少。本仓库用的是
PyBullet ``calculateInverseKinematics``：腕任务有，**姿势任务没有**。PyBullet 唯一能当
姿势任务用的东西是 ``restPoses``（冗余自由度走零空间时被拉向哪个参考姿势）。

所以本探针只回答一个问题，而且是可证伪的：

    同一个腕目标、只换 restPoses 里的肘角，肘连杆会不会真的移动？腕残差会不会坏？

做法（每台机器人、每侧，互不干扰）
----------------------------------
1. 把两侧都送回中性（加载姿势），记下基线：肘关节角 + 肩/肘/腕三个 FK 点 + |腕-肩|；
2. 把腕固定在 profile 的中性工作点，跑 ``--frames`` 帧，换三个变体各跑一遍
   （每次先回中性，免得上一个变体留在零空间里的姿势串味）：
   a. ``posture=None``（现状：零空间参考 = 加载姿势）
   b. ``posture={elbow_joint: 中性肘角 + delta}``
   c. ``posture={elbow_joint: 中性肘角 - delta}``
3. 量每个变体的：肘关节角、肘连杆位移、腕连杆位移、腕残差、|腕-肩|、
   上臂方向（肩->肘单位向量）相对基线的夹角。

判据
----
- ``posture_effective``：肘角变化 >= 0.05 rad 且肘连杆位移 >= 10 mm
  （说明零空间真的被 restPoses 拉动，姿势任务这条路走得通）；
- ``residual_ok``：腕残差 <= profile.max_residual_mm（姿势参考没有把腕任务挤掉）；
- ``flexion_sign``：哪个方向的 delta 让 |腕-肩| 变短，哪个方向就是「屈肘」。
  这个符号在实现里用来把「人手肘弯了多少」映射成关节角，必须量出来而不是猜。

用法::

    E:\\python3.11.7\\python.exe devtools\\probe_posture_task.py
    E:\\python3.11.7\\python.exe devtools\\probe_posture_task.py --robot h1_2 --side left
    E:\\python3.11.7\\python.exe devtools\\probe_posture_task.py --delta 0.8 --frames 12 ^
        --report outputs\\tmp_realtime\\probe_posture_task.json

退出码：0 = 至少一台机器人量到「姿势参考能拉动肘」；1 = 一台都没有（这条路走不通）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FRAMES = 8
DEFAULT_DELTA = 0.6
DEFAULT_SETTLE = 6
#: 肘关节名里的关键字（三台机器人都含 "elbow"，见 arm_ik.ELBOW_KEYWORD）。
ELBOW_KEYWORD = "elbow"


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/probe_posture_task.py",
        description="restPoses as a PostureTask: does the elbow follow a posture hint?")
    parser.add_argument("--robot", action="append", default=None,
                        help="only these robots (repeatable; default: all loaded)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help=f"elbow bend hint, rad (default {DEFAULT_DELTA})")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames per variant (default {DEFAULT_FRAMES})")
    parser.add_argument("--settle", type=int, default=DEFAULT_SETTLE,
                        help="frames used to return to neutral between variants "
                             f"(default {DEFAULT_SETTLE})")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _direction(points, start: str, end: str) -> np.ndarray:
    """两点之间的单位向量（退化时给零向量：探针要把退化报出来，而不是抛异常）。"""
    vector = np.asarray(points[end], dtype=float) - np.asarray(points[start], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3)


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _elbow_name(adapter, side: str) -> str:
    for _index, name in adapter.arm[side]:
        if ELBOW_KEYWORD in name.lower():
            return name
    raise RuntimeError(f"{adapter.robot_type}/{side}: no elbow joint in "
                       f"{[name for _index, name in adapter.arm[side]]}")


def _sample(adapter, side: str) -> dict:
    """这一侧现在的状态：肘角、三个 FK 点、|腕-肩|。"""
    values = adapter.joint_values(side)
    points = adapter.arm_points(side)
    return {
        "elbow_joint_rad": round(float(values[_elbow_name(adapter, side)]), 5),
        "points": {key: [round(float(value), 5) for value in point]
                   for key, point in points.items()},
        "reach_mm": round(1000.0 * float(np.linalg.norm(
            np.asarray(points["wrist"], dtype=float)
            - np.asarray(points["shoulder"], dtype=float))), 2),
    }


def _settle(scene, adapter, sides, neutral, frames) -> None:
    """把每一侧都写回中性关节角，让场景走几步，回到确定的起点。"""
    for _frame in range(max(2, frames)):
        for side in sides:
            adapter.apply(side, neutral[side])
        scene.step()


def _run_variant(scene, adapter, side, sides, neutral, target, posture, frames) -> dict:
    """固定腕目标 + 给定的 posture 参考跑 frames 帧，稳态量一次。"""
    residuals = []
    for _frame in range(frames):
        adapter.apply(side, adapter.solve(side, target, source="base", posture=posture))
        for other in sides:
            if other != side:
                adapter.apply(other, neutral[other])
        scene.step()
        residuals.append(adapter.residual_mm(side))
    state = _sample(adapter, side)
    state["residual_mm"] = round(float(residuals[-1]), 2)
    state["residual_mean_mm"] = round(float(np.mean(residuals)), 2)
    return state


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import SIDES, loadable_robots, profile_for
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    robots = tuple(args.robot) if args.robot else loadable_robots()
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    scene = ThreeRobotHandScene(sides=sides, gui=False, steps_per_frame=8)
    report = {"mode": "probe_posture_task", "delta_rad": float(args.delta),
              "frames": int(args.frames), "robots": {}}
    effective = 0
    try:
        driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                ik_iterations=200, residual_threshold=1e-5)
        for name in robots:
            adapter = driver.ik[name]
            profile = profile_for(name)
            report["robots"][name] = {"ik_mode": adapter.ik_mode, "sides": {}}
            for side in sides:
                elbow = _elbow_name(adapter, side)
                neutral = {other: adapter.joint_values(other) for other in sides}
                target = tuple(float(value) for value in profile.realtime_neutral(side))
                _settle(scene, adapter, sides, neutral, args.settle)
                baseline = _sample(adapter, side)
                variants = {"none": None,
                            "plus": {elbow: neutral[side][elbow] + args.delta},
                            "minus": {elbow: neutral[side][elbow] - args.delta}}
                measured = {}
                for label in ("none", "plus", "minus"):
                    _settle(scene, adapter, sides, neutral, args.settle)
                    measured[label] = _run_variant(
                        scene, adapter, side, sides, neutral, target,
                        variants[label], max(2, args.frames))
                    measured[label]["elbow_delta_rad"] = round(
                        measured[label]["elbow_joint_rad"] - baseline["elbow_joint_rad"], 5)
                    measured[label]["elbow_mm_moved"] = round(1000.0 * float(np.linalg.norm(
                        np.asarray(measured[label]["points"]["elbow"], dtype=float)
                        - np.asarray(baseline["points"]["elbow"], dtype=float))), 2)
                    measured[label]["wrist_mm_moved"] = round(1000.0 * float(np.linalg.norm(
                        np.asarray(measured[label]["points"]["wrist"], dtype=float)
                        - np.asarray(baseline["points"]["wrist"], dtype=float))), 2)
                    measured[label]["upper_arm_deg"] = round(_angle_deg(
                        _direction(baseline["points"], "shoulder", "elbow"),
                        _direction(measured[label]["points"], "shoulder", "elbow")), 2)
                plus, minus = measured["plus"], measured["minus"]
                flexion = 0
                if plus["reach_mm"] < minus["reach_mm"] - 1.0:
                    flexion = 1
                elif minus["reach_mm"] < plus["reach_mm"] - 1.0:
                    flexion = -1
                ok_posture = any(
                    abs(measured[label]["elbow_delta_rad"]) >= 0.05
                    and measured[label]["elbow_mm_moved"] >= 10.0
                    for label in ("plus", "minus"))
                ok_residual = all(measured[label]["residual_mm"] <= profile.max_residual_mm
                                  for label in ("none", "plus", "minus"))
                report["robots"][name]["sides"][side] = {
                    "elbow_joint": elbow, "baseline": baseline, "variants": measured,
                    "flexion_sign": flexion, "posture_effective": bool(ok_posture),
                    "residual_ok": bool(ok_residual),
                    "max_residual_mm": profile.max_residual_mm}
                effective += 1 if ok_posture else 0
                log("OK" if ok_posture else "FAIL",
                    f"{name}/{side}: elbow {elbow}; +delta -> "
                    f"{plus['elbow_delta_rad']:+.3f} rad / {plus['elbow_mm_moved']:.1f} mm "
                    f"(upper arm {plus['upper_arm_deg']:.1f} deg), -delta -> "
                    f"{minus['elbow_delta_rad']:+.3f} rad / "
                    f"{minus['elbow_mm_moved']:.1f} mm; reach "
                    f"{baseline['reach_mm']:.1f} -> {plus['reach_mm']:.1f} / "
                    f"{minus['reach_mm']:.1f} mm")
                log("OK" if ok_residual else "FAIL",
                    f"{name}/{side}: wrist residual none/plus/minus = "
                    f"{measured['none']['residual_mm']:.1f} / {plus['residual_mm']:.1f} / "
                    f"{minus['residual_mm']:.1f} mm (ceiling {profile.max_residual_mm} mm); "
                    f"flexion_sign={flexion:+d}")
    except Exception as error:  # noqa: BLE001 - 探针要把原因原样报出来
        log("FAIL", f"{type(error).__name__}: {error}")
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        scene.close()

    report["posture_effective_count"] = effective
    report["exit_code"] = 0 if effective else 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    if not effective:
        log("FAIL", "no robot let the elbow follow restPoses: a PyBullet posture task "
                    "does not steer this chain")
        return 1
    log("OK", f"{effective} (robot, side) pairs let the elbow follow the posture hint")
    log("INFO", "next: map the operator elbow angle onto that joint and feed it as posture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
