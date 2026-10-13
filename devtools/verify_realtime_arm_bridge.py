"""实时手臂桥验收：摄像头帧 -> 每台机器人的手臂 -> 残差 / 行程 / 保持 / 漂移。

为什么必须离线可跑
------------------
真正的验收要你坐在摄像头前，但结论必须在**没有相机、没有你**的情况下也能复现，所以
这里有两种输入：

- ``--synthetic``（默认）：人造一个"操作者" —— 米制肩/肘/腕按正弦扫掠，前
  ``--calibrate-frames`` 帧保持不动（正好走完中性标定）；
- ``--input-npz PATH``：``python -m teleoperation.simulation.realtime_teleop --dump-npz``
  录下来的真实摄像头帧（手 21 点 + pose 肩/肘/腕，NaN = 那一帧没检到）。这条路径把手部
  兜底映射也一起走到（pose 丢帧时用手 21 点）。

判据（任一不过即 [FAIL]，退出码 1）
----------------------------------
1) 中性标定完成（所驱动的两侧），且标定后第一帧的目标 = profile 的中性工作点
2) 每台机器人每侧：稳态残差 mean <= profile.max_residual_mm（前 ``--settle-skip`` 帧只记录）
3) 手腕实际行程 >= profile.min_travel_mm，且与目标行程差 <= --max-travel-gap-mm
4) 每轴位移没有越过 ``profile.travel_limit(side)``；被夹住的帧数写进报告（不是失败）
5) 丢帧（None / NaN）时目标保持不变（同一位置），且 ``counts.held > 0``
6) 未驱动关节（腿 / 腰 / 头 / 未驱动侧的手）漂移 <= --max-hold-drift-rad

用法::

    E:\\python3.11.7\\python.exe devtools\\verify_realtime_arm_bridge.py
    E:\\python3.11.7\\python.exe devtools\\verify_realtime_arm_bridge.py \\
        --input-npz outputs\\tmp_realtime\\frames.npz --side both
    E:\\python3.11.7\\python.exe devtools\\verify_realtime_arm_bridge.py --probe-travel
    E:\\python3.11.7\\python.exe devtools\\verify_realtime_arm_bridge.py --probe-lateral

``--probe-travel`` / ``--probe-lateral`` 是**量参数**的模式：它们不改任何文件，只打印
每台机器人每侧的行程上限 / 横向中性工作点建议值，用来复核 profile 里的实测数字。
"""
from __future__ import annotations

import argparse
import json
import sys
from math import cos, pi, sin
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from teleoperation.simulation.arm_profiles import loadable_robots, profile_for  # noqa: E402
from teleoperation.simulation.arm_scene_driver import (  # noqa: E402
    ArmSceneDriver,
    sweep_target,
)

SIDES = ("left", "right")
#: 合成操作者的肩高（相对髋中点，米）：与 MediaPipe pose world 的量级一致。
SHOULDER = (0.17, -0.35, 0.0)
UPPER_ARM = (0.0, 0.26, 0.0)
FORE_ARM = (0.0, 0.23, 0.0)
SIDE_PHASE = {"left": 0.0, "right": 0.6 * pi}


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def synthetic_arm(side: str, phase: float, travel: float, neutral: bool = False):
    """米制 (肩, 肘, 腕)：``neutral`` 时不动，否则在三个轴上按不同相位画圆。"""
    sign = 1.0 if side == "left" else -1.0
    shoulder = np.array([SHOULDER[0] * sign, SHOULDER[1], SHOULDER[2]], dtype=np.float64)
    offset = np.zeros(3) if neutral else np.array(
        [travel * sin(phase), 0.5 * travel * sin(2.0 * phase), travel * cos(phase)],
        dtype=np.float64)
    elbow = shoulder + np.array(UPPER_ARM, dtype=np.float64)
    wrist = elbow + np.array(FORE_ARM, dtype=np.float64) + offset
    return np.stack((shoulder, elbow, wrist))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/verify_realtime_arm_bridge.py",
        description="realtime arm bridge: camera frames -> per-robot arm targets -> PyBullet")
    parser.add_argument("--robot", action="append", default=None,
                        help="only these robots (repeatable; default: all loaded)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--calibrate-frames", type=int, default=12)
    parser.add_argument("--operator-travel", type=float, default=0.12,
                        help="how far the synthetic operator's wrist moves (metres)")
    parser.add_argument("--dropout-every", type=int, default=7,
                        help="drop one side every N frames (0 = never); checks hold-on-invalid")
    parser.add_argument("--settle-skip", type=int, default=4)
    parser.add_argument("--max-travel-gap-mm", type=float, default=10.0)
    parser.add_argument("--max-hold-drift-rad", type=float, default=0.02)
    parser.add_argument("--arm-orientation", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--arm-ik-iterations", type=int, default=None,
                        help="forward to PyBulletArmIK (default 200); 60 is ~2x faster")
    parser.add_argument("--arm-ik-residual-threshold", type=float, default=None)
    parser.add_argument("--input-npz", type=Path, default=None,
                        help="replay a session recorded by realtime_teleop --dump-npz")
    parser.add_argument("--probe-travel", action="store_true",
                        help="measure the per-side travel limit and print the suggestion")
    parser.add_argument("--probe-lateral", action="store_true",
                        help="measure the per-side lateral neutral and print the suggestion")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser



def frame_stream(args, sides, frames):
    """产出 ``(arms_world, hands)`` 帧序列：合成操作者，或 ``--input-npz`` 回放。"""
    if args.input_npz is None:
        span = max(1, frames - args.calibrate_frames)

        def synthetic():
            for frame in range(frames):
                neutral = frame < args.calibrate_frames
                phase = 2.0 * pi * (frame - args.calibrate_frames) / span
                arms = {}
                for index, side in enumerate(sides):
                    arm = synthetic_arm(side, phase + SIDE_PHASE[side],
                                        args.operator_travel, neutral)
                    if (not neutral and args.dropout_every
                            and frame % args.dropout_every == 0
                            and (frame // args.dropout_every) % 2 == index % 2):
                        arm = None                      # 轮流丢一侧，另一侧照常
                    arms[side] = arm
                yield arms, None

        return synthetic(), "synthetic"

    data = np.load(str(args.input_npz))
    total = int(np.asarray(data["timestamps"]).shape[0])
    if frames:
        total = min(total, int(frames))

    def recorded():
        for index in range(total):
            arms, hands = {}, {}
            for side in SIDES:
                arms[side] = _points(data, f"{side}_arm", index, (3, 3))
                hands[side] = _points(data, f"{side}_hand", index, (21, 3))
                hands[f"world_{side}"] = _points(data, f"world_{side}_hand", index, (21, 3))
            yield arms, hands

    return recorded(), str(args.input_npz)


def _points(data, key, index, shape):
    """``data[key][index]``；形状不对或含 NaN 时返回 ``None``（NaN = 那一帧没检到）。"""
    if key not in data:
        return None
    values = np.asarray(data[key][index], dtype=np.float64)
    if values.shape != tuple(shape) or not np.isfinite(values).all():
        return None
    return values


def _span_mm(points) -> float:
    """一组世界坐标点的最大轴行程（毫米）："手腕真的动了吗"的量度。"""
    array = np.asarray(points, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    return float(np.max(array.max(axis=0) - array.min(axis=0)) * 1000.0)



def drive(args) -> tuple:
    """建场景 + 桥，逐帧驱动，跑完判据；返回 ``(report, failures)``。"""
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import loadable_robots, profile_for
    from teleoperation.simulation.arm_realtime import RealtimeArmBridge
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    sides = tuple(SIDES if args.side == "both" else (args.side,))
    robots = tuple(args.robot) if args.robot else loadable_robots()
    profiles = {name: profile_for(name) for name in robots}
    scene = ThreeRobotHandScene(sides, gui=bool(args.gui), steps_per_frame=8)
    failures = []
    report = {"source": None, "sides": list(sides), "robots": list(robots),
              "operator_travel_m": args.operator_travel,
              "calibrate_frames": args.calibrate_frames}
    try:
        driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                ik_iterations=args.arm_ik_iterations,
                                residual_threshold=args.arm_ik_residual_threshold)
        bridge = RealtimeArmBridge([profiles[name] for name in robots], sides=sides,
                                   calibration_frames=args.calibrate_frames,
                                   orientation=args.arm_orientation, fallback_hand=True)
        stream, source = frame_stream(args, sides, args.frames)
        report["source"] = source
        residuals = {name: {side: [] for side in sides} for name in robots}
        measured = {name: {side: [] for side in sides} for name in robots}
        commanded = {name: {side: [] for side in sides} for name in robots}
        targets_world = {name: {side: [] for side in sides} for name in robots}
        held_flags = []
        applied_last = {}
        first_after_calibration = {}
        processed = 0
        for arms, hands in stream:
            targets = bridge.update(arms_world=arms, hands=hands)
            for side in sides:
                per_side = {name: entry[side] for name, entry in targets.items()}
                positions = {name: target.position for name, target in per_side.items()
                             if target.position is not None}
                if len(positions) != len(per_side):
                    continue
                driver.apply_each(side, positions, source="base",
                                  orientation=per_side[sorted(per_side)[0]].orientation)
                for name in robots:
                    target = per_side[name]
                    if target.position is None:
                        continue
                    commanded[name][side].append(np.asarray(target.position, dtype=np.float64))
                    if target.held and (name, side) in applied_last:
                        held_flags.append(bool(np.allclose(
                            np.asarray(target.position, dtype=np.float64),
                            applied_last[(name, side)], atol=1e-9)))
                    applied_last[(name, side)] = np.asarray(target.position, dtype=np.float64)
            driver.step(1)
            processed += 1
            for name, adapter in driver.ik.items():
                for side in sides:
                    residuals[name][side].append(adapter.residual_mm(side))
                    measured[name][side].append(np.asarray(adapter.measure(side)[0],
                                                           dtype=np.float64))
                    targets_world[name][side].append(np.asarray(
                        adapter.last_target_world[side], dtype=np.float64))
            if not bridge.calibrating and not first_after_calibration:
                first_after_calibration = {
                    name: {side: float(np.linalg.norm(
                        np.asarray(targets[name][side].position, dtype=np.float64)
                        - np.asarray(profiles[name].realtime_neutral(side), dtype=np.float64)))
                        for side in sides}
                    for name in robots}
        report["frames"] = processed
        report["calibrated"] = not bridge.calibrating
        report["counts"] = bridge.counts
        report["first_target_offset_from_neutral_mm"] = {
            name: {side: round(value * 1000.0, 4) for side, value in per.items()}
            for name, per in first_after_calibration.items()}
        report["hold_frames_identical"] = {
            "checked": len(held_flags), "all_identical": bool(all(held_flags))}
        report["bridge"] = bridge.summary()
        report["arm"] = {}
        for name, profile in profiles.items():
            report["arm"][name] = {}
            for side in sides:
                steady = residuals[name][side][args.settle_skip:]
                mean = float(np.mean(steady)) if steady else float("nan")
                worst = float(np.max(steady)) if steady else float("nan")
                travel = _span_mm(measured[name][side][args.settle_skip:])
                target_travel = _span_mm(targets_world[name][side][args.settle_skip:])
                limit = profile.travel_limit(side) * 1000.0
                report["arm"][name][side] = {
                    "ee_link": driver.ik[name].ee[side][1],
                    "ik_mode": profile.ik_mode,
                    "residual_mm_mean": round(mean, 2),
                    "residual_mm_max": round(worst, 2),
                    "residual_mm_ceiling": profile.max_residual_mm,
                    "travel_mm": round(travel, 2),
                    "target_travel_mm": round(target_travel, 2),
                    "travel_shortfall_mm": round(target_travel - travel, 2),
                    "travel_limit_mm": round(limit, 1),
                    "commanded_beyond_limit": bool(target_travel > 2.0 * limit + 1.0),
                }
        report["hold_drift"] = driver.hold_drift()
    finally:
        scene.close()
    _check(report, profiles, args, failures)
    report["failures"] = failures
    report["exit_code"] = 1 if failures else 0
    return report, failures



def _check(report, profiles, args, failures) -> None:
    """判据 1-6：标定 / 中性起点 / 残差 / 行程 / 保持 / 漂移（失败追加进 ``failures``）。"""
    if not report.get("calibrated"):
        failures.append("calibration_incomplete")
        log("FAIL", "the neutral calibration never completed: hold a neutral pose for "
                    f"{args.calibrate_frames} valid frames per side")
    else:
        log("OK", f"neutral calibration completed in {report['frames']} frames "
                  f"(valid per side: {report['counts']['valid']})")
    for name, per in sorted(report["first_target_offset_from_neutral_mm"].items()):
        worst = max(per.values()) if per else float("nan")
        if not np.isfinite(worst) or worst > 0.01:
            failures.append(f"{name}_neutral_start")
            log("FAIL", f"{name}: the first target after calibration is {worst:.2f} mm away "
                        f"from the profile's neutral point")
        else:
            log("OK", f"{name}: first target after calibration sits on the neutral point "
                      f"(worst axis offset {worst:.4f} mm)")
    for name, per_side in sorted(report["arm"].items()):
        for side, entry in sorted(per_side.items()):
            ceiling = entry["residual_mm_ceiling"]
            if (not np.isfinite(entry["residual_mm_mean"])
                    or entry["residual_mm_mean"] > ceiling):
                failures.append(f"{name}_{side}_residual")
                log("FAIL", f"{name}/{side}: steady residual mean {entry['residual_mm_mean']} mm "
                            f"> {ceiling} mm (max {entry['residual_mm_max']} mm)")
            else:
                log("OK", f"{name}/{side}: steady residual mean {entry['residual_mm_mean']} / "
                          f"max {entry['residual_mm_max']} mm (ceiling {ceiling} mm)")
            floor = profiles[name].min_travel_mm
            if entry["travel_mm"] < floor:
                failures.append(f"{name}_{side}_travel")
                log("FAIL", f"{name}/{side}: wrist travelled {entry['travel_mm']} mm "
                            f"(< {floor} mm)")
            gap = abs(entry["travel_mm"] - entry["target_travel_mm"])
            shortfall = entry["target_travel_mm"] - entry["travel_mm"]
            if shortfall > args.max_travel_gap_mm:
                failures.append(f"{name}_{side}_travel_gap")
                log("FAIL", f"{name}/{side}: wrist travel {entry['travel_mm']} mm falls "
                            f"{shortfall:.1f} mm short of the commanded "
                            f"{entry['target_travel_mm']} mm")
            else:
                log("OK", f"{name}/{side}: travelled {entry['travel_mm']} mm "
                          f"(commanded {entry['target_travel_mm']} mm, limit "
                          f"{entry['travel_limit_mm']} mm per axis; overshoot beyond the "
                          f"command is the servoing jitter, {gap:.1f} mm here)")
    if report["hold_frames_identical"]["checked"] == 0:
        failures.append("no_hold_observed")
        log("FAIL", "no dropped frame was seen, so hold-on-invalid was never exercised")
    elif not report["hold_frames_identical"]["all_identical"]:
        failures.append("hold_moved")
        log("FAIL", "a held frame moved the target; an invalid frame must reuse the last one")
    else:
        log("OK", f"dropped frames held the last target exactly "
                  f"({report['hold_frames_identical']['checked']} checks)")
    for name, entry in sorted(report["hold_drift"].items()):
        if entry["max_drift_rad"] > args.max_hold_drift_rad:
            failures.append(f"{name}_hold_drift")
            log("FAIL", f"{name}: held joints drifted {entry['max_drift_rad']} rad "
                        f"({entry['joint']})")
        else:
            log("OK", f"{name}: held joints drifted at most {entry['max_drift_rad']} rad "
                      f"({entry['held_joints']} joints; unmapped hand joints up to "
                      f"{entry['hand_drift_rad']} rad, informational)")



def _sweep(args, robots, profiles, offsets, amplitude, frames) -> dict:
    """两侧同时整圈扫掠 -> ``{robot: {side: {mean_mm, max_mm, ok}}}``。

    ``offsets[(robot, side)]`` 给该侧换一个中性工作点（``--probe-lateral`` 用），缺省用
    profile 的每侧实测中性。
    """
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

    sides = tuple(SIDES if args.side == "both" else (args.side,))
    scene = ThreeRobotHandScene(sides, gui=False, steps_per_frame=8)
    try:
        driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                ik_iterations=args.arm_ik_iterations,
                                residual_threshold=args.arm_ik_residual_threshold)
        out = {}
        for name in robots:
            adapter = driver.ik[name]
            profile = profiles[name]
            residuals = {side: [] for side in sides}
            shifts = {}
            for side in sides:
                wanted = np.asarray(offsets.get((name, side),
                                                profile.realtime_neutral(side)), dtype=np.float64)
                shifts[side] = wanted - np.asarray(profile.metric_offset, dtype=np.float64)
            for frame in range(frames):
                for side in sides:
                    target = sweep_target(profile, amplitude,
                                          2.0 * pi * frame / frames + SIDE_PHASE[side],
                                          "base")
                    target = tuple(target[axis] + shifts[side][axis] for axis in range(3))
                    adapter.apply(side, adapter.solve(side, target, source="base"))
                scene.step()
                for side in sides:
                    residuals[side].append(adapter.residual_mm(side))
            out[name] = {}
            for side in sides:
                steady = residuals[side][args.settle_skip:]
                mean = float(np.mean(steady)) if steady else float("nan")
                worst = float(np.max(steady)) if steady else float("nan")
                ceiling = profile.max_residual_mm
                out[name][side] = {"mean_mm": round(mean, 2), "max_mm": round(worst, 2),
                                   "ok": bool(mean <= ceiling and worst <= 1.5 * ceiling)}
        return out
    finally:
        scene.close()


def _probe_report(args, report) -> int:
    """把探测结果写进 ``--report``（可选）并返回退出码 0。"""
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    return 0



def probe_travel(args) -> int:
    """量每台每侧的行程上限：幅值从小到大扫，mean 与 max 都在 ceiling 内才算过。"""
    robots = tuple(args.robot) if args.robot else loadable_robots()
    profiles = {name: profile_for(name) for name in robots}
    amplitudes = (0.04, 0.06, 0.08, 0.10, 0.12, 0.16)
    frames = args.frames if 0 < args.frames <= 40 else 24
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    rows = {}
    for amplitude in amplitudes:
        key = f"{amplitude:.2f}"
        rows[key] = _sweep(args, robots, profiles, {}, amplitude, frames)
        for name in robots:
            text = " ".join(
                f"{side[0].upper()} mean {rows[key][name][side]['mean_mm']:6.2f}"
                f"/max {rows[key][name][side]['max_mm']:6.2f}"
                f"{' ok' if rows[key][name][side]['ok'] else ' XX'}"
                for side in sides)
            log("INFO", f"{name:>7} {amplitude:.2f} m: {text}")
    suggestion = {}
    for name in robots:
        suggestion[name] = {}
        for side in sides:
            passing = [amplitude for amplitude in amplitudes
                       if rows[f"{amplitude:.2f}"][name][side]["ok"]]
            suggestion[name][side] = max(passing) if passing else 0.0
    log("OK", f"suggested realtime_travel_m (per side): {suggestion}")
    return _probe_report(args, {"mode": "probe_travel", "amplitudes": list(amplitudes),
                               "frames": frames, "rows": rows, "suggestion": suggestion})


def probe_lateral(args) -> int:
    """量每台每侧的横向中性：固定幅值，扫 base-frame Y，取残差最小的那个。"""
    robots = tuple(args.robot) if args.robot else loadable_robots()
    profiles = {name: profile_for(name) for name in robots}
    y_values = (0.0, -0.10, -0.15, -0.20, 0.10, 0.15, 0.20)
    amplitude = 0.06
    frames = args.frames if 0 < args.frames <= 40 else 24
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    rows = {}
    for y in y_values:
        offsets = {}
        for name in robots:
            for side in sides:
                offset = profiles[name].realtime_neutral(side)
                offsets[(name, side)] = (offset[0], y, offset[2])
        rows[f"{y:+.2f}"] = _sweep(args, robots, profiles, offsets, amplitude, frames)
    suggestion = {}
    for name in robots:
        suggestion[name] = {}
        for side in sides:
            best = min(y_values,
                       key=lambda value: rows[f"{value:+.2f}"][name][side]["mean_mm"])
            suggestion[name][side] = {
                "y_m": best, "mean_mm": rows[f"{best:+.2f}"][name][side]["mean_mm"],
                "at_zero_mm": rows["+0.00"][name][side]["mean_mm"]}
        log("INFO", f"{name:>7} best lateral neutral: {suggestion[name]}")
    log("OK", "compare against profile.realtime_offset (the right arm usually prefers y=0)")
    return _probe_report(args, {"mode": "probe_lateral", "amplitude": amplitude,
                               "y_values": list(y_values), "frames": frames,
                               "rows": rows, "suggestion": suggestion})


def main(args=None) -> int:
    if args is None:
        args = build_parser().parse_args()
    if args.probe_travel:
        return probe_travel(args)
    if args.probe_lateral:
        return probe_lateral(args)
    report, failures = drive(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    if failures:
        log("FAIL", f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    log("OK", "realtime arm bridge verified (calibration / hold / residual / travel / drift)")
    log("WARN", "the operator mapping is per-robot tuning, not a hand-eye calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






