"""量一件事：把「肘」也当一个位置目标交给 IK，只动大臂时机器人会不会跟着动。

为什么需要这个探针
------------------
现场实测（2026-10-13）：双手握拳贴在胸前不动、只把肘（大臂）摆开，三台机器人**全都没有
反应**；大臂尤其 G1 / GR1-T2 基本不摆。原因有两条，本探针要把它们分开量出来：

1. **目标根本没动**：历史口径只喂「腕 - 肩」，手留在胸前、只摆肘时这个向量几乎不变；
2. **就算目标动了，肩也不爱转**：7 自由度冗余链只给腕一个位置时，求解器把解留在原地，
   哪几个关节动由零空间说了算（``devtools/probe_posture_task.py`` 量到 restPoses 的
   authority 只有 7-22%）。

论文的接口不是「只有腕的位置」：它有 FrameTask（腕位姿）+ 肘/整条臂的位置任务。
PyBullet 有 ``calculateInverseKinematics2``：一次交**两个**位置目标（腕 + 肘），这就是本机
能用的「肘位置任务」。

本探针只回答一个问题，且可证伪：

    腕目标一模一样、只多交一个肘目标，肩会不会真的转起来？腕残差会不会坏？

做法（每台机器人、每侧，互不干扰）
----------------------------------
1. 先回中性并静置，记基线：肩/肘关节角、肩/肘/腕三个 FK 点、|腕-肩|；
2. 四个变体各跑 ``--frames`` 帧（每次先回中性，免得上一个变体串味）：

   control     腕 = 中性、不给肘目标（期望：纹丝不动，现场症状的复现）
   wrist_only  腕 = 中性 + 横向 100 mm、不给肘目标（历史口径：腕动、肩也动一点）
   elbow_task  腕 = 中性、肘 = 中性肘 + 横向 100 mm（**本探针的主角**）
   both        腕 + 肘一起给（真实现场就是这一种）

3. 每个变体量：肩关节最大变化（rad）、肘关节变化、上臂方向转过多少度、肘连杆位移、
   腕连杆位移、腕残差、肘任务误差（FK 肘离肘目标还有多远）。

判据
----
- ``elbow_task_effective``：elbow_task 的上臂转角 >= 2 度且肘连杆位移 >= 10 mm；
- ``residual_ok``：elbow_task 的腕残差 <= profile.max_residual_mm；
- ``control_dead``：control 的上臂转角 <= 0.5 度（复现现场症状，证明对比有效）。

用法::

    E:\\python3.11.7\\python.exe devtools\\probe_elbow_task.py
    E:\\python3.11.7\\python.exe devtools\\probe_elbow_task.py --robot g1 --side left
    E:\\python3.11.7\\python.exe devtools\\probe_elbow_task.py --delta-mm 150 ^
        --report outputs\\tmp_realtime\\probe_elbow_task.json

退出码：0 = 至少一对 (机器人, 侧) 的肘任务真的把大臂带起来了；1 = 一台都没有。
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
DEFAULT_SETTLE = 6
DEFAULT_DELTA_MM = 100.0
#: 肩/肘关节名里的关键字（三台机器人都含，见 arm_ik.SHOULDER_KEYWORD / ELBOW_KEYWORD）。
SHOULDER_KEYWORD = "shoulder"
ELBOW_KEYWORD = "elbow"
#: 「大臂摆开」的横向方向：基座系 Y = 机器人左侧 -> 左臂 +Y、右臂 -Y。
LATERAL_SIGN = {"left": 1.0, "right": -1.0}
#: 四个变体（每个变体的腕目标 / 肘目标 / 肘任务遍序）：
#:
#: control           腕 = 中性、不给肘目标 -> 期望纹丝不动（现场症状的复现）
#: elbow_task_first  腕 = 中性、肘 = 球面上横移 delta（**故意与腕目标矛盾**，单独量肘的拉力）
#: consistent_first  腕与肘来自同一条「方向是你的、长度是机器人的」链，肘任务在第一遍
#: consistent_last   同上，但肘任务放最后一遍（候选默认口径）
VARIANTS = ("control", "elbow_task_first", "consistent_first", "consistent_last")


def log(mark: str, message: str) -> None:
    print(f"[{mark:<5}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/probe_elbow_task.py",
        description="elbow as a second IK position target: does the upper arm follow?")
    parser.add_argument("--robot", action="append", default=None,
                        help="only these robots (repeatable; default: all loaded)")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--delta-mm", type=float, default=DEFAULT_DELTA_MM,
                        help="lateral displacement of the varied target, mm "
                             f"(default {DEFAULT_DELTA_MM})")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames per variant (default {DEFAULT_FRAMES})")
    parser.add_argument("--settle", type=int, default=DEFAULT_SETTLE,
                        help="frames used to return to neutral between variants "
                             f"(default {DEFAULT_SETTLE})")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _joint_names(adapter, side: str, keyword: str) -> tuple:
    return tuple(name for _index, name in adapter.arm[side] if keyword in name.lower())


def _sample(adapter, side: str) -> dict:
    """这一侧现在的状态：肩/肘关节角、三个 FK 点、|腕-肩|。"""
    values = adapter.joint_values(side)
    points = adapter.arm_points(side)
    return {
        "shoulder_joints_rad": {name: round(float(values[name]), 5)
                                for name in _joint_names(adapter, side, SHOULDER_KEYWORD)},
        "elbow_joints_rad": {name: round(float(values[name]), 5)
                             for name in _joint_names(adapter, side, ELBOW_KEYWORD)},
        "points": {key: [round(float(value), 5) for value in point]
                   for key, point in points.items()},
        "reach_mm": round(1000.0 * float(np.linalg.norm(
            np.asarray(points["wrist"], dtype=float)
            - np.asarray(points["shoulder"], dtype=float))), 2),
    }


def _world_to_base(adapter, point) -> tuple:
    """世界坐标 -> 该机器人基座系（GR1-T2 的基座带旋转，不能只减平移）。"""
    p = adapter.p
    base_position, base_orientation = p.getBasePositionAndOrientation(
        adapter.body, physicsClientId=adapter.cid)
    inverse_position, inverse_orientation = p.invertTransform(base_position,
                                                              base_orientation)
    local, _orientation = p.multiplyTransforms(
        inverse_position, inverse_orientation, [float(v) for v in point],
        [0.0, 0.0, 0.0, 1.0])
    return tuple(float(value) for value in local)


def _upper_arm_deg(baseline_points, points) -> float:
    """上臂方向（肩->肘）相对基线转过的角度。"""
    left = (np.asarray(baseline_points["elbow"], dtype=float)
            - np.asarray(baseline_points["shoulder"], dtype=float))
    right = (np.asarray(points["elbow"], dtype=float)
             - np.asarray(points["shoulder"], dtype=float))
    norms = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if norms <= 1e-12:
        return 0.0
    cosine = float(np.clip(float(np.dot(left, right)) / norms, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _moved_mm(baseline_points, points, key: str) -> float:
    return round(1000.0 * float(np.linalg.norm(
        np.asarray(points[key], dtype=float)
        - np.asarray(baseline_points[key], dtype=float))), 2)


def _settle(scene, adapter, sides, neutral, frames) -> None:
    """把每一侧都写回中性关节角，让场景走几步，回到确定的起点。"""
    for _frame in range(max(2, frames)):
        for side in sides:
            adapter.apply(side, neutral[side])
        scene.step()


def _warm_to_neutral(scene, adapter, side, sides, neutral, wrist_base, frames) -> None:
    """先把这一侧跑到中性目标（不量），让基线是「已经到位」的姿势。

    少了这一步，基线就是**加载姿势**，第一个变体的位移里会混进「从加载姿势摆到中性目标」
    那一段 —— 实测会把 control 也量成 15-47 度，把结论带偏（2026-10-13 第一版探针）。
    """
    for _frame in range(max(2, frames)):
        adapter.apply(side, adapter.solve(side, wrist_base, source="base"))
        for other in sides:
            if other != side:
                adapter.apply(other, neutral[other])
        scene.step()


def _run_variant(scene, adapter, side, sides, neutral, wrist, elbow, frames,
                 elbow_pass: str = "first") -> dict:
    """固定腕目标（+ 可选肘目标）跑 frames 帧，稳态量一次。"""
    residuals = []
    for _frame in range(frames):
        adapter.apply(side, adapter.solve(side, wrist, source="base", elbow=elbow,
                                          elbow_pass=elbow_pass))
        for other in sides:
            if other != side:
                adapter.apply(other, neutral[other])
        scene.step()
        residuals.append(adapter.residual_mm(side))
    state = _sample(adapter, side)
    state["residual_mm"] = round(float(residuals[-1]), 2)
    state["residual_mean_mm"] = round(float(np.mean(residuals)), 2)
    target = adapter.last_elbow_world.get(side)
    state["elbow_target_world"] = (None if target is None
                                   else [round(float(value), 5) for value in target])
    state["elbow_task_err_mm"] = (None if target is None else round(
        1000.0 * float(np.linalg.norm(np.asarray(state["points"]["elbow"], dtype=float)
                                      - np.asarray(target, dtype=float))), 2))
    return state


def _shoulder_delta_rad(baseline: dict, state: dict) -> float:
    """所有肩关节里变化最大的那个（rad）。"""
    deltas = [abs(float(state["shoulder_joints_rad"][name])
                  - float(baseline["shoulder_joints_rad"][name]))
              for name in baseline["shoulder_joints_rad"]]
    return round(max(deltas) if deltas else 0.0, 5)


def _elbow_delta_rad(baseline: dict, state: dict) -> float:
    """所有肘关节里变化最大的那个（rad）。"""
    deltas = [abs(float(state["elbow_joints_rad"][name])
                  - float(baseline["elbow_joints_rad"][name]))
              for name in baseline["elbow_joints_rad"]]
    return round(max(deltas) if deltas else 0.0, 5)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
    from teleoperation.simulation.arm_profiles import SIDES, loadable_robots, profile_for
    from teleoperation.simulation.arm_scene_driver import ArmSceneDriver

    robots = tuple(args.robot) if args.robot else loadable_robots()
    sides = tuple(SIDES if args.side == "both" else (args.side,))
    scene = ThreeRobotHandScene(sides=sides, gui=False, steps_per_frame=8)
    report = {"mode": "probe_elbow_task", "delta_mm": float(args.delta_mm),
              "frames": int(args.frames), "variants": list(VARIANTS), "robots": {}}
    effective = 0
    try:
        import pybullet as p

        report["pybullet_has_ik2"] = bool(hasattr(p, "calculateInverseKinematics2"))
        log("INFO", "pybullet.calculateInverseKinematics2 available: "
                    f"{report['pybullet_has_ik2']}")
        driver = ArmSceneDriver(scene, sides=sides, robots=robots,
                                ik_iterations=200, residual_threshold=1e-5)
        for name in robots:
            adapter = driver.ik[name]
            profile = profile_for(name)
            report["robots"][name] = {"ik_mode": adapter.ik_mode, "sides": {}}
            for side in sides:
                neutral = {other: adapter.joint_values(other) for other in sides}
                wrist_base = tuple(float(value)
                                   for value in profile.realtime_neutral(side))
                _settle(scene, adapter, sides, neutral, args.settle)
                _warm_to_neutral(scene, adapter, side, sides, neutral, wrist_base,
                                 max(2, args.frames))
                baseline = _sample(adapter, side)
                # 肘目标：把「肘横移 delta」投影回**以肩为心、上臂长为半径**的球面上 ——
                # 方向是你的，长度是机器人自己的（论文 scale_joints 的口径）。直接给一个
                # 横移 100 mm 的肘点会与「腕不动」互相矛盾，IK2 只能两边各让一步。
                lateral = np.array([0.0,
                                    0.001 * args.delta_mm
                                    * float(LATERAL_SIGN.get(side, 1.0)), 0.0])
                shoulder = np.asarray(baseline["points"]["shoulder"], dtype=float)
                wrist_point = np.asarray(wrist_base, dtype=float)
                elbow_now = np.asarray(baseline["points"]["elbow"], dtype=float)
                wrist_now = np.asarray(baseline["points"]["wrist"], dtype=float)
                upper_length = float(np.linalg.norm(elbow_now - shoulder))
                fore_vector = wrist_now - elbow_now
                fore_length = float(np.linalg.norm(fore_vector))
                fore_dir = (fore_vector / fore_length if fore_length > 1e-9
                            else np.array([0.0, 0.0, 1.0]))
                direction = (elbow_now + lateral) - shoulder
                norm = float(np.linalg.norm(direction))
                direction = (direction / norm if norm > 1e-9
                             else np.array([0.0, 0.0, 1.0]))
                elbow_base = np.asarray(
                    _world_to_base(adapter, shoulder + upper_length * direction),
                    dtype=float)
                # 「一致链」：肘目标与腕目标由同一条链给出（方向是你的、长度是机器人的），
                # 所以两个目标是**同时可达**的 —— 直接给一个横移 100 mm 的肘点会与「腕不动」
                # 互相矛盾（前臂长度就得变），IK2 只能两边各让一步。
                consistent_elbow = shoulder + upper_length * direction
                consistent_wrist = consistent_elbow + fore_length * fore_dir
                consistent_elbow_base = np.asarray(
                    _world_to_base(adapter, consistent_elbow), dtype=float)
                consistent_wrist_base = np.asarray(
                    _world_to_base(adapter, consistent_wrist), dtype=float)
                targets = {
                    "control": (wrist_point, None, "first"),
                    "elbow_task_first": (wrist_point, elbow_base, "first"),
                    "consistent_first": (consistent_wrist_base, consistent_elbow_base,
                                         "first"),
                    "consistent_last": (consistent_wrist_base, consistent_elbow_base,
                                        "last"),
                }
                measured = {}
                for label in VARIANTS:
                    wrist, elbow, elbow_pass = targets[label]
                    _settle(scene, adapter, sides, neutral, args.settle)
                    state = _run_variant(
                        scene, adapter, side, sides, neutral,
                        tuple(float(value) for value in wrist),
                        None if elbow is None else tuple(float(value)
                                                        for value in elbow),
                        max(2, args.frames), elbow_pass=elbow_pass)
                    state["shoulder_delta_rad"] = _shoulder_delta_rad(baseline, state)
                    state["elbow_delta_rad"] = _elbow_delta_rad(baseline, state)
                    state["upper_arm_deg"] = round(
                        _upper_arm_deg(baseline["points"], state["points"]), 2)
                    state["elbow_mm_moved"] = _moved_mm(baseline["points"],
                                                        state["points"], "elbow")
                    state["wrist_mm_moved"] = _moved_mm(baseline["points"],
                                                        state["points"], "wrist")
                    measured[label] = state
                control = measured["control"]
                task = measured["consistent_last"]
                ok_task = bool(task["upper_arm_deg"] >= 2.0
                               and task["elbow_mm_moved"] >= 10.0)
                ok_residual = bool(task["residual_mm"] <= profile.max_residual_mm
                                   and (task["elbow_task_err_mm"] or 0.0)
                                   <= profile.max_residual_mm)
                ok_control = bool(control["upper_arm_deg"] <= 0.5)
                report["robots"][name]["sides"][side] = {
                    "baseline": baseline, "variants": measured,
                    "wrist_target_base_m": [round(float(value), 5)
                                            for value in wrist_base],
                    "elbow_target_base_m": [round(float(value), 5)
                                            for value in elbow_base],
                    "consistent_wrist_target_base_m": [round(float(value), 5)
                                                       for value in consistent_wrist_base],
                    "consistent_elbow_target_base_m": [round(float(value), 5)
                                                       for value in consistent_elbow_base],
                    "elbow_task_effective": ok_task,
                    "residual_ok": ok_residual,
                    "control_dead": ok_control,
                    "max_residual_mm": profile.max_residual_mm}
                effective += 1 if ok_task else 0
                log("OK" if ok_task else "FAIL",
                    f"{name}/{side}: upper arm control {control['upper_arm_deg']:.2f} / "
                    f"elbow_task_first {measured['elbow_task_first']['upper_arm_deg']:.2f} / "
                    f"consistent_first {measured['consistent_first']['upper_arm_deg']:.2f} / "
                    f"consistent_last {task['upper_arm_deg']:.2f} deg")
                log("OK" if ok_residual else "FAIL",
                    f"{name}/{side}: wrist residual control {control['residual_mm']:.1f} / "
                    f"consistent_first {measured['consistent_first']['residual_mm']:.1f} / "
                    f"consistent_last {task['residual_mm']:.1f} mm (ceiling "
                    f"{profile.max_residual_mm} mm); elbow task error "
                    f"first {measured['consistent_first']['elbow_task_err_mm']} / "
                    f"last {task['elbow_task_err_mm']} mm; elbow link moved "
                    f"last {task['elbow_mm_moved']:.1f} mm, shoulder joint "
                    f"{task['shoulder_delta_rad']:+.3f} rad")
    except Exception as error:  # noqa: BLE001 - 探针要把原因原样报出来
        log("FAIL", f"{type(error).__name__}: {error}")
        report["exception"] = f"{type(error).__name__}: {error}"
    finally:
        scene.close()

    report["elbow_task_effective_count"] = effective
    report["exit_code"] = 0 if effective else 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"report={args.report}")
    if not effective:
        log("FAIL", "no robot turned its upper arm towards the elbow target: "
                    "the elbow position task does not steer this chain")
        return 1
    log("OK", f"{effective} (robot, side) pairs let the elbow task move the upper arm")
    log("INFO", "next: feed a per-segment elbow target from the operator stream "
                "(--arm-map segments + --arm-elbow-task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


