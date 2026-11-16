"""探针：场景层摆好之后，机器人侧到底有哪些可用接口（为脚本策略打地基）。

只读，不改任何代码。打印：
* 动作空间维度 / 前后各几个名字；
* 两侧手臂关节名、灵巧手关节名；
* ``PyBulletArmIK`` 暴露的手掌（ee）/腕链接索引、肩/肘/腕锚点（基座系 + 世界系）；
* 场景层摆完后的物体位姿、桌面高度；
* 一次「IK 到手前方 30 cm」的实测：求解耗时、残差、240 步之后末端到底在哪。

用法::

    E:\\python3.11.7\\python.exe devtools\\probe_task_scene.py --robot h1_2 --task pushcube
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
DEVTOOLS = Path(__file__).resolve().parent
for extra in (ROOT / "src", DEVTOOLS):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import task_eval as te  # noqa: E402


def dump_ik(ik, side: str) -> dict:
    points = ik.arm_points(side)
    out = {
        "joints": [name for name, _ in ik.arm[side]],
        "joint_values": {name: round(float(v), 4)
                         for name, v in ik.joint_values(side).items()},
        "points_base": {key: [round(float(v), 4) for v in value]
                        for key, value in points.items()},
        "points_world": {key: [round(float(v), 4) for v in ik.to_world(value)]
                         for key, value in points.items()},
        "attrs": {key: (value if isinstance(value, (int, float, str, list, tuple, dict, type(None)))
                        else type(value).__name__)
                  for key, value in vars(ik).items() if key in
                  ("ee", "hand", "wrist", "side_links", "ee_index", "wrist_index",
                   "tool", "ik_mode", "neutral_orientation", "max_wrist_joint_step_deg")},
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="h1_2")
    parser.add_argument("--task", default="pushcube")
    parser.add_argument("--settle", type=int, default=te.SETTLE_STEPS)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from teleoperation.simulation.arm_ik import PyBulletArmIK
    from teleoperation.simulation.environment import SimulationEnv

    env = SimulationEnv(robot_type=args.robot, task_name=args.task, render=False,
                        record=False, data_dir=str(ROOT / "outputs" / "tmp_task_eval"))
    report = {"robot": args.robot, "task": args.task}
    try:
        env.reset(randomize=False)
        arm = te.measure_arm(args.robot)
        scene = te.TaskScene(env.client, env.task, robot=args.robot,
                             shoulder=arm.get("shoulder_m"), reach=arm.get("reach_m"),
                             name=args.task, robot_id=env.robot_id).prepare(args.settle)
        report["action_dim"] = int(env.action_dim or 0)
        report["action_names"] = list(env.action_joint_names)
        report["loader_hands"] = {side: list(names) for side, names
                                  in env.robot_loader.hand_joints.items()}
        report["scene"] = {key: scene.get(key) for key in
                           ("shift", "clamped", "compressed", "settle", "in_reach")}
        report["objects"] = te.object_positions(env)
        report["arm"] = {key: arm.get(key) for key in ("reach_m", "upper_m", "fore_m",
                                                       "shoulder_m")}

        ik = PyBulletArmIK(env.robot_id, env.client, robot_type=env.robot_type,
                           rest_poses="current")
        report["ik_sides"] = list(ik.arm)
        report["ik"] = {side: dump_ik(ik, side) for side in ik.arm}

        # 一次实测：把右手腕送到「肩前 0.30 m、向下一点」，看残差与 240 步后到位情况
        side = "right"
        points = ik.arm_points(side)
        shoulder = np.asarray(points["upper"], dtype=np.float64)
        target = ik.to_base(shoulder) + np.array([0.30, 0.0, -0.10])
        t0 = time.time()
        solution = ik.solve(side, target, source="base", posture=None)
        solve_s = time.time() - t0
        ik.apply(side, solution)
        for _ in range(240):
            p.stepSimulation(physicsClientId=env.client)
        reached = ik.arm_points(side)
        report["solve_test"] = {
            "target_base": [round(float(v), 4) for v in target],
            "solve_s": round(solve_s, 4),
            "residual_mm": round(ik.residual_mm(side), 2),
            "wrist_base_after": [round(float(v), 4) for v in reached["wrist"]],
            "elbow_base_after": [round(float(v), 4) for v in reached["elbow"]],
            "joint_values": {name: round(float(v), 3)
                             for name, v in ik.joint_values(side).items()},
        }
    finally:
        env.close()

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
        print("written:", args.json)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
