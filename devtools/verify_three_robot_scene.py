"""验收探针：无头跑一遍【三台机器人并排】实时场景，并出离屏截图。

为什么要这个文件
----------------
实时入口（``teleoperation.apps.realtime_hand_sim``）默认的显示目标是三台论文机器人
**H1-2 / GR1-T2 / G1 并排**，但 PyBullet 的 GUI 窗口只在交互式桌面下才看得到。本探针
用 DIRECT 客户端 + ``getCameraImage`` 离屏渲染把**同一个场景类**跑起来，做到"没有
显示器也能核对画面里到底是不是三台机器人、手指到底动没动"：

    1. 建 ``ThreeRobotHandScene``（与 ``--scene robots`` 完全同一个类）
    2. 用一段合成 18 维角度（全开 -> 握拳）驱动三台机器人的原装手
    3. 逐帧记录被映射关节的实际角度，检查它们真的转过 >= 0.05 rad
    4. 第 1 帧与最后一帧各渲染两张 PNG（iso / front），并比较像素差

退出码：0 = 三台机器人全部加载 + 至少一台的原装手真的动了；1 = 否则。

用法::

    E:\\python3.11.7\\python.exe devtools/verify_three_robot_scene.py `
        --out-dir outputs/tmp_probe/three_robot_scene `
        --report outputs/tmp_probe/three_robot_scene/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from teleoperation.apps.three_robot_scene import (  # noqa: E402
    ROBOT_ORDER, ThreeRobotHandScene,
)

# 离屏视角：(名字, yaw, pitch)
VIEWS = (("iso", 135.0, -15.0), ("front", 0.0, 0.0))
MOVE_THRESHOLD = 0.05      # rad：判定"这根手指真的转过"
HOLD_THRESHOLD = 0.05      # rad：被锁住的关节允许的最大漂移（超过就是"机器人自己歪了"）


def synthetic_angles(t: float) -> np.ndarray:
    """合成 18 维角度：t=0 全开，t=1 握拳（每维都落在 L21 限位内）。"""
    values = np.zeros(18, dtype=np.float64)
    for base in (1, 4, 7, 10):        # 食指 / 中指 / 无名指 / 小指
        values[base + 1] = 1.20 * t   # mcp_pitch
        values[base + 2] = 1.40 * t   # pip
    values[13] = 0.30 * t             # thumb_cmc_roll
    values[14] = 0.80 * t             # thumb_cmc_yaw
    values[15] = 0.60 * t             # thumb_cmc_pitch
    values[16] = 0.80 * t             # thumb_mcp
    values[17] = 0.80 * t             # thumb_ip
    return values


def render(p, scene, path: Path, yaw: float, pitch: float,
           width: int, height: int, distance: float | None = None) -> bool:
    """按场景自己的取景离屏渲染一张 PNG（CPU 光栅器，无显示器可用）。"""
    import cv2

    distance = float(scene.camera_distance if distance is None else distance)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(scene.camera_target), distance=distance,
        yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
    )
    # fov 与场景算距离时的假设一致（垂直 60 度 = 上下各 30 度），否则横向三台会被裁掉。
    projection = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / float(height), nearVal=0.1,
        farVal=distance * 4.0 + 10.0,
    )
    image = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER, physicsClientId=scene.cid,
    )
    rgba = np.asarray(image[2]).reshape(height, width, 4).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)))


def image_diff(first: Path, second: Path) -> float:
    """两张 PNG 的平均绝对像素差（0.0 表示画面完全一样）。"""
    import cv2

    a, b = cv2.imread(str(first)), cv2.imread(str(second))
    if a is None or b is None:
        return -1.0
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="无头验收【三台机器人并排】实时场景")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/tmp_probe/three_robot_scene"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    sides = tuple(part.strip() for part in args.sides.split(",") if part.strip())
    report = {"sides": list(sides), "frames": args.frames,
              "renderer": "ER_TINY_RENDERER", "move_threshold_rad": MOVE_THRESHOLD}
    scene = None
    status = 1
    try:
        started = time.perf_counter()
        scene = ThreeRobotHandScene(sides, gui=False)
        print(f"scene loaded: {[robot['type'] for robot in scene.robots]}  "
              f"camera_target={[round(value, 2) for value in scene.camera_target]}  "
              f"distance={scene.camera_distance:.2f} m", flush=True)
        report["scene_summary"] = scene.summary()
        report["camera"] = {"target": scene.camera_target,
                            "distance": scene.camera_distance}

        screenshots = {}
        history = {side: [] for side in sides}
        held_start = {
            robot["type"]: [float(scene.p.getJointState(
                robot["body"], index, physicsClientId=scene.cid)[0])
                for index in robot["hold_ids"]]
            for robot in scene.robots
        }
        last_index = max(args.frames - 1, 0)
        for index in range(args.frames):
            angles = synthetic_angles(index / max(last_index, 1))
            for side in sides:
                scene.apply(side, angles)
            scene.step()
            if index in (0, last_index):
                for name, yaw, pitch in VIEWS:
                    path = args.out_dir / f"{name}_frame{index:03d}.png"
                    # 取景距离按视角各自算（斜视要退得更远才装得下三台）。
                    distance = scene.frame_scene("full" if yaw else "front")
                    if render(scene.p, scene, path, yaw, pitch, args.width, args.height,
                              distance):
                        screenshots[f"{name}_frame{index:03d}"] = str(path)
            for side in sides:
                history[side].append(scene.joint_values(side))
        elapsed = time.perf_counter() - started

        movement = {}
        for side in sides:
            first, last = history[side][0], history[side][-1]
            deltas = {name: abs(last[name] - value) for name, value in first.items()}
            moving = sorted(name for name, delta in deltas.items()
                            if delta >= MOVE_THRESHOLD)
            movement[side] = {
                "mapped_joints": len(deltas),
                "max_delta_rad": round(max(deltas.values()), 4) if deltas else 0.0,
                "moving_joints": moving,
            }
        report["movement"] = movement
        held_drift = {}
        for robot in scene.robots:
            values = [float(scene.p.getJointState(robot["body"], index,
                                                  physicsClientId=scene.cid)[0])
                      for index in robot["hold_ids"]]
            held_drift[robot["type"]] = round(max(
                (abs(value - start) for value, start in
                 zip(values, held_start[robot["type"]])), default=0.0), 4)
        report["held_joint_drift_rad"] = held_drift
        report["screenshots"] = screenshots
        report["frame_diffs"] = {
            name: image_diff(args.out_dir / f"{name}_frame000.png",
                             args.out_dir / f"{name}_frame{last_index:03d}.png")
            for name, _yaw, _pitch in VIEWS
        }
        report["elapsed_seconds"] = round(elapsed, 3)

        print(f"sim frames={scene.frames} elapsed={elapsed:.2f} s", flush=True)
        for side in sides:
            info = movement[side]
            print(f"  {side}: mapped={info['mapped_joints']} joints  "
                  f"max_delta={info['max_delta_rad']:.4f} rad  "
                  f"moving={len(info['moving_joints'])}", flush=True)
            if info["moving_joints"]:
                print(f"    e.g. {', '.join(info['moving_joints'][:4])}", flush=True)
        for name, value in held_drift.items():
            print(f"  held joints {name}: max drift {value:.4f} rad "
                  f"(limit {HOLD_THRESHOLD})", flush=True)
        for name, value in report["frame_diffs"].items():
            print(f"  png diff {name}: {value:.2f} (mean abs pixel delta)", flush=True)
        print(f"screenshots: {len(screenshots)} -> {args.out_dir}", flush=True)

        loaded = len(scene.robots)
        moved = any(movement[side]["max_delta_rad"] >= MOVE_THRESHOLD for side in sides)
        still = all(value <= HOLD_THRESHOLD for value in held_drift.values())
        report["robots_loaded"] = loaded
        report["hands_moved"] = bool(moved)
        report["robots_still"] = bool(still)
        status = 0 if (loaded == len(ROBOT_ORDER) and moved and still) else 1
        if status != 0:
            print(f"[FAIL] robots_loaded={loaded}/{len(ROBOT_ORDER)} "
                  f"hands_moved={moved} robots_still={still}", flush=True)
    except Exception as error:  # noqa: BLE001 - 探针只报结论，不抛栈
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"[FAIL] {report['error']}", flush=True)
    finally:
        if scene is not None:
            try:
                scene.close()
            except Exception:  # noqa: BLE001 - 收尾失败不影响结论
                pass
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            print(f"report written: {args.report}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
