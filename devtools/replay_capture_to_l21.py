"""离线闭环：采集的原始 21 点 H5 -> 掌面局部 25 点 -> 重定向 -> PyBullet 渲染。

显示目标由 ``--scene`` 选择（**默认 robots**）：三台论文机器人 **H1-2 / GR1-T2 / G1 并排**、
各用**出厂自带**的灵巧手（``apps/three_robot_scene.py``）；``--scene hands`` 是旧行为，
只摆 1-2 只 LinkerHand L21 手。

为什么需要这个文件
------------------
实时入口 ``teleoperation.apps.realtime_hand_sim`` 的输入是**摄像头**：镜头前没有手时
MediaPipe 会连续检出 0 只手，实时链路就没有可重定向的输入（本机实测：同一台相机
亮度正常 114~147，120 帧全部 valid=0，而同一套采集脚本在 2026-09-12 采到的数据
643/701 帧有手，说明差异只在"镜头前有没有手"）。本文件把输入换成**文件**：任何符合
``left_hand_keypoints`` / ``right_hand_keypoints`` 形状 ``(T,21,3)`` 的 H5 都能驱动
**同一条**重定向链路，并把 L21 手在 PyBullet 里渲染出来，用于离线验证与截图取证。

数据流（全部复用既有模块，本文件不修改任何队友文件）：

    H5 原始 21 点
      -> RawHandFrame
      -> MediaPipePalmLocalProcessor     # 掌面局部对齐，21 点补成 25 点
      -> TemporalBuffer                  # 3 帧窗口（重定向的 receptive field）
      -> GeometricHandRetargeter         # 零权重几何后端，输出 18 维 L21 角度
      -> RealtimeL21Scene                # PyBullet resetJointState 写关节
      -> getCameraImage 截图 PNG         # 无显示环境下也能出证据

角度语义、标定方式、已知代价都写在 ``teleoperation/retargeting/hand/geometric.py``
的模块注释里；本文件只负责把文件输入接上，不改变任何数值。

用法::

    # 无头渲染（无人值守会话可用），每 25 帧存一组三视图 PNG
    python devtools/replay_capture_to_l21.py \
        --input-h5 "D:/.../visual_hand_data_20260912_112108.h5" \
        --out-dir outputs/tmp_probe/offline_l21 \
        --report outputs/tmp_probe/offline_l21/report.json

    # 本地看窗口（需要交互式桌面）
    python devtools/replay_capture_to_l21.py --input-h5 <raw.h5> --gui

    # 只看一只手（旧行为：只有 LinkerHand L21，没有机器人）
    python devtools/replay_capture_to_l21.py --input-h5 <raw.h5> --scene hands

    # 输入直接是 18 维角度 H5（left_angles / right_angles (T,18)）：
    # 跳过 21 点 -> 掌面局部 -> 重定向，直接用真实采集的角度驱动三台机器人。
    # 视频分辨率跟随 --width / --height；整段大分辨率可配 --stride 2 --fps 15 省一半时间。
    python devtools/replay_capture_to_l21.py \
        --angles-h5 datasets/raw/retarget_twohand_153542.h5 \
        --video replay_twohand_robots.mp4

退出码：0 = 至少一侧产生了有效角度且（如启用）写出了图像；1 = 没有任何有效角度。
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

import h5py  # noqa: E402  (sys.path bootstrap must run first)

from teleoperation.apps.realtime_hand_sim import RealtimeL21Scene  # noqa: E402
from teleoperation.apps.three_robot_scene import ThreeRobotHandScene  # noqa: E402
from teleoperation.contracts.hand import HandWindow  # noqa: E402
from teleoperation.contracts.observations import RawHandFrame  # noqa: E402
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM  # noqa: E402
from teleoperation.retargeting.hand.geometric import GeometricHandRetargeter  # noqa: E402
from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor  # noqa: E402
from teleoperation.retargeting.hand.temporal import TemporalBuffer  # noqa: E402

SIDES = ("left", "right")
SIDE_KEYS = {"left": "left_hand_keypoints", "right": "right_hand_keypoints"}
# 角度文件（契约 A~H）里的数据集名：每帧已经是 18 维 L21 角度，不用再过 21 点管线。
ANGLES_KEYS = {"left": "left_angles", "right": "right_angles"}
VALID_KEYS = {"left": "left_valid", "right": "right_valid"}
# 三视角：正/侧/斜，用来在截图里同时看清屈曲与展开。
VIEWS = {
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "iso": (40.0, -20.0),
}
# 默认视角；三台并排（--scene robots）时会自动换成 front + iso，见 run()。
DEFAULT_VIEWS = ["iso", "side"]
ROBOT_VIEWS = ["front", "iso"]


def load_capture(path: Path, sides) -> dict:
    """Read the raw two-hand capture; an all-zero frame counts as missing.

    The recorder writes ``(T,21,3)`` with zeros for a side that was not detected
    (this is the same convention ``retargeting/data.py`` documents). Zeros are
    turned into ``None`` here so the processor treats them exactly like a missing
    hand instead of a degenerate one.
    """
    with h5py.File(path, "r") as handle:
        missing = [SIDE_KEYS[side] for side in sides if SIDE_KEYS[side] not in handle]
        if missing:
            raise KeyError(f"{path} lacks datasets {missing}; found {sorted(handle.keys())}")
        arrays = {side: np.asarray(handle[SIDE_KEYS[side]][:], dtype=np.float32) for side in sides}
        timestamps = (
            np.asarray(handle["timestamps"][:], dtype=np.float64)
            if "timestamps" in handle else None
        )
        frame_ids = (
            np.asarray(handle["frame_ids"][:], dtype=np.int64)
            if "frame_ids" in handle else None
        )
        attributes = {key: handle.attrs[key] for key in handle.attrs}
    length = min(arrays[side].shape[0] for side in sides)
    for side in sides:
        if arrays[side].shape[1:] != (21, 3):
            raise ValueError(f"{SIDE_KEYS[side]} must be (T,21,3), got {arrays[side].shape}")
    return {
        "schema": "keypoints",
        "frames": length,
        "hands": arrays,
        "timestamps": timestamps,
        "frame_ids": frame_ids,
        "attributes": attributes,
    }


def load_angle_capture(path: Path, sides) -> dict:
    """Read a retargeted-angle H5 (``left_angles`` / ``right_angles`` ``(T,18)``).

    契约 A~H 的角度文件（``datasets/raw/retarget_twohand_153542.h5``、
    ``my_recording_angles.h5`` 这类）里每一帧已经是一个 18 维 L21 角度向量，
    21 点 -> 掌面局部 -> 重定向那一段不需要再跑一遍，所以这里直接读角度：
    缺帧按 ``left_valid`` / ``right_valid``（没有就整段视为有效）标记。
    """
    with h5py.File(path, "r") as handle:
        missing = [ANGLES_KEYS[side] for side in sides if ANGLES_KEYS[side] not in handle]
        if missing:
            raise KeyError(f"{path} lacks datasets {missing}; found {sorted(handle.keys())}")
        arrays = {side: np.asarray(handle[ANGLES_KEYS[side]][:], dtype=np.float64)
                  for side in sides}
        valid = {}
        for side in sides:
            key = VALID_KEYS[side]
            if key in handle:
                valid[side] = np.asarray(handle[key][:], dtype=bool)
            else:
                valid[side] = np.ones(arrays[side].shape[0], dtype=bool)
        timestamps = (
            np.asarray(handle["timestamps"][:], dtype=np.float64)
            if "timestamps" in handle else None
        )
        frame_ids = (
            np.asarray(handle["frame_ids"][:], dtype=np.int64)
            if "frame_ids" in handle else None
        )
        attributes = {key: handle.attrs[key] for key in handle.attrs}
    length = min(arrays[side].shape[0] for side in sides)
    for side in sides:
        if arrays[side].shape[1:] != (HAND_ANGLE_DIM,):
            raise ValueError(f"{ANGLES_KEYS[side]} must be (T,{HAND_ANGLE_DIM}), "
                             f"got {arrays[side].shape}")
    return {
        "schema": "angles",
        "frames": length,
        "angles": arrays,
        "valid": valid,
        "timestamps": timestamps,
        "frame_ids": frame_ids,
        "attributes": attributes,
    }


def frame_hand(points: np.ndarray):
    """Return the (21,3) landmark row, or None when the recorder wrote zeros."""
    if not np.isfinite(points).all():
        return None
    if not np.any(np.abs(points) > 1e-9):
        return None
    return points.astype(np.float64, copy=True)


def camera_lens(scene, distance: float):
    """离屏渲染用的 (fov, far)：三台机器人要 60 度视场 + 更远的裁剪面，L21 手沿用旧的。

    机器人的并排场景按垂直 60 度算取景距离（three_robot_scene.frame_scene），三台横向
    跨度约 5.2 m、相机距离 4 m 上下；沿用 L21 的 50 度视场 + farVal=3.0 会把最边上
    一台裁掉、并把机器人整段切掉。
    """
    robots = hasattr(scene, "frame_scene")
    if not robots:
        return 50.0, 3.0
    return 60.0, max(3.0, float(distance) * 4.0 + 10.0)


def render_png(scene, view: str, path: Path, width: int, height: int,
               target, distance: float) -> bool:
    """Render one view with the CPU rasterizer (works with no display attached)."""
    import cv2

    yaw, pitch = VIEWS[view]
    fov, far = camera_lens(scene, distance)
    view_matrix = scene.p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(target), distance=distance, yaw=yaw, pitch=pitch,
        roll=0, upAxisIndex=2,
    )
    projection = scene.p.computeProjectionMatrixFOV(
        fov=fov, aspect=width / float(height), nearVal=0.01, farVal=far,
    )
    image = scene.p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=projection,
        renderer=scene.p.ER_TINY_RENDERER, physicsClientId=scene.cid,
    )
    rgba = np.asarray(image[2]).reshape(height, width, 4).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)))


def describe(samples) -> dict | None:
    """Per-joint statistics; ``max_std`` proves the pose actually moved."""
    if not samples:
        return None
    values = np.stack(samples)
    return {
        "frames": int(values.shape[0]),
        "min": [float(item) for item in values.min(axis=0)],
        "max": [float(item) for item in values.max(axis=0)],
        "mean": [float(item) for item in values.mean(axis=0)],
        "std": [float(item) for item in values.std(axis=0)],
        "max_std": float(values.std(axis=0).max()),
    }


def run(args) -> int:
    import cv2

    sides = SIDES if args.hand == "both" else (args.hand,)
    angle_input = args.angles_h5 is not None
    source = Path(args.angles_h5 if angle_input else args.input_h5)
    capture = (load_angle_capture(source, sides) if angle_input
               else load_capture(source, sides))
    total = capture["frames"]
    frames = total if args.max_frames <= 0 else min(total, args.max_frames)
    out_dir = Path(args.out_dir)
    print(f"input={source} schema={capture['schema']}")
    print(f"frames_in_file={total} frames_used={frames} stride={args.stride} sides={sides}")
    print(f"attrs={capture['attributes']}")
    if angle_input:
        print("note: --angles-h5 already carries 18D angles per frame; "
              "--calibration-frames / --smoothing do not apply to this input")

    processor = MediaPipePalmLocalProcessor()
    buffer = TemporalBuffer(3)
    retargeter = GeometricHandRetargeter(calibration_frames=args.calibration_frames,
                                         smoothing=args.smoothing)
    samples = {side: [] for side in sides}
    detected = {side: 0 for side in sides}
    missing = {side: 0 for side in sides}
    rendered = []
    writer = None
    scene = None
    final = {}
    processed = 0
    elapsed = 0.0
    target = [0.0, 0.0, args.target_z]
    distance = float(args.distance)
    try:
        if args.scene == "robots":
            # 三台并排：取景由场景自己算（整机 + 横向约 5.2 m），不能沿用 L21 的 0.8 m。
            scene = ThreeRobotHandScene(sides, gui=args.gui,
                                        steps_per_frame=args.sim_steps)
            target = list(scene.camera_target)
            distance = float(scene.camera_distance)
            print(f"scene=three_robots target={[round(item, 2) for item in target]} "
                  f"distance={distance:.2f} m")
        else:
            scene = RealtimeL21Scene(sides, gui=args.gui)
            print("scene=hands (LinkerHand L21 only)")
        views = list(args.views)
        if args.scene == "robots" and views == DEFAULT_VIEWS:
            # side（yaw 90）是"沿着排看"，三台会前后遮挡；并排场景换成正面 + 斜视。
            views = list(ROBOT_VIEWS)
            print(f"note: --scene robots switches the default views to {views} "
                  f"(pass --views to override)")
        started = time.perf_counter()
        for index in range(0, frames, args.stride):
            timestamp = (float(capture["timestamps"][index])
                         if capture["timestamps"] is not None else float(index))
            if angle_input:
                # 角度文件：本帧已经是 18 维 L21 角度，直接写进场景（不再过 21 点管线）。
                for side in sides:
                    if not bool(capture["valid"][side][index]):
                        missing[side] += 1
                        continue
                    values = np.asarray(capture["angles"][side][index], dtype=np.float64)
                    if values.shape != (HAND_ANGLE_DIM,) or not np.isfinite(values).all():
                        raise ValueError(f"invalid {side} angles at frame {index}: "
                                         f"shape={values.shape}")
                    detected[side] += 1
                    scene.apply(side, values)
                    samples[side].append(values)
                processed += 1
                scene.step()
            else:
                hands = {side: frame_hand(capture["hands"][side][index]) for side in SIDES}
                for side in sides:
                    if hands[side] is None:
                        missing[side] += 1
                    else:
                        detected[side] += 1
                raw = RawHandFrame(hands, timestamp, "mediapipe_approx",
                                   {"path": str(source), "frame_index": int(index)})
                window = buffer.append(processor.process_frame(raw))
                processed += 1
                if window is not None:
                    result = retargeter.retarget(window)
                    for side in sides:
                        angles = getattr(result, side + "_angles")
                        if angles is None:
                            continue
                        values = np.asarray(angles.values, dtype=np.float64)
                        if values.shape != (HAND_ANGLE_DIM,) or not np.isfinite(values).all():
                            raise ValueError(f"invalid {side} angles: shape={values.shape}")
                        scene.apply(side, values)
                        samples[side].append(values)
                    scene.step()
            if args.video:
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_dir / args.video),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(args.fps, 1),
                                             (args.width, args.height))
                    print(f"video: {out_dir / args.video} "
                          f"{args.width}x{args.height} @ {max(args.fps, 1)} fps")
                render_tiny(scene, writer, target, distance, args.width, args.height)
            elif args.render_every and processed % args.render_every == 0:
                for view in views:
                    path = out_dir / f"frame{index:05d}_{view}.png"
                    if render_png(scene, view, path, args.width, args.height, target,
                                  distance):
                        rendered.append(str(path))
        elapsed = time.perf_counter() - started
        for side in sides:
            final[side] = scene.joint_values(side)
        if scene.gui and args.hold_seconds > 0:
            deadline = time.perf_counter() + args.hold_seconds
            while time.perf_counter() < deadline and scene.alive():
                scene.step()
                time.sleep(0.01)
    finally:
        if writer is not None:
            writer.release()
        if scene is not None:
            scene.close()

    print(f"\nprocessed={processed} frames in {elapsed:.2f} s "
          f"({processed / max(elapsed, 1e-9):.1f} frames/s)")
    calibration = ("n/a (--angles-h5: the 18D angles are already retargeted)"
                   if angle_input else
                   f"{retargeter.calibration_status} "
                   f"(open-hand reference = first {args.calibration_frames} valid frames)")
    print(f"calibration={calibration}")
    report = {
        "input_h5": str(source),
        "input_schema": capture["schema"],
        "input_attributes": {key: str(value) for key, value in capture["attributes"].items()},
        "frames_in_file": int(total),
        "frames_used": int(frames),
        "stride": int(args.stride),
        "processed": int(processed),
        "elapsed_seconds": round(float(elapsed), 3),
        "backend": "geometric",
        "scene": args.scene,
        "scene_summary": (scene.summary() if hasattr(scene, "summary") else None),
        "sides": {},
        "rendered_png": rendered,
        "video": None if args.video is None else str(out_dir / args.video),
    }
    for side in sides:
        entry = {
            "detected_frames": detected[side],
            "missing_frames": missing[side],
            "angles": describe(samples[side]),
            "final_joint_values": final.get(side),
        }
        report["sides"][side] = entry
        print(f"  {side}: detected={detected[side]} missing={missing[side]} "
              f"angles_produced={len(samples[side])}")
        if entry["angles"]:
            print(f"    angle min (18D): {[round(item, 3) for item in entry['angles']['min']]}")
            print(f"    angle max (18D): {[round(item, 3) for item in entry['angles']['max']]}")
            print(f"    angle max std: {entry['angles']['max_std']:.4f} rad "
                  f"(>0 means the rendered hand really moves, not a static pose)")
            print(f"    final joints: {entry['final_joint_values']}")
    if rendered:
        print(f"rendered {len(rendered)} PNG, first={rendered[0]}, last={rendered[-1]}")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {path}")
    if not any(samples[side] for side in sides):
        print("[FAIL] no valid angles were produced; check the H5 schema and hand data.")
        return 1
    return 0


def render_tiny(scene, writer, target, distance: float,
                width: int = 320, height: int = 240) -> None:
    """One ``width``x``height`` iso frame appended to the video writer.

    分辨率跟随 ``--width`` / ``--height``（默认 640x480）。320x240 下手指只占几个像素，
    看不出屈曲；调大后肉眼能直接看出握合，代价是渲染时间随像素数近似线性增长
    （本机实测 320x240 约 0.28 s/帧，640x480 约 0.35 s/帧），所以整段用大分辨率时可以
    ``--stride 2 --fps 15``：帧数减半、播放时长不变。
    """
    import cv2

    yaw, pitch = VIEWS["iso"]
    fov, far = camera_lens(scene, distance)
    view_matrix = scene.p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(target), distance=distance, yaw=yaw, pitch=pitch,
        roll=0, upAxisIndex=2,
    )
    projection = scene.p.computeProjectionMatrixFOV(
        fov=fov, aspect=width / float(height), nearVal=0.01, farVal=far,
    )
    image = scene.p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=projection,
        renderer=scene.p.ER_TINY_RENDERER, physicsClientId=scene.cid,
    )
    rgba = np.asarray(image[2]).reshape(height, width, 4).astype(np.uint8)
    writer.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python devtools/replay_capture_to_l21.py",
        description="offline capture H5 -> palm-local -> geometric retargeting -> L21 PyBullet",
    )
    parser.add_argument("--input-h5", type=Path, default=None,
                        help="raw capture H5 with left_hand_keypoints/right_hand_keypoints "
                             "(T,21,3); pass this or --angles-h5, not both")
    parser.add_argument("--angles-h5", type=Path, default=None,
                        help="retargeted 18D angle H5 with left_angles/right_angles (T,18); "
                             "skips the 21-landmark pipeline")
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--scene", choices=("robots", "hands"), default="robots",
                        help="robots: H1-2 / GR1-T2 / G1 side by side with their native "
                             "hands (default); hands: the LinkerHand L21 hands only")
    parser.add_argument("--sim-steps", type=int, default=8,
                        help="physics steps per frame for --scene robots (default 8)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--calibration-frames", type=int, default=15)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "tmp_probe" / "offline_l21")
    parser.add_argument("--render-every", type=int, default=25, help="0 = do not save PNG")
    parser.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=list(DEFAULT_VIEWS))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--distance", type=float, default=0.8)
    parser.add_argument("--target-z", type=float, default=0.28,
                        help="camera look-at height (L21 hands sit at z=0.3)")
    parser.add_argument("--video", type=str, default=None,
                        help="write this mp4 into --out-dir (iso view, every frame, size = --width x --height)")
    parser.add_argument("--gui", action="store_true", help="PyBullet GUI instead of DIRECT")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="with --gui, keep the last pose on screen for this long")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.input_h5 is None) == (args.angles_h5 is None):
        parser.error("pass exactly one of --input-h5 (raw 21 landmarks) "
                     "or --angles-h5 (18D angles)")
    try:
        return run(args)
    except Exception as error:  # noqa: BLE001 - CLI boundary reports instead of raising
        print(f"[FAIL] offline replay stopped: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
