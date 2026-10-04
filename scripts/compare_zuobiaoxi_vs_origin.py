"""Synchronously replay raw MediaPipe hands and exported L21 URDF FK poses.

The source panel preserves the image XY projection used by read_hand_xyz.py.
The robot panel projects FK nodes, not URDF mesh surfaces. No inference,
coordinate alignment, or training is performed here. Timestamps are seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from retargeting.config import ANGLE_LIMITS, JOINT_EDGES, JOINT_NAMES, L21
from retargeting.data import load_twohand_h5
from retargeting.simulation import iter_angle_h5


SOURCE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
NAME_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
ROBOT_EDGES = tuple((NAME_INDEX[a], NAME_INDEX[b]) for a, b in JOINT_EDGES)
ROBOT_TIPS = tuple(NAME_INDEX[f"{finger}_tip"] for finger in
                   ("thumb", "index", "middle", "ring", "pinky"))
COLORS = {"left": (255, 145, 65), "right": (80, 220, 100)}  # BGR
WIDTH, ROW_HEIGHT, HEADER = 600, 410, 80
WINDOW = "MediaPipe original vs L21 URDF FK"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True,
                        help="Original recorder H5 containing normalized 21-point hands")
    parser.add_argument("--angle-h5", type=Path, required=True,
                        help="Angle H5 previously produced by retargeting export")
    parser.add_argument("--side", choices=("both", "left", "right"), default="both")
    parser.add_argument("--view", choices=("yz", "xz", "xy", "iso"), default="yz")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed relative to source timestamps (seconds)")
    parser.add_argument("--frame", type=int, default=0, help="Starting array frame index")
    parser.add_argument("--paused", action="store_true", help="Start paused")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--labels", action="store_true", help="Show point indices")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "picture" / "compare_zuobiaoxi_vs_origin")
    parser.add_argument("--snapshot", type=Path,
                        help="Save the selected frame as a PNG and exit without opening a window")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and greater than zero")
    if args.image_width <= 0 or args.image_height <= 0:
        parser.error("Image dimensions must be greater than zero")
    if args.snapshot is not None and args.snapshot.suffix.lower() != ".png":
        parser.error("--snapshot must have a .png extension")
    return args


def load_recordings(source_path, angle_path, sides):
    ids, timestamps, source = load_twohand_h5(source_path, require_aligned=False)
    if len(ids) == 0:
        raise ValueError("Source H5 has no frames")
    if np.any(np.diff(ids) <= 0) or np.any(np.diff(timestamps) < 0):
        raise ValueError("Source frame IDs must increase and timestamps must not decrease")
    for side in sides:
        if source[side].shape != (len(ids), 21, 3):
            raise ValueError("Use the ORIGINAL 21-point recorder H5, not an aligned 25-point H5")
        if not np.isfinite(source[side]).all():
            raise ValueError(f"Source {side} contains NaN or Inf")
    frames = list(iter_angle_h5(angle_path))
    if len(frames) != len(ids):
        raise ValueError("Source and angle H5 frame counts differ")
    angle_ids = np.asarray([f.frame_id for f in frames])
    angle_times = np.asarray([f.timestamp for f in frames])
    if not np.array_equal(ids, angle_ids):
        raise ValueError("Source and angle H5 frame IDs differ")
    if not np.allclose(timestamps, angle_times, rtol=0, atol=1e-6):
        raise ValueError("Source and angle H5 timestamps differ; select matching recordings")
    with h5py.File(angle_path, "r") as f:
        attrs = dict(f.attrs)
    valid = {side: np.asarray([getattr(f, f"{side}_valid") for f in frames], dtype=bool)
             for side in sides}
    angles = {side: np.stack([getattr(f, side) for f in frames]) for side in sides}
    limits = np.asarray(ANGLE_LIMITS)
    for side in sides:
        selected = angles[side][valid[side]]
        if np.any(selected < limits[:, 0] - 1e-4) or np.any(selected > limits[:, 1] + 1e-4):
            raise ValueError(f"Valid {side} predictions exceed configured joint limits")
        if np.any(np.abs(angles[side][:, 0]) > 1e-6):
            raise ValueError(f"{side} root angle must remain zero")
    return ids, timestamps, source, angles, valid, attrs


def compute_fk(angles, side):
    urdf = L21.left_urdf if side == "left" else L21.right_urdf
    fk = create_hand_kinematics(urdf, L21.hand_kinematics_config(), device="cpu",
                              scale_factor=L21.training.robot_scale)
    output = []
    with torch.no_grad():
        for start in range(0, len(angles), 128):
            batch = angles[start:start + 128]
            nodes = torch.zeros((len(batch), len(JOINT_NAMES)), dtype=torch.float32)
            nodes[:, :18] = torch.from_numpy(batch.astype(np.float32))
            output.append(fk.forward(nodes)[2].cpu().numpy())
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise ValueError(f"{side} FK contains NaN or Inf")
    return result


def project(points, view):
    if view == "yz":
        return points[..., [1, 2]]
    if view == "xz":
        return points[..., [0, 2]]
    if view == "xy":
        return points[..., [0, 1]]
    # Orthographic oblique projection: the horizontal axis follows XY,
    # the vertical axis is tilted toward +Z. The pose itself is not changed.
    azimuth, elevation = np.deg2rad(-60), np.deg2rad(20)
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    vertical = np.array([-np.sin(elevation) * np.cos(azimuth),
                         -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)])
    return np.stack((points @ horizontal, points @ vertical), axis=-1)


def text(canvas, message, position, color=(225, 225, 225), scale=0.5):
    cv2.putText(canvas, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_skeleton(canvas, pixels, edges, color, tips, labels):
    for a, b in edges:
        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), color, 2, cv2.LINE_AA)
    for i, pixel in enumerate(pixels):
        point = tuple(pixel)
        cv2.circle(canvas, point, 5 if i in tips else 3,
                   (0, 190, 255) if i in tips else color, -1, cv2.LINE_AA)
        if labels:
            text(canvas, str(i), (int(pixel[0]) + 4, int(pixel[1]) - 4), scale=0.35)


def source_panel(points, side, args):
    panel = np.full((ROW_HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
    valid = np.any(np.abs(points) > 1e-6)
    text(panel, f"{side.upper()} | Original MediaPipe XY | {'DETECTED' if valid else 'MISSING'}",
         (15, 25), COLORS[side])
    text(panel, "Image coordinates: X right, Y down; no alignment applied", (15, 47), scale=0.44)
    if valid:
        scale = min((WIDTH - 40) / args.image_width, (ROW_HEIGHT - 90) / args.image_height)
        size = np.array([args.image_width, args.image_height]) * scale
        offset = np.array([(WIDTH - size[0]) / 2, 65 + (ROW_HEIGHT - 90 - size[1]) / 2])
        pixels = np.rint(points[:, :2] * size + offset).astype(np.int32)
        draw_skeleton(panel, pixels, SOURCE_EDGES, COLORS[side], (4, 8, 12, 16, 20), args.labels)
    else:
        text(panel, "No source hand detected in this frame", (100, 215))
    return panel


def robot_panel(points, side, state, args, bounds):
    panel = np.full((ROW_HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
    color = COLORS[side] if state == "VALID" else (125, 125, 125)
    text(panel, f"{side.upper()} | L21 URDF FK ({args.view.upper()}) | {state}",
         (15, 25), color)
    axes = {"yz": "horizontal +Y, vertical +Z", "xz": "horizontal +X, vertical +Z",
            "xy": "horizontal +X, vertical +Y", "iso": "orthographic oblique projection"}
    text(panel, axes[args.view] + "; display scale only", (15, 47), scale=0.44)
    projected = project(points, args.view)
    lower, upper = bounds
    extent = upper - lower
    scale = min((WIDTH - 100) / extent[0], (ROW_HEIGHT - 130) / extent[1])
    pixels = (projected - (lower + upper) / 2) * np.array([scale, -scale])
    pixels += np.array([WIDTH / 2, 65 + (ROW_HEIGHT - 100) / 2])
    draw_skeleton(panel, np.rint(pixels).astype(np.int32), ROBOT_EDGES, color, ROBOT_TIPS,
                  args.labels)
    if state != "VALID":
        message = "Invalid prediction: holding last valid angles" if state == "HOLD" else \
                  "No earlier valid prediction: displaying zero-angle reference"
        text(panel, message, (15, ROW_HEIGHT - 15), (100, 180, 255), scale=0.44)
    return panel


def render(index, ids, timestamps, source, positions, valid, seen_valid, args, sides, paused):
    header = np.full((HEADER, WIDTH * 2, 3), 28, dtype=np.uint8)
    status = "PAUSED" if paused else "PLAYING"
    text(header, f"index={index}/{len(ids)-1}  frame_id={ids[index]}  "
         f"time={timestamps[index]-timestamps[0]:.3f}s  speed={args.speed:g}x  {status}",
         (15, 23), scale=0.62)
    text(header, "q/Esc quit | Space pause | a/d previous/next | r restart | s save PNG | "
         "1 YZ  2 XZ  3 XY  4 ISO", (15, 48), scale=0.48)
    text(header, "Left: original image XY. Right: robot projection. Panel sizes do not imply equal physical scale.",
         (15, 69), scale=0.46)
    rows = [header]
    for side in sides:
        projected = project(positions[side], args.view).reshape(-1, 2)
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        margin = np.maximum((upper - lower) * 0.1, 0.005)
        state = "VALID" if valid[side][index] else ("HOLD" if seen_valid[side][index] else "ZERO")
        rows.append(np.hstack((source_panel(source[side][index], side, args),
                               robot_panel(positions[side][index], side, state, args,
                                           (lower - margin, upper + margin)))))
    return np.vstack(rows)


def save_png(path, canvas):
    path.parent.mkdir(parents=True, exist_ok=True)
    # imencode/tofile also supports Windows paths containing Chinese characters.
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Could not encode screenshot: {path}")
    encoded.tofile(str(path))
    print(f"已保存图片：{path.resolve()}")


def main():
    args = parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    ids, times, source, angles, valid, attrs = load_recordings(args.source_h5, args.angle_h5, sides)
    if not 0 <= args.frame < len(ids):
        raise ValueError(f"--frame must be in [0, {len(ids)-1}]")
    print(f"原始数据：{args.source_h5.resolve()}\n角度数据：{args.angle_h5.resolve()}")
    print(f"总帧数：{len(ids)}；帧编号和时间戳匹配；仅回放已有预测，不重新推理")
    print(f"角度文件记录的 coordinate_alignment：{attrs.get('coordinate_alignment', 'UNDECLARED')}")
    if attrs.get("identity_tracking", False):
        print("角度导出启用了身份跟踪，原始左右手槽位可能与跟踪后的槽位不同。")
    positions = {}
    for side in sides:
        print(f"{side}：原始检测帧 {np.any(np.abs(source[side]) > 1e-6, axis=(1, 2)).sum()}，"
              f"有效预测帧 {valid[side].sum()}；预计算 FK……")
        positions[side] = compute_fk(angles[side], side)
    seen_valid = {side: np.maximum.accumulate(valid[side]) for side in sides}
    if args.snapshot:
        save_png(args.snapshot, render(args.frame, ids, times, source, positions, valid,
                                     seen_valid, args, sides, True))
        return 0

    print("回放就绪：点击画面窗口后操作；空格暂停，a/d 逐帧，1/2/3/4 切换机器人视角，q 退出。")
    print("按采集时间戳播放且不丢帧；绘制较慢时会减速。本窗口不用于衡量实时推理速度。")
    index, paused, dirty = args.frame, args.paused, True
    canvas = None
    deadline = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1200, 820 if len(sides) == 2 else 490)
    try:
        while True:
            if dirty:
                canvas = render(index, ids, times, source, positions, valid, seen_valid,
                                args, sides, paused)
                cv2.imshow(WINDOW, canvas)
                dirty = False
                if not paused:
                    interval = (times[index + 1] - times[index]) / args.speed \
                        if index + 1 < len(ids) else 0.03 / args.speed
                    deadline = time.perf_counter() + max(0.001, float(interval))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' '):
                paused = not paused
                dirty = True
            elif key in (ord('a'), ord('d')):
                index = min(len(ids) - 1, max(0, index + (1 if key == ord('d') else -1)))
                paused, dirty = True, True
            elif key == ord('r'):
                index, dirty = 0, True
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                args.view = ("yz", "xz", "xy", "iso")[key - ord('1')]
                dirty = True
            elif key == ord('s'):
                save_png(args.output_dir / f"{args.angle_h5.stem}_{args.side}_{index:06d}_{args.view}.png",
                         canvas)
            if not paused and not dirty and time.perf_counter() >= deadline:
                if index + 1 < len(ids):
                    index += 1
                elif args.loop:
                    index = 0
                else:
                    paused = True
                    print("回放结束，停留在最后一帧。按 r 后按空格可重新播放，或按 q 退出。")
                dirty = True
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, OSError, RuntimeError, cv2.error) as exc:
        print(f"回放失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
