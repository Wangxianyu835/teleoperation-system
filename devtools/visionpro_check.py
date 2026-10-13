"""Vision Pro 采集自检：先证明"流是通的、形状是对的、数是像米的"，再谈训练。

两种用法
--------
1) 现场（连着串流，不落盘）::

       E:\\python3.11.7\\python.exe devtools\\visionpro_check.py --ip <头显IP> --frames 60

   打印每侧有效帧比例、25 点的手长中位数（判断单位是不是米）、腕部在世界坐标下
   三个轴的取值范围。把左手抬高/前伸时哪一轴在变，就是该轴的方向；本脚本只报告
   事实，不替你断言符号约定。

2) 离线（校验 visionpro_record.py 写出的 H5）::

       E:\\python3.11.7\\python.exe devtools\\visionpro_check.py --input-h5 datasets/raw/xxx.h5

判据（任何一条 [FAIL] 即退出码 1）
----------------------------------
1) 数据集齐全、形状自洽（帧数一致、点的维度对）
2) timestamps 严格递增、unix_ms 严格递增
3) 腕部/手部矩阵里没有 NaN 或 Inf（未检出的帧由 *_valid=False 标记，不是 NaN）
4) 有效帧里 25 点必须全有限；无有效帧只报 [WARN]
5) 属性声明齐全：source、timestamp_unit、missing_hand_policy、单位

尺度合理性（手长中位数落在 0.05 m 到 0.30 m）只报 [WARN]：它是数据检查，不是结构错误。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import h5py

SIDES = ("left", "right")
HAND_JOINTS = 25
# 项目 25 点里各指指尖：拇指 4、食指 9、中指 14、无名指 19、小指 24。
TIPS = {"thumb": 4, "index": 9, "middle": 14, "ring": 19, "little": 24}
WRIST = 0
# (T, 关节数, 4, 4)：一只手 25 个关节、前臂 2 个关节，每个关节一个 4x4 矩阵。
STACKED_MATRIX_DATASETS = tuple(
    [f"{side}_hand_transforms" for side in SIDES]
    + [f"{side}_forearm_transforms" for side in SIDES]
)
# (T, 4, 4)：单个位姿。
PLAIN_MATRIX_DATASETS = tuple(
    [f"{side}_wrist_transform" for side in SIDES] + ["head_transform"]
)
REQUIRED_MATRIX_DATASETS = (*STACKED_MATRIX_DATASETS, *PLAIN_MATRIX_DATASETS)
REQUIRED_POINT_DATASETS = tuple(f"{side}_hand_keypoints" for side in SIDES)
REQUIRED_VECTOR_DATASETS = tuple(
    [f"{side}_valid" for side in SIDES]
    + [f"{side}_pinch_distance" for side in SIDES]
    + ["frame_ids", "timestamps", "unix_ms"]
)
REQUIRED_ATTRS = (
    "source",
    "source_landmark_space",
    "derived_alignment_status",
    "length_unit",
    "timestamp_unit",
    "missing_hand_policy",
)


def _hand_lengths(points: np.ndarray) -> dict:
    """Wrist -> each fingertip distance per frame, ignoring degenerate rows."""
    usable = points[np.isfinite(points).all(axis=(1, 2)) & np.any(points, axis=(1, 2))]
    if usable.shape[0] == 0:
        return {}
    lengths = {}
    for name, tip in TIPS.items():
        offsets = usable[:, tip, :] - usable[:, WRIST, :]
        lengths[name] = float(np.median(np.linalg.norm(offsets, axis=1)))
    return lengths


def check_file(path: Path) -> int:
    failures: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        print(f"[FAIL] 文件不存在：{path}")
        return 1
    with h5py.File(path, "r") as handle:
        missing = [
            name for name in (
                *REQUIRED_MATRIX_DATASETS, *REQUIRED_POINT_DATASETS, *REQUIRED_VECTOR_DATASETS)
            if name not in handle
        ]
        if missing:
            failures.append(f"缺少数据集：{', '.join(missing)}")
        attrs = {key: handle.attrs[key] for key in handle.attrs}
        for name in REQUIRED_ATTRS:
            if name not in attrs:
                failures.append(f"缺少属性 {name}")
        arrays = {}
        for name in handle.keys():
            if isinstance(handle[name], h5py.Dataset):
                arrays[name] = np.asarray(handle[name][:])

    frames = int(arrays.get("timestamps", np.zeros(0)).shape[0])
    print(f"文件：{path}")
    print(f"帧数：{frames}  大小：{path.stat().st_size / 1024.0:.0f} KB")
    if frames == 0:
        failures.append("文件里没有任何帧")

    for name in REQUIRED_MATRIX_DATASETS:
        if name not in arrays:
            continue
        array = arrays[name]
        if array.shape[0] != frames:
            failures.append(f"{name} 帧数 {array.shape[0]} 与 timestamps 不一致")
        expected = (frames, 4, 4) if name in PLAIN_MATRIX_DATASETS else (frames, None, 4, 4)
        if len(array.shape) != len(expected) or array.shape[-2:] != (4, 4):
            failures.append(f"{name} 形状应为 {'(T,4,4)' if len(expected) == 3 else '(T,N,4,4)'}，实际 {array.shape}")
        elif len(expected) == 3 and array.shape[1:] != (4, 4):
            failures.append(f"{name} 形状应为 (T,4,4)，实际 {array.shape}")
        if not np.isfinite(array).all():
            failures.append(f"{name} 含 NaN/Inf")
    for name in REQUIRED_POINT_DATASETS:
        if name in arrays:
            array = arrays[name]
            if array.shape[0] != frames:
                failures.append(f"{name} 帧数 {array.shape[0]} 与 timestamps 不一致")
            if array.shape[1:] != (HAND_JOINTS, 3):
                failures.append(f"{name} 形状应为 (T,{HAND_JOINTS},3)，实际 {array.shape}")
    for name in ("timestamps", "unix_ms", "frame_ids"):
        if name in arrays and arrays[name].shape[0] != frames:
            failures.append(f"{name} 长度与 timestamps 不一致")
    if "timestamps" in arrays and frames > 1:
        if not np.all(np.diff(arrays["timestamps"]) > 0):
            failures.append("timestamps 不是严格递增")
    if "unix_ms" in arrays and frames > 1:
        if not np.all(np.diff(arrays["unix_ms"]) > 0):
            failures.append("unix_ms 不是严格递增")

    for side in SIDES:
        valid_key = f"{side}_valid"
        points_key = f"{side}_hand_keypoints"
        transforms_key = f"{side}_hand_transforms"
        if transforms_key in arrays and frames > 1:
            # 冻结尾段：摘下头显后 App 会继续发同一帧，看起来"有效"其实是假数据。
            transforms = arrays[transforms_key]
            same = np.all(transforms[1:] == transforms[:-1], axis=(1, 2, 3))
            run = 0
            freeze_start = None
            for index, flag in enumerate(same, start=1):
                run = run + 1 if flag else 0
                if run >= 30:
                    freeze_start = index - run + 1
                    break
            if freeze_start is not None:
                stamp = float(arrays["timestamps"][freeze_start]) if "timestamps" in arrays else float("nan")
                failures.append(
                    f"{side}_hand_transforms 从第 {freeze_start} 帧（t={stamp:.1f}s）起连续 30 帧以上"
                    "逐位完全相同：这是摘掉头显后的冻结数据，不是追踪结果，必须裁掉")
        if valid_key not in arrays or points_key not in arrays or frames == 0:
            continue
        valid = arrays[valid_key].astype(bool)
        points = arrays[points_key]
        ratio = float(valid.sum()) / float(frames)
        print(f"[{side}] 有效帧 {int(valid.sum())}/{frames}（{ratio * 100:.1f}%）")
        if valid.any():
            finite = np.isfinite(points[valid]).all(axis=(1, 2))
            if not finite.all():
                failures.append(f"{side}_hand_keypoints 在有效帧里仍有 NaN/Inf")
            base = points[np.isfinite(points).all(axis=(1, 2)) & np.any(points, axis=(1, 2))]
            if base.shape[0] == 0:
                # 典型情形：头显连着但没追踪到手，发来的全是单位矩阵。
                failures.append(
                    f"{side} 标了 {int(valid.sum())} 个有效帧，但里面没有任何关节几何"
                    "（很可能是连着但没追踪到手的单位矩阵）")
            else:
                lengths = _hand_lengths(points)
                text = ", ".join(f"{name}={value * 1000.0:.0f}mm" for name, value in lengths.items())
                print(f"    腕到指尖中位距离：{text}")
                reference = lengths.get("index", float("nan"))
                if not np.isfinite(reference) or not (0.05 <= reference <= 0.30):
                    warnings.append(
                        f"{side} 腕到食指指尖中位数 {reference:.4f} 不在 0.05-0.30 m："
                        "单位可能不是米，或坐标系不是腕相对"
                    )
                row_norms = np.linalg.norm(base, axis=2)
                print(f"    点位范数 中位={np.median(row_norms) * 1000.0:.1f}mm "
                      f"最大={row_norms.max() * 1000.0:.1f}mm")
        else:
            warnings.append(f"{side} 没有任何有效帧")
        key = f"{side}_pinch_distance"
        if key in arrays:
            values = arrays[key][np.isfinite(arrays[key])]
            if values.size:
                print(f"    pinch_distance 中位={np.median(values) * 1000.0:.1f}mm")
        spread_key = f"{side}_hand_spread"
        if spread_key in arrays:
            spreads = arrays[spread_key][np.isfinite(arrays[spread_key])]
            if spreads.size:
                print(f"    关节跨度 中位={np.median(spreads) * 100.0:.1f}cm "
                      f"最大={float(spreads.max()) * 100.0:.1f}cm")

    print("属性：")
    for key in REQUIRED_ATTRS:
        if key in attrs:
            print(f"  {key} = {attrs[key]}")
    print(f"  derived_origin = {attrs.get('derived_origin')}")
    print(f"  arm_source_note = {attrs.get('arm_source_note')}")

    for item in warnings:
        print(f"[WARN] {item}")
    for item in failures:
        print(f"[FAIL] {item}")
    print("[OK] 结构检查通过" if not failures else f"[FAIL] 共 {len(failures)} 项结构问题")
    return 1 if failures else 0


def check_stream(ip: str, frames: int, target_fps: float) -> int:
    try:
        from avp_stream import VisionProStreamer
    except ImportError:
        print("[FAIL] 缺少 avp_stream：pip install --upgrade avp_stream")
        return 1
    # ip 既可以是头显显示的局域网地址，也可以是跨网模式下的房间码。
    streamer = VisionProStreamer(ip=ip)
    period = 1.0 / target_fps if target_fps > 0 else 0.0
    collected = {side: [] for side in SIDES}
    wrist_positions = {side: [] for side in SIDES}
    pinch = {side: [] for side in SIDES}
    valid_counts = {side: 0 for side in SIDES}
    seen = 0
    print(f"采样 {frames} 帧（目标 {target_fps} fps）；请让双手在视野内自然开合。")
    while seen < frames:
        payload = streamer.get_latest()
        for side in SIDES:
            key = f"{side}_fingers"
            value = None
            try:
                value = payload[key]
            except Exception:
                try:
                    value = getattr(payload, side).transforms
                except Exception:
                    value = None
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.shape[0] < HAND_JOINTS:
                continue
            collected[side].append(array[:HAND_JOINTS])
            wrist = np.asarray(array[0], dtype=np.float32)
            wrist_positions[side].append(wrist[:3, 3] if wrist.shape == (4, 4) else wrist[:3])
            try:
                pinch[side].append(float(payload[f"{side}_pinch_distance"]))
            except Exception:
                pass
            valid_counts[side] += 1
        seen += 1
        if period:
            time.sleep(period)

    print(f"采样完成：{seen} 帧")
    failures = []
    for side in SIDES:
        if not collected[side]:
            failures.append(f"{side}_fingers 一帧都没收到")
            continue
        transforms = np.stack(collected[side])
        from teleoperation.retargeting.hand.topology import ensure_hand25, visionpro_fingers_to_points

        points = np.stack([ensure_hand25(visionpro_fingers_to_points(item, side))
                          for item in transforms])
        lengths = _hand_lengths(points)
        text = ", ".join(f"{name}={value * 1000.0:.0f}mm" for name, value in lengths.items())
        print(f"[{side}] 收到 {valid_counts[side]}/{seen} 帧  腕到指尖中位距离：{text}")
        positions = np.stack(wrist_positions[side])
        for axis, name in enumerate("xyz"):
            column = positions[:, axis]
            print(f"    腕位置 {name}: min={column.min():.3f} max={column.max():.3f} "
                  f"跨度={column.max() - column.min():.3f} m")
        reference = lengths.get("index", float("nan"))
        if not np.isfinite(reference) or not (0.05 <= reference <= 0.30):
            print(f"[WARN] {side} 手长中位数 {reference:.4f} 不像米；请检查上游单位")
        if pinch[side]:
            values = np.asarray(pinch[side], dtype=np.float32)
            print(f"    pinch_distance: min={values.min():.3f} max={values.max():.3f} m")
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1
    print("提示：把左手抬高 20 cm 再跑一次，哪一轴变化最大就是竖直方向（用于核对轴约定）。")
    print("[OK] 串流结构与尺度检查通过")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python devtools/visionpro_check.py",
        description="Vision Pro 串流/H5 自检",
    )
    parser.add_argument("--ip", help="头显地址（现场自检用）")
    parser.add_argument("--room-code", help="跨网模式的房间码，与 --ip 二选一")
    parser.add_argument("--input-h5", help="校验 visionpro_record.py 写出的 H5")
    parser.add_argument("--frames", type=int, default=60, help="现场采样帧数，默认 60")
    parser.add_argument("--target-fps", type=float, default=30.0, help="现场采样节奏")
    args = parser.parse_args(argv)
    if args.input_h5 and (args.ip or args.room_code):
        print("[FAIL] --input-h5 与 --ip/--room-code 不能同时给")
        return 1
    if args.ip and args.room_code:
        print("[FAIL] --ip 与 --room-code 只能给一个")
        return 1
    if args.input_h5:
        return check_file(Path(args.input_h5))
    if args.ip or args.room_code:
        return check_stream(args.ip or args.room_code, args.frames, args.target_fps)
    print("用法：--input-h5 <文件.h5>  或  --ip <头显IP>  /  --room-code <房间码>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
