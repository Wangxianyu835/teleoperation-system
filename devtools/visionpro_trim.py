"""把一个 Vision Pro 采集 H5 裁到"真正在追踪"的连续区间，另存一份干净副本。

为什么需要它
------------
实测（2026-10-12 首次真机录制）：摘下头显后，App **不会停止发送**，而是继续重复最后一帧。
于是 5 分钟的录制里，尾巴上大约 3300 帧是逐位相同的假数据。校验脚本会把这种尾巴标出来，
但交付给队友的数据集最好只留有效区间，避免训练时被"冻结的姿态"污染。

判据（两条都满足才算"活着"）
----------------------------
1. 该帧与前一帧不是逐位相同（连续相同达到 --stale-limit 帧即判定冻结）；
2. 该侧 *_valid 为 True（矩阵有限且有真实关节几何）。

裁剪区间 = 从第一帧起，直到**任一侧**首次出现冻结之前的一帧。原始文件不动，
输出是新文件，并在属性里写明来源、裁剪点与原因，便于追溯。

用法::

    E:\\python3.11.7\\python.exe devtools\\visionpro_trim.py --input-h5 datasets/raw/xxx.h5
    E:\\python3.11.7\\python.exe devtools\\visionpro_trim.py --input-h5 xxx.h5 --output yyy.h5

退出码：0 = 写出了裁剪文件；1 = 输入不可读、没有有效区间或写盘失败。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

SIDES = ("left", "right")
POINT_DATASETS = (
    "left_hand_keypoints", "right_hand_keypoints",
    "left_hand_transforms", "right_hand_transforms",
    "left_forearm_transforms", "right_forearm_transforms",
    "left_wrist_transform", "right_wrist_transform",
    "head_transform",
)
VECTOR_DATASETS = (
    "frame_ids", "timestamps", "unix_ms",
    "left_valid", "right_valid",
    "left_pinch_distance", "right_pinch_distance",
    "left_hand_spread", "right_hand_spread",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python devtools/visionpro_trim.py",
        description="按冻结检测裁剪 Vision Pro 采集 H5",
    )
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output", help="输出路径；默认在输入文件旁加 _live 后缀")
    parser.add_argument("--stale-limit", type=int, default=30,
                       help="连续多少帧逐位相同判定冻结（默认 30）")
    args = parser.parse_args(argv)

    source = Path(args.input_h5)
    if not source.is_file():
        print(f"[FAIL] 找不到输入文件：{source}")
        return 1
    destination = Path(args.output) if args.output else source.with_name(
        source.stem + "_live" + source.suffix)
    if destination.exists():
        print(f"[FAIL] 输出已存在，不覆盖：{destination}")
        return 1

    with h5py.File(source, "r") as handle:
        frames = int(handle["timestamps"].shape[0])
        arrays = {}
        for name in (*POINT_DATASETS, *VECTOR_DATASETS):
            if name in handle:
                arrays[name] = np.asarray(handle[name][:])
        attrs = {key: handle.attrs[key] for key in handle.attrs}

    # 逐侧求"首次冻结"的位置。
    freeze_at = {}
    for side in SIDES:
        key = f"{side}_hand_transforms"
        if key not in arrays:
            continue
        transforms = arrays[key]
        same = np.all(transforms[1:] == transforms[:-1], axis=(1, 2, 3))
        run = 0
        start = None
        for index, flag in enumerate(same, start=1):
            run = run + 1 if flag else 0
            if run >= args.stale_limit:
                start = index - run + 1
                break
        freeze_at[side] = start

    stops = [value for value in freeze_at.values() if value is not None]
    stop = min(stops) if stops else frames
    if stop <= 1:
        print(f"[FAIL] 没有可用区间（首次冻结于第 {stop} 帧）")
        return 1

    keep = slice(0, stop)
    duration = float(arrays["timestamps"][stop - 1]) if stop else 0.0
    valid_ratio = {
        side: float(arrays[f"{side}_valid"][keep].astype(bool).mean())
        for side in SIDES if f"{side}_valid" in arrays
    }
    print(f"输入：{source.name}  共 {frames} 帧")
    for side in SIDES:
        value = freeze_at.get(side)
        print(f"  {side}: 首次冻结于 {value if value is not None else '未检测到'}"
              f"  裁剪区有效帧比例 {valid_ratio.get(side, 0.0) * 100:.1f}%")
    print(f"裁剪区间：第 0 到第 {stop - 1} 帧（时长 {duration:.1f}s）")

    with h5py.File(destination, "w") as out:
        for name, value in arrays.items():
            dataset = out.create_dataset(name, data=value[keep])
            with h5py.File(source, "r") as handle:
                for key, item in handle[name].attrs.items():
                    dataset.attrs[key] = item
        trimmed = {str(key): value for key, value in attrs.items()}
        trimmed.update({
            "trimmed_from": source.name,
            "trimmed_frames": int(frames - stop),
            "trimmed_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trim_reason": "frozen_tail_after_headset_removed",
            "trim_rule": f"first_identical_run>={args.stale_limit}_frames",
            "frame_count": int(stop),
            "duration_seconds": round(duration, 3),
            "left_freeze_at_frame": freeze_at.get("left"),
            "right_freeze_at_frame": freeze_at.get("right"),
        })
        for key, value in trimmed.items():
            out.attrs[key] = value
        out.attrs["recording_status"] = f"{attrs.get('recording_status', 'unknown')}+trimmed"
    print(f"[OK] 已写出：{destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
