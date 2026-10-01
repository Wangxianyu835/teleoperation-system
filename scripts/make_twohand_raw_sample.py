"""生成一份【原始】双手关键点 H5（重定向流水线的入口数据，未对齐）

用途
----
队友的重定向链路需要「原始双手关键点录像」作为输入，但仓库里只有
**别人机器上跑出来的成品**（datasets/raw/retarget_twohand_153542.h5），
没有可复现的原始输入。本脚本用合成运动造一份，方便：

    * 在没有真实采集数据时，把整条链路（对齐 -> 训练 -> 导出 -> 回放）跑通
    * 当回归/冒烟测试的固定输入

数据契约（与 retargeting/data.py 的 root 布局一致）
--------------------------------------------------
    left_hand_keypoints   (T, 21, 3)  float32   MediaPipe 21 点顺序
    right_hand_keypoints  (T, 21, 3)  float32
    frame_ids             (T,)        int64     连续帧号
    timestamps            (T,)        float64   秒

[注意] 本脚本**故意不写** coordinate_frame 属性 —— 这样才是「原始」数据，
       必须先跑 scripts/align_h5_coordinates.py 对齐（训练/导出会强制检查）。

用法
----
    python scripts/make_twohand_raw_sample.py
    python scripts/make_twohand_raw_sample.py --frames 600 --fps 30
    python scripts/make_twohand_raw_sample.py --output tmp_motion/my_raw.h5

生成后完整链路见 docs/RETARGETING_PIPELINE.md（或一条命令：
python demo_retargeting_pipeline.py）。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 复用契约 G 的合成手（保证 21 点顺序、量纲与 sample 数据一致）
from make_sample_data import make_hand_keypoints  # noqa: E402

DEFAULT_OUTPUT = os.path.join('tmp_motion', 'raw_twohand_sample.h5')

# 左右手在 x 轴上分开（身份跟踪按「掌心位移」判定，重叠会被判成换手）
LEFT_OFFSET = np.array([0.70, 0.0, 0.0], dtype=np.float32)
RIGHT_OFFSET = np.array([0.30, 0.0, 0.0], dtype=np.float32)


def make_twohand_raw(frames=240, fps=30.0):
    """合成一段双手开合运动 -> (left, right, frame_ids, timestamps)"""
    left = make_hand_keypoints(frames, fps, side='left')[0] + LEFT_OFFSET
    right = make_hand_keypoints(frames, fps, side='right')[0] + RIGHT_OFFSET
    frame_ids = np.arange(frames, dtype=np.int64)
    timestamps = (frame_ids / float(fps)).astype(np.float64)
    return (left.astype(np.float32), right.astype(np.float32),
            frame_ids, timestamps)


def write_raw_h5(path, left, right, frame_ids, timestamps, fps=30.0):
    """按 root 布局写出【未对齐】的原始双手关键点文件"""
    import h5py
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('left_hand_keypoints', data=left)
        f.create_dataset('right_hand_keypoints', data=right)
        f.create_dataset('frame_ids', data=frame_ids)
        f.create_dataset('timestamps', data=timestamps)
        f.attrs['source'] = 'synthetic-twohand-raw'
        f.attrs['fps'] = float(fps)
        f.attrs['landmark_order'] = 'mediapipe21'
        # [注意] 这里【不】写 coordinate_frame：原始数据必须先对齐


def main():
    ap = argparse.ArgumentParser(
        description='生成原始（未对齐）双手关键点 H5，供重定向链路做入口数据')
    ap.add_argument('--output', default=DEFAULT_OUTPUT)
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--fps', type=float, default=30.0)
    args = ap.parse_args()

    left, right, frame_ids, timestamps = make_twohand_raw(args.frames, args.fps)
    write_raw_h5(args.output, left, right, frame_ids, timestamps, args.fps)

    size_kb = os.path.getsize(args.output) / 1024.0
    print('=' * 72)
    print('原始双手关键点已写出（未对齐）')
    print('=' * 72)
    print(f'  path            ：{args.output}  ({size_kb:.1f} KB)')
    print(f'  frames          ：{args.frames}   fps={args.fps}')
    print(f'  left  keypoints ：{left.shape}  right keypoints：{right.shape}')
    print('  coordinate_frame：<未设置>  <- 原始数据，下一步必须先对齐')
    print()
    print('下一步：')
    print('  python scripts/align_h5_coordinates.py '
          f'--input {args.output} \\')
    print('      --output tmp_motion/aligned_twohand_sample.h5')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
