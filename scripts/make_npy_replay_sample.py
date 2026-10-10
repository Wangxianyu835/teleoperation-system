"""生成最小的 .npy 回放样本（dict 帧序列）—— 让「npy 回放适配器」的复核可复现

背景
----
`input_adapters/npy_replay_adapter.py`（队友的）读取一个 ``.npy`` 帧日志，每帧是
``dict``：``frame_id`` / ``timestamp`` / ``left_hand`` / ``right_hand``，
手部关键点形状 ``(21, 3)``。``HandWindowBuffer`` 只在**攒满 3 帧**时才吐出一个 payload，
所以 **payload 数 = 帧数 - 2**（12 帧 -> 10；8 帧 -> 6）。

为什么要有这个脚本
------------------
样本文件 ``tmp_motion/npy_replay_sample.npy`` **不入库**（``tmp_*/`` 被 .gitignore 排除），
原本只有生成它的临时脚本躺在 tmp_motion 里 —— 别人 clone 后无法复现
docs/RETARGETING_PIPELINE.md 第 9 节写的那条复核命令。本脚本入库，专门解决这件事。

用法（在项目根目录）
--------------------
    python scripts/make_npy_replay_sample.py                          # 12 帧 -> payloads=10
    python scripts/make_npy_replay_sample.py --frames 8 --drop-right   # 8 帧、右手缺失 -> payloads=6
    python scripts/make_npy_replay_sample.py --out tmp_motion/my.npy

复核（完整两步）
----------------
    python scripts/make_npy_replay_sample.py
    python scripts/validate_retarget_input.py tmp_motion/npy_replay_sample.npy
    # 期望：payloads=10 / source=mediapipe_approx / left_shape=(3, 25, 3) / result=ok，退出码 0

只依赖 numpy，不经过 retargeting 包（队友算法装了没装都能跑）。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DEFAULT_OUT = os.path.join('tmp_motion', 'npy_replay_sample.npy')
SEED = 0                      # 固定种子 -> 生成结果可复现


def make_frames(n_frames=12, fps=30.0, drop_right=False):
    """合成帧序列：掌心大约在 z=0.4 m，21 个关键点带小幅抖动（确定性随机）

    Args:
        n_frames: 帧数（payload 数 = n_frames - 2）
        fps: 时间戳步长 = 1/fps
        drop_right: True 时右手留空（None），用来验证「单手 payload」路径
    Returns:
        list[dict]：可直接 np.save(..., allow_pickle=True) 的帧日志
    """
    rng = np.random.default_rng(SEED)
    frames = []
    for index in range(n_frames):
        hand = (rng.normal(scale=0.05, size=(21, 3))
                + np.array([0.0, 0.0, 0.4])).astype(np.float32)
        frames.append({
            'frame_id': index,
            'timestamp': index / fps,
            'left_hand': hand,
            'right_hand': None if drop_right else hand.copy(),
        })
    return frames


def main():
    ap = argparse.ArgumentParser(
        description='生成 .npy 回放样本（dict 帧序列），供 validate_retarget_input.py 复核')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help=f'输出路径（相对项目根；默认 {DEFAULT_OUT}）')
    ap.add_argument('--frames', type=int, default=12,
                    help='帧数（payload 数 = 帧数 - 2，默认 12 -> 10）')
    ap.add_argument('--fps', type=float, default=30.0, help='时间戳步长 = 1/fps')
    ap.add_argument('--drop-right', action='store_true',
                    help='右手留空，验证单手 payload（validate 会打印 right_shape=None）')
    args = ap.parse_args()

    frames = make_frames(args.frames, args.fps, args.drop_right)
    target = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(target, np.array(frames, dtype=object), allow_pickle=True)

    print(f'saved={args.out}')
    print(f'frames={len(frames)}')
    print(f'expected_payloads={max(0, len(frames) - 2)}')
    if args.drop_right:
        print('right_hand=None -> validate 期望 right_shape=None')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
