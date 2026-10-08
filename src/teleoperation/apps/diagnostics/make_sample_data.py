"""生成示例数据文件 —— 把「契约 G / H」变成可执行文档

用法：
    python -m teleoperation tools make-sample-data --kind hand       # 契约G：人类手部数据
    python -m teleoperation tools make-sample-data --kind actions    # 契约H：动作序列
    python -m teleoperation tools make-sample-data --kind all        # 两个都生成
    python -m teleoperation tools make-sample-data --kind hand --side right --frames 150

生成的文件放在 datasets/samples/ 下，可直接用 HDF5 查看器打开，
或作为队友实现时的「格式对照」。

契约定义见 docs/contracts.md
"""
import argparse
import os
import sys

import numpy as np

from teleoperation.paths import PROJECT_ROOT
ROOT = str(PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT_DIR = os.path.join(ROOT, 'datasets', 'samples')

# MediaPipe 手部 21 个关键点的官方顺序（契约G 要求必须按此顺序）
MEDIAPIPE_HAND_LANDMARKS = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]

# 每根手指在 21 点里的索引链（MCP -> PIP -> DIP -> TIP）
FINGER_CHAINS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}
FINGER_Y = {'thumb': -0.04, 'index': -0.02, 'middle': 0.0,
            'ring': 0.02, 'pinky': 0.04}
SEG_LEN = {'thumb': 0.035, 'index': 0.040, 'middle': 0.045,
           'ring': 0.040, 'pinky': 0.032}


def make_hand_keypoints(n_frames=120, fps=30.0, side='right'):
    """生成一段「手指开合」的合成手部关键点序列

    Returns:
        kp3d (T,21,3), kp2d (T,21,2), wrist (T,7), timestamps (T,)
    """
    t = np.arange(n_frames) / fps
    grasp = 0.5 + 0.45 * np.sin(2 * np.pi * 0.5 * t)   # 0(张开)~1(握紧)

    kp3d = np.zeros((n_frames, 21, 3), dtype=np.float32)
    handedness = -1.0 if side == 'right' else 1.0

    for f in range(n_frames):
        g = grasp[f]
        kp3d[f, 0] = [0.0, 0.0, 0.0]                    # WRIST 在原点
        for fname, chain in FINGER_CHAINS.items():
            p = np.array([0.02, FINGER_Y[fname], 0.0])  # MCP 起点
            kp3d[f, chain[0]] = p
            seg = SEG_LEN[fname]
            for j, idx in enumerate(chain[1:]):
                ang = g * (0.6 + 0.35 * j) * (np.pi / 2)
                p = p + np.array([seg * np.cos(ang), 0.0,
                                  -seg * np.sin(ang) * handedness])
                kp3d[f, idx] = p

    kp2d = kp3d[:, :, :2].copy()      # 2D：仅调试用（正交投影）

    wrist = np.zeros((n_frames, 7), dtype=np.float32)
    wrist[:, 0] = 0.05 * np.sin(2 * np.pi * 0.2 * t)   # [x,y,z,qx,qy,qz,qw]
    wrist[:, 1] = 0.5
    wrist[:, 2] = 0.9
    wrist[:, 6] = 1.0

    return kp3d, kp2d, wrist, t.astype(np.float64)


def main(args=None):
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    print('=' * 68)
    print('生成示例数据（契约 G / H）')
    print('=' * 68)

    # ---------- 契约G：人类手部数据 ----------
    if args.kind in ('hand', 'all'):
        print('\n【契约G】人类手部数据  ->  datasets/samples/'
              f'human_hand_demo_{args.side}.h5')
        kp3d, kp2d, wrist, ts = make_hand_keypoints(
            args.frames, args.fps, args.side)
        print(f"  keypoints_3d = {kp3d.shape}    "
              f"keypoints_2d = {kp2d.shape}")
        print(f"  wrist_pose   = {wrist.shape}    timestamps = {ts.shape}")
        print(f"  21 点顺序（前 5）: {MEDIAPIPE_HAND_LANDMARKS[:5]}")
        write_hand_file(
            os.path.join(OUT_DIR, f'human_hand_demo_{args.side}.h5'),
            kp3d, kp2d, wrist, ts, side=args.side, fps=args.fps)

    # ---------- 契约H：动作序列 ----------
    if args.kind in ('actions', 'all'):
        print(f'\n【契约H】动作序列  ->  datasets/samples/'
              f'actions_demo_{args.robot}.h5')
        import importlib.util
        from teleoperation.apps.replay import replay_actions as ra

        from teleoperation.apps.simulation import create_environment as SimulationEnv, run_episode, evaluate_task, benchmark_all_tasks
        env = SimulationEnv(robot_type=args.robot, task_name=args.task,
                            render=False, record=False)
        try:
            env.reset(randomize=False)
            dim = env.action_dim
        finally:
            env.close()

        actions = ra.make_dummy_actions(dim, n_steps=args.frames,
                                        fps=240.0)
        print(f"  actions = {actions.shape}  (action_dim={dim})")
        meta = {'robot_type': args.robot, 'task_name': args.task,
                'fps': 240.0, 'source': 'synthetic',
                'algo': 'sine-wave-dummy'}
        ra.save_action_file(
            os.path.join(OUT_DIR, f'actions_demo_{args.robot}.h5'),
            actions, meta)

    print('\n完成。')


from teleoperation.data.samples import write_hand_file
