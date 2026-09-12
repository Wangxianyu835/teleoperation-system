"""用机器人【原装的手】回放队友数据（降维映射）

为什么要这个脚本
----------------
队友的重定向目标手是 **LinkerHand L21**（17 可动关节），
而论文里三台机器人出厂装的是**各自的手**：

    H1-2   -> Inspire        12 关节/手
    GR1-T2 -> Fourier 原生手  11 关节/手
    G1     -> 轻量三指手       7 关节/手

本脚本把 18 维数据**降维**映射到机器人**原装**的手上 ——
机器人保持原装，不需要"换手"，适合与论文的 H1-2 基线对比。

[注意] 必须声明的代价（数据是有损的）
--------------------------------
L21 每根手指有 1 个 `*_mcp_roll`（侧摆），拇指还有 `cmc_roll`（对掌旋转），
**原装手上没有这些自由度，只能丢弃**：

    数据 17 个可动自由度  ->  原装手能表达的自由度
    H1-2     -> 12 / 17   （丢 4 个侧摆 + 拇指 cmc_roll）
    GR1-T2   -> 11 / 17   （再丢拇指 1 个）
    G1       ->  7 / 17   （只有食指/中指/拇指，无名指和小指整体丢弃）

映射原则（语义 1:1，不做无依据的"合并"）
--------------------------------------
    L21 的 mcp_pitch  <->  原装手的 proximal  （都是近端指节屈曲）
    L21 的 pip        <->  原装手的 intermediate/distal（都是远端指节屈曲）
    L21 的 *_mcp_roll ->   丢弃
    拇指 cmc_yaw/pitch ->   proximal_yaw/pitch
    拇指 mcp/ip        ->   intermediate/distal（各机器人略有差异）

符号方向自动处理
----------------
GR1-T2 和 G1 的屈曲关节限位是**负的**（如 `[-1.570, 0.000]`），
H1-2 是正的（`[0.000, 1.700]`）。本脚本用
    sign = +1 if |hi| >= |lo| else -1
自动判断，把数据映到正确的方向，避免手指"反着弯"。

用法
----
    python scripts/replay_hand_native.py --robot h1_2 --hand both --render
    python scripts/replay_hand_native.py --robot h1_2 --hand both --report
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from teleop.filters import detect_bad_frames, repair_bad_frames  # noqa: E402

# 映射表与工具统一放在 teleop/native_hand.py（正式接口，可被其它代码复用）
from teleop.native_hand import (  # noqa: E402
    DIM_NAMES, N_MOVABLE,
    build_mapping, coverage, dropped_dims, map_frame, read_joint_ranges,
)



def load_data(path):
    import h5py
    with h5py.File(path, 'r') as f:
        d = {
            'left_angles': np.asarray(f['left_angles'][:], dtype=np.float64),
            'right_angles': np.asarray(f['right_angles'][:], dtype=np.float64),
            'timestamps': np.asarray(f['timestamps'][:], dtype=np.float64),
        }
        for k in ('left_valid', 'right_valid'):
            d[k] = (np.asarray(f[k][:]).astype(bool) if k in f
                    else np.ones(len(d['timestamps']), bool))
    return d


# ----------------------------------------------------------------------
def main():
    import pybullet as p
    import pybullet_data

    ap = argparse.ArgumentParser(
        description='用机器人【原装的手】回放队友数据（降维映射）')
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--robot', default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--hand', default='both',
                    choices=['left', 'right', 'both'])
    ap.add_argument('--render', action='store_true', help='GUI 可视化')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--substeps', type=int, default=0,
                    help='每帧数据推进多少个物理步（默认 0 = 自动匹配数据时间，'
                         '约 8 步）。设小了会变成慢动作且关节滞后')
    ap.add_argument('--report', action='store_true',
                    help='只打印映射报告，不启动回放')
    ap.add_argument('--no-repair', action='store_true')
    args = ap.parse_args()

    print('=' * 96)
    print('原装手回放：18 维数据 -> 机器人自带灵巧手（降维映射）')
    print('=' * 96)
    print(f'机器人 = {args.robot}   手 = {args.hand}   数据 = {args.file}')

    data = load_data(args.file)
    from envs import RobotLoader

    cid = p.connect(p.GUI if args.render else p.DIRECT)
    if args.render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(1.0 / 240.0, physicsClientId=cid)
    p.loadURDF('plane.urdf', physicsClientId=cid)

    loader = RobotLoader(cid)
    robot_id = loader.load_robot(args.robot)['robot']

    sides = ['left', 'right'] if args.hand == 'both' else [args.hand]
    plans = {}
    for side in sides:
        ranges, names = read_joint_ranges(robot_id, cid)
        mapping = build_mapping(args.robot, side, ranges)
        dropped = dropped_dims(args.robot, side, mapping)

        print()
        print(f'--- {side} ---')
        print(f'  数据 {N_MOVABLE} 个可动维度 -> 原装手可表达 {len(mapping)} 个 '
              f'（覆盖率 {coverage(args.robot, side, mapping) * 100:.0f}%）')
        print(f'  {"数据维度":<22} {"语义":<18} {"原装手关节":<36} 符号')
        for d, jn, sg, lo, hi in mapping:
            print(f'  dim{d:<4}{DIM_NAMES[d]:<18} {jn:<36} {sg:+.0f}')
        print(f'  丢弃 {len(dropped)} 个维度（原装手没有这些自由度）:')
        print(f'    {[DIM_NAMES[d] for d in dropped]}')

        arr = data[f'{side}_angles']
        val = data[f'{side}_valid']
        bad = detect_bad_frames(arr, val)
        if bad and not args.no_repair:
            arr = repair_bad_frames(arr, bad)
            print(f'  已修复 {len(bad)} 个塌零坏帧（绝对下标）{bad}')
        plans[side] = (mapping, arr, val, names)

    if args.report:
        p.disconnect(cid)
        print()
        print('  （--report 模式：未启动回放）')
        return

    if args.render:
        kin = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.9, cameraYaw=135, cameraPitch=-12,
            cameraTargetPosition=[float(kin[0][0]), float(kin[0][1]), 1.0],
            physicsClientId=cid)
        print()
        print('  鼠标：左键旋转 / 右键平移 / 滚轮缩放')

    T = len(data['timestamps'])
    t0 = data['timestamps'][0]
    frame_dt = (float(np.median(np.diff(data['timestamps'])))
                if T > 1 else 1.0 / 30.0)
    dt = 1.0 / 240.0
    substeps = (args.substeps if args.substeps > 0
                else max(1, int(round(frame_dt / dt))))
    p.setTimeStep(dt, physicsClientId=cid)
    clip = {s: 0 for s in plans}
    tot = {s: 0 for s in plans}
    print(f'\n[回放] {T} 帧，开始...（数据 {frame_dt*1000:.1f} ms/帧，'
          f'物理步 {dt*1000:.2f} ms，每帧推进 {substeps} 步）')
    for i in range(T):
        for side, (mapping, arr, val, names) in plans.items():
            if not val[i]:
                continue
            joints, nclip = map_frame(arr[i], mapping)
            clip[side] += nclip
            tot[side] += len(mapping)
            for jn, v in joints.items():
                p.setJointMotorControl2(robot_id, names[jn],
                                        p.POSITION_CONTROL,
                                        targetPosition=v, force=200.0,
                                        physicsClientId=cid)
        for _ in range(substeps):
            p.stepSimulation(physicsClientId=cid)
        if args.render:
            wait = (data['timestamps'][i] - t0) / max(args.speed, 1e-6)
            time.sleep(max(0.0, min(wait, 0.05)))
        if (i + 1) % 200 == 0:
            print(f'    ... {i+1}/{T}')

    print('  回放完成。')
    for side in plans:
        if tot[side]:
            print(f'    {side}: 映射后贴到原装手限位的比例 '
                  f'{100.0 * clip[side] / tot[side]:.2f}%  '
                  f'({clip[side]}/{tot[side]})')
    if args.render:
        print('  窗口保持打开，可继续用鼠标查看。关闭窗口或 Ctrl+C 退出。')
        try:
            while p.isConnected(cid):
                p.stepSimulation(physicsClientId=cid)
                time.sleep(1.0 / 60.0)
        except (p.error, KeyboardInterrupt):
            pass
    else:
        p.disconnect(cid)


if __name__ == '__main__':
    main()


