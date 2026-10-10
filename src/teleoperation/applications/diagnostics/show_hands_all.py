"""三台机器人并排 + 手部由队友数据驱动（答辩 / 报告展示用）

画面内容
--------
    H1-2  /  GR1-T2  /  G1   三台论文机器人并排站立，
    各自用【原装】灵巧手，按**同一份队友重定向数据**同步屈伸。

范围声明（与当前算法能力一致）
------------------------------
    手指：OK   由队友重定向输出驱动（L21 18 维 -> 原装手降维映射）
    手臂：注意 数据里没有手臂关节角，因此保持中性姿态【静止】
          （重定向算法暂未输出手臂数据；这是已确认的阶段范围）

用法
----
    python -m teleoperation tools show-all-hands --render --loop 0
    python -m teleoperation tools show-all-hands --render --view front --loop 0
    python -m teleoperation tools show-all-hands --render --speed 0.5
    python -m teleoperation tools show-all-hands                 # 无头（只验证能跑）
"""
import argparse
import os
import sys
import time

import numpy as np

from teleoperation.paths import PROJECT_ROOT
ROOT = str(PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from teleoperation.retargeting.hand.postprocessing import (  # noqa: E402
    detect_bad_frames, detect_identity_swaps, repair_stream,
)
from teleoperation.robots.native_hand import N_MOVABLE, build_mapping, coverage
from teleoperation.applications.replay.hand_support import map_frame
from teleoperation.simulation.robot_loader import read_joint_ranges

# 三台机器人的横向摆位（米）
LAYOUT = [('h1_2', -2.4), ('gr1_t2', 0.0), ('g1', 2.4)]
COLORS = {'h1_2': [1.0, 0.35, 0.35],
          'gr1_t2': [0.35, 0.5, 1.0],
          'g1': [0.35, 1.0, 0.45]}


def joint_extent(p, body, cid):
    """合并某机器人所有 link 的 AABB -> (lo, hi)"""
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for i in range(-1, p.getNumJoints(body, physicsClientId=cid)):
        try:
            a, b = p.getAABB(body, i, physicsClientId=cid)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(a, dtype=float))
        hi = np.maximum(hi, np.asarray(b, dtype=float))
    return lo, hi


# ----------------------------------------------------------------------
def main(args=None):
    import pybullet as p
    import pybullet_data
    from teleoperation.simulation import RobotLoader

    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    print('=' * 96)
    print('三台机器人并排 + 手部由队友重定向数据驱动')
    print('=' * 96)
    print('  手指：由队友数据驱动 | 手臂：数据里没有，保持中性姿态静止')

    data = load_data(args.file)

    cid = p.connect(p.GUI if args.render else p.DIRECT)
    if args.render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF('plane.urdf', physicsClientId=cid)

    loader = RobotLoader(cid)
    scenes = []          # [(name, rid, plans, hold_ids, hold_tgt, hud, cov)]

    swaps = detect_identity_swaps(data['left_angles'], data['right_angles'],
                                  data['left_valid'], data['right_valid'])
    swap_by_side = {}
    for s in swaps:
        swap_by_side.setdefault(s['side'], []).append(s['frame'])
    if swaps:
        print(f'  身份切换检测：{len(swaps)} 处 -> '
              f'{[s["frame"] for s in swaps]}（用 hold 修复，不用插值）')

    for name, dx in LAYOUT:
        rid = loader.load_robot(name)['robot']
        # 横向摆开（URDF 默认都在原点，否则会重叠）
        bp, bo = p.getBasePositionAndOrientation(rid, physicsClientId=cid)
        p.resetBasePositionAndOrientation(
            rid, [bp[0] + dx, bp[1], bp[2]], bo, physicsClientId=cid)

        ranges, names = read_joint_ranges(rid, cid)
        plans = {}
        for side in ('left', 'right'):
            mp = build_mapping(name, side, ranges)
            if not mp:
                continue
            arr = data[f'{side}_angles']
            val = data[f'{side}_valid']
            bad = detect_bad_frames(arr, val)
            fr = sorted(set(swap_by_side.get(side, [])))
            if not args.no_repair:
                arr = repair_stream(arr, val, bad, fr)[0]
            plans[side] = (mp, arr, val, names)

        hand_ids = set()
        for mp, _a, _v, nms in plans.values():
            for _d, jn, *_ in mp:
                if jn in nms:
                    hand_ids.add(nms[jn])
        hold_ids, hold_tgt = [], []
        for i in range(p.getNumJoints(rid, physicsClientId=cid)):
            info = p.getJointInfo(rid, i, physicsClientId=cid)
            if info[2] == p.JOINT_FIXED or i in hand_ids:
                continue
            hold_ids.append(i)
            hold_tgt.append(p.getJointState(rid, i,
                                            physicsClientId=cid)[0])

        lo, hi = joint_extent(p, rid, cid)
        hud = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
               float(hi[2] + 0.30)]
        cov = '  '.join(
            f'{s[0].upper()} {len(mp)}/{N_MOVABLE}'
            f'({coverage(name, s, mp) * 100:.0f}%)'
            for s, (mp, _a, _v, _n) in plans.items())
        scenes.append((name, rid, plans, hold_ids, hold_tgt, hud, cov))
        print(f'  {name:<8} 摆位 x={dx:+.1f}   手部 {cov}   '
              f'锁住 {len(hold_ids)} 个非手部关节')

    if not scenes:
        print('没有可用的机器人，退出')
        p.disconnect(cid)
        return


    # ---- 相机：框住三台 ----
    if args.render:
        lo_all = np.array([np.inf] * 3)
        hi_all = np.array([-np.inf] * 3)
        for _n, rid, *_rest in scenes:
            a, b = joint_extent(p, rid, cid)
            lo_all = np.minimum(lo_all, a)
            hi_all = np.maximum(hi_all, b)
        ctr = (lo_all + hi_all) / 2
        # 按「水平跨度」和「高度」两个约束取更远的那个。
        # 注意：pybullet 的窗口比正方形宽（常见 16:9），
        # 用正方形的 60 度 FOV 估距离会退得太远、机器人显小。
        aspect = 16.0 / 9.0
        half_v = float(np.radians(30.0))
        half_h = float(np.arctan(aspect * np.tan(half_v)))
        pad = 0.45
        width = float(hi_all[0] - lo_all[0])
        height = float(hi_all[2] - lo_all[2])
        dist = float(max((width / 2.0 + pad) / np.tan(half_h),
                         (height / 2.0 + pad) / np.tan(half_v),
                         2.5))
        yaw = 135.0 if args.view == 'full' else 0.0
        p.resetDebugVisualizerCamera(
            cameraDistance=round(dist, 2), cameraYaw=yaw, cameraPitch=-10,
            cameraTargetPosition=[float(ctr[0]), float(ctr[1]),
                                  float(ctr[2])],
            physicsClientId=cid)
        print(f'\n  相机：{"整机斜视" if yaw == 135 else "正面"}  '
              f'距离 {dist:.2f} m')
        print(f'  三台横向跨度 {width:.2f} m / 高 {height:.2f} m  '
              f'（按 16:9 窗口自动取景）')
        print('  鼠标：左键旋转 / 右键平移 / 滚轮缩放')

    T = len(data['timestamps'])
    t0 = data['timestamps'][0]
    frame_dt = (float(np.median(np.diff(data['timestamps'])))
                if T > 1 else 1.0 / 30.0)
    dt = 1.0 / 240.0
    substeps = (args.substeps if args.substeps > 0
                else max(1, int(round(frame_dt / dt))))
    p.setTimeStep(dt, physicsClientId=cid)
    print(f'\n[回放] {T} 帧 x {len(scenes)} 台机器人'
          f'（数据 {frame_dt*1000:.1f} ms/帧，每帧推进 {substeps} 步）')

    hud_ids = [-1] * len(scenes)
    rounds = 0
    try:
        while args.loop == 0 or rounds < args.loop:
            rounds += 1
            if args.render and args.loop != 1:
                more = ('（无限循环，关窗或 Ctrl+C 退出）' if args.loop == 0
                        else f'/{args.loop}')
                print(f'  第 {rounds} 遍{more}')
            for i in range(T):
                for si, (name, rid, plans, hold_ids, hold_tgt,
                         hud, cov) in enumerate(scenes):
                    for _side, (mp, arr, val, nms) in plans.items():
                        if not val[i]:
                            continue
                        joints, _c = map_frame(arr[i], mp)
                        for jn, v in joints.items():
                            p.setJointMotorControl2(
                                rid, nms[jn], p.POSITION_CONTROL,
                                targetPosition=v, force=200.0,
                                physicsClientId=cid)
                    for k, j in enumerate(hold_ids):
                        p.setJointMotorControl2(
                            rid, j, p.POSITION_CONTROL,
                            targetPosition=hold_tgt[k], force=500.0,
                            physicsClientId=cid)
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=cid)
                if args.render:
                    for si, (name, rid, plans, hold_ids, hold_tgt,
                             hud, cov) in enumerate(scenes):
                        hud_ids[si] = p.addUserDebugText(
                            f'{name.upper()}   帧 {i+1}/{T}\n{cov}',
                            hud, textSize=1.1, textColorRGB=COLORS[name],
                            lifeTime=0, replaceItemUniqueId=hud_ids[si],
                            physicsClientId=cid)
                    wait = ((data['timestamps'][i] - t0)
                            / max(args.speed, 1e-6))
                    time.sleep(max(0.0, min(wait, 0.05)))
                if (i + 1) % 200 == 0:
                    print(f'    ... {i+1}/{T}')
    except p.error as exc:
        print(f'\n  GUI 窗口已关闭（{exc}），提前结束。')
        return
    except KeyboardInterrupt:
        print('\n  已中断（Ctrl+C）。')
        return

    print('  回放完成。')
    if args.render:
        print('  窗口保持打开，可继续用鼠标查看。关窗或 Ctrl+C 退出。')
        try:
            while p.isConnected(cid):
                p.stepSimulation(physicsClientId=cid)
                time.sleep(1.0 / 60.0)
        except (p.error, KeyboardInterrupt):
            pass
    else:
        p.disconnect(cid)


from teleoperation.data.hand_h5 import read_native_angles as load_data
