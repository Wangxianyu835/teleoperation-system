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
    python -m teleoperation replay native-hand --robot h1_2 --hand both --render
    python -m teleoperation replay native-hand --robot h1_2 --hand both --report
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

from teleoperation.retargeting.hand.postprocessing import detect_bad_frames, detect_identity_swaps, repair_stream
from teleoperation.tools.hand_reports import print_identity_swap_report

# 映射表与工具统一放在 robots/native_hand.py（正式接口，可被其它代码复用）
from teleoperation.robots.native_hand import DIM_NAMES, N_MOVABLE, build_mapping, coverage, dropped_dims
from teleoperation.apps.replay.hand_support import map_frame
from teleoperation.simulation.robot_loader import read_joint_ranges


# ----------------------------------------------------------------------
def compute_robot_extent(p, robot_id, cid):
    """把机器人【所有 link】的 AABB 合起来 -> (中心, 最大边长, 最高点)

    用来自动算相机距离，保证【整机】都在画面里（不再靠手调数字）。
    """
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for i in range(-1, p.getNumJoints(robot_id, physicsClientId=cid)):
        try:
            a, b = p.getAABB(robot_id, i, physicsClientId=cid)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(a, dtype=float))
        hi = np.maximum(hi, np.asarray(b, dtype=float))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        return np.zeros(3), 1.0, 1.0
    return (lo + hi) / 2.0, float(np.max(hi - lo)), float(hi[2])


def hand_extent(p, robot_id, cid, link_ids):
    """Frame the actual finger links rather than an estimate of body height."""
    boxes = [p.getAABB(robot_id, index, physicsClientId=cid) for index in link_ids]
    if not boxes:
        raise ValueError('No mapped hand links are available for the close-up')
    lo = np.min([box[0] for box in boxes], axis=0)
    hi = np.max([box[1] for box in boxes], axis=0)
    return (lo + hi) / 2, float(np.max(hi - lo))


def style_hands(p, robot_id, cid, plans):
    """Display-only colors; retain the recorded targets and native mapping."""
    for index in range(-1, p.getNumJoints(robot_id, physicsClientId=cid)):
        p.changeVisualShape(robot_id, index, rgbaColor=[0.72, 0.76, 0.82, 1.0],
                            textureUniqueId=-1, physicsClientId=cid)
    for side, (mapping, _arr, _valid, names) in plans.items():
        color = (0.28, 0.70, 1.0, 1.0) if side == 'left' else (1.0, 0.62, 0.22, 1.0)
        for _dim, name, *_ in mapping:
            p.changeVisualShape(robot_id, names[name], rgbaColor=color,
                                textureUniqueId=-1, specularColor=[0.15] * 3,
                                physicsClientId=cid)


def setup_camera(p, view, robot_id, cid, hand_pose=None, hand_links=None):
    """按 --view 设置相机；full/front/side 都自动框住整机

    Args:
        view: 'full' | 'front' | 'side' | 'hands'
        hand_pose: 'hands' 视图用的目标点（手的世界坐标）

    Returns:
        (target, 说明字符串)
    """
    ctr, size, top = compute_robot_extent(p, robot_id, cid)
    fov_v = 60.0
    # 1.6 倍留白：1.35 时机器人的脚/头会贴到画面边缘，看起来"卡边"
    auto = max(1.2, size * 1.6 / (2.0 * np.tan(np.radians(fov_v / 2.0))))

    if view in ('hands', 'left-hand', 'right-hand'):
        if hand_links:
            center, hand_size = hand_extent(p, robot_id, cid, hand_links)
            tgt = center.tolist()
            distance = max(0.26, hand_size * 1.7)
        else:
            tgt = list(hand_pose) if hand_pose is not None else [ctr[0], ctr[1], top * 0.85]
            distance = 1.1
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=135,
                                     cameraPitch=-12,
                                     cameraTargetPosition=tgt,
                                     physicsClientId=cid)
        return tgt, f'{view}（实际手部中心，距离 {distance:.2f} m）'

    yaw = {'full': 135.0, 'front': 0.0, 'side': 90.0}[view]
    tgt = [ctr[0], ctr[1], ctr[2]]
    p.resetDebugVisualizerCamera(cameraDistance=round(auto, 2),
                                 cameraYaw=yaw, cameraPitch=-8,
                                 cameraTargetPosition=tgt,
                                 physicsClientId=cid)
    return tgt, (f'整机（自动距离 {auto:.2f} m，'
                 f'机器人高 {size:.2f} m，yaw={yaw:.0f}）')


# ----------------------------------------------------------------------
def main(args=None):
    import pybullet as p
    import pybullet_data

    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")
    if not np.isfinite(args.speed) or args.speed <= 0:
        args._parser.error('--speed must be finite and positive')
    if args.loop < 0 or args.substeps < 0:
        args._parser.error('--loop and --substeps must be nonnegative')
    if args.view in ('left-hand', 'right-hand') and args.hand not in ('both', args.view[:-5]):
        args._parser.error('the selected close-up side must also be selected by --hand')

    print('=' * 96)
    print('原装手回放：18 维数据 -> 机器人自带灵巧手（降维映射）')
    print('=' * 96)
    print(f'机器人 = {args.robot}   手 = {args.hand}   数据 = {args.file}')

    data = load_data(args.file)
    frame_count = len(data['timestamps'])
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end_frame <= frame_count:
        args._parser.error(f'frame range must satisfy 0 <= start < end <= {frame_count}')
    recording_start = args.start_frame
    timestamps = data['timestamps'][recording_start:end_frame]
    print(f'  原始帧区间 [{recording_start}, {end_frame})；速度 {args.speed:g}x')
    from teleoperation.simulation import RobotLoader

    cid = p.connect(p.GUI if args.render else p.DIRECT)
    if args.render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(1.0 / 240.0, physicsClientId=cid)
    from teleoperation.simulation.urdf_loader import create_ground
    create_ground(p, physicsClientId=cid)

    loader = RobotLoader(cid)
    robot_id = loader.load_robot(args.robot)['robot']

    sides = ['left', 'right'] if args.hand == 'both' else [args.hand]

    # ---- 先做「左右手身份切换」检测（必须两侧一起看）----
    # 场景：一手从画面消失时，MediaPipe 会把另一只手误标到它头上，
    #       表现为「本侧突变 + 另一侧同时消失」。实测 right[535] 就是这种。
    swaps = detect_identity_swaps(
        data['left_angles'], data['right_angles'],
        data['left_valid'], data['right_valid'])
    print_identity_swap_report(swaps, '左右手身份切换检测（MediaPipe 标签污染）')
    swap_by_side = {}
    for s in swaps:
        swap_by_side.setdefault(s['side'], []).append(s['frame'])

    plans = {}
    adapters = {}
    for side in sides:
        ranges, names = read_joint_ranges(robot_id, cid)
        adapters[side] = NativeHandAdapter(args.robot, side, ranges, args.limit_mode)
        mapping = adapters[side].mapping
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
        fronts = sorted(set(swap_by_side.get(side, [])))
        if args.no_repair:
            if fronts or bad:
                print(f'  检出 身份切换 {fronts} / 塌零 {bad}'
                      f'（--no-repair，未修复）')
        else:
            arr, held, interp = repair_stream(arr, val, bad, fronts)
            if held:
                print(f'  已【保持上一有效姿态】修复身份切换帧: {held}')
            if interp:
                print(f'  已【线性插值】修复塌零帧: {interp}')
            if not held and not interp:
                print('  未检出需要修复的帧  [OK]')
        # Keep detection/repair on the full stream so selecting a clip does not
        # change the targets for the same original recording frame.
        plans[side] = (mapping, arr[recording_start:end_frame],
                       val[recording_start:end_frame], names)

    if args.report:
        p.disconnect(cid)
        print()
        print('  （--report 模式：未启动回放）')
        return

    # ---- 相机：自动框住【整机】----
    if args.render:
        style_hands(p, robot_id, cid, plans)
        ctr, _size, top = compute_robot_extent(p, robot_id, cid)
        hand_target = [float(ctr[0]), float(ctr[1]), float(top * 0.85)]
        focus = list(plans) if args.view == 'hands' else [args.view[:-5]] if args.view.endswith('-hand') else []
        hand_links = [plans[side][3][name] for side in focus
                      for _dim, name, *_ in plans[side][0]]
        _tgtl, desc = setup_camera(p, args.view, robot_id, cid, hand_target, hand_links)
        print()
        print(f'  相机视角：{desc}')
        print('  鼠标：左键旋转 / 右键平移 / 滚轮缩放')

    # ---- 锁住【没有数据驱动】的关节（手臂 / 腿 / 腰）----
    # 机器人是 useFixedBase=True，但手臂关节若没有电机，
    # 会在重力下慢慢垂下来 —— 看起来像"散架"，不像一台完整的机器人在站着。
    # 这里用位置控制把它们保持在当前（ARM_NEUTRAL 中性）姿态。
    hand_ids = set()
    for _mapping, _arr, _val, names in plans.values():
        for _d, jn, *_ in _mapping:
            if jn in names:
                hand_ids.add(names[jn])
    hold_ids, hold_tgt = [], []
    for i in range(p.getNumJoints(robot_id, physicsClientId=cid)):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED or i in hand_ids:
            continue
        hold_ids.append(i)
        hold_tgt.append(p.getJointState(robot_id, i,
                                        physicsClientId=cid)[0])
    print(f'  已锁住 {len(hold_ids)} 个非手部关节'
          f'（手臂/腿/腰，数据里没有它们）')

    T = len(timestamps)
    t0 = timestamps[0]
    frame_dt = (float(np.median(np.diff(timestamps)))
                if T > 1 else 1.0 / 30.0)
    dt = 1.0 / 240.0
    substeps = (args.substeps if args.substeps > 0
                else max(1, int(round(frame_dt / dt))))
    p.setTimeStep(dt, physicsClientId=cid)
    clip = {s: 0 for s in plans}
    tot = {s: 0 for s in plans}
    cov = {s: (f'{len(plans[s][0])}/{N_MOVABLE}'
               f'({coverage(args.robot, s, plans[s][0]) * 100:.0f}%)')
           for s in plans}

    hud_id, hud_pos = -1, None
    if args.render:
        c2, _s2, top2 = compute_robot_extent(p, robot_id, cid)
        hud_pos = [float(c2[0]), float(c2[1]), float(top2 + 0.35)]

    print(f'\n[回放] {T} 帧，开始...（数据 {frame_dt*1000:.1f} ms/帧，'
          f'物理步 {dt*1000:.2f} ms，每帧推进 {substeps} 步）')

    rounds = 0
    try:
        while args.loop == 0 or rounds < args.loop:
            rounds += 1
            loop_started = time.monotonic()
            if args.render and args.loop != 1:
                more = ('（无限循环，Ctrl+C 退出）' if args.loop == 0
                        else f'/{args.loop}')
                print(f'  第 {rounds} 遍{more}')
            for i in range(T):
                for side, (mapping, arr, val, names) in plans.items():
                    if not val[i]:
                        continue
                    dofs = L21HandAngles(arr[i]).native_mapping_dofs()
                    joints, nclip = adapters[side].map(dofs)
                    clip[side] += nclip
                    tot[side] += len(mapping)
                    for jn, v in joints.items():
                        p.setJointMotorControl2(robot_id, names[jn],
                                                p.POSITION_CONTROL,
                                                targetPosition=v, force=200.0,
                                                physicsClientId=cid)
                for k, j in enumerate(hold_ids):
                    p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                            targetPosition=hold_tgt[k],
                                            force=500.0,
                                            physicsClientId=cid)
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=cid)
                if hud_pos is not None:
                    # 把「第几遍」显示在画面上 —— 只在控制台打印容易让人
                    # 误以为"数据是无限的、机器人一直在动"
                    loop_txt = ('无限循环' if args.loop == 0
                                else f'共 {args.loop} 遍')
                    hud_id = p.addUserDebugText(
                        f'{args.robot}  |  第 {rounds} 遍（{loop_txt}）'
                        f'  |  原始帧 {recording_start+i}（片段 {i+1}/{T}）  |  '
                        f'手部映射  左 {cov.get("left", "-")}  '
                        f'右 {cov.get("right", "-")}\n'
                        f'手指：队友重定向输出（L21 -> 原装手）      '
                        f'手臂：数据里没有，保持中性姿态静止',
                        hud_pos, textSize=1.0, textColorRGB=[1.0, 0.95, 0.3],
                        lifeTime=0, replaceItemUniqueId=hud_id,
                        physicsClientId=cid)
                if args.render:
                    # Schedule against wall time; the old absolute per-frame
                    # sleep was capped at 50 ms and made --speed ineffective.
                    wait = (timestamps[i] - t0) / args.speed - (time.monotonic() - loop_started)
                    if wait > 0:
                        time.sleep(wait)
                if (i + 1) % 200 == 0:
                    print(f'    ... {i+1}/{T}')
    except p.error as exc:
        # 手动关掉 GUI 窗口时 pybullet 会抛 "Not connected to physics server"
        print(f'\n  GUI 窗口已关闭（{exc}），提前结束。')
        return
    except KeyboardInterrupt:
        print('\n  已中断（Ctrl+C）。')
        return

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


from teleoperation.data.hand_h5 import read_native_angles as load_data

from teleoperation.contracts.hand import L21HandAngles
from teleoperation.robots.native_hand import NativeHandAdapter
