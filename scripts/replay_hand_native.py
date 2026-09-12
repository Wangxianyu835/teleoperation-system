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

from teleop.filters import (  # noqa: E402
    detect_bad_frames, detect_identity_swaps, print_identity_swap_report,
    repair_stream,
)

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


def setup_camera(p, view, robot_id, cid, hand_pose=None):
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

    if view == 'hands':
        tgt = list(hand_pose) if hand_pose else [ctr[0], ctr[1], top * 0.85]
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=135,
                                     cameraPitch=-15,
                                     cameraTargetPosition=tgt,
                                     physicsClientId=cid)
        return tgt, '手部特写'

    yaw = {'full': 135.0, 'front': 0.0, 'side': 90.0}[view]
    tgt = [ctr[0], ctr[1], ctr[2]]
    p.resetDebugVisualizerCamera(cameraDistance=round(auto, 2),
                                 cameraYaw=yaw, cameraPitch=-8,
                                 cameraTargetPosition=tgt,
                                 physicsClientId=cid)
    return tgt, (f'整机（自动距离 {auto:.2f} m，'
                 f'机器人高 {size:.2f} m，yaw={yaw:.0f}）')


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
    ap.add_argument('--view', default='full',
                    choices=['full', 'front', 'side', 'hands'],
                    help='相机视角：full=整机(默认,自动框住整个机器人) / '
                         'front=正面 / side=侧面 / hands=手部特写')
    ap.add_argument('--loop', type=int, default=1,
                    help='播放几遍（默认 1；设 0 表示无限循环，方便演示）')
    ap.add_argument('--limit-mode', default='clamp',
                    choices=['clamp', 'rescale'],
                    help='数据超出原装手限位时怎么办：'
                         'clamp=截断（默认，忠实映射）；'
                         'rescale=按比例缩放（保运动形状但改变语义，'
                         '用于判断"是不是被限位卡住了"）')
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
        plans[side] = (mapping, arr, val, names)

    if args.report:
        p.disconnect(cid)
        print()
        print('  （--report 模式：未启动回放）')
        return

    # ---- 相机：自动框住【整机】----
    if args.render:
        ctr, _size, top = compute_robot_extent(p, robot_id, cid)
        hand_target = [float(ctr[0]), float(ctr[1]), float(top * 0.85)]
        _tgtl, desc = setup_camera(p, args.view, robot_id, cid, hand_target)
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
            if args.render and args.loop != 1:
                more = ('（无限循环，Ctrl+C 退出）' if args.loop == 0
                        else f'/{args.loop}')
                print(f'  第 {rounds} 遍{more}')
            for i in range(T):
                for side, (mapping, arr, val, names) in plans.items():
                    if not val[i]:
                        continue
                    joints, nclip = map_frame(arr[i], mapping,
                                              limit_mode=args.limit_mode)
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
                        f'  |  数据帧 {i+1}/{T}  |  '
                        f'手部映射  左 {cov.get("left", "-")}  '
                        f'右 {cov.get("right", "-")}\n'
                        f'手指：队友重定向输出（L21 -> 原装手）      '
                        f'手臂：数据里没有，保持中性姿态静止',
                        hud_pos, textSize=1.0, textColorRGB=[1.0, 0.95, 0.3],
                        lifeTime=0, replaceItemUniqueId=hud_id,
                        physicsClientId=cid)
                if args.render:
                    wait = (data['timestamps'][i] - t0) / max(args.speed, 1e-6)
                    time.sleep(max(0.0, min(wait, 0.05)))
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


if __name__ == '__main__':
    main()


