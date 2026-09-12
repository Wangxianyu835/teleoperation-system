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

# 18 维数据的语义名（dim 0 是 hand_base_link 固定占位）
DIM_NAMES = {
    0: 'hand_base_link(占位)',
    1: 'index_mcp_roll', 2: 'index_mcp_pitch', 3: 'index_pip',
    4: 'middle_mcp_roll', 5: 'middle_mcp_pitch', 6: 'middle_pip',
    7: 'ring_mcp_roll', 8: 'ring_mcp_pitch', 9: 'ring_pip',
    10: 'pinky_mcp_roll', 11: 'pinky_mcp_pitch', 12: 'pinky_pip',
    13: 'thumb_cmc_roll', 14: 'thumb_cmc_yaw', 15: 'thumb_cmc_pitch',
    16: 'thumb_mcp', 17: 'thumb_ip',
}

# 关节名前缀（不同机器人命名风格不同）
PREFIX = {
    'h1_2':   {'left': 'L_', 'right': 'R_'},
    'gr1_t2': {'left': 'L_', 'right': 'R_'},
    'g1':     {'left': 'left_hand_', 'right': 'right_hand_'},
}

# 语义名 -> 原装手关节名模板（{P} 会被替换成上面的前缀）
# 未列出的语义名 = 原装手没有这个自由度，只能丢弃
NATIVE_MAP = {
    'h1_2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_mcp':        '{P}thumb_intermediate_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'gr1_t2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'g1': {
        # G1 只有 食指 / 中指 / 拇指 三根
        'index_mcp_pitch':  '{P}index_0_joint',
        'index_pip':        '{P}index_1_joint',
        'middle_mcp_pitch': '{P}middle_0_joint',
        'middle_pip':       '{P}middle_1_joint',
        'thumb_cmc_yaw':    '{P}thumb_0_joint',
        'thumb_cmc_pitch':  '{P}thumb_1_joint',
        'thumb_mcp':        '{P}thumb_2_joint',
    },
}


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


def build_mapping(robot_type, side, joint_ranges):
    """生成映射表：[(数据维度, 原装手关节名, 符号, 下限, 上限), ...]

    Args:
        robot_type: 'h1_2' / 'gr1_t2' / 'g1'
        side: 'left' / 'right'
        joint_ranges: {关节名: (lo, hi)} —— 从 pybullet 读到的真实限位

    符号规则：原装手的屈曲方向有两种约定
        H1-2   : [0.000, +1.700]  屈曲为正  -> sign = +1
        GR1-T2 : [-1.570, 0.000]  屈曲为负  -> sign = -1
        G1     : 左右手还不一样
    用「限位主要落在哪半轴」自动判定，避免手指反着弯。
    """
    pref = PREFIX[robot_type][side]

    def sign_of(lo, hi):
        return 1.0 if abs(hi) >= abs(lo) else -1.0

    table = NATIVE_MAP[robot_type]
    out = []
    for dim, sem in DIM_NAMES.items():
        tpl = table.get(sem)
        if tpl is None:
            continue                      # 原装手没有这个自由度 -> 丢弃
        jn = tpl.format(P=pref)
        if jn not in joint_ranges:
            continue
        lo, hi = joint_ranges[jn]
        out.append((dim, jn, sign_of(lo, hi), lo, hi))
    return out


def read_joint_ranges(robot_id, side, cid):
    """读取机器人的 {关节名: (下限, 上限)} 与 {关节名: 关节索引}

    ★ 必须用 info[1]（【关节名】，如 `L_index_proximal_joint`），
      不能用 info[12]（那是 child link 名，如 `L_index_proximal`）。
      NATIVE_MAP 里的模板用的是关节名。
    """
    import pybullet as p
    del side
    ranges, names = {}, {}
    n = p.getNumJoints(robot_id, physicsClientId=cid)
    for i in range(n):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED:
            continue
        jn = info[1].decode() if isinstance(info[1], bytes) else info[1]
        ranges[jn] = (float(info[8]), float(info[9]))
        names[jn] = i
    return ranges, names


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
        ranges, names = read_joint_ranges(robot_id, side, cid)
        mapping = build_mapping(args.robot, side, ranges)
        used = {d for d, *_ in mapping}
        dropped = [d for d in DIM_NAMES if d != 0 and d not in used]

        print()
        print(f'--- {side} ---')
        print(f'  数据 17 个可动维度 -> 原装手可表达 {len(mapping)} 个 '
              f'（覆盖率 {100.0 * len(mapping) / 17:.0f}%）')
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
    clip = {s: 0 for s in plans}
    tot = {s: 0 for s in plans}
    print(f'\n[回放] {T} 帧，开始...')
    for i in range(T):
        for side, (mapping, arr, val, names) in plans.items():
            if not val[i]:
                continue
            for d, jn, sg, lo, hi in mapping:
                v = sg * float(arr[i, d])
                if v < lo or v > hi:
                    clip[side] += 1
                tot[side] += 1
                v = min(max(v, lo), hi)
                p.setJointMotorControl2(robot_id, names[jn],
                                        p.POSITION_CONTROL,
                                        targetPosition=v, force=200.0,
                                        physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        if args.render:
            dt = (data['timestamps'][i] - t0) / max(args.speed, 1e-6)
            time.sleep(max(0.0, min(dt, 0.05)))
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


