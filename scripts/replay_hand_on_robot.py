"""把 LinkerHand l21 装到仿真机器人的腕部，用手部数据驱动手指

用途
----
让画面里出现「机器人 + l21 灵巧手」，比孤立的手更接近真实，
也为将来接入手臂数据做准备（现在手臂保持不动）。

技术做法
--------
1. 用 RobotLoader 加载机器人（与论文一致的 H1-2 / GR1-T2 / G1）
2. 找到「手基座 link」（每种机器人都有，见下）
3. 隐藏机器人自带的手（否则会和 l21 重叠）
4. 用 pybullet 的 **固定约束** 把 l21 手锁到「手基座 link」上
   → 以后手臂动了，手会自动跟随
5. 每帧用队友的 18 维数据驱动 l21 的 17 个关节

各种机器人的手基座 link（实测自 URDF，2026-09-12）
--------------------------------------------------
    h1_2   : L_hand_base_link  / R_hand_base_link   （父级是 *_wrist_yaw_link）
    gr1_t2 : l_hand_base_link  / r_hand_base_link   （父级是 *_end_effector_link）
    g1     : left_hand_palm_link / right_hand_palm_link（父级是 *_wrist_yaw_link）

用法
----
    python scripts/replay_hand_on_robot.py --robot h1_2 --hand right --render
    python scripts/replay_hand_on_robot.py --robot g1 --hand both --render
    # 手的位置/朝向不对时，用下面两个参数可视化微调：
    python scripts/replay_hand_on_robot.py --robot h1_2 --hand right --render \
        --mount-offset 0 0 0.05 --mount-rpy 0 0 1.5708
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

# 各机器人「手基座 link」的候选名（按优先级）
HAND_BASE_CANDIDATES = {
    'h1_2':   {'left': ['L_hand_base_link'],
               'right': ['R_hand_base_link']},
    'gr1_t2': {'left': ['l_hand_base_link'],
               'right': ['r_hand_base_link']},
    'g1':     {'left': ['left_hand_palm_link'],
               'right': ['right_hand_palm_link']},
}

# 判断「某个 link 属于机器人自带的手」（用于隐藏）
HAND_LINK_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky',
                  'hand_base', 'hand_palm', 'hand_')


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


def _link_name_of_joint(rid, i, cid):
    """取关节 i 的 child link 名

    注意：pybullet 的 getJointInfo **不返回 child link 的索引**；
    在树形结构中「关节 i 的 child link 索引就是 i 本身」。
    info[12] 是 child link 的【名字】（bytes）。
    """
    info = p_getJointInfo(rid, i, cid)
    nm = info[12]
    return nm.decode() if isinstance(nm, bytes) else nm


def _match_side(name, side):
    """判断 link 名是否属于指定侧（兼容 L_/R_ 与 left_/right_ 两种命名）"""
    low = (name or '').lower()
    if side == 'left':
        return low.startswith('l_') or low.startswith('left')
    return low.startswith('r_') or low.startswith('right')


def find_hand_base_link(rid, robot_type, side, cid):
    """在机器人里找「手基座 link」的索引（= 该 link 对应关节的索引）"""
    n = p_getNumJoints(rid, cid)
    name2link = {}
    for i in range(n):
        name2link[_link_name_of_joint(rid, i, cid)] = i
    # 优先用已知候选名（实测自 URDF）
    for cand in HAND_BASE_CANDIDATES.get(robot_type, {}).get(side, []):
        if cand in name2link:
            return name2link[cand], cand
    # 兜底：按名字找
    for ln, li in name2link.items():
        low = ln.lower()
        if _match_side(ln, side) and ('hand_base' in low
                                      or 'hand_palm' in low
                                      or 'end_effector' in low):
            return li, ln
    return None, None


def robot_hand_link_indices(rid, side, cid):
    """找出机器人自带「手」的所有 link 索引（用于隐藏）"""
    idxs = set()
    n = p_getNumJoints(rid, cid)
    for i in range(n):
        ln = _link_name_of_joint(rid, i, cid)
        low = (ln or '').lower()
        if not low:
            continue
        if _match_side(ln, side) and any(k in low for k in HAND_LINK_KEYS):
            idxs.add(i)
    return idxs


# --- 薄封装（延迟 import pybullet，便于模块被 import 时不连接）---
def p_getNumJoints(rid, cid):
    import pybullet as p
    return p.getNumJoints(rid, physicsClientId=cid)


def p_getJointInfo(rid, i, cid):
    import pybullet as p
    return p.getJointInfo(rid, i, physicsClientId=cid)


# ----------------------------------------------------------------------
def main():
    import pybullet as p
    import pybullet_data
    import importlib.util

    ap = argparse.ArgumentParser(
        description='把 LinkerHand l21 装到机器人腕部并回放手部数据')
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--robot', default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--hand', default='both',
                    choices=['left', 'right', 'both'])
    ap.add_argument('--render', action='store_true', help='GUI 可视化')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--mount-offset', nargs=3, type=float,
                    default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'),
                    help='在「手基座 link」坐标系里的安装平移（米）')
    ap.add_argument('--mount-rpy', nargs=3, type=float,
                    default=[0.0, 0.0, 0.0], metavar=('R', 'P', 'Y'),
                    help='安装旋转（弧度）')
    ap.add_argument('--no-hide-robot-hand', action='store_true',
                    help='不隐藏机器人自带的手（便于对比位置）')
    args = ap.parse_args()

    print('=' * 88)
    print('方案B：把 LinkerHand l21 装到机器人腕部')
    print('=' * 88)
    print(f'机器人 = {args.robot}   手 = {args.hand}   数据 = {args.file}')

    data = load_data(args.file)
    spec = importlib.util.spec_from_file_location(
        'rha', os.path.join(ROOT, 'scripts', 'replay_hand_angles.py'))
    rha = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rha)

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
    print(f'  机器人 bodyId = {robot_id}')

    if not args.no_hide_robot_hand:
        hidden = 0
        for side in ('left', 'right'):
            for li in robot_hand_link_indices(robot_id, side, cid):
                try:
                    p.changeVisualShape(robot_id, li,
                                        rgbaColor=[0, 0, 0, 0],
                                        physicsClientId=cid)
                    hidden += 1
                except p.error:
                    pass
        print(f'  已隐藏机器人自带的手：{hidden} 个 link')

    mounts = {}
    sides = ['left', 'right'] if args.hand == 'both' else [args.hand]
    for side in sides:
        model = f'l21_{side}'
        urdf = os.path.join(rha.HAND_ROOT, model,
                            f'linkerhand_{model}.urdf')
        if not os.path.isfile(urdf):
            print(f'  [FAIL] 缺模型 {urdf}')
            continue
        mlink, mname = find_hand_base_link(robot_id, args.robot, side, cid)
        if mlink is None:
            print(f'  [FAIL] 找不到 {side} 的手基座 link')
            continue
        print(f'  {side}: 手基座 link = {mname} (index={mlink})')

        p.stepSimulation(physicsClientId=cid)
        ls = p.getLinkState(robot_id, mlink, computeForwardKinematics=True,
                            physicsClientId=cid)
        base_pos, base_orn = ls[4], ls[5]
        print(f'    世界位姿 pos={tuple(round(float(v), 4) for v in base_pos)}')

        off_pos, off_orn = p.multiplyTransforms(
            base_pos, base_orn, args.mount_offset,
            p.getQuaternionFromEuler(args.mount_rpy), physicsClientId=cid)
        p.setAdditionalSearchPath(os.path.dirname(urdf),
                                  physicsClientId=cid)
        hand_id = p.loadURDF(urdf, off_pos, off_orn,
                             useFixedBase=False, physicsClientId=cid)
        cj = p.createConstraint(
            parentBodyUniqueId=robot_id, parentLinkIndex=mlink,
            childBodyUniqueId=hand_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=args.mount_offset,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=p.getQuaternionFromEuler(args.mount_rpy),
            childFrameOrientation=[0, 0, 0, 1], physicsClientId=cid)
        print(f'    已加载 l21 + 固定约束 (id={cj})')

        limits = rha.parse_joint_limits(urdf)
        name2idx = {}
        for i in range(p.getNumJoints(hand_id, physicsClientId=cid)):
            ji = p.getJointInfo(hand_id, i, physicsClientId=cid)
            nm = ji[1].decode() if isinstance(ji[1], bytes) else ji[1]
            name2idx[nm] = i
        dim2j = {d: name2idx[jn] for d, jn in enumerate(rha.MAPPING_18)
                 if jn and jn in name2idx}
        arr = data[f'{side}_angles']
        val = data[f'{side}_valid']
        bad = detect_bad_frames(arr[val])
        if bad:
            arr = arr.copy()
            arr[np.where(val)[0]] = repair_bad_frames(arr[val], bad)
            print(f'    已修复 {len(bad)} 个坏帧 {bad}')
        mounts[side] = (hand_id, dim2j, limits, arr, val)
        print(f'    可驱动关节 = {len(dim2j)}/17')

    if not mounts:
        print('没有可挂载的手，退出')
        p.disconnect(cid)
        return

    replay_loop(p, args, data, mounts, robot_id, cid, rha)


# ----------------------------------------------------------------------
def replay_loop(p, args, data, mounts, robot_id, cid, rha):
    """相机设置 + 回放主循环"""
    if args.render:
        kin = p.getBasePositionAndOrientation(robot_id, physicsClientId=cid)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.2, cameraYaw=45, cameraPitch=-18,
            cameraTargetPosition=[float(kin[0][0]), float(kin[0][1]), 0.9],
            physicsClientId=cid)
        print()
        print('  鼠标：左键旋转 / 右键平移 / 滚轮缩放')
        print('  手的位置或朝向不对时，用 --mount-offset / --mount-rpy 微调')

    T = len(data['timestamps'])
    t0 = data['timestamps'][0]
    print(f'\n[回放] {T} 帧，开始...')
    for i in range(T):
        for side, (hid, dim2j, limits, arr, val) in mounts.items():
            if not val[i]:
                continue
            for d, j in dim2j.items():
                jn = rha.MAPPING_18[d]
                lo, hi = limits.get(jn, (-9.0, 9.0))
                v = min(max(float(arr[i, d]), lo), hi)
                p.setJointMotorControl2(hid, j, p.POSITION_CONTROL,
                                        targetPosition=v, force=20.0,
                                        physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        if args.render:
            dt = (data['timestamps'][i] - t0) / max(args.speed, 1e-6)
            time.sleep(max(0.0, min(dt, 0.05)))
        if (i + 1) % 200 == 0:
            print(f'    ... {i+1}/{T}')

    print('  回放完成。')
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


