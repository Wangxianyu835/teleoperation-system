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
    python -m teleoperation replay mounted-hand --robot h1_2 --hand right --render
    python -m teleoperation replay mounted-hand --robot g1 --hand both --render
    # 手的位置/朝向不对时，用下面两个参数可视化微调：
    python -m teleoperation replay mounted-hand --robot h1_2 --hand right --render \
        --mount-offset 0 0 0.05 --mount-rpy 0 0 1.5708
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


# ---------------------- 「手坐标系」推断与自动安装 ----------------------
# 目标手的坐标系用三根正交轴描述（都在「手基座 link」坐标系里）：
#   e1 = 手指伸展方向（从掌根指向指尖）
#   e2 = 拇指侧方向（垂直于 e1，指向拇指）
#   e3 = e1 × e2
FINGER_KEYS = ('index', 'middle', 'ring', 'pinky')
THUMB_KEY = 'thumb'


# --- 薄封装（延迟 import pybullet，便于模块被 import 时不连接）---


# ----------------------------------------------------------------------
def main(args=None):
    import pybullet as p
    import pybullet_data
    import importlib.util

    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    print('=' * 88)
    print('方案B：把 LinkerHand l21 装到机器人腕部')
    print('=' * 88)
    print(f'机器人 = {args.robot}   手 = {args.hand}   数据 = {args.file}')

    data = load_data(args.file)
    from teleoperation.apps.replay import replay_hand_angles as rha

    from teleoperation.simulation import RobotLoader

    cid = p.connect(p.GUI if args.render else p.DIRECT)
    if args.render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(args.dt, physicsClientId=cid)
    from teleoperation.simulation.urdf_loader import create_ground, load_urdf
    create_ground(p, physicsClientId=cid)

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

    # ---- 先做「左右手身份切换」检测（必须两侧一起看）----
    swaps = detect_identity_swaps(
        data['left_angles'], data['right_angles'],
        data['left_valid'], data['right_valid'])
    print_identity_swap_report(swaps, '左右手身份切换检测（MediaPipe 标签污染）')
    swap_by_side = {}
    for s in swaps:
        swap_by_side.setdefault(s['side'], []).append(s['frame'])

    for side in sides:
        model = f'l21_{side}'
        urdf = os.path.join(rha.HAND_ROOT, side,
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

        p.setAdditionalSearchPath(os.path.dirname(urdf),
                                  physicsClientId=cid)
        # 先按「零偏移」加载，用来测量 l21 自身的坐标系
        hand_id = load_urdf(urdf, basePosition=base_pos,
                            baseOrientation=base_orn,
                            useFixedBase=False, physicsClientId=cid)
        for i in range(p.getNumJoints(hand_id, physicsClientId=cid)):
            p.resetJointState(hand_id, i, 0.0, physicsClientId=cid)

        # ★ 修「手抖」（2026-09-12 定位）
        #   l21 的 URDF 质量/惯量接近 0：base link 质量 1.5785e-07 kg、
        #   惯量对角 (0, 0, 0)，各指节 0.0004~0.003 kg，且 damping=friction=0。
        #   位置控制相对这么小的惯量增益过大 -> 过冲 / 数值发散，
        #   肉眼看到的就是「手抖」。重标到物理合理量级即可：
        #       实测 稳态误差 0.4380 -> 0.0114，超调 0.4800 -> 0.0005
        #   （注意：这是为了让控制稳定，不代表真实质量；做接触/抓取实验时需谨慎）
        if not args.no_fix_jitter:
            for i in range(p.getNumJoints(hand_id, physicsClientId=cid)):
                p.changeDynamics(hand_id, i, mass=0.02,
                                 localInertiaDiagonal=[1e-6, 1e-6, 1e-6],
                                 physicsClientId=cid)
            print('    已重标质量/惯量（修手抖）: mass=0.02 kg, '
                  'inertia=1e-6（--no-fix-jitter 可关）')

        mount_rpy = list(args.mount_rpy)
        rob_frame = None
        if not args.no_auto_mount:
            rob_frame = measure_hand_frame(
                robot_id, base_pos, base_orn, cid,
                link_indices=robot_hand_link_indices(robot_id, side, cid))
            # ★ 必须用手的「实际」base 位姿作参考系：
            #   pybullet 的 loadURDF 对自由刚体按「质心」摆放 basePosition，
            #   实际 base 与传入值有偏移（l21 约 0.076 m）。若用传入值作参考，
            #   手指/拇指方向会被算歪（实测约 15 度）。
            hb_pos, hb_orn = p.getBasePositionAndOrientation(
                hand_id, physicsClientId=cid)
            l21_frame = measure_hand_frame(hand_id, hb_pos, hb_orn, cid)
            off = float(np.linalg.norm(np.asarray(hb_pos)
                                       - np.asarray(base_pos)))
            if off > 1e-4:
                print(f'    注：loadURDF 的 base 实际偏移了 {off:.4f} m'
                      f'（质心 vs 原点约定），已按实际位姿测量')
            auto_rpy, det = compute_auto_mount_rpy(rob_frame, l21_frame)
            if auto_rpy is None:
                print('    自动安装失败（几何不足），改用 --mount-rpy')
            else:
                print('    自动安装朝向 rpy = ({:+.4f}, {:+.4f}, {:+.4f})'
                      '  [det={:+.4f}]'.format(*auto_rpy, det))
                if abs(det - 1.0) > 1e-4:
                    print('    [WARN] det != 1：左右手存在镜像，朝向可能不准')
                if list(args.mount_rpy) == [0.0, 0.0, 0.0]:
                    mount_rpy = auto_rpy

        off_pos, off_orn = p.multiplyTransforms(
            base_pos, base_orn, args.mount_offset,
            p.getQuaternionFromEuler(mount_rpy), physicsClientId=cid)
        p.resetBasePositionAndOrientation(hand_id, off_pos, off_orn,
                                          physicsClientId=cid)
        # 自检：安装后再测一次 l21 坐标系，应与机器人自带手重合
        if rob_frame is not None:
            got = measure_hand_frame(hand_id, base_pos, base_orn, cid)
            if got is not None:
                a_f = _vec_angle(got[:, 0], rob_frame[:, 0])
                a_t = _vec_angle(got[:, 1], rob_frame[:, 1])
                ok = a_f < 0.5 and a_t < 0.5
                print('    自检：手指方向差 {:.2f}deg / 拇指方向差 {:.2f}deg'
                      '  {}'.format(a_f, a_t,
                                    '[OK]' if ok else '[WARN 朝向可能有偏差]'))
        cj = p.createConstraint(
            parentBodyUniqueId=robot_id, parentLinkIndex=mlink,
            childBodyUniqueId=hand_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=args.mount_offset,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=p.getQuaternionFromEuler(mount_rpy),
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
        bad = detect_bad_frames(arr, val)
        fronts = sorted(set(swap_by_side.get(side, [])))
        arr, held, interp = repair_stream(arr, val, bad, fronts)
        if held:
            print(f'    已【保持上一有效姿态】修复身份切换帧: {held}')
        if interp:
            print(f'    已【线性插值】修复塌零帧: {interp}')
        if not held and not interp:
            print('    未检出需要修复的帧  [OK]')
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
    frame_dt = (float(np.median(np.diff(data['timestamps'])))
                if T > 1 else 1.0 / 30.0)
    substeps = (args.substeps if args.substeps > 0
                else max(1, int(round(frame_dt / args.dt))))
    print(f'\n[回放] {T} 帧，开始...（数据 {frame_dt*1000:.1f} ms/帧，'
          f'物理步 {args.dt*1000:.2f} ms，每帧推进 {substeps} 步）')
    for i in range(T):
        for side, (hid, dim2j, limits, arr, val) in mounts.items():
            if not val[i]:
                continue
            adapter = L21HandAdapter(rha.MAPPING_18, limits, default_limits=(-9.0, 9.0))
            targets = adapter.map(L21HandAngles(arr[i]).native_mapping_dofs())
            for d, j in dim2j.items():
                v = targets[rha.MAPPING_18[d]]
                p.setJointMotorControl2(hid, j, p.POSITION_CONTROL,
                                        targetPosition=v, force=20.0,
                                        physicsClientId=cid)
        for _ in range(substeps):
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


from teleoperation.data.hand_h5 import read_native_angles as load_data

from teleoperation.simulation.hand_mount import _collect_link_positions, _euler_from_matrix_xyz, _finger_tips, _link_name_of_joint, _mat_to_euler_xyz, _match_side, _thumb_tip, _vec_angle, compute_auto_mount_rpy, compute_hand_frame, find_hand_base_link, measure_hand_frame, p_getJointInfo, p_getNumJoints, robot_hand_link_indices

from teleoperation.robots.l21 import L21HandAdapter
from teleoperation.contracts.hand import L21HandAngles
