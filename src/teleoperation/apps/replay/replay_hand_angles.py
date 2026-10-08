"""离线回放：队友重定向输出的手部角度 -> LinkerHand l21 模型

★ 数据来源：队友的重定向算法输出（当前是「契约外」的中间格式）
  datasets/raw/retarget_twohand_153542.h5
     left_angles / right_angles  (T, 18)  float32
     left_valid  / right_valid   (T,)     bool     <- 无效帧（同时是全零帧）
     timestamps                  (T,)     float64

★ 18 维的含义（通过「维度实测范围 vs 关节限位」匹配推导，零越界）：
     dim 0                -> 占位（恒 0）
     dims 1,2,3           -> index:  mcp_roll, mcp_pitch, pip
     dims 4,5,6           -> middle: mcp_roll, mcp_pitch, pip
     dims 7,8,9           -> ring:   mcp_roll, mcp_pitch, pip
     dims 10,11,12        -> pinky:  mcp_roll, mcp_pitch, pip
     dims 13,14,15,16,17  -> thumb:  cmc_roll, cmc_yaw, cmc_pitch, mcp, ip

     推导依据（关键证据）：
       dim 13 [-0.5997,-0.5875] <-> thumb_cmc_roll  (±0.60)   精确到 0.0003
       dim 14 [+0.0067,+1.5953] <-> thumb_cmc_yaw   (0~1.60)  精确
       dim  1 [+0.1771,+0.1800] <-> index_mcp_roll  (±0.18)
     即「每 3 个一组 = (侧摆, MCP屈曲, PIP屈曲)」，与 l21 的 URDF 顺序一致。

用法：
    python -m teleoperation replay l21-hand --check                  # 只做静态检查
    python -m teleoperation replay l21-hand --hand right --render    # GUI 回放（右手）
    python -m teleoperation replay l21-hand --hand both --render --smooth 5
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

# LinkerHand 模型根目录
from teleoperation.paths import L21_ROOT
HAND_ROOT = str(L21_ROOT)

# ======================================================================
# ★★★ 映射表（dim 索引 -> LinkerHand l21 关节名）★★★
#  推导方式：把「每个维度的实测范围」与「l21 关节限位」逐一匹配，结果零越界。
#  若队友给出了官方顺序，请以此处为准修改（其它代码无需改动）。
# ======================================================================

# l21 的 URDF 关节顺序（用于交叉校验映射是否与 URDF 一致）


# 平滑 / 诊断相关实现统一放在 retargeting/hand/postprocessing.py（单一实现，避免重复）
from teleoperation.retargeting.hand.postprocessing import KalmanSmoother, moving_average, classify_jumps, compare_methods, detect_bad_frames, repair_bad_frames
from teleoperation.tools.hand_reports import print_jump_report, print_metrics_table, print_bad_frames_report


# ----------------------------------------------------------------------
def static_check(data, angles, valid, side, limits, tol=0.02):
    """静态检查：维度 / 范围 / 越界 / 跳变"""
    print()
    print('=' * 94)
    print(f'【静态检查】{side}   维度 = {angles.shape[1]}   '
          f'总帧 = {len(angles)}   有效帧 = {int(valid.sum())}')
    print('=' * 94)
    A = angles[valid]
    if len(A) == 0:
        print('  没有有效帧！')
        return False

    idx = np.where(valid)[0]
    ts = data['timestamps']
    print(f'  有效帧范围: {idx[0]} ~ {idx[-1]}  (共 {len(idx)} 帧, '
          f'{ts[idx[-1]] - ts[idx[0]]:.2f} s)')

    print()
    print(f'  {"dim":>4s} {"实测范围":>22s} {"映射关节":<22s} {"限位":>20s}  状态')
    print('  ' + '-' * 90)
    n_ok, n_bad, n_none = 0, 0, 0
    for d in range(A.shape[1]):
        lo, hi = float(A[:, d].min()), float(A[:, d].max())
        jname = MAPPING_18[d] if d < len(MAPPING_18) else None
        if jname is None:
            print(f'  {d:>4d}   [{lo:+.4f}, {hi:+.4f}]  {"(占位/未映射)":<22s} '
                  f'{"-":>20s}  --')
            n_none += 1
            continue
        if jname not in limits:
            print(f'  {d:>4d}   [{lo:+.4f}, {hi:+.4f}]  {jname:<22s} '
                  f'{"不存在于URDF":>20s}  [FAIL]')
            n_bad += 1
            continue
        llo, lhi = limits[jname]
        lim_s = f'{llo:+.3f}~{lhi:+.3f}'
        if lo >= llo - tol and hi <= lhi + tol:
            print(f'  {d:>4d}   [{lo:+.4f}, {hi:+.4f}]  {jname:<22s} '
                  f'{lim_s:>20s}  [OK]')
            n_ok += 1
        else:
            print(f'  {d:>4d}   [{lo:+.4f}, {hi:+.4f}]  {jname:<22s} '
                  f'{lim_s:>20s}  [越界]')
            n_bad += 1

    print()
    print(f'  映射检查: OK={n_ok}  越界={n_bad}  未映射={n_none}')
    dmax = np.abs(np.diff(A, axis=0)).max(axis=1)
    big = int((dmax > 0.3).sum())
    print(f'  帧间跳变(>0.3rad): {big}/{len(dmax)}   最大 {dmax.max():.4f} rad '
          f'({np.degrees(dmax.max()):.1f}°)')
    return n_bad == 0


# ----------------------------------------------------------------------
def replay_pybullet(angles, valid, timestamps, urdf_path, urdf_dir,
                    joints_order, limits, base_pos, speed=1.0, gui=True):
    """在 pybullet 里回放（gui=False 则用 DIRECT 无头模式，便于自动验证）"""
    import pybullet as p

    cid = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(urdf_dir, physicsClientId=cid)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    rid = p.loadURDF(urdf_path, base_pos, useFixedBase=True,
                     physicsClientId=cid)

    if gui:
        # 把手放到视野中央（否则 20cm 的手在默认视角下几乎看不见）
        p.resetDebugVisualizerCamera(
            cameraDistance=0.55, cameraYaw=40, cameraPitch=-20,
            cameraTargetPosition=list(base_pos), physicsClientId=cid)

    name2idx = {}
    for i in range(p.getNumJoints(rid, physicsClientId=cid)):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        nm = info[1].decode() if isinstance(info[1], bytes) else info[1]
        name2idx[nm] = i
    print(f'  已加载 {os.path.basename(urdf_path)}：{len(name2idx)} 个关节')

    dim2jidx = {d: name2idx[jn] for d, jn in enumerate(joints_order)
                if jn and jn in name2idx}
    print(f'  可驱动的动作维度: {len(dim2jidx)}')

    adapter = L21HandAdapter(joints_order, limits)
    t0 = timestamps[0]
    n_frames = 0
    final_q = {}
    for i in range(len(angles)):
        if not valid[i]:
            continue
        targets = adapter.map(L21HandAngles(angles[i]).native_mapping_dofs())
        for d, jidx in dim2jidx.items():
            val = targets[joints_order[d]]
            p.setJointMotorControl2(rid, jidx, p.POSITION_CONTROL,
                                    targetPosition=val, force=5.0,
                                    physicsClientId=cid)
        for _ in range(5):                        # 多步进一点，让姿态收敛
            p.stepSimulation(physicsClientId=cid)
        n_frames += 1
        if gui:
            dt = (timestamps[i] - t0) / max(speed, 1e-6)
            time.sleep(max(0.0, min(dt, 0.05)))
        # 记录末帧关节角（用于无头模式验证）
        final_q = {jn: p.getJointState(rid, name2idx[jn],
                                       physicsClientId=cid)[0]
                   for jn in joints_order if jn and jn in name2idx}
    print(f'  回放完成：{n_frames} 帧' + ('' if gui else '（无头模式）'))

    if not gui:
        # 无头模式：打印末帧关节角，确认确实驱动了
        print('  末帧关节角（前 6 个）:')
        for jn, v in list(final_q.items())[:6]:
            print(f'    {jn:<22s} {v:+.4f} rad')

    if gui:
        print('  回放结束。窗口保持打开，可继续用鼠标查看末帧姿态。')
        print('  退出方式：关闭 pybullet 窗口，或在终端按 Ctrl+C')
        try:
            while p.isConnected(cid):
                p.stepSimulation(physicsClientId=cid)
                time.sleep(1.0 / 60.0)
        except (p.error, KeyboardInterrupt):
            pass
        print('  已退出。')
    else:
        p.disconnect(cid)
    return final_q


# ----------------------------------------------------------------------
def main(args=None):
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    print('=' * 94)
    print('回放重定向输出的手部角度  ->  LinkerHand l21')
    print('=' * 94)
    print('数据文件 =', args.file)

    data = load_data(args.file)
    ok_all = True
    sides = [args.hand] if args.hand != 'both' else ['left', 'right']

    for side in sides:
        model = f'l21_{side}'
        urdf = os.path.join(HAND_ROOT, side, f'linkerhand_{model}.urdf')
        if not os.path.isfile(urdf):
            print(f'\n[FAIL] 找不到模型：{urdf}')
            ok_all = False
            continue
        limits = parse_joint_limits(urdf)
        raw = data[f'{side}_angles']
        valid = data[f'{side}_valid']

        # ---- 坏帧检测（整帧塌零：多维同时掉到 0 而前后帧正常） ----
        # ★ 必须传【完整数组 + valid】：detect_bad_frames 返回的是【绝对下标】。
        #   旧版传的是 raw[valid]（压缩数组），返回压缩下标，会被误当成绝对下标
        #   （2026-09-12 就把 left[522] 误报成了 left[497]）。
        bad = detect_bad_frames(raw, valid)
        print_bad_frames_report(bad, len(raw), f'{side} 坏帧检测（整帧塌零）')
        if bad and not args.no_repair:
            raw = repair_bad_frames(raw, bad)
            print(f'  [修复] 已用前后帧线性插值修复 {len(bad)} 个坏帧'
                  f'（如不想修复请加 --no-repair）')

        # ---- 跳变性质诊断（决定"该不该滤波"） ----
        if args.analyze or args.compare:
            print_jump_report(classify_jumps(raw[valid]),
                              f'{side} 数据跳变诊断'
                              f'（孤立尖峰 vs 成片真实运动）')

        # ---- 对比多种平滑方法 ----
        if args.compare:
            dt = 1.0 / max(args.fps, 1e-6)
            methods = {
                'ma3': moving_average(raw[valid], 3),
                'ma7': moving_average(raw[valid], 7),
                'kf': KalmanSmoother(dt=dt, process_noise=args.kalman_q,
                                     measurement_noise=args.kalman_r
                                     ).filter_sequence(raw[valid]),
            }
            print_metrics_table(
                compare_methods(raw[valid], methods, fps=args.fps),
                f'{side} 平滑效果对比（原始 vs 移动平均 vs 卡尔曼）')

        # ---- 选择平滑方式 ----
        method = 'raw（未平滑）'
        angles = raw
        if args.kalman:
            kf = KalmanSmoother(dt=1.0 / max(args.fps, 1e-6),
                                process_noise=args.kalman_q,
                                measurement_noise=args.kalman_r,
                                gate=args.gate)
            angles = kf.filter_sequence(raw)
            method = (f'kalman(q={args.kalman_q:g}, r={args.kalman_r:g}, '
                      f'gate={args.gate:g})')
        elif args.smooth > 1:
            angles = moving_average(raw, args.smooth)
            method = f'moving_average(win={args.smooth})'
        if method != 'raw（未平滑）':
            print(f'\n  [平滑] {side}: {method}')

        ok_all = static_check(data, angles, valid, side, limits) and ok_all

        if (args.render or args.headless_replay) and not args.check:
            gui = bool(args.render)
            print(f'\n【回放】{model}  '
                  f'{"GUI（关窗口退出）" if gui else "无头模式"}')
            if args.hand == 'both':
                base = [-0.3, 0, 0.3] if side == 'left' else [0.3, 0, 0.3]
            else:
                base = [0, 0, 0.3]
            replay_pybullet(angles, valid, data['timestamps'], urdf,
                            os.path.join(HAND_ROOT, side),
                            MAPPING_18, limits, base,
                            speed=args.speed, gui=gui)

    print()
    print('=' * 94)
    print('结论:', '所有映射维度都落在 l21 关节限位内  [OK]' if ok_all
          else '存在越界或缺失，请检查映射表/模型选择  [FAIL]')
    print('=' * 94)


from teleoperation.data.hand_h5 import read_native_angles as load_data

from teleoperation.data.urdf import parse_joint_limits

from teleoperation.robots.l21 import MAPPING_18, L21_JOINT_ORDER

from teleoperation.robots.l21 import L21HandAdapter
from teleoperation.contracts.hand import L21HandAngles
