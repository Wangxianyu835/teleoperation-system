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
    python scripts/replay_hand_angles.py --check                  # 只做静态检查
    python scripts/replay_hand_angles.py --hand right --render    # GUI 回放（右手）
    python scripts/replay_hand_angles.py --hand both --render --smooth 5
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

# LinkerHand 模型根目录
HAND_ROOT = os.path.join(ROOT, 'linkerhand_sdk', 'ros1', 'src',
                         'assets', 'robots', 'hands', 'linker_hand')

# ======================================================================
# ★★★ 映射表（dim 索引 -> LinkerHand l21 关节名）★★★
#  推导方式：把「每个维度的实测范围」与「l21 关节限位」逐一匹配，结果零越界。
#  若队友给出了官方顺序，请以此处为准修改（其它代码无需改动）。
# ======================================================================
MAPPING_18 = [
    None,                     # dim 0  占位
    'index_mcp_roll',         # 1
    'index_mcp_pitch',        # 2
    'index_pip',              # 3
    'middle_mcp_roll',        # 4
    'middle_mcp_pitch',       # 5
    'middle_pip',             # 6
    'ring_mcp_roll',          # 7
    'ring_mcp_pitch',         # 8
    'ring_pip',               # 9
    'pinky_mcp_roll',         # 10
    'pinky_mcp_pitch',        # 11
    'pinky_pip',              # 12
    'thumb_cmc_roll',         # 13
    'thumb_cmc_yaw',          # 14
    'thumb_cmc_pitch',        # 15
    'thumb_mcp',              # 16
    'thumb_ip',               # 17
]

# l21 的 URDF 关节顺序（用于交叉校验映射是否与 URDF 一致）
L21_JOINT_ORDER = [
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip',
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip',
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip',
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip',
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
]


def load_data(path):
    """读取队友输出的 H5"""
    import h5py
    if not os.path.isfile(path):
        raise FileNotFoundError(f'找不到数据文件：{path}')
    with h5py.File(path, 'r') as f:
        d = {
            'left_angles': np.asarray(f['left_angles'][:], dtype=np.float64),
            'right_angles': np.asarray(f['right_angles'][:], dtype=np.float64),
            'timestamps': np.asarray(f['timestamps'][:], dtype=np.float64),
        }
        for k in ('left_valid', 'right_valid'):
            d[k] = (np.asarray(f[k][:]).astype(bool) if k in f
                    else np.ones(len(d['timestamps']), dtype=bool))
    return d


def parse_joint_limits(urdf_path):
    """从 URDF 读关节限位 {关节名: (lower, upper)}"""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall('joint'):
        if j.get('type') == 'fixed':
            continue
        lim = j.find('limit')
        if lim is None:
            continue
        try:
            out[j.get('name')] = (float(lim.get('lower', -3.15)),
                                  float(lim.get('upper', 3.15)))
        except (TypeError, ValueError):
            pass
    return out


# 平滑 / 诊断相关实现统一放在 teleop/filters.py（单一实现，避免重复）
from teleop.filters import (            # noqa: E402
    KalmanSmoother, moving_average, classify_jumps, print_jump_report,
    compare_methods, print_metrics_table,
    detect_bad_frames, repair_bad_frames, print_bad_frames_report,
)


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

    t0 = timestamps[0]
    n_frames = 0
    final_q = {}
    for i in range(len(angles)):
        if not valid[i]:
            continue
        a = angles[i]
        for d, jidx in dim2jidx.items():
            val = float(a[d])
            jn = joints_order[d]
            if jn in limits:                      # 裁剪到限位内（安全）
                lo, hi = limits[jn]
                val = min(max(val, lo), hi)
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
def main():
    ap = argparse.ArgumentParser(
        description='回放重定向输出的手部角度（LinkerHand l21）')
    ap.add_argument('--file', type=str,
                    default=os.path.join(ROOT, 'datasets', 'raw',
                                         'retarget_twohand_153542.h5'),
                    help='队友输出的 h5 文件')
    ap.add_argument('--hand', choices=['left', 'right', 'both'],
                    default='right')
    ap.add_argument('--check', action='store_true',
                    help='只做静态检查，不渲染')
    ap.add_argument('--render', action='store_true',
                    help='GUI 可视化回放（不加则只做静态检查）')
    ap.add_argument('--headless-replay', action='store_true',
                    help='无头模式回放（不开窗口，用于自动验证）')
    ap.add_argument('--smooth', type=int, default=0,
                    help='滑动平均窗口（奇数，如 5）')
    ap.add_argument('--kalman', action='store_true',
                    help='启用卡尔曼滤波（搭配 --kalman-q/--kalman-r 调参）')
    ap.add_argument('--kalman-q', type=float, default=1.0,
                    help='卡尔曼过程噪声 q（默认 1.0）')
    ap.add_argument('--kalman-r', type=float, default=0.01,
                    help='卡尔曼观测噪声 r（默认 0.01）')
    ap.add_argument('--gate', type=float, default=0.0,
                    help='创新门控倍数（默认 0=关闭；仅孤立尖峰数据才需要）')
    ap.add_argument('--analyze', action='store_true',
                    help='打印跳变性质诊断（孤立尖峰 vs 成片真实运动）')
    ap.add_argument('--no-repair', action='store_true',
                    help='不修复「整帧异常」坏帧（默认会自动检测并插值修复）')
    ap.add_argument('--compare', action='store_true',
                    help='对比 原始/移动平均/卡尔曼 的平滑指标')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='数据帧率（用于滤波器 dt 与指标换算，默认 30）')
    ap.add_argument('--speed', type=float, default=1.0, help='播放倍速')
    args = ap.parse_args()

    print('=' * 94)
    print('回放重定向输出的手部角度  ->  LinkerHand l21')
    print('=' * 94)
    print('数据文件 =', args.file)

    data = load_data(args.file)
    ok_all = True
    sides = [args.hand] if args.hand != 'both' else ['left', 'right']

    for side in sides:
        model = f'l21_{side}'
        urdf = os.path.join(HAND_ROOT, model, f'linkerhand_{model}.urdf')
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
                            os.path.join(HAND_ROOT, model),
                            MAPPING_18, limits, base,
                            speed=args.speed, gui=gui)

    print()
    print('=' * 94)
    print('结论:', '所有映射维度都落在 l21 关节限位内  [OK]' if ok_all
          else '存在越界或缺失，请检查映射表/模型选择  [FAIL]')
    print('=' * 94)


if __name__ == '__main__':
    main()


