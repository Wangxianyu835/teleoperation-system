"""原装手回放【运动量实测】：确认三台机器人的手是不是真的会动

为什么需要这个脚本
------------------
只靠肉眼看 GUI，有两类问题很难发现：

  1. 关节【根本没动】 —— 映射生成了指令，但关节限位极窄 / 关节被锁 /
     力矩不够，手指在外观上几乎没有变化（尤其 G1 的轻量三指手很小）。
  2. 手指动了，但【方向反了】 —— 符号判定（sign = +1 if |hi| >= |lo| else -1）
     在某些关节上判断错误时，手指会朝手背方向弯。

本脚本把同一份数据在 DIRECT（无界面）模式下真跑一遍，
逐关节量出【实际转过的弧度】与【跟踪误差】，并给出 PASS / FAIL。

判据
----
  travel = max(实际角度) - min(实际角度)
  [PASS] 该关节 travel >= 0.05 rad（约 2.9 度，肉眼可见）
  [WARN] 0 < travel < 0.05 rad（动了但幅度小）
  [FAIL] travel ~ 0（完全没动）

用法
----
    python scripts/check_native_hand_motion.py
    python scripts/check_native_hand_motion.py --file datasets/raw/my_recording_angles.h5
    python scripts/check_native_hand_motion.py --robots h1_2 gr1_t2 g1 --limit-mode clamp
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from teleop.filters import (  # noqa: E402
    detect_bad_frames, detect_identity_swaps, repair_stream,
)
from teleop.native_hand import (  # noqa: E402
    DIM_NAMES, N_MOVABLE, build_mapping, dropped_dims,
    map_frame, read_joint_ranges,
)

# travel 判据（弧度）
PASS_RAD = 0.05
WARN_RAD = 0.005

DT = 1.0 / 240.0
FORCE = 200.0          # 与 replay_hand_native.py 保持一致


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


def joint_names_of(mapping):
    return [jn for _d, jn, *_ in mapping]


def replay_robot(robot_type, data, limit_mode, verbose=True):
    """把数据回放到指定机器人上，返回逐关节的运动量统计"""
    import pybullet as p
    import pybullet_data
    from envs import RobotLoader

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setTimeStep(DT, physicsClientId=cid)
        loader = RobotLoader(cid)
        rid = loader.load_robot(robot_type)['robot']

        swaps = detect_identity_swaps(data['left_angles'],
                                      data['right_angles'],
                                      data['left_valid'], data['right_valid'])
        swap_by_side = {}
        for s in swaps:
            swap_by_side.setdefault(s['side'], []).append(s['frame'])

        frame_dt = (float(np.median(np.diff(data['timestamps'])))
                    if len(data['timestamps']) > 1 else 1.0 / 30.0)
        substeps = max(1, int(round(frame_dt / DT)))

        T = len(data['timestamps'])
        out = {}
        for side in ('left', 'right'):
            ranges, names = read_joint_ranges(rid, cid)
            mapping = build_mapping(robot_type, side, ranges)
            arr = data[f'{side}_angles'].copy()
            val = data[f'{side}_valid']
            bad = detect_bad_frames(arr, val)
            fronts = sorted(set(swap_by_side.get(side, [])))
            arr, _held, _interp = repair_stream(arr, val, bad, fronts)

            # 记录每帧的【实际】关节角度
            trace = {jn: [] for jn in joint_names_of(mapping)}
            cmd_last, clip, tot = {}, 0, 0
            for i in range(T):
                if not val[i]:
                    continue
                joints, nclip = map_frame(arr[i], mapping, limit_mode=limit_mode)
                clip += nclip
                tot += len(mapping)
                for jn, v in joints.items():
                    p.setJointMotorControl2(rid, names[jn], p.POSITION_CONTROL,
                                            targetPosition=v, force=FORCE,
                                            physicsClientId=cid)
                    cmd_last[jn] = v
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=cid)
                for jn in joint_names_of(mapping):
                    trace[jn].append(
                        p.getJointState(rid, names[jn], physicsClientId=cid)[0])

            # 末帧误差必须在【稳定后】测：数据放完时关节通常还差一帧没跟上，
            # 直接用末帧位置去比会得到约 0.15 rad 的固定滞后（三台机器人相同），
            # 那是采样相位差，不是跟踪失败。这里再空推 120 步让关节收敛。
            settle = 0.0
            if cmd_last:
                for _ in range(120):
                    for jn, v in cmd_last.items():
                        p.setJointMotorControl2(rid, names[jn],
                                                p.POSITION_CONTROL,
                                                targetPosition=v, force=FORCE,
                                                physicsClientId=cid)
                    p.stepSimulation(physicsClientId=cid)

            rows = []
            for d, jn, sg, lo, hi in mapping:
                tr = np.asarray(trace[jn], dtype=float)
                travel = float(tr.max() - tr.min()) if tr.size else 0.0
                err = (abs(float(cmd_last.get(jn, 0.0))
                           - p.getJointState(rid, names[jn],
                                             physicsClientId=cid)[0])
                       if tr.size else 0.0)
                settle = max(settle, err)
                rows.append({
                    'dim': d, 'sem': DIM_NAMES[d], 'joint': jn, 'sign': sg,
                    'lo': lo, 'hi': hi, 'travel': travel, 'err': err,
                    'final': float(tr[-1]) if tr.size else 0.0,
                })
            out[side] = {
                'rows': rows,
                'dropped': dropped_dims(robot_type, side, mapping),
                'clip': clip, 'tot': tot,
            }
            if verbose:
                print()
                print(f'  --- {robot_type} / {side} ---')
                print(f'  可表达 {len(mapping)}/{N_MOVABLE} 维，'
                      f'丢弃 {len(dropped_dims(robot_type, side, mapping))} 维；'
                      f'限位截断 {100.0 * clip / max(tot, 1):.1f}% '
                      f'({clip}/{tot})')
                print(f'  {"语义":<18} {"关节名":<34} {"符号":<5} '
                      f'{"限位":<22} {"转动量":<10} 判定')
                for r in rows:
                    verdict = ('[PASS]' if r['travel'] >= PASS_RAD
                               else ('[WARN]' if r['travel'] > WARN_RAD
                                     else '[FAIL]'))
                    lim = f'[{r["lo"]:+.2f},{r["hi"]:+.2f}]'
                    print(f'  {r["sem"]:<18} {r["joint"]:<34} {r["sign"]:+.0f}   '
                          f'{lim:<22} {r["travel"]:<10.3f} {verdict}')
        return out
    finally:
        p.disconnect(cid)


def main():
    ap = argparse.ArgumentParser(description='原装手回放运动量实测')
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'my_recording_angles.h5'))
    ap.add_argument('--robots', nargs='+', default=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--limit-mode', default='clamp',
                    choices=['clamp', 'rescale'])
    ap.add_argument('--quiet', action='store_true', help='只打印汇总')
    args = ap.parse_args()

    print('=' * 96)
    print('原装手回放运动量实测：同一份数据 -> 三台机器人的手（DIRECT 无界面）')
    print('=' * 96)
    print(f'数据 = {args.file}')

    data = load_data(args.file)

    summary = []
    for rn in args.robots:
        res = replay_robot(rn, data, args.limit_mode, verbose=not args.quiet)
        n_pass = n_warn = n_fail = 0
        for side in res:
            for r in res[side]['rows']:
                if r['travel'] >= PASS_RAD:
                    n_pass += 1
                elif r['travel'] > WARN_RAD:
                    n_warn += 1
                else:
                    n_fail += 1
        max_err = max((r['err'] for side in res for r in res[side]['rows']),
                      default=0.0)
        summary.append((rn, n_pass, n_warn, n_fail, max_err))

    print()
    print('=' * 96)
    print('汇总')
    print('=' * 96)
    print(f'  {"机器人":<10} {"会动":<8} {"幅度偏小":<12} {"没动":<8} '
          f'{"最大跟踪误差":<18} 结论')
    for rn, n_pass, n_warn, n_fail, max_err in summary:
        ok = (n_fail == 0 and max_err < 0.05)
        print(f'  {rn:<10} {n_pass:<8} {n_warn:<12} {n_fail:<8} '
              f'{max_err:<18.4f} '
              f'{"[PASS] 可以直接演示" if ok else "[FAIL] 需检查"}')
    print('=' * 96)


if __name__ == '__main__':
    main()
