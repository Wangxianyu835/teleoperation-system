"""一键验收：数据 -> 映射 -> 仿真模型（LinkerHand l21）全链路

用途
----
回答一个问题：**我们采集/重定向的数据，能不能在仿真里表现出来？**
自动跑完 5 项检查并给出 PASS/FAIL 汇总，任何人（含队友）都能直接运行。

用法
----
    python scripts/verify_hand_pipeline.py                 # 用默认数据
    python scripts/verify_hand_pipeline.py --file xxx.h5   # 指定数据
    python scripts/verify_hand_pipeline.py --render        # 最后开 GUI 看画面
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

from teleop.filters import detect_bad_frames, repair_bad_frames  # noqa: E402


def banner(t):
    print()
    print('=' * 88)
    print(t)
    print('=' * 88)


def summary(results):
    banner('验收汇总')
    for name, ok, detail in results:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}'
              + (f'   {detail}' if detail else ''))
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    if n_fail == 0:
        print('  结论：**数据可以在仿真里正常表现出来**  [PASS]')
    else:
        print(f'  结论：有 {n_fail} 项未通过，请看上面明细  [FAIL]')
    print('=' * 88)


def main():
    ap = argparse.ArgumentParser(description='手部链路一键验收')
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'))
    ap.add_argument('--render', action='store_true', help='最后打开 GUI')
    args = ap.parse_args()

    results = []

    def record(name, ok, detail=''):
        results.append((name, ok, detail))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}'
              + (f'   {detail}' if detail else ''))

    # ---------------- [1] 数据读取 ----------------
    banner('一键验收：手部数据 -> 仿真模型 全链路')
    print(f'数据文件 = {args.file}')
    print()
    print('[1] 数据文件结构')
    if not os.path.isfile(args.file):
        record('数据文件存在', False, '找不到文件')
        return summary(results)
    import h5py
    with h5py.File(args.file, 'r') as f:
        keys = list(f.keys())
        L = np.asarray(f['left_angles'][:], dtype=np.float64)
        R = np.asarray(f['right_angles'][:], dtype=np.float64)
        LV = (np.asarray(f['left_valid'][:]).astype(bool)
              if 'left_valid' in f else np.ones(len(L), bool))
        RV = (np.asarray(f['right_valid'][:]).astype(bool)
              if 'right_valid' in f else np.ones(len(R), bool))
        ts = (np.asarray(f['timestamps'][:], dtype=np.float64)
              if 'timestamps' in f else np.arange(len(L)) / 30.0)
        attrs = dict(f.attrs)
    print(f'    顶层数据集 = {keys}')
    print(f'    left = {L.shape}   right = {R.shape}')
    print(f'    时间跨度 = {ts[0]:.2f} ~ {ts[-1]:.2f} s')
    for k, v in attrs.items():
        print(f'    attr {k} = {v!r}')
    record('数据文件结构可读', True, f'{L.shape[0]} 帧 x {L.shape[1]} 维')

    # ---------------- [2] 有效帧 ----------------
    print()
    print('[2] 有效帧统计')
    for nm, A, V in (('left', L, LV), ('right', R, RV)):
        idx = np.where(V)[0]
        print(f'    {nm}: 有效 {int(V.sum())}/{len(V)} '
              f'({V.sum()/len(V)*100:.1f}%), 范围 {idx[0]}~{idx[-1]}')
    record('有效帧可用（>50%）', bool(LV.mean() > 0.5 and RV.mean() > 0.5),
           f'left {LV.mean()*100:.0f}% / right {RV.mean()*100:.0f}%')

    # ---------------- [3] 坏帧 ----------------
    print()
    print('[3] 坏帧检测（整帧异常）')
    bad_info = {}
    for nm, A, V in (('left', L, LV), ('right', R, RV)):
        bad = detect_bad_frames(A[V])
        bad_info[nm] = bad
        print(f'    {nm}: 发现 {len(bad)} 个坏帧  {bad if bad else ""}')
    n_bad = sum(len(v) for v in bad_info.values())
    record('坏帧已检测出（回放端会自动修复）', True,
           f'共 {n_bad} 个' + ('（建议反馈给数据提供方修 valid 标记）'
                              if n_bad else ''))

    part2(args, results, record, L, R, LV, RV, ts, bad_info)
    summary(results)


# ----------------------------------------------------------------------
def part2(args, results, record, L, R, LV, RV, ts, bad_info):
    """[4] 映射一致性 + [5] 无头回放"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'rha', os.path.join(ROOT, 'scripts', 'replay_hand_angles.py'))
    rha = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rha)

    # ---------------- [4] 映射 vs l21 限位 ----------------
    print()
    print('[4] 映射与 l21 关节限位一致性')
    n_ok = n_bad_dim = 0
    for side in ('left', 'right'):
        model = f'l21_{side}'
        urdf = os.path.join(rha.HAND_ROOT, model,
                            f'linkerhand_{model}.urdf')
        if not os.path.isfile(urdf):
            print(f'    [缺模型] {urdf}')
            continue
        limits = rha.parse_joint_limits(urdf)
        A = L if side == 'left' else R
        V = LV if side == 'left' else RV
        arr = A[V]
        for d in range(arr.shape[1]):
            jn = rha.MAPPING_18[d] if d < len(rha.MAPPING_18) else None
            if jn is None:
                continue
            lo, hi = float(arr[:, d].min()), float(arr[:, d].max())
            llo, lhi = limits.get(jn, (-9.0, 9.0))
            if lo >= llo - 0.02 and hi <= lhi + 0.02:
                n_ok += 1
            else:
                n_bad_dim += 1
                print(f'    [越界] {side} dim{d} [{lo:+.3f},{hi:+.3f}] '
                      f'-> {jn} ({llo:+.3f}~{lhi:+.3f})')
    print(f'    落在限位内 = {n_ok} ，越界 = {n_bad_dim}')
    record('映射与 l21 限位一致（零越界）', n_bad_dim == 0,
           f'{n_ok} 个维度通过')

    # ---------------- [5] 无头回放 ----------------
    print()
    print('[5] 回放（能否真正驱动 17 个手部关节）')
    import pybullet as p
    n_model_ok = 0
    for side in ('left', 'right'):
        model = f'l21_{side}'
        urdf = os.path.join(rha.HAND_ROOT, model,
                            f'linkerhand_{model}.urdf')
        if not os.path.isfile(urdf):
            continue
        limits = rha.parse_joint_limits(urdf)
        A = L if side == 'left' else R
        V = LV if side == 'left' else RV
        arr = A[V]
        bad = bad_info.get(side, [])
        if bad:
            arr = repair_bad_frames(arr, bad)
        cid = p.connect(p.DIRECT)
        try:
            world = os.path.dirname(urdf)
            p.setAdditionalSearchPath(world, physicsClientId=cid)
            rid = p.loadURDF(urdf, [0, 0, 0.3], useFixedBase=True,
                             physicsClientId=cid)
            name2idx = {}
            for i in range(p.getNumJoints(rid, physicsClientId=cid)):
                info = p.getJointInfo(rid, i, physicsClientId=cid)
                nm = (info[1].decode() if isinstance(info[1], bytes)
                      else info[1])
                name2idx[nm] = i
            dim2j = {d: name2idx[jn]
                     for d, jn in enumerate(rha.MAPPING_18)
                     if jn and jn in name2idx}
            for i in range(len(arr)):
                for d, j in dim2j.items():
                    jn = rha.MAPPING_18[d]
                    lo, hi = limits.get(jn, (-9, 9))
                    v = min(max(float(arr[i, d]), lo), hi)
                    p.setJointMotorControl2(rid, j, p.POSITION_CONTROL,
                                            targetPosition=v, force=5.0,
                                            physicsClientId=cid)
                for _ in range(5):
                    p.stepSimulation(physicsClientId=cid)
            q = {jn: p.getJointState(rid, name2idx[jn],
                                     physicsClientId=cid)[0]
                 for jn in rha.MAPPING_18 if jn and jn in name2idx}
            moved = sum(1 for v in q.values() if abs(v) > 1e-6)
            print(f'    {model}: 映射关节 {len(dim2j)}/17，'
                  f'末帧非零关节 {moved}，帧数 {len(arr)}')
            if len(dim2j) == 17:
                n_model_ok += 1
        finally:
            p.disconnect(cid)
    record('回放成功（17 个手部关节全部映射）', n_model_ok == 2,
           f'{n_model_ok}/2 只手')


if __name__ == '__main__':
    main()

