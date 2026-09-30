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

from teleop.filters import (  # noqa: E402
    detect_bad_frames, detect_identity_swaps, repair_bad_frames,
    repair_stream,
)


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

    # ---------------- [3] 坏帧（整帧塌零） ----------------
    print()
    print('[3] 坏帧检测（整帧塌零）+ 检测器自检')
    bad_info = {}
    for nm, A, V in (('left', L, LV), ('right', R, RV)):
        bad = detect_bad_frames(A, V)
        bad_info[nm] = bad
        print(f'    {nm}: 检出 {len(bad)} 个塌零坏帧'
              f'{"（绝对下标）" + str(bad) if bad else ""}')
    n_bad = sum(len(v) for v in bad_info.values())

    # 自检 A：注入一个真·塌零帧，必须被精确检出
    V = LV
    v_idx = np.where(V)[0]
    probe = int(v_idx[len(v_idx) // 2])
    inj = L.copy()
    inj[probe] = 0.0
    hit_inject = detect_bad_frames(inj, V)
    ok_inject = (hit_inject == [probe])

    # 自检 B：真实数据里被【旧判据】误报过的快速运动帧，不应再被检出
    FP_OLD = (435, 522)          # 旧判据曾误报（435 绝对是「手张开」的 V 形谷底）
    real_bad = set(bad_info['left']) | set(bad_info['right'])
    ok_no_fp = all(f not in real_bad for f in FP_OLD)

    print(f'    自检A 注入塌零帧 {probe} -> 检出 {hit_inject}  '
          f'{"[OK]" if ok_inject else "[FAIL]"}')
    print(f'    自检B 真实快动作 {FP_OLD} 未被误判  '
          f'{"[OK]" if ok_no_fp else "[FAIL]"}')
    record('坏帧检测器自检（只抓塌零、不误判快动作）',
           bool(ok_inject and ok_no_fp),
           f'注入帧精确检出；真实快动作零误报；本次数据检出 {n_bad} 个塌零帧')

    # ---------------- [3b] 左右手身份切换 ----------------
    print()
    print('[3b] 左右手身份切换检测（MediaPipe 标签污染）')
    swaps = detect_identity_swaps(L, R, LV, RV)
    for s in swaps:
        print(f'    {s["frame"]}: {s["side"]} 的标签被 {s["other"]} 污染  '
              f'd_same={s["d_same"]:.4f}  d_cross={s["d_cross"]:.4f}  '
              f'另一侧同时消失={s["other_lost"]}')
    if not swaps:
        print('    未发现身份切换')
    # 自检：关掉 valid 门控后会误报多少？（证明门控必要）
    no_gate = detect_identity_swaps(L, R, LV, RV, use_valid_gate=False)
    has535 = any(s['frame'] == 535 and s['side'] == 'right' for s in no_gate)
    # [注意] 535 这个【参考特征帧】只存在于队友那份数据
    #   （datasets/raw/retarget_twohand_153542.h5，557 帧）。
    #   换成别的数据（如自己采集的 my_recording_angles.h5，809 帧）时，
    #   不能因为"没有这一帧"就判 FAIL —— 那会让人误以为数据有问题。
    #   因此只有【参考数据】才套用「必须命中 right[535]」这条断言。
    is_ref = os.path.basename(args.file) == 'retarget_twohand_153542.h5'
    if is_ref:
        ref_ok = (len(swaps) == 1 and swaps[0]['frame'] == 535
                  and swaps[0]['side'] == 'right'
                  and swaps[0]['other_lost'])
        ref_note = ('参考数据：right[535] 命中' if ref_ok
                    else '参考数据：未命中 right[535]（疑似回归）')
    else:
        ref_ok = True
        ref_note = (f'非参考数据（{os.path.basename(args.file)}），'
                    f'跳过"必须命中 535"断言，本数据检出 {len(swaps)} 处')
    # 通用判据（任何数据都成立）：valid 门控只会【减少】检出，不会凭空造出
    gated = {(s['frame'], s['side']) for s in swaps}
    ungated = {(s['frame'], s['side']) for s in no_gate}
    ok_subset = gated.issubset(ungated)
    print(f'    自检 关闭 valid 门控 -> 误报 {len(no_gate)} 处'
          f'（其中含 535 吗: {has535}）；说明「另一侧同时消失」这个门控是必要的')
    print(f'    门控后检出 {len(swaps)} 处，是否为未门控检出的子集: {ok_subset}')
    record('身份切换检测（门控不增加误报 + 参考帧断言）',
           bool(ref_ok and ok_subset),
           f'{ref_note}；关掉 valid 门控会误报 {len(no_gate)} 处')

    part2(args, results, record, L, R, LV, RV, ts, bad_info, swaps)
    summary(results)


# ----------------------------------------------------------------------
def part2(args, results, record, L, R, LV, RV, ts, bad_info, swaps):
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
        bad = bad_info.get(side, [])
        swap_f = [s['frame'] for s in swaps if s['side'] == side]
        # bad / swap_f 都是【绝对下标】：先修完整序列，再筛有效帧
        # 组合策略：身份切换帧「保持」、塌零帧「插值」
        arr = repair_stream(A, V, bad, swap_f)[0][V]
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

    part3(args, results, record)


# ----------------------------------------------------------------------
def part3(args, results, record):
    """[6] 原装手降维映射（当前项目采用的方案：不换手）"""
    import pybullet as p
    from envs import RobotLoader
    from teleop.native_hand import (build_mapping, coverage, dropped_dims,
                                    map_frame, read_joint_ranges)

    print()
    print('[6] 原装手降维映射（本项目采用：用机器人【自带】的手，不换手）')

    expected = {'h1_2': 12, 'gr1_t2': 11, 'g1': 7}
    cid = p.connect(p.DIRECT)
    loader = RobotLoader(cid)
    all_ok = True
    rows = []
    try:
        for rn in ('h1_2', 'gr1_t2', 'g1'):
            rid = loader.load_robot(rn)['robot']
            ranges, names = read_joint_ranges(rid, cid)
            per = {}
            for side in ('left', 'right'):
                mp = build_mapping(rn, side, ranges)
                per[side] = mp
                ok_side = (len(mp) == expected[rn])
                all_ok &= ok_side
                print(f'    {rn:<8} {side:<5} 可表达 {len(mp)}/17 '
                      f'({coverage(rn, side, mp) * 100:.0f}%)  '
                      f'丢弃 {len(dropped_dims(rn, side, mp))} 个  '
                      f'{"[OK]" if ok_side else "[FAIL]"}')
            rows.append((rn, rid, per, names))
            p.removeBody(rid)
    finally:
        p.disconnect(cid)

    # 真回放一次：以 H1-2 右手为例，确认映射后的关节能【跟上命令】
    # 注意：只在 valid=True 的帧上跑 —— 无效帧的角度是被置零的，
    #       拿它当"末帧"会误判成"关节没动"。
    import h5py
    ok_replay = False
    detail = ''
    try:
        with h5py.File(args.file, 'r') as f:
            ar = np.asarray(f['right_angles'][:], dtype=np.float64)
            av = np.asarray(f['right_valid'][:]).astype(bool)
        idx = np.where(av)[0]
        cid = p.connect(p.DIRECT)
        p.setTimeStep(1.0 / 240.0, physicsClientId=cid)
        loader = RobotLoader(cid)
        rid = loader.load_robot('h1_2')['robot']
        ranges, names = read_joint_ranges(rid, cid)
        mp = build_mapping('h1_2', 'right', ranges)
        step = max(1, len(idx) // 60)
        # 让仿真时间跟上数据时间：每 step 帧数据 = step*33ms，
        # 按 1/240 时间步就需要 step*8 个物理步，否则关节必然滞后
        nsub = max(1, int(round(step * (1.0 / 30.0) / (1.0 / 240.0))))
        joints = {}
        for i in idx[::step]:
            joints, _ = map_frame(ar[i], mp)
            for jn, v in joints.items():
                p.setJointMotorControl2(rid, names[jn], p.POSITION_CONTROL,
                                        targetPosition=v, force=200.0,
                                        physicsClientId=cid)
            for _ in range(nsub):
                p.stepSimulation(physicsClientId=cid)
        err = max(abs(p.getJointState(rid, names[jn],
                                      physicsClientId=cid)[0] - v)
                  for jn, v in joints.items())
        ok_replay = (err < 0.05)
        detail = (f'H1-2 右手 {len(joints)} 个映射关节跟踪误差 '
                  f'{err:.4f} rad（每采样 {nsub} 物理步）'
                  f'{"[OK]" if ok_replay else "[FAIL]"}')
        p.removeBody(rid)
        p.disconnect(cid)
    except Exception as exc:                  # noqa: BLE001
        detail = f'回放异常: {type(exc).__name__}: {exc}'

    print(f'    {detail}')
    record('原装手映射可真正驱动（12/11/7，符号自动判定）',
           bool(all_ok and ok_replay),
           'h1_2 12/17, gr1_t2 11/17, g1 7/17；' + detail)



if __name__ == '__main__':
    main()

