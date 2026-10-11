"""Formatting of hand diagnostic results."""

def print_jump_report(info, title='跳变性质诊断'):
    """把 classify_jumps 的结果打印成报告"""
    if not info:
        print('（数据不足，无法诊断）')
        return
    print()
    print('=' * 92)
    print(title)
    print('=' * 92)
    print(f"  帧数 = {info['n_frames']}   超阈值帧 = {info['n_jumps']} "
          f"({info['jump_ratio']*100:.1f}%)")
    print(f"  ★ 孤立尖峰（疑似噪声）   = {info['n_isolated']}")
    print(f"  ★ 成片快速运动（真实动作）= {info['n_sustained']}")

    v = info['verdict']
    concl = {
        'clean': '没有跳变 —— 数据很干净，无需滤波',
        'noise': '以孤立尖峰为主 —— 是噪声，【建议】滤波',
        'real_motion': '以成片快动作为主 —— 是真实运动，【不要】滤波！',
        'mixed': '两者都有 —— 建议只对孤立尖峰做轻度处理',
    }[v]
    print(f'  -> 结论：{concl}')

    if info['timeline']:
        print()
        print('  跳变的分布（按 50 帧分段）：')
        for s, e, c in info['timeline']:
            print(f'    帧 {s:>4d}~{e:>4d}: {c:>3d}  {"#" * c}')

    if info['isolated_frames']:
        print()
        print(f"  孤立尖峰的帧号：{info['isolated_frames'][:30]}")
    print('=' * 92)


def print_identity_swap_report(swaps, title='左右手身份切换检测'):
    """打印身份切换报告"""
    print()
    print('=' * 96)
    print(title)
    print('=' * 96)
    if not swaps:
        print('  未发现左右手身份切换  [OK]')
    else:
        print(f'  发现 {len(swaps)} 处：')
        print(f'  {"帧":>6} {"侧":>6} {"换成了谁":>9} {"d_same":>9} '
              f'{"d_cross":>9}  另一侧同时消失')
        for s in swaps:
            print(f'  {s["frame"]:>6} {s["side"]:>6} {s["other"]:>9} '
                  f'{s["d_same"]:>9.4f} {s["d_cross"]:>9.4f}  '
                  f'{"是" if s["other_lost"] else "否"}')
        print()
        print('  修复策略：这些帧用 hold_last_valid()'
              '【保持上一有效姿态】，' + '不要用插值')
        print('            （插值会被同样受污染的下一帧拉偏）')
    print('=' * 96)


def print_bad_frames_report(bad_frames, n_frames, title='坏帧检测'):
    """打印坏帧报告（bad_frames 为【绝对下标】）"""
    print()
    print('=' * 92)
    print(title)
    print('=' * 92)
    if not bad_frames:
        print(f'  共 {n_frames} 帧，未发现「整帧塌零」坏帧  [OK]')
        print('  说明：数据里成片的快速运动是【真实动作】，不算坏帧。')
    else:
        print(f'  共 {n_frames} 帧，发现 {len(bad_frames)} 个坏帧（整帧塌零）:')
        print(f'    【绝对下标】{bad_frames}')
        print('  [!] 建议：')
        print('     1) 回放端会用 repair_bad_frames() 插值修复（--no-repair 可关）')
        print('     2) 若能定位成因，请在导出侧修正（但需先核对原始数据）')
    print('=' * 92)


def print_metrics_table(rows, title='平滑效果对比'):
    """把 compare_methods 的结果打印成表格"""
    keys = ['max_delta', 'n_jumps', 'jump_ratio', 'rms_vel',
            'rms_accel', 'noise_ratio']
    head = ['方法', '最大单帧变化', '跳变帧数', '跳变比例',
            'rms速度', 'rms加速度', '噪声比']
    print()
    print('=' * 96)
    print(title)
    print('=' * 96)
    print(f'{head[0]:<22s} {head[1]:>13s} {head[2]:>10s} {head[3]:>10s} '
          f'{head[4]:>10s} {head[5]:>11s} {head[6]:>10s}')
    print('-' * 96)
    for name, m in rows:
        if not m:
            continue
        print(f'{name:<22s} {m["max_delta"]:>13.4f} {m["n_jumps"]:>10d} '
              f'{m["jump_ratio"]*100:>9.2f}% {m["rms_vel"]:>10.4f} '
              f'{m["rms_accel"]:>11.4f} {m["noise_ratio"]:>10.4f}')
    print('=' * 96)
    print('说明：最大单帧变化 / 跳变帧数 越少越好；'
          'rms加速度 越小越平滑；噪声比 = rms加速度/rms速度')

