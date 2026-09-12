"""离线回放脚本 —— 把重定向算法输出的动作序列灌进仿真环境

契约 H 的参考实现（格式定义见 docs/OFFLINE_PIPELINE.md）

用法：
    # ① 用假数据验证链路（不需要真实数据）
    python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render

    # ② 导出「契约H 示例文件」给队友A 照着写
    python scripts/replay_actions.py --dummy --save-actions datasets/samples/actions_demo_h1_2.h5

    # ③ 回放真实数据
    python scripts/replay_actions.py --file datasets/actions/pushcube_h1_2.h5

    # ④ 带可视化
    python scripts/replay_actions.py --file xxx.h5 --render

    # ⑤ 查看动作空间定义
    python scripts/replay_actions.py --describe --robot h1_2

支持格式：.npz / .h5 / .hdf5
"""
import argparse
import os
import sys
import time

import numpy as np

# 把项目根目录加入 sys.path（本脚本在 scripts/ 子目录下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# --------------------------------------------------------------------------
# 文件读写（契约H）
# --------------------------------------------------------------------------
def load_action_file(path):
    """读取契约H 的动作序列文件，返回 (actions, meta)"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到动作文件：{path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == '.npz':
        z = np.load(path, allow_pickle=True)
        if 'actions' not in z.files:
            raise ValueError("npz 中缺少必需的 'actions' 数组")
        actions = np.asarray(z['actions'], dtype=np.float32)
        meta = {}
        for k in z.files:
            if k == 'actions':
                continue
            v = z[k]
            meta[k] = v.item() if getattr(v, 'shape', ()) == () else v
        return actions, meta

    if ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            if 'actions' not in f:
                raise ValueError("h5 中缺少必需的 'actions' 数据集")
            actions = np.asarray(f['actions'][:], dtype=np.float32)
            meta = {}
            for k, v in f.attrs.items():
                meta[k] = v.decode() if isinstance(v, bytes) else v
            if 'timestamps' in f:
                meta['timestamps'] = np.asarray(f['timestamps'][:])
        return actions, meta

    raise ValueError(f"不支持的文件格式：{ext}（请用 .npz / .h5 / .hdf5）")


def save_action_file(path, actions, meta):
    """按契约H 写出动作序列文件（用于生成示例数据）"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    meta = dict(meta)

    if ext == '.npz':
        payload = {'actions': np.asarray(actions, dtype=np.float32)}
        payload.update(meta)
        np.savez_compressed(path, **payload)
    elif ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'w') as f:
            f.create_dataset('actions',
                             data=np.asarray(actions, dtype=np.float32))
            if 'timestamps' in meta:
                f.create_dataset(
                    'timestamps',
                    data=np.asarray(meta.pop('timestamps'), dtype=np.float64))
            for k, v in meta.items():
                f.attrs[k] = v
    else:
        raise ValueError(f"不支持的文件格式：{ext}")

    print(f"  已写出：{path}  ({os.path.getsize(path)/1024:.1f} KB)")


# --------------------------------------------------------------------------
# 假数据（用于验证链路 / 生成契约H 示例）
# --------------------------------------------------------------------------
def make_dummy_actions(action_dim, n_steps=480, fps=240.0):
    """生成一段正弦摆动的假动作序列（左臂 / 右臂 / 手指）"""
    t = np.arange(n_steps) / fps
    actions = np.zeros((n_steps, action_dim), dtype=np.float32)

    # 左臂 7 关节（索引 0~6）
    actions[:, 0] = 0.3 * np.sin(2 * np.pi * 0.5 * t)   # shoulder_pitch
    actions[:, 1] = 0.2 * np.sin(2 * np.pi * 0.4 * t)   # shoulder_roll
    actions[:, 3] = 0.5 * np.sin(2 * np.pi * 0.6 * t)   # elbow
    # 右臂 7 关节（索引 7~13）
    actions[:, 7] = -actions[:, 0]
    actions[:, 8] = 0.2 * np.sin(2 * np.pi * 0.4 * t + 1.0)
    actions[:, 10] = 0.5 * np.sin(2 * np.pi * 0.6 * t + 1.0)
    # 手部（索引 14 之后）：缓慢开合
    if action_dim > 14:
        grasp = (0.3 + 0.2 * np.sin(2 * np.pi * 0.3 * t))[:, None]
        actions[:, 14:] = np.repeat(grasp, action_dim - 14, axis=1)

    return actions


# --------------------------------------------------------------------------
# 维度校验 & 回放
# --------------------------------------------------------------------------
def validate_dim(actions, expected_dim, robot_type):
    """校验动作维度与机器人是否匹配"""
    if actions.ndim != 2:
        raise ValueError(
            f"actions 应为二维 (T, action_dim)，实际 shape={actions.shape}")
    got = actions.shape[1]
    if got != expected_dim:
        raise ValueError(
            f"动作维度不匹配！\n"
            f"  文件里的 action_dim = {got}\n"
            f"  但 robot_type='{robot_type}' 期望 {expected_dim}\n"
            f"  请检查 robot_type 是否正确，或重定向算法的输出维度。\n"
            f"  （h1_2=38 / gr1_t2=36 / g1=28）")
    return got


def replay(env, actions, randomize=False, verbose=True, label='',
           already_reset=False):
    """把动作序列逐步灌进环境

    Args:
        already_reset: True 表示调用方已经 reset 过
                       （用于「先校验动作空间、再回放」的流程）

    Returns:
        dict: {success, done, finished, steps, total_steps, elapsed}
    """
    if not already_reset:
        env.reset(randomize=randomize)
    total = len(actions)

    success = False
    done = False
    elapsed = 0.0
    i = 0
    t0 = time.time()
    for i in range(total):
        obs, success, done, elapsed = env.step(actions[i])
        if verbose and (i + 1) % 240 == 0:
            print(f"    ... {i+1}/{total} 步  ({time.time()-t0:.1f}s)")
        if done:
            break

    result = {
        'success': bool(success),
        'done': bool(done),
        'finished': bool(done),     # 是否在动作序列播完前就判定结束
        'steps': i + 1,
        'total_steps': total,
        'elapsed': float(elapsed),
    }
    if verbose:
        if success:
            status = '[OK] SUCCESS'
        elif done:
            status = '[FAIL] FAILED (timeout)'
        else:
            status = '— 序列播放完，但任务未完成'
        print(f"  [{label}] {status}")
        print(f"      播放 {result['steps']}/{total} 步, "
              f"仿真耗时 {result['elapsed']:.2f}s")
    return result


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='离线回放：把动作序列灌进仿真环境（契约H）')
    ap.add_argument('--file', type=str, default=None,
                    help='动作序列文件 (.npz/.h5/.hdf5)，契约H 格式')
    ap.add_argument('--dummy', action='store_true',
                    help='用假数据（正弦摆动）而非真实文件')
    ap.add_argument('--save-actions', type=str, default=None,
                    help='把动作序列另存为契约H 示例文件')
    ap.add_argument('--describe', action='store_true',
                    help='只打印动作空间定义后退出')
    ap.add_argument('--robot', type=str, default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--task', type=str, default='pushcube')
    ap.add_argument('--render', action='store_true', help='开启 GUI 可视化')
    ap.add_argument('--no-render', action='store_true', help='无头模式（默认）')
    ap.add_argument('--record', action='store_true', help='记录数据到 HDF5')
    ap.add_argument('--randomize', action='store_true',
                    help='回放时启用域随机化（默认关闭，保证可复现）')
    ap.add_argument('--steps', type=int, default=480,
                    help='--dummy 模式生成的步数')
    args = ap.parse_args()

    from envs import SimulationEnv
    import tasks.all_tasks          # noqa: F401  ← 触发 30 个任务注册
    from tasks import get_task      # noqa: F401

    # ---- describe：只打印动作空间定义
    if args.describe:
        import pybullet as p
        from envs import RobotLoader
        cid = p.connect(p.DIRECT)
        try:
            loader = RobotLoader(cid)
            loader.load_robot(args.robot)
            loader.describe_action_space()
        finally:
            p.disconnect(cid)
        return

    # ---- 准备动作序列
    meta = {}
    if args.dummy:
        probe = SimulationEnv(robot_type=args.robot, task_name=args.task,
                              render=False, record=False)
        try:
            probe.reset(randomize=False)
            action_dim = probe.action_dim
        finally:
            probe.close()
        print(f"\n[假数据模式] robot={args.robot}  action_dim={action_dim}  "
              f"steps={args.steps}")
        actions = make_dummy_actions(action_dim, n_steps=args.steps)
        meta = {'robot_type': args.robot, 'task_name': args.task,
                'fps': 240.0, 'source': 'dummy', 'algo': 'sine-wave-dummy'}
        if args.save_actions:
            save_action_file(args.save_actions, actions, dict(meta))
    else:
        if not args.file:
            ap.error('请指定 --file <动作文件>，或用 --dummy 生成假数据')
        print(f"\n[读取动作文件] {args.file}")
        actions, meta = load_action_file(args.file)
        print(f"  帧数 = {actions.shape[0]}  action_dim = {actions.shape[1]}")
        if meta.get('robot_type'):
            args.robot = str(meta['robot_type'])
        if meta.get('task_name'):
            args.task = str(meta['task_name'])
        print(f"  robot_type = {args.robot}   task_name = {args.task}")
        if args.save_actions:
            save_action_file(args.save_actions, actions, dict(meta))

    # ---- 建立环境并回放
    if args.render and args.no_render:
        ap.error('--render 与 --no-render 不能同时使用')
    render = bool(args.render) and not args.no_render

    print(f"\n[建立仿真环境] robot={args.robot} task={args.task} "
          f"render={render} record={args.record}")
    env = SimulationEnv(robot_type=args.robot, task_name=args.task,
                        render=render, record=args.record)
    try:
        # 注意 动作空间是在 reset() 里「按机器人确定」的，所以必须先 reset
        env.reset(randomize=args.randomize)
        if not env.action_dim:
            raise RuntimeError("env.action_dim 未初始化，请先调用 env.reset()")

        validate_dim(actions, env.action_dim, args.robot)
        print(f"  [OK] 动作维度校验通过（{env.action_dim}）")
        print(f"  [OK] 动作空间前 3 个: {env.action_joint_names[:3]}")
        print(f"\n[开始回放] {len(actions)} 帧")
        result = replay(env, actions, label=f"{args.task}/{args.robot}",
                        already_reset=True)
        print(f"\n[结果] {result}")
    finally:
        env.close()


if __name__ == '__main__':
    main()

