"""离线回放脚本 —— 把重定向算法输出的动作序列灌进仿真环境

契约 H 的参考实现（格式定义见 docs/contracts.md）

用法：
    # ① 用假数据验证链路（不需要真实数据）
    python -m teleoperation replay actions --dummy --robot h1_2 --task pushcube --no-render

    # ② 导出「契约H 示例文件」给队友A 照着写
    python -m teleoperation replay actions --dummy --save-actions datasets/samples/actions_demo_h1_2.h5

    # ③ 回放真实数据
    python -m teleoperation replay actions --file datasets/actions/pushcube_h1_2.h5

    # ④ 带可视化
    python -m teleoperation replay actions --file xxx.h5 --render

    # ⑤ 查看动作空间定义
    python -m teleoperation replay actions --describe --robot h1_2

支持格式：.npz / .h5 / .hdf5
"""
import argparse
import os
import sys
import time

import numpy as np

# 默认资源由仓库路径模块定位
from teleoperation.paths import PROJECT_ROOT
ROOT = str(PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# --------------------------------------------------------------------------
# 文件读写（契约H）
# --------------------------------------------------------------------------


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
def main(args=None):
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    from teleoperation.apps.simulation import create_environment as SimulationEnv, run_episode, evaluate_task, benchmark_all_tasks
    import teleoperation.simulation.tasks          # noqa: F401  <- 触发 30 个任务注册
    from teleoperation.simulation.tasks import get_task      # noqa: F401

    # ---- describe：只打印动作空间定义
    if args.describe:
        import pybullet as p
        from teleoperation.simulation import RobotLoader
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
            args._parser.error('请指定 --file <动作文件>，或用 --dummy 生成假数据')
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
        args._parser.error('--render 与 --no-render 不能同时使用')
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


from teleoperation.data.actions import load_action_file, save_action_file
