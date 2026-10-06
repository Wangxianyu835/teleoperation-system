"""
TeleOpBench 仿真平台主入口
用法:
    python main.py                        # 列出所有任务并交互式选择
    python main.py --task pushcube        # 运行指定任务
    python main.py --demo                 # 运行演示（无控制器模式）
    python main.py --benchmark            # 对Level 1任务进行基准测试
    python main.py --task pushcube --no-render  # 无头模式
"""

import sys
import os
import argparse
import time
from functools import partial

# 确保项目目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入所有任务（触发注册）
from tasks.all_tasks import *  # noqa: F401
from tasks import list_tasks, get_task, get_tasks_by_level
from envs import SimulationEnv


def demo_controller(obs, *, env):
    """演示用简单控制器：左右臂交替摆动 + 手指缓慢开合

    注意 动作向量长度由机器人决定（H1-2=38 / GR1-T2=36 / G1=28），
       不要硬编码 28！真实长度见 env.action_dim。
       索引含义见 env.action_joint_names（左臂7 + 右臂7 + 左手N + 右手N）。

    真实遥操作时，把这里换成重定向算法的输出（obs -> action）。
    """
    import numpy as np
    # run_episode() owns reset/robot loading. Read the current robot's action
    # space here, after that reset, rather than caching an uninitialized value.
    action_dim = env.action_dim
    if action_dim is None:
        raise RuntimeError('demo_controller requires env.reset() before generating actions')
    t = time.time()
    action = np.zeros(action_dim)
    s = 0.3 * np.sin(t * 2.0)
    # 左臂 7 关节（索引 0~6）
    action[0] = s                          # shoulder_pitch
    action[1] = 0.2 * np.sin(t * 1.5)      # shoulder_roll
    action[3] = 0.5 * np.sin(t * 1.8)      # elbow
    # 右臂 7 关节（索引 7~13）
    action[7] = -s                         # shoulder_pitch
    action[8] = 0.2 * np.sin(t * 1.5 + 3.0)
    action[10] = 0.5 * np.sin(t * 1.8 + 1.0)   # elbow
    # 手部（索引 14 之后）：缓慢开合
    if action_dim > 14:
        action[14:] = 0.3 + 0.2 * np.sin(t * 1.2)
    return action


def interactive_menu():
    """交互式菜单"""
    print("\n" + "=" * 60)
    print("  TeleOpBench - 双臂灵巧遥操作仿真基准平台")
    print("  (基于 PyBullet 实现)")
    print("=" * 60)
    print("\n可用任务列表：\n")

    all_tasks = list_tasks()
    for level in range(1, 5):
        level_tasks = get_tasks_by_level(level)
        print(f"  [Level {level}] 难度级别")
        for t in level_tasks:
            task_cls = get_task(t)
            doc = (task_cls.__doc__ or "").strip()
            print(f"    {t:<20} - {doc}")
        print()

    print("-" * 60)
    print("  输入任务名运行，或输入:")
    print("    list     - 重新列出任务")
    print("    benchmark- 对全部Level 1任务进行基准测试")
    print("    demo     - 运行演示")
    print("    quit     - 退出")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='TeleOpBench 仿真平台')
    parser.add_argument('--task', type=str, default=None,
                        help='任务名 (e.g., pushcube, pickcube)')
    parser.add_argument('--robot', type=str, default='h1_2',
                        choices=['h1_2', 'gr1_t2', 'g1'],
                        help='机器人类型')
    parser.add_argument('--no-render', action='store_true',
                        help='无GUI渲染（无头模式）')
    parser.add_argument('--no-record', action='store_true',
                        help='不记录数据')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--demo', action='store_true',
                       help='运行 3 次正弦波演示；默认 pushcube，可用 --task 指定任务')
    modes.add_argument('--benchmark', action='store_true',
                       help='对 Level 1 任务进行基准测试；不能与 --task 同用')
    parser.add_argument('--trials', type=int, default=5,
                        help='基准测试每个任务的试验次数')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='数据存储目录')
    args = parser.parse_args()
    if args.benchmark and args.task is not None:
        parser.error('--benchmark cannot be combined with --task')

    # 交互模式
    choice = None   # 注意 必须初始化：带 --task/--demo/--benchmark 时不会进入下面的交互分支
    if args.task is None and not args.demo and not args.benchmark:
        interactive_menu()

        while True:
            choice = input("\n> ").strip().lower()

            if choice == 'quit' or choice == 'exit':
                break
            elif choice == 'list':
                interactive_menu()
            elif choice == 'demo':
                args.demo = True
                args.task = 'pushcube'
                break
            elif choice == 'benchmark':
                args.benchmark = True
                break
            elif choice in list_tasks():
                args.task = choice
                break
            else:
                print(f"未知选择: {choice}")

    if choice in ('quit', 'exit'):
        return

    if args.demo and args.task is None:
        args.task = 'pushcube'

    # 创建仿真环境
    env = SimulationEnv(
        robot_type=args.robot,
        task_name=args.task or 'pushcube',
        render=not args.no_render,
        record=not args.no_record,
        data_dir=args.data_dir,
    )

    # Bind the environment without resetting it; each episode performs exactly
    # one reset, then the callback reads its actual action space.
    controller = partial(demo_controller, env=env)

    try:
        if args.task:
            print(f"\n启动任务: {args.task}")
            print(f"机器人: {args.robot}")
            print(f"渲染: {'开' if not args.no_render else '关'}")
            print(f"数据记录: {'开' if not args.no_record else '关'}")

            # 运行任务
            if args.demo:
                print("\n[演示模式] 使用简单正弦波控制器...")
                env.evaluate_task(num_trials=3, controller=controller)
            else:
                env.run_episode(controller=controller)
        elif args.benchmark:
            print(f"\n[基准测试模式] 测试 Level 1 任务...")
            level1_tasks = get_tasks_by_level(1)
            env.benchmark_all_tasks(
                task_list=level1_tasks,
                controller=controller,
                num_trials=args.trials,
            )

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        env.close()
        print("\n仿真环境已关闭。")


if __name__ == '__main__':
    main()
