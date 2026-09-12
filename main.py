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

# 确保项目目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入所有任务（触发注册）
from tasks.all_tasks import *  # noqa: F401
from tasks import list_tasks, get_task, get_tasks_by_level
from envs import SimulationEnv


def demo_controller(obs):
    """
    演示用简单控制器：交替移动双臂
    实际使用时，这里的输入来自王宪雨的重定向算法
    """
    import numpy as np
    # 生成28维度的周期性关节角度
    t = time.time()
    action = np.zeros(28)
    # 简单的正弦波运动
    action[0] = 0.3 * np.sin(t * 2.0)       # 左肩pitch
    action[1] = 0.2 * np.sin(t * 1.5 + 1.0) # 左肩roll
    action[4] = 0.5 * np.sin(t * 1.8)        # 左肘
    action[7] = 0.3 * np.sin(t * 2.0 + 2.0)  # 右肩pitch
    action[8] = 0.2 * np.sin(t * 1.5 + 3.0)  # 右肩roll
    action[11] = 0.5 * np.sin(t * 1.8 + 1.0) # 右肘
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
    parser.add_argument('--demo', action='store_true',
                        help='演示模式')
    parser.add_argument('--benchmark', action='store_true',
                        help='基准测试模式')
    parser.add_argument('--trials', type=int, default=5,
                        help='基准测试每个任务的试验次数')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='数据存储目录')
    args = parser.parse_args()

    # 交互模式
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

    # 创建仿真环境
    env = SimulationEnv(
        robot_type=args.robot,
        task_name=args.task or 'pushcube',
        render=not args.no_render,
        record=not args.no_record,
        data_dir=args.data_dir,
    )

    try:
        if args.task:
            print(f"\n启动任务: {args.task}")
            print(f"机器人: {args.robot}")
            print(f"渲染: {'开' if not args.no_render else '关'}")
            print(f"数据记录: {'开' if not args.no_record else '关'}")

            # 运行任务
            if args.demo:
                print("\n[演示模式] 使用简单正弦波控制器...")
                env.evaluate_task(num_trials=3, controller=demo_controller)
            else:
                env.run_episode(controller=demo_controller)
        elif args.benchmark:
            print(f"\n[基准测试模式] 测试 Level 1 任务...")
            level1_tasks = get_tasks_by_level(1)
            env.benchmark_all_tasks(
                task_list=level1_tasks,
                controller=demo_controller,
                num_trials=args.trials,
            )

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        env.close()
        print("\n仿真环境已关闭。")


if __name__ == '__main__':
    main()