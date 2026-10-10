def main(args=None):
    """快速测试 - 验证所有模块能正确导入"""
    import sys
    import os


    print("测试1: 导入基础模块...")
    try:
        from teleoperation.evaluation import MetricsTracker
        print("  [OK] utils.MetricsTracker")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n测试2: 导入任务模块...")
    try:
        from teleoperation.simulation.tasks import register_builtin_tasks
        register_builtin_tasks()
        from teleoperation.simulation.tasks import list_tasks, get_task, get_tasks_by_level
        registered_tasks = list_tasks()
        print(f"  [OK] 已注册 {len(registered_tasks)} 个任务")
        for level in range(1, 5):
            level_tasks = get_tasks_by_level(level)
            print(f"    Level {level}: {level_tasks}")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n测试3: 导入环境模块（★ 必须检查「可调用」，None 也算失败）...")
    try:
        from teleoperation.simulation import RobotLoader, SensorRecorder, DomainRandomizer, SimulationEnv
        for name, obj in [('envs.RobotLoader', RobotLoader),
                          ('envs.SensorRecorder', SensorRecorder),
                          ('envs.DomainRandomizer', DomainRandomizer),
                          ('envs.SimulationEnv', SimulationEnv)]:
            # 注意 只检查 import 会漏掉 None（导入异常被静默吞掉的情况）
            assert obj is not None, f"{name} 是 None —— 导入被静默吞掉了！"
            assert callable(obj), f"{name} 不可调用"
            print(f"  [OK] {name}")
    except Exception as e:
        import traceback
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n测试4: 创建PyBullet实例（无头模式）...")
    try:
        import pybullet as p
        client = p.connect(p.DIRECT)
        assert client >= 0
        p.disconnect(client)
        print("  [OK] PyBullet 连接正常")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n测试5: 创建仿真环境实例...")
    try:
        # 只测试导入和初始化逻辑，不实际渲染
        from teleoperation.simulation.tasks import register_builtin_tasks
        register_builtin_tasks()  # noqa
        task_cls = get_task('pushcube')
        print(f"  [OK] 获取任务类: pushcube -> {task_cls.__name__}")
        print(f"    描述: {task_cls.__doc__}")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n测试6: ★ 端到端实测（无头跑 300 步：H1-2 + pushcube）...")
    try:
        import numpy as np
        from teleoperation.simulation.tasks import register_builtin_tasks
        register_builtin_tasks()  # noqa
        from teleoperation.applications.simulation import create_environment as SimulationEnv, run_episode, evaluate_task, benchmark_all_tasks
        env = SimulationEnv(robot_type='h1_2', task_name='pushcube',
                            render=False, record=False)
        try:
            env.reset(randomize=True)
            assert isinstance(env.robot_id, int), \
                f"robot_id 应为 int，实际 {type(env.robot_id).__name__}"
            assert env.action_dim > 0, "action_dim 必须 > 0"
            # ★ 回归断言：动作绝不能误触腿部关节
            leg_idx = {i for nm, i in env.robot_loader.all_joints.items()
                       if ('hip' in nm or 'knee' in nm or 'ankle' in nm)}
            overlap = leg_idx & set(env.action_joint_indices)
            assert not overlap, f"动作空间误触腿部关节索引：{sorted(overlap)}"
            action = np.zeros(env.action_dim)
            for _ in range(300):
                obs, success, done, elapsed = env.step(action)
                if done:
                    break
            print(f"  [OK] 端到端跑通（robot_id={env.robot_id}, "
                  f"action_dim={env.action_dim}）")
            print(f"    任务对象 = {list(env.task.objects.keys())}")
            print(f"    观测键   = {list(obs.keys())}")
        finally:
            env.close()
    except Exception as e:
        import traceback
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("所有测试完成！")
