"""快速测试 - 验证所有模块能正确导入"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试1: 导入基础模块...")
try:
    from utils import MetricsTracker
    print("  ✓ utils.MetricsTracker")
except Exception as e:
    print(f"  ✗ {e}")

print("\n测试2: 导入任务模块...")
try:
    from tasks.all_tasks import *
    from tasks import list_tasks, get_task, get_tasks_by_level
    tasks = list_tasks()
    print(f"  ✓ 已注册 {len(tasks)} 个任务")
    for level in range(1, 5):
        level_tasks = get_tasks_by_level(level)
        print(f"    Level {level}: {level_tasks}")
except Exception as e:
    print(f"  ✗ {e}")

print("\n测试3: 导入环境模块...")
try:
    from envs import RobotLoader, SensorRecorder, DomainRandomizer, SimulationEnv
    print("  ✓ envs.RobotLoader")
    print("  ✓ envs.SensorRecorder")
    print("  ✓ envs.DomainRandomizer")
    print("  ✓ envs.SimulationEnv")
except Exception as e:
    print(f"  ✗ {e}")

print("\n测试4: 创建PyBullet实例（无头模式）...")
try:
    import pybullet as p
    client = p.connect(p.DIRECT)
    assert client >= 0
    p.disconnect(client)
    print("  ✓ PyBullet 连接正常")
except Exception as e:
    print(f"  ✗ {e}")

print("\n测试5: 创建仿真环境实例...")
try:
    # 只测试导入和初始化逻辑，不实际渲染
    from tasks.all_tasks import *  # noqa
    task_cls = get_task('pushcube')
    print(f"  ✓ 获取任务类: pushcube -> {task_cls.__name__}")
    print(f"    描述: {task_cls.__doc__}")
except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 50)
print("所有测试完成！")