"""
仿真环境 — 第三部分: 接收重定向数据, 驱动 H1-2 在 PyBullet 中执行

完整流水线:
  王健杰(摄像头) -> 王宪雨(重定向) -> 你(仿真环境显示机器人)

你负责的部分: 读取 H1DemoJointCommand -> 应用到 H1-2 -> 记录任务结果
当前用 H1DemoJointCommand.mock_demo() 模拟重定向数据, 等同学完成后替换为真实数据
"""

import sys, os, time
import pybullet as p, pybullet_data, numpy as np
from teleoperation.simulation.robot_loader import RobotLoader
from teleoperation.simulation.tasks.registry import get_task
from teleoperation.applications.demo.types import H1DemoJointCommand
from teleoperation.evaluation.metrics import MetricsTracker
from teleoperation.simulation.tasks import register_builtin_tasks
register_builtin_tasks()


# ═══════════════════════════════════════════════
# 你只需关心这个函数: 接收关节指令, 应用到机器人
# ═══════════════════════════════════════════════
def apply_joints_to_robot(robot_id, joint_cmd: H1DemoJointCommand,
                           arm_joints: dict, hand_joints: dict):
    """将 H1DemoJointCommand 应用到 H1-2 (你的核心工作)

    肖奕阳把重定向算法输出的数据封装成 H1DemoJointCommand,
    你在这里把它们逐一设到 H1-2 的对应关节上
    """
    for joint, target in map_demo_arm_targets(joint_cmd.left_arm, joint_cmd.right_arm, arm_joints):
        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target, force=200)


# ═══════════════════════════════════════════════
# 主程序: 加载场景 -> 循环接收数据 -> 驱动机器人
# ═══════════════════════════════════════════════
def main(args=None):
    print("=" * 55)
    print("  TeleOpBench 仿真环境 (第三部分)")
    print("  H1-2 机器人 + pushcube 任务")
    print("  等待接收重定向数据...")
    print("=" * 55)

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")

    # 加载 H1-2 机器人
    loader = RobotLoader(client)
    robot = loader.load_robot()
    robot_id = robot['robot']

    # 加载 pushcube 任务
    TaskClass = get_task('pushcube')
    task = TaskClass(client)
    task.set_robot(robot_id)
    task.reset()

    p.resetDebugVisualizerCamera(3.5, 50, -25, [0, 0, 1.1])

    # 指标追踪
    metrics = MetricsTracker()
    task_start = time.time()

    print("\n机器人已加载, H1-2 双臂摆动中...")
    print("鼠标: 左键旋转 | 右键平移 | 滚轮缩放 | Ctrl+C 退出\n")

    sim = 0
    try:
        while True:
            sim += 1
            t = sim / 240.0

            # ═══════ 接收重定向数据 ═══════
            # TODO: 替换为肖奕阳的真实重定向输出
            # 当前用 mock_demo() 模拟正弦波摆动
            cmd = H1DemoJointCommand.mock_demo(t)

            # ═══════ 应用到机器人 ═══════
            apply_joints_to_robot(robot_id, cmd,
                                   loader.all_joints, {})

            p.stepSimulation()

            # 任务检查
            if sim % 120 == 0:
                success = task.check_success()
                if success:
                    elapsed = time.time() - task_start
                    print(f"  [OK] TASK COMPLETED! ({elapsed:.1f}s)")
                    metrics.record('pushcube', True, elapsed)
                    break

            if sim % 1200 == 0:
                print(f"  [{sim//240:3d}s] 等待重定向数据...")

            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        try:
            metrics.print_summary()
            p.disconnect()
        except Exception:
            pass
        print("仿真环境已关闭")


from teleoperation.robots.demo_h1 import map_demo_arm_targets
