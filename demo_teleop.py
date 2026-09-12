"""
仿真环境 — 第三部分: 接收重定向数据, 驱动 H1-2 在 PyBullet 中执行

完整流水线:
  王健杰(摄像头) → 王宪雨(重定向) → 你(仿真环境显示机器人)

你负责的部分: 读取 RobotJointCommand → 应用到 H1-2 → 记录任务结果
当前用 RobotJointCommand.mock_demo() 模拟重定向数据, 等同学完成后替换为真实数据
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pybullet as p, pybullet_data, numpy as np
from envs.robot_loader import RobotLoader
from tasks.task_registry import get_task
from teleop.pipeline_data import RobotJointCommand
from utils.metrics import MetricsTracker
import tasks.all_tasks as _


# ═══════════════════════════════════════════════
# 你只需关心这个函数: 接收关节指令, 应用到机器人
# ═══════════════════════════════════════════════
def apply_joints_to_robot(robot_id, joint_cmd: RobotJointCommand,
                           arm_joints: dict, hand_joints: dict):
    """将 RobotJointCommand 应用到 H1-2 (你的核心工作)

    肖奕阳把重定向算法输出的数据封装成 RobotJointCommand,
    你在这里把它们逐一设到 H1-2 的对应关节上
    """
    client_id = 0  # PyBullet 默认 client

    # 左臂 7 DoF
    left_names = [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_pitch_joint',
        'left_elbow_roll_joint', 'left_wrist_pitch_joint',
        'left_wrist_yaw_joint',
    ]
    for i, name in enumerate(left_names):
        if name in arm_joints and i < len(joint_cmd.left_arm):
            p.setJointMotorControl2(
                robot_id, arm_joints[name],
                p.POSITION_CONTROL,
                targetPosition=float(joint_cmd.left_arm[i]),
                force=200,
            )

    # 右臂 7 DoF
    right_names = [
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_pitch_joint',
        'right_elbow_roll_joint', 'right_wrist_pitch_joint',
        'right_wrist_yaw_joint',
    ]
    for i, name in enumerate(right_names):
        if name in arm_joints and i < len(joint_cmd.right_arm):
            p.setJointMotorControl2(
                robot_id, arm_joints[name],
                p.POSITION_CONTROL,
                targetPosition=float(joint_cmd.right_arm[i]),
                force=200,
            )


# ═══════════════════════════════════════════════
# 主程序: 加载场景 → 循环接收数据 → 驱动机器人
# ═══════════════════════════════════════════════
def main():
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
            cmd = RobotJointCommand.mock_demo(t)

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


if __name__ == '__main__':
    main()