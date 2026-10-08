"""遥操作完整流水线 - VR + 手套 -> 机器人 -> 任务执行

论文 Section 3.2: 统一模块化接口
  Vision Pro -> 坐标转换 -> IK -> 双臂关节
  LinkerHand -> 重定向 -> 灵巧手关节
  -> 仿真环境 -> 任务执行 -> 数据记录
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

# 导入项目模块
from teleoperation.simulation.robot_loader import RobotLoader
from teleoperation.data.recorder import SensorRecorder
from teleoperation.simulation.sensors import SimulationSensors
from teleoperation.simulation.randomization import DomainRandomizer
from teleoperation.evaluation.metrics import MetricsTracker

from teleoperation.apps.demo.vr import VRInterface
from teleoperation.apps.demo.hand import HandInterface

from teleoperation.simulation.tasks.registry import get_task, list_tasks, get_tasks_by_level
from teleoperation.simulation.tasks import register_builtin_tasks
register_builtin_tasks()  # noqa

# ============================================================
# TRON2 机器人配置 (逐际动力)
# 等 URDF 到手后替换 robot_loader.py 里的路径
# ============================================================
# xArm7 关节索引 (int) — IK 返回 tuple，需用整数索引
# joint1=0, joint2=1, ..., joint7=6
TRON2_CONFIG = {
    'left_arm_indices': [0, 1, 2, 3, 4, 5, 6],
    'right_arm_indices': [0, 1, 2, 3, 4, 5, 6],
    'left_ee_idx': 7,   # 末端执行器索引
    'right_ee_idx': 7,
    'pelvis_pos': np.array([0.0, 0.0, 0.9]),
}


class TeleopPipeline:
    """
    VR + 手套 遥操作完整流水线

    数据流:
      Vision Pro ──-> VRInterface ──-> IK ──-> 双臂关节
      LinkerHand ──-> HandInterface ──-> 灵巧手关节
                          ↓
      [仿真循环] <-── 所有关节角度
                          ↓
      任务检查 + 数据记录
    """

    def __init__(self, task_name: str = 'pushcube',
                 robot_type: str = 'xarm7_ability',
                 hand_mode: str = 'simulator',
                 render: bool = True,
                 record: bool = True,
                 data_dir: str = './data/'):
        """
        Args:
            task_name: 30个任务名之一
            robot_type: 机器人类型
            hand_mode: 'simulator' | 'glove' | 'vr'
            render: 是否 GUI 渲染
            record: 是否记录数据
        """
        self.task_name = task_name
        self.trial_num = 0

        # ========== 1. 连接仿真 ==========
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # ========== 2. 加载模块 ==========
        self.robot_loader = RobotLoader(self.client)
        self.recorder = SensorRecorder(data_dir) if record else None
        self.sensors = SimulationSensors()
        self.randomizer = DomainRandomizer()
        self.metrics = MetricsTracker()

        # 遥操作接口
        self.vr = VRInterface(self.client)
        self.hand = HandInterface(mode=hand_mode)

        # 机器人状态
        self.robot_ids = None
        self.task = None

    def setup_scene(self):
        """初始化场景：地面 + 机器人 + 任务"""
        # 地面
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # 加载机器人
        print(f"\n[Pipeline] 加载机器人...")
        self.robot_ids = self.robot_loader.load_robot()

        # 创建任务
        print(f"[Pipeline] 加载任务: {self.task_name}")
        TaskClass = get_task(self.task_name)
        self.task = TaskClass(self.client)
        self.task.set_robot(self.robot_ids['left'])
        self.task.reset()

        # 相机
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=50, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.8],
            physicsClientId=self.client
        )

    def run_episode(self, max_steps: int = 5000, verbose: bool = True) -> dict:
        """
        运行一个完整的遥操作 episode

        Returns:
            dict: {'success', 'elapsed', 'steps'}
        """
        self.trial_num += 1
        # 只在第一次 setup，后续 episode 只 reset 任务
        if self.robot_ids is None:
            self.setup_scene()
        else:
            self.task.reset()

        robot_id = self.robot_ids['left']
        left_indices = TRON2_CONFIG['left_arm_indices']
        right_indices = TRON2_CONFIG['right_arm_indices']
        pelvis = TRON2_CONFIG['pelvis_pos']

        start_time = time.time()

        if verbose:
            print(f"\n{'='*50}")
            print(f"  Teleop Episode {self.trial_num}: {self.task_name}")
            print(f"  {'='*25}")
            print(f"  遥操作模式: Vision Pro (模拟) + 手套 (模拟)")
            print(f"  机器人: TRON2 (当前: xArm7 Ability)")

        # ========== 主循环 ==========
        for step_i in range(max_steps):
            # --- 1. VR 遥操作 -> 手臂关节 ---
            vr_result = self.vr.process_frame(
                robot_id=robot_id,
                left_arm_indices=left_indices,
                right_arm_indices=right_indices,
                left_ee_idx=TRON2_CONFIG['left_ee_idx'],
                right_ee_idx=TRON2_CONFIG['right_ee_idx'],
                pelvis_pos=pelvis,
            )

            # --- 2. 手套 -> 手指关节 ---
            hand_joints = self.hand.get_both_hands()

            # --- 3. 应用手臂关节 ---
            for j_idx, angle in zip(left_indices, vr_result['left_arm']):
                p.setJointMotorControl2(
                    robot_id, j_idx, p.POSITION_CONTROL,
                    targetPosition=angle, force=200,
                    physicsClientId=self.client
                )

            # --- 4. 应用手指关节 ---
            if 'right' in hand_joints:
                self.hand.apply_to_robot(
                    self.robot_ids['right'], hand_joints['right']
                )

            # --- 5. 物理步进 ---
            p.stepSimulation(physicsClientId=self.client)

            # --- 6. 检查成功 ---
            success = self.task.check_success()
            elapsed = time.time() - start_time
            timeout = elapsed > self.task.max_time
            done = success or timeout

            # --- 7. 记录数据 ---
            if self.recorder:
                self.recorder.record_step(self.sensors.collect(
                    robot_id, self.task.objects, self.client, elapsed))

            # --- 8. 结束判断 ---
            if done:
                self.metrics.record(self.task_name, success, elapsed)
                if self.recorder:
                    if success:
                        self.recorder.mark_success(elapsed)
                    self.recorder.save_episode(self.trial_num)

                if verbose:
                    status = "[OK] SUCCESS" if success else "[FAIL] TIMEOUT"
                    print(f"  [{self.task_name}] Episode {self.trial_num}: "
                          f"{status} | {elapsed:.1f}s | {step_i} 步")
                return {'success': success, 'elapsed': elapsed, 'steps': step_i}

            # 延迟以匹配实时 (240 Hz)
            time.sleep(1.0 / 240.0)

        return {'success': False, 'elapsed': time.time() - start_time,
                'steps': max_steps}

    def evaluate_task(self, task_name: str = None, num_trials: int = 5,
                      verbose: bool = True):
        """对单个任务进行评估，跑多次"""
        if task_name:
            self.task_name = task_name

        print(f"\n{'#'*50}")
        print(f"# TeleOpBench Evaluation")
        print(f"# Task: {self.task_name}")
        print(f"# Trials: {num_trials}")
        print(f"{'#'*50}")

        for i in range(num_trials):
            self.run_episode(verbose=verbose)

        self.metrics.print_summary()

    def benchmark_level(self, level: int = 1, num_trials: int = 3):
        """对指定难度等级的全部任务进行基准测试"""
        selected_tasks = get_tasks_by_level(level)
        print(f"\nLevel {level} 任务 ({len(selected_tasks)}个): {selected_tasks}")

        for task_name in selected_tasks:
            self.task_name = task_name
            for i in range(num_trials):
                self.run_episode(verbose=(i == 0))

        self.metrics.print_summary()

    def close(self):
        """关闭所有连接"""
        self.hand.close()
        p.disconnect(physicsClientId=self.client)
        print("[Pipeline] 仿真环境已关闭")
