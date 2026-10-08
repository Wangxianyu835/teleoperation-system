"""仿真环境核心类 - 集成机器人、任务、记录、随机化"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from .robot_loader import RobotLoader
from .randomization import DomainRandomizer
from teleoperation.simulation.tasks import get_task, list_tasks
from teleoperation.simulation.tasks import register_builtin_tasks
register_builtin_tasks()          # noqa: F401  ★ 触发 30 个任务注册
from teleoperation.evaluation import MetricsTracker


class SimulationEnv:
    """双臂灵巧遥操作仿真环境"""

    def __init__(self, robot_type: str = 'h1_2',
                 task_name: str = 'pushcube',
                 render: bool = True,
                 record: bool = True,
                 data_dir: str = './data/', recording=None):
        """
        Args:
            robot_type: 'h1_2', 'gr1_t2', 'g1'
            task_name: 30个任务名之一
            render: 是否GUI渲染
            record: 是否记录数据
            data_dir: 数据存储路径
        """
        self.robot_type = robot_type
        self.task_name = task_name
        self.record = record
        if record and recording is None:
            raise ValueError("Recording requires an application-provided recording observer")
        self.recording = recording

        # 连接 PyBullet
        self._render = bool(render)
        if render:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        # 基本设置
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # 初始化和加载
        self._init_modules(data_dir)
        self.episode_count = 0

        # 机器人 / 动作空间占位
        # 注意 真正的值在 reset() 里「按机器人确定」后填充。
        #    这里先占位，是为了在 reset() 之前访问时能得到明确信息，
        #    而不是抛出难懂的 AttributeError: 'SimulationEnv' object has no
        #    attribute 'action_dim'
        self.robot_id = None
        self.action_dim = None
        self.action_joint_names = []
        self.action_joint_indices = []

    def _init_modules(self, data_dir: str):
        """初始化各模块"""
        self.robot_loader = RobotLoader(self.client)
        self.randomizer = DomainRandomizer()
        self.metrics = MetricsTracker()

    def reset(self, randomize: bool = True):
        """重置环境"""
        # 清除场景
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # 加载地面
        self._load_ground()

        # 加载机器人（load_robot 返回 dict，取 'robot' 作为 body id）
        robot_info = self.robot_loader.load_robot(self.robot_type)
        self.robot_id = (robot_info['robot']
                         if isinstance(robot_info, dict) else robot_info)

        # ★ 动作空间（由机器人决定，不是固定 28 维）
        self.action_joint_names = self.robot_loader.action_joint_names
        self.action_joint_indices = self.robot_loader.action_joint_indices
        self.action_dim = len(self.action_joint_indices)
        if not getattr(self, '_action_space_logged', False):
            print(f"  [SimulationEnv] 动作空间维度 = {self.action_dim} "
                  f"(左臂7 + 右臂7 + 左手{len(self.robot_loader.hand_joints['left'])} "
                  f"+ 右手{len(self.robot_loader.hand_joints['right'])})")
            self._action_space_logged = True

        # 创建任务
        TaskClass = get_task(self.task_name)
        self.task = TaskClass(self.client)
        self.task.set_robot(self.robot_id)
        self.task.reset()

        # 域随机化
        if randomize:
            self.randomizer.randomize(self.task.objects, self.client)

        # 重置记录器
        if self.recording is not None: self.recording.reset()
        self.start_time = time.time()
        self.task_start_time = time.time()

        self.episode_count += 1

    def _load_ground(self):
        """加载地面"""
        # RobotLoader changes the search path; subsequent episode resets must
        # still resolve the same ground asset without depending on that path.
        p.loadURDF(f"{pybullet_data.getDataPath()}/plane.urdf", physicsClientId=self.client)

    def step(self, action: np.ndarray = None, duration: float = 1.0 / 240.0):
        """
        执行一步仿真
        Args:
            action: 按当前机器人 action_joint_names 排列的完整原生动作
            duration: 步长时间
        Returns:
            obs, success, done, elapsed
        """
        # 应用动作（★ 必须用 action_joint_indices 映射，绝不能按位置映射到关节索引）
        if action is not None:
            validate_native_action(action, self.action_dim)
            self.task.apply_action(
                action,
                joint_indices=getattr(self, 'action_joint_indices', None),
            )

        # 物理步进
        p.stepSimulation(physicsClientId=self.client)

        # 检查成功
        success = self.task.check_success()
        elapsed = time.time() - self.task_start_time

        # 记录数据
        if self.record:
            self.recording.capture(self.robot_id, self.task.objects, self.client, elapsed)

        # 判断结束
        timeout = elapsed > self.task.max_time
        done = success or timeout

        if success and self.recording is not None and self.recording.completion_time == 0.0:
            self.recording.mark_success(elapsed)

        if done:
            self._finalize_episode(success, elapsed)

        return self._get_obs(), success, done, elapsed

    def _get_obs(self) -> dict:
        """获取观测"""
        obs = {
            'task_name': self.task_name,
            'robot_type': self.robot_type,
        }
        if self.task is not None:
            for name, obj_id in self.task.objects.items():
                pos, orn = p.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self.client)
                obs[f'{name}_pos'] = pos
                obs[f'{name}_orn'] = orn
        return obs

    def _finalize_episode(self, success: bool, elapsed: float):
        """结束episode，保存数据"""
        self.metrics.record(self.task_name, success, elapsed)

        if self.record:
            self.recording.save_episode(self.episode_count)


    def close(self):
        """关闭仿真"""
        p.disconnect(physicsClientId=self.client)

from teleoperation.robots.action_mapping import validate_native_action
