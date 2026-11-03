"""仿真环境核心类 - 集成机器人、任务、记录、随机化"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from .robot_loader import RobotLoader
from .randomization import DomainRandomizer
from .urdf_loader import create_ground
from teleoperation.simulation.tasks import get_task, list_tasks
from teleoperation.simulation.tasks import register_builtin_tasks
register_builtin_tasks()          # noqa: F401  ★ 触发 30 个任务注册
from teleoperation.evaluation import MetricsTracker


DEFAULT_DT = 1.0 / 240.0     # 物理步长（reset() 里也按这个 setTimeStep）


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

        # 仿真时钟与「连续成立窗口」的占位值：同上面的理由，reset() 之前访问
        # 也要有确定值（例如步数用尽时 _finalize_episode() 会读 self.sim_time）。
        self.dt = DEFAULT_DT
        self.sim_time = 0.0
        self._success_streak = 0
        self._success_hold_steps = 1

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

        # 域随机化（关节体不参与：门/抽屉的位姿由关节决定，硬搬会撕开铰链）
        if randomize:
            self.randomizer.randomize(self.task.objects, self.client,
                                      skip_names=self.task.jointed_objects)
            # 随机化会带来 ±1 cm 的 z 抖动，可能又把物体压进桌面，
            # 所以再吸附一次，并把结果记为「初始位姿」（相对判据的基准）
            self.task.snap_to_supports()
            self.task.capture_initial_state()

        # 重置记录器
        if self.recording is not None: self.recording.reset()

        # 时间口径：sim_time 是仿真时间（超时判据 + 报表主口径，与机器快慢无关）；
        # task_start_time 只用来算墙钟 elapsed（接口契约里的返回值，看实时因子）。
        self.dt = DEFAULT_DT
        self.sim_time = 0.0
        self.start_time = time.time()
        self.task_start_time = time.time()

        # 判据需连续成立的步数（仿真时间换算）：0.1 s @ 240 Hz = 24 步
        self._success_streak = 0
        hold = getattr(self.task, 'hold_steps', None)
        self._success_hold_steps = max(1, int(hold(self.dt))) if callable(hold) else 1

        self.episode_count += 1

    def _load_ground(self):
        """加载地面"""
        create_ground(p, physicsClientId=self.client)

    def step(self, action: np.ndarray = None, duration: float = 1.0 / 240.0):
        """
        执行一步仿真
        Args:
            action: 按当前机器人 action_joint_names 排列的完整原生动作
            duration: 保留参数（本实现每次只推进 1 个物理步，不按 duration 细分）
        Returns:
            obs, success, done, elapsed
            elapsed 是**墙钟**秒数（接口契约不变，用来算实时因子）；
            仿真时间见 env.sim_time，它才是超时判据与报表的口径。
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
        self.sim_time += self.dt

        # 成功判定：判据必须**连续成立** N 步（默认 0.1 s 仿真时间）才算成功。
        # 只判一步会收下瞬时状态 —— 例如物体被弹起来从目标区上空飞过、
        # 抽屉被撞得短暂到位又弹回，这些都不是「做成了」。
        if self.task.check_success():
            self._success_streak += 1
        else:
            self._success_streak = 0
        success = self._success_streak >= self._success_hold_steps

        elapsed = time.time() - self.task_start_time        # 墙钟（参考用）

        # 记录数据
        if self.record:
            self.recording.capture(self.robot_id, self.task.objects, self.client, elapsed)

        # 判断结束：超时按**仿真时间**比（旧版比墙钟，慢机器上会提前超时，
        # 同一策略在快慢不同的机器上成绩不同）
        timeout = self.sim_time > self.task.max_time
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
            for name in self.task.objects:
                pos, orn = self.task.get_object_pose(name)
                obs[f'{name}_pos'] = pos
                obs[f'{name}_orn'] = orn
        return obs

    def _finalize_episode(self, success: bool, elapsed: float = None,
                          wall_elapsed: float = None):
        """结束 episode：记一次试验（仿真时间与墙钟都记）

        Args:
            success: 是否成功
            elapsed: 兼容旧调用点的墙钟读数（applications/simulation.py 在步数用尽时
                     传进来的是 time.time() 之差）。**完成时间一律取 self.sim_time**：
                     仿真时间不随机器快慢变，成绩才可比。
            wall_elapsed: 显式墙钟（优先于 elapsed），只用于报表的 WallT 列
        """
        if wall_elapsed is None and elapsed is not None:
            wall_elapsed = float(elapsed)
        self.metrics.record(self.task_name, success, self.sim_time,
                            wall_time=wall_elapsed)

        if self.record:
            self.recording.save_episode(self.episode_count)


    def close(self):
        """关闭仿真"""
        p.disconnect(physicsClientId=self.client)

from teleoperation.robots.action_mapping import validate_native_action
