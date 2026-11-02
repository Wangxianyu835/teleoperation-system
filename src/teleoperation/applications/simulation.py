"""Application-owned episode loops and simulation/data recording composition."""
import time
from teleoperation.simulation.environment import SimulationEnv
from teleoperation.simulation.sensors import SimulationSensors
from teleoperation.simulation.tasks import list_tasks
from teleoperation.data.recorder import SensorRecorder


class EpisodeRecording:
    def __init__(self, data_dir):
        self.recorder = SensorRecorder(data_dir)
        self.sensors = SimulationSensors()

    @property
    def completion_time(self): return self.recorder.completion_time
    def reset(self): self.recorder.reset()
    def capture(self, robot_id, objects, client, elapsed):
        self.recorder.record_step(self.sensors.collect(robot_id, objects, client, elapsed))
    def mark_success(self, elapsed): self.recorder.mark_success(elapsed)
    def save_episode(self, episode): return self.recorder.save_episode(episode)


def create_environment(*args, record=True, data_dir="./data/", **kwargs):
    # Keep recorder creation/reset timing even for non-recording legacy applications.
    recording = EpisodeRecording(data_dir)
    return SimulationEnv(*args, record=record, data_dir=data_dir, recording=recording, **kwargs)


def run_episode(env, controller=None, max_steps: int = 10000,
                randomize: bool = True, verbose: bool = True):
    """
    运行一个完整的episode
    Args:
        controller: 控制器函数，输入obs，输出action
        max_steps: 最大步数
        randomize: 是否随机化场景
        verbose: 是否打印信息
    Returns:
        success, elapsed_time
    """
    env.reset(randomize=randomize)

    for step_i in range(max_steps):
        # 获取控制指令
        if controller is not None:
            action = controller(env._get_obs())
        else:
            action = None

        obs, success, done, elapsed = env.step(action)

        if done:
            if verbose:
                status = "[OK] SUCCESS" if success else "[FAIL] FAILED (timeout)"
                print(f"[{env.task_name}] Episode {env.episode_count}: "
                      f"{status} | Time: {elapsed:.2f}s | Steps: {step_i}")
            return success, elapsed

        # 渲染步进（GUI 模式下放慢速度以便观察）
        # 注意 旧代码是 `if env.client == p.GUI:`，但 client 是【连接 id】(0)，
        #    p.GUI 是【连接类型常量】(1)，两者永不相等 -> 已改用 env._render 标志
        if env._render:
            time.sleep(1.0 / 240.0)

    # 注意 步数用尽也必须**记一次试验**。旧代码直接 `return False, task.max_time`：
    #    timeout 判据比的是墙钟（task.max_time=300 s），而 max_steps=10000 在 240 Hz 下
    #    只跑 ~42 s 仿真就退出了，于是 done 永远为 False、MetricsTracker 一行都不记，
    #    `sim run --benchmark` 打印的是只有表头的空成绩单（2026-11-02 实测确认）。
    #    同时 elapsed 用真实的墙钟读数，不要伪造 task.max_time。
    elapsed = time.time() - env.task_start_time
    env._finalize_episode(False, elapsed)
    if verbose:
        # 注意 这里不能打 "[FAIL] ..."：队友的 tests/test_application_entrypoints.py 会把
        #    stdout 里的 "[FAIL]" 当成评估失败信号（它故意用 max_steps=5 跑有界试验）。
        print(f"[{env.task_name}] Episode {env.episode_count}: step budget exhausted "
              f"| Time: {elapsed:.2f}s | Steps: {max_steps}")
    return False, elapsed


def evaluate_task(env, num_trials: int = 5, controller=None):
    """对一个任务进行评估，跑多次"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {env.task_name} ({num_trials} trials)")
    print(f"Robot: {env.robot_type}")
    print(f"{'='*60}")

    for i in range(num_trials):
        success, elapsed = run_episode(env, controller=controller,
                                            verbose=True)

    env.metrics.print_summary()


def benchmark_all_tasks(env, task_list: list = None,
                        controller=None, num_trials: int = 3):
    """对任务列表进行基准测试"""
    if task_list is None:
        task_list = list_tasks()

    print(f"\n{'#'*60}")
    print(f"# TeleOpBench Benchmark: {len(task_list)} teleoperation.simulation.tasks x {num_trials} trials")
    print(f"{'#'*60}")

    for task_name in task_list:
        env.task_name = task_name
        for i in range(num_trials):
            run_episode(env, controller=controller, verbose=(i == 0))

    env.metrics.print_summary()

