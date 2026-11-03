"""任务评估指标：成功率、完成时间统计（仿真时间 + 墙钟）

每个 trial 记两个时间：
  sim_time   仿真时间（秒）—— 与机器快慢无关，跨机器 / 跨版本可比，报表主口径
  wall_time  真实耗时（秒）—— 只反映实时因子（real-time factor），不要拿来比机器

旧版只有墙钟：同一个策略在慢机器上会更早撞 timeout（`max_time` 按墙钟比），
成绩单因此不可比（2026-11-02 修，见 docs/SIM_EVALUATION.md）。
没有成功 trial 时平均完成时间无定义，返回 nan（旧版返回 inf，报表里印成 `inf`）。
"""

import numpy as np
from collections import defaultdict


class MetricsTracker:
    """跟踪每个任务的性能指标

    results[task_name] = [(success, sim_time, wall_time), ...]
    wall_time 允许为 None（只有仿真时间的调用方）。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.results = defaultdict(list)  # {task_name: [(success, sim, wall), ...]}

    def record(self, task_name: str, success: bool,
               completion_time: float, wall_time: float = None):
        """记一次试验

        Args:
            task_name: 任务名
            success: 是否成功
            completion_time: 完成耗时（秒）—— 约定传**仿真时间**
            wall_time: 真实耗时（秒），可选；不传就不统计墙钟
        """
        self.results[task_name].append(
            (bool(success), float(completion_time),
             None if wall_time is None else float(wall_time)))

    def get_success_rate(self, task_name: str) -> float:
        records = self.results[task_name]
        if not records:
            return 0.0
        return sum(1 for success, _, _ in records if success) / len(records)

    def get_avg_time(self, task_name: str) -> float:
        """成功 trial 的平均仿真时间（秒）；没有任何成功 trial 时返回 nan。"""
        records = self.results[task_name]
        if not records:
            return 0.0
        times = [sim for success, sim, _ in records if success]
        if not times:
            return float('nan')
        return float(np.mean(times))

    def get_avg_episode_time(self, task_name: str) -> float:
        """全部 trial（成功 + 失败）的平均仿真时间（秒）。"""
        records = self.results[task_name]
        if not records:
            return 0.0
        return float(np.mean([sim for _, sim, _ in records]))

    def get_avg_wall_time(self, task_name: str) -> float:
        """全部 trial 的平均墙钟耗时（秒）；没有任何墙钟记录时返回 nan。"""
        records = self.results[task_name]
        walls = [wall for _, _, wall in records if wall is not None]
        if not walls:
            return float('nan')
        return float(np.mean(walls))

    def get_summary(self) -> dict:
        summary = {}
        for task_name in self.results:
            summary[task_name] = {
                'success_rate': self.get_success_rate(task_name),
                'avg_completion_time': self.get_avg_time(task_name),
                'avg_episode_time': self.get_avg_episode_time(task_name),
                'avg_wall_time': self.get_avg_wall_time(task_name),
                'num_trials': len(self.results[task_name]),
            }
        return summary

    @staticmethod
    def _cell(value, fmt: str = '{:.2f}') -> str:
        return '-' if np.isnan(value) else fmt.format(value)

    def print_summary(self):
        print("\n" + "=" * 78)
        print(f"{'Task':<25} {'Success%':>10} {'SimT(s)':>10} "
              f"{'WallT(s)':>10} {'Trials':>8}")
        print("-" * 78)
        for task_name, stats in self.get_summary().items():
            # SimT = 成功 trial 的平均仿真时间（没有成功就印 '-'）
            # WallT = 全部 trial 的平均墙钟耗时
            print(f"{task_name:<25} {stats['success_rate']*100:>9.1f}% "
                  f"{self._cell(stats['avg_completion_time']):>10} "
                  f"{self._cell(stats['avg_wall_time']):>10} "
                  f"{stats['num_trials']:>8}")
        print("=" * 78)
        print("SimT = 仿真时间（可比口径）；WallT = 真实耗时（看实时因子）")
