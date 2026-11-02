"""任务评估指标：成功率、完成时间统计"""

import time
import numpy as np
from collections import defaultdict


class MetricsTracker:
    """跟踪每个任务的性能指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.results = defaultdict(list)  # {task_name: [(success, time), ...]}

    def record(self, task_name: str, success: bool, completion_time: float):
        self.results[task_name].append((success, completion_time))

    def get_success_rate(self, task_name: str) -> float:
        records = self.results[task_name]
        if not records:
            return 0.0
        successes = sum(1 for s, _ in records if s)
        return successes / len(records)

    def get_avg_time(self, task_name: str) -> float:
        """成功完成的平均耗时（秒）。没有任何成功 trial 时返回 nan。

        旧版返回 inf，报表里直接印成 `inf`，看起来像坏了；而且它掩盖了
        「0% 成功」这个真正的信息 —— 失败 trial 的耗时要用 get_avg_episode_time()。
        """
        records = self.results[task_name]
        if not records:
            return 0.0
        times = [t for s, t in records if s]  # 只统计成功完成的
        if not times:
            return float('nan')
        return float(np.mean(times))

    def get_avg_episode_time(self, task_name: str) -> float:
        """全部 trial（成功 + 失败）的平均耗时（秒）。"""
        records = self.results[task_name]
        if not records:
            return 0.0
        return float(np.mean([t for _, t in records]))

    def get_summary(self) -> dict:
        summary = {}
        for task_name in self.results:
            summary[task_name] = {
                'success_rate': self.get_success_rate(task_name),
                'avg_completion_time': self.get_avg_time(task_name),
                'avg_episode_time': self.get_avg_episode_time(task_name),
                'num_trials': len(self.results[task_name]),
            }
        return summary

    def print_summary(self):
        print("\n" + "=" * 82)
        print(f"{'Task':<25} {'Success%':>10} {'AvgTime(s)':>12} "
              f"{'EpTime(s)':>11} {'Trials':>8}")
        print("-" * 82)
        for task_name, stats in self.get_summary().items():
            # 没有成功 trial 时平均完成时间无定义，印 '-' 而不是 inf
            avg_time = stats['avg_completion_time']
            avg_text = '-' if np.isnan(avg_time) else f'{avg_time:.2f}'
            print(f"{task_name:<25} {stats['success_rate']*100:>9.1f}% "
                  f"{avg_text:>11} {stats['avg_episode_time']:>10.2f} "
                  f"{stats['num_trials']:>8}")
        print("=" * 82)
