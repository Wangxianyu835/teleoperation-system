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
        records = self.results[task_name]
        if not records:
            return 0.0
        times = [t for s, t in records if s]  # 只统计成功完成的
        if not times:
            return float('inf')
        return np.mean(times)

    def get_summary(self) -> dict:
        summary = {}
        for task_name in self.results:
            summary[task_name] = {
                'success_rate': self.get_success_rate(task_name),
                'avg_completion_time': self.get_avg_time(task_name),
                'num_trials': len(self.results[task_name]),
            }
        return summary

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"{'Task':<25} {'Success%':>10} {'AvgTime(s)':>12} {'Trials':>8}")
        print("-" * 70)
        for task_name, stats in self.get_summary().items():
            print(f"{task_name:<25} {stats['success_rate']*100:>9.1f}% "
                  f"{stats['avg_completion_time']:>11.2f} {stats['num_trials']:>8}")
        print("=" * 70)
