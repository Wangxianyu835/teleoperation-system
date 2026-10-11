# 函数 register_builtin_tasks（注册/分发）；链路：仿真环境→任务注册/执行→评估记录
"""Explicit and idempotent registration of the thirty built-in tasks."""
from .registry import get_task, list_tasks, get_tasks_by_level, register_task
from .base import BaseTask


def register_builtin_tasks():
    from . import basic, tools, bimanual, sequences
