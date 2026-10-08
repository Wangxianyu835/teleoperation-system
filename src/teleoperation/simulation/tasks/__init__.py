"""Explicit and idempotent registration of the thirty built-in tasks."""
from .registry import get_task, list_tasks, get_tasks_by_level, register_task
from .base import BaseTask


def register_builtin_tasks():
    from . import basic, tools, bimanual, sequences
