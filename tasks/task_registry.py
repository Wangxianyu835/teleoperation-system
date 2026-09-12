"""任务注册表 - 管理30个任务类"""

from typing import Type, List
from .base_task import BaseTask


# 先注册为空，等导入具体任务后填充
TASK_REGISTRY: dict = {}


def register_task(name: str):
    """装饰器：注册任务类"""
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str) -> Type[BaseTask]:
    """获取任务类"""
    if name not in TASK_REGISTRY:
        raise KeyError(f"Task '{name}' not found. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """列出所有已注册任务"""
    return sorted(TASK_REGISTRY.keys())


def get_tasks_by_level(level: int) -> List[str]:
    """按难度级别获取任务列表"""
    level_tasks = {
        1: ['pushcube', 'pickcube', 'pickplacecube', 'uprearcup', 'balltrashcan'],
        2: ['rotatefaucet', 'rotatehearth', 'openmicrowave', 'closemicrowave',
            'opendrawer', 'closedrawer', 'liftmug', 'openlaptop', 'ballmug',
            'pourwater', 'breadtoaster'],
        3: ['ballbimanual', 'potbimanual', 'pottomato', 'pottray',
            'stackboxes', 'panhearth', 'tidyuptable', 'pottomatoout', 'plateoven'],
        4: ['pottomatoplate', 'penbrushpot', 'drawerbook', 'twistbottlecaps',
            'stacktoyblocks'],
    }
    return level_tasks.get(level, [])