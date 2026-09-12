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
        hint = ""
        if not TASK_REGISTRY:
            hint = ("\n提示：任务注册表为空 —— 30 个任务是靠 "
                    "`import tasks.all_tasks` 的副作用注册的。\n"
                    "      请在使用 get_task() 前先 import tasks.all_tasks；"
                    "或直接用 SimulationEnv（它已自动导入）。")
        raise KeyError(
            f"Task '{name}' not found. Available: {list(TASK_REGISTRY.keys())}"
            f"{hint}")
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