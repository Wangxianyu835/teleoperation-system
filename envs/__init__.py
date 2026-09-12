from .robot_loader import RobotLoader
from .sensor_recorder import SensorRecorder
from .domain_randomizer import DomainRandomizer

# ⚠️ 不要用 try/except 吞掉这里的导入异常！
# 之前写成：
#     try:
#         from .simulation_env import SimulationEnv
#     except (ImportError, ValueError):
#         SimulationEnv = None
# 结果把 simulation_env.py 的导入错误静默隐藏，只在运行时暴露成
# "TypeError: 'NoneType' object is not callable"，极难排查。
# 让错误直接抛出，才能第一时间发现。
from .simulation_env import SimulationEnv

__all__ = ['RobotLoader', 'SensorRecorder', 'DomainRandomizer', 'SimulationEnv']
