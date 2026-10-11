# 函数 __getattr__（内部辅助计算）；链路：应用→仿真环境/机器人→传感器/记录
"""Simulation components load when explicitly requested."""
def __getattr__(name):
    if name == "SimulationEnv":
        from .environment import SimulationEnv
        return SimulationEnv
    if name == "RobotLoader":
        from .robot_loader import RobotLoader
        return RobotLoader
    if name == "SensorRecorder":
        from teleoperation.data.recorder import SensorRecorder
        return SensorRecorder
    if name == "DomainRandomizer":
        from .randomization import DomainRandomizer
        return DomainRandomizer
    raise AttributeError(name)
