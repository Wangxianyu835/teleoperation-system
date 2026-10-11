# 函数 iter_angle_h5（读取/解析）；链路：CLI/调用方→应用编排→输入/算法/存储/仿真
"""Compose strict canonical angle reading with the original hold algorithm."""
from teleoperation.data.hand_h5 import iter_angle_h5 as read_angle_frames
from teleoperation.retargeting.hand.postprocessing import hold_angle_frames


def iter_angle_h5(path):
    yield from hold_angle_frames(read_angle_frames(path))
