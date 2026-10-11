# 函数 align（坐标处理）；export（保存/导出）；inspect/realtime（业务处理）；链路：CLI/调用方→应用编排→输入/算法/存储/仿真
"""Hand application dispatch; dependencies load only for the selected workflow."""
def align(args):
    from .hand_align import run
    return run(args)
def export(args):
    from .hand_export import run
    return run(args)
def inspect(args):
    from .hand_inspect import run
    return run(args)
def realtime(args):
    from .hand_realtime import run
    return run(args)
