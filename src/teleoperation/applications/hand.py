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
