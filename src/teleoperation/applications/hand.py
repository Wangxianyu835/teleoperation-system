"""Hand application dispatch; dependencies load only for the selected workflow."""
def align(args):
    if getattr(args, 'input_profile', None) is not None:
        from .native_hand_align import run as align_native
        return align_native(args)
    from .hand_align import run
    return run(args)
def export(args):
    if getattr(args, 'backend', None) is not None or getattr(args, 'hand_model', None) is not None:
        if args.backend is None or args.hand_model is None:
            raise ValueError('Native export requires both --hand-model and --backend')
        from .hand_control import export as export_native
        return export_native(args)
    from .hand_export import run
    return run(args)
def inspect(args):
    from .hand_inspect import run
    return run(args)
def realtime(args):
    from .hand_realtime import run
    return run(args)
