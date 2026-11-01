"""Unified export application, also used by the hand dispatch module."""


def run(args):
    from .hand_control import export
    return export(args)
