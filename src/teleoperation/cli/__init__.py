"""A single CLI; all workflow execution is dispatched to apps."""
import argparse
from importlib import import_module
from .registry import COMMANDS


def build_parser():
    from . import schemas
    parser = argparse.ArgumentParser(prog="python -m teleoperation")
    groups = parser.add_subparsers(dest="group", required=True)
    commands = {}
    for group, name, configure, handler in COMMANDS:
        if group not in commands:
            commands[group] = groups.add_parser(group).add_subparsers(dest="command", required=True)
        child = commands[group].add_parser(name)
        getattr(schemas, configure)(child)
        child.set_defaults(_parser=child)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    module, function = args._handler.split(":")
    result = getattr(import_module(module), function)(args)
    return 0 if result is None else result
