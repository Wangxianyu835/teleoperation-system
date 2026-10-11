# 函数 _configure_utf8_stdio（内部辅助计算）；build_parser（读取/解析）；main（入口/工作流）；链路：命令行→参数/路由→应用
"""A single CLI; all workflow execution is dispatched to applications."""
import argparse
from importlib import import_module
import sys
from .registry import COMMANDS
from .checkpoints import CheckpointArgumentParser


def _configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def build_parser():
    from . import schemas
    parser = CheckpointArgumentParser(prog="python -m teleoperation")
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
    _configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    module, function = args._handler.split(":")
    result = getattr(import_module(module), function)(args)
    return 0 if result is None else result
