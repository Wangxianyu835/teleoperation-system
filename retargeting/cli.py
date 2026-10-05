"""Single public CLI; optional command dependencies load only when selected."""

from __future__ import annotations

import argparse
from importlib import import_module


class _CommandParser(argparse.ArgumentParser):
    def __init__(self, *args, module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._module = module
        self._configured = False

    def parse_known_args(self, args=None, namespace=None):
        if self._module is not None and not self._configured:
            import_module(self._module).configure_parser(self)
            self._configured = True
        return super().parse_known_args(args, namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m retargeting")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_CommandParser)
    for name, module, help_text in (
        ("train", "training", "train the shared two-hand L21 model"),
        ("export", "inference", "export L21 angles from aligned Hand25 H5"),
        ("inspect", "inspect", "validate and summarize exported angle H5"),
        ("align", "preprocessing", "align raw MediaPipe H5 to palm-local L21 coordinates"),
        ("realtime", "mediapipe", "MediaPipe palm-local inference; optional live visualization"),
    ):
        commands.add_parser(name, module=f"retargeting.{module}", help=help_text)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
