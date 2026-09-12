"""Command-line interface for the L21 retargeting pipeline."""

from __future__ import annotations

import argparse

from retargeting import inference, inspect, training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m retargeting")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="export L21 angles from a two-hand keypoint H5"
    )
    inference.configure_parser(export_parser)

    train_parser = subparsers.add_parser(
        "train", help="train the shared two-hand L21 model"
    )
    training.configure_parser(train_parser)

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate and summarize an exported angle H5"
    )
    inspect.configure_parser(inspect_parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

