"""Parse shared or paired checkpoints without loading model dependencies."""

import argparse
from pathlib import Path

from teleoperation.contracts.checkpoints import validate_checkpoint_selection


def add_checkpoint_arguments(parser, *, default=None, prefix="", required=False):
    stem = f"{prefix}-" if prefix else ""
    for name in ("checkpoint", "left-checkpoint", "right-checkpoint"):
        parser.add_argument(f"--{stem}{name}", type=Path)
    selections = list(parser.get_default("_checkpoint_selections") or ())
    selections.append((prefix, default, required))
    parser.set_defaults(_checkpoint_selections=selections)


class CheckpointArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        for prefix, default, required in getattr(parsed, "_checkpoint_selections", ()):
            stem = f"{prefix}_" if prefix else ""
            names = [stem + name for name in ("checkpoint", "left_checkpoint", "right_checkpoint")]
            values = [getattr(parsed, name) for name in names]
            if all(value is None for value in values):
                values[0] = default
                setattr(parsed, names[0], default)
            try:
                validate_checkpoint_selection(*values, required=required)
            except ValueError as error:
                parsed._parser.error(f"{prefix + ': ' if prefix else ''}{error}")
        return parsed
