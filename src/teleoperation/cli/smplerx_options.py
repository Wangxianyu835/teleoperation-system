"""Shared command-line options for the isolated SMPLer-X worker."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_smplerx_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit paths for the external S32 inference process.

    The main teleoperation environment does not import SMPLer-X.  ``None`` is
    intentional for the paths: the application validates them before opening
    a camera, so a missing model produces a useful configuration error.
    """

    parser.add_argument(
        "--smplerx-worker-python",
        type=Path,
        default=None,
        help="Python interpreter for the isolated SMPLer-X worker",
    )
    parser.add_argument(
        "--smplerx-root",
        type=Path,
        default=None,
        help="checked-out SMPLer-X source root",
    )
    parser.add_argument(
        "--smplerx-checkpoint",
        type=Path,
        default=None,
        help="S32 SMPLer-X checkpoint",
    )
    parser.add_argument(
        "--smplx-model-dir",
        type=Path,
        default=None,
        help="directory containing SMPL-X body-model assets",
    )
    parser.add_argument(
        "--smplerx-model",
        choices=("s32",),
        default="s32",
        help="SMPLer-X model variant (the camera arm route currently fixes S32)",
    )
