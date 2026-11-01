"""Shared command-line options for the isolated SMPLer-X worker."""

from __future__ import annotations

import argparse
from pathlib import Path


def smplerx_defaults():
    root = Path(__file__).resolve().parents[3]
    return dict(smplerx_worker_python=root / '.venv/smplerx-worker/Scripts/python.exe',
                smplerx_root=root / '.venv/smplerx-assets/SMPLer-X',
                smplerx_checkpoint=root / '.venv/smplerx-assets/smpler_x_s32.pth.tar',
                smplx_model_dir=root / 'datasets/smpl_models')


def add_smplerx_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit paths for the external S32 inference process.

    The main environment does not import SMPLer-X. Defaults use local ignored
    assets; applications validate these paths before opening a camera.
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
    parser.set_defaults(**smplerx_defaults())
