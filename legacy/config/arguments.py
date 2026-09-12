"""Command-line arguments shared by the training entry points."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hand retargeting model")

    parser.add_argument(
        "-ht",
        "--hand-type",
        default="right",
        choices=("left", "right", "both"),
        help="hand data side to load",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="h5",
        choices=("h5",),
        help="dataset folder name under dataset/data",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("input/visual_hand_data_20260912_112108.h5"),
        help="path to the two-hand visual keypoint H5 file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        help="checkpoint/log root directory",
    )

    parser.add_argument(
        "-s",
        "--stride",
        default=1,
        type=int,
        help="chunk size used by the training generator",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="number of training epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1024,
        type=int,
        help="batch size in predicted frames",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "-lrd",
        "--lr-decay",
        default=0.99,
        type=float,
        help="learning rate decay per epoch",
    )
    parser.add_argument(
        "-no-da",
        "--no-data-augmentation",
        dest="data_augmentation",
        action="store_false",
        help="disable train-time data augmentation",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.2,
        type=float,
        help="fraction of raw H5 frames reserved for validation",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="optional single-hand checkpoint used to initialize the shared model",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="training device",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="random seed for dataset shuffling and torch",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=20,
        type=int,
        help="epochs without validation improvement before stopping",
    )
    parser.add_argument(
        "--left-coordinate-mode",
        choices=("none", "mirror_x"),
        default="none",
        help="coordinate normalization applied to left-hand offline H5 data",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="isolated name for two-hand model, log, and metric outputs",
    )

    parser.set_defaults(data_augmentation=False)
    return parser.parse_args()
