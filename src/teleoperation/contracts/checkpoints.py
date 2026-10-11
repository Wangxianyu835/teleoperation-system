"""Checkpoint selection rules shared by CLI and hand-model factories."""


def validate_checkpoint_selection(checkpoint, left_checkpoint, right_checkpoint, *, required=False):
    paired = left_checkpoint is not None or right_checkpoint is not None
    if paired and checkpoint is not None:
        raise ValueError("--checkpoint cannot be combined with --left-checkpoint/--right-checkpoint")
    if paired and (left_checkpoint is None or right_checkpoint is None):
        raise ValueError("Both --left-checkpoint and --right-checkpoint are required together")
    if required and checkpoint is None and not paired:
        raise ValueError("Provide --checkpoint or both --left-checkpoint and --right-checkpoint")
