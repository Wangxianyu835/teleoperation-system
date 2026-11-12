"""Retired custom camera adapter command; use the shared control source."""


def main(args=None):
    raise ValueError('Legacy dual realtime adapters do not provide the fitted control contract; use robot control --robot tron2a_l21 --source camera --arm-fitting-dir DIR, with --input-body for recorded raw SMPL poses')
