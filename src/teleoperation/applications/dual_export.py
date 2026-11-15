"""Retired derived-point export command; use accepted unified recordings."""


def main(args):
    raise ValueError('Legacy dual export cannot create fitted SMPL/PINK commands from derived points; use robot control --robot tron2a_l21 --input-body RAW_POSE.h5 --arm-fitting-dir DIR --record-results OUTPUT.h5')
