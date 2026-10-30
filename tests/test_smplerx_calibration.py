"""Calibration counts independent fresh body samples, not camera polls."""
import unittest
from types import SimpleNamespace
import numpy as np
from teleoperation.contracts.observations import RawTeleopObservation, RawHandFrame, UpperBodyObservation
from teleoperation.retargeting.arm.calibration import NeutralPoseCollector
from tests.test_smplerx_retargeter import calibration
from teleoperation.retargeting.arm.retargeter import ArmRetargeter


def observation(sequence, timestamp):
    points = np.array([[.2, .1, 0], [.2, .4, 0], [.3, .6, 0.]])
    body = UpperBodyObservation(
        {s: points for s in ('left', 'right')}, {s: True for s in ('left', 'right')},
        timestamp, 'smplx_image_normalized', 'normalized',
        world_arms={s: points for s in ('left', 'right')},
        metadata={'source': 'smplerx', 'sample_sequence': sequence, 'session_id': 's',
                  'source_coordinate_frame': 'smplx_camera_root_relative_m', 'source_units': 'm'})
    hands = RawHandFrame({'left': None, 'right': None}, timestamp + .01, 'hands',
                         {'timestamp_unit': 'monotonic_s'})
    return RawTeleopObservation(hands, body)


class SmplerXCalibrationTests(unittest.TestCase):
    def test_dropped_camera_frames_are_allowed_and_duplicates_do_not_count(self):
        config = SimpleNamespace(sample_count=3, max_sample_gap=.5, stability_tolerance=.015, min_visibility=.7,
                                 min_elbow_bend_degrees=10, max_elbow_bend_degrees=100)
        collector = NeutralPoseCollector(config)
        for tick, (seq, stamp) in enumerate([(1, 1.), (4, 1.1), (4, 1.1), (7, 1.2)]):
            collector.add(observation(seq, stamp), 1. + tick / 10)
        self.assertTrue(collector.ready)
        self.assertEqual(len(collector.samples), 3)

    def test_retargering_uses_same_shoulder_reference_as_calibration(self):
        retargeter = ArmRetargeter(calibration(), input_mode='smplerx')
        raw = observation(1, 1.)
        target = retargeter.update(raw.upper_body, 'left')
        points = raw.upper_body.world_arms['left']
        np.testing.assert_allclose(target.pose[:3], points[2] - points[0], atol=1e-7)
