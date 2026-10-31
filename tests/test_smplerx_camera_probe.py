"""Prevent diagnostic reports from confusing detector and policy failures."""
import unittest

from devtools import probe_smplerx_camera as probe


class CameraProbeTests(unittest.TestCase):
    def test_network_output_during_shape_warmup_is_not_detection_failure(self):
        self.assertEqual(probe.classify(False, 'calibrating body shape: hold a neutral pose',
                                        1, True), 'shape_initializing')

    def test_filtered_boxes_and_missing_boxes_are_distinguished(self):
        self.assertEqual(probe.classify(False, 'no person detected', 0, False),
                         'no_accepted_person')
        boxes = [[0, 0, 100, 200, .9], [0, 0, 49, 200, .99],
                 [0, 0, 100, 149, .99], [0, 0, 100, 200, .49]]
        self.assertEqual(probe.detection_report(boxes)['accepted_count'], 1)

    def test_expiry_and_identity_rejection_are_not_model_failures(self):
        self.assertEqual(probe.classify(False, 'input frame expired before inference',
                                        None, False), 'input_expired')
        self.assertEqual(probe.classify(False, 'operator identity uncertain; start a new session and recalibrate',
                                        2, False), 'operator_rejected')

    def test_nonfinite_raw_outputs_are_reported_without_invalid_json(self):
        import numpy as np
        import json
        report = probe.array_report({'bad': np.array([float('nan'), float('inf')]),
                                     'ok': np.array([1., 2.])})
        self.assertFalse(report['bad']['finite'])
        self.assertEqual(report['ok']['shape'], [2])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
