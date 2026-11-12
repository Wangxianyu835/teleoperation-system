import unittest

import numpy as np

from teleoperation.contracts.observations import (
    RawHandFrame,
    RawTeleopObservation,
    UpperBodyObservation,
)


class BodyObservationContractTests(unittest.TestCase):
    def test_body_result_is_optional_and_requires_the_typed_worker_contract(self):
        from dataclasses import replace
        from tests.test_smplerx_camera_source import body_result
        raw=RawTeleopObservation(RawHandFrame({'left':None,'right':None}),
                                UpperBodyObservation({'left':None,'right':None},{'left':False,'right':False}))
        self.assertIsNone(raw.body_result)
        result=body_result()
        self.assertIs(replace(raw,body_result=result).body_result,result)
        with self.assertRaises(TypeError):replace(raw,body_result={'arrays':{}})

    def test_optional_metadata_and_wrists_preserve_old_construction(self):
        points = {"left": np.zeros((3, 3)), "right": np.zeros((3, 3))}
        body = UpperBodyObservation({"left": points["left"], "right": None},
                                    {"left": True, "right": False})
        self.assertEqual(body.metadata, {})
        hands = RawHandFrame({"left": None, "right": None})
        observation = RawTeleopObservation(hands, body)
        self.assertIsNone(observation.wrists)


if __name__ == "__main__":
    unittest.main()
