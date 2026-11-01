import unittest

import numpy as np

from teleoperation.contracts.observations import (
    RawHandFrame,
    RawTeleopObservation,
    UpperBodyObservation,
)


class BodyObservationContractTests(unittest.TestCase):
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
