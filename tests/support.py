"""Explicit test composition for canonical-window algorithm tests."""
from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.retargeting.hand.topology import ensure_hand25


class CanonicalWindowFixture:
    def __init__(self, receptive_field=3, scale_factor=1.0):
        self.scale_factor = scale_factor
        self.buffer = TemporalBuffer(receptive_field, reset_on_missing=False)
    def update(self, left_hand=None, right_hand=None, timestamp=None, source="unknown", metadata=None):
        hands = {side: None if points is None else ensure_hand25(points, self.scale_factor) for side, points in (("left", left_hand), ("right", right_hand))}
        return self.update_canonical(hands["left"], hands["right"], timestamp, source, metadata)
    def update_canonical(self, left_hand=None, right_hand=None, timestamp=None, source="unknown", metadata=None):
        window = self.buffer.append(CanonicalHandFrame({"left": left_hand, "right": right_hand}, timestamp, source, metadata or {}))
        return None if window is None else window.to_payload()
    def reset(self): self.buffer.reset()
    def reset_side(self, side): self.buffer.reset_side(side)
