"""Sensor geometry adapters; no network, optimizer or robot-specific outputs."""
import numpy as np

from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.coordinates import unified_metric_metadata
from teleoperation.retargeting.hand.coordinates import align_mediapipe_hand


class VisionProMetricAdapter:
    """SDK joint translations -> anatomical palm coordinates in metres.

    The SDK's 25-point ordering matches hand25. Side-specific proper rotations
    restore anatomical axes before the common palm basis/right canonicalization.
    These are the anatomical axes used by the existing native VR adapter, not
    the reflected left-to-right mapping of the original TransHandR checkpoint.
    """
    AXES = {
        'left': np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]),
        'right': np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]]),
    }

    def adapt(self, raw):
        if not isinstance(raw, RawHandFrame) or raw.source != 'visionpro':
            raise ValueError('VR adapter requires raw Vision Pro transforms')
        if raw.metadata.get('units', 'm') != 'm' or raw.metadata.get('source_scale', 1.) != 1.:
            raise ValueError('VR transforms must retain SDK metre dimensions')
        hands, reasons = {}, {}
        for side in ('left', 'right'):
            value = raw.hands[side]
            hands[side], reasons[side] = None, 'missing VR hand'
            if value is None:
                continue
            if raw.geometry_encoding.get(side) != 'transforms25':
                raise ValueError('VR geometry must declare transforms25')
            matrices = np.asarray(value)
            if matrices.shape != (25, 4, 4) or not np.isfinite(matrices).all():
                reasons[side] = 'nonfinite or malformed VR transforms'
                continue
            if not np.allclose(matrices[:, 3], [0., 0., 0., 1.]):
                reasons[side] = 'VR transforms must be homogeneous'
                continue
            points = matrices[:, :3, 3] @ self.AXES[side].T
            aligned = align_mediapipe_hand(points, side=side)
            if not np.any(aligned):
                reasons[side] = 'zero or degenerate VR palm'
                continue
            hands[side], reasons[side] = aligned, None
        return CanonicalHandFrame(hands, raw.timestamp, raw.source,
            {**unified_metric_metadata(), 'invalid_reasons': reasons})
