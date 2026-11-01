"""Body source rates count distinct accepted model samples, never cached frames."""
from collections import deque
import numpy as np


class BodySourceMetrics:
    def __init__(self):
        self.first = self.last = None
        self.frames = 0
        self.sides = {side: dict(fresh=0, held=0, last_id=None, ages=deque(maxlen=1800), states={})
                      for side in ('left', 'right')}

    def update(self, body, capture_time_s):
        stamp = float(capture_time_s)
        if self.first is None:
            self.first = stamp
        self.last, self.frames = stamp, self.frames + 1
        metadata = body.metadata
        for side, values in self.sides.items():
            sample = (metadata.get('session_id'), metadata.get('body_sample_ids', {}).get(side))
            fresh = bool(body.valid[side] and metadata.get('body_fresh', {}).get(side)
                         and sample != values['last_id'])
            values['fresh'] += int(fresh)
            values['held'] += int(not fresh)
            if fresh:
                values['last_id'] = sample
            age = metadata.get('body_age_s', {}).get(side)
            if age is not None and np.isfinite(age):
                values['ages'].append(float(age))
            state = metadata.get('tracking_state', {}).get(side, 'unknown')
            values['states'][state] = values['states'].get(state, 0) + 1

    def summary(self):
        duration = 0. if self.first is None else self.last - self.first
        return dict(frames=self.frames, duration_s=duration, age_history_capacity=1800,
                    sides={side: dict(reliable_new_samples=value['fresh'],
                       reliable_new_result_hz=value['fresh']/duration if duration > 0 else None,
                       hold_fraction=value['held']/self.frames if self.frames else None,
                       source_age_s_p95=float(np.percentile(value['ages'],95)) if value['ages'] else None,
                       states=dict(value['states'])) for side,value in self.sides.items()})
