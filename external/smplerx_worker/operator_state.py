"""Conservative session identity and stable shape policy; no model dependencies."""
import numpy as np


class OperatorState:
    def __init__(self, samples=5, shape_tolerance=.5, absence_s=2.):
        self.samples, self.shape_tolerance, self.absence_s = samples, shape_tolerance, absence_s
        self.window = []
        self.fixed_shape = None
        self.last_seen = None
        self.last_box = None
        self.locked = False

    def shape_sample(self, shape):
        if self.fixed_shape is not None:
            return self.fixed_shape.copy()
        shape = np.asarray(shape, dtype=np.float32)
        if shape.shape != (10,) or not np.isfinite(shape).all():
            self.window.clear()
            return None
        if self.window and np.max(np.abs(shape - np.median(self.window, axis=0))) > self.shape_tolerance:
            self.window.clear()
        self.window.append(shape.copy())
        if len(self.window) < self.samples:
            return None
        self.fixed_shape = np.median(self.window, axis=0)
        self.window.clear()
        return self.fixed_shape.copy()

    def person(self, box, now_s, *, ambiguous=False):
        if ambiguous or (self.last_seen is not None and now_s - self.last_seen > self.absence_s):
            self.locked = True
        if self.locked:
            return 'operator identity uncertain; start a new session and recalibrate'
        if box is None:
            return 'no person detected'
        box = np.asarray(box, dtype=float)
        if self.last_box is not None:
            intersection = np.maximum(0., np.minimum(box[2:], self.last_box[2:]) - np.maximum(box[:2], self.last_box[:2])).prod()
            area = (box[2:] - box[:2]).prod() + (self.last_box[2:] - self.last_box[:2]).prod() - intersection
            if area <= 0 or intersection / area < .1:
                self.locked = True
                return 'operator identity uncertain; start a new session and recalibrate'
        self.last_box, self.last_seen = box.copy(), float(now_s)
        return ''
