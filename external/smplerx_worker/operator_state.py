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
        self._uncertain_since = None
        self._recovering = False
        self._recovery_box = None
        self._recovery_count = 0
        self.selection = {}

    @staticmethod
    def _iou(a, b):
        intersection = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
        union = np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - intersection
        return float(intersection / union) if union > 0 else 0.

    def select_person(self, detections, now_s):
        """Associate filtered boxes with the existing operator, never by size/score.

        A unique strong match must overlap the previous box by >= .5 IoU and
        exceed every other match by >= .25. Otherwise hold and recover below.
        This is spatial continuity, not biometric identity recognition.
        """
        boxes = np.asarray(detections, dtype=float).reshape(-1, 5)
        selected = 0 if len(boxes) == 1 else None
        policy = 'single_person'
        if len(boxes) > 1 and self.last_box is not None:
            overlaps = np.array([self._iou(self.last_box, box[:4]) for box in boxes])
            order = np.argsort(overlaps)
            best, runner_up = order[-1], order[-2]
            if overlaps[best] >= .5 and overlaps[best] - overlaps[runner_up] >= .25:
                selected, policy = int(best), 'previous_operator_iou'
        box = None if selected is None else boxes[selected, :4]
        reason = self.person(box, now_s, ambiguous=len(boxes) > 1 and selected is None)
        self.selection = dict(accepted_count=len(boxes), selected_index=selected,
                              policy=policy if selected is not None else 'hold',
                              recovering=self._recovering, recovery_count=self._recovery_count,
                              locked=self.locked)
        return (None if reason else box.copy()), reason

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
        if ((self.last_seen is not None and now_s - self.last_seen > self.absence_s)
                or (self._uncertain_since is not None and now_s - self._uncertain_since > self.absence_s)):
            self.locked = True
        if self.locked:
            return 'operator identity uncertain; start a new session and recalibrate'
        if ambiguous:
            if self._uncertain_since is None:
                self._uncertain_since = float(now_s)
            self._recovering = True
            self._recovery_box, self._recovery_count = None, 0
            return 'operator ambiguous; hold until the same operator is clear'
        if box is None:
            self._recovery_box, self._recovery_count = None, 0
            return 'no person detected'
        box = np.asarray(box, dtype=float)
        if self._recovering:
            if self.last_box is not None and self._iou(box, self.last_box) < .5:
                self._recovery_box, self._recovery_count = None, 0
                return 'operator recovering: no match to the previous operator; hold'
            consistent = self._recovery_box is not None and self._iou(box, self._recovery_box) >= .5
            self._recovery_count = self._recovery_count + 1 if consistent else 1
            self._recovery_box = box.copy()
            if self._recovery_count < 3:
                return f'operator recovering: {self._recovery_count}/3 consistent frames; hold'
            self._recovering, self._uncertain_since = False, None
            self._recovery_box, self._recovery_count = None, 0
        if self.last_box is not None:
            if self._iou(box, self.last_box) < .1:
                self.locked = True
                return 'operator identity uncertain; start a new session and recalibrate'
        self.last_box, self.last_seen = box.copy(), float(now_s)
        return ''
