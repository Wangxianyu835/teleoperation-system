from __future__ import annotations
import numpy as np
from teleoperation.contracts.constants import *
MEDIAPIPE_HAND_KEYPOINTS = 21
VISIONPRO_SOURCE = "visionpro"
MEDIAPIPE_APPROX_SOURCE = "mediapipe_approx"
from itertools import product
DEFAULT_MAX_CENTER_DISPLACEMENT = 0.08
DEFAULT_MAX_SHAPE_RMSE = 0.05
class HandIdentityTracker:
    """Keep MediaPipe detections attached to continuous left/right tracks."""

    def __init__(
        self,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        self.max_center_displacement = float(max_center_displacement)
        self.max_shape_rmse = float(max_shape_rmse)
        if self.max_center_displacement <= 0 or self.max_shape_rmse <= 0:
            raise ValueError("Hand tracking thresholds must be positive")
        self._previous: dict[str, np.ndarray | None] = {
            side: None for side in HAND_SIDES
        }

    def update(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | None]:
        detections = [
            (side, np.asarray(points, dtype=np.float32))
            for side, points in (("left", left_hand), ("right", right_hand))
            if _is_trackable_hand(points)
        ]
        return self.update_detections(detections)

    def update_detections(
        self,
        detections: list[tuple[str, np.ndarray]],
    ) -> dict[str, np.ndarray | None]:
        assigned = self.assign(detections)
        self._previous = {
            side: None if assigned[side] is None else assigned[side].copy()
            for side in HAND_SIDES
        }
        return assigned

    def assign(
        self,
        detections: list[tuple[str, np.ndarray]],
    ) -> dict[str, np.ndarray | None]:
        """Assign labeled detections using temporal continuity as the authority."""
        candidates: list[tuple[str, np.ndarray]] = []
        for label, points in detections:
            if label not in HAND_SIDES:
                raise ValueError(f"Invalid hand side: {label}")
            array = np.asarray(points, dtype=np.float32)
            if _is_trackable_hand(array):
                candidates.append((label, array))
        if len(candidates) > len(HAND_SIDES):
            raise ValueError("At most two hand detections are supported")

        result = {side: None for side in HAND_SIDES}
        if not candidates:
            return result
        if all(self._previous[side] is None for side in HAND_SIDES):
            for label, points in candidates:
                if result[label] is None:
                    result[label] = points
            return result

        choices = (None,) + HAND_SIDES
        plausible_existing: list[set[str]] = []
        for _, points in candidates:
            matches = set()
            for side in HAND_SIDES:
                previous = self._previous[side]
                if previous is None:
                    continue
                center_displacement, shape_rmse = _continuity_metrics(
                    points,
                    previous,
                )
                if (
                    center_displacement <= self.max_center_displacement
                    and shape_rmse <= self.max_shape_rmse
                ):
                    matches.add(side)
            plausible_existing.append(matches)

        best_score = None
        best_assignment = None
        for targets in product(choices, repeat=len(candidates)):
            assigned_sides = [target for target in targets if target is not None]
            if len(set(assigned_sides)) != len(assigned_sides):
                continue

            accepted = 0
            label_matches = 0
            total_cost = 0.0
            valid = True
            for candidate_index, ((label, points), target) in enumerate(
                zip(candidates, targets)
            ):
                if target is None:
                    continue
                previous = self._previous[target]
                if previous is None:
                    if target != label or plausible_existing[candidate_index]:
                        valid = False
                        break
                    cost = 0.0
                else:
                    metrics = _continuity_metrics(points, previous)
                    if (
                        metrics[0] > self.max_center_displacement
                        or metrics[1] > self.max_shape_rmse
                    ):
                        valid = False
                        break
                    cost = (
                        metrics[0] / self.max_center_displacement
                        + metrics[1] / self.max_shape_rmse
                    )
                accepted += 1
                label_matches += int(target == label)
                total_cost += cost

            if not valid:
                continue
            score = (accepted, label_matches, -total_cost)
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = targets

        if best_assignment is not None:
            for (_, points), target in zip(candidates, best_assignment):
                if target is not None:
                    result[target] = points
        return result

    def reset(self) -> None:
        for side in HAND_SIDES:
            self._previous[side] = None


def _is_trackable_hand(points: np.ndarray | None) -> bool:
    if points is None:
        return False
    array = np.asarray(points)
    return bool(
        array.ndim == 2
        and array.shape[1:] == (HAND_COORDS,)
        and array.shape[0] in (MEDIAPIPE_HAND_KEYPOINTS, HAND_KEYPOINTS)
        and np.isfinite(array).all()
        and not np.all(array == 0)
    )


def _continuity_metrics(
    current: np.ndarray,
    previous: np.ndarray,
) -> tuple[float, float]:
    if current.shape != previous.shape:
        return float("inf"), float("inf")
    center_displacement = float(
        np.linalg.norm(_palm_center(current) - _palm_center(previous))
    )
    current_shape = current - current[0:1]
    previous_shape = previous - previous[0:1]
    shape_rmse = float(np.sqrt(np.mean((current_shape - previous_shape) ** 2)))
    return center_displacement, shape_rmse


def _palm_center(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == MEDIAPIPE_HAND_KEYPOINTS:
        palm_indices = (0, 5, 9, 13, 17)
    elif points.shape[0] == HAND_KEYPOINTS:
        palm_indices = (0, 6, 11, 16, 21)
    else:
        raise ValueError(f"Unsupported hand point count: {points.shape[0]}")
    return points[np.asarray(palm_indices)].mean(axis=0)
