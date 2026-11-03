"""Robot-specific action encoding, with no cross-robot arm conversion."""
import numpy as np
from .partial import PartialLimbVector


def validate_native_action(action, action_dim):
    if isinstance(action, PartialLimbVector): raise TypeError("PartialLimbVector is not a complete robot action")
    if not isinstance(action, np.ndarray): raise TypeError("Native action must be np.ndarray")
    if action.shape != (action_dim,): raise ValueError(f"Native action must have shape ({action_dim},), got {action.shape}")
    if not np.isfinite(action).all(): raise ValueError("Native action must be finite")
    return action
