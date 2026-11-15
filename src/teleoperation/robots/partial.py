"""Partial diagnostic vectors never constitute executable robot actions."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartialLimbVector:
    values: np.ndarray
    limbs: tuple[str, ...]
