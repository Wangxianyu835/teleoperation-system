"""PyBullet URDF loading helpers that work under non-ASCII project paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_urdf(urdf_path: str | Path, **kwargs: Any) -> int:
    """Load a URDF through an ASCII-safe relative filename.

    PyBullet on Windows can fail when an absolute URDF path contains
    non-ASCII characters.  Loading through a relative filename while the
    URDF directory is the current working directory avoids that limitation.
    """
    import pybullet as p

    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF was not found: {path}")

    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        return p.loadURDF(path.name, **kwargs)
    finally:
        os.chdir(previous)


def create_ground(pybullet: Any, physicsClientId: int = 0) -> int:
    """Create the standard ground plane without loading pybullet_data assets."""
    half_extents = [100.0, 100.0, 5.0]
    collision = pybullet.createCollisionShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=physicsClientId,
    )
    visual = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        physicsClientId=physicsClientId,
    )
    return pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, -5.0],
        physicsClientId=physicsClientId,
    )
