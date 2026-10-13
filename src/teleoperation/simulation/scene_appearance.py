"""Runtime display settings for the realtime robot scene."""
from __future__ import annotations

import numpy as np


def apply_appearance(scene, g1_color=(0.65, 0.80, 0.95),
                     light_position=(2.0, -4.0, 6.0), shadows=False):
    """Set display-only properties without modifying the robot assets."""
    light = np.asarray(light_position, dtype=float)
    if light.shape != (3,) or not np.isfinite(light).all():
        raise ValueError("light_position must contain three finite values")
    color = None if g1_color is None else np.asarray(g1_color, dtype=float)
    if color is not None and (color.shape != (3,) or not np.isfinite(color).all()
                              or np.any(color < 0) or np.any(color > 1)):
        raise ValueError("g1_color must contain three RGB values between 0 and 1")
    p, cid = scene.p, scene.cid
    p.configureDebugVisualizer(lightPosition=light.tolist(), physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(bool(shadows)), physicsClientId=cid)
    changed = 0
    if color is not None:
        for robot in scene.robots:
            if robot["type"] != "g1":
                continue
            # Use the visual-shape list, including the base, rather than assuming
            # every joint has a mesh. Clear textures so the selected RGB is visible.
            for link in {shape[1] for shape in p.getVisualShapeData(
                    robot["body"], physicsClientId=cid)}:
                p.changeVisualShape(robot["body"], link, rgbaColor=[*color.tolist(), 1.0],
                                    textureUniqueId=-1, physicsClientId=cid)
                changed += 1
    return {"g1_color": None if color is None else color.tolist(),
            "g1_recolored_links": changed, "light_position": light.tolist(),
            "shadows": bool(shadows)}
