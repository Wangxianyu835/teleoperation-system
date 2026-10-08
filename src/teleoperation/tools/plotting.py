"""Display-only projections and skeleton drawing shared by live/offline tools."""

import cv2
import numpy as np
from teleoperation.retargeting.hand.config import JOINT_EDGES, JOINT_NAMES

SOURCE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
NAME_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
ROBOT_EDGES = tuple((NAME_INDEX[a], NAME_INDEX[b]) for a, b in JOINT_EDGES)
ROBOT_TIPS = tuple(NAME_INDEX[f"{finger}_tip"] for finger in
                   ("thumb", "index", "middle", "ring", "pinky"))
COLORS = {"left": (255, 145, 65), "right": (80, 220, 100)}  # BGR


def project(points, view):
    if view == "yz":
        return points[..., [1, 2]]
    if view == "xz":
        return points[..., [0, 2]]
    if view == "xy":
        return points[..., [0, 1]]
    # Orthographic oblique projection: the horizontal axis follows XY,
    # the vertical axis is tilted toward +Z. The pose itself is not changed.
    azimuth, elevation = np.deg2rad(-60), np.deg2rad(20)
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    vertical = np.array([-np.sin(elevation) * np.cos(azimuth),
                         -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)])
    return np.stack((points @ horizontal, points @ vertical), axis=-1)


def text(canvas, message, position, color=(225, 225, 225), scale=0.5):
    cv2.putText(canvas, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_skeleton(canvas, pixels, edges, color, tips, labels):
    for a, b in edges:
        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), color, 2, cv2.LINE_AA)
    for i, pixel in enumerate(pixels):
        point = tuple(pixel)
        cv2.circle(canvas, point, 5 if i in tips else 3,
                   (0, 190, 255) if i in tips else color, -1, cv2.LINE_AA)
        if labels:
            text(canvas, str(i), (int(pixel[0]) + 4, int(pixel[1]) - 4), scale=0.35)
