"""Camera image and landmark overlay for offline H5 recording."""
#录制时打印窗口


import time

import cv2
import numpy as np

from .plotting import COLORS, SOURCE_EDGES, draw_skeleton, text


WINDOW = "Offline H5 hand recording"


def render_recording_preview(frame, hand_side, count, elapsed, fps):
    image = frame.get("image_bgr")
    if image is None:
        raise ValueError("Camera image is unavailable for recording preview")
    image = image.copy()
    height, width = image.shape[:2]
    statuses = []
    for side in ("left", "right"):
        selected = hand_side in (side, "both")
        points = frame[side]
        status = ("DETECTED" if points is not None else "MISSING") if selected else "NOT RECORDED"
        statuses.append(f"{side.upper()}: {status}")
        if points is not None:
            pixels = np.rint(np.clip(points[:, :2] * [width, height], -10000, 10000)).astype(np.int32)
            color = COLORS[side] if selected else (130, 130, 130)
            draw_skeleton(image, pixels, SOURCE_EDGES, color, (4, 8, 12, 16, 20), True)
            wrist = pixels[0]
            text(image, side.upper(), (int(wrist[0]) + 8, int(wrist[1]) + 20), color)
    header = np.full((90, width, 3), 25, dtype=np.uint8)
    text(header, f"REC {hand_side.upper()} | Frames {count} | {elapsed:.1f}s | FPS {fps:.1f}",
         (10, 22), (80, 220, 255), scale=0.55)
    text(header, " | ".join(statuses), (10, 47), scale=0.48)
    text(header, "q / Esc / close window: stop and save H5", (10, 73), scale=0.48)
    return np.vstack((header, image))


class RecordingPreview:
    def __init__(self, hand_side):
        self.hand_side = hand_side
        self._shown = False
        self._started = time.perf_counter()
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        except cv2.error as error:
            raise RuntimeError("Cannot open recording preview; use --no-preview for capture without a window") from error

    def show(self, frame, count):
        if self._shown and not self._visible():
            return "window_closed"
        elapsed = time.perf_counter() - self._started
        canvas = render_recording_preview(frame, self.hand_side, count, elapsed,
                                          count / max(elapsed, 1e-9))
        cv2.imshow(WINDOW, canvas)
        self._shown = True
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "keyboard_quit"
        if not self._visible():
            return "window_closed"
        return None

    def _visible(self):
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            # Closing the window using its title-bar button may destroy it first.
            pass
