"""Realtime statistics and bounded resource teardown."""
import statistics
import threading
import numpy as np
from teleoperation.robots.l21 import L21_JOINT_ORDER


def _describe(vectors, mapping=None) -> dict:
    """Frame count plus per-dimension observed range for one side."""
    mapping = mapping if mapping is not None else list(L21_JOINT_ORDER)
    array = np.asarray(vectors, dtype=np.float64).reshape(-1, len(mapping))
    return {
        "frames": int(array.shape[0]),
        "min": [float(value) for value in array.min(axis=0)],
        "max": [float(value) for value in array.max(axis=0)],
        "mapping": list(mapping),
    }


def _loop_fps(marks) -> float:
    """Loop rate over the last second of frame marks (0.0 when undefined)."""
    if len(marks) < 2:
        return 0.0
    cutoff = marks[-1] - 1.0
    recent = [mark for mark in marks if mark >= cutoff]
    if len(recent) < 2:
        recent = marks[-2:]
    return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-9)


def _median_interval(marks) -> float:
    if len(marks) < 2:
        return 0.0
    return statistics.median([later - earlier for earlier, later in zip(marks, marks[1:])])


RELEASE_TIMEOUT_SECONDS = 2.0


def _drain_in_background(action, label: str, timeout: float = RELEASE_TIMEOUT_SECONDS) -> None:
    """Run a slow teardown step in a daemon thread; never block the exit.

    ``MediaPipeCameraInput.release()`` spends about 42 s inside
    ``landmarker.close()`` (TFLite/XNNPACK teardown) on this machine, and
    ``p.disconnect()`` hangs when the PyBullet GUI client already died. Both look
    like a freeze. We wait at most ``timeout`` seconds and then continue;
    a teardown still running at that point remains in the daemon thread.
    """

    def run_action() -> None:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - teardown must not hide the summary
            print(f"note: {label} failed: {type(error).__name__}: {error}", flush=True)

    thread = threading.Thread(target=run_action, name="teardown", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"note: {label} is still running after {timeout:.0f} s; exiting without "
              f"waiting.", flush=True)
