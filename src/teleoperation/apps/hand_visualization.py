"""Show live raw hands, palm-local windows, and palm_local_v2 L21 predictions.

Two rows (left/right), three columns (raw image XY / palm-local / L21 FK).
Missing sides clear immediately; the robot panel waits for three new valid
frames. Darker palm skeletons are older frames from the actual model window.
Only display projections and zoom change here; model inputs remain untouched.

Run with --model-asset-path pointing to hand_landmarker.task. q/Esc quits,
1/2/3/4 selects YZ/XZ/XY/ISO, w toggles window traces, +/- zooms the aligned
panels, and s saves a PNG. --headless --frames N can verify rendering without a
window. No H5 recording, robot commands, smoothing, or identity tracking.
"""

from __future__ import annotations

from pathlib import Path
import time
import sys

import cv2
import numpy as np
import torch

from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.paths import DEFAULT_REALTIME_SNAPSHOT
from teleoperation.retargeting.hand.config import L21
from teleoperation.apps.hand_realtime import (
    MediaPipeCameraWorkflow,
    load_palm_local_retargeter,
)
from teleoperation.retargeting.hand.angles import angle18_to_nodes
from teleoperation.tools.plotting import (
    COLORS, ROBOT_EDGES, ROBOT_TIPS, SOURCE_EDGES, draw_skeleton, project, text,
)


SIDES = ("left", "right")
WINDOW = "MediaPipe realtime: raw -> palm-local -> window -> L21"
WIDTH, HEIGHT, HEADER = 420, 320, 100
PALM_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)
PALM_TIPS = (4, 9, 14, 19, 24)


def make_fk():
    """Create the existing, unchanged URDF FK once per side."""
    return {
        side: create_hand_kinematics(
            getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(), device="cpu",
            scale_factor=L21.training.robot_scale,
        )
        for side in SIDES
    }


def run(args):
    canvas = None
    created_window = False
    completed = 0
    resets = {side: 0 for side in SIDES}
    outputs = {side: 0 for side in SIDES}
    zoom, traces = 1.0, True
    try:
        if args.device == "cpu":
            torch.set_num_threads(1)
        model = load_palm_local_retargeter(args.checkpoint, args.device)
        retargeter = PoseTransformerRetargeter(model, args.device)
        fk = make_fk()
        print(f"Checkpoint: {args.checkpoint}\nCamera: {args.camera_index}\n"
              "q/Esc quit; missing sides are cleared without holding old poses.", flush=True)
        with MediaPipeCameraWorkflow(args.model_asset_path, args.camera_index) as camera:
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                created_window = True
                cv2.resizeWindow(WINDOW, 1150, 676)
            started = time.perf_counter()
            while args.frames == 0 or completed < args.frames:
                previous = dict(camera.processor.valid_streak)
                window = camera.next_window()
                payload = None if window is None else window.to_payload()
                result = None if window is None else retargeter.retarget(window)
                predictions = {side: None if result is None or getattr(result, side + "_angles") is None else getattr(result, side + "_angles").values for side in SIDES}
                positions = {side: None for side in SIDES}
                completed += 1
                for side in SIDES:
                    if previous[side] and camera.processor.valid_streak[side] == 0:
                        resets[side] += 1
                        print(f"frame={completed} {side}: RESET", flush=True)
                    if predictions[side] is not None:
                        positions[side] = fk_positions(fk[side], predictions[side])
                        outputs[side] += 1
                        if previous[side] == 2:
                            print(f"frame={completed} {side}: window ready after 3 valid frames", flush=True)
                fps = completed / max(time.perf_counter() - started, 1e-9)
                canvas = render(camera.last_raw_frame, camera.processor.current_hands, payload,
                                predictions, positions, camera.processor.valid_streak,
                                camera.processor.invalid_reasons, resets, args, completed, fps, traces, zoom)
                if not args.headless:
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
                        args.view = {ord('1'): 'yz', ord('2'): 'xz', ord('3'): 'xy', ord('4'): 'iso'}[key]
                    elif key == ord('w'):
                        traces = not traces
                    elif key in (ord('+'), ord('=')):
                        zoom = min(zoom * 1.2, 8.0)
                    elif key == ord('-'):
                        zoom = max(zoom / 1.2, 0.125)
                    elif key == ord('s'):
                        save_png(args.snapshot or DEFAULT_REALTIME_SNAPSHOT, canvas)
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as error:
        print(f"Realtime visualization failed: {error}", file=sys.stderr)
        return 1
    finally:
        if created_window:
            cv2.destroyAllWindows()
    if args.snapshot is not None and canvas is not None:
        save_png(args.snapshot, canvas)
    print(f"Frames={completed} outputs={outputs} resets={resets}; resources released", flush=True)
    if not any(outputs.values()):
        print("No real-hand model output observed during this run.")
    return 0

from teleoperation.tools.hand_visualization import _panel, _plot, render, save_png

from teleoperation.retargeting.hand.fk_diagnostics import fk_positions

from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
