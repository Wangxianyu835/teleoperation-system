from pathlib import Path
import cv2
import numpy as np
from .plotting import COLORS, ROBOT_EDGES, ROBOT_TIPS, SOURCE_EDGES, draw_skeleton, project, text
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


def _panel(title, subtitle, color):
    panel = np.full((HEIGHT, WIDTH, 3), 18, dtype=np.uint8)
    text(panel, title, (12, 24), color, scale=0.52)
    text(panel, subtitle, (12, 45), scale=0.40)
    return panel


def _plot(panel, points, edges, tips, color, args, radius, labels=False):
    projected = project(points, args.view)
    # Fixed origin and isotropic scale; do not recenter/rescale each new frame.
    pixels = projected * np.array([1, -1]) * (185 / (2 * radius))
    pixels += np.array([WIDTH / 2, 105])
    pixels = np.rint(np.clip(pixels, -10000, 10000)).astype(np.int32)
    # Clip drawing to the plot area so zoom cannot obscure status text.
    plot_area = panel[60:250]
    cv2.line(plot_area, (12, 105), (WIDTH - 12, 105), (35, 35, 35), 1)
    cv2.line(plot_area, (WIDTH // 2, 5), (WIDTH // 2, 185), (35, 35, 35), 1)
    draw_skeleton(plot_area, pixels, edges, color, tips, labels)


def render(raw_frame, palms, payload, angles, robot_points, streaks, reasons,
           resets, args, frame_index, fps, traces=True, zoom=1.0):
    """Render current data only; no historical robot pose is cached here."""
    header = np.full((HEADER, WIDTH * 3, 3), 28, dtype=np.uint8)
    timestamp = raw_frame["timestamp"]
    text(header, f"frame={frame_index}  timestamp={timestamp} unix_ms  FPS={fps:.1f}  "
         f"view={args.view.upper()}  zoom={zoom:.2f}", (14, 22), scale=0.57)
    text(header, f"{args.checkpoint.parent.name}/{args.checkpoint.name} | palm_local_to_l21_v1 | "
         "identity tracking OFF | fixed display scales",
         (14, 45), scale=0.48)
    text(header, "q/Esc quit | 1 YZ  2 XZ  3 XY  4 ISO | w window traces | +/- zoom | s PNG", (14, 68), scale=0.48)
    text(header, "Raw: normalized image XY. Palm: normalized coordinates. Robot: URDF units. Screen distances differ.",
         (14, 89), scale=0.43)
    rows = [header]
    for side in SIDES:
        color = COLORS[side]
        raw, palm, prediction = raw_frame[side], palms[side], angles[side]
        window = None if payload is None else payload["hands"][side]
        status = "INVALID" if palm is None else f"WARMUP {streaks[side]}/3" if window is None else "VALID"
        raw_panel = _panel(f"{side.upper()} | RAW 21 | {'MISSING' if raw is None else 'DETECTED'}",
                           "Image X right, Y down; unmirrored", color)
        if raw is not None:
            width = raw_frame.get("metadata", {}).get("image_width", 640)
            height = raw_frame.get("metadata", {}).get("image_height", 480)
            scale = min((WIDTH - 40) / width, 185 / height)
            size = np.array([width, height]) * scale
            offset = np.array([(WIDTH - size[0]) / 2, 5 + (185 - size[1]) / 2])
            pixels = np.rint(np.clip(raw[:, :2] * size + offset, -10000, 10000)).astype(np.int32)
            draw_skeleton(raw_panel[60:250], pixels, SOURCE_EDGES, color, (4, 8, 12, 16, 20), args.labels)
            text(raw_panel, f"shape={raw.shape} dtype={raw.dtype} finite={bool(np.isfinite(raw).all())}",
                 (12, 285), scale=0.40)
        else:
            text(raw_panel, "No current hand", (110, 165), color)

        palm_panel = _panel(f"{side.upper()} | PALM-LOCAL 25 | {status}",
                            f"{args.view.upper()} | streak={streaks[side]}/3 | resets={resets[side]}", color)
        if palm is not None:
            if traces and window is not None:
                for age, alpha in ((0, 0.25), (1, 0.55)):
                    _plot(palm_panel, window[age], PALM_EDGES, (), tuple(int(c * alpha) for c in color),
                          args, args.palm_radius / zoom)
            _plot(palm_panel, palm, PALM_EDGES, PALM_TIPS, color, args, args.palm_radius / zoom, args.labels)
            text(palm_panel, f"shape={palm.shape} finite={bool(np.isfinite(palm).all())}", (12, 268), scale=0.42)
            text(palm_panel, f"window={None if window is None else window.shape}", (12, 289), scale=0.42)
            text(palm_panel, "traces: t-2 dark / t-1 medium / t current" if traces else "current frame only",
                 (12, 309), scale=0.38)
        else:
            text(palm_panel, "INVALID - side buffer cleared", (45, 165), color, scale=0.50)
            reason = reasons[side] or "missing"
            text(palm_panel, reason[:57], (12, 285), scale=0.37)
            if len(reason) > 57:
                text(palm_panel, reason[57:114], (12, 306), scale=0.37)

        robot_panel = _panel(f"{side.upper()} | L21 FK | {status}",
                             f"{args.view.upper()} | current 18D prediction only", color)
        if prediction is not None:
            _plot(robot_panel, robot_points[side], ROBOT_EDGES, ROBOT_TIPS, color, args,
                  args.robot_radius / zoom, args.labels)
            text(robot_panel, f"angles={prediction.shape} finite={bool(np.isfinite(prediction).all())}",
                 (12, 269), scale=0.42)
            text(robot_panel, f"range=[{prediction.min():.3f}, {prediction.max():.3f}] rad", (12, 291), scale=0.42)
        else:
            text(robot_panel, "No current prediction", (90, 155), color, scale=0.52)
            text(robot_panel, "Waiting for 3 consecutive valid frames", (25, 184), scale=0.44)
        rows.append(np.hstack((raw_panel, palm_panel, robot_panel)))
    return np.vstack(rows)


def save_png(path, canvas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())
    print(f"Saved {path.resolve()}", flush=True)

