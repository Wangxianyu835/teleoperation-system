"""Fixed-seed synthetic camera arm geometry; no claims of real capture."""
import json

import numpy as np


SCENARIOS = (
    ("neutral", 3.), ("static", 2.), ("static_noise", 2.),
    ("move_left", 3.), ("move_right", 3.), ("move_both", 3.),
    ("body_translation", 2.), ("drop_left", .5), ("recover_left", 2.),
    ("drop_both", .5), ("recover_both", 2.), ("step_both", 2.),
    ("clamp", 2.), ("settle", 2.), ("pose_fallback", 2.),
)


def palm_points():
    points = [(.5, .5, 0.)]
    for base_x in (.06, .03, 0., -.03, -.05):
        for joint in range(1, 5):
            points.append((.5 + base_x, .5 - .035 * joint, 0.))
    return np.asarray(points, dtype=float)


def synthetic_recording(rate=30, seed=20261022, scenarios=SCENARIOS):
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    rng = np.random.default_rng(seed)
    data = {key: [] for key in ("left_arm", "right_arm", "left_hand", "right_hand",
                                "world_left_hand", "world_right_hand")}
    labels, intervals = [], []
    hand = palm_points()
    for name, duration in scenarios:
        count = max(2, int(round(duration * rate)))
        begin = len(labels)
        intervals.append({"name": name, "start_frame": begin, "end_frame_exclusive": begin + count,
                          "start_s": begin / rate, "end_s": (begin + count) / rate})
        for index in range(count):
            phase = 2 * np.pi * index / max(1, count - 1)
            for side in ("left", "right"):
                shoulder = np.array([.17 if side == "left" else -.17, -.35, 0.])
                points = np.stack((shoulder, shoulder + [0, .26, .04], shoulder + [0, .49, 0]))
                moving = name == "move_both" or name == "move_" + side
                if moving:
                    points[2] += .015 * np.array([np.sin(phase), .5 * np.sin(2 * phase), np.sin(phase)])
                elif name == "static_noise":
                    points[2] += rng.normal(0, .002, 3)
                elif name == "body_translation":
                    points += [.05 * np.sin(phase), .03 * np.sin(phase), .04 * np.sin(phase)]
                elif name == "step_both":
                    points[2] += [.02, 0, 0]
                elif name in ("recover_left", "recover_both"):
                    if side == "left" or name == "recover_both":
                        points[2] += [.015, -.01, .01]
                elif name == "clamp":
                    points[2] += [.8, .8, -.8]
                if name == "drop_both" or name == "drop_left" and side == "left":
                    points[:] = np.nan
                data[side + "_arm"].append(points)
                data[side + "_hand"].append(hand if name == "pose_fallback" else np.full((21, 3), np.nan))
                data["world_" + side + "_hand"].append(
                    (hand - hand[0]) * .19 if name == "pose_fallback" else np.full((21, 3), np.nan))
            labels.append(name)
    result = {key: np.stack(value) for key, value in data.items()}
    result.update(timestamps=np.arange(len(labels), dtype=float) / rate,
                  scenario=np.asarray(labels), metadata_json=np.asarray(json.dumps({
                      "schema": "synthetic_arm_validation_v1", "source": "synthetic", "seed": seed,
                      "sample_rate_hz": rate, "timestamp_unit": "seconds", "length_unit": "m",
                      "point_order": ["shoulder", "elbow", "wrist"], "intervals": intervals,
                  })))
    return result
