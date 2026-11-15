"""Vision Pro dex calibration and acceptance helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path


from teleoperation.data.hand_recording import iter_dex_observations
from teleoperation.retargeting.hand.dex_native import DexInputAdapter, DexCalibration


def canonical_window(frame):
    """Declare the explicit single-frame VR profile as a hand window."""
    from teleoperation.contracts.hand import HandWindow
    return HandWindow({side: None if points is None else points[None]
                       for side, points in frame.hands.items()}, frame.timestamp, frame.source, frame.metadata)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_frames(path, source):
    adapter = DexInputAdapter(source)
    return [adapter.adapt(raw) for raw in iter_dex_observations(path, source)]


def calibrate(args):
    if args.source != "visionpro":
        raise ValueError("Fixed dex calibration supports source visionpro only")
    frames = recording_frames(args.input, args.source)
    if not frames:
        raise ValueError("Empty calibration recording")
    calibration = DexCalibration.fit(frames, frames[0].timestamp + args.start, args.duration, file_hash(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(args.output)
    print(f"Saved fixed {calibration.profile} calibration: {args.output}")
    return 0
