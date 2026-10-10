"""Camera -> raw two-hand H5 -> palm-local H5 for hand train/export.

Run ``python -m teleoperation hand record --frames 900 --hand-side left``.
With --frames 0, q/Esc, closing the preview or Ctrl+C finishes and aligns.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from teleoperation.applications.hand_align import align_h5
from teleoperation.data.hand_recording import RawHandH5Writer
from teleoperation.inputs.mediapipe import MediaPipeCameraInput


def _output_paths(args, started_at):
    stem = f"visual_hand_data_{args.hand_side}_{started_at:%Y%m%d_%H%M%S}"
    raw = (Path(args.output) if args.output is not None else
           Path(args.output_dir) / f"{stem}.h5")
    aligned = (Path(args.aligned_output) if args.aligned_output is not None else
               raw.with_name(f"aligned_{raw.name}"))
    # Automatic names may collide when two captures start in the same second.
    if args.output is None:
        index = 1
        while raw.exists() or (args.aligned_output is None and aligned.exists()):
            raw = Path(args.output_dir) / f"{stem}_{index:02d}.h5"
            if args.aligned_output is None:
                aligned = raw.with_name(f"aligned_{raw.name}")
            index += 1
    if raw.resolve() == aligned.resolve():
        raise ValueError("Raw and aligned output paths must be different")
    for path in (raw, aligned):
        if path.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError(f"Output must be an H5 file: {path}")
        if path.exists():
            raise FileExistsError(f"Recording output already exists: {path}")
    return raw, aligned


def run(args):
    if args.frames < 0:
        raise ValueError("--frames must be nonnegative; 0 records until Ctrl+C")
    if min(args.width, args.height, args.fps) <= 0:
        raise ValueError("--width, --height and --fps must be positive")
    if args.hand_side not in ("left", "right", "both"):
        raise ValueError("--hand-side must be left, right or both")
    started_at = datetime.now(timezone(timedelta(hours=8)))
    raw_path, aligned_path = _output_paths(args, started_at)
    capture = MediaPipeCameraInput(
        model_asset_path=args.model_asset_path, camera_index=args.camera_index,
        width=args.width, height=args.height, fps=args.fps,
        invalid_hand_as_missing=True,
        include_image=not args.no_preview,
    )
    failure = None
    preview = None
    try:
        if not args.no_preview:
            from teleoperation.tools.recording_preview import RecordingPreview
            preview = RecordingPreview(args.hand_side)
        with RawHandH5Writer(raw_path, {
            "camera_index": args.camera_index,
            "requested_width": args.width, "requested_height": args.height,
            "requested_fps": args.fps, "model_asset_path": str(args.model_asset_path),
            "recording_hand_side": args.hand_side,
            "recording_started_at": started_at.isoformat(timespec="seconds"),
            "recording_timezone": "Asia/Shanghai",
        }) as writer:
            print(f"Recording raw hands to {raw_path}", flush=True)
            print("Ctrl+C stops capture and creates the aligned H5.", flush=True)
            if preview is not None:
                print("Preview: q/Esc or close the window to stop and save.", flush=True)
            stop_reason = "frame_limit"
            try:
                while args.frames == 0 or writer.count < args.frames:
                    frame = capture.next_frame()
                    hands = {side: frame[side] if args.hand_side in (side, "both") else None
                             for side in ("left", "right")}
                    writer.append(hands, frame["timestamp"] / 1000.0)
                    if writer.count == 1:
                        # Persist actual dimensions, which may differ from the request.
                        for name in ("image_width", "image_height"):
                            if name in frame.get("metadata", {}):
                                writer.set_metadata(name, frame["metadata"][name])
                    if writer.count % 30 == 0:
                        print(f"frames={writer.count} detected={writer.hand_counts}", flush=True)
                    if preview is not None:
                        preview_stop = preview.show(frame, writer.count)
                        if preview_stop is not None:
                            stop_reason = preview_stop
                            break
            except KeyboardInterrupt:
                stop_reason = "keyboard_interrupt"
                print("\nCapture stopped; saving recording.", flush=True)
            except (RuntimeError, ValueError, OSError) as error:
                stop_reason = "error"
                failure = error
            writer.finish(stop_reason)
    finally:
        try:
            capture.close()
        finally:
            if preview is not None:
                preview.close()

    print(f"raw_output={raw_path}")
    print(f"frames={writer.count} detected={writer.hand_counts}")
    if failure is not None:
        print(f"Recording failed: {failure}. Saved raw frames remain at {raw_path}.")
        return 1
    if writer.count == 0:
        print("No frames captured; no aligned file was created.")
        return 1
    # A second recording may have created this path while capture was running.
    if aligned_path.exists():
        raise FileExistsError(f"Aligned output already exists; raw recording retained: {aligned_path}")
    report = align_h5(raw_path, aligned_path)
    print(f"aligned_output={aligned_path}")
    print("coordinate_alignment=palm_local_to_l21_v1")
    for side, statistics in report["sides"].items():
        print(f"{side}_aligned_frames={statistics['nonzero_frames']}")
    return 0


if __name__ == "__main__":
    import sys
    from teleoperation.cli import main

    raise SystemExit(main(["hand", "record", *sys.argv[1:]]))
