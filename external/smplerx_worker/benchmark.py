"""Sequential single-model benchmark. Static-image runs are not camera acceptance."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    parser = argparse.ArgumentParser()
    for name in ('smplerx-root', 'checkpoint', 'body-model-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-video', type=Path)
    inputs.add_argument('--input-image', type=Path)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--duration-s', type=float, default=0.)
    args = parser.parse_args()
    if args.samples < 1 or args.duration_s < 0:
        parser.error('samples must be positive; duration must be nonnegative')
    from external.smplerx_worker.adapter import SmplerXAdapter, SmplerXAssets
    from teleoperation.contracts.body import BodyFrameId
    adapter = SmplerXAdapter(SmplerXAssets(args.smplerx_root.resolve(), args.checkpoint.resolve(), args.body_model_dir.resolve()))
    capture = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            import numpy as np
            import torch
            adapter.load()
            image = None
            if args.input_video:
                capture = cv2.VideoCapture(str(args.input_video))
                if not capture.isOpened():
                    raise ValueError('Cannot open benchmark video')
            else:
                image = cv2.imread(str(args.input_image))
                if image is None:
                    raise ValueError('Cannot read benchmark image')
            durations, valid, reasons, memory = [], [], {}, []
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            measured_start = None
            sequence = 0
            while len(durations) < args.samples or (measured_start is not None and time.monotonic() - measured_start < args.duration_s):
                stamp = time.monotonic()
                if capture is not None:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                else:
                    bgr = image
                result = adapter.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), BodyFrameId('benchmark', sequence, stamp))
                torch.cuda.synchronize()
                sequence += 1
                if sequence <= 5:
                    continue
                if measured_start is None:
                    measured_start = stamp
                durations.append((time.monotonic() - stamp) * 1000.)
                valid.append(all(result.valid.values()))
                if result.reason:
                    reasons[result.reason] = reasons.get(result.reason, 0) + 1
                memory.append(torch.cuda.memory_allocated())
            elapsed = 0. if measured_start is None else time.monotonic() - measured_start
            rate = sum(valid) / elapsed if elapsed else 0.
            p95 = float(np.percentile(durations, 95)) if durations else None
            report = dict(status='passed' if durations else 'no_samples', scope='single_model_sequential',
                source='video' if capture is not None else 'repeated_static_image',
                real_camera_verified=False, motion_accuracy_verified=False,
                model_hash=result.model_hash if sequence else None, samples=len(durations), reliable_results=sum(valid),
                reliable_new_result_hz=rate, hold_fraction=1. - sum(valid)/len(valid) if valid else None,
                latency_ms=dict(p50=float(np.median(durations)) if durations else None, p95=p95),
                measured_seconds=elapsed, total_seconds=time.monotonic()-started, warmup_frames=5,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_growth_bytes=memory[-1]-memory[0] if memory else None, invalid_reasons=reasons,
                device=torch.cuda.get_device_name(), torch=torch.__version__,
                engineering_target_met=bool(len(durations)>=300 and rate>=10 and p95 is not None and p95<=150))
        print(json.dumps(report, allow_nan=False))
        return 0 if durations else 1
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            if capture is not None:
                capture.release()
            adapter.close()


if __name__ == '__main__':
    raise SystemExit(main())
