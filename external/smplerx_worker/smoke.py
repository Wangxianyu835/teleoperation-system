"""Real model smoke test; all diagnostics go to stderr and JSON to stdout."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository))
sys.path.insert(0, str(repository / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplerx-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--body-model-dir', type=Path, required=True)
    parser.add_argument('--input-image', type=Path, required=True)
    args = parser.parse_args()
    from external.smplerx_worker.adapter import SmplerXAdapter, SmplerXAssets
    adapter = SmplerXAdapter(SmplerXAssets(args.smplerx_root.resolve(), args.checkpoint.resolve(),
                                          args.body_model_dir.resolve()))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            import torch
            from teleoperation.contracts.body import BodyFrameId
            image = cv2.imread(str(args.input_image))
            if image is None:
                raise ValueError(f'Cannot read smoke input image: {args.input_image}')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            adapter.load()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for sequence in range(3):
                stamp = time.monotonic()
                result = adapter.infer(rgb, BodyFrameId('smoke', sequence, stamp))
                torch.cuda.synchronize()
                durations.append((time.monotonic() - stamp) * 1000)
            report = {'status': 'passed' if all(result.valid.values()) else 'no_valid_operator',
                      'reason': result.reason, 'model_hash': result.model_hash,
                      'input_shape': list(rgb.shape),
                      'output_shapes': {key: list(value.shape) for key, value in result.arrays.items()},
                      'frame_latency_ms': durations, 'warmup_frames': 1,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
        print(json.dumps(report, allow_nan=False))
        return 0 if report['status'] == 'passed' else 1
    except Exception as error:
        print(json.dumps({'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}))
        return 1
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            adapter.close()


if __name__ == '__main__':
    raise SystemExit(main())
