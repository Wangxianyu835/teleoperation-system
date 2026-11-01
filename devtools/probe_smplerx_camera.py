"""Observe camera -> detector -> raw S32 output -> application result, without robots.

The launcher uses the existing isolated worker interpreter. Observation hooks
live only in this process; the backend, thresholds and upstream are unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def array_report(arrays):
    import numpy as np
    report = {}
    for key, value in arrays.items():
        a = np.asarray(value)
        finite = bool(np.isfinite(a).all())
        report[key] = dict(shape=list(a.shape), dtype=str(a.dtype), finite=finite,
                           min=float(a.min()) if a.size and finite else None,
                           max=float(a.max()) if a.size and finite else None)
    return report


def detection_report(boxes):
    import numpy as np
    from external.smplerx_worker.operator_state import PERSON_SCORE_THRESHOLD
    boxes = np.asarray(boxes).reshape(-1, 5)
    entries = []
    for x1, y1, x2, y2, score in boxes:
        reasons = []
        if score < PERSON_SCORE_THRESHOLD:
            reasons.append(f'score_below_{PERSON_SCORE_THRESHOLD:g}')
        if x2 - x1 < 50:
            reasons.append('width_below_50px')
        if y2 - y1 < 150:
            reasons.append('height_below_150px')
        entries.append(dict(xyxy=[float(x1), float(y1), float(x2), float(y2)],
                            score=float(score), accepted=not reasons, rejected_by=reasons))
    return dict(raw_count=len(entries), accepted_count=sum(x['accepted'] for x in entries),
                boxes=entries)


def classify(valid, reason, accepted_count, network_ran):
    if valid:
        return 'valid_body'
    if 'expired' in reason:
        return 'input_expired'
    if 'identity uncertain' in reason:
        return 'operator_rejected'
    if 'operator ambiguous' in reason:
        return 'operator_ambiguous'
    if 'operator recovering' in reason:
        return 'operator_recovering'
    if network_ran and 'calibrating body shape' in reason:
        return 'shape_initializing'
    if accepted_count == 0:
        return 'no_accepted_person'
    return 'result_rejected'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def parser():
    from teleoperation.cli.smplerx_options import add_smplerx_arguments
    p = argparse.ArgumentParser(description=__doc__)
    add_smplerx_arguments(p)
    p.add_argument('--camera-id', type=int, default=0)
    p.add_argument('--camera-backend', choices=('auto', 'dshow', 'msmf'), default='auto')
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--input-image', type=Path, help='repeat a saved image instead of opening a camera')
    p.add_argument('--preview', action='store_true', help='show saved frame with boxes and arm projections; Q quits')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / ('smplerx_probe_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--isolated', action='store_true', help=argparse.SUPPRESS)
    return p


def run(args):
    import cv2
    import numpy as np
    from external.smplerx_worker.adapter import SmplerXAdapter, SmplerXAssets
    from teleoperation.contracts.body import BodyFrameId

    output = args.output_dir
    report = dict(status='starting', mode='image' if args.input_image else 'camera',
                  python=sys.executable, camera_id=args.camera_id, frames=[],
                  note='Direct existing backend in isolated worker environment; IPC/tracker/IK not exercised.')
    adapter = SmplerXAdapter(SmplerXAssets(args.smplerx_root, args.smplerx_checkpoint,
                                         args.smplx_model_dir))
    cap, hook = None, None
    observed = {}
    try:
        report['stage'] = 'model_load'
        print('Loading S32 and detector before capturing frames...', flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            adapter.load()
        backend = adapter._backend
        report['model_hash'] = backend.model_hash
        original_detect = backend.detect

        def detect(*a, **kw):
            result = original_detect(*a, **kw)
            observed['detections'] = detection_report(result[0])
            return result

        def forward_hook(module, inputs, out):
            observed['raw'] = {key: value.detach().cpu().numpy().copy()
                               for key, value in out.items() if backend.torch.is_tensor(value)}

        backend.detect = detect
        hook = backend.model.register_forward_hook(forward_hook)
        report['stage'] = 'camera_open' if not args.input_image else 'image_read'
        still = None
        if args.input_image:
            still = cv2.imread(str(args.input_image))
            if still is None:
                raise RuntimeError(f'Cannot read image: {args.input_image}')
        else:
            api = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}[args.camera_backend]
            cap = cv2.VideoCapture(args.camera_id, api)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera {args.camera_id} ({args.camera_backend})')
            report['camera_backend'] = cap.getBackendName()

        for sequence in range(args.frames):
            observed.clear()
            report['stage'] = 'capture'
            ok, bgr = cap.read() if cap is not None else (True, still.copy())
            if not ok or bgr is None:
                raise RuntimeError('Camera frame read failed')
            stamp = time.monotonic()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_id = BodyFrameId('camera-probe', sequence, stamp)
            folder = output / f'{sequence:05d}'
            folder.mkdir()
            row = dict(frame_id=asdict(frame_id), input=array_report({'rgb': rgb})['rgb'])
            report['stage'] = 'inference'
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = adapter.infer(rgb, frame_id)
                row.update(valid=dict(result.valid), reason=result.reason,
                           completed_time_s=result.completed_time_s,
                           metadata=result.metadata, arrays=array_report(result.arrays))
                np.savez_compressed(folder / 'result.npz', **result.arrays)
                row['diagnosis'] = classify(all(result.valid.get(s, False) for s in ('left', 'right')),
                                            result.reason, observed.get('detections', {}).get('accepted_count'),
                                            'raw' in observed)
            finally:
                row['elapsed_ms'] = (time.monotonic() - stamp) * 1000
                row['detections'] = observed.get('detections')
                row['network_ran'] = 'raw' in observed
                row['raw_output'] = array_report(observed.get('raw', {}))
                if 'raw' in observed:
                    np.savez_compressed(folder / 'raw_model.npz', **observed['raw'])
                if not cv2.imwrite(str(folder / 'input.png'), bgr):
                    raise OSError('Cannot save input frame')
                write_json(folder / 'frame.json', row)
                report['frames'].append(row)
            canvas = bgr.copy()
            for box in (row['detections'] or {}).get('boxes', []):
                if box['score'] < .1:
                    continue
                x1, y1, x2, y2 = map(int, box['xyxy'])
                color = (0, 255, 0) if box['accepted'] else (0, 128, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"person {box['score']:.3f}", (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            if 'image_joints' in result.arrays:
                points = result.arrays['image_joints']
                for indices in ((16, 18, 20), (17, 19, 21)):
                    xy = [(int(points[i, 0] * bgr.shape[1]), int(points[i, 1] * bgr.shape[0])) for i in indices]
                    for a, b in zip(xy, xy[1:]):
                        cv2.line(canvas, a, b, (255, 0, 255), 2)
            cv2.putText(canvas, row['diagnosis'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if not cv2.imwrite(str(folder / 'overlay.png'), canvas):
                raise OSError('Cannot save overlay')
            print(f"[{sequence + 1}/{args.frames}] {row['diagnosis']} "
                  f"accepted={(row['detections'] or {}).get('accepted_count')} "
                  f"network={row['network_ran']} valid={row['valid']} "
                  f"{row['elapsed_ms']:.0f}ms reason={row['reason']}", flush=True)
            write_json(output / 'report.json', report)
            if args.preview:
                cv2.imshow('SMPLer-X camera probe (Q quits)', canvas)
                if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                    break
        valid_count = sum(row.get('diagnosis') == 'valid_body' for row in report['frames'])
        report.update(status='passed' if valid_count else 'no_valid_body', valid_frames=valid_count,
                      network_frames=sum(row['network_ran'] for row in report['frames']), stage='complete')
        return 0 if valid_count else 1
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        if hook is not None:
            hook.remove()
        if cap is not None:
            cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        with contextlib.redirect_stdout(sys.stderr):
            adapter.close()
        write_json(output / 'report.json', report)
        print(f"{report['status']}: {output / 'report.json'}", flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.frames < 1:
        p.error('--frames must be positive')
    for name in ('smplerx_worker_python', 'smplerx_root', 'smplerx_checkpoint', 'smplx_model_dir', 'output_dir', 'input_image'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.isolated:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return run(args)
    # Never overwrite an earlier diagnostic run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [str(args.smplerx_worker_python), str(Path(__file__).resolve()),
               *(sys.argv[1:] if argv is None else argv), '--isolated',
               '--output-dir', str(args.output_dir)]
    try:
        with (args.output_dir / 'worker.stderr.log').open('w', encoding='utf-8') as log:
            return subprocess.call(command, cwd=ROOT, stderr=log)
    except OSError as error:
        write_json(args.output_dir / 'report.json', dict(status='failed', stage='worker_launch', error=str(error)))
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
