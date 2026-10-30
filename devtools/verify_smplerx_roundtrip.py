"""Real subprocess result -> tracker -> H5 -> replay equivalence and overlay."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=40, help='include five shape initialization frames and thirty neutral samples')
    args = parser.parse_args()
    if args.frames < 6:
        parser.error('--frames must be at least six')
    import cv2
    import numpy as np
    from teleoperation.cli.smplerx_options import smplerx_defaults
    from teleoperation.inputs.smplerx_process import SmplerXWorkerTransport
    from teleoperation.inputs.smplerx import SmplerXClient, encode_packet
    from teleoperation.inputs.body_tracking import BodyTracker
    from teleoperation.contracts.body import BodyFrameId
    from teleoperation.contracts.observations import CameraFrame, RawHandFrame, RawTeleopObservation
    from teleoperation.data.body_recording import capture_sample, save_body_recording, load_body_recording, replay_observation
    from types import SimpleNamespace
    options = smplerx_defaults()
    transport = SmplerXWorkerTransport(worker_python=options['smplerx_worker_python'],
        smplerx_root=options['smplerx_root'], checkpoint=options['smplerx_checkpoint'],
        body_model_dir=options['smplx_model_dir'])
    client = SmplerXClient(transport, session_id='roundtrip')
    rows, results, tracker = [], [], BodyTracker()
    try:
        bgr = cv2.imread(str(args.input_image))
        if bgr is None:
            raise ValueError('Cannot read input image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.monotonic()
        for sequence in range(args.frames):
            stamp = time.monotonic()
            client.submit(BodyFrameId('roundtrip', sequence, stamp), rgb)
            deadline, result = time.monotonic() + 120., None
            while result is None:
                result = client.poll()
                if time.monotonic() > deadline:
                    raise TimeoutError('Worker result timeout')
                if result is None:
                    time.sleep(.002)
            tracked = tracker.update(result, now_s=time.monotonic())
            raw = RawTeleopObservation(RawHandFrame({'left': None, 'right': None}, stamp, 'verification_no_hands'),
                                      tracked.body, tracked.wrists)
            source = SimpleNamespace(session_id='roundtrip', last_frame=CameraFrame(rgb, stamp, sequence), last_result=result)
            rows.append(capture_sample(source, raw))
            results.append(raw)
        if not all(results[-1].upper_body.valid.values()):
            raise RuntimeError(f'Real worker produced no accepted bilateral body result: {result.reason}; '
                               f'{tracked.state_by_side}; {tracked.reason_by_side}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = save_body_recording(args.output_dir / 'body.h5', rows,
            dict(source='smplerx', model_id=result.model_id, model_hash=result.model_hash, time_unit='monotonic_s',
                 verification_source='repeated_static_image_no_hands'))
        restored = load_body_recording(path)
        for original, row in zip(results, restored.samples):
            replayed = replay_observation(row)
            for side in ('left', 'right'):
                a,b = original.upper_body.world_arms[side], replayed.upper_body.world_arms[side]
                if a is not None:
                    np.testing.assert_array_equal(a,b)
            if encode_packet('observation', {'body': vars(original.upper_body)}) != encode_packet('observation', {'body': vars(replayed.upper_body)}):
                raise AssertionError('Body observation changed during persistence')
        canvas = bgr.copy()
        points = result.arrays['image_joints']
        for side, indices in (('left',(16,18,20)),('right',(17,19,21))):
            xy = [(int(points[i,0]*canvas.shape[1]), int(points[i,1]*canvas.shape[0])) for i in indices]
            for a,b in zip(xy,xy[1:]):
                cv2.line(canvas,a,b,(0,255,0),2)
            for name,point in zip(('S','E','W'),xy):
                cv2.circle(canvas,point,4,(0,0,255),-1)
                cv2.putText(canvas,side[0]+name,point,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
        cv2.imwrite(str(args.output_dir / 'projection.png'),canvas)
        report = dict(status='passed', samples=len(rows), model_hash=result.model_hash,
                      body_roundtrip_max_error_m=0., total_seconds=time.monotonic()-started,
                      raw_model_and_rgb_preserved=True, live_hands_exercised=False, real_camera_verified=False)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        client.close()


if __name__ == '__main__':
    main()
