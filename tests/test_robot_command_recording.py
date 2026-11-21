from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_output import OutputReceipt


def receipt(i,seconds,steps=8):
    return OutputReceipt(i,seconds,{}, {'world':steps},{'world':1/240})


def append_right_hand(writer, spec, i):
    frame = spec.reference_frame(i, i / 100.)
    writer.append({'g1': frame}, replace(receipt(i, frame.timestamp), statuses={'g1': 'submitted'}))


def interrupted_writer(path, stage):
    """Signal the parent at a real HDF5 operation and wait to be killed."""
    from teleoperation.data.robot_command_recording import RobotCommandWriter

    def ready():
        path.with_suffix('.ready').touch()
        time.sleep(60)
        raise TimeoutError(stage)

    spec = load_robot_control_spec('g1', resource_overrides={'parts': 'hands', 'side': 'right'})
    with RobotCommandWriter(path, {'g1': spec}, {}) as writer:
        write = h5py.Dataset.__setitem__
        flush = writer.stream.flush

        def on_write(dataset, index, value):
            if stage == 'reserved' and index == 0 and dataset is writer.frame_ids:
                ready()
            if dataset is writer.receipts and ((stage == 'partial' and index == 0)
                    or (stage == 'next' and index == writer.RESERVE_FRAMES)):
                ready()
            return write(dataset, index, value)

        def on_flush():
            if stage == 'before_flush':
                ready()
            flush()
            if stage == 'after_flush':
                ready()

        with patch.object(h5py.Dataset, '__setitem__', on_write), patch.object(writer.stream, 'flush', on_flush):
            for i in range(writer.RESERVE_FRAMES + 1):
                append_right_hand(writer, spec, i)


class RobotRecordingTests(unittest.TestCase):
    def test_interrupted_writer_never_publishes(self):
        from teleoperation.data.robot_command_recording import load_robot_recording
        from teleoperation import runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            for stage in ('reserved', 'partial', 'before_flush', 'after_flush', 'next'):
                with self.subTest(stage=stage):
                    path = Path(directory) / (stage + '.h5')
                    code = ('from tests.test_robot_command_recording import interrupted_writer, Path; '
                            'import sys; interrupted_writer(Path(sys.argv[1]), sys.argv[2])')
                    args = [sys.executable, '-E', '-s', runtime_environment.__file__, '-c', code, str(path), stage]
                    with subprocess.Popen(args) as child:
                        try:
                            deadline = time.monotonic() + 45
                            while not path.with_suffix('.ready').exists():
                                self.assertIsNone(child.poll(), 'child exited before handshake')
                                self.assertLess(time.monotonic(), deadline, 'handshake timed out')
                                time.sleep(.02)
                        finally:
                            child.kill()
                            child.wait(timeout=10)
                    self.assertFalse(path.exists())
                    pending, = path.parent.glob(path.name + '.pending-*.h5')
                    with self.assertRaises((OSError, ValueError)):
                        load_robot_recording(pending, ('g1',))

    def test_io_failures_preserve_complete_rows_and_prevent_publication(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter, load_robot_recording

        spec = load_robot_control_spec('g1', resource_overrides={'parts': 'hands', 'side': 'right'})
        with tempfile.TemporaryDirectory() as directory, patch.object(RobotCommandWriter, 'RESERVE_FRAMES', 2):
            for stage in ('grow', 'write', 'block_flush', 'trim', 'close_flush', 'close', 'publish', 'interrupt'):
                with self.subTest(stage=stage):
                    path = Path(directory) / (stage + '.h5')
                    writer = RobotCommandWriter(path, {'g1': spec}, {})
                    if stage == 'grow':
                        # The first extent succeeds before the second one fails.
                        with patch.object(writer.timestamps, 'resize', side_effect=OSError('grow')):
                            with self.assertRaisesRegex(OSError, stage):
                                append_right_hand(writer, spec, 0)
                        self.assertEqual(writer.capacity, 0)
                    else:
                        append_right_hand(writer, spec, 0)
                    if stage == 'write':
                        write = h5py.Dataset.__setitem__
                        def fail_write(dataset, index, value):
                            if dataset is writer.receipts:
                                raise OSError('write')
                            write(dataset, index, value)
                        with patch.object(h5py.Dataset, '__setitem__', fail_write), self.assertRaisesRegex(OSError, stage):
                            append_right_hand(writer, spec, 1)
                    elif stage == 'block_flush':
                        with patch.object(writer.stream, 'flush', side_effect=[OSError(stage), None]), self.assertRaisesRegex(OSError, stage):
                            append_right_hand(writer, spec, 1)
                    elif stage in ('trim', 'close_flush', 'close'):
                        obj, method = {'trim': (writer.timestamps, 'resize'), 'close_flush': (writer.stream, 'flush'),
                                       'close': (writer.stream, 'close')}[stage]
                        original = getattr(obj, method)
                        calls = 0
                        def fail_once(*args, **kwargs):
                            nonlocal calls
                            calls += 1
                            if calls == 1:
                                raise OSError(stage)
                            return original(*args, **kwargs)
                        with patch.object(obj, method, fail_once), self.assertRaisesRegex(OSError, stage):
                            writer.close()
                    elif stage == 'publish':
                        path.write_bytes(b'other publisher')
                        with self.assertRaises(FileExistsError):
                            writer.close()
                        self.assertEqual(path.read_bytes(), b'other publisher')
                    elif stage == 'interrupt':
                        with self.assertRaises(KeyboardInterrupt), writer:
                            raise KeyboardInterrupt('operator')
                    writer.close()
                    writer.close()
                    self.assertEqual(path.exists(), stage == 'publish')
                    with h5py.File(writer.temporary_path) as stream:
                        self.assertEqual(stream.attrs['recording_status'], 'failed')
                        if stage != 'trim':
                            self.assertEqual(len(stream['frame_ids']), {'grow': 0, 'block_flush': 2}.get(stage, 1))
                        self.assertTrue(stream.attrs['failure_reason'])
                    with self.assertRaises(ValueError):
                        load_robot_recording(writer.temporary_path, ('g1',))

    def test_reservation_flush_and_final_lengths(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter

        spec = load_robot_control_spec('g1', resource_overrides={'parts': 'hands', 'side': 'right'})
        initial = spec.reference_frame(0, 0.)
        block = RobotCommandWriter.RESERVE_FRAMES
        self.assertEqual(block, 1024)
        with tempfile.TemporaryDirectory() as directory:
            for count in (0, 1, 1023, 1024, 1025, 2048, 2050):
                path = Path(directory) / f'{count}.h5'
                with self.subTest(count=count), RobotCommandWriter(path, {'g1': spec}, {}) as writer:
                    flushed_counts = []
                    growth = []
                    flush, resize = writer.stream.flush, h5py.Dataset.resize
                    def record_flush():
                        flushed_counts.append(writer.count)
                        flush()
                    def record_resize(dataset, size, axis=None):
                        growth.append((dataset.name, size))
                        resize(dataset, size, axis)
                    with patch.object(writer.stream, 'flush', record_flush), patch.object(h5py.Dataset, 'resize', record_resize):
                        for i in range(count):
                            frame = replace(initial, frame_id=i, timestamp=i / 100)
                            r = replace(receipt(i, frame.timestamp), statuses={'g1': 'submitted'})
                            writer.append({'g1': frame}, r)
                        self.assertEqual(flushed_counts, list(range(block, count + 1, block)))
                        capacity = ((count + block - 1) // block) * block
                        self.assertTrue(all(ds.shape[0] == capacity for ds in writer.datasets))
                        self.assertEqual(growth, [(ds.name, extent) for extent in range(block, capacity + 1, block)
                                                  for ds in writer.datasets])
                with h5py.File(path) as stream:
                    def check_length(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            self.assertEqual(obj.shape[0], count, name)
                    stream.visititems(check_length)
                    np.testing.assert_array_equal(stream['frame_ids'][:], np.arange(count))
                    self.assertEqual(stream.attrs['recording_status'], 'complete')

    def test_target_metadata_round_trip_and_v1_without_invented_target_identity(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter,load_robot_recording
        spec=load_robot_control_spec('g1',resource_overrides={'parts':'hands','side':'right'})
        frame=spec.reference_frame(0,.2)
        part=replace(frame.parts['right_hand'],source_timestamp=123.,source_timebase='body_s',
                     observed=True,status='tracking',target_sequence=19,target_expires_time=.5,source_is_new=False)
        batch={'g1':replace(frame,parts={'right_hand':part})}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'commands.h5'
            with RobotCommandWriter(path,{'g1':spec},{}) as writer:
                writer.append(batch,replace(receipt(0,.2),statuses={'g1':'submitted'}))
            restored=next(load_robot_recording(path,('g1',)).iter_frames())[0]['g1'].parts['right_hand']
            self.assertEqual(restored.target_sequence,19)
            self.assertEqual(restored.target_expires_time,.5)
            self.assertFalse(restored.source_is_new)
            with h5py.File(path,'r+') as stream:
                self.assertEqual(stream.attrs['schema_version'],2)
                stream.attrs['schema']='robot_command_results_v1'
                del stream.attrs['schema_version']
                for field in ('target_sequence','target_expires_time','source_is_new'):
                    del stream['robots/g1/parts/right_hand'][field]
            restored=next(load_robot_recording(path,('g1',)).iter_frames())[0]['g1'].parts['right_hand']
            self.assertIsNone(restored.target_sequence)
            self.assertIsNone(restored.target_expires_time)
            self.assertTrue(restored.source_is_new)

    def test_failed_dispatch_keeps_receipt_in_unpublished_recording(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter
        spec=load_robot_control_spec('g1',resource_overrides={'parts':'hands','side':'right'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'commands.h5'
            writer=RobotCommandWriter(path,{'g1':spec},{})
            r=replace(receipt(0,0.),statuses={'g1':'failed'},steps={'world':None})
            writer.append({'g1':spec.reference_frame(0,0.)},r)
            writer.close()
            self.assertFalse(path.exists())
            with h5py.File(writer.temporary_path) as stream:
                import json
                saved=json.loads(stream['execution/receipts'][0])
                self.assertEqual(saved['statuses'],{'g1':'failed'})
                self.assertEqual(saved['steps'],{'world':None})
                self.assertEqual(stream.attrs['recording_status'],'failed')

    def test_all_layouts_round_trip_source_clocks_and_holding(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter,load_robot_recording
        specs={n:load_robot_control_spec(n) for n in ('h1_2','gr1_t2','g1','tron2a_l21')}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'commands.h5'
            with RobotCommandWriter(path,specs,{'geometry':'simplified'}) as writer:
                for i in range(3):
                    batch={n:replace(s.reference_frame(i,i/30),parts={key:replace(p,
                        source_timestamp=100.+i/30 if key.endswith('_hand') else 99.+i/30,
                        source_timebase='hands_s' if key.endswith('_hand') else 'body_s',
                        status='holding' if i==1 else 'tracking',observed=i!=1)
                        for key,p in s.reference_frame(i,i/30).parts.items()}) for n,s in specs.items()}
                    r=receipt(i,i/30,7+i)
                    r=replace(r,statuses={n:'submitted' for n in specs})
                    writer.append(batch,r)
                self.assertFalse(path.exists())
            recording=load_robot_recording(path,('g1',))
            rows=list(recording.iter_frames())
            self.assertEqual(len(rows),3)
            self.assertEqual(list(rows[0][0]),['g1'])
            frame=rows[1][0]['g1']
            self.assertEqual(frame.parts['left_arm'].status,'holding')
            self.assertEqual(frame.parts['left_arm'].source_timebase,'body_s')
            self.assertAlmostEqual(frame.parts['right_hand'].source_timestamp,100+1/30)
            self.assertEqual(rows[2][1].steps,{'world':9})
            with h5py.File(path) as f:
                self.assertEqual(f.attrs['recording_status'],'complete')
                self.assertEqual(f.attrs['angle_unit'],'rad')
            with self.assertRaises(FileExistsError):RobotCommandWriter(path,specs,{})

    def test_abort_invalid_and_tampered_commands_never_replay(self):
        from teleoperation.data.robot_command_recording import RobotCommandWriter,load_robot_recording
        spec=load_robot_control_spec('g1',resource_overrides={'parts':'hands','side':'right'})
        specs={'g1':spec}; batch={'g1':spec.reference_frame(0,0.)}
        r=replace(receipt(0,0.),statuses={'g1':'submitted'})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'commands.h5'
            writer=RobotCommandWriter(path,specs,{})
            with self.assertRaises(ValueError):writer.append({},r)
            writer.append(batch,r)
            with self.assertRaises(ValueError):writer.append(batch,r)
            writer.abort('capture lost'); writer.close()
            self.assertFalse(path.exists())
            with h5py.File(writer.temporary_path) as f:self.assertEqual(f.attrs['recording_status'],'failed')
            with self.assertRaises(ValueError):load_robot_recording(writer.temporary_path,('g1',))
            with RobotCommandWriter(path,specs,{}) as writer:writer.append(batch,r)
            with h5py.File(path,'r+') as f:f['robots/g1/parts/right_hand/positions'][0,0]=np.nan
            with self.assertRaises(ValueError):load_robot_recording(path,('g1',))
