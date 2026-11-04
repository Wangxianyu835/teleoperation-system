from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from teleoperation.robots.control_spec import load_robot_control_spec
from teleoperation.simulation.robot_output import OutputReceipt


def receipt(i,seconds,steps=8):
    return OutputReceipt(i,seconds,{}, {'world':steps},{'world':1/240})


class RobotRecordingTests(unittest.TestCase):
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
