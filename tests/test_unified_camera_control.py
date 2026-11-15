"""Run recorded raw poses through real PINK, motor output and command recording."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from teleoperation.cli import main
from teleoperation.contracts.body import BodyFrameId, SmplerXResult
from teleoperation.contracts.observations import CameraFrame,RawHandFrame,UpperBodyObservation,RawTeleopObservation
from teleoperation.data.body_recording import capture_sample,save_body_recording
from tests import test_arm_pipeline as fixtures


class UnifiedCameraControlTests(unittest.TestCase):
    def test_pump_preserves_pending_body_identity_while_hands_advance(self):
        import threading
        from teleoperation.applications.unified_camera_control import ObservationPump
        inferred=fixtures.result(7,stamp=100.)
        body=UpperBodyObservation({'left':None,'right':None},{'left':False,'right':False})
        first=RawTeleopObservation(RawHandFrame({'left':None,'right':None},100.,'camera'),body,body_result=inferred)
        latest=RawTeleopObservation(RawHandFrame({'left':None,'right':None},100.1,'camera'),body)
        class Source:
            def __init__(self):
                self.samples=iter((first,latest));self.drained=threading.Event();self.released=threading.Event()
            def next_teleop_observation(self):
                try:return next(self.samples)
                except StopIteration:
                    self.drained.set()
                    self.released.wait(5.)
                    raise StopIteration
            def release(self):self.released.set()
        source=Source();pump=ObservationPump(source)
        try:
            self.assertTrue(source.drained.wait(2.))
            taken=pump.take()
            self.assertIs(taken.hands,latest.hands)
            self.assertEqual(taken.hands.timestamp,100.1)
            self.assertIs(taken.body_result,inferred)
            self.assertEqual(taken.body_result.frame_id.sequence,7)
            self.assertEqual(taken.body_result.frame_id.capture_time_s,100.)
            self.assertIsNone(pump.take())
        finally:
            pump.close()
        self.assertTrue(source.released.is_set())
        self.assertFalse(pump.thread.is_alive())

    def test_recorded_poses_run_through_continuous_control_and_replay(self):
        pipeline,body=fixtures.ArmPipelineTests().make_pipeline()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            # Only profile loading/reconstruction resource selection is replaced.
            # PINK, assembler, output, URDF, physical stepping and H5 are real.
            (directory/'h1_2.json').write_text('{}')
            rows=[]
            points=body.points
            for index in range(4):
                stamp=100.+index/30
                result=SmplerXResult(BodyFrameId('test',index,stamp),stamp+.001,'synthetic','synthetic',
                    {'smplx_root_pose':np.zeros(3),'smplx_body_pose':np.zeros(63)}, {'left':True,'right':True})
                arms={side:np.array([points[side+'_'+j] for j in ('shoulder','elbow','wrist')]) for side in ('left','right')}
                raw=RawTeleopObservation(RawHandFrame({'left':None,'right':None},stamp,'mediapipe_approx'),
                    UpperBodyObservation(arms,{'left':True,'right':True},stamp,'smplx_image_normalized','normalized',
                        world_arms=arms,metadata={'source':'smplerx'}),body_result=result)
                source=SimpleNamespace(session_id='test',last_frame=CameraFrame(np.zeros((8,8,3),np.uint8),stamp,index),last_result=result)
                rows.append(capture_sample(source,raw))
            recording=directory/'body.h5'
            save_body_recording(recording,rows,metadata={'source':'smplerx','model_id':'synthetic','model_hash':'synthetic','time_unit':'monotonic_s'})
            def build(args,specs,backends):
                actual=backends['h1_2'].read_state()
                pipeline.solvers['h1_2'].reset({n:q for key,part in actual.items() for n,q in zip(specs['h1_2'].parts[key].joint_names,part)})
                return pipeline
            report=directory/'report.json'
            commands=directory/'commands.h5'
            with patch('teleoperation.applications.unified_camera_control.build_arm_pipeline',side_effect=build):
                self.assertEqual(main(['robot','control','--robot','h1_2','--parts','arms','--input-body',str(recording),
                    '--arm-fitting-dir',str(directory),'--headless','--record-results',str(commands),'--report',str(report)]),0)
            import json
            data=json.loads(report.read_text())
            self.assertEqual(data['body_reconstructions'],4)
            self.assertGreater(data['frames'],data['observations'])
            self.assertEqual(data['arm_method'],'smpl_segment_pink_v1')
            with patch('teleoperation.retargeting.arm.pink_ik.PinkArmIK',side_effect=AssertionError('replay constructed IK')):
                self.assertEqual(main(['robot','replay','--robot','h1_2','--input',str(commands),'--headless']),0)


if __name__=='__main__':unittest.main()
