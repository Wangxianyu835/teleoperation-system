import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from teleoperation.contracts.hand_control import HandJointResult
from teleoperation.data.hand_joint_recording import HandJointResultWriter, iter_joint_commands
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.robots.hand_spec import load_hand_spec


class HandJointRecordingTests(unittest.TestCase):
    def test_independent_raw_and_expanded_command_round_trip_and_strict_replay(self):
        spec = load_hand_spec('h1_2','right')
        specs = {spec.key:spec}
        commands = JointCommandPipeline(specs,filter_mode='none')
        raw = HandJointResult(*spec.key,spec.spec_hash,spec.independent_joint_names,spec.q_reference,1.)
        accepted = commands.update(raw,timestamp=1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results.h5'
            with HandJointResultWriter(path,specs,{'method':'nn','input_profile':'test'}) as writer:
                writer.append(0,1.,{spec.key:raw},{spec.key:accepted})
            replayed = list(iter_joint_commands(path,specs))
            np.testing.assert_array_equal(replayed[0][1][spec.key].positions,accepted.positions)
            with h5py.File(path,'r+') as stream:
                self.assertEqual(stream.attrs['schema'],'hand_joint_results_v1')
                self.assertEqual(stream['h1_2/right/raw_positions'].shape,(1,6))
                self.assertEqual(stream['h1_2/right/command_positions'].shape,(1,12))
                stream['h1_2/right'].attrs['spec_hash']='a'*64
            with self.assertRaises(ValueError): list(iter_joint_commands(path,specs))


if __name__ == '__main__': unittest.main()
