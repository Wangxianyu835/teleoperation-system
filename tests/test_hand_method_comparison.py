from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from teleoperation.contracts.hand_control import HandJointResult
from teleoperation.data.hand_joint_recording import HandJointResultWriter
from teleoperation.learning.hand_evaluation import compare_hand_methods
from teleoperation.retargeting.hand.hand_model import HandTargets, DifferentiableHandModel
from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
from teleoperation.robots.hand_spec import load_hand_spec


class HandMethodComparisonTests(unittest.TestCase):
    def test_common_valid_timestamps_raw_and_command_are_evaluated_separately(self):
        spec=load_hand_spec('l21','right'); specs={spec.key:spec}
        vectors=DifferentiableHandModel(spec).vectors(torch.tensor(spec.q_reference[None],dtype=torch.float32)).numpy()[0]
        with tempfile.TemporaryDirectory() as directory:
            files=[]
            for method in ('nn','dex'):
                path=Path(directory)/(method+'.h5'); files.append(path)
                commands=JointCommandPipeline(specs,filter_mode='none')
                with HandJointResultWriter(path,specs,{'backend':method,'input_sha256':'same','input_profile':'camera_palm_length_window3_v1','command_policy':commands.summary()}) as writer:
                    for i in range(5):
                        valid=i>=2 and not (method=='nn' and i==3)
                        raw=HandJointResult(*spec.key,spec.spec_hash,spec.independent_joint_names,
                                           spec.q_reference if valid else None,i/30,valid)
                        accepted=commands.update(raw,timestamp=i/30)
                        writer.append(i,i/30,{spec.key:raw},{spec.key:accepted},
                                      {spec.key:HandTargets(vectors,vectors*spec.reference_palm_length_m,i/30)})
            report=compare_hand_methods(files[0],files[1],specs)
            hand=report['targets']['l21/right']
            self.assertEqual(hand['common_valid_frames'],2)
            self.assertEqual(hand['nn']['invalid_frames'],3)
            self.assertLess(hand['nn']['raw']['tip_error_normalized_mean'],1e-6)
            self.assertEqual(hand['dex']['command']['speed_violations'],0)
            self.assertFalse(report['cross_operator_generalization_verified'])


if __name__=='__main__': unittest.main()
