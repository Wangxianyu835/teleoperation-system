from dataclasses import replace
import unittest
from unittest.mock import Mock

from teleoperation.contracts.hand_control import HandJointCommand
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.simulation.hand_executor import HandExecutor


class HandExecutorTests(unittest.TestCase):
    def test_resolves_exact_target_and_rejects_results_wrong_names_or_coupling(self):
        spec = load_hand_spec('h1_2','right')
        named = spec.expand(spec.q_reference)
        command = HandJointCommand(*spec.key,spec.spec_hash,spec.joint_names,[named[n] for n in spec.joint_names],0.)
        scene = Mock()
        executor = HandExecutor({spec.key:spec},{spec.key:scene.apply_native})
        executor.apply(command)
        scene.apply_native.assert_called_once_with(command)
        for bad in (replace(command,spec_hash='a'*64),replace(command,side='left'),replace(command,joint_names=tuple(reversed(command.joint_names)))):
            with self.assertRaises(ValueError): executor.apply(bad)
        q = command.positions.copy(); q[-1] += .1
        with self.assertRaises(ValueError): executor.apply(replace(command,positions=q))


if __name__ == '__main__': unittest.main()
