"""Dispatch accepted named commands to an explicitly bound target hand."""
from teleoperation.contracts.hand_control import HandJointCommand


class HandExecutor:
    def __init__(self, specs, bindings):
        self.specs, self.bindings = dict(specs), dict(bindings)
        if not self.specs or set(self.specs) != set(self.bindings):
            raise ValueError('Executor needs one binding for every target specification')

    def apply(self, command):
        if not isinstance(command,HandJointCommand) or command.key not in self.specs:
            raise ValueError('Executor requires an accepted command for an active target')
        spec = self.specs[command.key]
        if command.spec_hash != spec.spec_hash or command.joint_names != spec.joint_names:
            raise ValueError('Executor command target specification mismatch')
        named = command.named_positions()
        expanded = spec.expand([named[n] for n in spec.independent_joint_names])
        if any(abs(named[n]-expanded[n]) > 1e-6 for n in spec.joint_names):
            raise ValueError('Executor command violates target hand coupling')
        self.bindings[command.key](command)
