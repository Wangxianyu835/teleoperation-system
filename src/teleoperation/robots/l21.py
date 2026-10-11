"""Original L21 eighteen-value protocol and actuator names."""
MAPPING_18 = [
    None,                     # dim 0  占位
    'index_mcp_roll',         # 1
    'index_mcp_pitch',        # 2
    'index_pip',              # 3
    'middle_mcp_roll',        # 4
    'middle_mcp_pitch',       # 5
    'middle_pip',             # 6
    'ring_mcp_roll',          # 7
    'ring_mcp_pitch',         # 8
    'ring_pip',               # 9
    'pinky_mcp_roll',         # 10
    'pinky_mcp_pitch',        # 11
    'pinky_pip',              # 12
    'thumb_cmc_roll',         # 13
    'thumb_cmc_yaw',          # 14
    'thumb_cmc_pitch',        # 15
    'thumb_mcp',              # 16
    'thumb_ip',               # 17
]

L21_JOINT_ORDER = [
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip',
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip',
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip',
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip',
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
]


class L21HandAdapter:
    def __init__(self, joints_order, limits, default_limits=None):
        self.joints_order, self.limits, self.default_limits = joints_order, limits, default_limits

    def map(self, dofs):
        from teleoperation.contracts.hand import L21HandDOFs
        if not isinstance(dofs, L21HandDOFs): raise TypeError("L21 hand mapping requires L21HandDOFs")
        targets = {}
        for index, name in enumerate(self.joints_order):
            if not name: continue
            value = float(dofs.values[index - 1])
            bounds = self.limits.get(name, self.default_limits)
            if bounds is not None:
                lo, hi = bounds
                value = min(max(value, lo), hi)
            targets[name] = value
        return targets
