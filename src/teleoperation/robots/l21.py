"""L21 native independent actuator order and bounded named mapping."""
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
        import numpy as np
        values = np.asarray(dofs)
        if values.shape != (17,) or not np.isfinite(values).all():
            raise ValueError("L21 hand mapping requires 17 finite independent coordinates")
        targets = {}
        for index, name in enumerate(self.joints_order):
            if not name: continue
            value = float(values[index])
            bounds = self.limits.get(name, self.default_limits)
            if bounds is not None:
                lo, hi = bounds
                value = min(max(value, lo), hi)
            targets[name] = value
        return targets
