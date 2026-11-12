"""Shared native L21 hand factory; legacy arm baselines live in devtools."""

def load_camera_hand_retargeter(checkpoint, device="cpu"):
    """Load only the canonical shared native L21 camera checkpoint."""
    from teleoperation.robots.hand_spec import load_hand_spec
    from .hand_backend import build_hand_retargeter_from_specs
    specs = {("l21", side): load_hand_spec("l21", side) for side in ("left", "right")}
    return build_hand_retargeter_from_specs(specs, 'nn', checkpoints={'l21': checkpoint}, device=device)
