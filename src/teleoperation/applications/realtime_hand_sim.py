"""Native L21 scene and explicit scene selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from teleoperation.simulation.urdf_loader import prepare_physics
from teleoperation.data.urdf import parse_joint_limits
from teleoperation.paths import (
    DEFAULT_MEDIAPIPE_ASSET,
    L21_ROOT,
    PROJECT_ROOT,
)
from teleoperation.robots.l21 import L21_JOINT_ORDER, L21HandAdapter

SIDES = ("left", "right")
URDF_NAME = "linkerhand_l21_{side}.urdf"
ASSET_NAME = "hand_landmarker.task"
# 双手并排摆位；单只手时放在原点前方。
BASE_POSITIONS = {"left": (-0.15, 0.0, 0.3), "right": (0.15, 0.0, 0.3)}
# 摄像头窗口：标题、指尖下标（复用 tools/plotting 的 COLORS 与 SOURCE_EDGES 连线）。


def resolve_model_asset(path: Path | None) -> Path:
    """Return the MediaPipe asset path, or raise with the exact places searched."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([DEFAULT_MEDIAPIPE_ASSET, PROJECT_ROOT / ASSET_NAME, PROJECT_ROOT / "outputs" / ASSET_NAME])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        f"MediaPipe model asset not found; searched: {searched}. Download the "
        f"official hand_landmarker.task (float16, about 7.5 MB) and put it in one "
        f"of those paths, or pass --model-asset-path."
    )


class RealtimeL21Scene:
    """PyBullet L21 scene; native named commands plus explicit historical layout reader.

    Only the native joints in ``L21_JOINT_ORDER`` are written; any
    extra URDF joints (if a model ever has them) stay at their loaded state.
    """

    def __init__(self, sides, gui: bool = True):
        self.cid = -1
        try:
            self._initialize(sides,gui)
        except BaseException:
            if self.cid >= 0 and hasattr(self,'p'):
                self.close()
            raise

    def _initialize(self, sides, gui):
        import pybullet as p

        if not sides:
            raise ValueError("at least one hand side is required")
        self.p = p
        self.sides = tuple(sides)
        self.gui = bool(gui)
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0 or not p.isConnected(self.cid):
            raise RuntimeError(
                "PyBullet GUI connection could not be created (no display or a "
                "non-interactive session); rerun with --headless."
            )
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # PyBullet 的 GUI 服务端是异步起来的：本机实测第一次 showGUI 后立刻调用
        # setGravity 会报 "Not connected to physics server"。这里重试到服务端应答为止。
        # 这是可视化场景：不模拟重力，否则 resetJointState 写入的角度会在
        # stepSimulation 后被重力拖走（探针实测拇指自由关节漂移约 1e-3 rad）。
        prepare_physics(p.setGravity, (0, 0, 0), client_id=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        self.bodies = {}
        self.base_positions = {}
        self.view = "full"
        self.joint_index = {}
        self.adapters = {}
        for side in self.sides:
            directory = L21_ROOT / side
            urdf = directory / URDF_NAME.format(side=side)
            if not urdf.is_file():
                raise FileNotFoundError(f"L21 model not found: {urdf}")
            p.setAdditionalSearchPath(str(directory), physicsClientId=self.cid)
            base = list(BASE_POSITIONS[side]) if len(self.sides) > 1 else [0.0, 0.0, 0.3]
            self.base_positions[side] = base
            body = p.loadURDF(str(urdf), base, useFixedBase=True, physicsClientId=self.cid)
            names = {}
            for index in range(p.getNumJoints(body, physicsClientId=self.cid)):
                info = p.getJointInfo(body, index, physicsClientId=self.cid)
                name = info[1].decode() if isinstance(info[1], bytes) else info[1]
                names[name] = index
            missing = [name for name in L21_JOINT_ORDER if name not in names]
            if missing:
                raise RuntimeError(f"{urdf.name} lacks L21 joints: {missing}")
            self.bodies[side] = body
            self.joint_index[side] = names
            self.adapters[side] = L21HandAdapter(list(L21_JOINT_ORDER), parse_joint_limits(urdf))
        self.frame_scene("full")
        self.applied = {side: 0 for side in self.sides}
        self.loss_reported = False

    def frame_scene(self, view="full") -> None:
        """Front: both palms toward the camera; full: original URDF layout.

        Both native palms face +X. A -90 degree world-Z rotation points them
        toward a yaw=0 camera on -Y, with the fingers still pointing upward.
        Only rigid display poses change; native joint angles remain untouched.
        """
        if view not in ("full", "front"):
            raise ValueError(f"Unknown L21 view: {view}")
        if not self.alive():
            return
        if view != self.view:
            orientation = self.p.getQuaternionFromEuler([0, 0, -np.pi / 2 if view == "front" else 0])
            for side, body in self.bodies.items():
                dynamics = self.p.getDynamicsInfo(body, -1, physicsClientId=self.cid)
                # resetBasePositionAndOrientation expects the inertial/COM pose,
                # whereas loadURDF takes the root-link pose (different on left).
                position, rotation = self.p.multiplyTransforms(
                    self.base_positions[side], orientation, dynamics[3], dynamics[4])
                self.p.resetBasePositionAndOrientation(body, position, rotation, physicsClientId=self.cid)
            self.view = view
        if self.gui:
            self.p.resetDebugVisualizerCamera(
                cameraDistance=0.7 if len(self.sides) > 1 else 0.55,
                cameraYaw=0 if view == "front" else 40,
                cameraPitch=0 if view == "front" else -20,
                cameraTargetPosition=[0.0, 0.0, 0.43 if view == "front" else 0.3],
                physicsClientId=self.cid,
            )


    def apply_native(self, command):
        """Apply expanded named L21 commands without a historical angle layout."""
        from teleoperation.simulation.hand_executor import HandExecutor
        from teleoperation.robots.hand_spec import load_hand_spec
        if not hasattr(self, '_native_executor'):
            specs = {('l21',side):load_hand_spec('l21',side) for side in self.sides}
            def write(value):
                if not self.alive():
                    raise RuntimeError('L21 simulator connection closed')
                for name, angle in value.named_positions().items():
                    self.p.resetJointState(self.bodies[value.side],self.joint_index[value.side][name],angle,physicsClientId=self.cid)
                self.applied[value.side] += 1
            self._native_executor = HandExecutor(specs,{key:write for key in specs})
        self._native_executor.apply(command)

    def step(self) -> None:
        """Step physics; a lost connection is reported once and then ignored.

        The PyBullet GUI connection can drop mid-run (window closed, or a session
        that cannot create an OpenGL window). Stepping a dead server raises
        ``Not connected to physics server``, which would also skip the summary and
        the ``--report`` file; instead we note it once and let the loop exit.
        """
        if not self.alive():
            if not self.loss_reported:
                self.loss_reported = True
                print("PyBullet connection lost (GUI window closed or no display in this "
                      "session); stopping and still writing the summary. Use --headless "
                      "when the GUI cannot open here.", flush=True)
            return
        self.p.stepSimulation(physicsClientId=self.cid)

    def alive(self) -> bool:
        """False once the user closes the GUI window."""
        return bool(self.p.isConnected(self.cid))

    def close(self) -> None:
        if self.p.isConnected(self.cid):
            self.p.disconnect(self.cid)

    def joint_values(self, side: str) -> dict:
        """Current L21 joint values, for reporting what the sim actually shows."""
        if side not in self.bodies or not self.alive():
            return {}
        body = self.bodies[side]
        index = self.joint_index[side]
        return {
            name: float(self.p.getJointState(body, index[name], physicsClientId=self.cid)[0])
            for name in L21_JOINT_ORDER
        }


def build_scene(args, sides):
    """Construct a native hand or whole-robot simulation scene."""
    if args.scene == "hands":
        return RealtimeL21Scene(sides, gui=not args.headless)
    from teleoperation.applications.three_robot_scene import ThreeRobotHandScene

    return ThreeRobotHandScene(sides, gui=not args.headless,
                               steps_per_frame=args.sim_steps)
