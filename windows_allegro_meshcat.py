"""Windows 11 demo: webcam -> MediaPipe -> DexPilot -> Allegro via MeshCat.

Put this file in the *root* of a cloned dexsuite/dex-retargeting v0.4.6
repository. Requires its recursively initialized `assets` submodule.

This is an adaptation, not an official dexsuite example. Untested on a
Windows physical machine. Press Q in the webcam window to quit.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig


ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "example" / "vector_retargeting"
if not EXAMPLES.is_dir():
    raise RuntimeError(
        f"Missing example directory: {EXAMPLES}. Put this script in the repo root."
    )
sys.path.insert(0, str(EXAMPLES))
from single_hand_detector import SingleHandDetector  # noqa: E402


def build_robot_and_viewer():
    urdf_base = ROOT / "assets" / "robots" / "hands"
    urdf = urdf_base / "allegro_hand" / "allegro_hand_right.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(
            f"Missing {urdf}. Execute: git submodule update --init --recursive"
        )

    RetargetingConfig.set_default_urdf_dir(urdf_base)
    config_file = get_default_config_path(
        RobotName.allegro, RetargetingType.dexpilot, HandType.right
    )
    config = RetargetingConfig.load_from_file(config_file)
    retargeting = config.build()

    # Reuse *the exact same* Pinocchio joint model as Dex Retargeting. This
    # prevents the joint-order mismatch of rebuilding a separate model.
    model = retargeting.optimizer.robot.model

    # OBJ visual meshes referenced relative to the hand's URDF directory.
    visual_model = pin.buildGeomFromUrdf(
        model,
        str(urdf),
        pin.GeometryType.VISUAL,
        package_dirs=[str(urdf.parent), str(urdf_base)],
    )
    viz = MeshcatVisualizer(model, None, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel(rootNodeName="allegro")
    viz.display(pin.neutral(model))
    print("MeshCat URL:", viz.viewer.url())
    print("Joint degrees of freedom:", model.nq)
    return retargeting, viz


def make_camera(camera_id):
    # DirectShow is usually more reliable for Windows USB webcams.
    cam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release()
        cam = cv2.VideoCapture(camera_id)
    if not cam.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_id}; try --camera 1 or close other apps."
        )
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam


def main():
    parser = argparse.ArgumentParser(description="Allegro webcam + MeshCat demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID, often 0")
    parser.add_argument(
        "--model-only", action="store_true", help="Test the 3D viewer without webcam"
    )
    parser.add_argument(
        "--selfie", action="store_true", help="Try if your right hand is not detected"
    )
    args = parser.parse_args()

    retargeting, viz = build_robot_and_viewer()
    if args.model_only:
        input("If the Allegro hand is visible in MeshCat, press Enter to exit... ")
        return

    detector = SingleHandDetector(hand_type="Right", selfie=args.selfie)
    camera = make_camera(args.camera)
    print("Point your RIGHT hand at the camera. Press Q in the camera window to exit.")
    print("If you see 'No right hand', retry with --selfie.")

    try:
        while True:
            success, bgr = camera.read()
            if not success:
                print("Camera frame capture failed.")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, joints_3d, landmarks_2d, _ = detector.detect(rgb)
            status = "No right hand"
            if joints_3d is not None:
                try:
                    indices = retargeting.optimizer.target_link_human_indices
                    origin = indices[0, :]
                    target = indices[1, :]
                    reference = joints_3d[target, :] - joints_3d[origin, :]
                    qpos = retargeting.retarget(reference)
                    viz.display(np.asarray(qpos, dtype=float))
                    status = "Tracking / retargeting"
                except Exception as exc:
                    status = f"Retarget failed: {type(exc).__name__}"
                    print("Retargeting error:", repr(exc))

                if landmarks_2d is not None:
                    detector.draw_skeleton_on_image(
                        bgr, landmarks_2d, style="default"
                    )

            cv2.putText(
                bgr,
                status,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if joints_3d is not None else (0, 0, 255),
                2,
            )
            cv2.imshow("Windows Allegro Teleop - press Q to quit", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
