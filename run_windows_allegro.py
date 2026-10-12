"""Windows entry point for the validated dex-retargeting v0.4.6 project environment.

The source checkout may remain in ./dex-retargeting. Use the project environment
Python after dependency installation. --model-only does not open the camera.
"""
import argparse
from pathlib import Path
import sys


def source_path(argument):
    root = Path(__file__).resolve().parent
    candidates = [Path(argument).resolve()] if argument else [root / "dex-retargeting", root]
    for candidate in candidates:
        if (candidate / "dex_retargeting/retargeting_config.py").is_file():
            if not (candidate / "example/vector_retargeting/single_hand_detector.py").is_file():
                raise FileNotFoundError(f"Missing hand detector in {candidate}")
            return candidate
    raise FileNotFoundError("Source checkout not found. Specify its directory with --repo.")


def build_robot_and_viewer(repo, open_browser):
    import numpy as np
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    from dex_retargeting.constants import HandType, RetargetingType, RobotName, get_default_config_path
    from dex_retargeting.retargeting_config import RetargetingConfig

    urdf_base = repo / "assets/robots/hands"
    urdf = urdf_base / "allegro_hand/allegro_hand_right.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"Missing Allegro URDF: {urdf}")
    RetargetingConfig.set_default_urdf_dir(urdf_base)
    config_file = get_default_config_path(RobotName.allegro, RetargetingType.dexpilot, HandType.right)
    retargeting = RetargetingConfig.load_from_file(config_file).build()
    # Display the model owned by the optimizer, preserving its configuration order.
    robot = retargeting.optimizer.robot
    model = robot.model
    if model.nq != 16 or model.nv != 16:
        raise RuntimeError(f"Expected 16 Allegro DOFs; nq={model.nq}, nv={model.nv}")
    model_names = [name for i, name in enumerate(model.names) if model.nqs[i] > 0]
    if list(retargeting.joint_names) != model_names:
        raise RuntimeError("Retargeting and viewer joint names differ")
    visual_model = pin.buildGeomFromUrdf(
        model, str(urdf), pin.GeometryType.VISUAL,
        package_dirs=[str(urdf.parent), str(urdf_base)],
    )
    if not visual_model.ngeoms:
        raise RuntimeError("No visual geometry loaded from Allegro URDF")
    viz = MeshcatVisualizer(model, None, visual_model)
    viz.initViewer(open=open_browser)
    try:
        viz.loadViewerModel(rootNodeName="allegro")
        # MeshCat 0.3.2 exposes camera nodes, rather than Pinocchio's newer camera helpers.
        from meshcat.transformations import translation_matrix
        viz.viewer["/Cameras/default"].set_transform(translation_matrix([0.0, 0.0, 0.10]))
        viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0.40, 0.0, 0.0])
    except Exception:
        if viz.viewer.window.server_proc is not None:
            viz.viewer.window.server_proc.kill()
            viz.viewer.window.server_proc.wait(timeout=10)
        raise
    q0 = np.clip(pin.neutral(model), model.lowerPositionLimit, model.upperPositionLimit)
    retargeting.set_qpos(q0)
    viz.display(q0)
    print("MeshCat URL:", viz.viewer.url())
    print("Joint names:", model_names)
    print("Visual geometries:", visual_model.ngeoms)
    return retargeting, viz


def make_camera(camera_id):
    import cv2

    camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera {camera_id}; try --camera 1 or close other camera apps")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def track_camera(repo, retargeting, viz, camera_id, selfie):
    import cv2
    import numpy as np

    sys.path.insert(0, str(repo / "example/vector_retargeting"))
    from single_hand_detector import SingleHandDetector

    detector = SingleHandDetector(hand_type="Right", selfie=selfie)
    camera = None
    try:
        camera = make_camera(camera_id)
        print("Point your RIGHT hand at the camera. Press Q to quit.")
        indices = np.asarray(retargeting.optimizer.target_link_human_indices)
        robot = retargeting.optimizer.robot
        lower, upper = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
        while True:
            success, bgr = camera.read()
            if not success:
                raise RuntimeError("Camera frame capture failed")
            _, joints_3d, landmarks_2d, _ = detector.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            status = "No right hand; try --selfie if needed"
            if joints_3d is not None:
                reference = joints_3d[indices[1], :] - joints_3d[indices[0], :]
                qpos = np.asarray(retargeting.retarget(reference), dtype=float)
                if qpos.shape != (16,) or not np.isfinite(qpos).all():
                    raise RuntimeError("Retargeting returned invalid joint positions")
                # Upstream NLopt bounds include a 1e-3 tolerance; enforce URDF limits for display.
                if np.any(qpos < lower - 1.1e-3) or np.any(qpos > upper + 1.1e-3):
                    raise RuntimeError("Retargeting returned positions outside URDF joint limits")
                viz.display(np.clip(qpos, lower, upper))
                status = "Tracking / retargeting"
            if landmarks_2d is not None:
                detector.draw_skeleton_on_image(bgr, landmarks_2d, style="default")
            cv2.putText(bgr, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if joints_3d is not None else (0, 0, 255), 2)
            cv2.imshow("Windows Allegro Teleop - press Q to quit", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    finally:
        if camera is not None:
            camera.release()
        detector.hand_detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Path to the dex-retargeting checkout")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--selfie", action="store_true")
    parser.add_argument("--model-only", action="store_true", help="Display the model without opening a camera")
    parser.add_argument("--no-browser", action="store_true", help="Print the MeshCat URL without opening a browser")
    args = parser.parse_args()
    repo = source_path(args.repo)
    sys.path.insert(0, str(repo))
    retargeting, viz = build_robot_and_viewer(repo, not args.no_browser)
    if args.model_only:
        input("Inspect the Allegro hand in MeshCat; press Enter to exit... ")
    else:
        track_camera(repo, retargeting, viz, args.camera, args.selfie)


if __name__ == "__main__":
    main()
