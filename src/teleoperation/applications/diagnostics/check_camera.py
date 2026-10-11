def main(args=None):
    """测试联想摄像头 - RGB 画面采集"""
    import sys, cv2, os
    from teleoperation.inputs.camera import CameraInterface

    cam = CameraInterface(0)
    cam.start()
    print("摄像头已启动! 画面正常采集")
    print("按 q 键退出")

    while True:
        rgb, bgr = cam.capture_frame()
        if bgr is None:
            continue

        # 显示帧率
        cv2.putText(bgr, "RGB Camera (联想电脑)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        key = cam.show_preview(bgr, "TeleOpBench Camera")
        if key == ord('q'):
            break

    cam.stop()
    print("摄像头已关闭")
