def main(args=None):
    """LinkerHand 灵巧手独立展示 - 从 SDK 加载完整 3D 模型"""
    import pybullet as p
    import pybullet_data
    import time
    import os
    from teleoperation.paths import PROJECT_ROOT

    # SDK 中的灵巧手模型路径（相对本脚本定位，不依赖盘符）
    HAND_DIR = os.path.normpath(os.path.join(
        str(PROJECT_ROOT),
        "linkerhand_sdk", "ros1", "src", "assets", "robots", "hands", "linker_hand",
    ))

    if not os.path.isdir(HAND_DIR):
        raise SystemExit(
            "未找到 LinkerHand 模型目录：\n"
            f"  {HAND_DIR}\n\n"
            "请先克隆 SDK 后重试：\n"
            "  git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git linkerhand_sdk\n"
        )

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    from teleoperation.simulation.urdf_loader import create_ground, load_urdf
    create_ground(p)

    # 加载左右灵巧手（统一为 L21，17 个可动关节 —— 与队友重定向目标一致）
    # ★ 2026-09-12 型号统一：原先加载的是 l7（每手 7 关节、无 *_mcp_roll），
    #   与队友模型的输出（L21）不匹配，已按队友确认统一为 L21。
    p.setAdditionalSearchPath(HAND_DIR)

    # 右手 - L21
    right_hand = load_urdf(
        os.path.join(HAND_DIR, "l21_right", "linkerhand_l21_right.urdf"),
        basePosition=[-0.2, 0, 0.6],
        baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]),
        useFixedBase=True
    )

    # 左手 - L21
    left_hand = load_urdf(
        os.path.join(HAND_DIR, "l21_left", "linkerhand_l21_left.urdf"),
        basePosition=[0.2, 0, 0.6],
        baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]),
        useFixedBase=True
    )

    print(f"右手加载: {right_hand}, 关节数: {p.getNumJoints(right_hand)}")
    print(f"左手加载: {left_hand}, 关节数: {p.getNumJoints(left_hand)}")

    # 关节信息
    for i in range(p.getNumJoints(right_hand)):
        info = p.getJointInfo(right_hand, i)
        print(f"  右手关节{i}: {info[1].decode() if isinstance(info[1],bytes) else info[1]}")

    p.resetDebugVisualizerCamera(
        cameraDistance=0.8, cameraYaw=180, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    print(f"\n{'='*50}")
    print(f"  灵心巧手 LinkerHand 独立模型展示")
    print(f"  左手: L21  |  右手: L21   （17 个可动关节 / 手）")
    print(f"  每只手 17 关节, 带完整 STL 网格")
    print(f"  鼠标滚轮可放大看手指细节")
    print(f"{'='*50}")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        p.disconnect()
