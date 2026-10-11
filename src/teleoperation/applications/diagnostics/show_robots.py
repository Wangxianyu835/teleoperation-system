"""同时显示 TeleOpBench 三种机器人：H1-2 / GR1-T2 / G1

用法:
    python -m teleoperation tools show-robots

三个机器人并排加载，双臂摆动演示。
鼠标左键旋转、右键平移、滚轮缩放。Ctrl+C 退出。
"""
import os
import re
import time
import pybullet as p
import pybullet_data
import numpy as np
from teleoperation.simulation.urdf_loader import create_ground, load_urdf

# 项目内机器人资产根目录（相对本脚本，不依赖盘符）
from teleoperation.paths import ROBOTS_ROOT
ROBOTS = str(ROBOTS_ROOT)

# 三种机器人：名称 -> (URDF 相对路径, 搜索路径, 世界坐标, 标签色)
ROBOT_SPECS = [
    ("H1-2",   os.path.join("h1_2", "h1_2.urdf"),
     os.path.join("h1_2"), [-2.2, 0, 1.00], [1.0, 0.15, 0.15]),
    ("GR1-T2", os.path.join("gr1", "urdf", "robot.urdf"),
     os.path.join("gr1", "urdf"), [0.0, 0, 0.85], [0.15, 0.25, 1.0]),
    ("G1",     os.path.join("g1", "g1_29dof_with_hand_lock_waist.urdf"),
     os.path.join("g1"), [2.2, 0, 0.80], [0.15, 1.0, 0.20]),
]

# 用于找“肩部摆动关节”的关节名片段（从左/右臂各找一个）
SHOULDER_PATTERN = re.compile(r"(shoulder_pitch|upper_arm_pitch|arm_pitch)", re.I)


def find_arm_joints(robot_id, client):
    """返回 {side: joint_index}，side 为 'left'/'right'，各取一个肩部摆动关节"""
    found = {}
    n = p.getNumJoints(robot_id, physicsClientId=client)
    for i in range(n):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        if info[2] == p.JOINT_FIXED:
            continue
        name = info[1].decode() if isinstance(info[1], bytes) else info[1]
        if not SHOULDER_PATTERN.search(name):
            continue
        if name.lower().startswith("left") and "left" not in found:
            found["left"] = i
        elif name.lower().startswith("right") and "right" not in found:
            found["right"] = i
        if "left" in found and "right" in found:
            break
    return found


def add_ground_marker(position, color):
    """脚下放一个醒目的彩色圆盘（半径 0.7m）"""
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.03)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.03,
                              rgbaColor=color + [1.0])
    p.createMultiBody(0, col, vis, [position[0], position[1], 0.015])


def add_head_marker(position, color):
    """头顶上方放一个醒目的彩色方块（边长 0.5m）"""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25] * 3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25] * 3,
                              rgbaColor=color + [1.0])
    p.createMultiBody(0, col, vis, [position[0], position[1], position[2]])


def main(args=None):
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    create_ground(p, physicsClientId=client)

    robots = []  # [(name, robot_id, joints_dict, base)]
    for name, urdf_rel, search_rel, base, color in ROBOT_SPECS:
        search_path = os.path.join(ROBOTS, search_rel)
        urdf_path = os.path.join(ROBOTS, urdf_rel)
        p.setAdditionalSearchPath(search_path, physicsClientId=client)
        try:
            rid = load_urdf(
                urdf_path,
                basePosition=base,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                physicsClientId=client,
            )
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            continue

        joints = find_arm_joints(rid, client)
        robots.append((name, rid, joints, base))

        # 彩色地面圆盘 + 头顶方块
        add_ground_marker(base, color)
        add_head_marker([base[0], base[1], base[2] + 1.5], color)

        # 3D 文字标签：永远面向相机，任意视角都可见
        p.addUserDebugText(
            name, [base[0], base[1], base[2] + 1.9],
            textColorRGB=color, textSize=1.5,
            lifeTime=0, physicsClientId=client,
        )

        print(f"[OK] {name}: ID={rid}, "
              f"{p.getNumJoints(rid, physicsClientId=client)} 关节, "
              f"摆动关节={joints}, 位置=({base[0]}, {base[1]}, {base[2]})")

    if not robots:
        print("没有任何机器人加载成功，退出。")
        p.disconnect(client)
        return

    # 斜俯视相机：既能看清立体轮廓，又能让三台沿 X 轴排开全部入画
    p.resetDebugVisualizerCamera(
        cameraDistance=12.0,
        cameraYaw=0,
        cameraPitch=-55,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client,
    )

    print("\n三个机器人已加载（带彩色圆盘/方块/3D文字标签）：")
    print("  左: H1-2(红)   中: GR1-T2(蓝)   右: G1(绿)")
    print("若某台不在视野，滚轮缩小 / 右键平移即可。")
    print("Ctrl+C 退出。")

    sim = 0
    try:
        while True:
            sim += 1
            a = 0.3 * np.sin(sim / 240.0 * 0.5)
            for name, rid, joints, base in robots:
                for side, idx in joints.items():
                    target = -0.3 + (a if side == "left" else -a)
                    p.setJointMotorControl2(rid, idx, p.POSITION_CONTROL,
                                            targetPosition=target, force=200,
                                            physicsClientId=client)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    except p.error:
        # GUI 窗口被手动关闭时触发
        pass
    finally:
        try:
            p.disconnect(client)
        except p.error:
            pass
        print("仿真环境已关闭")


