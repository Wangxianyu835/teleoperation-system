"""LinkerHand 灵巧手独立展示 - 从 SDK 加载完整 3D 模型"""
import pybullet as p
import pybullet_data
import time
import os

# SDK 中的灵巧手模型路径（相对本脚本定位，不依赖盘符）
HAND_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
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
p.loadURDF("plane.urdf")

# 加载左右灵巧手（l7 = 7关节右手, l7_left = 7关节左手）
p.setAdditionalSearchPath(HAND_DIR)

# 右手 - l7
right_hand = p.loadURDF(
    "l7_right/linkerhand_l7_right.urdf",
    basePosition=[-0.2, 0, 0.6],
    baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]),
    useFixedBase=True
)

# 左手 - l7 (与右手配套的 l7 型号)
left_hand = p.loadURDF(
    "l7_left/linkerhand_l7_left.urdf",
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
print(f"  左手: l7  |  右手: l7")
print(f"  每只手 ~17 关节, 带完整 STL 网格")
print(f"  鼠标滚轮可放大看手指细节")
print(f"{'='*50}")

try:
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
except KeyboardInterrupt:
    p.disconnect()