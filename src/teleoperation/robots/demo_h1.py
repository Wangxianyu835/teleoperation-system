# 函数 map_demo_arm_targets（映射/转换）；常量 H1_ARM_NAMES（配置/索引）；链路：算法输出→机器人动作映射→执行命令
"""Original H1 demonstration arm joint mapping."""
H1_ARM_NAMES = (
    'shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint',
    'elbow_pitch_joint', 'elbow_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint',
)


def map_demo_arm_targets(left_arm, right_arm, arm_joints):
    targets = []
    for side, values in (("left", left_arm), ("right", right_arm)):
        for index, suffix in enumerate(H1_ARM_NAMES):
            name = side + "_" + suffix
            if name in arm_joints and index < len(values):
                targets.append((arm_joints[name], float(values[index])))
    return targets
