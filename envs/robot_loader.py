"""机器人模型加载 —— 支持 TeleOpBench 论文的三种人形机器人

支持的机器人（与论文一致，实测数据 2026-09-12）：
    h1_2   : Unitree H1-2   —— 55 关节（51 可动）| 双臂各 7 DoF | 每手 12 关节
    gr1_t2 : Fourier GR1-T2 —— 70 关节（54 可动）| 双臂各 7 DoF | 每手 11 关节
    g1     : Unitree G1     —— 53 关节（41 可动）| 双臂各 7 DoF | 每手  7 关节

★★★ 动作空间约定（重定向算法必须遵守）★★★
    动作向量按【关节名】顺序排列，通过 action_joint_names / action_joint_indices 获取：

        action = [左臂 7] + [右臂 7] + [左手 N_L] + [右手 N_R]
                   0 ~ 6      7 ~ 13    14 ~ 14+N_L-1      其余

    维度不是固定的！用 len(loader.action_joint_indices) 或 env.action_dim 获取。

    ⚠️ 绝不能假设「动作向量第 i 个元素 -> 关节索引 i」！
       因为 URDF 前 12 个关节是【腿部】（左右髋/膝/踝），
       按索引直接映射会导致「用腿做动作、手臂不动」的严重错位。
       必须使用 action_joint_indices 做映射。
"""
import os

import pybullet as p

# 项目内机器人资产根目录（相对本文件定位，不依赖盘符）
DEFAULT_ROBOTS_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'robots', 'from_teleopbench',
))

# 三种机器人的配置：URDF 相对路径 + 初始高度 + 说明
ROBOT_SPECS = {
    'h1_2': {
        'urdf': os.path.join('h1_2', 'h1_2.urdf'),
        'base_z': 1.00,
        'desc': 'Unitree H1-2 (55关节, 双臂7DoF, 每手12关节)',
    },
    'gr1_t2': {
        'urdf': os.path.join('gr1', 'urdf', 'robot.urdf'),
        'base_z': 0.85,
        'desc': 'Fourier GR1-T2 (70关节, 双臂7DoF, 每手11关节)',
    },
    'g1': {
        'urdf': os.path.join('g1', 'g1_29dof_with_hand_lock_waist.urdf'),
        'base_z': 0.80,
        'desc': 'Unitree G1 (53关节, 双臂7DoF, 每手7关节)',
    },
}

# 判定「手臂关节」的关键词（三种机器人通用）
_ARM_KEYWORDS = ('shoulder', 'elbow', 'wrist')

# 手臂初始姿态（只对名字匹配的关节生效，其他机器人自动跳过）
ARM_NEUTRAL = {
    'left_shoulder_pitch_joint': -0.3,
    'left_elbow_pitch_joint': 1.2,
    'left_elbow_joint': 1.2,
    'right_shoulder_pitch_joint': -0.3,
    'right_elbow_pitch_joint': 1.2,
    'right_elbow_joint': 1.2,
}


def _is_arm_joint(name, side):
    """是否属于指定侧的手臂关节（含 shoulder / elbow / wrist）"""
    return name.startswith(side + '_') and any(k in name for k in _ARM_KEYWORDS)


def _is_hand_joint(name, side):
    """是否属于指定侧的灵巧手关节"""
    if side == 'left':
        return name.startswith('L_') or name.startswith('left_hand_')
    return name.startswith('R_') or name.startswith('right_hand_')

class RobotLoader:
    """加载机器人并构建「关节名 -> 索引」映射表"""

    def __init__(self, client, robots_root=None):
        self.client = client
        self.robots_root = robots_root or DEFAULT_ROBOTS_ROOT
        self.robot_type = None
        self.robot_id = None

        self.all_joints = {}                             # {关节名: 索引} 全部可动关节
        self.arm_joints = {'left': [], 'right': []}      # 手臂关节名列表（各 7 个）
        self.hand_joints = {'left': [], 'right': []}     # 灵巧手关节名列表（各 N 个）

        # ★ 动作空间映射（与 action 向量一一对应）
        self.action_joint_names = []
        self.action_joint_indices = []

    # ------------------------------------------------------------------ #
    def load_robot(self, robot_type='h1_2'):
        """加载指定机器人

        Args:
            robot_type: 'h1_2' | 'gr1_t2' | 'g1'

        Returns:
            dict: {'robot': body_id, 'left': body_id, 'right': body_id}
                  （为兼容 demo_teleop.py / teleop_pipeline.py 的旧写法而保留）
        """
        if robot_type not in ROBOT_SPECS:
            raise ValueError(
                f"未知的 robot_type='{robot_type}'，可选：{list(ROBOT_SPECS)}")

        spec = ROBOT_SPECS[robot_type]
        urdf_path = os.path.join(self.robots_root, spec['urdf'])
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(
                f"找不到机器人模型：\n  {urdf_path}\n"
                f"robots/from_teleopbench/ 未纳入 Git，需从 TeleOpBench 获取"
                f"或由队友拷贝后再运行。")

        self.robot_type = robot_type
        print(f"  加载 {spec['desc']}")

        p.setAdditionalSearchPath(os.path.dirname(urdf_path),
                                  physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            urdf_path, [0, 0, spec['base_z']],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True, physicsClientId=self.client,
        )

        self._build_joint_maps()
        self._apply_neutral_pose()

        return {'robot': self.robot_id,
                'left': self.robot_id,
                'right': self.robot_id}

    # ------------------------------------------------------------------ #
    def _build_joint_maps(self):
        """遍历所有关节，构建映射表 + 动作空间定义"""
        self.all_joints = {}
        self.arm_joints = {'left': [], 'right': []}
        self.hand_joints = {'left': [], 'right': []}

        n = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(n):
            info = p.getJointInfo(self.robot_id, i,
                                  physicsClientId=self.client)
            if info[2] == p.JOINT_FIXED:
                continue
            name = info[1].decode() if isinstance(info[1], bytes) else info[1]
            self.all_joints[name] = i
            for side in ('left', 'right'):
                if _is_arm_joint(name, side):
                    self.arm_joints[side].append(name)
                    break
                if _is_hand_joint(name, side):
                    self.hand_joints[side].append(name)
                    break

        # 动作空间 = 左臂 + 右臂 + 左手 + 右手
        self.action_joint_names = (
            self.arm_joints['left'] + self.arm_joints['right']
            + self.hand_joints['left'] + self.hand_joints['right'])
        self.action_joint_indices = [self.all_joints[nm]
                                     for nm in self.action_joint_names]

        print(f"    共 {n} 关节（{len(self.all_joints)} 可动）")
        print(f"    左臂 {len(self.arm_joints['left'])} | "
              f"右臂 {len(self.arm_joints['right'])} | "
              f"左手 {len(self.hand_joints['left'])} | "
              f"右手 {len(self.hand_joints['right'])}")
        print(f"    ★ 动作空间维度 = {len(self.action_joint_indices)}")

        for side in ('left', 'right'):
            cnt = len(self.arm_joints[side])
            if cnt != 7:
                print(f"    ⚠️ 警告：{side} 手臂识别到 {cnt} 个关节（预期 7）")

    def _apply_neutral_pose(self):
        """把手臂摆到中性姿态（只影响名字匹配的关节）"""
        for name, val in ARM_NEUTRAL.items():
            idx = self.all_joints.get(name)
            if idx is not None:
                p.resetJointState(self.robot_id, idx, val,
                                  physicsClientId=self.client)

    # ------------------------------------------------------------------ #
    def set_arm_pose(self, side, angles):
        """按「关节名后缀」设置手臂姿态（兼容旧代码）"""
        prefix = side + '_'
        for k, v in angles.items():
            jname = k if k.startswith(prefix) else prefix + k
            idx = self.all_joints.get(jname)
            if idx is not None:
                p.setJointMotorControl2(
                    self.robot_id, idx, p.POSITION_CONTROL,
                    targetPosition=v, force=200,
                    physicsClientId=self.client)

    # ------------------------------------------------------------------ #
    def describe_action_space(self):
        """打印动作空间定义（供队友对接参考）"""
        print(f"\n动作空间（robot_type={self.robot_type}, "
              f"dim={len(self.action_joint_indices)}）:")
        for i, (nm, jidx) in enumerate(zip(self.action_joint_names,
                                           self.action_joint_indices)):
            print(f"  action[{i:2d}]  ->  关节索引 {jidx:3d}  {nm}")