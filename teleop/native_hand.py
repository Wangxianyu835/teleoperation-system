"""L21 数据 -> 机器人【原装灵巧手】的降维映射（本项目正式采用的方案）

决策背景（2026-09-12）
---------------------
队友的重定向目标手是 **LinkerHand L21**（17 可动关节），
而论文里三台机器人出厂装的是**各自的手**：

    H1-2   -> Inspire          12 关节/手
    GR1-T2 -> Fourier 原生手    11 关节/手
    G1     -> 轻量三指手         7 关节/手

本项目**决定在仿真里使用机器人原装的手**（不换手），
因此需要把 18 维 L21 数据降维映射到原装手上。

映射原则：语义 1:1，不做无依据的「合并」
--------------------------------------
    L21 mcp_pitch  <->  原装手 proximal      （近端指节屈曲）
    L21 pip        <->  原装手 intermediate  （远端指节屈曲）
    L21 *_mcp_roll ->   丢弃（原装手没有侧摆自由度）
    拇指 cmc_yaw/pitch -> proximal_yaw/pitch
    拇指 mcp/ip        -> intermediate/distal（各机器人略有差异）

[注意] 这是**有损**映射，必须在报告里声明丢弃的自由度：
    H1-2   丢 5 个（4 个 *_mcp_roll + 拇指 cmc_roll）
    GR1-T2 丢 6 个（再丢拇指 1 个）
    G1     丢 10 个（另丢无名指/小指整根 + 拇指 1 个）

符号方向自动判定（关键，否则手指会反着弯）
----------------------------------------
    H1-2   [0.000, +1.700]  屈曲为正
    GR1-T2 [-1.570, 0.000]  屈曲为负
    G1     left 负 / right 正（左右手还不一样）
规则：sign = +1 if |hi| >= |lo| else -1
"""
import numpy as np

# 18 维数据的语义名（dim 0 是 hand_base_link 固定占位）
DIM_NAMES = {
    0: 'hand_base_link(占位)',
    1: 'index_mcp_roll', 2: 'index_mcp_pitch', 3: 'index_pip',
    4: 'middle_mcp_roll', 5: 'middle_mcp_pitch', 6: 'middle_pip',
    7: 'ring_mcp_roll', 8: 'ring_mcp_pitch', 9: 'ring_pip',
    10: 'pinky_mcp_roll', 11: 'pinky_mcp_pitch', 12: 'pinky_pip',
    13: 'thumb_cmc_roll', 14: 'thumb_cmc_yaw', 15: 'thumb_cmc_pitch',
    16: 'thumb_mcp', 17: 'thumb_ip',
}

# 原装手的关节名前缀
PREFIX = {
    'h1_2':   {'left': 'L_', 'right': 'R_'},
    'gr1_t2': {'left': 'L_', 'right': 'R_'},
    'g1':     {'left': 'left_hand_', 'right': 'right_hand_'},
}

# 语义名 -> 原装手关节名模板（{P} 替换成上面的前缀）
# 未列出的语义名 = 原装手没有这个自由度，只能丢弃
NATIVE_MAP = {
    'h1_2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_mcp':        '{P}thumb_intermediate_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'gr1_t2': {
        'index_mcp_pitch':  '{P}index_proximal_joint',
        'index_pip':        '{P}index_intermediate_joint',
        'middle_mcp_pitch': '{P}middle_proximal_joint',
        'middle_pip':       '{P}middle_intermediate_joint',
        'ring_mcp_pitch':   '{P}ring_proximal_joint',
        'ring_pip':         '{P}ring_intermediate_joint',
        'pinky_mcp_pitch':  '{P}pinky_proximal_joint',
        'pinky_pip':        '{P}pinky_intermediate_joint',
        'thumb_cmc_yaw':    '{P}thumb_proximal_yaw_joint',
        'thumb_cmc_pitch':  '{P}thumb_proximal_pitch_joint',
        'thumb_ip':         '{P}thumb_distal_joint',
    },
    'g1': {
        # G1 只有 食指 / 中指 / 拇指 三根
        'index_mcp_pitch':  '{P}index_0_joint',
        'index_pip':        '{P}index_1_joint',
        'middle_mcp_pitch': '{P}middle_0_joint',
        'middle_pip':       '{P}middle_1_joint',
        'thumb_cmc_yaw':    '{P}thumb_0_joint',
        'thumb_cmc_pitch':  '{P}thumb_1_joint',
        'thumb_mcp':        '{P}thumb_2_joint',
    },
}

# l21 真正可动的维度数（dim 1~17）
N_MOVABLE = 17

# L21 各维的关节限位（来自 l21 URDF，可用 scripts/replay_hand_angles.py
# 里的 parse_joint_limits() 复核）。只有 --limit-mode rescale 会用到。
L21_LIMITS = {
    1: (-0.18, 0.18), 2: (0.0, 1.57), 3: (0.0, 1.57),
    4: (-0.18, 0.18), 5: (0.0, 1.57), 6: (0.0, 1.57),
    7: (-0.18, 0.18), 8: (0.0, 1.57), 9: (0.0, 1.57),
    10: (-0.18, 0.18), 11: (0.0, 1.57), 12: (0.0, 1.57),
    13: (-0.60, 0.60), 14: (0.0, 1.60), 15: (0.0, 1.00),
    16: (0.0, 1.57), 17: (0.0, 1.57),
}


def read_joint_ranges(robot_id, client):
    """读取 {关节名: (下限, 上限)} 与 {关节名: 关节索引}

    [注意] 必须用 info[1]（【关节名】，如 `L_index_proximal_joint`），
    不能用 info[12]（那是 child link 名 `L_index_proximal`）。
    一开始用错会导致覆盖率 0%。
    """
    import pybullet as p
    ranges, names = {}, {}
    for i in range(p.getNumJoints(robot_id, physicsClientId=client)):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        if info[2] == p.JOINT_FIXED:
            continue
        jn = info[1].decode() if isinstance(info[1], bytes) else info[1]
        ranges[jn] = (float(info[8]), float(info[9]))
        names[jn] = i
    return ranges, names


def build_mapping(robot_type, side, joint_ranges):
    """生成映射表：[(数据维度, 原装手关节名, 符号, 下限, 上限), ...]"""
    pref = PREFIX[robot_type][side]

    def sign_of(lo, hi):
        return 1.0 if abs(hi) >= abs(lo) else -1.0

    table = NATIVE_MAP[robot_type]
    out = []
    for dim, sem in DIM_NAMES.items():
        tpl = table.get(sem)
        if tpl is None:
            continue                      # 原装手没有这个自由度
        jn = tpl.format(P=pref)
        if jn not in joint_ranges:
            continue
        lo, hi = joint_ranges[jn]
        out.append((dim, jn, sign_of(lo, hi), lo, hi))
    return out


def link_extent(robot_id, link_ids, client):
    """合并若干 link 的 AABB -> (中心, 各轴边长向量)；无有效 link 时返回 (None, None)

    用途：给「手部特写」相机算目标点与距离。
    [注意] 必须用 AABB，不能用关节原点 —— 关节原点只是 link 坐标系原点，
    手指网格可能偏离好几厘米。而且三台机器人的手在【世界】里的方位
    完全不同（实测右手：H1-2 x=+0.36 / GR1-T2 x=-0.19 / G1 x=+0.25），
    用固定 yaw 的相机会把镜头塞进 GR1-T2 的躯干内部，只能拍到面片背面。

    返回 size 用【向量】而不是标量：`--hand both` 时两只手合起来是一个
    横长条（实测 GR1-T2 宽 0.55 m），用最大边长会算错相机距离，
    用对角线长度 |size| 才是稳的。
    """
    import pybullet as p
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for i in link_ids:
        try:
            a, b = p.getAABB(robot_id, i, physicsClientId=client)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(a, dtype=float))
        hi = np.maximum(hi, np.asarray(b, dtype=float))
    if not np.isfinite(lo).all():
        return None, None
    return (lo + hi) / 2.0, (hi - lo)


def dropped_dims(robot_type, side, mapping):
    """返回被丢弃的数据维度（原装手没有对应自由度）"""
    del side
    used = {d for d, *_ in mapping}
    return [d for d in DIM_NAMES if d != 0 and d not in used]


def map_frame(row, mapping, limit_mode='clamp'):
    """把一帧 18 维数据映射成 {关节名: 目标角度}（已按符号与限位处理）

    Args:
        row: (18,) 一帧 L21 角度
        mapping: build_mapping() 的输出
        limit_mode:
          'clamp'   （默认，忠实映射）
                    超出原装手限位的值直接【截断】。语义准确，
                    但数据超出原装手行程时手指会"卡"在限位上。
          'rescale' （视觉辅助，不是真实映射）
                    用 L21 的限位把比例线性映射到原装手的限位：
                    保运动形状、幅度等比缩放，不会卡住。
                    代价：改变了角度语义，报告里必须声明。
                    注意 实测触发原因：L21 拇指 cmc_pitch 行程 0~1.0，
                       而 H1-2 原装手 thumb_proximal_pitch 只有 -0.1~0.6
                       -> 右撇子有 50% 的帧被截断。

    Returns:
        (joints: dict, n_clipped: int)
    """
    row = np.asarray(row, dtype=np.float64)
    joints, clipped = {}, 0
    for d, jn, sg, lo, hi in mapping:
        v = sg * float(row[d])
        if limit_mode == 'rescale' and d in L21_LIMITS:
            l21lo, l21hi = L21_LIMITS[d]
            span = l21hi - l21lo
            v = lo + (v - l21lo) / span * (hi - lo) if span > 1e-9 else lo
            v = min(max(v, lo), hi)
        else:
            if v < lo or v > hi:
                clipped += 1
            v = min(max(v, lo), hi)
        joints[jn] = v
    return joints, clipped


def coverage(robot_type, side, mapping):
    """覆盖率（可表达自由度 / 17）"""
    del side
    return len(mapping) / float(N_MOVABLE)
