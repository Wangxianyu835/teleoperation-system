"""Existing robot geometry queries and mounted-hand coordinate math."""
import numpy as np
HAND_BASE_CANDIDATES = {
    'h1_2':   {'left': ['L_hand_base_link'],
               'right': ['R_hand_base_link']},
    'gr1_t2': {'left': ['l_hand_base_link'],
               'right': ['r_hand_base_link']},
    'g1':     {'left': ['left_hand_palm_link'],
               'right': ['right_hand_palm_link']},
}

# 判断「某个 link 属于机器人自带的手」（用于隐藏）
HAND_LINK_KEYS = ('thumb', 'index', 'middle', 'ring', 'pinky',
                  'hand_base', 'hand_palm', 'hand_')


def _link_name_of_joint(rid, i, cid):
    """取关节 i 的 child link 名

    注意：pybullet 的 getJointInfo **不返回 child link 的索引**；
    在树形结构中「关节 i 的 child link 索引就是 i 本身」。
    info[12] 是 child link 的【名字】（bytes）。
    """
    info = p_getJointInfo(rid, i, cid)
    nm = info[12]
    return nm.decode() if isinstance(nm, bytes) else nm


def _match_side(name, side):
    """判断 link 名是否属于指定侧（兼容 L_/R_ 与 left_/right_ 两种命名）"""
    low = (name or '').lower()
    if side == 'left':
        return low.startswith('l_') or low.startswith('left')
    return low.startswith('r_') or low.startswith('right')


def find_hand_base_link(rid, robot_type, side, cid):
    """在机器人里找「手基座 link」的索引（= 该 link 对应关节的索引）"""
    n = p_getNumJoints(rid, cid)
    name2link = {}
    for i in range(n):
        name2link[_link_name_of_joint(rid, i, cid)] = i
    # 优先用已知候选名（实测自 URDF）
    for cand in HAND_BASE_CANDIDATES.get(robot_type, {}).get(side, []):
        if cand in name2link:
            return name2link[cand], cand
    # 兜底：按名字找
    for ln, li in name2link.items():
        low = ln.lower()
        if _match_side(ln, side) and ('hand_base' in low
                                      or 'hand_palm' in low
                                      or 'end_effector' in low):
            return li, ln
    return None, None


def robot_hand_link_indices(rid, side, cid):
    """找出机器人自带「手」的所有 link 索引（用于隐藏）"""
    idxs = set()
    n = p_getNumJoints(rid, cid)
    for i in range(n):
        ln = _link_name_of_joint(rid, i, cid)
        low = (ln or '').lower()
        if not low:
            continue
        if _match_side(ln, side) and any(k in low for k in HAND_LINK_KEYS):
            idxs.add(i)
    return idxs


def _collect_link_positions(n_links, get_name, get_pos):
    """收集 link 名与位置（供下面推断坐标系用）"""
    out = []
    for i in range(n_links):
        nm = get_name(i)
        if nm:
            out.append((nm, np.asarray(get_pos(i, nm), dtype=float)))
    return out


def _finger_tips(link_positions):
    """按手指分组，每根手指取离掌根最远的那个 link"""
    best = {}
    for nm, pos in link_positions:
        low = nm.lower()
        for k in FINGER_KEYS:
            if k in low:
                d = float(np.linalg.norm(pos))
                if k not in best or d > best[k][0]:
                    best[k] = (d, pos)
                break
    return [v[1] for v in best.values()]


def _thumb_tip(link_positions):
    best = None
    for nm, pos in link_positions:
        if THUMB_KEY in nm.lower():
            d = float(np.linalg.norm(pos))
            if best is None or d > best[0]:
                best = (d, pos)
    return best[1] if best else None


def compute_hand_frame(link_positions):
    """由 link 位置推断手的正交坐标系 M=[e1|e2|e3]（3x3）

    e1 = 手指方向（四指指尖的均值方向）
    e2 = 拇指侧（拇指尖方向去掉 e1 分量后归一化）
    e3 = e1 x e2
    """
    tips = _finger_tips(link_positions)
    thumb = _thumb_tip(link_positions)
    if len(tips) < 2 or thumb is None:
        return None
    mean_tip = np.mean(np.array(tips), axis=0)
    n1 = float(np.linalg.norm(mean_tip))
    if n1 < 1e-9:
        return None
    e1 = mean_tip / n1
    t = np.asarray(thumb, dtype=float)
    t = t - float(np.dot(t, e1)) * e1
    n2 = float(np.linalg.norm(t))
    if n2 < 1e-9:
        return None
    e2 = t / n2
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3])


def _mat_to_euler_xyz(M):
    """把旋转矩阵转成 pybullet 的 getQuaternionFromEuler 用的 (r, p, y)

    pybullet 的约定是 R = Rz(yaw) * Ry(pitch) * Rx(roll)，
    本函数按同一约定反解，保证 p.getQuaternionFromEuler(rpy) 能还原 M。
    """
    m = [float(M[r][c]) for r in range(3) for c in range(3)]
    return list(_euler_from_matrix_xyz(m))


def _euler_from_matrix_xyz(m):
    """3x3（行主序 list）-> (roll, pitch, yaw)"""
    sy = -m[6]
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    if abs(sy) < 1.0 - 1e-9:
        roll = float(np.arctan2(m[7], m[8]))
        yaw = float(np.arctan2(m[3], m[0]))
    else:
        roll = float(np.arctan2(-m[5], m[4]))
        yaw = 0.0
    return roll, pitch, yaw


def _vec_angle(u, v):
    """两个向量之间的夹角（度）"""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c = float(np.dot(u, v)) / (float(np.linalg.norm(u))
                               * float(np.linalg.norm(v)) + 1e-12)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def compute_auto_mount_rpy(robot_frame, hand_frame):
    """R_mount = M_robot * M_hand^T（把 l21 的坐标系对到机器人的手基座坐标系）"""
    if robot_frame is None or hand_frame is None:
        return None, 0.0
    R = np.asarray(robot_frame, dtype=float) @ np.asarray(
        hand_frame, dtype=float).T
    det = float(np.linalg.det(R))
    return _mat_to_euler_xyz(R), det


def measure_hand_frame(body_id, base_pos, base_orn, cid, link_indices=None):
    """测出某只手的坐标系（表达在 base_pos/base_orn 这个「手基座」系里）

    link_indices=None 表示取全部 link（用于 l21 手本身）。
    """
    import pybullet as p
    inv_p, inv_o = p.invertTransform(base_pos, base_orn,
                                     physicsClientId=cid)
    n = p.getNumJoints(body_id, physicsClientId=cid)
    idxs = list(range(n)) if link_indices is None else sorted(link_indices)
    links = []
    for li in idxs:
        nm = _link_name_of_joint(body_id, li, cid)
        st = p.getLinkState(body_id, li, computeForwardKinematics=True,
                            physicsClientId=cid)
        rel, _ = p.multiplyTransforms(inv_p, inv_o, st[4], st[5],
                                      physicsClientId=cid)
        links.append((nm, rel))
    return compute_hand_frame(links)


def p_getNumJoints(rid, cid):
    import pybullet as p
    return p.getNumJoints(rid, physicsClientId=cid)


def p_getJointInfo(rid, i, cid):
    import pybullet as p
    return p.getJointInfo(rid, i, physicsClientId=cid)


FINGER_KEYS = ("index", "middle", "ring", "pinky")
THUMB_KEY = "thumb"
