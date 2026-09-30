"""离屏渲染三台机器人【原装手】的姿态快照（无需显示器，产出 PNG）

用途
----
把队友的重定向数据在【三台机器人】的原装手上同时回放，
再把右手特写渲染成图片拼成条带，肉眼一次比完：

    h1_2（12 关节） / gr1_t2（11 关节） / g1（7 关节）

为什么不用 GUI 截图
------------------
DIRECT 模式 + getCameraImage 可以在没有显示器（CI / 远程桌面）时
同样出图，且相机指向【手部 link 的真实质心】，
不依赖「整机包围盒猜位置」这种近似。

两个离屏渲染的坑（都已处理，改代码前请先读）
--------------------------------------------
1. H1-2 的网格材质在 getCameraImage 下渲染成【纯黑】：把相机拉到
   0.8 m 依旧只有剪影（全黑图仅 133 KB），手指弯曲在图上完全看不出。
   -> 默认用 paint_readable() 把整机刷成浅灰；--no-paint 可关掉。
2. 相机必须先扫一遍所有取样帧、按【姿态并集】定一张固定相机：
   每格各算各的相机 -> 六格缩放各不相同，并排反而不好比姿态；
   只按第一格算 -> 第一格恰好是握紧姿态时，后面张开的帧会被裁掉。

用法
----
    python scripts/render_hand_snapshots.py
    python scripts/render_hand_snapshots.py --robots gr1_t2 g1 --n 5
    python scripts/render_hand_snapshots.py --file datasets/raw/xxx.h5
    python scripts/render_hand_snapshots.py --hand left --out outputs/hands
    python scripts/render_hand_snapshots.py --video          # 另出每台一段 MP4
    python scripts/render_hand_snapshots.py --video --stacked  # 再拼三台同框
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from teleop.filters import (  # noqa: E402
    detect_bad_frames, detect_identity_swaps, repair_stream,
)
from teleop.native_hand import (  # noqa: E402
    build_mapping, link_extent, map_frame, read_joint_ranges,
)

DT = 1.0 / 240.0
FORCE = 200.0
W, H = 480, 480


def load_data(path):
    import h5py
    with h5py.File(path, 'r') as f:
        d = {
            'left_angles': np.asarray(f['left_angles'][:], dtype=np.float64),
            'right_angles': np.asarray(f['right_angles'][:], dtype=np.float64),
            'timestamps': np.asarray(f['timestamps'][:], dtype=np.float64),
        }
        for k in ('left_valid', 'right_valid'):
            d[k] = (np.asarray(f[k][:]).astype(bool) if k in f
                    else np.ones(len(d['timestamps']), bool))
    return d


def body_center(p, rid, cid):
    """整机 AABB 中心：用来判断'哪一侧是身体外'"""
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for i in range(-1, p.getNumJoints(rid, physicsClientId=cid)):
        try:
            a, b = p.getAABB(rid, i, physicsClientId=cid)
        except p.error:
            continue
        lo = np.minimum(lo, np.asarray(a, dtype=float))
        hi = np.maximum(hi, np.asarray(b, dtype=float))
    if not np.isfinite(lo).all():
        return np.zeros(3)
    return (lo + hi) / 2.0


def hand_eye(hand_ctr, body_ctr, dist):
    """相机位置：沿【躯干 -> 手】方向再往外推 dist，保证在身体外侧

    ★ 不能用固定 yaw：三台机器人的手在世界的方位完全不同
      （H1-2 右手在 x=+0.36，GR1-T2 右手在 x=-0.19，G1 在 x=+0.25），
      固定 yaw=135 会把相机塞进 GR1-T2 的躯干内部，只能拍到面片背面。
    """
    u = hand_ctr - body_ctr
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        u = np.array([0.0, -1.0, 0.0])
    else:
        u = u / n
    return hand_ctr + u * dist


def flex_score(row, mapping):
    """一帧的"屈曲程度"：映射后各关节偏离行程中点的绝对量之和"""
    s = 0.0
    for d, _jn, sg, lo, hi in mapping:
        mid = 0.5 * (lo + hi)
        s += abs(sg * float(row[d]) - mid)
    return s


def pick_frames_by_flexion(arr, val, mapping, n_frames):
    """按屈曲程度排序取样：保证取到【最张开】和【最握紧】的帧

    [已不是默认] 现改用 pick_frames_diverse()：flex 对"整体张开/握紧"
    敏感，但实测对 GR1-T2 / G1 会选到两帧几乎一样的姿态
    （像素平均差仅 0.5，而关节其实摆了 1.4 rad）。
    保留此函数是因为"按张开程度排序看"仍有诊断价值。

    随机或等间隔取样可能全落在姿态相近的区间，
    导致肉眼看"手指根本没动"（其实是取帧问题，不是映射问题）。
    """
    idx = np.where(val)[0]
    if len(idx) == 0:
        return []
    scores = np.array([flex_score(arr[i], mapping) for i in idx])
    order = np.argsort(scores)
    picks = [int(round(t)) for t in np.linspace(0, len(order) - 1, n_frames)]
    return [int(idx[order[k]]) for k in dict.fromkeys(picks)]


def pick_frames_diverse(arr, val, mapping, n_frames):
    """按【映射后关节角的差异】取样：保证取到彼此最不相同的几帧

    [为什么不用屈曲程度排序] flex = 各关节偏离限位中点之和，它对
    "整体张开 / 握紧"敏感，但对【手指之间的相对差异】不敏感：
    实测按 flex 取到的最张开/最握紧两帧，GR1-T2 / G1 渲染出来几乎
    一模一样（像素平均差仅 0.5），而这些关节在数据里其实摆了 1.4 rad。
    这里改成【最远点采样】：在映射后的关节角空间里，每次挑一个离已选
    帧最远的帧 -> 几格必然是几个最不相同的姿态。
    """
    idx = np.where(val)[0]
    if len(idx) == 0:
        return []
    if len(idx) <= n_frames:
        return [int(i) for i in idx]
    mat = np.zeros((len(idx), len(mapping)))
    for k, fr in enumerate(idx):
        joints, _c = map_frame(arr[fr], mapping)
        for j, (_d, jn, *_rest) in enumerate(mapping):
            mat[k, j] = joints[jn]
    span = mat.max(axis=0) - mat.min(axis=0)
    span[span < 1e-9] = 1.0
    mat = (mat - mat.min(axis=0)) / span           # 每个关节归一到 [0,1]
    # 起点 = 离"平均姿态"最远的那帧；之后每次取离已选集合最远的帧
    picks = [int(np.argmax(np.linalg.norm(mat - mat.mean(axis=0), axis=1)))]
    dmin = np.linalg.norm(mat - mat[picks[0]], axis=1)
    while len(picks) < min(n_frames, len(idx)):
        k = int(np.argmax(dmin))
        picks.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(mat - mat[k], axis=1))
    return [int(idx[k]) for k in sorted(picks)]



def paint_readable(p, cid, rid):
    """把整机视觉颜色刷成浅灰（只改外观，不动物理）

    [必须] 实测 H1-2 的网格材质在 getCameraImage 下渲染成**纯黑**：
    把相机拉到 0.8 m 依旧只有剪影（全黑图仅 133 KB），
    手指弯曲在图上完全看不出来 —— 刷成浅灰后手指轮廓才可读。
    三台同色还有个好处：并排时差异只来自【姿态】，不来自材质。
    """
    for i in range(-1, p.getNumJoints(rid, physicsClientId=cid)):
        try:
            p.changeVisualShape(rid, i, rgbaColor=[0.82, 0.83, 0.88, 1.0],
                                physicsClientId=cid)
        except p.error:
            continue


def render(p, cid, eye, target):
    view = p.computeViewMatrix(cameraEyePosition=[float(v) for v in eye],
                               cameraTargetPosition=[float(v) for v in target],
                               cameraUpVector=[0.0, 0.0, 1.0])
    proj = p.computeProjectionMatrixFOV(50.0, W / float(H), 0.01, 5.0)
    _w, _h, rgb, _dep, _seg = p.getCameraImage(
        W, H, view, proj, renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid)
    img = np.asarray(rgb, dtype=np.uint8)
    if img.ndim == 1:
        # ER_TINY_RENDERER 返回的是扁平 RGBA 像素，且上下翻转
        # （硬件渲染器才会直接返回 (H, W, 4)）——两种都要能吃下
        img = img.reshape(H, W, 4)[::-1]
    return img[:, :, :3].copy()


def render_robot_strip(p, cid, robot_type, data, hand, n_frames,
                       dist, scale, paint=True):
    """渲染一台机器人的手部姿态条带 -> (图像数组, 每格标签, 关节数)"""
    import pybullet_data
    from envs import RobotLoader
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    loader = RobotLoader(cid)
    rid = loader.load_robot(robot_type)['robot']
    ranges, names = read_joint_ranges(rid, cid)
    mapping = build_mapping(robot_type, hand, ranges)

    # ---- 同 replay_hand_native：坏帧/身份切换修复 ----
    swaps = detect_identity_swaps(data['left_angles'], data['right_angles'],
                                  data['left_valid'], data['right_valid'])
    fronts = sorted({s['frame'] for s in swaps if s['side'] == hand})
    arr = data[f'{hand}_angles'].copy()
    val = data[f'{hand}_valid']
    arr, _h, _i = repair_stream(arr, val, detect_bad_frames(arr, val), fronts)

    # ---- 非手部关节锁定在初始姿态（与真实回放一致）----
    # [注意] hand_ids 存的是【关节索引】，排除时必须比索引；
    #   早期这里写成 info[1].decode() in hand_ids（名字比索引）永远为 False，
    #   结果手部关节也被 500 力矩锁在初始姿态 —— 画面上 4 张快照一模一样。
    hand_ids = {names[jn] for _d, jn, *_ in mapping}
    hold_ids, hold_tgt = [], []
    for i in range(p.getNumJoints(rid, physicsClientId=cid)):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED or i in hand_ids:
            continue
        hold_ids.append(i)
        hold_tgt.append(p.getJointState(rid, i, physicsClientId=cid)[0])

    # 在【有效帧】里按屈曲程度取样，保证取到最张开/最握紧的帧
    picks = pick_frames_diverse(arr, val, mapping, n_frames)

    joint_ids = [names[jn] for _d, jn, *_ in mapping]
    body_ctr = body_center(p, rid, cid)
    if paint:
        paint_readable(p, cid, rid)

    # 相机取景：先把所有取样帧扫一遍（直接写关节角、不渲染），
    # 用【并集 AABB】定相机 —— 一张固定相机，且张开/握紧都不出画。
    # 若只按第一格定相机：第一格恰好是握紧姿态时，后面张开的帧会被裁掉。
    union_lo = np.array([np.inf] * 3)
    union_hi = np.array([-np.inf] * 3)
    for fr in picks:
        joints, _c = map_frame(arr[fr], mapping)
        for jn, v in joints.items():
            p.resetJointState(rid, names[jn], float(v), physicsClientId=cid)
        c, s = link_extent(rid, joint_ids, cid)
        if c is None:
            continue
        union_lo = np.minimum(union_lo, c - s / 2.0)
        union_hi = np.maximum(union_hi, c + s / 2.0)
    if np.isfinite(union_lo).all():
        cam_ctr = (union_lo + union_hi) / 2.0
        cam_diag = float(np.linalg.norm(union_hi - union_lo))
    else:
        cam_ctr, cam_diag = np.zeros(3), 0.2
    d = dist if dist > 0 else max(0.25, cam_diag * 1.15)
    eye = hand_eye(cam_ctr, body_ctr, d)
    cam = (cam_ctr, union_hi - union_lo, eye, d)

    images, used = [], []
    for fr in picks:
        joints, _c = map_frame(arr[fr], mapping)
        for _ in range(200):                      # 收敛到该帧姿态
            for jn, v in joints.items():
                p.setJointMotorControl2(rid, names[jn], p.POSITION_CONTROL,
                                        targetPosition=v, force=FORCE,
                                        physicsClientId=cid)
            for k, j in enumerate(hold_ids):
                p.setJointMotorControl2(rid, j, p.POSITION_CONTROL,
                                        targetPosition=hold_tgt[k],
                                        force=500.0, physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
        # 手部 AABB（正式接口在 teleop.native_hand.link_extent）
        images.append(render(p, cid, cam[2], cam[0]))
        used.append(fr)
    p.removeBody(rid, physicsClientId=cid)
    labels = [f'frame {fr}  flex {flex_score(arr[fr], mapping):.2f}'
              for fr in used]
    cam_desc = (f'手部 AABB 中心 {np.round(cam[0], 3)} / 尺寸 '
                f'{np.round(cam[1], 3)} m / 相机距离 {cam[3]:.2f} m')
    return np.hstack(images), labels, len(mapping), cam_desc


def video_check(path, out_png, picks=None):
    """抽几帧做像素差自检（MP4 没法直接看，就抽帧看 + 存对比图）

    picks 默认 [首, 中, 末]；传「最张开 / 最握紧」的帧号更有说服力
    （首尾姿态相近时，默认取样会误报"几乎无变化"）。
    """
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    if picks is None:
        picks = (0, n // 2, n - 1)
    frames = []
    for k in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(int(k), n - 1)))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if len(frames) < 2:
        return None
    diffs = [float(np.mean(np.abs(frames[k].astype(int)
                                 - frames[k - 1].astype(int))))
             for k in range(1, len(frames))]
    cv2.imwrite(out_png, np.hstack(frames))
    return diffs


def stack_videos(paths, names, out_path, fps):
    """把三台各自录好的 MP4 按帧【横向拼接】成一段（帧本来就同步）

    三台的 frames 都取自同一份数据的同一 valid 掩码与同一 stride，
    所以第 k 帧必然对应同一个数据帧 —— 拼起来就是"同一时刻三台的手"，
    比三段视频来回切着看直观得多。
    """
    import cv2
    caps = [cv2.VideoCapture(p) for p in paths]
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    n = min(counts) if counts else 0
    if n <= 0:
        for c in caps:
            c.release()
        return None, counts
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             float(fps), (w * len(caps), h))
    if not writer.isOpened():
        for c in caps:
            c.release()
        return None, counts
    try:
        for _k in range(n):
            tiles = []
            for c, nm in zip(caps, names):
                ok, fr = c.read()
                if not ok:
                    fr = np.full((h, w, 3), 255, np.uint8)
                else:
                    cv2.putText(fr, nm, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (30, 30, 30), 2, cv2.LINE_AA)
                tiles.append(fr)
            writer.write(np.hstack(tiles))
    finally:
        writer.release()
        for c in caps:
            c.release()
    return n, counts


def render_robot_video(p, cid, robot_type, data, hand, out_path, fps,
                       max_frames, dist, paint=True):
    """把一台机器人的手部回放录成 MP4（相机在参考帧上算一次后【固定】）

    [注意] 不能每帧重算相机 —— 相机跟着手跑，画面会抖到看不出动作。
    所以先在【屈曲中位数】那帧上算一次手部 AABB，得到相机位置，之后固定。
    """
    import cv2
    import pybullet_data
    from envs import RobotLoader

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    loader = RobotLoader(cid)
    rid = loader.load_robot(robot_type)['robot']
    ranges, names = read_joint_ranges(rid, cid)
    mapping = build_mapping(robot_type, hand, ranges)

    swaps = detect_identity_swaps(data['left_angles'], data['right_angles'],
                                  data['left_valid'], data['right_valid'])
    fronts = sorted({s['frame'] for s in swaps if s['side'] == hand})
    arr = data[f'{hand}_angles'].copy()
    val = data[f'{hand}_valid']
    arr, _h, _i = repair_stream(arr, val, detect_bad_frames(arr, val), fronts)

    # 非手部关节锁在初始姿态（与单台回放一致）
    # [注意] hand_ids 存的是【关节索引】，排除时必须比索引
    hand_ids = {names[jn] for _d, jn, *_ in mapping}
    hold_ids, hold_tgt = [], []
    for i in range(p.getNumJoints(rid, physicsClientId=cid)):
        info = p.getJointInfo(rid, i, physicsClientId=cid)
        if info[2] == p.JOINT_FIXED or i in hand_ids:
            continue
        hold_ids.append(i)
        hold_tgt.append(p.getJointState(rid, i, physicsClientId=cid)[0])

    valid = np.where(val)[0]
    if len(valid) == 0:
        p.removeBody(rid, physicsClientId=cid)
        return None
    stride = max(1, int(len(valid) // max(1, max_frames)))
    frames = [int(i) for i in valid[::stride]]

    def apply_pose(fr):
        """按数据直接写关节角（精确复现，不靠力矩收敛）"""
        joints, _c = map_frame(arr[fr], mapping)
        for jn, v in joints.items():
            p.resetJointState(rid, names[jn], float(v), physicsClientId=cid)

    scores = np.array([flex_score(arr[i], mapping) for i in valid])
    ref = int(valid[int(np.argsort(scores)[len(valid) // 2])])
    if paint:
        paint_readable(p, cid, rid)
    joint_ids = [names[jn] for _d, jn, *_ in mapping]

    # 相机：扫一遍取样帧（直接写关节角、不渲染）取【并集 AABB】，
    # 一张固定相机 + 全程不出画
    union_lo = np.array([np.inf] * 3)
    union_hi = np.array([-np.inf] * 3)
    for fr in frames:
        apply_pose(fr)
        c, s = link_extent(rid, joint_ids, cid)
        if c is None:
            continue
        union_lo = np.minimum(union_lo, c - s / 2.0)
        union_hi = np.maximum(union_hi, c + s / 2.0)
    if np.isfinite(union_lo).all():
        ctr = (union_lo + union_hi) / 2.0
        size_vec = union_hi - union_lo
        cam_diag = float(np.linalg.norm(size_vec))
    else:
        ctr, size_vec, cam_diag = np.zeros(3), np.ones(3) * 0.15, 0.2
    d = dist if dist > 0 else max(0.25, cam_diag * 1.15)
    eye = hand_eye(ctr, body_center(p, rid, cid), d)

    # 自检取帧：用【姿态最不相同】的三帧比 —— 首尾姿态相近时默认取样会
    # 误报"几乎无变化"；按屈曲取帧对 GR1-T2/G1 也会选到几乎一样的帧
    pos = {fr: k for k, fr in enumerate(frames)}
    check_pos = [pos[fr] for fr in pick_frames_diverse(arr, val, mapping, 3)
                 if fr in pos]
    if len(check_pos) < 2:
        check_pos = [0, len(frames) // 2, len(frames) - 1]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             float(fps), (W, H))
    if not writer.isOpened():
        p.removeBody(rid, physicsClientId=cid)
        return None
    try:
        for fr in frames:
            apply_pose(fr)
            writer.write(cv2.cvtColor(render(p, cid, eye, ctr),
                                      cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        p.removeBody(rid, physicsClientId=cid)
    return {'path': out_path, 'n': len(frames), 'fps': fps, 'ref': ref,
            'cam': (ctr, size_vec, eye, d), 'check_pos': check_pos}


def main():
    import pybullet as p
    import cv2

    ap = argparse.ArgumentParser(description='离屏渲染三台机器人的原装手快照')
    ap.add_argument('--file', default=os.path.join(
        ROOT, 'datasets', 'raw', 'my_recording_angles.h5'))
    ap.add_argument('--robots', nargs='+', default=['h1_2', 'gr1_t2', 'g1'])
    ap.add_argument('--hand', default='right', choices=['left', 'right'])
    ap.add_argument('--n', type=int, default=4, help='每台渲染几张（按屈曲取帧）')
    ap.add_argument('--dist', type=float, default=0.0,
                    help='相机到手距离(m)；0 = 按手的实际尺寸自动算')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs',
                                                  'hand_snapshots'))
    ap.add_argument('--video', action='store_true',
                    help='额外给每台机器人录一段 MP4（相机固定，便于反复观看）')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--video-frames', type=int, default=600,
                    help='每个 MP4 最多多少帧（数据更长时等间隔抽帧）')
    ap.add_argument('--no-paint', dest='paint', action='store_false',
                    help='不刷浅灰：H1-2 网格材质在离屏渲染下是纯黑剪影，'
                         '默认刷成浅灰才看得清手指')
    ap.add_argument('--stacked', action='store_true',
                    help='配合 --video：把三台按帧横向拼成 three_robots.mp4'
                         '（同一条数据下三台姿态直接对比）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.file)

    p.connect(p.DIRECT)
    try:
        p.setTimeStep(DT)
        for rn in args.robots:
            img, labels, n_map, cam = render_robot_strip(
                p, 0, rn, data, args.hand, args.n, args.dist, args.scale,
                args.paint)
            if args.scale != 1.0:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)
            # 每格贴标签，避免"这是哪台机器人/哪一帧"分不清
            h, w = img.shape[:2]
            cw = w // max(len(labels), 1)
            for k, lab in enumerate(labels):
                cv2.putText(img, f'{rn} {args.hand[0].upper()} {lab}',
                            (k * cw + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            path = os.path.join(args.out, f'hand_{rn}.png')
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f'  {rn:<8} 可表达 {n_map}/17 关节  ->  {path}')
            print(f'           {cam}（相机 = 取样帧姿态并集取景，固定）')
            # 自检：相邻快照的像素平均差 —— 若接近 0，说明画面里姿态没变，
            # 要么取帧没取到差异，要么驱动没生效（避免"看起来一样"蒙混过关）
            if len(labels) > 1:
                print('           取样帧 = ' + ' | '.join(labels))
                diffs = [float(np.mean(np.abs(
                    img[:, k * cw:(k + 1) * cw].astype(int)
                    - img[:, (k - 1) * cw:k * cw].astype(int))))
                    for k in range(1, len(labels))]
                flag = '' if max(diffs) >= 1.0 else '  [注意] 几乎无变化，需排查'
                print('           相邻快照像素平均差 = '
                      + ' '.join(f'{d:.1f}' for d in diffs) + flag)
        if args.video:
            for rn in args.robots:
                out_v = os.path.join(args.out, f'hand_{rn}.mp4')
                info = render_robot_video(p, 0, rn, data, args.hand, out_v,
                                          args.fps, args.video_frames,
                                          args.dist, args.paint)
                if info is None:
                    print(f'  {rn:<8} MP4 输出失败（编码器不可用？）')
                    continue
                print(f'  {rn:<8} 视频 {info["n"]} 帧 @{info["fps"]}fps'
                      f' -> {out_v}')
                print(f'           相机固定（取样帧姿态并集取景）'
                      f' / 参考帧 {info["ref"]} / 手部 AABB '
                      f'中心 {np.round(info["cam"][0], 3)} / '
                      f'距离 {info["cam"][3]:.2f} m')
                png = os.path.join(args.out, f'video_check_{rn}.png')
                diffs = video_check(out_v, png, info['check_pos'])
                if diffs:
                    flag = ('' if max(diffs) >= 1.0
                            else '  [注意] 几乎无变化，需排查')
                    print('           MP4 抽帧自检(姿态最不相同的三帧) 像素平均差 = '
                          + ' '.join(f'{d:.1f}' for d in diffs) + flag)
                    print(f'           抽帧对比图 -> {png}')
        if args.video and args.stacked and len(args.robots) > 1:
            vids = [os.path.join(args.out, f'hand_{rn}.mp4')
                    for rn in args.robots]
            out_s = os.path.join(args.out, 'three_robots.mp4')
            n, counts = stack_videos(vids, args.robots, out_s, args.fps)
            if n:
                print(f'  三台并排(帧同步) {n} 帧 -> {out_s}')
                print(f'           各台帧数 {counts}（一致 = 可逐帧对比）')
            else:
                print('  三台并排拼接失败（帧数不一致或编码器不可用）')
    finally:
        p.disconnect()
    print(f'\n  图片目录：{args.out}')


if __name__ == '__main__':
    main()

