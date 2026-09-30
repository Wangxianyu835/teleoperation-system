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

用法
----
    python scripts/render_hand_snapshots.py
    python scripts/render_hand_snapshots.py --robots gr1_t2 g1 --n 5
    python scripts/render_hand_snapshots.py --file datasets/raw/xxx.h5
    python scripts/render_hand_snapshots.py --hand left --out outputs/hands
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
                       dist, scale):
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
    picks = pick_frames_by_flexion(arr, val, mapping, n_frames)

    joint_ids = [names[jn] for _d, jn, *_ in mapping]
    body_ctr = body_center(p, rid, cid)
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
        ctr, size_vec = link_extent(rid, joint_ids, cid)
        if ctr is None:
            ctr, size_vec = np.zeros(3), np.array([0.15, 0.15, 0.15])
        diag = float(np.linalg.norm(size_vec))
        d = dist if dist > 0 else max(0.35, diag * 1.4)
        eye = hand_eye(ctr, body_ctr, d)
        images.append(render(p, cid, eye, ctr))
        used.append(fr)
        cam = (ctr, size_vec, eye, d)
    p.removeBody(rid, physicsClientId=cid)
    labels = [f'frame {fr}  flex {flex_score(arr[fr], mapping):.2f}'
              for fr in used]
    cam_desc = (f'手部 AABB 中心 {np.round(cam[0], 3)} / 尺寸 '
                f'{np.round(cam[1], 3)} m / 相机距离 {cam[3]:.2f} m')
    return np.hstack(images), labels, len(mapping), cam_desc


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
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.file)

    p.connect(p.DIRECT)
    try:
        p.setTimeStep(DT)
        for rn in args.robots:
            img, labels, n_map, cam = render_robot_strip(
                p, 0, rn, data, args.hand, args.n, args.dist, args.scale)
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
            print(f'           {cam}')
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
    finally:
        p.disconnect()
    print(f'\n  图片目录：{args.out}')


if __name__ == '__main__':
    main()

