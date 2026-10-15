"""把 30 个仿真环境逐个建起来、跑一小段、离屏截图（无需显示器）

用途
----
平台的 30 个分层任务环境（Level 1~4，实现在 tasks/all_tasks.py，注册表见
tasks/task_registry.py）分散在 30 个任务类里，平时只能一个一个进 main.py 看。
本脚本一次性把它们【逐个建起来 -> 跑若干步 -> 出一张截图】，并给一份
实测报告（每个环境里有几个物体、物体落在哪、截图是否正常），
用来回答两个问题：

    1. 30 个环境到底能不能都跑起来？（谁报错、错在哪）
    2. 每个环境长什么样？（30 张图 + 一张 6x5 总览拼图）

为什么不用 GUI 截图
------------------
DIRECT 模式 + getCameraImage 在没有显示器（CI / 远程桌面）时同样出图，
相机按【场景实际包围盒】自动取景，不依赖手工调相机。
要肉眼看窗口时加 --gui（此时不下发离屏相机，画面直接显示在窗口里）。

离屏渲染的坑（与 scripts/render_hand_snapshots.py 同源，改一处记得同步另一处）
--------------------------------------------------------------------------
1. H1-2 的网格材质在 getCameraImage 下渲染成【纯黑剪影】：手臂/手指在图上
   完全看不出 -> 默认 paint_readable() 把整机刷成浅灰（只改外观，不动物理），
   --no-paint 可关掉。
2. ER_TINY_RENDERER 返回的是【扁平 RGBA】像素，必须先 reshape(H, W, 4)；
   但【行序是自上而下、还是自下而上】跟 PyBullet / 渲染器版本有关：
   本机 PyBullet 202010061 + ER_TINY_RENDERER 实测行序已经是自上而下，
   再按旧经验 [::-1] 会得到【上下颠倒】的图（桌子跑到机器人上方），
   所以默认【不翻】，留 --vflip 给换版本/换渲染器时用。
   实测对照：tmp_motion/task_env_smoke（默认、正确）对比 --vflip（颠倒）。
3. 地面 plane.urdf 的 AABB 是无限大，取景时必须排除，只按
   「机器人 + 任务物体」算包围盒（本脚本显式传 body id，不遍历所有 body）。
4. OpenCV 在 Windows 上按 ANSI 代码页打开文件路径 -> 路径含中文时
   `cv2.imwrite` **静默失败**、`cv2.imread` 读不出来。本脚本一律走
   `cv2.imencode` / `cv2.imdecode` + Python 二进制读写（见 save_cv_image），
   所以 `--out` / `--copy-to` 给中文目录（例如 C:\\Users\\王宪雨\\...）也能用。

用法
----
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py --robot gr1_t2 --levels 1 2
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py --tasks pushcube pickcube
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py --steps 960 --size 1280 960
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py --gui          # 开窗口一个个看
    E:\\python3.11.7\\python.exe scripts\\capture_task_envs.py --copy-to "C:\\Users\\<you>\\Pictures\\Screenshots"

产物（默认目录 outputs/task_env_shots/，该目录被 .gitignore 排除，不入库）
------------------------------------------------------------------------
    01_pushcube_L1.png ... 30_stacktoyblocks_L4.png   每个环境一张
    00_overview_all30.png                             6x5 拼图（一眼看全 30 个）
    report.json                                       每个环境的实测数据（可机读）
    report.txt                                        人能读的汇总表

退出码：0 = 所要求的环境全部建成且截图正常；1 = 有环境失败（报告里能看到错因）
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 默认相机方向（从场景中心往外推的方向）：
# 站在操作台【远侧偏左、略高】往机器人正面看 —— 桌面上的物体不被机器人挡住
CAM_DIR = np.array([0.95, -1.00, 0.42])
CAM_UP = [0.0, 0.0, 1.0]


def task_universe():
    """返回 [(任务名, 难度级)]，顺序 = Level 1..4 内的注册顺序"""
    import tasks.all_tasks          # noqa: F401  触发 30 个任务注册
    from tasks import get_tasks_by_level, list_tasks

    pairs = []
    for level in (1, 2, 3, 4):
        for name in get_tasks_by_level(level):
            pairs.append((name, level))
    # 防御：注册表里有、但没被列进任何 level 的任务也要跑（否则"30 个"会悄悄少）
    listed = {name for name, _ in pairs}
    for name in list_tasks():
        if name not in listed:
            pairs.append((name, 0))
    return pairs


def paint_readable(p, cid, rid):
    """把整机视觉颜色刷成浅灰（只改外观，不动物理）

    [必须] H1-2 的网格材质在 getCameraImage 下渲染成纯黑剪影，
    刷成浅灰后手臂/手指轮廓才可读（细节见模块 docstring 第 1 条）。
    """
    for i in range(-1, p.getNumJoints(rid, physicsClientId=cid)):
        try:
            p.changeVisualShape(rid, i, rgbaColor=[0.82, 0.83, 0.88, 1.0],
                                physicsClientId=cid)
        except p.error:
            continue


def scene_bounds(p, cid, ids):
    """场景包围盒（只看机器人 + 任务物体，排除无限大的地面）"""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for body in ids:
        if body is None:
            continue
        for link in range(-1, p.getNumJoints(body, physicsClientId=cid)):
            try:
                a, b = p.getAABB(body, link, physicsClientId=cid)
            except p.error:
                continue
            lo = np.minimum(lo, np.asarray(a, dtype=float))
            hi = np.maximum(hi, np.asarray(b, dtype=float))
    if not np.isfinite(lo).all():
        return None, None
    return lo, hi


def camera_pose(bounds, fov, fit, aspect):
    """按包围盒算相机（eye, target）

    [为什么不是球拟合] 第一版用「半径 / tan(fov/2)」把包围盒当球来算，
    实测取景太远（30 个环境的桌子 1.0 x 1.6 m + 机器人 1.8 m 高，
    对角线 2.6 m 当直径 -> 相机退到 3.2 m，画面里一半是空地板）。
    这里改成【把 8 个角点投到相机基上，逐一算"要让它在画面里需要多远"，
    取最大值】-> 画面正好贴边。

    Args:
        bounds: (lo, hi) 场景包围盒
        fov:    垂直视场角（度）
        fit:    余量系数（1.0 = 刚好贴边）
        aspect: 宽/高（水平方向按它放宽）
    """
    lo, hi = bounds
    center = (lo + hi) / 2.0
    forward = CAM_DIR / float(np.linalg.norm(CAM_DIR))
    right = np.cross(forward, CAM_UP)
    right = right / float(np.linalg.norm(right))
    up = np.cross(right, forward)
    tan_v = np.tan(np.radians(fov) / 2.0)
    tan_h = tan_v * float(aspect)
    need = 0.0
    for sx in (lo[0], hi[0]):
        for sy in (lo[1], hi[1]):
            for sz in (lo[2], hi[2]):
                rel = np.array([sx, sy, sz]) - center
                toward_cam = float(np.dot(rel, forward))
                x = abs(float(np.dot(rel, right)))
                y = abs(float(np.dot(rel, up)))
                need = max(need, toward_cam + x / tan_h, toward_cam + y / tan_v)
    dist = fit * max(need, 0.3)
    return center + forward * dist, center


def render(p, cid, size, eye, target, fov, vflip=False):
    """离屏渲染一帧 -> (H, W, 3) uint8 RGB

    vflip：把行序反过来。本机不需要（见 docstring 坑 2），换渲染器时再试。
    """
    width, height = size
    view = p.computeViewMatrix(cameraEyePosition=[float(v) for v in eye],
                               cameraTargetPosition=[float(v) for v in target],
                               cameraUpVector=CAM_UP, physicsClientId=cid)
    proj = p.computeProjectionMatrixFOV(fov, width / float(height),
                                        0.05, 12.0, physicsClientId=cid)
    _w, _h, rgb, _depth, _seg = p.getCameraImage(
        width, height, view, proj, renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid)
    img = np.asarray(rgb, dtype=np.uint8)
    if img.ndim == 1:      # 坑 2：扁平 RGBA（硬件渲染器才直接给 (H, W, 4)）
        img = img.reshape(height, width, 4)
        if vflip:
            img = img[::-1]
    return img[:, :, :3].copy()


def frame_stats(img):
    """截图自检：像素均值/标准差/颜色数/背景占比

    目的：抓住「渲染失败 -> 纯色/纯黑图」这种"文件有了、内容是空的"情况，
    不要只看文件存在就当成功（约定 5：结论要不变量支撑）。
    """
    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = counts.max() / float(len(flat))
    return {
        'pixel_mean': round(float(flat.mean()), 2),
        'pixel_std': round(float(flat.std()), 2),
        'unique_colors': int(len(colors)),
        'background_ratio': round(float(bg), 4),
    }


def controller_for(env):
    """复用 main.py 的演示控制器（单一实现，避免这里再抄一份正弦波）

    main.demo_controller 按模块级 _ACTION_DIM 生成动作，所以先把当前环境的
    真实动作维度写回去（H1-2=38 / GR1-T2=36 / G1=28，动作空间由机器人决定）。
    """
    import main as platform_main
    platform_main._ACTION_DIM = int(env.action_dim)
    return platform_main.demo_controller


def run_one(task_name, level, args, index, total):
    """建环境 -> 跑 --steps 步 -> 截图；返回一条报告记录"""
    import pybullet as p
    from envs import SimulationEnv

    rec = {'index': index, 'task': task_name, 'level': level,
           'robot': args.robot, 'status': 'FAIL', 'error': None,
           'png': None, 'objects': {}, 'steps': 0, 'success': False}
    env = None
    t0 = time.time()
    try:
        env = SimulationEnv(robot_type=args.robot, task_name=task_name,
                            render=args.gui, record=False,
                            data_dir=args.data_dir)
        env.reset(randomize=args.randomize)
        rec['action_dim'] = int(env.action_dim)
        doc = (env.task.__class__.__doc__ or '').strip().splitlines()
        rec['doc'] = doc[0] if doc else ''
        controller = controller_for(env)

        steps = 0
        success = False
        for _ in range(args.steps):
            _obs, success, done, _elapsed = env.step(controller(None))
            steps += 1
            if done:            # 任务自己判成功/超时（少数任务一 reset 就判成功）
                break
        rec['steps'] = steps
        rec['success'] = bool(success)

        for name, obj_id in env.task.objects.items():
            pos, _orn = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=env.client)
            rec['objects'][name] = [round(float(v), 4) for v in pos]

        if args.gui:
            rec['status'] = 'OK'
            rec['note'] = 'GUI 模式：画面已在窗口里显示，未写 PNG'
            env.close()
            rec['seconds'] = round(time.time() - t0, 2)
            return rec

        if not args.no_paint:
            paint_readable(p, env.client, env.robot_id)
        ids = [env.robot_id] + list(env.task.objects.values())
        lo, hi = scene_bounds(p, env.client, ids)
        rec['scene_bounds'] = (None if lo is None else
                               [[round(float(v), 3) for v in lo],
                                [round(float(v), 3) for v in hi]])
        if args.eye is not None:
            eye = np.asarray(args.eye, dtype=float)
            target = np.asarray(args.target, dtype=float)
        else:
            eye, target = camera_pose((lo, hi), args.fov, args.fit,
                                      args.width / float(args.height))
        rec['eye'] = [round(float(v), 3) for v in eye]
        rec['target'] = [round(float(v), 3) for v in target]

        img = render(p, env.client, (args.width, args.height), eye, target,
                     args.fov, args.vflip)
        rec.update(frame_stats(img))
        path = os.path.join(args.out,
                            f'{index:02d}_{task_name}_L{level}.png')
        write_png(path, img)
        rec['png'] = path
        rec['png_bytes'] = os.path.getsize(path)
        if rec['pixel_std'] < 2.0 or rec['unique_colors'] < 50:
            rec['error'] = (f"截图内容近乎纯色 (std={rec['pixel_std']}, "
                            f"colors={rec['unique_colors']}) —— 渲染可能没生效")
        else:
            rec['status'] = 'OK'
        env.close()
    except Exception as error:                                     # noqa: BLE001
        rec['error'] = f'{type(error).__name__}: {error}'
        rec['traceback'] = traceback.format_exc().strip().splitlines()[-6:]
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    rec['seconds'] = round(time.time() - t0, 2)
    flag = '[OK]  ' if rec['status'] == 'OK' else '[FAIL]'
    print(f'  {flag} {index:>2}/{total} L{level} {task_name:<18} '
          f'steps={rec["steps"]:<4} obj={len(rec["objects"]):<2} '
          f'{rec["seconds"]:>5.2f}s'
          + ('' if rec['status'] == 'OK' else f'  <- {rec["error"]}'))
    return rec


def save_cv_image(path, bgr):
    """保存 OpenCV 图（BGR）-> PNG

    [必须] 不用 `cv2.imwrite`：OpenCV 在 Windows 上按 **ANSI 代码页** 打开文件
    路径，路径里含中文（例如用户目录 `C:\\Users\\王宪雨\\...`）时会**静默返回
    False、什么都不写**（不抛异常）；`imencode` + 二进制写让 Python 处理路径，
    中文目录也能正常出图。
    """
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError(f'cv2.imencode 失败: {path}')
    with open(path, 'wb') as fh:
        fh.write(buf.tobytes())


def read_cv_image(path):
    """读 OpenCV 图（BGR）—— 同 save_cv_image：`cv2.imread` 打不开中文路径"""
    import cv2
    with open(path, 'rb') as fh:
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_png(path, rgb):
    """写 PNG（入参是 RGB，(H, W, 3)）"""
    import cv2
    save_cv_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def overview_sheet(records, out, size, cols=6):
    """把所有截图缩小后贴成一张总览图，一眼看全 30 个环境"""
    import cv2
    ok = [r for r in records if r.get('png') and os.path.exists(r['png'])]
    if not ok:
        return None
    tw, th = max(size[0] // 3, 160), max(size[1] // 3, 120)
    rows = int(np.ceil(len(ok) / float(cols)))
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for k, rec in enumerate(ok):
        img = read_cv_image(rec['png'])
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        label = f"{rec['task']} L{rec['level']}"
        cv2.rectangle(img, (0, 0), (tw, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = img
    path = os.path.join(out, '00_overview_all%02d.png' % len(ok))
    save_cv_image(path, sheet)
    return path


def write_report(records, args, sheet, out):
    """写 report.json / report.txt，并返回 (失败数, 成功数)"""
    ok = [r for r in records if r['status'] == 'OK']
    bad = [r for r in records if r['status'] != 'OK']
    payload = {
        'robot': args.robot,
        'requested': len(records),
        'ok': len(ok),
        'failed': len(bad),
        'steps_per_env': args.steps,
        'size': [args.width, args.height],
        'randomize': bool(args.randomize),
        'overview': sheet,
        'records': records,
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'report.json'), 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    lines = ['=' * 78,
             f"30 个仿真环境截图报告  robot={args.robot}  "
             f"共请求 {len(records)} 个环境",
             '=' * 78,
             f"{'#':>3} {'L':>2} {'task':<18} {'obj':>4} {'steps':>5} "
             f"{'sec':>6} {'std':>7} {'colors':>7}  status"]
    for rec in records:
        lines.append(
            f"{rec['index']:>3} {rec['level']:>2} {rec['task']:<18} "
            f"{len(rec['objects']):>4} {rec['steps']:>5} "
            f"{rec.get('seconds', 0):>6.2f} "
            f"{rec.get('pixel_std', 0):>7.2f} "
            f"{rec.get('unique_colors', 0):>7}  {rec['status']}"
            + ('' if rec['status'] == 'OK' else f"  {rec['error']}"))
    lines.append('-' * 78)
    lines.append(f"成功 {len(ok)} / {len(records)}；失败 {len(bad)}")
    if sheet:
        lines.append(f"总览拼图: {sheet}")
    for rec in bad:
        lines.append(f"[FAIL] {rec['task']}: {rec['error']}")
        for tb in rec.get('traceback', []):
            lines.append('        ' + tb)
    text = '\n'.join(lines)
    with open(os.path.join(out, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    print('\n' + text)
    return len(bad), len(ok)


def copy_out(records, sheet, dest):
    """把截图复制到用户指定目录（--copy-to）

    注意：--copy-to 是用户【明确要求】的落点（例如自己的
    Pictures\\Screenshots）。约定 1 禁的是项目产物落 C 盘：依赖 / 缓存 /
    临时文件；仓库里始终保留一份权威副本（outputs/，已被 .gitignore 排除）。
    """
    os.makedirs(dest, exist_ok=True)
    files = [r['png'] for r in records
             if r.get('png') and os.path.exists(r['png'])]
    if sheet and os.path.exists(sheet):
        files.append(sheet)
    total = 0
    for src in files:
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
        total += os.path.getsize(src)
    print(f"  已复制 {len(files)} 个文件（{total / 1024.0 / 1024.0:.2f} MB）"
          f" -> {dest}")
    return len(files)


def main():
    ap = argparse.ArgumentParser(
        description='把 30 个仿真环境逐个跑起来并离屏截图')
    ap.add_argument('--robot', default='h1_2',
                    choices=['h1_2', 'gr1_t2', 'g1'], help='机器人类型')
    ap.add_argument('--tasks', nargs='+', default=None,
                    help='只跑这些任务（默认全部 30 个）')
    ap.add_argument('--levels', nargs='+', type=int, default=None,
                    help='只跑这些难度级（1~4）')
    ap.add_argument('--steps', type=int, default=480,
                    help='每个环境跑多少物理步（默认 480 = 2 秒 @240Hz）')
    ap.add_argument('--size', nargs=2, type=int, default=[960, 720],
                    metavar=('W', 'H'), help='截图分辨率（默认 960x720）')
    ap.add_argument('--fov', type=float, default=50.0, help='相机视场角')
    ap.add_argument('--fit', type=float, default=1.05,
                    help='取景余量：1.0 = 场景包围盒刚好贴边（默认 1.05）')
    ap.add_argument('--eye', nargs=3, type=float, default=None,
                    metavar=('X', 'Y', 'Z'), help='手动指定相机位置')
    ap.add_argument('--target', nargs=3, type=float, default=None,
                    metavar=('X', 'Y', 'Z'),
                    help='手动指定相机注视点（必须与 --eye 同时给出）')
    ap.add_argument('--randomize', action='store_true',
                    help='开启域随机化（默认关：截图要可复现、可比）')
    ap.add_argument('--gui', action='store_true',
                    help='开 PyBullet 窗口一个个看（不下发离屏相机，不写 PNG）')
    ap.add_argument('--no-paint', action='store_true',
                    help='不刷浅灰：H1-2 网格材质在离屏渲染下是纯黑剪影，'
                         '默认刷成浅灰才看得清手臂')
    ap.add_argument('--vflip', action='store_true',
                    help='把离屏图的行序反过来。本机不需要（实测行序已经正确）；'
                         '换 PyBullet / 渲染器版本后若出现【上下颠倒】的图再打开')
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs',
                                                  'task_env_shots'))
    ap.add_argument('--data-dir',
                    default=os.path.join(ROOT, 'outputs', 'task_env_shots',
                                         '_recorder'),
                    help='记录器目录（默认 --no-record，这里只是给构造器一个落点）')
    ap.add_argument('--copy-to', default=None,
                    help='额外把截图复制到这个目录（例如自己的图片文件夹）')
    args = ap.parse_args()

    if (args.eye is None) != (args.target is None):
        raise SystemExit('--eye 与 --target 必须同时给出')
    args.width, args.height = int(args.size[0]), int(args.size[1])

    os.makedirs(args.out, exist_ok=True)
    universe = task_universe()
    if args.levels:
        universe = [x for x in universe if x[1] in args.levels]
    if args.tasks:
        want = [t.lower() for t in args.tasks]
        known = {name for name, _ in task_universe()}
        missing = [t for t in want if t not in known]
        if missing:
            raise SystemExit(f'未知任务: {missing}（见 tasks/task_registry.py）')
        universe = [x for x in universe if x[0] in want]
    if not universe:
        raise SystemExit('没有匹配到任何任务')

    print('=' * 78)
    print(f"30 个仿真环境截图  robot={args.robot}  环境数={len(universe)}  "
          f"每环境 {args.steps} 步  截图 {args.width}x{args.height}")
    print(f"输出目录: {args.out}")
    print('=' * 78)

    records = []
    for index, (task_name, level) in enumerate(universe, 1):
        records.append(run_one(task_name, level, args, index, len(universe)))

    if args.gui:
        print(f"\n  GUI 模式结束：{len(records)} 个环境已逐个显示")
        return 0

    sheet = overview_sheet(records, args.out, (args.width, args.height))
    failed, _ok = write_report(records, args, sheet, args.out)
    if args.copy_to:
        copy_out(records, sheet, args.copy_to)
    print(f"\n  截图目录: {args.out}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())



