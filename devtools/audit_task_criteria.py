"""30 个内置任务的「判据 + 场景物理一致性」体检。

为什么需要它：`sim run --benchmark` 表里全是 0.0% 时，你无法区分
「策略不行」和「任务本身坏了」。这个脚本在**完全不给动作**的情况下跑 1 秒，
把「任务坏掉」的三类症状单独抓出来：

  t0_success       t=0 就 check_success()=True  -> 白送成功（判据是绝对阈值，
                   初始摆位恰好满足）。例：旧的 liftmug / opendrawer / openmicrowave。
  settle_drift     静置 1 s 后物体自己动了 > --max-drift（默认 2 cm）。
                   典型原因：物体中心写成桌面高度 0.42，忽略自己的半高，
                   嵌进桌面 2~10 cm，PyBullet 把它弹起来再落地；
                   或者门/抽屉是自由刚体，被挤飞。
  settled_success  静置结束后 check_success()=True -> 重力自己把任务做完了。
  gravity_success  静置期间**任何一步** check_success()=True（包含上面那种，也包含
                   「被弹起来那一刻成立、落回去又不成立」的闪烁型白送）。
  fell_through     物体中心低于桌面 - 5 cm -> 掉到地上（摆位穿空/无支撑）。
  unreachable      机器人臂展够不到物体（默认 WARN，加 --reach-strict 才算 FAIL）。
                   够不到的任务成功率恒为 0，与「策略不行」是两件事。

用法：
    E:\\python3.11.7\\python.exe devtools\\audit_task_criteria.py
    E:\\python3.11.7\\python.exe devtools\\audit_task_criteria.py --robot gr1_t2 --json out.json
    E:\\python3.11.7\\python.exe devtools\\audit_task_criteria.py --tasks pushcube,liftmug
    E:\\python3.11.7\\python.exe devtools\\audit_task_criteria.py --reach-strict

退出码：0 = 30/30 全过；1 = 有任务不合格（CI/回归门用退出码判断，别只看输出）。
"""

import argparse
import json
import os
import sys

import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from teleoperation.simulation.environment import SimulationEnv            # noqa: E402
from teleoperation.simulation.tasks import list_tasks                    # noqa: E402


def probe_actuation(env, task, max_steps: int = 180, drive_force: float = 200.0) -> dict:
    """驱动门/抽屉的关节，确认「关节真能动」且「动到位能满足判据」。

    只查静置漂移是不够的：一个焊死的门同样「不漂移」。这里反过来验一次。

    实现说明（踩过的坑）：本机 pybullet (202010061) 对「静态底座多体的 link」
    调用 p.applyExternalForce 完全没有反应（200 N 推不动 0.15 kg 抽屉），
    但关节电机与 p.applyExternalTorque 都正常。所以这里用电机驱动代替
    「机器人推」，只验证机构与判据的可达性。
    """
    results = {}
    quarter_turn = float(np.pi / 2.0)
    for name, meta in list(getattr(task, 'panel_joints', {}).items()):
        body = task.objects[name]
        link = task._links[name]
        if meta['kind'] == 'prismatic':
            targets = [float(meta['limits'][0]), float(meta['limits'][1])]
        else:
            targets = [quarter_turn, -quarter_turn]
        outcome = {'moved': False, 'success': False, 'steps': 0,
                   'travel_m': 0.0, 'turned_deg': 0.0}
        angle0 = float(p.getJointState(body, link, physicsClientId=env.client)[0])
        pos0, orn0 = task.get_object_pose(name)
        for target in targets:
            # 每次尝试都先复位到初始关节角：上一次的阻塞/楔死会污染下一次
            p.resetJointState(body, link, angle0, targetVelocity=0.0,
                              physicsClientId=env.client)
            p.stepSimulation(physicsClientId=env.client)
            p.setJointMotorControl2(body, link, p.POSITION_CONTROL,
                                    targetPosition=target, force=float(drive_force),
                                    physicsClientId=env.client)
            for step in range(max_steps):
                p.stepSimulation(physicsClientId=env.client)
                if task.check_success():
                    outcome['success'] = True
                    outcome['steps'] = step + 1
                    break
            pos_after, orn_after = task.get_object_pose(name)
            travelled = float(np.linalg.norm(pos_after - pos0))
            dot = abs(float(np.dot(orn0, orn_after)))
            turned_deg = float(np.degrees(2.0 * np.arccos(min(1.0, dot))))
            if travelled > 0.02 or turned_deg > 5.0:
                outcome['moved'] = True
                outcome['travel_m'] = round(travelled, 4)
                outcome['turned_deg'] = round(turned_deg, 2)
            if outcome['success']:
                break
        results[name] = outcome
    return results


def _hand_length(cid: int, body: int, wrist_index: int) -> float:
    """腕点到**该腕之后**最远手部 link 的距离（世界系），当作「手」能用的那截。

    注意 不能扫全表里名字带 hand/finger 的 link：左右手都带 hand，
    量出来会是「到另一只手」的距离（g1 约 0.9 m），臂展虚高 0.5 m。
    所以只沿腕关节的运动学子树往下走。
    """
    num_joints = p.getNumJoints(body, physicsClientId=cid)
    children = {}
    for j in range(num_joints):
        parent = p.getJointInfo(body, j, physicsClientId=cid)[16]
        children.setdefault(parent, []).append(j)
    wrist_pos = np.array(p.getLinkState(body, wrist_index,
                                        computeForwardKinematics=True,
                                        physicsClientId=cid)[0], dtype=float)
    longest = 0.0
    stack = list(children.get(wrist_index, []))
    while stack:
        j = stack.pop()
        # 注意 getJointInfo()[14] 是 parentFramePos（父连杆系里的关节位置），
        # 不是世界坐标；连杆世界坐标必须用 getLinkState()[0]。
        pos = np.array(p.getLinkState(body, j, computeForwardKinematics=True,
                                      physicsClientId=cid)[0], dtype=float)
        longest = max(longest, float(np.linalg.norm(pos - wrist_pos)))
        stack.extend(children.get(j, []))
    return longest


def probe_reach(env, task) -> dict:
    """机器人够得到场景里的东西吗？（纯几何，和判据无关）

    为什么要单独查：**判据可达 != 机器人够得着**。本机实测三个机器人的手
    （上臂 + 前臂 + 手）只有 0.40~0.65 m 臂展，而 30 个任务把物体摆在
    x=0.5 / z=0.45 —— 离肩点 0.82 m(g1) / 1.10 m(h1_2) / 0.93 m(gr1_t2)。
    这类任务在任何策略下都是 0 分，必须和「策略不行」分开报。

    臂展 = 上臂 + 前臂（FK 关键点量出来）+ 手长（腕点到最远手部 link）。
    距离 = 肩点 -> 物体中心；margin = 臂展 - 距离，< 0 就是够不到。
    """
    body = getattr(env, 'robot_id', None)
    if body is None:
        return {}
    cid = env.client
    try:
        from teleoperation.simulation.arm_ik import PyBulletArmIK
    except Exception as exc:                        # 没装 IK 依赖也别让体检挂掉
        return {'error': f'{type(exc).__name__}: {exc}'}
    objects = {name: np.array(task.get_object_position(name), dtype=float)
               for name in task.objects if name != 'table'}
    if not objects:
        return {}

    sides = {}
    for side in ('left', 'right'):
        try:
            ik = PyBulletArmIK(body, cid, robot_type=env.robot_type, sides=(side,))
            points = ik.arm_points(side)
            wrist_index = ik._limb_link(side, 'wrist')
        except Exception:
            continue
        shoulder = np.asarray(points['shoulder'], dtype=float)
        elbow = np.asarray(points['elbow'], dtype=float)
        wrist = np.asarray(points['wrist'], dtype=float)
        arm_len = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
        reach = arm_len + _hand_length(cid, body, wrist_index)
        nearest = min(objects, key=lambda n: float(np.linalg.norm(objects[n] - shoulder)))
        distance = float(np.linalg.norm(objects[nearest] - shoulder))
        sides[side] = {'reach_m': round(reach, 3), 'nearest': nearest,
                       'distance_m': round(distance, 3),
                       'margin_m': round(reach - distance, 3)}
    if not sides:
        return {}
    best = max(sides.values(), key=lambda entry: entry['margin_m'])
    return {'sides': sides, 'nearest': best['nearest'],
            'best_margin_m': best['margin_m']}


def reach_margin(row: dict) -> float:
    """该任务的「臂展余量」（米）；没测到就返回 0（不报警）。"""
    return float((row.get('reach') or {}).get('best_margin_m', 0.0))


def audit_one(task_name: str, robot: str, settle_steps: int,
              max_drift: float, dt: float) -> dict:
    """跑单个任务：t=0 判据 / 静置漂移 / 静置后判据。"""
    # record=False：体检不需要录像，也不需要 application 提供的 recording 观察者
    env = SimulationEnv(robot_type=robot, task_name=task_name,
                        render=False, record=False)
    row = {'task': task_name, 'robot': robot}
    try:
        env.reset(randomize=False)          # 关随机化：体检要可复现
        task = env.task
        row['t0_success'] = bool(task.check_success())
        row['objects'] = sorted(n for n in task.objects if n != 'table')
        # 机器人够得到吗？（和判据无关的几何事实，见 probe_reach 的说明）
        row['reach'] = probe_reach(env, task)

        start = {name: task.get_object_position(name) for name in task.objects}
        # 静置期间**任何一步**判据变 True 都算「重力/弹跳自己完成」：
        # 只看最后一步会漏掉「被弹起来的那一刻判据成立、落回去又不成立」这种闪烁型白送。
        first_success_step = None
        for step in range(settle_steps):
            p.stepSimulation(physicsClientId=env.client)
            if first_success_step is None and task.check_success():
                first_success_step = step + 1      # 1 = 静置第 1 步
        row['first_success_step'] = first_success_step
        row['gravity_success'] = first_success_step is not None

        drifts = {}
        for name, pos0 in start.items():
            pos1 = task.get_object_position(name)
            drifts[name] = float(abs(pos1[2] - pos0[2]))
            row.setdefault('max_xy_drift', 0.0)
            row['max_xy_drift'] = max(
                row['max_xy_drift'],
                float(((pos1[0] - pos0[0]) ** 2 + (pos1[1] - pos0[1]) ** 2) ** 0.5))
        worst = max(drifts, key=lambda k: drifts[k]) if drifts else None
        row['max_drop_m'] = float(drifts[worst]) if worst else 0.0
        row['max_drop_object'] = worst
        row['settled_success'] = bool(task.check_success())

        # 掉穿桌面：排除本来就摆在地上的家具（桌面以下但仍在 0.1 m 以上的不算）
        below = [name for name in task.objects
                 if name != 'table'
                 and task.get_object_position(name)[2] < task.table_top_z - 0.05]
        row['below_table'] = below
        row['settle_seconds'] = round(settle_steps * dt, 3)

        # 作动探针：门/抽屉推得动吗？推到头能满足判据吗？
        actuation = probe_actuation(env, task)
        if actuation:
            row['actuation'] = actuation
    finally:
        env.close()

    failures = []
    if row['t0_success']:
        failures.append('t0_success')
    if row['max_drop_m'] > max_drift or row['max_xy_drift'] > max_drift:
        failures.append('settle_drift')
    # 静置期间任何一步成立都算被重力/弹跳凑成（含最后一步仍成立的情况）
    if row['gravity_success']:
        failures.append('gravity_success')
    if row['below_table']:
        failures.append('fell_through')
    for panel, outcome in (row.get('actuation') or {}).items():
        if not outcome.get('moved'):
            failures.append(f'joint_stuck:{panel}')
    row['failures'] = failures
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot', default='g1', choices=['h1_2', 'gr1_t2', 'g1'])
    parser.add_argument('--tasks', default='', help='逗号分隔；默认全部 30 个')
    parser.add_argument('--settle-steps', type=int, default=240,
                        help='静置步数（默认 240 = 240 Hz 下的 1 秒）')
    parser.add_argument('--max-drift', type=float, default=0.02,
                        help='允许的自发位移（米，默认 2 cm）')
    parser.add_argument('--json', default='', help='把明细写到这个 JSON')
    parser.add_argument('--reach-strict', action='store_true',
                        help='臂展不足也算 FAIL（默认只 WARN：这是场景布局问题，不是判据问题）')
    args = parser.parse_args()

    task_names = ([t.strip() for t in args.tasks.split(',') if t.strip()]
                  if args.tasks else list_tasks())
    dt = 1.0 / 240.0

    print(f'[audit] robot={args.robot} tasks={len(task_names)} '
          f'settle={args.settle_steps} steps ({args.settle_steps * dt:.2f} s) '
          f'max_drift={args.max_drift * 1000:.0f} mm')
    print(f'{"task":<18}{"t0":<6}{"drift":<9}{"obj":<16}{"grav":<7}{"reach":<9}result')
    print('-' * 72)

    rows = []
    for name in task_names:
        try:
            row = audit_one(name, args.robot, args.settle_steps, args.max_drift, dt)
        except Exception as exc:                        # 单个任务炸掉不算整体崩
            print(f'{name:<18}[ERROR] {type(exc).__name__}: {exc}')
            rows.append({'task': name, 'failures': ['error'], 'error': str(exc)})
            continue
        rows.append(row)
        status = 'PASS' if not row['failures'] else 'FAIL ' + ','.join(row['failures'])
        margin = reach_margin(row)
        reach_text = f'{margin * 100:+.0f}cm'
        print(f'{name:<18}{str(row["t0_success"]):<6}'
              f'{row["max_drop_m"] * 1000:>6.1f}mm '
              f'{(row["max_drop_object"] or "-"):<16}'
              f'{str(row.get("gravity_success")):<7}'
              f'{reach_text:<9}{status}')

    failed = [r for r in rows if r.get('failures')]
    # 臂展：够不到不是「判据坏了」，但一样让成功率恒为 0，所以必须显式报出来
    short = [r for r in rows if reach_margin(r) < 0.0]
    if short and args.reach_strict:
        failed = failed + short
    print('-' * 72)
    print(f'[audit] PASS={len(rows) - len(failed)} FAIL={len(failed)}')
    if failed:
        for row in failed:
            print(f'  - {row["task"]}: {row["failures"]}')

    if short:
        worst = min(short, key=reach_margin)
        nearest = (worst.get('reach') or {}).get('nearest')
        print(f'[audit] WARN reach: {len(short)}/{len(rows)} 个任务的全部物体都在臂展之外'
              f'（最差 {worst["task"]}: margin={reach_margin(worst):.3f} m, nearest={nearest}）')
        print('        这不是策略问题：物体要摆进手臂工作空间（docs/SIM_EVALUATION.md 第 4 节）')
        if not args.reach_strict:
            print('        加 --reach-strict 可把它变成 FAIL')

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({'robot': args.robot, 'settle_steps': args.settle_steps,
                       'max_drift_m': args.max_drift, 'rows': rows},
                      handle, ensure_ascii=False, indent=2)
        print(f'[audit] 明细写入 {args.json}')

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
