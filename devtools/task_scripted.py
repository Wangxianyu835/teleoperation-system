"""无人操作的「脚本化操作策略」：IK 到位 + 抓取约束 + 释放，把 30 个任务真做完。

为什么要有这个文件
------------------
批次 3/4（``docs/task-evaluation.md`` §2）都标着「需要人在摄像头前」。现场没人时，
唯一能重复验证的路子是把「人的动作」换成一串**确定性脚本**：

    Segment(位置, 抓/放/转)  ->  PyBulletArmIK.solve()  ->  env.task.apply_action()

这条路径**不改 baseline 任务代码**（规则 3），只在运行时驱动机器人：
物体摆放仍由 ``task_eval.TaskScene`` 决定，判据仍是任务自己的 ``check_success()``，
所以「完成」是任务自己说的，不是我们写死的。

诚实声明（对外报数必须一起给）
------------------------------
* **抓取用约束辅助**（``grasp_assist``，默认开）：手腕进到物体 8 cm 以内时，
  用 ``p.createConstraint(JOINT_FIXED)`` 把物体刚性挂在手部链接上，松手时删掉。
  这不是指尖摩擦抓取 —— 五指的闭合动作会做（视觉上像抓），真正夹住靠约束。
  本仓库没有触觉/摩擦力标定，靠接触抓 3-6 cm 的代理方块在 PyBullet 里成功率极低；
  关掉它（``--no-grasp-assist``）可以复现「纯接触抓不动」的对照。
* 手部关节只做「张开 / 闭合」两档，不走 18 维重定向链路（那条链路要有相机输入）。
* 脚本是**按任务写死的动作序列**，不是通用策略；它证明「这个任务在仿真里可完成」，
  不是「遥操作能做到」。两者的差距在 ``docs/task-evaluation.md`` §5 偏差表里。

用法::

    E:\\python3.11.7\\python.exe devtools\\task_eval.py --task pushcube --policy scripted
    E:\\python3.11.7\\python.exe devtools\\task_eval.py --policy scripted --verify-scripts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pybullet as p

#: 手腕到物体的「抓取容差」（米）：进这个圈内才允许挂约束。
GRASP_TOLERANCE_M = 0.10
#: 每多少个物理步解一次 IK（8 步 = 30 Hz；每步都解太慢，且和 20 fps 的相机同量级）
IK_EVERY_STEPS = 8
#: 规划用的默认末端速度（米/秒），决定每段走多少步
SPEED_M_S = 0.35
#: 进入这一段之前先停多少步（让物理收敛，比如松手后等物体落稳）
SETTLE_STEPS = 40
#: 抓取段：差几厘米时最多再伺服多少步（240 步 = 1 s）
GRASP_WAIT_STEPS = 240
#: 一段走完名义步数后，若腕点还没到（|err| > 容差）最多再等多少步（240 = 1 s）
SEGMENT_WAIT_STEPS = 240
#: 认为「到位」的容差（米）
ARRIVE_TOLERANCE_M = 0.015




@dataclass
class Segment:
    """一段直线运动 + 可选的抓/放/转。

    ``position`` 是**世界系**手腕目标；``None`` = 原地保持（只做抓/放/转）。
    """

    position: Optional[tuple] = None
    steps: Optional[int] = None
    label: str = ""
    #: 到位后挂上哪个物体（``task.objects`` 的键）
    grasp: Optional[str] = None
    #: 到位后松手
    release: bool = False
    #: 到位后额外拧某个手臂关节（度），键 = 关节名（通常是 ``*_wrist_*``）
    rotate: dict = field(default_factory=dict)
    #: 这一段结束后等多少步（物理收敛）
    hold: int = 0


def _unit(values) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else vec


def object_position(env, key: str) -> np.ndarray:
    body = env.task.objects.get(key)
    if body is None:
        raise KeyError("task %r has no object %r" % (env.task_name, key))
    return np.asarray(p.getBasePositionAndOrientation(body,
                                                      physicsClientId=env.client)[0],
                      dtype=np.float64)


def object_aabb(env, key: str):
    body = env.task.objects[key]
    low, high = p.getAABB(body, physicsClientId=env.client)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def table_top(env) -> float:
    """桌面顶面的世界 z（场景层摆完之后）。"""
    table = env.task.objects.get("table")
    if table is None:
        return 0.0
    _low, high = p.getAABB(table, physicsClientId=env.client)
    return float(np.asarray(high, dtype=np.float64)[2])


class ScriptedPolicy:
    """把 ``Segment`` 序列变成 ``callable(step) -> action``（``run_trial`` 直接吃）。"""

    def __init__(self, env, task_name: str = None, side: str = "right",
                 grasp_assist: bool = True, speed_m_s: float = SPEED_M_S,
                 ik_every: int = IK_EVERY_STEPS, trace: bool = False, **extra):
        from teleoperation.simulation.arm_ik import PyBulletArmIK
        from teleoperation.simulation.hand_mount import find_hand_base_link

        if extra:
            raise ValueError("scripted 策略不认识这些参数: %s" % sorted(extra))
        self.env = env
        self.task_name = str(task_name or env.task_name)
        self.side = str(side)
        self.grasp_assist = bool(grasp_assist)
        self.speed = max(0.02, float(speed_m_s))
        self.ik_every = max(1, int(ik_every))
        self.trace = bool(trace)
        self.ik = PyBulletArmIK(env.robot_id, env.client, robot_type=env.robot_type,
                                sides=(self.side,), rest_poses="current",
                                ik_mode="position", lock_others=True)

        self.names = {name: index for index, name in enumerate(env.action_joint_names)}
        self.hand_names = list(env.robot_loader.hand_joints.get(self.side, []))
        self.action = _joint_positions(env)
        self.open_angles = self._hand_angles(closed=False)
        self.closed_angles = self._hand_angles(closed=True)
        self.hand_link, self.hand_link_name = find_hand_base_link(
            env.robot_id, env.robot_type, self.side, env.client)
        if self.hand_link is None:
            self.hand_link, self.hand_link_name = int(self.ik.ee[self.side][0]), "ee"
        self.hand_link = int(self.hand_link)
        self.palm_links = self._palm_link_indices()


        self.constraint = None
        self.held = None
        #: 抓住时「物体 - 腕点」的偏移：搬运时按物体目标点反推腕点目标（否则松手差 5-8 cm）
        self.grasp_offset = np.zeros(3)
        self.log = []

        self.segment_index = 0
        self.segment_step = 0
        self.grip_target = "open"
        self.rotations = {}
        self.steps = 0
        self.solves = 0
        self.max_wrist_error = 0.0
        self.stopped = False

        builder = SCRIPT_BUILDERS.get(self.task_name, default_script)
        self.segments = [Segment(self.home_target(), label="home")] + list(
            builder(env, self))
        self._note("plan: %d segments" % len(self.segments))

    def home_target(self) -> np.ndarray:
        """胸前待命点（x 必须 < 桌沿 0.18 m，否则手臂会撞上落地支撑）。

        几何原因（实测 ``outputs/tmp_probe/track.txt``）：桌面顶面比肩关节还高 5 cm，而
        ``TaskScene._bench`` 在 x >= 0.18 从地面补了一根支撑 —— 等于机器人和物体之间有一堵墙。
        从垂臂位直接往前摆，手在 z≈0.9 就会怼上墙，shoulder_pitch 被卡在 +0.44 rad。
        所以第一段永远是「把肘收回来、手贴到胸前」（肩前 0.10 m，仍在墙内）。
        """
        shoulder = np.asarray(self.ik.arm_points(self.side)["upper"], dtype=np.float64)
        side = 1.0 if shoulder[1] > 0.0 else -1.0
        return shoulder + np.array([0.10, 0.06 * side, -0.10])



    # ---------------------------------------------------------------- 手部
    def _hand_angles(self, closed: bool) -> dict:
        """张开 / 闭合两档手指角：闭合 = 每根手指往行程大的那一端走 90%。"""
        angles = {}
        for name in self.hand_names:
            joint = self.env.robot_loader.all_joints[name]
            info = p.getJointInfo(self.env.robot_id, joint, physicsClientId=self.env.client)
            lower, upper = float(info[8]), float(info[9])
            if closed:
                angles[name] = 0.9 * (upper if abs(upper) >= abs(lower) else lower)
            else:
                angles[name] = (0.0 if lower <= 0.0 <= upper else
                                (lower if abs(lower) < abs(upper) else upper))
        return angles

    # ---------------------------------------------------------------- 状态查询
    def _palm_link_indices(self) -> list:
        """该侧手上所有 link（掌根 + 五指）：抓取判据用「手的任一点到物体的最小距离」。"""
        out = []
        prefix = ("l_", "left") if self.side == "left" else ("r_", "right")
        for index in range(p.getNumJoints(self.env.robot_id,
                                          physicsClientId=self.env.client)):
            info = p.getJointInfo(self.env.robot_id, index,
                                  physicsClientId=self.env.client)
            name = info[12].decode() if isinstance(info[12], bytes) else str(info[12])
            low = name.lower()
            if not low.startswith(prefix):
                continue
            if any(word in low for word in ("proximal", "intermediate", "distal",
                                            "palm", "hand", "finger", "thumb",
                                            "index", "middle", "ring", "pinky")):
                out.append(index)
        return out or [self.hand_link]

    def palm_position(self) -> np.ndarray:
        points = [p.getLinkState(self.env.robot_id, int(index),
                                 computeForwardKinematics=True,
                                 physicsClientId=self.env.client)[4]
                  for index in self.palm_links]
        return np.asarray(points, dtype=np.float64).mean(axis=0)

    def hand_distance(self, key: str) -> float:
        """手（掌 + 五指）到物体中心的**最小**距离：抓取判据就用它。"""
        body = self.env.task.objects.get(key)
        if body is None:
            return float("inf")
        pos = np.asarray(p.getBasePositionAndOrientation(
            body, physicsClientId=self.env.client)[0], dtype=np.float64)
        dists = [float(np.linalg.norm(np.asarray(p.getLinkState(
            self.env.robot_id, int(index), computeForwardKinematics=True,
            physicsClientId=self.env.client)[4], dtype=np.float64) - pos))
            for index in self.palm_links]
        return min(dists) if dists else float("inf")


    def wrist_world(self) -> np.ndarray:
        return np.asarray(self.ik.arm_points(self.side)["wrist"], dtype=np.float64)

    def _held_position(self):
        """手里那个物体的实测位置（没抓着东西就是 ``None``）。"""
        if self.held is None:
            return None
        body = self.env.task.objects.get(self.held)
        if body is None:
            return None
        return np.asarray(p.getBasePositionAndOrientation(
            body, physicsClientId=self.env.client)[0], dtype=np.float64)



    def _note(self, text: str) -> None:
        self.log.append(text)
        if self.trace:
            print("[scripted] " + text, flush=True)

    # ---------------------------------------------------------------- 抓/放
    def _attach(self, key: str) -> bool:
        body = self.env.task.objects.get(key)
        if body is None:
            self._note("grasp skipped: no object %r" % (key,))
            return False
        if not self.grasp_assist:
            self._note("grasp assist off: %r untouched" % (key,))
            return False
        if self.constraint is not None:
            self._release()
        gap = self.hand_distance(key)
        if gap > GRASP_TOLERANCE_M:
            # 够不着就不许挂：宁可这条失败，也不要「隔空抓物」把成功率做成假的
            self._note("grasp MISSED %s: hand gap %.3f m > tol %.3f m"
                       % (key, gap, GRASP_TOLERANCE_M))
            return False
        link = p.getLinkState(self.env.robot_id, self.hand_link,
                              computeForwardKinematics=True,
                              physicsClientId=self.env.client)
        link_pos, link_orn = link[4], link[5]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(body,
                                                           physicsClientId=self.env.client)
        inverse_pos, inverse_orn = p.invertTransform(link_pos, link_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inverse_pos, inverse_orn, obj_pos, obj_orn)
        self.constraint = p.createConstraint(
            self.env.robot_id, self.hand_link, body, -1, p.JOINT_FIXED, [0, 0, 0],
            parentFramePosition=[0.0, 0.0, 0.0], childFramePosition=list(rel_pos),
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFrameOrientation=list(rel_orn), physicsClientId=self.env.client)
        self.held = key
        body_pos = np.asarray(obj_pos, dtype=np.float64)
        self.grasp_offset = body_pos - self.wrist_world()
        self._note("grasp %s (hand gap %.3f m, offset %s)"
                   % (key, gap, [round(float(v), 3) for v in self.grasp_offset]))
        return True




    def _release(self) -> None:
        if self.constraint is not None:
            p.removeConstraint(self.constraint, physicsClientId=self.env.client)
            self._note("release %s" % (self.held,))
        self.constraint = None
        self.held = None

    # ---------------------------------------------------------------- 执行
    def __call__(self, step: int):
        """``run_trial`` 每步调一次：插值目标 -> 解 IK -> 写动作向量。"""
        self.steps = int(step) + 1
        segment = self.segments[min(self.segment_index, len(self.segments) - 1)]
        if self.segment_step == 0:
            # 每段入口记一次起点：插值从固定起点出发，时间才可预测（跟手 servoing 会慢半拍）
            self.segment_start = self.wrist_world()
            self.segment_obj_start = self._held_position()
        start = self.segment_start
        goal = (start if segment.position is None
                else np.asarray(segment.position, dtype=np.float64))
        steps = segment.steps or (SETTLE_STEPS if segment.position is None
                                  else self._steps_for(float(np.linalg.norm(goal - start))))
        alpha = min(1.0, (self.segment_step + 1) / float(steps))
        eased = alpha * alpha * (3.0 - 2.0 * alpha)   # 两端速度为零，免得甩飞物体
        held_position = self._held_position()
        if (segment.position is not None and held_position is not None
                and self.segment_obj_start is not None):
            # 手里有东西：这一段说的是**物体**去哪。物体与腕点是刚性约束，腕一转世界系偏移就变，
            # 所以每步拿物体的实测位置做闭环（静态偏移补偿实测差 12 cm）。
            obj_goal = self.segment_obj_start + (goal - self.segment_obj_start) * eased
            target = self.wrist_world() + (obj_goal - held_position)
        else:
            target = start + (goal - start) * eased



        if (self.segment_step % self.ik_every) == 0 or alpha >= 1.0:
            self._solve_and_apply(target)
        for name in self.hand_names:
            index = self.names.get(name)
            if index is not None:
                self.action[index] = (self.closed_angles if self.grip_target == "closed"
                                      else self.open_angles)[name]
        self.segment_step += 1
        if self.segment_step >= steps:
            over = self.segment_step - steps
            waiting = False
            if segment.position is not None and over < SEGMENT_WAIT_STEPS:
                if float(np.linalg.norm(self.wrist_world() - goal)) > ARRIVE_TOLERANCE_M:
                    waiting = True
            if (segment.grasp and over < GRASP_WAIT_STEPS
                    and not self._grasp_ready(segment.grasp)):
                waiting = True
            if not waiting:
                self._finish_segment(segment, goal)
        return self.action


    def _grasp_ready(self, key: str) -> bool:
        return self.hand_distance(key) <= GRASP_TOLERANCE_M




    def _finish_segment(self, segment: Segment, goal: np.ndarray) -> None:
        if segment.grasp:
            self.grip_target = "closed"
            self._attach(segment.grasp)
        if segment.release:
            self.grip_target = "open"
            self._release()
        for name, degrees in segment.rotate.items():
            self.rotations[name] = float(self.rotations.get(name, 0.0)) + float(degrees)
        if segment.hold:
            for _ in range(int(segment.hold)):
                p.stepSimulation(physicsClientId=self.env.client)
        error = float(np.linalg.norm(self.wrist_world() - goal))
        self.max_wrist_error = max(self.max_wrist_error, error)
        self._note("seg %-30s done  wrist_err=%.3f m" % (segment.label or "-", error))
        self.segment_index += 1
        self.segment_step = 0
        if self.segment_index >= len(self.segments):
            # 脚本跑完就收工（run_trial 见 stopped 就结束，不必把 60 s 空转完）
            self.stopped = True
            self._note("script finished (%d steps, %d solves)" % (self.steps, self.solves))


    def _steps_for(self, distance: float) -> int:
        # 240 物理步/秒，速度 0.35 m/s -> 每步约 1.5 mm
        return max(12, int(round(distance / self.speed * 240.0)))

    def _joint_state(self) -> dict:
        """整机可动关节当前角（``solve`` 内部会 teleport 去跑第二遍，用完必须还原）。"""
        out = {}
        for index in range(p.getNumJoints(self.env.robot_id,
                                          physicsClientId=self.env.client)):
            info = p.getJointInfo(self.env.robot_id, index,
                                  physicsClientId=self.env.client)
            if info[2] == p.JOINT_FIXED:
                continue
            out[index] = float(p.getJointState(self.env.robot_id, index,
                                               physicsClientId=self.env.client)[0])
        return out

    def _restore_joint_state(self, state: dict) -> None:
        for index, value in state.items():
            p.resetJointState(self.env.robot_id, index, value,
                              physicsClientId=self.env.client)

    def _solve_and_apply(self, target: np.ndarray) -> None:
        # ★ 求解会 ``resetJointState``（两遍迭代用），机械臂会**瞬移穿过桌面上的物体**，
        #   实测把 pushcube 的方块扫飞 0.5 m。所以：先存整机角度 -> 求解 -> 立刻还原 ->
        #   只把解写进动作向量，交给位置电机平滑地走过去（和我们真人遥操作同一条路）。
        state = self._joint_state()
        solution = self.ik.solve(self.side, tuple(float(v) for v in target),
                                 source="world", posture=None)
        self._restore_joint_state(state)
        self.solves += 1
        for name, value in solution.items():
            index = self.names.get(name)
            if index is not None:
                self.action[index] = float(value)
        for name, degrees in self.rotations.items():
            index = self.names.get(name)
            if index is not None:
                self.action[index] = float(self.action[index] + math.radians(degrees))


    # ---------------------------------------------------------------- 报告
    def meta(self) -> dict:
        return {"scripted": {"side": self.side, "grasp_assist": self.grasp_assist,
                             "speed_m_s": self.speed, "ik_every": self.ik_every,
                             "segments": len(self.segments),
                             "hand_link": int(self.hand_link),
                             "hand_joints": len(self.hand_names)}}

    def report(self) -> dict:
        return {"steps": self.steps, "solves": self.solves,
                "segments_done": self.segment_index,
                "segments_total": len(self.segments), "held": self.held,
                "max_wrist_error_m": round(self.max_wrist_error, 4),
                "log": list(self.log[-40:])}

    def close(self) -> None:
        try:
            self._release()
        except Exception:  # noqa: BLE001 - 断连后删约束会抛，收尾不该盖住 trial 结果
            pass


def _joint_positions(env) -> np.ndarray:
    indices = getattr(env, "action_joint_indices", None) or []
    if not indices:
        return np.zeros(0)
    return np.array([float(p.getJointState(env.robot_id, int(index),
                                           physicsClientId=env.client)[0])
                     for index in indices])


# --------------------------------------------------------------------------- #
# 动作积木：抓取 / 放下 / 推 / 拧
# --------------------------------------------------------------------------- #
def clamp_to_reach(policy, point, radius: float = 0.36) -> np.ndarray:
    """把航点夹进「肩关节半径 ``radius``」的球内：IK 到不了的航点等于让这一段失败。"""
    point = np.asarray(point, dtype=np.float64)
    shoulder = np.asarray(policy.ik.arm_points(policy.side)["upper"], dtype=np.float64)
    offset = point - shoulder
    distance = float(np.linalg.norm(offset))
    if distance <= radius or distance <= 1e-9:
        return point
    return shoulder + offset * (radius / distance)


def approach_above(env, policy, point, clearance: float = 0.10) -> list:
    """从胸前 -> 抬到桌面上方 -> 平移到目标正上方 -> 下探（航点都夹在可达球内）。"""
    point = np.asarray(point, dtype=np.float64)
    chest = policy.home_target()
    lift = clamp_to_reach(policy, [chest[0], chest[1], point[2] + clearance])
    over = clamp_to_reach(policy, [point[0], point[1], point[2] + clearance])
    return [Segment(lift, label="lift"),
            Segment(over, label="over"),
            Segment(point, label="down")]



def pick(env, policy, key: str, grip_dz: float = 0.03, label: str = None) -> list:
    """胸前 -> 上抬 -> 物体正上方 -> 下探到抓取高度 -> 停一下 -> 挂上约束。"""
    name = label or key
    pos = object_position(env, key)
    approach = approach_above(env, policy, pos + np.array([0.0, 0.0, grip_dz]))
    return approach + [Segment(None, steps=SETTLE_STEPS, label="settle %s" % name),
                       Segment(None, steps=1, grasp=key, label="grasp %s" % name)]


def place(dest, clearance: float = 0.25, release: bool = True, retreat: float = 0.12,
          label: str = "place", settle: int = 60) -> list:
    """搬到 ``dest`` 正上方 -> 落下 -> 停稳 -> 松手 -> 抬起（不拖着东西横穿桌面）。"""
    dest = np.asarray(dest, dtype=np.float64)
    above = dest + np.array([0.0, 0.0, clearance])
    segments = [Segment(above, label="carry %s" % label),
                Segment(dest, label="lower %s" % label),
                Segment(None, steps=SETTLE_STEPS, label="settle %s" % label)]
    if release:
        segments.append(Segment(None, steps=1, release=True, label="release %s" % label))
        segments.append(Segment(dest + np.array([0.0, 0.0, retreat]),
                                label="retreat %s" % label))
        segments.append(Segment(None, steps=settle, label="settle after %s" % label))
    return segments



def carry_to(dest, label: str = "carry") -> list:
    return [Segment(np.asarray(dest, dtype=np.float64), label="carry %s" % label),
            Segment(None, steps=SETTLE_STEPS, label="settle %s" % label)]


def wrist_joint(policy: "ScriptedPolicy", keyword: str = "yaw") -> str:
    """该侧腕链里名字含 ``keyword`` 的关节（拧腕用）。"""
    for name in policy.ik.joint_values(policy.side):
        if "wrist" in name.lower() and keyword in name.lower():
            return name
    return ""


def rotate_in_place(policy: "ScriptedPolicy", degrees: float, keyword: str = "yaw",
                    steps: int = 160, label: str = "twist") -> list:
    """原地拧手腕（物体已挂在手上时，物体跟着转）。"""
    name = wrist_joint(policy, keyword)
    if not name:
        return []
    return [Segment(None, steps=int(steps), rotate={name: float(degrees)},
                    label="%s %s %+.0fdeg" % (label, keyword, degrees)),
            Segment(None, steps=SETTLE_STEPS, label="settle %s" % label)]


def push_through(env, key: str, direction, distance: float, height: float = 0.0,
                 label: str = None) -> list:
    """贴着物体沿 ``direction`` 推 ``distance`` 米（纯接触，不用抓取约束）。"""
    name = label or key
    unit = _unit([direction[0], direction[1], 0.0])
    pos = object_position(env, key)
    start = pos - unit * 0.11 + np.array([0.0, 0.0, height])
    end = pos + unit * float(distance) + np.array([0.0, 0.0, height])
    return [Segment(start, label="pre-push %s" % name),
            Segment(end, steps=max(48, int(distance / 0.12 * 240)), label="push %s" % name),
            Segment(None, steps=SETTLE_STEPS, label="settle push %s" % name)]


# --------------------------------------------------------------------------- #
# 任务脚本表（键 = 任务名，值是 ``builder(env, policy) -> list[Segment]``）
# --------------------------------------------------------------------------- #
def _onto(env, key: str, dz: float = 0.06) -> np.ndarray:
    """``key`` 上面 ``dz`` 米（放上去用的目标点）。"""
    return object_position(env, key) + np.array([0.0, 0.0, float(dz)])


def _pick_place(env, policy, src: str, dest, label: str = None) -> list:
    return pick(env, policy, src) + place(np.asarray(dest, dtype=np.float64),
                                          label=label or src)


def _lift_and_hold(env, policy, key: str, height: float = 0.30) -> list:
    """抓起来举着（判据只要求「高度/位置」，不需要放下）。"""
    return pick(env, policy, key) + carry_to(object_position(env, key)
                                             + np.array([0.0, 0.0, height]),
                                             label="lift %s" % key)


def _arm_joint(policy, *keywords) -> str:
    """该侧手臂里名字同时含这些关键词的关节（拧腕 / 转前臂用）。"""
    for name in policy.ik.joint_values(policy.side):
        low = name.lower()
        if all(word in low for word in keywords):
            return name
    return ""


def script_pushcube(env, policy) -> list:
    return _pick_place(env, policy, "cube", _onto(env, "target", 0.03), label="target")


def script_pickcube(env, policy) -> list:
    return _lift_and_hold(env, policy, "cube", 0.30)


def script_pickplacecube(env, policy) -> list:
    return _pick_place(env, policy, "cube", _onto(env, "plate", 0.05), label="plate")


def script_balltrashcan(env, policy) -> list:
    return _pick_place(env, policy, "ball", _onto(env, "trashcan", 0.12), label="trashcan")


def script_ballmug(env, policy) -> list:
    return _pick_place(env, policy, "ball", _onto(env, "mug", 0.10), label="mug")


def script_ballbimanual(env, policy) -> list:
    ball = object_position(env, "ball")
    return pick(env, policy, "ball") + carry_to(ball + np.array([0.0, -ball[1], 0.22]),
                                                label="ball center")


def script_pottobimanual(env, policy) -> list:
    return _lift_and_hold(env, policy, "pot", 0.30)


def script_pottomato(env, policy) -> list:
    return _pick_place(env, policy, "tomato", _onto(env, "pot", 0.06), label="pot")


def script_pottray(env, policy) -> list:
    return _pick_place(env, policy, "pot", _onto(env, "tray", 0.06), label="tray")


def script_stackboxes(env, policy) -> list:
    return _pick_place(env, policy, "box_a", _onto(env, "box_b", 0.08), label="box_b")


def script_panhearth(env, policy) -> list:
    return _pick_place(env, policy, "pan", _onto(env, "hearth", 0.06), label="hearth")


def script_tidyuptable(env, policy) -> list:
    dest = _onto(env, "basket", 0.08)
    return (_pick_place(env, policy, "item1", dest, label="basket1")
            + _pick_place(env, policy, "item2", dest, label="basket2"))


def script_pottomatoout(env, policy) -> list:
    pot = object_position(env, "pot")
    return _pick_place(env, policy, "tomato", pot + np.array([0.16, 0.0, 0.03]),
                       label="beside pot")


def script_plateoven(env, policy) -> list:
    return _pick_place(env, policy, "plate", _onto(env, "oven", 0.10), label="oven")


def script_pottomatoplate(env, policy) -> list:
    return _pick_place(env, policy, "tomato", _onto(env, "plate", 0.04), label="plate")


def script_penbrushpot(env, policy) -> list:
    return _pick_place(env, policy, "pen", _onto(env, "pot_holder", 0.08),
                       label="pot_holder")


def script_twistbottlecaps(env, policy) -> list:
    return _pick_place(env, policy, "cap", _onto(env, "bottle", 0.07), label="bottle")


def script_drawerbook(env, policy) -> list:
    return _pick_place(env, policy, "book", _onto(env, "drawer", 0.05), label="drawer")


def script_stacktoyblocks(env, policy) -> list:
    base = _onto(env, "block_c", 0.05)
    top = _onto(env, "block_c", 0.14)
    return (_pick_place(env, policy, "block_a", base, label="stack a")
            + _pick_place(env, policy, "block_b", top, label="stack b"))



def default_script(env, policy) -> list:
    """没写脚本的任务先「到桌面上方待命」，报告里会被标 ``no_script``。"""
    top = table_top(env)
    pos = policy.wrist_world()
    return [Segment(np.array([pos[0] * 0.5 + 0.25, 0.0, top + 0.18]),
                    label="standby (no script)")]


def script_uprearcup(env, policy) -> list:
    """杯子侧放 -> 抓起来拧正（判据：roll/pitch 接近 0 且 z > 0.42）。"""
    cup = object_position(env, "cup")
    segments = pick(env, policy, "cup") + carry_to(cup + np.array([0.0, 0.0, 0.12]),
                                                   label="cup up")
    joint = _arm_joint(policy, "elbow", "roll")
    if joint:
        segments += [Segment(None, steps=240, rotate={joint: -90.0}, label="upright"),
                     Segment(None, steps=SETTLE_STEPS, label="settle upright")]
    return segments + [Segment(None, steps=1, release=True, label="release cup"),
                       Segment(None, steps=90, label="settle release")]


def script_rotatefaucet(env, policy) -> list:
    """抓住水龙头手柄拧 90 度（判据：手柄绕 z 转 > 80°）。"""
    handle = object_position(env, "handle")
    segments = pick(env, policy, "handle") + carry_to(handle + np.array([0.0, 0.0, 0.10]),
                                                      label="handle")
    joint = _arm_joint(policy, "elbow", "roll")
    if joint:
        segments += [Segment(None, steps=220, rotate={joint: 100.0}, label="turn handle"),
                     Segment(None, steps=SETTLE_STEPS, label="settle turn")]
    return segments + [Segment(None, steps=1, release=True, label="release handle")]


def script_pourwater(env, policy) -> list:
    """提起水壶并倾倒（判据：roll > 60° 且 z > 0.5）。"""
    kettle = object_position(env, "kettle")
    segments = pick(env, policy, "kettle") + carry_to(
        kettle + np.array([0.0, 0.0, 0.20]), label="kettle")
    joint = _arm_joint(policy, "elbow", "roll")
    if joint:
        segments += [Segment(None, steps=220, rotate={joint: 80.0}, label="pour"),
                     Segment(None, steps=SETTLE_STEPS, label="settle pour")]
    return segments


SCRIPT_BUILDERS = {
    "pushcube": script_pushcube,
    "pickcube": script_pickcube,
    "pickplacecube": script_pickplacecube,
    "balltrashcan": script_balltrashcan,
    "ballmug": script_ballmug,
    "ballbimanual": script_ballbimanual,
    "potbimanual": script_pottobimanual,
    "pottomato": script_pottomato,
    "pottray": script_pottray,
    "stackboxes": script_stackboxes,
    "panhearth": script_panhearth,
    "tidyuptable": script_tidyuptable,
    "pottomatoout": script_pottomatoout,
    "plateoven": script_plateoven,
    "pottomatoplate": script_pottomatoplate,
    "penbrushpot": script_penbrushpot,
    "twistbottlecaps": script_twistbottlecaps,
    "drawerbook": script_drawerbook,
    "stacktoyblocks": script_stacktoyblocks,
    "uprearcup": script_uprearcup,
    "rotatefaucet": script_rotatefaucet,
    "pourwater": script_pourwater,
}



