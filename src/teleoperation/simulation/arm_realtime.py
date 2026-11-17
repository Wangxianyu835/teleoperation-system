"""Camera arm landmarks (metres) -> per-robot arm targets: relative retargeting.

为什么是「相对」而不是标定
--------------------------
论文链路要 SMPLer-X 的手腕 6D 位姿 + 手眼标定 + PINK 全身 IK；本机没有相机内参/外参，
所以这里做的是**相对重定向**：以一次「中性姿势」为原点，把操作者手腕相对肩的位移按
比例搬进机器人的基座系。

    d_op   = (W - S) - (W0 - S0)            # 你移动了多少（操作者系，米）
    d_base = gain * M(d_op)                 # M = 轴向置换 + 符号（约定，可覆盖）
    target = profile.metric_offset + d_base # 机器人基座系，夹在 profile 的行程内

好处：不需要手眼标定 —— 人整体前后左右挪动时肩和腕一起动，d_op 不变，机器人不会
被你的站位带跑。坏处同样如实说：**这不是论文数值**，gain / 轴向 / 行程都是本机调参，
``docs/multi-robot-arm.md`` 第 7 节记了它们是怎么量出来的。

三种口径（``MAP_MODES``，见 docs/multi-robot-arm.md 8.10 / 8.11）
--------------------------------------------------------------
``chain`` 与 ``segments`` 都还在「**位移**」这一层做文章，所以幅度天生挂在「中性姿势下
肩到腕有多长」上：你双手握拳贴胸（``|W0 - S0|`` 只有约 0.27 m）再摆大臂，位移虽然记下来
了，可它能给出的目标位移也被这根短基线吃掉 —— 现场那三台「幅度很微小 / 映射完全不对」
就是这个。``posture`` 换一层：不搬位移，搬**方向**，长度用机器人自己量出来的连杆：

    u = unit(你的上臂方向)        v = unit(你的前臂方向)
    elbow_target = 肩锚点 + L_u(robot) * M(u)
    wrist_target = elbow_target + L_f(robot) * M(v)

于是「你的上臂抬起 40 度」在 g1（两段 0.193+0.184 m）和 h1_2（0.331+0.229 m）上都是
40 度，肘弯的角度也一比一 —— 幅度不再被你的站位、姿势或机器人臂长放大缩小；两个目标
又都由机器人**自己的**连杆长从它**自己的**肩点推出来，所以永远同时可达，IK2 不再互相
拉扯（这正是 ``segments`` + ``elbow_task`` 抽抽的原因）。

数据从哪来（同一帧，来自 ``inputs/mediapipe.py`` 的 ``next_teleop_observation``）
-------------------------------------------------------------------------------
``upper_body.world_arms[side]`` = (肩, 肘, 腕) 3x3、米制、原点=髋中点 -> **位置**用它；
``world_hands[side]`` = 手 21 点（度量 world）-> **姿态的第三轴**用它（掌面法向，只取方向：
手部 world 的原点是「手的几何中心」，不能当位置用）；``hands[side]`` 是图像归一化口径，
形状不对或各向异性，这里**不用它**算姿态。

退化输入（该侧没检到 / 非有限 / 掌面基不可张成）一律不猜：保持上一帧目标，并在
``ArmTarget.reason`` 里写明原因。

姿态目标（论文的腕部完整位姿，去掉需要标定的那一半）
--------------------------------------------------
论文给 IK 的是「手腕 4x4 位姿」（SMPL-X 旋转链算出来），不是只给一个点；我们用**相对
中性**的写法把它补上：

    R_now  = palm_orientation_frame(手 21 点, 肩肘腕)  # 前臂方向 + **掌面法向**
    R_rel  = R_neutral^T @ R_now                      # 相对你站好的那一帧
    R_target(robot) = R_robot_neutral_ee @ R_rel      # 每台各自乘自己的（在 arm_ik 里做）

两个好处：轴向约定在相减里自然约掉（不需要把相机外参标出来），而且同一份目标被三台
机器人各自乘上自己的中性 EE 朝向，天生是「每台一套」。

第三轴取**掌面法向**而不是手臂平面法向（2026-11-16 现场根因）：三点几何里「绕前臂轴的
自转」是自由度、量不到 —— 手怎么拧，机器人手腕一动不动。掌面法向量得到那一维。没有
21 点手 / 掌面退化时退回旧的三点基（``wrist_orientation_frame``，行为与改动前逐字一致），
并把用的是哪条基写进 ``info["orientation_source"]``。仍然缺的：三点近共线（直臂）时三点
基给不出姿态，而掌面基不受这一点限制（它不需要手臂平面）。

轴向约定（默认值，不是标定）
--------------------------
操作者系（MediaPipe pose world 口径）：x 图像向右、y 向下、z 远离相机为正。
机器人基座系（URDF 惯例）：X 前、Y 左、Z 上。

    robot X <- -op_z     手远离相机 = 远离你 = 机器人往前伸
    robot Y <- +op_x     图像向右 = 你的左侧 = 机器人 +Y
    robot Z <- -op_y     图像向下 = 机器人向下

相机在你正前方、机器人也朝着你时这组符号自洽；相机挪到侧面或机器人背对你时必须改
（``--arm-axis-order`` / ``--arm-axis-signs``）。真值要在你坐到位后实测确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import acos, degrees
from pathlib import Path

import numpy as np

from teleoperation.simulation.arm_profiles import SIDES, ArmProfile

AXIS_NAMES = ("X", "Y", "Z")
#: 默认轴向映射，见模块 docstring（约定，不是标定）。
DEFAULT_AXIS_ORDER = ("Z", "X", "Y")
DEFAULT_AXIS_SIGNS = (-1.0, 1.0, -1.0)
#: 手臂长度归一化的参考值（米）：高于/矮于这个臂长的操作者由 gain 自动补偿。
REFERENCE_REACH_M = 0.60
#: 归一化系数的允许区间，防止一次坏标定把 gain 推到离谱的值。
ARM_LENGTH_BAND = (0.5, 2.0)
#: 比这更短的「肩到腕」认为 Pose 这帧不可信（正常人手臂 >= 0.3 m）。
MIN_OPERATOR_REACH_M = 0.15
#: 基座系里的默认行程上限（米，每轴），profile 没有 realtime_travel_m 时用它。
DEFAULT_MAX_TRAVEL_M = 0.10
#: 没有 pose 手臂时回落到「手 21 点 -> 手腕目标」的映射比例（沿用 wrist_targets 默认）。
FALLBACK_IMAGE_SCALE = 0.7
#: 肘平面可信判据：|上臂 x 前臂|（= 上臂与前臂夹角的正弦）小于它时这一帧不给姿态。
MIN_ELBOW_PLANE_SINE = 0.05
#: 掌面法向取点：腕 0 / 食指 MCP 5 / 小指 MCP 17。与 ``--wrist-trace`` 的 ``palm_normal``
#: 是**同一口径**（现场取证看到的那一列，就是这里喂给 IK 的那一维）。
PALM_NORMAL_SLOTS = (0, 5, 17)

#: 左侧的掌面法向要**取反**：MediaPipe 21 点里食指 MCP 在拇指侧、小指 MCP 在另一侧，
#: 左右手的关键点绕序是**镜像**的，于是同一个 ``cross(index - wrist, pinky - wrist)`` 在右手
#: 给出「掌心朝外」、在左手给出「掌心朝内」。``y = z x x`` 是推算出来的，所以那一帧形式上
#: 仍是正交旋转（det = +1），差别藏在「z 指反」里 —— 等价于**整帧绕前臂轴多转 180 度**。
#:
#: 后果（2026-11-16 深夜现场 + 离线取证 ``outputs/tmp_probe/left_right_palm_sign.py``）：
#: 绕前臂的自转在 ``R_neutral^T @ R_now`` 里**不受影响**（同一个叉乘在两侧都被用，共轭约掉，
#: 实测两侧给的都是 +40.0 度），但腕的 pitch/yaw 两个方向在左侧整体反号 —— 于是左手的目标
#: 姿态系统性偏了 180 度：关节被一路推到行程两端（同一跑报告 ``arm.wrist_usage``：h1_2 左
#: ``wrist_pitch`` usage 1.000 顶满、``wrist_yaw`` 0.000 顶死另一端，右 0.283/0.248 全在中段；
#: g1 左 ``wrist_roll`` 1.000 顶满、右 0.184），现场看到的就是「右手跟得上、左手反着拧，而且
#: 左边特别抖」（``bridge.counts.direction_glitch`` 左 77 / 右 39）。
#:
#: 取反之后左侧掌面基与右侧同手性，同一个世界旋转在两侧语义一致。**不改**旋前那一维的含义
#: （它本来就对），改的是 pitch/yaw 那两维。
MIRRORED_PALM_SIDES = ("left",)


#: 掌面法向投影到「前臂法平面」之后的长度下限（= sin(法向, 前臂)）。手掌自然朝前时它
#: 接近 1；接近 0 说明掌面法向与前臂几乎平行（手腕折成 90 度、或 21 点乱跳），此时滚转
#: 定不出来 —— 退回三点基，不猜。
MIN_PALM_SINE = 0.05
#: 掌面基这一帧给不出时，还能沿用**上一次**掌面旋转多久（秒；``0`` = 关，A/B 用）。
#:
#: 为什么要保持（2026-11-16 第一次带 ``--wrist-trace`` 的现场，CSV 见
#: ``outputs/tmp_probe/wrist_live_hold.csv``）：掌面基依赖**米制 21 点手**，而手部模型在
#: 那场里只有约 20% 的帧给出完整手点，``ori_source`` 于是逐帧在 ``palm_normal`` /
#: ``three_point`` 之间来回切。两套基量的不是同一件事：``three_point`` 对旋前完全不敏感，
#: 所以「丢掉一帧掌面基」等于把腕目标从「拧了 90 度」瞬间改成「没拧」—— 机器人照着做就是
#: 猛甩一下；掌面基回来之后，同一份姿态在腕链上的两个对称解又落在另一支上（实测 g1 左腕
#: ``left_wrist_pitch_joint`` 在 -1.512 / +1.619 rad 两颗限位之间跳，而限位正是
#: ±1.614429558 rad），现场看到的就是「抽搐」。丢一两帧时沿用上一帧的掌面旋转，比换一套
#: 量不到旋前的基更接近事实：腕的目标本来是连续的。超过窗口才如实退回 ``three_point``
#: （无限保持会让腕锁在旧姿态上不跟手）。
PALM_HOLD_S = 0.5
#: 中性掌面基的**可信度闸门**：样本帧数至少占标定帧的 ``PALM_MIN_SAMPLE_FRACTION``，
#: 且这几帧彼此相对均值的最大夹角不得超过 ``PALM_MAX_SPREAD_DEG``。不合格就当没有
#: （``orientation_palm`` 给 None，退回三点基），原因写进 ``neutral[side]["palm_note"]``。
#:
#: 为什么逐帧的 ``MIN_PALM_SINE`` 不够（2026-11-17 现场 15:40 那场，报告
#: ``outputs/tmp_realtime/arm_run_20261117_154001.json``）：标定那 30 帧里左手
#: ``palm_samples=0``（手还没进画面）、右手 **12**（手刚进画面那几帧）。左手机缘巧合地
#: 退回三点基，右手却把这 12 帧的均值当成基准，于是同一场里：
#: ``arm.residual_mm`` 左 h1_2/gr1_t2/g1 = 3.3/2.5/9.4 mm，右 **120.2/45.3/256.2 mm**；
#: ``arm.orientation_error_deg`` 左 0.6/1.6/0.7 度，右 **72.3/24.9/122.5 度**；
#: ``arm.wrist_usage`` 里 h1_2 右腕 ``elbow_roll/wrist_pitch/wrist_yaw`` 三个都
#: ``at_limit: true``（左三个 0.385-0.634，无一个顶限位）。现场看到的就是
#: **「右手卡住不动、手腕怎么拧都不对」**。
#: 三个腕关节同时顶限位不是行程问题（§8.19 的桩回放里 h1_2 左右都只用到 17 度、
#: 没有 ``at_limit``），而是基准本身偏了一个大角度 —— 目标越拧越顶，最后钉死在限位上。
#:
#: 判据为什么是「帧数比例 + 散度」两条：帧数挡住「手只露了一下」，散度挡住
#: 「露的那几下彼此不一致（半遮挡 / 掌面翻号）」—— 一组自相矛盾的样本，均值再准也不是
#: 任何一帧的姿势。两条都不过时宁可退回三点基（旋前不跟，但**不会把腕推进限位**），
#: 也不拿一份可疑的基准去驱动六个腕关节。
PALM_MIN_SAMPLE_FRACTION = 0.5
PALM_MAX_SPREAD_DEG = 25.0
#: 中性里**没有**可用掌面基时，运行中补采一次「迟到基准」：要**连续** ``PALM_LATE_FRAMES``
#: 帧、且这几帧彼此夹角都不超过 ``PALM_LATE_SPREAD_DEG`` 才认。
#:
#: 为什么必须有（2026-11-17 16:00 那场的 ``--wrist-trace`` 统计，
#: ``outputs/tmp_realtime/live_run_1600_wrist.csv``）：2500+ 行里 ``ori_source`` 是
#: ``palm_normal`` 的只有 **3 行**（右 0 行），其余全是 ``three_point``。三点基的第三轴
#: 取「上臂 x 前臂」的手臂平面法向，**对绕前臂的自转完全不敏感** —— 于是整场「手怎么拧、
#: 机器人手腕一动不动」，这不是桥没接上，是那一维在数学上就没被喂进去
#: （见 ``palm_orientation_frame``）。而这份基准一旦被闸门判死（或手点晚到、窗口已关），
#: 之前的行为就是**整场失去旋前**，没有任何补救路径。
#:
#: 迟到基准补的就是这条路：中性没有掌面基时，等手上真的出现一段**稳定**的掌面基，
#: 就地把它当基准。代价是「零点」取在那一刻而不是标定那一刻 —— 所以要求连续 8 帧一致
#: （人站着不动时手点很稳，8 帧 ≈ 0.8-1.3 s），并在报告里写 ``palm_reference_source="late"``
#: 与 ``palm_reference_frame``，不假装它就是中性。
#:
#: 只在**标定完成之后**才采（``calibrating`` 为 False）：标定期间 ``_refresh_neutral``
#: 每帧都会重写 ``orientation_palm``，那时候采会被下一帧盖掉。
PALM_LATE_FRAMES = 8
PALM_LATE_SPREAD_DEG = 12.0

#: 「你的两条手臂是不是镜像的」：两条上臂方向相对**你肩膀连线**的镜像偏差超过这么多度，
#: 就记一帧 ``same_side``（= 两侧朝同一边摆，不是镜像手势）。
#:
#: 为什么需要这个数字（2026-11-17 现场："屈臂、双手放胸前、把肘向外打开，左臂跟着右臂的
#: 方向动"）：桥的这一层**不可能**把镜像手势算成同向 —— 轴映射 ``AxisMap`` 是两侧共用的
#: 一个线性映射（置换+符号），而按侧的三个量（肩锚点 / 上臂长 / 前臂长）实测左右镜像到
#: 0.07 mm（``outputs/tmp_probe/probe_arm_anchor_mirror.py``：h1_2 0.07 / g1 0.01 /
#: gr1_t2 5.4 mm），线性映射 + 镜像输入 -> 输出必然镜像。所以现场那句「左跟着右」只可能
#: 来自**输入本身不镜像**（一侧手点被挡/跑偏，或手势本来就不对称）或者 IK 那一层。
#:
#: 这个量在**操作者自己的坐标系**里算，与 ``AxisMap`` 无关：
#:
#:     lateral = unit(左肩 - 右肩)          # 操作者自己的左右轴，每帧现算
#:     dev     = angle(u_left, mirror(u_right, lateral))
#:
#: ``dev`` 小 = 输入是镜像的（桥会如实给出镜像的两条手臂）；``dev`` 大 = 输入不是镜像手势，
#: 桥只能照抄 —— 报告里如实写出来，别让人再去猜「是不是桥算错了」。
SYMMETRY_SAME_SIDE_DEG = 45.0
#: 对称性采样最多记多少帧（够算中位/最大，又不至于跑一晚吃掉内存）。
SYMMETRY_MAX_SAMPLES = 20000
#: ``track`` 模式下，这一侧**还没见过掌面基**时先不把帧计入中性标定，最多等这么多帧。
#:
#: 为什么必须有这一条（2026-11-17 15:52 那次重启的现场日志
#: ``outputs/tmp_realtime/live_run_1552.log``）：手臂 pose 关键点从**第 1 帧**就有，
#: 手部 21 点要 5-7 s 才锁上 —— 于是「30 帧中性窗口」在任何一帧掌面基出现之前就烧完了：
#:     frame=20  hand valid=left=0 right=0 invalid=8 | cal=track left=20/30 right=0/30
#:     frame=100 hand valid=left=0 ... | cal=track left=26/30 ...
#: 手册那句「站好别动 30 帧」在这里是**做不到**的：人还没走到手部模型能认出来的距离，
#: 窗口就关了，这一整场就注定只有三点基（拧腕不跟），而后加的闸门只会如实把那份
#: 12/30 帧的坏基判死 —— 丢旋前的原因没解决，只是不再把腕推进限位。
#:
#: 语义边界：只等**计入标定**这件事，**不等跟手** —— ``track`` 模式下样本照旧每帧追加、
#: 临时中性照旧发布，手臂从看见你手臂的第一帧就开始跟。到 ``PALM_WAIT_FRAMES`` 还没有
#: 手点就照旧计数（等于回到改动前的行为），所以不会出现「永远标定不完」。
#: ``hold`` 模式完全不受影响（历史行为逐字不变）。
PALM_WAIT_FRAMES = 90
#: 腕姿态目标（下发给 IK 的那一份相对旋转）的最大角速度（度/秒）：``0`` = 关（A/B 基线）。
#:
#: 为什么 ``PALM_HOLD_S`` 之后还需要这一层（2026-11-16 第二次现场，CSV 见
#: ``outputs/tmp_probe/wrist_live_hold2.csv``，分析脚本 ``outputs/tmp_probe/analyze_hold2*.py``）：
#: 加了保持窗口后「换基那一下」确实不跳了（进 ``palm_hold`` 的帧 |d腕| 只有 0.000-0.010 rad），
#: 但同一场里仍有 **1-2% 的掌面帧本身是坏的**：``palm_turn`` 的逐帧角速度 p50 13-50 度/秒、
#: p90 114-210、p99 740-833、最大 1252（人手腕极限约 500-700），也就是偶发一帧就要求转
#: 100-180 度。这些帧直接把腕目标拨过去，实测 g1 左腕 4 秒内 0.75 -> 3.14 -> -0.49 rad
#: （3.14 = 腕行程尽头），现场看到的就是「抽搐」。另一种同源跳变是**换基**：掌面基丢超过
#: ``PALM_HOLD_S`` 后如实退回 ``three_point``，两套基的目标差 2.6-3.0 rad，那一下也是瞬时的。
#:
#: 这里是**限速**而不是「保持」或低通：按这一帧的真实时长放行 ``rate * dt`` 那么多度，
#: 轴不变。真手速之内照跟（p50 只有 13-50 度/秒，限速根本不介入），超了就把尖峰削成一段
#: 连续运动；与 ``_guard_direction`` 的「保持」不同，它**不会卡住**持续的快动作，也不会在
#: 尖峰过去之后来回振（削掉的角度不累积成追赶）。与 ``--arm-slew-mm`` 是同一哲学，只是
#: 作用在旋转上。数字依据：人前臂旋前/旋后的峰值约 500-700 度/秒，取 540 落在里面。
MAX_ORIENTATION_RATE_DEG_S = 540.0
#: 腕姿态目标**每帧**最多走多少度（与帧长无关）；``0`` = 只按速率（旧行为）。
#:
#: 为什么在速率上限之外还要这一条（2026-11-16 晚，现场 16 分钟实跑
#: ``outputs/tmp_realtime/arm_run_20261116_213650.json``：5940 帧 / 967.7 秒 =
#: **5.92 fps**，``median_frame_interval_ms`` 155.4）：``rate * dt`` 在 155 ms 一帧时
#: 等于 **84 度/帧**，于是一帧 150 度的坏掌面帧仍被放行 84 度 —— 速率上限在这个帧率下
#: 等于没拦，现场看到的还是「抽一下」（同一份报告里 ``arm.residual_mm`` 49-329 mm、
#: 左臂 2251/5940 = 38% 的帧没有手点，也都是同一个低帧率现场的产物）。
#:
#: 分工：速率管「这个速度人体可能吗」，这一条管「这一帧别走太远」，与帧率脱钩 ——
#: 6 fps 与 30 fps 都最多 30 度/帧。两条同时生效时取较小的那个。
#:
#: 代价如实说：帧长超过 ``30/540 = 55.6 ms``（低于 18 fps）时这一条比速率更紧，会削掉
#: **真实**的快速转腕（0.5 秒一帧里人真能转 150 度）。18 fps 以上一个数都不改（30 fps
#: 下速率只放 18 度/帧）。要在低帧率下把「真快」与「坏帧」分开，只能等「缓一帧回看」
#: （给每帧加一帧延迟），本轮没做。
MAX_ORIENTATION_STEP_DEG = 30.0
#: 操作者 3D 手臂点的自适应低通（One-Euro 形式）：慢速/静止时多滤，快速运动时
#: 截止频率随速度上升，避免固定强滤波造成明显拖影。
ARM_INPUT_FILTER_MIN_CUTOFF_HZ = 1.0
ARM_INPUT_FILTER_BETA = 8.0
#: 手掌目标旋转的自适应低通。这里的速度单位是度/秒，所以 beta 比米制位置小。
HAND_TARGET_FILTER_MIN_CUTOFF_HZ = 0.8
HAND_TARGET_FILTER_BETA = 0.01
#: 目标超出行程上限后的做法：``clip`` = 逐轴剪断（**默认**）；``scale`` = 整条位移等比缩到上限。
#: 2026-10-13 单轴隔离 + 正弦跟随实测（E:\cache\tmp\probe_axis_response.py / probe_sine.py，
#: 数字写进 docs/multi-robot-arm.md §8.8）：一次真实现场复合位移 (-9.3, +6.4, +24.5) cm
#: 里只有 op_z 超限，``scale`` 会把横向一起压到 0.16-0.3 倍 —— g1 左腕横向只剩 1.31 cm、
#: 肩 roll 只转 2.0 度、肘 2.8 度（现场"大臂不左右摆、小臂不动"）；逐轴剪断后是 3.44 cm、
#: roll 5.5 度、肘 7.1 度，横摆传递比 0.43 -> 0.49，而残差反而更小（15.1 -> 13.5 mm）。
#: ``scale`` 当初是为了躲"盒角 IK 到不了"，实测那个担心只在 gain 被坏标定放大到 2.19 时成立。
TRAVEL_MODES = ("clip", "scale")
#: 中性标定没攒够 ``calibration_frames`` 帧时怎么办（``--arm-calibrate-mode``）：
#:
#: ``hold``  历史行为：攒够帧之前目标一直钉在中性工作点上（``valid=False, held=True``）。
#: ``track`` 只要这一侧有 1 帧有效手臂，就先把**已有样本的中位数**当临时中性开始跟手；
#:           攒够帧时中性收敛到最终中位数（与 ``hold`` 的终值逐位相同），HUD 显示
#:           ``cal=track n/N``。
#:
#: 为什么要有 ``track``（2026-10-14 那场 live_round2，85% 的帧没有任何有效手臂数据）：
#: ``hold`` 下进度卡在 ``cal=left=15/30 right=16/30`` 整整 4 分钟，三台机器人全部停在中性
#: 工作点上 —— 现场看到的就是「手臂卡在胸前举着」。这不是臂映射的错，是「没有数据就永远不
#: 开始」。``track`` 让「只要看见一次就动」，同时把「为什么这一帧没跟手」逐帧记到
#: ``counts["held_reason"]``（见 ``HOLD_REASONS``），下一次排查不用再靠猜。
CALIBRATE_MODES = ("hold", "track")
#: 没跟手的那一帧属于哪一类（``summary()["counts"]["held_reason"]``）：标定中 / 这一帧没有
#: pose 手臂关键点 / 有手臂但短于 ``MIN_OPERATOR_REACH_M``（关键点抖动） / 其他（例如这一侧
#: 没有 pose 后端）。现场排查"手臂不动"先看这四个数。
HOLD_REASONS = ("calibrating", "no_pose_landmarks", "reach_below_min", "other")
#: 「操作者位移 -> 目标位移」的口径：
#:
#: ``chain``    肩->腕整段一个 gain（历史口径；等效于论文那条「只给腕一个目标」的路径）
#: ``segments`` 论文 ``scale_joints`` 的两段式：上臂段与前臂段**各自**按
#:              「机器人连杆长 / 你的连杆长」缩放后相加：dT = k_u*d(上臂) + k_f*d(前臂)
#:
#: 为什么必须有第二口径（2026-10-13 现场实测的症状）：只喂「腕 - 肩」时，**只动大臂
#: （肘摆开、手留在胸前）这一刻 d_op 几乎不变，机器人一点反应都没有** —— 现场"双手握拳
#: 贴胸、只摆大臂/小臂"三台全零反应就是这个原因。三点数据里肘那一点本来就在手上，
#: 只是 ``chain`` 把它扔掉了。数字见 docs/multi-robot-arm.md §8.10。
#:
#: ``posture`` 方向口径（2026-10-13 晚，用户当场判定 chain/segments 都不像）：
#: 不用位移，用**单位方向** + 机器人自己的连杆长（``set_robot_links``）与自己的肩/肘/腕
#: 锚点（``set_robot_anchors``）。见模块 docstring 与 docs/multi-robot-arm.md §8.11。
#: 它与前两种是**并列**关系：不需要 ``elbow_task``（肘目标由口径自身给出，见
#: ``_posture_target``），也不受 ``travel_limit`` 逐轴夹断的影响（目标天然落在机器人
#: 自己那根「上臂+前臂」可达球面上，逐轴夹断正是把 g1 的幅度压成 40 mm 的那个上限）。
MAP_MODES = ("chain", "segments", "posture")
#: 腕姿态目标的坐标系口径：
#: ``hand_base`` = 直接目标可见手掌的手指/拇指/掌面三轴（CLI 默认）；
#: ``wrist`` = 手相对前臂的局部旋转（历史实验口径）；
#: ``hand`` = 历史口径，整只手掌坐标系相对中性的旋转（包含前臂方向变化）。
ORIENTATION_MODES = ("hand_base", "wrist", "hand")
#: 21 点掌面基 -> 机器人原生手掌几何轴的逐侧符号。
#: 掌面基为了左右同手性，左手的 ``y=z*x`` 实际指向小指侧；原生手几何帧的 y 是拇指侧、
#: z=x*y。左手因此要同时翻 y/z（det 仍为 +1），右手保持原样。
HAND_SEMANTIC_SIGNS = {
    "left": np.array([1.0, -1.0, -1.0]),
    "right": np.array([1.0, 1.0, 1.0]),
}
#: 中性标定量出来的「上臂 / 前臂」段长（米）的合理区间，**超出就当标定坏了**。
#: 为什么需要（2026-11-17 那三场 live 的实测）：``k = 机器人段长 / 你的段长``，而你的段长是
#: 深度压扁的受害者 —— 同一个人同一天三场的 ``k_前臂`` 分别是 1.02 / 1.34 / **3.79**
#: （h1_2 前臂 0.229 m ÷ 你的 0.060 m），同一份记录的右臂却是 0.99（0.23 m）。
#: 一个人左右前臂不可能差 4 倍，所以 k=3.79 只能说明那一场的姿态关键点被压扁了。
#: 这种 k 一进 ``segments`` 口径就会把前臂位移放大 3.8 倍（现场"左手抽"的半个原因），
#: 所以宁可判它不可用、退回整段 gain（``info["map"]`` 会写 ``chain (segments unavailable)``），
#: 并且把原因写进 ``segment_scale_notes`` —— 别让它静默生效。
#: 0.12 m 是成年人前臂的宽松下限（真实 0.20-0.32），0.45 m 是上限（超过就是关键点跳飞了）。
SEGMENT_LENGTH_BAND_M = (0.12, 0.45)
#: 左右两侧**量出来的**同一条段长的允许比值（大/小）。超过就是标定坏了。
#:
#: 为什么带宽护栏不够（2026-11-17 追到机制）：三道护栏里 ``SEGMENT_LENGTH_BAND_M`` 只拦
#: "明显不像人"的段长，而现场最普通的一次坏标定是 **前臂 0.17 m vs 0.23 m**（比值 1.35）
#: —— 两端都在带宽内，护栏一声不吭，``k_前臂`` 却从 0.99 变成 1.35，前臂位移被放大 35%。
#: 那一场的 ``--wrist-trace`` 逐侧对比（outputs/tmp_probe/wrist_trace_compare.py）显示：
#: 左右 ``palm_normal``/``palm_hold`` 比例几乎相同（57/9 vs 58/8，**输入侧一样**），
#: 但左手 ``residual_mm`` 中位 50.7 而右手 26.1，且 ``err`` 的带符号分量左手是右手的
#: 1.6 倍（dz -69 mm vs -44 mm）—— 目标被放大 -> 甩得更远 -> 更难够到 -> residual 翻倍。
#: 同一个人左右上臂/前臂的真实差异 <10%（解剖学量级），所以比值超过 1.25 只能是标定误差。
#:
#: **只告警，不改控制**：单次采集分不清是"左手量短了"还是"右手量长了"，而静默把某一侧
#: 从 ``segments`` 换成 ``chain`` 又是一次"配方被偷偷换掉"（历史判定 chain 两侧都不像）。
#: 真要改的是标定：重标 / 加大 ``--arm-calibrate-frames``，告警就是把手指指向那里。
SEGMENT_ASYMMETRY_MAX = 1.25
#: 人手臂关节的最大角速度（度/秒）：单帧方向变化超过它，就当作关键点跳变而不是你的手。
#: 上限的依据：肩关节峰值约 500 度/秒（日常摆动 100-300），所以 600 是"比人快"的界。
#: 实测这本来历（outputs/tmp_realtime/live_long_frames.npz，口径见
#: outputs/tmp_realtime/check_arm_jitter.py）：每帧方向变化中位 4.7-6.9 度（10 fps ->
#: 470-690 度/秒）、p90 21-35 度、**最大 153 度** —— 最后那一档是 MediaPipe 的跳变，不是手在动。
#: 方向口径是 1:1 搬方向，于是机器人被要求在一帧里把肘转 153 度：IK 要么翻支路、要么抽一下
#: （现场「H1-2 手臂和手掌交界处偶尔抽搐」就是这个）。按这一帧的实际时长换算，不是写死度数。
MAX_DIRECTION_RATE_DEG_S = 600.0
#: 拿不到上一帧间隔时的名义帧长（秒）：与 realtime_teleop.NOMINAL_FRAME_DT 同一口径。
NOMINAL_FRAME_DT_S = 1.0 / 30.0
#: 跳变护栏最多**连续**拦几帧：同一个「太快」的方向连拦这么多帧还没变，就不是手抖而是真的在快速
#: 移动（或者关键点自己的偏移），必须放过去 —— 硬拦会让手臂卡在旧方向上不动。
#: 实测（2026-10-13 200 帧回放）：不设这个上限时 g1 左臂的方向误差反而从 17.6 度涨到 43.3 度，
#: 因为左臂录制里有 30% 的帧被判跳变，其中一部分是持续的真实移动。
GLITCH_HOLD_FRAMES = 3
#: 横向护栏默认值（米，基座系）：左腕 y 不小于 +它、右腕 y 不大于 -它。0 = 关闭。
#: 历史：2026-10-13 白天为了"窗口里永不跨中线"设成 0.02（h1_2 的中性工作点只有
#: y=+-0.05 m，而每轴行程上限是 0.10 / 0.15 m，于是「目标」可以合法地走到身体另一侧，
#: 两前臂就贴到一起 —— 实测 1560 帧，前臂间距最小 2.21 mm）。护栏只动 y，其他轴照跟。
#: 2026-10-13 晚现场反馈"大臂不能左右摆"后**默认改回 0**：护栏挡掉的正是"往身体中线
#: 方向摆"那半程（实测左腕 ±9 cm 摆动的传递比 0.49 -> 0.43，右腕 9.0 cm 的向内请求
#: 只落 2.9 cm），而跨中线本身是操作者把右手横过身体中线的忠实映射。
#: 要恢复旧行为：`--arm-lateral-guard-mm 20`（折中 10）。见 docs/multi-robot-arm.md §8.9。
DEFAULT_LATERAL_GUARD_M = 0.0
#: 臂长归一化的可信下限（米）：标定出来的「肩到腕」短于它，说明那一场 Pose 关节是坏的
#: （实测 live_long 一场：左 0.2742 / 右 0.3976，正常人同一个人两侧差 < 5%），
#: 这时不敢再用 ``arm_length_reference / reach`` 去放大，退回 ``metric_scale``。
MIN_PLAUSIBLE_ARM_REACH_M = 0.30
#: 左右两侧标定臂长的最大相对差：超过它认为这一场标定不可信（左右同一个人不该差这么多）。
ARM_REACH_SIDE_TOLERANCE = 0.25
SOURCE_POSE = "pose"
SOURCE_HAND = "hand"
SOURCE_NONE = "none"


def _is_points(value, shape) -> bool:
    """``True`` 表示这是一组给定形状、全有限、可用的点。"""
    if value is None:
        return False
    array = np.asarray(value, dtype=np.float64)
    return array.shape == tuple(shape) and bool(np.isfinite(array).all())


def _unit_direction(vector):
    """单位化一个方向向量；``None`` 表示退化（缺数据 / 非有限 / 长度接近 0）。

    退化不抛异常：调用方据此退回别的口径，并把原因写进那一帧的 ``info``。
    """
    if vector is None:
        return None
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3:
        return None
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return values / norm


def _slerp_direction(start, end, gain: float):
    """两个单位方向之间按夹角插值；``gain=1`` 就是 ``end``（方向口径的 gain 语义）。

    方向口径里 gain 不再是「位移乘多少倍」，而是「机器人离你的方向有多近」：
    1.0 = 完全对齐你的方向（默认），0.5 = 停在机器人中性方向与你的方向正中，0 = 回中性。
    用夹角插值而不是线性插值再单位化，是因为后者在夹角大时角速度不均匀（看起来会先快后慢）。
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    gain = float(min(max(gain, 0.0), 1.0))
    if gain >= 1.0:
        return end
    if gain <= 0.0:
        return start
    angle = float(np.arccos(float(np.clip(np.dot(start, end), -1.0, 1.0))))
    if angle <= 1e-6:
        return end
    sine = float(np.sin(angle))
    if abs(sine) <= 1e-9:
        # 夹角 180 度：最小旋转不唯一，绕一个与自身垂直的固定轴转，至少结果是确定的。
        axis = np.cross(start, np.array([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(start, np.array([1.0, 0.0, 0.0]))
        axis = axis / float(np.linalg.norm(axis))
        cross = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
        rotation = (np.eye(3) + np.sin(gain * angle) * cross
                    + (1.0 - np.cos(gain * angle)) * (cross @ cross))
        return rotation @ start
    return ((np.sin((1.0 - gain) * angle) / sine) * start
            + (np.sin(gain * angle) / sine) * end)


def _angle_between_deg(left, right) -> float:
    """两个方向的夹角（度）：报告里用来说明「你的上臂转了多少度」。"""
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(degrees(acos(cosine)))


def quaternion_angle_deg(left, right) -> float:
    """两个单位四元数（``[x, y, z, w]``，PyBullet 序）之间的转角（度，0-180）；坏输入给 NaN。

    口径与 ``--wrist-trace`` 的 ``ori_turn_deg`` 逐字相同（``2*acos(|w|)``，``q`` 与 ``-q``
    是同一个姿态，所以取绝对值），这样现场看到的那个数和限速用的那个数是同一个数。
    """
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != 4 or right.size != 4:
        return float("nan")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(left_norm) or not np.isfinite(right_norm) \
            or left_norm <= 0.0 or right_norm <= 0.0:
        return float("nan")
    cosine = abs(float(np.dot(left, right)) / (left_norm * right_norm))
    return float(degrees(2.0 * acos(min(1.0, max(-1.0, cosine)))))


def _quaternion_multiply(left, right):
    """``left * right``（Hamilton 积，``[x, y, z, w]`` 序）：先把 ``right`` 转再转 ``left``。"""
    x1, y1, z1, w1 = (float(value) for value in np.asarray(left, dtype=np.float64).reshape(4))
    x2, y2, z2, w2 = (float(value) for value in np.asarray(right, dtype=np.float64).reshape(4))
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dtype=np.float64)


def _quaternion_scaled(quaternion, fraction: float):
    """把 ``quaternion`` 的转角按 ``fraction``（0..1）缩，**轴不变**；= 与单位四元数做 slerp。

    用途只有一处：限速时「只走 limit 那么多度」（见 ``MAX_ORIENTATION_RATE_DEG_S``）。
    转轴保持不变是必须的 —— 换轴等于换姿态方向，削尖峰会变成另一条曲线。
    """
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    values = values / norm
    if float(values[3]) < 0.0:
        values = -values
    half = float(acos(min(1.0, max(-1.0, float(values[3])))))
    sine = float(np.sin(half))
    if not np.isfinite(sine) or sine <= 1e-9:
        # 转角 0：怎么缩都还是它自己。
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = values[:3] / sine
    scaled_half = half * float(min(max(fraction, 0.0), 1.0))
    return np.array([axis[0] * np.sin(scaled_half), axis[1] * np.sin(scaled_half),
                     axis[2] * np.sin(scaled_half), np.cos(scaled_half)], dtype=np.float64)


def _filter_alpha(cutoff_hz: float, dt: float) -> float:
    """First-order low-pass coefficient for ``old + alpha * (new - old)``."""
    cutoff_hz = max(1e-6, float(cutoff_hz))
    dt = max(1e-6, float(dt))
    return float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt))


class AdaptiveArmPointFilter:
    """Per-side One-Euro filter for the three metric pose points.

    MediaPipe pose world points jitter by several degrees per frame while the
    operator is nearly still. A fixed low-pass removes this but adds obvious lag
    during intentional motion. The cutoff rises with measured point speed, so slow
    frames are smoothed while fast reaches stay responsive.
    """

    def __init__(self, min_cutoff_hz=ARM_INPUT_FILTER_MIN_CUTOFF_HZ,
                 beta=ARM_INPUT_FILTER_BETA):
        self.min_cutoff_hz = max(0.0, float(min_cutoff_hz))
        self.beta = max(0.0, float(beta))
        self._history = {}
        self._last_speed = {}
        self._last_target = {}
        self._filtered = {}

    def reset(self, side: str) -> None:
        self._history.pop(side, None)
        self._last_speed.pop(side, None)
        self._last_target.pop(side, None)
        self._filtered.pop(side, None)

    def update(self, side: str, points, dt: float):
        current = np.asarray(points, dtype=np.float64)
        if current.shape != (3, 3) or not np.isfinite(current).all():
            return None
        previous = self._filtered.get(side)
        history = self._history.setdefault(side, [])
        history.append(current.copy())
        del history[:-3]
        if previous is None or len(history) == 1:
            target = current.copy()
            self._history[side] = [target.copy()]
            self._last_target[side] = target.copy()
            self._filtered[side] = current.copy()
            self._last_speed[side] = 0.0
            return current.copy()
        # A one-frame spike is removed by selecting the 3-frame coordinate median.
        # Persistent motion changes the median by the second frame, so real motion
        # opens the bandwidth instead of being mistaken for noise forever.
        target = np.median(np.stack(history), axis=0)
        last_target = self._last_target.get(side, target)
        speed = float(np.linalg.norm(target - last_target) / max(1e-6, dt))
        smooth_speed = 0.7 * self._last_speed.get(side, speed) + 0.3 * speed
        self._last_speed[side] = smooth_speed
        if self.min_cutoff_hz <= 0.0:
            filtered = target.copy()
        else:
            cutoff = self.min_cutoff_hz + self.beta * smooth_speed
            alpha = _filter_alpha(cutoff, dt)
            filtered = previous + alpha * (target - previous)
        self._last_target[side] = target.copy()
        self._filtered[side] = filtered
        return filtered


class AdaptiveRotationFilter:
    """Per-side quaternion filter whose bandwidth follows angular speed."""

    def __init__(self, min_cutoff_hz=HAND_TARGET_FILTER_MIN_CUTOFF_HZ,
                 beta=HAND_TARGET_FILTER_BETA):
        self.min_cutoff_hz = max(0.0, float(min_cutoff_hz))
        self.beta = max(0.0, float(beta))
        self._history = {}
        self._last_speed = {}
        self._last_target = {}
        self._filtered = {}

    def reset(self, side: str) -> None:
        self._history.pop(side, None)
        self._last_speed.pop(side, None)
        self._last_target.pop(side, None)
        self._filtered.pop(side, None)

    @staticmethod
    def _slerp(first, second, fraction: float):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        if float(np.dot(first, second)) < 0.0:
            second = -second
        cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
        if cosine > 0.9995:
            value = first + fraction * (second - first)
            return value / max(1e-12, float(np.linalg.norm(value)))
        half = float(np.arccos(cosine))
        sine = float(np.sin(half))
        value = (float(np.sin((1.0 - fraction) * half) / sine) * first
                 + float(np.sin(fraction * half) / sine) * second)
        return value / max(1e-12, float(np.linalg.norm(value)))

    def update(self, side: str, quaternion, dt: float):
        current = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(current))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        current = current / norm
        previous = self._filtered.get(side)
        history = self._history.setdefault(side, [])
        history.append(current.copy())
        del history[:-3]
        if previous is None or len(history) == 1:
            self._history[side] = [current.copy()]
            self._last_target[side] = current.copy()
            self._filtered[side] = current.copy()
            self._last_speed[side] = 0.0
            return current.copy()
        # Pick the medoid of three quaternions. A single bad palm frame cannot win
        # the median; two consecutive frames in a new direction can.
        target = min(
            enumerate(history),
            key=lambda item: (sum(
                quaternion_angle_deg(item[1], other) for other in history),
                -item[0]))[1]
        last_target = self._last_target.get(side, target)
        speed = quaternion_angle_deg(last_target, target) / max(1e-6, dt)
        smooth_speed = 0.7 * self._last_speed.get(side, speed) + 0.3 * speed
        self._last_speed[side] = smooth_speed
        if self.min_cutoff_hz <= 0.0:
            filtered = target.copy()
        else:
            cutoff = self.min_cutoff_hz + self.beta * smooth_speed
            filtered = self._slerp(previous, target, _filter_alpha(cutoff, dt))
        self._last_target[side] = target.copy()
        self._filtered[side] = filtered
        return filtered


def elbow_bend_degrees(shoulder, elbow, wrist) -> float:
    """肘关节弯曲角（度）：0 = 完全伸直，180 = 完全折起。

    纯几何诊断量，用来对比「你的肘弯了多少、机器人的肘弯了多少」；NaN/inf 输入返回
    ``float("nan")``，调用方自己决定要不要展示（不参与控制）。
    """
    upper = np.asarray(elbow, dtype=np.float64) - np.asarray(shoulder, dtype=np.float64)
    lower = np.asarray(elbow, dtype=np.float64) - np.asarray(wrist, dtype=np.float64)
    norms = (float(np.linalg.norm(upper)), float(np.linalg.norm(lower)))
    if not all(np.isfinite(norms)) or min(norms) <= 0.0:
        return float("nan")
    cosine = float(np.dot(upper, lower)) / (norms[0] * norms[1])
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def arm_reach(arm_world) -> float:
    """上臂 + 前臂长度之和（米）；输入不可用时返回 NaN。"""
    if not _is_points(arm_world, (3, 3)):
        return float("nan")
    points = np.asarray(arm_world, dtype=np.float64)
    return float(np.linalg.norm(points[1] - points[0])
                 + np.linalg.norm(points[2] - points[1]))


def wrist_orientation_frame(arm_world):
    """(肩, 肘, 腕) 3x3 点 -> 3x3 腕坐标系（列 = x/y/z 基）；退化时 None。

    与论文的差别写在模块 docstring：论文从 SMPL-X 的旋转链拿手腕 4x4，我们用手上有的
    三点。定义（列向量）：

        x = 前臂方向 (腕 - 肘)        手臂指哪儿（前臂方向，2 个自由度）
        z = 上臂 x 前臂 的单位化       肘的拐向（手臂平面，1 个自由度）
        y = z x x                    补成右手系

    位置 3 + x 2 + z 1 = 6 个自由度，正是论文那条「手腕完整位姿」约束在三点输入下的
    等价物；缺的只有手的轴向扭转（那要用 21 点，见模块 docstring）。

    直臂（三点近共线）时上臂与「肘 -> 腕」的夹角正弦趋零，平面不定：返回 ``None``，
    由调用方保持上一帧，而不是猜一个方向。
    """
    if not _is_points(arm_world, (3, 3)):
        return None
    points = np.asarray(arm_world, dtype=np.float64)
    upper = points[1] - points[0]
    forearm = points[2] - points[1]
    upper_norm = float(np.linalg.norm(upper))
    forearm_norm = float(np.linalg.norm(forearm))
    if upper_norm <= 0.0 or forearm_norm <= 0.0:
        return None
    x_axis = forearm / forearm_norm
    normal = np.cross(upper / upper_norm, x_axis)
    sine = float(np.linalg.norm(normal))
    if not np.isfinite(sine) or sine < MIN_ELBOW_PLANE_SINE:
        return None
    z_axis = normal / sine
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def relative_wrist_rotation(arm_world, neutral_frame):
    """这一帧的腕坐标系相对中性那一帧的旋转 ``R_neutral^T @ R_now``；给不出时 None。

    论文那条「手腕完整位姿」约束的可用部分：轴向约定在相减里约掉，所以不需要相机外参。
    """
    return _relative_to_neutral(wrist_orientation_frame(arm_world), neutral_frame)


def _relative_to_neutral(frame, neutral_frame):
    """``R_neutral^T @ R_now``；任一帧给不出时 ``None``（两种基共用这一个减法）。

    相对写法有两个好处：轴向约定在相减里自然约掉（不需要相机外参），而且同一份目标能被
    三台机器人各自乘上自己的中性 EE 朝向。``frame`` 与 ``neutral`` 必须**同一套构造**
    —— 混用三点基与掌面基得到的差不是姿态误差，是垃圾。
    """
    if frame is None or neutral_frame is None:
        return None
    return np.asarray(neutral_frame, dtype=np.float64).T @ np.asarray(frame, dtype=np.float64)


def palm_orientation_frame(hand_world, arm_world, side):
    """(手 21 点度量world, (肩, 肘, 腕), 侧) -> 3x3 腕坐标系（列 = x/y/z 基）；给不出时 None。

    ``side`` 是**必须**给的（``"left"`` / ``"right"``）：关键点绕序左右镜像，掌面法向要按侧
    取反才能让两侧同手性，否则左手那一帧等于绕前臂多转 180 度（见 ``MIRRORED_PALM_SIDES``）。
    不给侧就等于把 2026-11-16 深夜那个「左手反着拧」的 bug 又装回去，所以这里不给默认值。

    与 ``wrist_orientation_frame`` 只差**第三轴是谁给的**：那一版的 z 取「上臂 x 前臂」
    的手臂平面法向 —— 绕前臂轴的自转（旋前/旋后）在那套几何里是自由度、量不到，
    于是 ``R_neutral^T @ R_now`` 对「手怎么拧」完全不敏感。掌面法向才量得到那一维，
    这正是 2026-11-16 现场「手怎么拧，机器人手腕一动不动」缺的那一维。

        x = 前臂方向 (腕 - 肘)        与 ``wrist_orientation_frame`` 逐字相同
        z = 掌面法向                  腕 0 / 食指 MCP 5 / 小指 MCP 17 的叉乘，
                                      投影到 x 的法平面再单位化
        y = z x x                    补成右手系

    x 故意仍取 **pose 的前臂方向**、不取掌面长轴（0 -> 9）：中性那一帧与这一帧的 x 定义
    必须一致，否则相减得到的不是旋前角；手部关键点只负责补上 z。

    ``arm_world`` 必须有：只靠手点也能凑出 x，但那与中性帧的口径不一致，宁可不给。
    退化（没有 pose 手臂 / 手点不全 / 掌面近退化 / 法向几乎与前臂平行）一律 ``None``，
    由调用方退回三点基或位置模式 —— 与 ``wrist_orientation_frame`` 一样不猜。
    """
    if not _is_points(arm_world, (3, 3)) or not _is_points(hand_world, (21, 3)):
        return None
    arm = np.asarray(arm_world, dtype=np.float64)
    hand = np.asarray(hand_world, dtype=np.float64)
    x_axis = _unit_direction(arm[2] - arm[1])
    if x_axis is None:
        return None
    origin, index, pinky = (hand[slot] for slot in PALM_NORMAL_SLOTS)
    normal = np.cross(index - origin, pinky - origin)
    if side in MIRRORED_PALM_SIDES:
        # 见 MIRRORED_PALM_SIDES：左手的关键点绕序与右手镜像，不取反就是「掌心朝内」。
        normal = -normal
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0.0:
        return None
    # 先归一化再投影：门槛是**夹角的正弦**，不能拿未归一化的叉乘长度（量纲 m^2）去比。
    normal = normal / length
    normal = normal - float(np.dot(normal, x_axis)) * x_axis
    sine = float(np.linalg.norm(normal))
    if not np.isfinite(sine) or sine < MIN_PALM_SINE:
        return None
    z_axis = normal / sine
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def anatomical_hand_frame(hand_world, arm_world, side):
    """21 点手 -> ``[手指方向, 拇指侧, 掌面法向]``，左右都保持右手系。

    ``palm_orientation_frame`` 的第三轴经过左右镜像修正，适合做相对旋转；但它的左手
    y 轴实际落在小指侧，不能直接套到机器人原生手的“拇指侧”轴。这里只做逐侧符号
    变换，不重新估法向，轴仍严格正交。
    """
    frame = palm_orientation_frame(hand_world, arm_world, side)
    if frame is None:
        return None
    signs = HAND_SEMANTIC_SIGNS.get(side)
    if signs is None:
        return None
    return frame @ np.diag(signs)


def relative_palm_rotation(hand_world, arm_world, neutral_frame, side):
    """这一帧的**掌面基**相对中性那一帧的旋转 ``R_neutral^T @ R_now``；给不出时 None。

    与 ``relative_wrist_rotation`` 的差别只有基的构造（见 ``palm_orientation_frame``）：
    它对绕前臂轴的自转敏感，于是「手旋前 90 度」真的会变成 90 度的姿态目标。``side`` 必须
    带下去 —— 掌面法向要按侧取反，否则左手那一帧等于绕前臂多转 180 度。
    """
    return _relative_to_neutral(palm_orientation_frame(hand_world, arm_world, side),
                                neutral_frame)


def relative_palm_wrist_rotation(hand_world, arm_world, neutral_local_frame, side):
    """手相对**前臂**的局部旋转；前臂本身的姿态变化由手臂位置任务负责。

    历史口径 ``relative_palm_rotation`` 返回 ``R_hand_neutral^T R_hand_now``。当操作者
    抬臂/摆肘时，这个旋转里会包含整条前臂的方向变化；把它交给只转腕的 IK，会让腕部
    用尽行程去补偿肩肘，现场表现为“摆臂时腕卡死”或“只有整条臂在动”。

    这里用三点基 ``wrist_orientation_frame`` 给前臂定义一个独立坐标系 ``F``，再用
    掌面基 ``H`` 表示手在前臂上的局部姿态：

        W_neutral = F_neutral^T H_neutral
        W_now     = F_now^T H_now
        W_relative = W_neutral^T W_now

    整个人和手臂作为一个刚体转动时，``F`` 与 ``H`` 同步转动，``W_relative`` 严格是
    单位阵；只有手掌相对前臂旋前/旋后、腕屈/腕偏时它才变化。这正好对应机器人已经由
    位置 IK 完成的前臂方向，加一个只由腕链兑现的局部姿态。

    ``neutral_local_frame`` 是标定（或迟到补采）时保存的 ``F^T H``。当前帧的掌面基和
    前臂三点基必须都可用；否则返回 ``None``，由调用方保持上一帧或只走位置模式，不从
    缺失的数据里猜一个腕角。
    """
    if neutral_local_frame is None:
        return None
    current_wrist = wrist_orientation_frame(arm_world)
    current_palm = palm_orientation_frame(hand_world, arm_world, side)
    if current_wrist is None or current_palm is None:
        return None
    current_local = current_wrist.T @ current_palm
    rotation = np.asarray(neutral_local_frame, dtype=np.float64).T @ current_local
    # 两次相减都来自实测基，乘积理论上仍是真旋转；投影一次，避免浮点误差进入 IK。
    left, _singular, right_t = np.linalg.svd(rotation)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _average_rotation(matrices):
    """一组 3x3 旋转的均值 -> 最接近的真旋转（SVD 投影）；没有可用值时 None。"""
    usable = [np.asarray(matrix, dtype=np.float64) for matrix in matrices
              if matrix is not None]
    if not usable:
        return None
    mean = np.mean(np.stack(usable, axis=0), axis=0)
    left, _singular, right_t = np.linalg.svd(mean)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _rotation_angle_deg(first, second) -> float:
    """两个 3x3 旋转之间的夹角（度）：``acos((tr(a^T b) - 1) / 2)``。

    ``_rotation_spread_deg`` 与「迟到基准」的连续性判据（``PALM_LATE_SPREAD_DEG``）共用
    这一个口径 —— 「这几帧彼此差多少度」在报告里和在这里必须是同一个数。
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _rotation_spread_deg(matrices, centre) -> float:
    """一组 3x3 旋转相对 ``centre`` 的**最大**夹角（度）；没有可用值 = ``0.0``。

    用途只有一个：判「这几帧掌面基是不是同一个姿势」（见 ``PALM_MIN_SAMPLE_FRACTION``）。
    标定时手刚进画面 / 半遮挡 / 掌面翻号，逐帧的法向可以差上百十度，``_average_rotation``
    照样算得出一个真旋转 —— 但那不是任何一帧的姿势，拿它当基准就是把腕目标整体拨偏。
    """
    if centre is None:
        return 0.0
    worst = 0.0
    for matrix in matrices:
        if matrix is None:
            continue
        worst = max(worst, _rotation_angle_deg(matrix, centre))
    return worst


def _palm_cluster(palms):
    """一组掌面基里**彼此一致的最大一簇**：``(均值, 簇内最大夹角, 被丢掉的帧数)``。

    为什么不直接对全部样本取均值（``_average_rotation``）：标定窗口里总有几帧掌面翻号 /
    半遮挡，实测两场（2026-11-17 15:40 与 16:00）掌面基的散度是 55 与 **165** 度 ——
    165 度不是「人转了手」，是那几帧的法向取反了。把正反两拨一起平均，得到的既不是正
    也不是反（接近正交），而这份垃圾基准会被``PALM_MAX_SPREAD_DEG`` 判死，于是**整场
    丢掉旋前**（同一场 ``ori_source`` 里 ``palm_normal`` 只有 3 行，见 ``PALM_LATE_FRAMES``）。

    改成「最大一致簇」：以每一帧为候选中心，数有多少帧落在 ``PALM_MAX_SPREAD_DEG`` 之内，
    取最大的那一簇做均值，其余帧数如实报成 ``palm_outliers``。闸门的两条判据随之改成
    「簇的大小够不够占标定帧的比例」+「簇内散度」—— 少数坏帧不再一票否决，多数一致
    仍然一票否决（12/30 帧凑出来的基准依旧过不去）。
    """
    usable = [np.asarray(matrix, dtype=np.float64) for matrix in palms if matrix is not None]
    if not usable:
        return None, 0.0, 0
    best = None
    for candidate in usable:
        members = [other for other in usable
                   if _rotation_angle_deg(other, candidate) <= PALM_MAX_SPREAD_DEG]
        if best is None or len(members) > len(best):
            best = members
    basis = _average_rotation(best)
    if basis is None:                     # 理论到不了（每帧至少和自己一致），保险
        return None, 0.0, len(usable)
    return basis, _rotation_spread_deg(best, basis), len(usable) - len(best)



@dataclass(frozen=True)
class AxisMap:
    """操作者系的轴到机器人基座系的轴：每个机器人轴由哪一个操作者轴、带什么符号。

    ``order``/``signs`` 按机器人轴顺序 (X, Y, Z) 给：``order[i]`` 是被取用的操作者轴名，
    ``signs[i]`` 是它的正负号。默认 (``("Z","X","Y")``, ``(-1,1,-1)``) 即模块 docstring 里
    那三行。这里只做置换与符号，不含旋转矩阵拟合 —— 真正的相机外参要标定。
    """

    order: tuple = DEFAULT_AXIS_ORDER
    signs: tuple = DEFAULT_AXIS_SIGNS

    def __post_init__(self) -> None:
        if sorted(self.order) != sorted(AXIS_NAMES):
            raise ValueError(
                f"axis order must be a permutation of X/Y/Z, got {tuple(self.order)!r}")
        if len(self.signs) != 3:
            raise ValueError(f"axis signs must have 3 entries, got {tuple(self.signs)!r}")
        for sign in self.signs:
            if float(sign) not in (-1.0, 1.0):
                raise ValueError(f"axis signs must be +1 or -1, got {tuple(self.signs)!r}")

    def matrix(self) -> np.ndarray:
        """3x3 矩阵 ``M``：``d_base = M @ d_op``。"""
        index = {name: position for position, name in enumerate(AXIS_NAMES)}
        matrix = np.zeros((3, 3), dtype=np.float64)
        for row, (name, sign) in enumerate(zip(self.order, self.signs)):
            matrix[row, index[name]] = float(sign)
        return matrix

    def apply(self, vector) -> np.ndarray:
        """把操作者系的位移向量转到机器人基座系的轴序（只换轴，不缩放）。"""
        values = np.asarray(vector, dtype=np.float64).reshape(3)
        return self.matrix() @ values

    def describe(self) -> dict:
        """JSON 化：给报告和命令行回显用。"""
        return {"order": list(self.order), "signs": [float(sign) for sign in self.signs],
                "robot_axes_from_operator": [
                    f"{name}_robot = {sign:+.0f} * {self.order[row]}_op"
                    for row, (name, sign) in enumerate(
                        zip(AXIS_NAMES, self.signs))]}

    @classmethod
    def parse(cls, order: str, signs: str) -> "AxisMap":
        """命令行写法：``--arm-axis-order ZXY --arm-axis-signs -1,1,-1``（逗号或空格都认）。"""
        letters = str(order).strip().upper().replace(",", "").replace(" ", "")
        values = [part for part in str(signs).replace(" ", ",").split(",") if part]
        if len(values) == 1 and len(values[0]) >= 3:
            values = [values[0][index] for index in range(3)]
        return cls(tuple(letters), tuple(float(value) for value in values))

    def with_defaults(self, order=None, signs=None) -> "AxisMap":
        """用可选覆盖值得到一个新的映射（``None`` = 保持原值）。"""
        return AxisMap(tuple(order) if order else self.order,
                       tuple(signs) if signs else self.signs)


@dataclass(frozen=True)
class ArmTarget:
    """一台机器人的一侧、某一帧的手臂目标（机器人基座系）。

    ``valid=True`` 表示这一帧真的解算出了新目标；``held=True`` 表示沿用上一帧（该侧
    没数据 / 还在标定 / 目标非法）。``source`` 说明用的是 pose 手臂还是手部兜底映射，
    ``info`` 里是可检查的诊断量（你移动了多少、肘弯了多少、有没有被行程夹住）。

    ``orientation`` 是**相对中性**的四元数（``R_neutral^T @ R_now``）：喂给
    ``PyBulletArmIK.solve(orientation_frame="neutral")`` 才会乘上本机的中性 EE 朝向，
    直接当绝对姿态用是错的。
    """

    side: str
    position: tuple | None = None
    #: 肘的目标点（基座系）。只有 ``elbow_task=True`` 且量到了锚点才给 —— 这是论文的
    #: 「肘位置任务」（``PyBulletArmIK.solve(elbow=...)``）：只给腕一个位置时，7 自由度
    #: 冗余链会让整条臂纹丝不动（实测 0.01 度），把肘也交进去肩才会转。腕目标与它来自
    #: 同一条按连杆长缩放的链，所以两个目标同时可达。
    elbow: tuple | None = None
    orientation: tuple | None = None
    valid: bool = False
    held: bool = False
    reason: str = ""
    source: str = SOURCE_NONE
    info: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """JSON 化（报告与 HUD 共用）。"""
        return {
            "side": self.side,
            "position": None if self.position is None else [round(float(v), 5) for v in self.position],
            "elbow": None if self.elbow is None else [round(float(v), 5) for v in self.elbow],
            "orientation": None if self.orientation is None else [round(float(v), 6) for v in self.orientation],
            "valid": bool(self.valid),
            "held": bool(self.held),
            "reason": self.reason,
            "source": self.source,
            "info": self.info,
        }


def _neutral_default() -> dict:
    return {"offset": None, "reach": None, "orientation": None,
            # 掌面基的中性腕坐标系（第三轴 = 掌面法向，见 ``palm_orientation_frame``）。
            # 老记录 / 标定期间一直没手点时是 None -> 姿态退回三点基（历史行为）。
            "orientation_palm": None,
            # 手相对前臂的局部姿态 ``F_neutral^T H_neutral``：腕姿态的 ``wrist`` 口径
            # 用它作为零点；它和绝对掌面基分开存，晚期补采时不会把前臂方向变化算进腕。
            "orientation_palm_local": None,
            # 上臂段 (肘-肩) 与「前臂+手」段 (腕-肘) 的中性向量：两段式映射要用的基准。
            # 老记录里没有这两项 -> None，那一侧自动退回整段 gain（不会算错，只是不生效）。
            "upper": None, "fore": None,
            "samples": 0, "complete": False}


def _neutral_rotation(orientation):
    """中性记录里的腕坐标系 -> 3x3；缺 / 为空 / 形状不对时 ``None``。

    兼容三种写法：``None``（老记录，那时还不记姿态）、JSON 里的 9 个数、已经是 3x3 的数组。
    """
    if orientation is None:
        return None
    values = np.asarray(orientation, dtype=np.float64)
    if values.size != 9:
        return None
    return values.reshape(3, 3)


def load_neutral(path) -> dict:
    """读一份 ``--save-neutral`` 写出（或 ``--neutral-json`` 指定）的中性姿势记录。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    neutral = {}
    for side in SIDES:
        entry = payload.get("sides", {}).get(side, {})
        offset = entry.get("offset")
        orientation = entry.get("orientation")
        neutral[side] = {
            "offset": None if offset is None else np.asarray(offset, dtype=np.float64),
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
            # 中性那一帧的腕坐标系（相对姿态的基准）。老记录没有这一段 -> None，
            # 于是这一侧退化成位置模式，而不是把缺的姿态当成零。
            "orientation": _neutral_rotation(orientation),
            # 同一帧的**掌面基**（第三轴 = 掌面法向）：只有它带旋前/旋后。老记录没有
            # 这一段 -> None，姿态自动退回三点基，行为与改动前逐字一致。
            "orientation_palm": _neutral_rotation(entry.get("orientation_palm")),
            # 掌面基相对前臂基的局部姿态。新记录直接存；老记录读入时按
            # ``F_neutral^T H_neutral`` 现场补齐，兼容已有 neutral JSON。
            "orientation_palm_local": _neutral_rotation(
                entry.get("orientation_palm_local")),
            # 两段式映射（--arm-map segments）的基准：上臂段 / 「前臂+手」段。
            "upper": _neutral_vector(entry.get("upper"), 3),
            "fore": _neutral_vector(entry.get("fore"), 3),
            "samples": int(entry.get("samples", 0)),
            "complete": bool(entry.get("complete", False)) and offset is not None,
            # 掌面基的来路 / 零点偏移（报告与 CLI 的 ``[INFO]`` 用；缺了也能跑）。
            "palm_reference_source": str(entry.get("palm_reference_source") or ""),
            "palm_reference_frame": entry.get("palm_reference_frame"),
            "palm_reference_pose_deg": entry.get("palm_reference_pose_deg"),
        }
        record = neutral[side]
        if (record["orientation_palm_local"] is None
                and record["orientation"] is not None
                and record["orientation_palm"] is not None):
            record["orientation_palm_local"] = (
                record["orientation"].T @ record["orientation_palm"])
    return {"sides": neutral, "axis_map": payload.get("axis_map"),
            "created": payload.get("created"), "path": str(path)}


def _neutral_vector(value, size: int):
    """中性记录里的一段向量（上臂段 / 前臂段）-> ``ndarray``；缺 / 形状不对时 ``None``。

    跟 ``_neutral_rotation`` 一个口径：老记录（或手写错的记录）不给这一段时返回 ``None``，
    调用方据此退回整段 gain，而不是把 ``None`` 当成零向量算出错误位移。
    """
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size:
        return None
    return vector


def save_neutral(path, sides: dict, extra: dict | None = None) -> Path:
    """把中性姿势写进 JSON，方便下次直接复用（不用再站一次）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "realtime_arm_neutral_v2", "extra": extra or {}, "sides": {}}
    for side, entry in sides.items():
        offset = entry.get("offset")
        payload["sides"][side] = {
            "offset": None if offset is None else [float(v) for v in np.asarray(offset).reshape(3)],
            "reach": None if entry.get("reach") is None else float(entry["reach"]),
            # 9 个数（行主序）：中性那一帧的腕坐标系，读回来就是相对姿态的基准。
            "orientation": (None if entry.get("orientation") is None else
                            [float(value) for value in
                             np.asarray(entry["orientation"], dtype=np.float64).reshape(9)]),
            # 同一帧的掌面基（第三轴 = 掌面法向）。没有就是 null：读回来时姿态退回三点基。
            "orientation_palm": (None if entry.get("orientation_palm") is None else
                                 [float(value) for value in
                                  np.asarray(entry["orientation_palm"],
                                             dtype=np.float64).reshape(9)]),
            "orientation_palm_local": (
                None if entry.get("orientation_palm_local") is None else
                [float(value) for value in
                 np.asarray(entry["orientation_palm_local"],
                            dtype=np.float64).reshape(9)]),
            # 两段式映射的基准（上臂段 / 「前臂+手」段，米）。没采到就写 null。
            "upper": None if entry.get("upper") is None else
                     [float(v) for v in np.asarray(entry["upper"]).reshape(3)],
            "fore": None if entry.get("fore") is None else
                    [float(v) for v in np.asarray(entry["fore"]).reshape(3)],
            "samples": int(entry.get("samples", 0)),
            "complete": bool(entry.get("complete", False)),
            # 掌面基的来路（标定那次 / 运行中补采）+ 零点偏移：只影响报告的可解释性，
            # 少了也能跑（读回来当作 "neutral"）。
            "palm_reference_source": str(entry.get("palm_reference_source", "")),
            "palm_reference_frame": entry.get("palm_reference_frame"),
            "palm_reference_pose_deg": entry.get("palm_reference_pose_deg"),
        }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _side_map(value) -> dict:
    """``{side: ...}`` 的宽松读取：支持 dict / ``UpperBodyObservation`` / ``RawHandFrame``。"""
    if value is None:
        return {}
    if hasattr(value, "world_arms"):
        return dict(value.world_arms or {})
    if isinstance(value, dict):
        return value
    return {}


def _hand_frames(hands) -> tuple:
    """-> ``(normalized, world)`` 两套 21 点字典；缺哪套哪套为 ``{}``。"""
    normalized, world = {}, {}
    if hands is None:
        return normalized, world
    if hasattr(hands, "hands"):
        normalized = dict(getattr(hands, "hands") or {})
        world = dict(getattr(hands, "world_hands") or {})
        return normalized, world
    if isinstance(hands, dict):
        for side in SIDES:
            normalized[side] = hands.get(side)
            world[side] = hands.get(f"world_{side}")
    return normalized, world


class RealtimeArmBridge:
    """一帧摄像头数据 -> 每台机器人每侧一个 ``ArmTarget``（基座系，可直接喂 IK）。

    用法（每个摄像头帧调用一次）::

        targets = bridge.update(arms_world=obs.upper_body, hands=raw_frame)
        for robot_type, per_side in targets.items():
            for side, target in per_side.items():
                if target.position is not None:
                    driver.apply_each(side, {robot_type: target.position},
                                      source="base", orientation=target.orientation)

    这个类只做数学与状态：不 import PyBullet、不开摄像头、不写关节。因此它可以被单元
    测试直接驱动（``tests/test_arm_realtime_bridge.py``），仿真侧只负责把 ``position``
    交给 ``PyBulletArmIK.solve(source="base")``。
    """

    def __init__(self, profiles=(), sides=SIDES, axis_map=None,
                 calibration_frames: int = 30,
                 calibrate_mode: str = "hold",
                 palm_wait_frames: int = PALM_WAIT_FRAMES,
                 arm_length_reference: float = REFERENCE_REACH_M,
                 normalise_arm_length: bool = True,
                 arm_length_band=ARM_LENGTH_BAND,
                 orientation: str = "auto",
                 orientation_mode: str = "hand",
                 hand_roll_deg: float = 180.0,
                 jitter_filter: bool = False,
                 palm_hold_s: float = PALM_HOLD_S,
                 max_orientation_rate_deg_s: float = MAX_ORIENTATION_RATE_DEG_S,
                 max_orientation_step_deg: float = MAX_ORIENTATION_STEP_DEG,
                 gains=None, max_travel_m=None, neutral=None,
                 travel_mode: str = "clip",
                 lateral_guard_m: float = DEFAULT_LATERAL_GUARD_M,
                 gain_quality_guard: bool = True,
                 min_plausible_reach_m: float = MIN_PLAUSIBLE_ARM_REACH_M,
                 arm_reach_side_tolerance: float = ARM_REACH_SIDE_TOLERANCE,
                 fallback_hand: bool = True,
                 fallback_scale: float = FALLBACK_IMAGE_SCALE,
                 map_mode: str = "chain",
                 robot_links=None,
                 elbow_task: bool = False,
                 elbow_gain: float = 1.0,
                 posture_gain: float = 1.0,
                 max_direction_rate_deg_s: float = MAX_DIRECTION_RATE_DEG_S,
                 frame_dt: float | None = None,
                 anchors=None):
        self.sides = tuple(sides)
        for side in self.sides:
            if side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if map_mode not in MAP_MODES:
            raise ValueError(f"map_mode must be one of {MAP_MODES}, got {map_mode!r}")
        if elbow_task and map_mode not in ("segments", "posture"):
            raise ValueError(
                "elbow_task needs map_mode='segments' or map_mode='posture' (in posture "
                "the elbow target is built by that recipe itself): the elbow and the "
                "wrist target must come from the same length-scaled chain, otherwise "
                "the two position tasks fight each other (measured 2026-10-13: with the "
                "wrist pinned and the elbow told to move 100 mm, the wrist residual "
                "went to 107 mm and the elbow still missed its target by 200+ mm)")
        if int(calibration_frames) < 1:
            raise ValueError("calibration_frames must be >= 1")
        if calibrate_mode not in CALIBRATE_MODES:
            raise ValueError(f"calibrate_mode must be one of {CALIBRATE_MODES},"
                             f" got {calibrate_mode!r}")
        if orientation not in ("off", "auto", "on"):
            raise ValueError(
                f"orientation must be 'off', 'auto' or 'on', got {orientation!r}")
        if orientation_mode not in ORIENTATION_MODES:
            raise ValueError(
                f"orientation_mode must be one of {ORIENTATION_MODES}, "
                f"got {orientation_mode!r}")
        self.profiles = {}
        for profile in profiles:
            if isinstance(profile, ArmProfile):
                self.profiles[profile.robot_type] = profile
            else:
                raise TypeError(f"profiles must contain ArmProfile, got {profile!r}")
        if not self.profiles:
            raise ValueError("at least one ArmProfile is required")
        self.robots = tuple(self.profiles)
        self.axis_map = axis_map or AxisMap()
        # 「只动大臂也要有反应」的那一条开关：chain = 历史口径（肩->腕整段），
        # segments = 论文 scale_joints 的两段式（靠 set_robot_links 得到的连杆长比缩放），
        # posture = 方向口径（你的单位方向 + 机器人自己的连杆长，不搬位移）。
        self.map_mode = str(map_mode)
        self.robot_links = {}
        self.set_robot_links(robot_links)
        #: 「肘位置任务」：把肘也当目标交给 IK（只在 segments 口径下有意义，见 __init__）。
        self.elbow_task = bool(elbow_task)
        #: 肘目标的力度 0-1：1 = 整段上臂位移（肩最积极、腕残差最大），越小越像只喂腕。
        #: 两个位置目标在 PyBullet 里是等权的（没有优先级），所以这条是唯一的折中旋钮。
        if not 0.0 <= float(elbow_gain) <= 1.0:
            raise ValueError(f"elbow_gain must be within 0..1, got {elbow_gain!r}")
        self.elbow_gain = float(elbow_gain)
        #: 方向口径的力度 0-1（``map_mode="posture"``）：1 = 机器人方向完全对齐你的方向
        #: （默认），0.5 = 停在机器人中性方向与你方向的夹角正中，0 = 回中性。它不是位移
        #: 缩放，见 _slerp_direction。
        if not 0.0 <= float(posture_gain) <= 1.0:
            raise ValueError(f"posture_gain must be within 0..1, got {posture_gain!r}")
        self.posture_gain = float(posture_gain)
        self.robot_anchors = {}
        self.set_robot_anchors(anchors)
        self.calibration_frames = int(calibration_frames)
        #: 标定没攒够帧时的口径（``hold`` / ``track``，见 ``CALIBRATE_MODES``）。
        self.calibrate_mode = str(calibrate_mode)
        self.arm_length_reference = float(arm_length_reference)
        self.normalise_arm_length = bool(normalise_arm_length)
        self.arm_length_band = (float(arm_length_band[0]), float(arm_length_band[1]))
        self.orientation = orientation
        #: ``hand_base`` 直接目标可见手掌三轴；``wrist`` 只传局部腕旋转；``hand`` 保留
        #: 历史整手相对旋转。CLI 默认 ``hand_base``；库默认 ``hand``，避免旧探针静默换口径。
        self.orientation_mode = str(orientation_mode)
        #: 操作者掌面约定 -> 机器人原生掌面约定的绕手指轴滚转。默认 180°：
        #: MediaPipe 的掌心正反与机器人原生手掌可视正面相反，若直接映射会出现
        #: “拇指内侧 vs 外侧、掌面正反互换”。0° 仅用于 A/B。
        self.hand_roll_deg = float(hand_roll_deg)
        if not np.isfinite(self.hand_roll_deg):
            raise ValueError("hand_roll_deg must be finite")
        self._hand_roll_matrix = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, np.cos(np.radians(self.hand_roll_deg)),
              -np.sin(np.radians(self.hand_roll_deg))],
             [0.0, np.sin(np.radians(self.hand_roll_deg)),
              np.cos(np.radians(self.hand_roll_deg))]], dtype=np.float64)
        #: MediaPipe 慢速抖动的前置自适应滤波。关闭只为 A/B；快速运动时带宽自动上升。
        self.jitter_filter = bool(jitter_filter)
        self._arm_filter = AdaptiveArmPointFilter()
        self._hand_filter = AdaptiveRotationFilter()
        #: 这一帧当前手掌几何坐标系（列 = 指向前 / 拇指侧 / 掌面法向），已由
        #: ``AxisMap`` 换到机器人基座轴。``hand_base`` 模式直接消费它。
        self._current_hand_frames = {side: None for side in self.sides}
        #: 掌面基丢帧时沿用上一次掌面旋转的窗口（秒，见 ``PALM_HOLD_S``）。0 = 立即退回三点基。
        if float(palm_hold_s) < 0.0:
            raise ValueError("palm_hold_s must be >= 0 (0 turns the hold off)")
        self.palm_hold_s = float(palm_hold_s)
        self.gains = {str(name): float(value) for name, value in (gains or {}).items()}
        self.max_travel_m = (None if max_travel_m is None else float(max_travel_m))
        if travel_mode not in TRAVEL_MODES:
            raise ValueError(f"travel_mode must be one of {TRAVEL_MODES}, got {travel_mode!r}")
        if float(lateral_guard_m) < 0.0:
            raise ValueError("lateral_guard_m must be >= 0 (0 turns the guard off)")
        if float(min_plausible_reach_m) <= 0.0:
            raise ValueError("min_plausible_reach_m must be > 0")
        self.travel_mode = str(travel_mode)
        self.lateral_guard_m = float(lateral_guard_m)
        self.gain_quality_guard = bool(gain_quality_guard)
        self.min_plausible_reach_m = float(min_plausible_reach_m)
        self.arm_reach_side_tolerance = float(arm_reach_side_tolerance)
        self.fallback_hand = bool(fallback_hand)
        self.fallback_scale = float(fallback_scale)
        self.frames = 0
        self._samples = {side: [] for side in self.sides}
        self.neutral = {side: _neutral_default() for side in self.sides}
        #: 标定期间**真的拿到过**掌面基的帧数。0 = 这一整场量不到旋前/旋后（见
        #: ``_raw_orientation``：掌面基要跟中性里的 ``orientation_palm`` 相减，它是 None 就
        #: 只能退回三点基）。2026-11-16 晚那场就是标定那 30 帧手不在画面里，整场丢旋前跟踪。
        self._palm_samples = {side: 0 for side in self.sides}
        #: 这一侧**已经见过**掌面基（之后丢了就不再等，见 ``PALM_WAIT_FRAMES``）。
        self._palm_seen = {side: False for side in self.sides}
        #: 为了等手点，已经**放过**多少帧没计入标定（到 ``palm_wait_frames`` 就停止等）。
        self._palm_wait = {side: 0 for side in self.sides}
        #: 「迟到基准」的候选窗口：每侧最近几帧掌面基（见 ``PALM_LATE_FRAMES``）。
        #: 只在标定完成后、且中性里没有掌面基时才攒；一断帧就清空（必须**连续**）。
        self._palm_late = {side: [] for side in self.sides}
        #: 真正计入标定的帧数（``calibration_frames`` 与 HUD 进度都按它）。
        #: 与 ``len(_samples[side])`` 的差 = 为等手点而放过的那几帧。
        self._counted = {side: 0 for side in self.sides}
        #: 「你的两条手臂是否镜像」的逐帧镜像偏差（度，见 ``SYMMETRY_SAME_SIDE_DEG``）。
        #: 只在这里攒数字，不参与任何控制量；报告里给中位/最大与 ``same_side`` 帧数。
        self._symmetry = []
        #: 攒不进去（超过 ``SYMMETRY_MAX_SAMPLES``）之后又来了多少帧 —— 计数照旧，样本封顶。
        self._symmetry_extra = 0
        #: 等手点的上限（帧；``0`` = 不等，A/B 与历史行为）。
        self.palm_wait_frames = max(0, int(palm_wait_frames))
        self._last_state = {side: None for side in self.sides}
        #: 关键点跳变护栏的角速度上限（度/秒）：0 = 关（A/B 用）。见 ``_guard_direction``。
        self.max_direction_rate_deg_s = max(0.0, float(max_direction_rate_deg_s))
        #: 关键点跳变护栏用的「上一帧通过的方向」（每侧 ``(upper, fore)`` 米制向量）。
        self._last_direction = {}
        #: 每侧连续被拦的帧数（见 ``GLITCH_HOLD_FRAMES``）。
        self._glitch_streak = {}
        #: 每侧最近一次可用的掌面旋转 ``(四元数, 距今秒数)``；丢帧时按 ``PALM_HOLD_S``
        #: 沿用它（见 ``_orientation_frame``）。
        self._palm_hold = {side: None for side in self.sides}
        #: 腕姿态目标的角速度上限（度/秒）：0 = 关（A/B 用）。见 ``_clamp_orientation``。
        self.max_orientation_rate_deg_s = max(0.0, float(max_orientation_rate_deg_s))
        #: 腕姿态目标**每帧**最多走多少度（与帧长无关）：0 = 只按速率，见
        #: ``MAX_ORIENTATION_STEP_DEG``。
        self.max_orientation_step_deg = max(0.0, float(max_orientation_step_deg))
        #: 每侧上一帧**放行**的姿态目标（限速的链式基准，见 ``_clamp_orientation``）。
        self._last_orientation = {}
        #: 这一帧的时长（秒）：护栏的角速度上限按它换算。见 _guard_direction。
        #: 构造时给一个值 = 这个调用方按固定帧率跑（``update(frame_dt=...)`` 仍可逐帧覆盖）。
        self.frame_dt = NOMINAL_FRAME_DT_S
        if frame_dt is not None:
            value = float(frame_dt)
            if value > 0.0:
                self.frame_dt = value
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},
                       "fallback": {side: 0 for side in self.sides},
                       "oriented": {side: 0 for side in self.sides},
                       #: 掌面基这一帧丢了、沿用上一帧掌面旋转的帧数（见 ``PALM_HOLD_S``）。
                       "palm_held": {side: 0 for side in self.sides},
                       #: 运行中补采到「迟到基准」的次数（0/1；见 ``PALM_LATE_FRAMES``）。
                       "palm_late": {side: 0 for side in self.sides},
                       #: 姿态目标被限速削过的帧数（见 ``MAX_ORIENTATION_RATE_DEG_S``）。
                       "orientation_clamped": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides},
                       "guarded": {side: 0 for side in self.sides},
                       "segments": {side: 0 for side in self.sides},
                       "posture": {side: 0 for side in self.sides},
                       "elbow": {side: 0 for side in self.sides},
                       "direction_glitch": {side: 0 for side in self.sides},
                       "direction_glitch_kept": {side: 0 for side in self.sides},
                       "gain_guarded": {side: 0 for side in self.sides},
                       #: 「这一帧没跟手」按 ``HOLD_REASONS`` 分类，现场排查先看它。
                       "held_reason": {side: {kind: 0 for kind in HOLD_REASONS}
                                       for side in self.sides}}
        if neutral is not None:
            self.load_neutral(neutral)

    # ------------------------------------------------------------------ #
    # 中性姿势（标定的替代品）：站好一次，记下「腕 - 肩」与臂长
    # ------------------------------------------------------------------ #
    def load_neutral(self, payload) -> None:
        """装入一份中性记录（``load_neutral()`` 的返回值或它的 ``sides`` 段）。"""
        sides = payload.get("sides", payload) if isinstance(payload, dict) else {}
        for side in self.sides:
            entry = sides.get(side) or {}
            offset = entry.get("offset")
            self.neutral[side] = {
                "offset": None if offset is None else np.asarray(offset, dtype=np.float64).reshape(3),
                "reach": None if entry.get("reach") is None else float(entry["reach"]),
                # 姿态基准：老记录里没有 -> None（这一侧退化成位置模式）。
                "orientation": _neutral_rotation(entry.get("orientation")),
                # 掌面基的姿态基准：老记录里没有 -> None（姿态退回三点基，同改动前）。
                "orientation_palm": _neutral_rotation(entry.get("orientation_palm")),
                # 两段式映射的基准：老记录里没有 -> None（那一侧退回整段 gain）。
                "upper": _neutral_vector(entry.get("upper"), 3),
                "fore": _neutral_vector(entry.get("fore"), 3),
                "samples": int(entry.get("samples", 0)),
                # 标定期间拿到掌面基的帧数（0/缺 = 这一整场量不到旋前，见 _palm_samples）。
                "palm_samples": int(entry.get("palm_samples", 0)),
                # 掌面基不可用的原因 / 那几帧彼此的散度（空串 + 缺 = 能用或没样本）。
                "palm_note": str(entry.get("palm_note") or ""),
                "palm_spread_deg": (None if entry.get("palm_spread_deg") is None
                                    else float(entry["palm_spread_deg"])),
                # 掌面基是标定那次采的还是运行中补采的（``neutral`` / ``late`` / 空串）。
                # 补采的那个零点不是标定那一刻，报告里必须分辨得出来（见 PALM_LATE_FRAMES）。
                "palm_reference_source": str(entry.get("palm_reference_source") or ""),
                "palm_reference_frame": (None if entry.get("palm_reference_frame") is None
                                         else int(entry["palm_reference_frame"])),
                # 迟到基准取零点时离标定姿势多远（度）：腕从那个姿势起读数为零。
                "palm_reference_pose_deg": (None if entry.get("palm_reference_pose_deg") is None
                                            else float(entry["palm_reference_pose_deg"])),
                "palm_outliers": int(entry.get("palm_outliers") or 0),
                "complete": bool(entry.get("complete", False)) and offset is not None,
            }

    def neutral_record(self) -> dict:
        """``{side: {offset, reach, upper, fore, samples, complete}}``（可直接 ``save_neutral``）。"""
        return {side: dict(entry) for side, entry in self.neutral.items()}

    # ------------------------------------------------------------------ #
    # 两段式映射（论文 scale_joints）：把「只动大臂」也变成目标位移
    # ------------------------------------------------------------------ #
    def set_robot_links(self, links) -> None:
        """每台机器人每侧的实测连杆长度（米）：``{robot: {side: {"upper": .., "fore": ..}}}``。

        仿真实例加载完之后用 FK 量出来（``realtime_teleop`` 启动时做一次），量不到就为空。
        没有它就没有「机器人连杆长 / 你的连杆长」这个比值，``map_mode="segments"`` 会退回
        整段 gain；理由是写进 ``summary()["robot_links"]`` 的，别让下一个人猜。
        """
        self.robot_links = {}
        for robot_type, per_side in dict(links or {}).items():
            entries = {}
            for side, value in dict(per_side or {}).items():
                if not isinstance(value, dict):
                    continue
                upper = float(value.get("upper") or 0.0)
                fore = float(value.get("fore") or 0.0)
                if upper > 0.0 and fore > 0.0:
                    entries[str(side)] = {"upper": upper, "fore": fore}
            if entries:
                self.robot_links[str(robot_type)] = entries

    def set_robot_anchors(self, anchors) -> None:
        """每台机器人每侧的中性腕/肘点（**基座系**，米）：``{robot: {side: {"elbow","wrist"}}}``。

        同样由仿真实例在加载后量出来（FK 到那个姿势），只在 ``elbow_task=True`` 时用：肘目标
        = 中性肘 + M(上臂段位移)，也就是「从机器人自己的中性肘点出发、按你的上臂方向转」。

        ``map_mode="posture"`` 还要 ``"shoulder"``（同一个中性姿势下的肩 FK 点）：腕/肘目标
        就是「肩锚点 + 机器人自己的连杆长 * 你的单位方向」。老调用方不给它不会报错，只是
        那一侧在方向口径下退回位移口径，并把原因写进那一帧的 ``info``。

        **这个 "shoulder" 必须是上臂的肩端**（= 肘关节的父连杆原点，``arm_points()["upper"]``），
        即与肘刚性相连的那个点：只有它到肘的距离**与姿势无关**，配方的两个目标才同时可达。
        用名字匹配到的肩关节连杆会随 shoulder_roll 摆动 48-102 mm（实测），配方在非中性姿势下
        就几何上不可达 —— 现场「前臂弯曲方向相反」的根因，见 ``_posture_target``。
        ``realtime_teleop.build_scene`` 与 ``arm_scene_driver._shoulder_shift`` 都按同一口径取点。
        """
        self.robot_anchors = {}
        for robot_type, per_side in dict(anchors or {}).items():
            entries = {}
            for side, value in dict(per_side or {}).items():
                if not isinstance(value, dict):
                    continue
                elbow = _neutral_vector(value.get("elbow"), 3)
                wrist = _neutral_vector(value.get("wrist"), 3)
                if elbow is not None and wrist is not None:
                    entry = {"elbow": elbow, "wrist": wrist}
                    shoulder = _neutral_vector(value.get("shoulder"), 3)
                    if shoulder is not None:
                        entry["shoulder"] = shoulder
                    entries[str(side)] = entry
            if entries:
                self.robot_anchors[str(robot_type)] = entries

    def _human_segments(self, side):
        """中性标定量出来的 ``(上臂长, 前臂长)``（米）；还没标定就是 ``None``。"""
        entry = self.neutral[side]
        if entry.get("upper") is None or entry.get("fore") is None:
            return None
        return (float(np.linalg.norm(entry["upper"])),
                float(np.linalg.norm(entry["fore"])))

    def segment_scale_note(self, side) -> str:
        """这一侧的 ``k`` 为什么不能用（空串 = 能用 / 还没标定）。

        判据只涉及**你的**段长，与机器人无关，所以一条中性记录就能定论：标定那
        ``--arm-calibrate-frames`` 帧里手臂被深度压扁（离镜头太远 / 侧身 / 遮挡）时，
        量出来的前臂会比真实短一截，``k`` 就被放大成几倍。见 ``SEGMENT_LENGTH_BAND_M``。
        """
        lengths = self._human_segments(side)
        if lengths is None:
            return ""
        low, high = SEGMENT_LENGTH_BAND_M
        for name, length in zip(("upper arm", "forearm"), lengths):
            if low <= length <= high:
                continue
            return (f"{side} calibrated {name} {length:.3f} m is outside "
                    f"{low:.2f}-{high:.2f} m (pose landmarks bad)")
        return ""

    def palm_basis_note(self, side) -> str:
        """这一侧的掌面基为什么被判不可用（空串 = 能用 / 还没有样本）。

        与 ``segment_scale_note`` 同一个套路：**判据只涉及采集过程，不涉及机器人**，
        所以一次标定就能定论，且不合格时只会让这一侧退回三点基（旋前不跟），
        不会把六个腕关节推向限位（见 ``PALM_MIN_SAMPLE_FRACTION`` 的实测数字）。
        """
        return str(self.neutral[side].get("palm_note") or "")

    def _palm_basis_note(self, side, samples: int, spread_deg: float,
                         outliers_spread_deg: float = None) -> str:
        """``palm_basis_note`` 的判据本体：``samples`` = **多数簇**里的帧数（见 ``_palm_cluster``）。

        样本数为 ``0`` 时返回空串 —— 那是「这一侧根本没手点」，由
        ``realtime_teleop._warn_missing_palm_basis`` 按 ``palm_samples=0`` 报，
        与「有样本但不可信」是两件不同的事，不能混成一条。

        **必须是过半**（``<=`` 就判死，不是 ``<``）：恰好一半的帧彼此一致 = 另一半的帧
        说的是别的事，那份均值仍然不代表任何一帧。现场那两场（15:40 / 16:00）的散度
        55 / 165 度就是这种「一半正一半反」的样子，只看「有没有 15 帧」是拦不住的。

        ``outliers_spread_deg`` = 散出去的那几帧相对基准的最大夹角（簇外散度）：两条判据
        都不成立时说清是哪一条 —— 「帧数不够」和「帧数够但彼此矛盾」现场要改的事不一样。
        """
        if samples <= 0:
            return ""
        frames = int(self.calibration_frames)
        wanted = max(2, int(round(float(frames) * PALM_MIN_SAMPLE_FRACTION)))
        reasons = []
        if samples <= wanted:
            reasons.append(f"only {samples}/{frames} calibration frames are consistent "
                           f"(needs a clear majority, > {wanted}): the hand was barely in "
                           f"view while the neutral was built")
        if spread_deg > PALM_MAX_SPREAD_DEG:
            reasons.append(f"the {samples} consistent frames still disagree by "
                           f"{spread_deg:.1f} deg (max {PALM_MAX_SPREAD_DEG:.0f}): the hand "
                           f"moved while the neutral was built")
        if not reasons:
            return ""
        if outliers_spread_deg is not None and outliers_spread_deg > PALM_MAX_SPREAD_DEG:
            # 只作证据：散出去的那几帧离基准有多远（``_palm_cluster`` 已经把它们踢出簇，
            # 但它们正是「为什么会判死」的现场解释 —— 掌面翻号 / 半遮挡）。
            reasons.append(f"the rest disagree by up to {outliers_spread_deg:.1f} deg "
                           f"(flipped normals / half-hidden frames)")
        return f"{side} palm basis rejected: " + "; ".join(reasons)

    def segment_asymmetry_note(self) -> str:
        """左右量出来的段长差得太多时的原因文本；空串 = 正常 / 数据不够。

        与 ``segment_scale_note`` 分开的原因：那一条是**逐侧**的（一侧坏了只废掉那一侧），
        这一条必须**两侧都在**才有意义（比的是你的左臂和你的右臂）。判据同样只涉及你的
        段长、与机器人无关，所以一台机器人的中性记录就够判所有机器人。

        实测三场（2026-11-17）：右前臂一直是 0.23 m，左前臂 0.23 / 0.17 / 0.060 m
        —— 0.23 那场比值 1.00 不响，0.17 那场 1.35 响，0.060 那场已被带宽护栏拦下。
        见 ``SEGMENT_ASYMMETRY_MAX``。
        """
        lengths = {side: self._human_segments(side) for side in self.sides}
        if len(lengths) < 2 or any(value is None for value in lengths.values()):
            return ""
        # 只有 left/right 都在 ``sides`` 里时才比得了（单侧跑时这一条无意义）。
        if "left" not in lengths or "right" not in lengths:
            return ""
        left, right = lengths["left"], lengths["right"]
        worst, worst_name, worst_index = 1.0, "", 1
        for name, index in (("upper arm", 0), ("forearm", 1)):
            values = (left[index], right[index])
            if min(values) <= 1e-6:
                return ""
            ratio = max(values) / min(values)
            if ratio > worst:
                worst, worst_name, worst_index = ratio, name, index
        if worst <= SEGMENT_ASYMMETRY_MAX:
            return ""
        # 「哪一侧可疑」要按**最不对称的那一段**判，不能写死前臂（上臂也可能不对称）。
        shorter = "left" if left[worst_index] < right[worst_index] else "right"
        return (f"left/right calibrated segments differ {worst:.2f}x on {worst_name} "
                f"(upper {left[0]:.3f}/{right[0]:.3f} m, forearm {left[1]:.3f}/"
                f"{right[1]:.3f} m); the shorter one ({shorter}) is the suspect "
                f"(depth flattening only shortens), recalibrate with both arms in frame")

    def _segment_scales(self, profile, side):
        """``(k_上臂, k_前臂)`` = 机器人连杆长 / 你的连杆长；缺数据或标定不可信就 ``None``。"""
        robot = self.robot_links.get(profile.robot_type, {}).get(side)
        lengths = self._human_segments(side)
        if not robot or lengths is None:
            return None
        human_upper, human_fore = lengths
        if human_upper <= 1e-6 or human_fore <= 1e-6:
            return None
        # 坏标定不许进链路：宁可退回整段 gain，也不要把前臂位移放大 3.8 倍。
        if self.segment_scale_note(side):
            return None
        return (float(robot["upper"]) / human_upper, float(robot["fore"]) / human_fore)

    def _segment_delta(self, profile, side, state):
        """论文 ``scale_joints`` 的位移形式（操作者系，米）：``k_u*Δ(上臂) + k_f*Δ(前臂)``。

        为什么它比「腕 - 肩」强：你握拳贴胸、只把肘摆开时，「腕 - 肩」几乎不动（所以历史口径
        零反应），但上臂段与「前臂+手」段各自都在转 —— 两段分开算就抓得到。缺连杆长 / 缺中性
        记录里那两段向量时返回 ``None``，调用方退回整段 gain。

        注意它**不是万能的**：这个和等于 ``(R-I)(L_u*u + L_f*v)``，所以当机器人的两段比例
        与你的接近时，绕「肩-腕」轴的纯摆动仍会互相抵消（实测 g1 2.0 mm、gr1_t2 8-10 mm，
        而比例差得多的 h1_2 有 68-74 mm）。要那种动作有反应就得靠肘位置任务
        （``elbow_task``），这也是为什么两个开关是一对（见 docs/multi-robot-arm.md 8.10）。
        """
        terms = self._segment_terms(profile, side, state)
        return None if terms is None else terms[0] + terms[1]

    def _segment_terms(self, profile, side, state):
        """两段各自的那一项（操作者系，米）：``(上臂段, 前臂段)``；缺数据返回 ``None``。

        第 1 项就是肘的位移：``L_上臂 * (单位方向_now - 单位方向_中性)``。腕目标 = 中性腕 +
        M(两项之和)，肘目标 = 中性肘 + M(第 1 项) —— **两个目标来自同一个和里的同一项**，
        所以同时可达，IK2 才解得动（实测对不上的目标会让腕残差涨到 107 mm）。
        """
        scales = self._segment_scales(profile, side)
        if scales is None:
            return None
        upper, fore = state.get("upper"), state.get("fore")
        entry = self.neutral[side]
        if upper is None or fore is None:
            return None
        k_upper, k_fore = scales
        return (k_upper * (np.asarray(upper, dtype=np.float64) - entry["upper"]),
                k_fore * (np.asarray(fore, dtype=np.float64) - entry["fore"]))

    # ------------------------------------------------------------------ #
    # 方向口径（posture）：搬方向不搬位移，长度用机器人自己的
    # ------------------------------------------------------------------ #
    def _posture_terms(self, profile, side, state):
        """``{你的上臂方向, 你的前臂方向, 机器人上臂方向, 机器人前臂方向, 两根连杆}``。

        缺任何一项返回 ``None``（缺什么由 ``_posture_unavailable`` 说明）。你的方向先过
        ``axis_map``（与位移口径同一条轴向约定），再与机器人自己的中性方向按 ``posture_gain``
        做夹角插值 —— ``gain=1`` 就是完全用你的方向。
        """
        anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
        if anchor is None or "shoulder" not in anchor:
            return None
        links = self.robot_links.get(profile.robot_type, {}).get(side)
        if not links:
            return None
        entry = self.neutral[side]
        u_op = _unit_direction(state.get("upper"))
        v_op = _unit_direction(state.get("fore"))
        u_op_rest = _unit_direction(entry.get("upper"))
        v_op_rest = _unit_direction(entry.get("fore"))
        if u_op is None or v_op is None or u_op_rest is None or v_op_rest is None:
            return None
        # rest_u 是**刚体**上臂方向（肩锚点 -> 肘）：gain<1 时它就是插值的基准方向，
        # 必须与 operator rest 对应同一个姿势（锚点口径见 _posture_target 的说明）。
        rest_u = _unit_direction(anchor["elbow"] - anchor["shoulder"])
        rest_v = _unit_direction(anchor["wrist"] - anchor["elbow"])
        u_base = _unit_direction(self.axis_map.apply(u_op))
        v_base = _unit_direction(self.axis_map.apply(v_op))
        if (rest_u is None or rest_v is None
                or u_base is None or v_base is None):
            return None
        upper = float(links["upper"])
        fore = float(links["fore"])
        if not (np.isfinite(upper) and np.isfinite(fore)) or upper <= 0.0 or fore <= 0.0:
            return None
        return {
            "shoulder": np.asarray(anchor["shoulder"], dtype=np.float64),
            # 机器人这一侧最终要指的方向：gain=1 就是你的方向本身。
            "u": _slerp_direction(rest_u, u_base, self.posture_gain),
            "v": _slerp_direction(rest_v, v_base, self.posture_gain),
            "u_rest": rest_u, "v_rest": rest_v,
            "u_op": u_op, "v_op": v_op,
            "u_op_rest": u_op_rest, "v_op_rest": v_op_rest,
            "upper": upper, "fore": fore,
        }

    def _posture_unavailable(self, profile, side) -> str:
        """方向口径这一侧缺什么 —— 写进那一帧的 ``info``，别让下一个人猜。"""
        anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
        if anchor is None:
            return "no anchors measured for this robot/side"
        if "shoulder" not in anchor:
            return "no shoulder anchor measured for this robot/side"
        if not self.robot_links.get(profile.robot_type, {}).get(side):
            return "no robot link lengths measured for this robot/side"
        return "no usable operator arm directions in this frame"

    def _posture_target(self, profile, side, state):
        """方向口径的一对目标：``(肘点, 腕点, terms)``（基座系）；缺数据返回 ``None``。

        肩锚点是机器人自己的中性**上臂肩端**（= 肘的父连杆原点，见 ``set_anchors``），两根骨头
        是它自己的实测长度，所以：

            elbow_target = shoulder + L_u * u          # 你的上臂方向
            wrist_target = elbow_target + L_f * v      # 你的前臂方向

        两个目标同时可达**有一个前提**：``L_u`` 必须是「肩锚点 -> 肘」在**任意姿势下都不变**的
        那个长度，``L_f`` 同理。若锚点取的是名字匹配的肩关节连杆，它的 roll 关节一转，肘到锚点
        的真实距离就变 48.5-101.7 mm（gr1_t2 左实测 +101.6 mm），此时「肩 + L_u*u」这个点根本
        不在手臂可达的球面上，7 自由度链只能把误差摊到肘和腕上 —— 现场看到的就是
        「H1-2 / GR1-T2 前臂弯曲方向相反」，而残差（这场肘 222 mm / 腕 98 mm）全在求解器里，
        物理没错。改用刚体肩端后残差降到毫米级，见 ``devtools/probe_arm_segment_rigidity.py``
        与 docs/multi-robot-arm.md。

        ``segments`` 那两个目标各自被逐轴夹断过（g1 每轴 40 mm），是幅度小与抽抽的来源；
        这里不夹断，因为目标本来就落在机器人自己的可达球面上。
        """
        terms = self._posture_terms(profile, side, state)
        if terms is None:
            return None
        elbow_point = terms["shoulder"] + terms["upper"] * terms["u"]
        wrist_point = elbow_point + terms["fore"] * terms["v"]
        return elbow_point, wrist_point, terms

    @property
    def calibrating(self) -> bool:
        """还有哪一侧没攒够中性帧。"""
        return any(not self.neutral[side]["complete"] for side in self.sides)

    def calibration_progress(self) -> dict:
        """``{side: (计入标定的帧数, 需要的帧数)}``（放过等手点的帧不算，见 ``PALM_WAIT_FRAMES``）。"""
        return {side: (self._counted[side], self.calibration_frames)
                for side in self.sides}

    def status(self) -> str:
        """HUD 用的一行中性标定状态（``track`` 模式多一个前缀，表示已经在跟手）。

        等手点的那几个侧面挂 ``+w``：``cal=track left=20/30 w=12 right=0/30 w=0`` ——
        现场一眼能分清「我没站好」与「它没在等」。
        """
        if not self.calibrating:
            return "cal=ok"
        parts = " ".join(f"{side}={self._counted[side]}/{self.calibration_frames}"
                         + (f" w={self._palm_wait[side]}" if self._palm_wait[side] else "")
                         for side in self.sides)
        if self.calibrate_mode == "track":
            return f"cal=track {parts}"
        return f"cal={parts}"


    # ------------------------------------------------------------------ #
    # 每帧入口
    # ------------------------------------------------------------------ #
    def update(self, arms_world=None, hands=None, frame_dt=None) -> dict:
        """一帧摄像头数据 -> ``{robot_type: {side: ArmTarget}}``。

        ``arms_world`` 是 Pose 的 (肩, 肘, 腕) 米制数据（``UpperBodyObservation`` 或它的
        ``world_arms`` 字典），``hands`` 是同一帧的手部关键点（``RawHandFrame`` 或
        ``{side: normalized, world_side: world}`` 字典）。两者都可以是 ``None``：某一侧
        没数据时该侧保持上一帧目标，不会拿瞎猜的位置去解 IK。

        ``frame_dt`` = 上一帧到这一帧的秒数（拿不到给 ``None`` -> 名义 1/30 s）：关键点跳变
        护栏的角速度上限按它换算（见 ``_guard_direction``）。现场摄像头 8-12 fps 时必须给真值，
        否则名义 33 ms 会把 600 度/秒算成 20 度/帧，把真实的快速摆动也拦掉。
        """
        if frame_dt is not None:
            try:
                value = float(frame_dt)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                self.frame_dt = value
        # 掌面基的保持窗口按这一帧的真实时长走（见 ``PALM_HOLD_S``）：现场 8-12 fps，
        # 写死「几帧」会把窗口绑在帧率上。
        for side, held in self._palm_hold.items():
            if held is not None:
                self._palm_hold[side] = (held[0], held[1] + self.frame_dt)
        normalized, world = _hand_frames(hands)
        raw_arms = _side_map(arms_world)
        # 「你的两条手臂是否镜像」——只记数字，不参与控制（见 SYMMETRY_SAME_SIDE_DEG）。
        self._note_symmetry(raw_arms)
        arms = dict(raw_arms)
        if self.jitter_filter:
            for side in self.sides:
                filtered = self._arm_filter.update(
                    side, raw_arms.get(side), self.frame_dt)
                if filtered is not None:
                    arms[side] = filtered
        inputs = {}
        for side in self.sides:
            inputs[side] = self._frame_inputs(side, arms.get(side), normalized.get(side),
                                              world.get(side))
            frame = anatomical_hand_frame(
                inputs[side].get("world_hand"), inputs[side].get("arm"), side)
            target_frame = (None if frame is None
                            else self.axis_map.matrix() @ frame @ self._hand_roll_matrix)
            if target_frame is not None and self.jitter_filter:
                from teleoperation.retargeting.arm.wrist_targets import (
                    matrix_from_quaternion, rotation_matrix_to_quaternion)

                target_quaternion = rotation_matrix_to_quaternion(target_frame)
                filtered_quaternion = self._hand_filter.update(
                    side, target_quaternion, self.frame_dt)
                if filtered_quaternion is not None:
                    target_frame = matrix_from_quaternion(filtered_quaternion)
            self._current_hand_frames[side] = target_frame
            self._collect_calibration(side, inputs[side])
        self.frames += 1
        self.counts["frames"] = self.frames
        states = {side: self._operator_state(side, inputs[side]) for side in self.sides}
        for side in self.sides:
            if states[side].get("orientation") is not None:
                self.counts["oriented"][side] += 1
        targets = {}
        for robot_type, profile in self.profiles.items():
            targets[robot_type] = {
                side: self._target_from_state(profile, side, states[side])
                for side in self.sides}
        return targets

    def _frame_inputs(self, side, arm, hand, world_hand) -> dict:
        """一帧一侧的原始输入：可用的 pose 手臂点 / 手部点，以及不可用时的原因。"""
        reason = ""
        ok = _is_points(arm, (3, 3))
        if ok:
            reach = arm_reach(arm)
            if reach < MIN_OPERATOR_REACH_M:
                ok = False
                reason = (f"pose arm reach {reach * 1000.0:.0f} mm is below "
                          f"{MIN_OPERATOR_REACH_M * 1000.0:.0f} mm; frame rejected")
        if not ok and not reason:
            reason = "no usable pose arm landmarks for this side"
        return {
            "side": side,
            "arm": np.asarray(arm, dtype=np.float64) if ok else None,
            "hand": np.asarray(hand, dtype=np.float64) if _is_points(hand, (21, 3)) else None,
            "world_hand": (np.asarray(world_hand, dtype=np.float64)
                           if _is_points(world_hand, (21, 3)) else None),
            "reason": reason,
        }

    @staticmethod
    def _mirror_across(vector, axis) -> np.ndarray:
        """``vector`` 相对「法向 = ``axis`` 的平面」的镜像（只反掉沿 ``axis`` 的那一维）。"""
        values = np.asarray(vector, dtype=np.float64)
        normal = np.asarray(axis, dtype=np.float64)
        return values - 2.0 * float(np.dot(values, normal)) * normal

    def _note_symmetry(self, arms) -> None:
        """记一帧「两条上臂是不是镜像」（度，见 ``SYMMETRY_SAME_SIDE_DEG``）。

        ``lateral`` 用**你这一帧的两肩连线**现算，所以这个量与 ``AxisMap``、与相机外参都无关：
        它只回答「输入本身是不是镜像手势」。两侧缺任一侧就**不记**（不拿单侧数据凑数）。
        只攒数字，不参与任何控制量。
        """
        left, right = arms.get("left"), arms.get("right")
        if not (_is_points(left, (3, 3)) and _is_points(right, (3, 3))):
            return
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        lateral = _unit_direction(left[0] - right[0])
        u_left = _unit_direction(left[1] - left[0])
        u_right = _unit_direction(right[1] - right[0])
        if lateral is None or u_left is None or u_right is None:
            return
        mirrored = _unit_direction(self._mirror_across(u_right, lateral))
        if mirrored is None:
            return
        cosine = float(np.dot(u_left, mirrored))
        deviation = float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))
        if len(self._symmetry) < SYMMETRY_MAX_SAMPLES:
            self._symmetry.append(deviation)
        else:
            self._symmetry_extra += 1

    def symmetry_summary(self) -> dict:
        """「你的两条手臂是否镜像」的报告段（度）。见 ``SYMMETRY_SAME_SIDE_DEG``。

        ``median_deg`` 小 -> 输入是镜像手势，桥会如实给出镜像的两条手臂；大 -> 两侧朝同一边
        摆（或一侧手点被挡/跑偏），桥只能照抄。两种情况的处理完全不同，所以必须能分开看。
        """
        samples = list(self._symmetry)
        if not samples:
            return {"frames": 0, "frames_not_sampled": self._symmetry_extra,
                    "median_deg": None, "p90_deg": None, "max_deg": None,
                    "same_side_frames": 0, "threshold_deg": SYMMETRY_SAME_SIDE_DEG,
                    "note": "no frame had both arms usable"}
        values = np.asarray(samples, dtype=np.float64)
        median = float(np.median(values))
        note = ("both arms were mirror images of each other (the bridge copies that as is)"
                if median <= SYMMETRY_SAME_SIDE_DEG else
                "both arms moved the same way instead of as mirror images: the bridge can only "
                "copy that (a hidden/misdetected side or a gesture that is not mirrored)")
        return {"frames": len(samples), "frames_not_sampled": self._symmetry_extra,
                "median_deg": round(median, 2),
                "p90_deg": round(float(np.percentile(values, 90.0)), 2),
                "max_deg": round(float(values.max()), 2),
                "same_side_frames": int(np.count_nonzero(
                    values > SYMMETRY_SAME_SIDE_DEG)),
                "threshold_deg": SYMMETRY_SAME_SIDE_DEG, "note": note}

    def _collect_calibration(self, side, data) -> None:
        """中性姿势采样：只取有效帧，攒够 ``calibration_frames`` 帧后取中位数。

        除了「腕 - 肩」与臂长，还记下每帧的腕坐标系（``wrist_orientation_frame``）：
        标定完成后取均值（SVD 投影回真旋转）作为**姿态基准**。直臂帧是 ``None``，跳过。
        2026-10-13 起还记上臂段 (肘-肩) 与「前臂+手」段 (腕-肘) 的中性向量：两段式映射
        （``map_mode="segments"``）拿它们当基准 —— 同一份中性记录多吃四个数，不用多站一次。
        2026-11-16 起再多记一帧**掌面基**（``palm_orientation_frame``，第三轴 = 掌面法向）：
        三点基量不到旋前/旋后，掌面基量得到。两份基准都留，是因为它们**不能混用** ——
        相减的两帧必须同一套构造（见 ``_relative_to_neutral``）。

        这一帧的中位数写进 ``neutral[side]`` 的动作在 ``_refresh_neutral``：``hold`` 模式下
        只有攒够帧那一次才写（历史行为），``track`` 模式下每来一帧都刷新（临时中性）。
        """
        entry = self.neutral[side]
        if entry["complete"] or data["arm"] is None:
            return
        points = data["arm"]
        palm_frame = palm_orientation_frame(data.get("world_hand"), points, side)
        if palm_frame is not None:
            self._palm_samples[side] += 1
            self._palm_seen[side] = True
        # 等手点：这一侧还没见过掌面基时先不计数（样本照旧追加，跟手不受影响）。
        # 见 ``PALM_WAIT_FRAMES``：手臂第 1 帧就有、手部 21 点要 5-7 s，不等就必然
        # 「窗口在手伸进来之前烧完」。
        if self._waiting_for_palm(side, palm_frame):
            self._palm_wait[side] += 1
        else:
            self._counted[side] += 1
        self._samples[side].append((points[2] - points[0], arm_reach(points),
                                    wrist_orientation_frame(points),
                                    points[1] - points[0], points[2] - points[1],
                                    # 掌面基：没手点 / 掌面退化时是 None，
                                    # 均值那边照样能出（全是 None 就给 None）。
                                    palm_frame))
        self._refresh_neutral(side)

    def _waiting_for_palm(self, side, palm_frame) -> bool:
        """这一帧要不要**放过**（只追加样本、不计入标定），就为了等一次手点。

        只在 ``track`` 模式下、且这一侧**从没见过**掌面基、且还没等满 ``palm_wait_frames``
        时成立。``hold`` 模式永远 False（历史行为逐字不变）。
        """
        if palm_frame is not None:
            return False
        if self.calibrate_mode != "track":
            return False
        if self._palm_seen[side]:
            return False
        return self._palm_wait[side] < self.palm_wait_frames

    def _samples_enough(self, side) -> bool:
        """这一侧标定够不够：按**计入的帧数**算（``_counted``，见 ``PALM_WAIT_FRAMES``）。"""
        return self._counted[side] >= self.calibration_frames

    def _refresh_neutral(self, side) -> None:
        """把已有样本的中位数写进 ``neutral[side]``；攒够 ``calibration_frames`` 才算完成。

        ``hold``：样本不够直接返回 —— ``neutral[side]`` 一直是空的，那一侧就停在
        profile 的中性工作点上（历史行为，改前逐字一致）。
        ``track``：样本有 1 帧就发布一个**临时**中性（``complete`` 仍为 False），攒够帧后
        自动收敛成与 ``hold`` 同一个终值（同一批样本、同一个中位数）。

        「攒够」按 ``_counted`` 算而不是 ``len(samples)``：为了等手点而放过的那一帧也是
        样本（临时中性要用），但不能算作标定进度（见 ``PALM_WAIT_FRAMES``）。
        """
        samples = self._samples[side]
        if not samples:
            return
        enough = self._samples_enough(side)
        if not enough and self.calibrate_mode != "track":
            return
        entry = self.neutral[side]
        offsets = np.stack([sample[0] for sample in samples], axis=0)
        reaches = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        uppers = np.stack([sample[3] for sample in samples], axis=0)
        fores = np.stack([sample[4] for sample in samples], axis=0)
        entry["offset"] = np.median(offsets, axis=0)
        entry["reach"] = float(np.median(reaches))
        entry["orientation"] = _average_rotation([sample[2] for sample in samples])
        # 掌面基取**最大一致簇**的均值（见 ``_palm_cluster``）：正反两拨一起平均只会
        # 得到一个既不正也不反的垃圾基准。全是 None（这一侧一直没手点 / 老记录）就给 None，
        # 姿态那边自动退回三点基 —— 绝不把「没有」当成单位阵。
        palms = [sample[5] for sample in samples if sample[5] is not None]
        basis, spread, outliers = _palm_cluster(palms)
        # 簇外散度：散出去的那几帧相对基准有多远（0 = 全部帧都进了一个簇）。
        overall = _rotation_spread_deg([palm for palm in palms], basis)
        # 样本不够 / 彼此不一致 -> 这份基准不可用（见 PALM_MIN_SAMPLE_FRACTION）：
        # 现场「右手卡住不动、手腕怎么拧都不对」就是拿 12/30 帧凑出来的基准驱动腕链。
        # ``samples`` 只数簇内的帧：散出去的几帧本来就不该算「这一侧采到了」。
        palm_note = self._palm_basis_note(side, len(palms) - outliers, spread, overall)
        entry["orientation_palm"] = None if (palm_note or basis is None) else basis
        # 局部基准必须逐帧做 ``F_i^T H_i`` 再平均，不能拿平均后的 F 和 H 现乘：
        # 后者在标定期间手臂稍微移动时会凭空产生腕部偏置。簇外掌面帧不参与。
        local_samples = [
            sample[2].T @ sample[5]
            for sample in samples
            if sample[2] is not None and sample[5] is not None
            and (basis is None
                 or _rotation_angle_deg(sample[5], basis) <= PALM_MAX_SPREAD_DEG)]
        entry["orientation_palm_local"] = (
            None if palm_note or basis is None
            else _average_rotation(local_samples))
        #: 这份基准是「标定那一次采的」还是「运行中补采的迟到基准」（见 ``PALM_LATE_FRAMES``）。
        #: 空串 = 没有可用基准（``orientation_palm`` 为 None）。
        entry["palm_reference_source"] = ("" if entry["orientation_palm"] is None else "neutral")
        #: 迟到基准是在第几帧采的（``palm_reference_source="late"`` 时才有意义）。
        entry["palm_reference_frame"] = None
        #: 这几帧掌面基相对基准的最大夹角（度）：大于 ``PALM_MAX_SPREAD_DEG`` = 标定时手在动 /
        #: 半遮挡，均值不代表任何一帧。现场只能从这一列看出「基是凑出来的」。
        entry["palm_spread_deg"] = round(overall, 2)
        #: 掌面基为什么不可用（空串 = 能用 / 这一侧没有手点）。
        entry["palm_note"] = palm_note
        #: 有掌面基却被判「不在多数簇里」的帧数（掌面翻号 / 半遮挡，见 ``_palm_cluster``）。
        entry["palm_outliers"] = int(outliers)
        entry["upper"] = np.median(uppers, axis=0)
        entry["fore"] = np.median(fores, axis=0)
        entry["samples"] = int(len(samples))
        #: 这批样本里有几帧真的带掌面基：0 就说明这一整场量不到旋前（现场只看 ``ori_source``
        #: 全是 ``three_point`` 是看不出来的，要这一列才知道「标定时机不对」）。
        entry["palm_samples"] = int(self._palm_samples[side])
        entry["complete"] = bool(enough)

    def _acquire_palm_reference(self, side, data) -> None:
        """中性里没有掌面基时，运行中补采一次「迟到基准」（见 ``PALM_LATE_FRAMES``）。

        判据是**连续** ``PALM_LATE_FRAMES`` 帧、每帧与前几帧的夹角都不超过
        ``PALM_LATE_SPREAD_DEG``：断帧 / 一致性被打破就整窗清空重来。这条不是为了
        「把噪声平均掉」，而是为了确认「人此刻是站着不动的」—— 零点取在这一刻，只有
        静止时才谈得上零点。

        采到之后写进 ``neutral[side]``（与 ``_refresh_neutral`` 同一组键），于是
        ``_raw_orientation`` 从下一帧起就走 ``palm_normal`` 那条路，旋前/旋后**真的**
        进得去。`palm_reference_source="late"` / `palm_reference_frame` 如实记下它不是
        标定那一帧，报告与 CLI 都能分辨。

        为什么不在标定期间采：``_refresh_neutral`` 每帧都重写 ``orientation_palm``，
        采到的值下一帧就被盖掉（所以 ``self.calibrating`` 时直接返回）。
        """
        if self.orientation == "off":
            return
        entry = self.neutral.get(side)
        if entry is None or self.calibrating or not self._has_neutral(side):
            return
        buffer = self._palm_late[side]
        if entry.get("orientation_palm") is not None:
            # 已经有基准（标定那次采的，或本侧已经补采过）：这个窗口不再需要。
            buffer.clear()
            return
        frame = palm_orientation_frame(data.get("world_hand"), data.get("arm"), side)
        if frame is None:
            buffer.clear()
            return
        if buffer and _rotation_spread_deg(buffer + [frame], frame) > PALM_LATE_SPREAD_DEG:
            # 这一帧和窗口里已经攒下的不一致：窗口作废、从这一帧重新开始数。
            buffer.clear()
        buffer.append(frame)
        if len(buffer) < PALM_LATE_FRAMES:
            return
        basis = _average_rotation(buffer)
        if basis is None:
            buffer.clear()
            return
        entry["orientation_palm"] = basis
        entry["palm_reference_source"] = "late"
        entry["palm_reference_frame"] = int(self.frames)
        entry["palm_spread_deg"] = round(_rotation_spread_deg(buffer, basis), 2)
        # 这个零点取在**此刻**，所以「此刻你的前臂方向」与中性那一刻差多少，腕就带着
        # 多少度的固定偏移（相对映射本来就是这样：零点挪了，读数就挪了）。必须报出来 ——
        # 现场看到「腕立起来时是歪的」要能对上这个数，而不是以为桥又坏了。
        arm = data.get("arm")
        wrist_now = wrist_orientation_frame(arm)
        if wrist_now is not None:
            # 迟到基的``局部零点``也取在补采这一刻；不能继续拿标定时的前臂基与
            # 现在的手掌基相减，否则前臂已经转过的 80-90 度会原样变成腕部偏置。
            entry["orientation_palm_local"] = wrist_now.T @ basis
        fore_now = _unit_direction(None if arm is None else np.asarray(arm)[2] - np.asarray(arm)[1])
        fore_rest = _unit_direction(entry.get("fore"))
        entry["palm_reference_pose_deg"] = (None if fore_now is None or fore_rest is None
                                            else round(_angle_between_deg(fore_now, fore_rest), 1))
        # 判死的那条理由已经不成立了：现在有基准，且它是刚刚逐帧核对过的。
        entry["palm_note"] = ""
        self.counts["palm_late"][side] += 1
        buffer.clear()

    def palm_reference_note(self, side) -> str:
        """迟到基准采到了就返回一行说明（空串 = 没有迟到基准）。

        CLI 用它打一条 ``[INFO]``：不解释的话，现场看到「腕突然开始跟手拧了」会
        以为是自己按错了什么键 —— 实际是这一侧第 N 帧才第一次看见稳定的手。
        """
        entry = self.neutral.get(side) or {}
        if entry.get("palm_reference_source") != "late":
            return ""
        offset = entry.get("palm_reference_pose_deg")
        suffix = ("" if offset is None else
                  f", {offset:.0f} deg away from the calibration pose"
                  f" (the wrist reads zero from that pose)")
        return (f"{side} palm reference acquired at frame {entry.get('palm_reference_frame')} "
                f"(the neutral had none){suffix}: pronation now tracks twists from this pose")

    def _guard_direction(self, side, upper, fore):
        """单帧方向变化超过人手臂的物理上限 -> 判为关键点跳变，保持上一帧。

        为什么（录音实测，见 ``MAX_DIRECTION_RATE_DEG_S``）：操作者每帧方向变化中位 4.7-6.9 度，
        但**最大 153 度** —— MediaPipe 的跳变。方向口径 1:1 搬方向，跳变就直接变成「一帧里把肘
        转 153 度」的请求，IK 只好翻支路或抽一下（现场第二个症状）。这里是**保持**而不是平滑：
        真手速 470-690 度/秒（中位）照样通过，只有比人快的才被拦住，且会记进这一帧的 info。

        返回 ``(upper, fore, 拐角 度)``：拐角 0 = 这一帧通过；非 0 = 被拦下了多少度。
        """
        upper = np.asarray(upper, dtype=np.float64)
        fore = np.asarray(fore, dtype=np.float64)
        if self.max_direction_rate_deg_s <= 0.0:
            return upper, fore, 0.0
        previous = self._last_direction.get(side)
        if previous is None:
            self._last_direction[side] = (upper, fore)
            return upper, fore, 0.0
        step = 0.0
        for old, new in zip(previous, (upper, fore)):
            old_unit = _unit_direction(old)
            new_unit = _unit_direction(new)
            if old_unit is None or new_unit is None:
                continue
            step = max(step, _angle_between_deg(old_unit, new_unit))
        limit = self.max_direction_rate_deg_s * self.frame_dt
        if step > limit:
            streak = self._glitch_streak.get(side, 0) + 1
            if streak <= GLITCH_HOLD_FRAMES:
                self._glitch_streak[side] = streak
                self.counts["direction_glitch"][side] += 1
                return previous[0], previous[1], step
            # 连续拦满 -> 放过去（见 GLITCH_HOLD_FRAMES）
            self.counts["direction_glitch_kept"][side] += 1
        self._glitch_streak[side] = 0
        self._last_direction[side] = (upper, fore)
        return upper, fore, 0.0

    def _operator_state(self, side, data) -> dict:
        """一侧这一帧的「操作者层」结论：用哪条来源、位移多少、姿态、还是保持上一帧。"""
        entry = self.neutral[side]
        if data["arm"] is None:
            # 断流（没检到 / 帧被拒）之后再回来：方向基准丢掉，否则长间隔后的正常移动会被
            # 跳变护栏当成跳变（实测左臂 30% 的帧被判跳变，一部分就是这么来的）。
            self._last_direction.pop(side, None)
            self._glitch_streak.pop(side, None)
        # 中性里没有掌面基时，在这里补采一次「迟到基准」（见 ``PALM_LATE_FRAMES``）：
        # 放在姿态计算之前，采到的那一帧起 ``_raw_orientation`` 就走掌面基。
        self._acquire_palm_reference(side, data)
        if self._has_neutral(side) and data["arm"] is not None:
            points = data["arm"]
            offset = points[2] - points[0]
            orientation, orientation_source = self._orientation_frame(data)
            state = {
                "source": SOURCE_POSE,
                "valid": True,
                "held": False,
                "reason": "",
                "offset": offset,
                "delta_op": offset - entry["offset"],
                # 两段式映射用的两段向量（操作者系，米）。历史口径不用它们，但记着不贵。
                # 过一遍关键点跳变护栏：单帧转角超过人手臂上限就保持上一帧（见 _guard_direction）。
                "upper": points[1] - points[0],
                "fore": points[2] - points[1],
                "reach": arm_reach(points),
                "elbow_bend_deg": elbow_bend_degrees(points[0], points[1], points[2]),
                "orientation": orientation,
                "orientation_source": orientation_source,
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
            state["upper"], state["fore"], state["direction_glitch_deg"] = \
                self._guard_direction(side, state["upper"], state["fore"])
            self._last_state[side] = state
            self.counts["valid"][side] += 1
            return state
        if data["arm"] is None and self.fallback_hand and data["hand"] is not None:
            orientation, orientation_source = self._orientation_frame(data)
            state = {
                "source": SOURCE_HAND,
                "valid": True,
                "held": False,
                "reason": "",
                "offset": None,
                "delta_op": None,
                "upper": None,
                "fore": None,
                "reach": float("nan"),
                "elbow_bend_deg": float("nan"),
                "orientation": orientation,
                "orientation_source": orientation_source,
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
            self._last_state[side] = state
            self.counts["valid"][side] += 1
            self.counts["fallback"][side] += 1
            return state
        previous = self._last_state[side]
        if previous is None:
            self.counts["held"][side] += 1
            self._count_hold(side, data)
            return {
                "source": SOURCE_NONE,
                "valid": False,
                "held": True,
                "reason": self._hold_reason(side, data),
                "offset": None,
                "delta_op": None,
                "upper": None,
                "fore": None,
                "reach": float("nan"),
                "elbow_bend_deg": float("nan"),
                "orientation": None,
                "hand": data["hand"],
                "fallback_reason": data["reason"],
            }
        held = dict(previous)
        held.update({"valid": False, "held": True,
                     "reason": self._hold_reason(side, data)})
        self.counts["held"][side] += 1
        self._count_hold(side, data)
        return held

    def _has_neutral(self, side) -> bool:
        """这一侧现在有没有可用的中性基准。

        ``hold``：必须有攒够帧的完整中性（``complete``）—— 与改前逐字一致。
        ``track``：``_refresh_neutral`` 一发布临时中性（第 1 帧有效数据）就算有，
        所以「只要看见一次就开始跟手」。
        """
        entry = self.neutral[side]
        if entry["offset"] is None:
            return False
        return bool(entry["complete"]) or self.calibrate_mode == "track"

    def _count_hold(self, side, data) -> None:
        """把「这一帧为什么没跟手」记进 ``counts["held_reason"]``（``HOLD_REASONS``）。

        现场的「HUD 上 cal 一直不动、手臂一动不动」里，两种情形长得一模一样：
        没检到关键点（`no_pose_landmarks`）和这一帧被臂长判据拒了（`reach_below_min`，
        关键点抖动导致「肩-腕」< 150 mm）。分开计数才不用下一次再猜。
        """
        reason = data.get("reason") or ""
        if not self._has_neutral(side):
            kind = "calibrating"
        elif data.get("arm") is not None:
            kind = "other"
        elif "pose arm reach" in reason:
            kind = "reach_below_min"
        elif reason:
            kind = "no_pose_landmarks"
        else:
            kind = "other"
        self.counts["held_reason"][side][kind] += 1

    def _hold_reason(self, side, data) -> str:
        """保持上一帧时解释为什么（标定中 / 没检到 / 帧被拒）。"""
        if not self._has_neutral(side):
            prefix = "tracking" if self.calibrate_mode == "track" else "calibrating"
            return (f"{prefix} {len(self._samples[side])}/{self.calibration_frames}"
                    f" ({data['side']})")
        return data["reason"] or "no arm data in this frame"


    # ------------------------------------------------------------------ #
    # 操作者层 -> 机器人层
    # ------------------------------------------------------------------ #
    def _target_from_state(self, profile, side, state) -> ArmTarget:
        """把「操作者层」结论换算成某一台机器人基座系里的目标。

        ``position`` 是基座系的点；``orientation`` 是**相对中性**的四元数（同一份目标给
        所有机器人，各自乘自己的中性 EE 朝向 —— 见 ``PyBulletArmIK.solve`` 的
        ``orientation_frame``）。"""
        gain = self._gain(profile, side, state)
        gain_reach, gain_source = self._gain_reference(side, state)
        trusted, trust_reason = self._arm_length_trust(side)
        if not trusted:
            # 这一帧用的是 metric_scale（臂长归一化被护栏关掉），如实写进报告：
            # 否则 gain_source 会说 "neutral"，看报告的人会以为归一化生效了。
            gain_source = "metric_scale"
            if self.neutral[side]["complete"]:
                # 标定没攒够帧的那几帧不计数：那时目标是中性工作点，归一化本来就不参与。
                self.counts["gain_guarded"][side] += 1
        neutral = self._neutral(profile, side)
        limit = self._travel_limit(profile, side)
        info = {
            "source": state["source"],
            "gain": round(gain, 4),
            "gain_reach_m": _round_or_none(gain_reach, 4),
            "gain_source": gain_source,
            "gain_trusted": bool(trusted),
            "gain_guard_reason": "" if trusted else trust_reason,
            "travel_limit_m": limit,
            "travel_mode": self.travel_mode,
            "lateral_guard_m": self.lateral_guard_m,
            "neutral_m": [round(float(value), 5) for value in neutral],
            "operator_reach_m": _round_or_none(state.get("reach"), 4),
            "elbow_bend_deg": _round_or_none(state.get("elbow_bend_deg"), 1),
            # 这一帧的姿态用的是哪条基（palm_normal / three_point / None）。现场排查先看它：
            # three_point 说明掌面基退化了 —— 那时腕对旋前不敏感是"正常"的。
            "orientation_source": state.get("orientation_source"),
            "orientation_mode": self.orientation_mode,
            "hand_roll_deg": self.hand_roll_deg,
        }
        if state["source"] == SOURCE_HAND and state["hand"] is not None:
            return self._fallback_target(profile, side, state, info)
        if state["source"] == SOURCE_POSE and state["delta_op"] is not None:
            # 位移口径：chain = 历史口径（肩->腕整段，只动大臂时几乎不动）；segments =
            # 论文 scale_joints 的两段式（上臂段 + 前臂段各自按连杆长比缩放）。取不到连杆长 /
            # 老中性记录缺那两段向量时退回 chain，并把原因写进这一帧的 info。
            delta_op = state["delta_op"]
            info["map"] = self.map_mode
            if self.map_mode == "posture":
                # 方向口径：腕/肘目标都从机器人自己的肩锚点 + 自己的连杆长推出来，
                # 与「你移动了多少」无关（所以不夹行程、不需要 gain）。
                built = self._posture_target(profile, side, state)
                if built is None:
                    info["map"] = ("chain (posture unavailable: "
                                   + self._posture_unavailable(profile, side) + ")")
                else:
                    elbow_point, position, terms = built
                    self.counts["posture"][side] += 1
                    self.counts["elbow"][side] += 1
                    guarded = self._apply_lateral_guard(side, position)
                    if guarded:
                        self.counts["guarded"][side] += 1
                    # 报告里最该看的一对数：你的上臂相对中性转了多少度 / 机器人这一侧的上臂
                    # 方向相对它的中性转了多少度。gain=1 时两个数应当几乎相等 —— 这就是
                    # 「幅度不再被臂长和站位缩放」的可证伪证据（现场症状正好相反）。
                    info.update({
                        "posture_gain": self.posture_gain,
                        "operator_direction_glitch_deg": _round_or_none(
                            state.get("direction_glitch_deg"), 1),
                        "operator_upper_deg": round(_angle_between_deg(
                            terms["u_op_rest"], terms["u_op"]), 2),
                        "operator_fore_deg": round(_angle_between_deg(
                            terms["v_op_rest"], terms["v_op"]), 2),
                        "robot_upper_deg": round(_angle_between_deg(
                            terms["u_rest"], terms["u"]), 2),
                        "robot_fore_deg": round(_angle_between_deg(
                            terms["v_rest"], terms["v"]), 2),
                        "posture_elbow_bend_deg": round(elbow_bend_degrees(
                            terms["shoulder"], elbow_point, position), 1),
                        "posture_reach_m": round(float(np.linalg.norm(
                            position - terms["shoulder"])), 4),
                        "posture_upper_m": round(float(terms["upper"]), 4),
                        "posture_fore_m": round(float(terms["fore"]), 4),
                        # 目标由机器人自己的肩锚点 + 自己的连杆长推出来，天然落在它自己的
                        # 可达球面上，所以逐轴行程上限不参与（也不该参与：g1 的每轴 40 mm
                        # 正是把大臂幅度压没的那个夹子）。这一帧如实留下"没夹"的痕迹。
                        "travel_limit_m": None,
                        "travel_limit_applied": False,
                        "clamped": False,
                        "elbow_target": "posture",
                    })
                    return ArmTarget(
                        side=side, position=tuple(float(value) for value in position),
                        elbow=tuple(float(value) for value in elbow_point),
                        orientation=self._orientation_for(profile, state),
                        valid=bool(state["valid"]), held=bool(state["held"]),
                        reason=state["reason"], source=SOURCE_POSE, info=info)
            if self.map_mode == "segments":
                mapped = self._segment_delta(profile, side, state)
                if mapped is not None:
                    delta_op = mapped
                    self.counts["segments"][side] += 1
                    info["segment_scales"] = [round(float(value), 4)
                                              for value in self._segment_scales(profile, side)]
                else:
                    info["map"] = "chain (segments unavailable)"
            # 肘位置任务：肘目标 = 机器人自己的中性肘点 + M(上臂段位移)。它与腕目标出自
            # 同一个和里的同一项，所以两个目标同时可达（见 _segment_terms）。逐轴夹到与腕
            # 相同的行程上限：不夹的话一次方向反转可以要出 2*上臂长 的位移。
            elbow_point = None
            if self.elbow_task:
                terms = self._segment_terms(profile, side, state)
                anchor = self.robot_anchors.get(profile.robot_type, {}).get(side)
                if terms is None:
                    info["elbow_task"] = "off (no link lengths / old neutral record)"
                elif anchor is None:
                    info["elbow_task"] = "off (no elbow anchor measured for this robot)"
                else:
                    elbow_delta = np.clip(
                        np.asarray(self.axis_map.apply(terms[0]), dtype=np.float64)
                        * self.elbow_gain,
                        -limit, limit)
                    elbow_point = tuple(float(value) for value in (
                        np.asarray(anchor["elbow"], dtype=np.float64) + elbow_delta))
                    info["elbow_task"] = "on"
                    info["elbow_gain"] = self.elbow_gain
                    info["elbow_delta_base_m"] = [round(float(value), 4)
                                                  for value in elbow_delta]
                    self.counts["elbow"][side] += 1
            info["delta_map_m"] = [round(float(value), 4) for value in delta_op]
            delta = np.asarray(gain * self.axis_map.apply(delta_op),
                               dtype=np.float64)
            request = float(np.max(np.abs(delta))) if delta.size else 0.0
            clamped = request > limit
            if clamped:
                self.counts["clamped"][side] += 1
            if clamped and self.travel_mode == "scale" and request > 0.0:
                # 等比缩到上限：方向不变，最远的那一轴正好落在上限。代价是**没超限的轴也
                # 跟着变小**（2026-10-13 实测：现场 op_z 抖动常是最大轴，于是横向/竖向摆幅
                # 一起被吃掉，见 §8.8）；默认已改成 else 分支的逐轴剪断。
                delta = delta * (limit / request)
            else:
                # 逐轴剪断：只压超限的那一轴，其它轴保持原样。实测盒角 IK 到得了
                # （g1 三轴都剪断后残差 5.9 mm，判据 25 mm），所以"手再动也不跟"没有出现。
                delta = np.clip(delta, -limit, limit)
            position = np.asarray(neutral, dtype=np.float64) + delta
            guarded = self._apply_lateral_guard(side, position)
            if guarded:
                self.counts["guarded"][side] += 1
            info.update({
                "delta_op_m": [round(float(value), 4) for value in state["delta_op"]],
                "delta_base_m": [round(float(value), 4) for value in delta],
                "clamped": clamped,
                "request_m": round(request, 5),
                "guarded": guarded,
                "fallback_reason": state.get("fallback_reason", ""),
            })
            return ArmTarget(
                side=side, position=tuple(float(value) for value in position),
                elbow=elbow_point,
                orientation=self._orientation_for(profile, state),
                valid=bool(state["valid"]), held=bool(state["held"]),
                reason=state["reason"], source=SOURCE_POSE, info=info)
        # 还没有任何目标（标定中 / 一侧一直没数据）：停在 profile 的中性工作点，
        # 这样手臂不会因为没人管而垂下去（held=True 会把它写进报告）。
        info["fallback_reason"] = state.get("fallback_reason", "")
        return ArmTarget(
            side=side,
            position=tuple(float(value) for value in self._neutral(profile, side)),
            orientation=None, valid=False, held=True,
            reason=state["reason"] or "no target yet", source=SOURCE_NONE, info=info)

    def _fallback_target(self, profile, side, state, info) -> ArmTarget:
        """pose 手臂这帧不可信时的兜底：沿用既有「21 点 -> 手腕目标」映射。

        它输出**已经在机器人基座系**里的位置（``base_offset`` 直接取 profile 的
        ``metric_offset``），所以调用方要用 ``source="base"``，不能再叠一次 profile 缩放。
        """
        from teleoperation.retargeting.arm.wrist_targets import (
            WristMapping, wrist_pose_from_landmarks)

        mapping = WristMapping(scale=self.fallback_scale,
                               base_offset=self._neutral(profile, side))
        target = wrist_pose_from_landmarks(state["hand"], side, mapping)
        info["fallback_scale"] = self.fallback_scale
        info["fallback_reason"] = state.get("fallback_reason", "")
        if not target.valid:
            info["fallback_reason"] = target.reason or "hand landmarks unusable"
            return ArmTarget(
                side=side,
                position=tuple(float(value) for value in self._neutral(profile, side)),
                orientation=None, valid=False, held=True,
                reason=info["fallback_reason"], source=SOURCE_NONE, info=info)
        # 兜底路径没有肘，给不出「相对中性」的姿态（见 _orientation 的说明）。如实写进
        # info：不要把掌面基的绝对姿态塞进一个相对目标的接口，那会让腕停在一个随机角度。
        info["orientation_source"] = "none (hand fallback has no elbow)"
        # 护栏对两条路径一视同仁：兜底映射同样可以把手腕送到身体另一侧。
        position = np.asarray(target.position, dtype=np.float64)
        guarded = self._apply_lateral_guard(side, position)
        info["guarded"] = guarded
        if guarded:
            self.counts["guarded"][side] += 1
        return ArmTarget(
            side=side, position=tuple(float(value) for value in position),
            orientation=None, valid=True, held=False, reason="",
            source=SOURCE_HAND, info=info)

    def _orientation(self, data) -> tuple | None:
        """相对中性姿势的腕部姿态四元数（机器人无关；IK 那边再乘本机的中性 EE 朝向）。

        论文喂给 IK 的是「手腕 4x4 位姿」（SMPL-X 旋转链）；我们用手上的点造一个腕
        坐标系再减掉中性那一帧（模块 docstring 有公式）。顺序与日期：

        1. **掌面基**（2026-11-16 起优先）：``palm_orientation_frame`` —— x = 前臂方向、
           z = 掌面法向。只有这一条量得到绕前臂轴的自转（旋前/旋后）。前提是这一帧有
           21 点**度量 world 手**、且中性记录里有 ``orientation_palm``。
        2. **三点基**（历史路径）：``wrist_orientation_frame`` —— z 取手臂平面法向，
           量不到旋前。掌面给不出时用它，行为与 2026-11-16 之前逐字一致。

        两条都要求这一帧有 pose 手臂（兜底的手 21 点路径没有肘，给不出前臂方向），
        中性记录里有对应基准。都给不出就返回 ``None`` —— 这一帧退回位置模式，不猜姿态。

        返回的是 ``R_neutral^T @ R_now``；乘本机的中性 EE 朝向由
        ``PyBulletArmIK.solve(orientation_frame="neutral")`` 做，所以这里不需要知道是
        哪台机器人 —— 同一份目标三台各自乘自己的。
        """
        return self._orientation_frame(data)[0]

    def _orientation_frame(self, data):
        """``(四元数 or None, 用的是哪条基)``；基 ∈ ``{"palm_normal", "palm_hold", "three_point", None}``。

        返回值单独带一条「基」的名字，是因为现场排查必须先分清**两种失败**：
        「掌面基退化、退回三点基」（``three_point`` 但 ``ori_turn`` 对旋前不敏感）与
        「根本没算出来」（``None``，这一帧位置模式）。两者在报告里长得一样，性质不同。

        ``palm_hold`` = 这一帧的掌面基给不出（手点丢了 / 掌面退化），但上一次给出它不超过
        ``PALM_HOLD_S``，于是沿用那一次的相对旋转。理由见 ``PALM_HOLD_S``：两套基量的不是
        同一件事，逐帧来回切就是把腕目标在「拧了 90 度」和「没拧」之间来回拨。窗口过了才
        如实退回 ``three_point``。

        算出来的旋转还要过一遍 ``_clamp_orientation``（角速度上限），但**保持窗口里存的是
        原始测量值**：限速削过的那一份只给 IK，不能反过来污染下一次的「上一帧掌面旋转」。
        """
        rotation, source = self._raw_orientation(data)
        if rotation is None:
            return None, None
        clamped, _ = self._clamp_orientation(data.get("side"), rotation)
        return clamped, source

    def _raw_orientation(self, data):
        """``_orientation_frame`` 的测量部分（不含限速）：``(四元数 or None, 基名 or None)``。"""
        if self.orientation == "off":
            return None, None
        side = data.get("side")
        entry = self.neutral.get(side)
        if entry is None:
            return None, None
        arm = data.get("arm")
        if self.orientation_mode in ("hand_base", "wrist"):
            # 只取手相对前臂的局部旋转：前臂方向变化已经由 posture/位置 IK 完成，
            # 再塞给腕会让腕替肩肘补偿，甚至顶在限位上。
            relative = relative_palm_wrist_rotation(
                data.get("world_hand"), arm,
                entry.get("orientation_palm_local"), side)
            if relative is not None:
                from teleoperation.retargeting.arm.wrist_targets import (
                    rotation_matrix_to_quaternion)

                held = tuple(float(value) for value in
                             rotation_matrix_to_quaternion(relative))
                self._palm_hold[side] = (held, 0.0)
                return held, "palm_normal"
            held = self._palm_hold.get(side)
            if held is not None and held[1] <= self.palm_hold_s:
                self.counts["palm_held"][side] += 1
                return held[0], "palm_hold"
            # 三点基测不到手相对前臂的姿态（它把前臂方向也吃了进去），不伪造零腕角。
            return None, None
        # 掌面基优先：它比三点基多一维（旋前/旋后）。
        relative = relative_palm_rotation(data.get("world_hand"), arm,
                                          entry.get("orientation_palm"), side)
        if relative is not None:
            from teleoperation.retargeting.arm.wrist_targets import (
                rotation_matrix_to_quaternion)

            held = tuple(float(value) for value in
                         rotation_matrix_to_quaternion(relative))
            # 这一帧重新拿到掌面基：窗口重新计时。
            self._palm_hold[side] = (held, 0.0)
            return held, "palm_normal"
        held = self._palm_hold.get(side)
        if held is not None and held[1] <= self.palm_hold_s:
            self.counts["palm_held"][side] += 1
            return held[0], "palm_hold"
        # 掌面基超出窗口（或从来没有）：如实退回三点基 —— 它量不到旋前，不假装量到。
        relative = relative_wrist_rotation(arm, entry.get("orientation"))
        if relative is None:
            return None, None
        from teleoperation.retargeting.arm.wrist_targets import (
            rotation_matrix_to_quaternion)

        return (tuple(float(value) for value in
                      rotation_matrix_to_quaternion(relative)), "three_point")

    def _orientation_limit_deg(self) -> float:
        """这一帧最多放行多少度：``min(rate * dt, step)``；``rate <= 0`` = 整个机制关掉（0）。

        两条上限分工见 ``MAX_ORIENTATION_RATE_DEG_S`` / ``MAX_ORIENTATION_STEP_DEG``：
        速率那条按**这一帧的真实时长**换算（6 fps 下 540 度/秒 = 84 度/帧，几乎不拦），
        步长那条与帧长无关（任何帧率都最多 ``MAX_ORIENTATION_STEP_DEG`` 度）。

        ``step <= 0`` = 只按速率（历史行为）；``rate <= 0`` 时一律 0 —— A/B 基线要求
        「关掉就是逐位透传」，不能被步长那条单独接管。
        """
        if self.max_orientation_rate_deg_s <= 0.0:
            return 0.0
        limit = self.max_orientation_rate_deg_s * self.frame_dt
        if self.max_orientation_step_deg > 0.0:
            limit = min(limit, self.max_orientation_step_deg)
        return limit

    def _clamp_orientation(self, side, quaternion):
        """单帧姿态变化超过上限 -> 只放行 ``limit`` 那么多度（见 ``_orientation_limit_deg``）。

        返回 ``(放行后的四元数, 被削掉的角度)``。基准是**上一帧放行后的值**，所以削掉的角
        不会累积成「追赶」（下一帧从削过的位置继续算）；这也正是它和低通的区别：低通在尖峰
        之后要来回振几下才收敛，限速只是把一跳变成一段连续运动，不会过冲。

        为什么对 ``three_point`` 也生效：换基（掌面基丢超过 ``PALM_HOLD_S``）那一下两套基的
        目标差 2.6-3.0 rad（实测 ``outputs/tmp_probe/wrist_live_hold2.csv`` f=246/439/550），
        那是**真实的口径切换**而不是噪声，但同样不该在一帧里完成。

        与 ``_guard_direction`` 的一处不同：``data["arm"]`` 断流时**不清**这里的基准。跳变
        护栏判的是「这是不是手在动」，断流后必须重新起算；限速管的是「命令连不连续」，断流
        之后的大幅转向同样该是一段连续运动，清掉基准反而会把那一跳放过去。要清用 ``reset()``。
        """
        if quaternion is None:
            return None, 0.0
        quaternion = tuple(float(value) for value in
                           np.asarray(quaternion, dtype=np.float64).reshape(4))
        previous = self._last_orientation.get(side)
        limit = self._orientation_limit_deg()
        if previous is None or limit <= 0.0:
            self._last_orientation[side] = quaternion
            return quaternion, 0.0
        step = quaternion_angle_deg(previous, quaternion)
        if not np.isfinite(step) or step <= limit:
            self._last_orientation[side] = quaternion
            return quaternion, 0.0
        relative = _quaternion_multiply(tuple(float(-value) for value in previous[:3])
                                        + (float(previous[3]),), quaternion)
        clamped = tuple(float(value) for value in
                        _quaternion_multiply(previous,
                                             _quaternion_scaled(relative, limit / step)))
        self._last_orientation[side] = clamped
        self.counts["orientation_clamped"][side] += 1
        return clamped, float(step - limit)

    def _orientation_wanted(self, profile) -> bool:
        """要不要带姿态约束：``auto`` = 只在 profile 声明 ``ik_mode="pose"`` 时带。"""
        if self.orientation == "off":
            return False
        if self.orientation == "on":
            return True
        return profile.ik_mode == "pose"

    def _orientation_for(self, profile, state):
        return state.get("orientation") if self._orientation_wanted(profile) else None

    def orientation_target(self) -> dict:
        """``{side: 相对中性的四元数 or None}``：机器人无关的那一份姿态目标。

        为什么单独暴露这一份：同一份相对旋转要发给三台（各自乘自己的中性 EE 朝向），
        而 ``ArmTarget.orientation`` 是按 profile 的 ``ik_mode`` 门控的 —— 调用方若写
        ``per_side[sorted(per_side)[0]].orientation``，就会挑到一台 position-only 的
        机器人（``sorted(['g1','gr1_t2','h1_2'])[0]`` 是 ``g1``，它的 orientation 是
        ``None``），于是姿态约束被**静默丢掉**：实测 2026-10-12，现场那次 h1_2 的
        ``ik_mode="pose"`` 就是这么失效的（残差里 ``orientation_error_deg`` 只有 0.5 度，
        因为 IK 拿"它自己现在的姿态"当了目标）。

        ``orientation="off"`` 时一律 None；否则给这一帧算出来的相对旋转（只要这一帧的
        三点给得出腕坐标系）。position-only 的机器人收到它也会忽略（``arm_ik.solve``
        只在 ``ik_mode="pose"`` 时才用它）。
        """
        if self.orientation == "off":
            return {side: None for side in self.sides}
        return {side: (self._last_state.get(side) or {}).get("orientation")
                for side in self.sides}

    def hand_frame_target(self) -> dict:
        """Current palm frame in robot-base axes: ``x=fingers, y=thumb, z=normal``.

        This is the absolute target for ``orientation_mode="hand_base"``. Unlike the
        relative wrist modes it keeps the visible native hand aligned with the mapped
        operator hand at every pose, including a straight raised arm and a horizontal
        palm. ``None`` on a side means this frame has no usable pose arm + world hand.
        """
        return {side: (None if self._current_hand_frames.get(side) is None
                       else self._current_hand_frames[side].copy())
                for side in self.sides}

    @staticmethod
    def _bend_of(upper, fore):
        """``(上臂向量, 前臂向量)`` -> 肘弯（度），口径与 ``elbow_bend_degrees`` 完全一致。

        直接用那个函数而不是另写一个夹角：报告里的 ``elbow_bend_deg`` 就是它算的，
        「你弯了多少」和「报告里写着你弯了多少」必须是同一个数（180 = 伸直，0 = 折起）。
        造三个点喂进去即可：肩 = 原点、肘 = 上臂向量、腕 = 上臂向量 + 前臂向量。
        """
        u = _unit_direction(upper)
        v = _unit_direction(fore)
        if u is None or v is None:
            return None
        origin = np.zeros(3, dtype=np.float64)
        return float(elbow_bend_degrees(origin, u, u + v))

    def elbow_bend_target(self) -> dict:
        """``{side: 这一帧的肘弯（度）or None}``：机器人无关的那一份「姿势」目标。

        为什么要单独暴露（与 ``orientation_target`` 同一个理由）：这是**操作者层**的量 ——
        同一份给三台，各自的肘关节名、正方向与「加载姿势下自己的夹角」由
        ``PyBulletArmIK.posture_from_bend`` 自己量、自己换。调用方若从某台机器人的
        ``ArmTarget`` 里挑，就会挑到 position-only 那台而把这一项静默丢掉（姿态口径踩过一次，
        见 ``orientation_target``）。

        口径同 ``elbow_bend_degrees``：180 = 完全伸直，0 = 完全折起。给的是**绝对**值而不是
        相对中性的增量：位置任务命令的夹角就是这个绝对角，而每台机器人加载姿势下自己的夹角
        各不相同（g1 172.9 / gr1_t2 111.2 / h1_2 155.2 度），只有把绝对角交给它、由它减掉自己
        的那个数，参考才与命令同向（见 ``PyBulletArmIK.posture_from_bend``）。

        这一帧没有 pose 手臂就给 ``None``：调用方如实退回历史行为。断流的那一帧沿用上一帧
        （``_last_state``）——否则每丢一帧参考就弹回加载姿势，而「一断流肘就掉回去」正是现场
        那一件事。
        """
        out = {}
        for side in self.sides:
            state = self._last_state.get(side) or {}
            out[side] = self._bend_of(state.get("upper"), state.get("fore"))
        return out

    def _gain(self, profile, side, state) -> float:
        """米/米。``profile.metric_scale`` 是本机调参，再按你标定时的臂长归一化。

        臂长固定取**中性标定**那一侧的值（一个常数），不用当帧的肩-肘-腕长度 —— 否则
        肘一弯 gain 就漂，动作幅度会跟着姿势变。

        2026-10-13 加**标定可信度护栏**（``gain_quality_guard``）：归一化系数是
        ``参考臂长 / 标定臂长``，一旦标定那 60 帧的 Pose 关节坏了（实测现场那场
        左 0.2742 m / 右 0.3976 m，同一个人两侧差 31%），它就把动作放到 2 倍上限
        （h1_2 左 gain 1.4），再叠上被夹断的行程就成了「手一动就钉死在盒角」。
        不可信时退回 ``profile.metric_scale``：宁可幅度小，也不放大噪声。
        """
        gain = float(self.gains.get(profile.robot_type, profile.metric_scale))
        if not self.normalise_arm_length:
            return gain
        reach = self.neutral[side]["reach"]
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            reach = state.get("reach")
        if reach is None or not np.isfinite(reach) or float(reach) <= 0.0:
            return gain
        if self.gain_quality_guard:
            trusted, _reason = self._arm_length_trust(side)
            if not trusted:
                return gain
        scale = self.arm_length_reference / float(reach)
        return gain * min(max(scale, self.arm_length_band[0]), self.arm_length_band[1])

    def _reach_plausible(self, reach) -> bool:
        """标定出来的「肩到腕」是不是一个像样的手臂长度（米）。"""
        if reach is None:
            return False
        number = float(reach)
        return bool(np.isfinite(number)) and number >= self.min_plausible_reach_m

    def _arm_length_trust(self, side) -> tuple:
        """``(这一侧能不能用臂长归一化, 原因)``。

        两条判据：本侧标定臂长要像人（``>= min_plausible_reach_m``）；两侧都像人时，
        两边还要互相对得上（同一个人左右差 < ``arm_reach_side_tolerance``）。
        实测反例（现场那场 live_long）：左 0.2742 / 右 0.3976 m —— 一条判据都不满足。
        """
        reach = self.neutral[side]["reach"]
        if not self._reach_plausible(reach):
            shown = _round_or_none(reach, 4)
            if not self.neutral[side]["complete"]:
                # 标定还没攒够帧：这时目标就是中性工作点本身，归一化用不上，也没必要报警。
                return False, f"{side} calibrated arm reach {shown} m is not usable yet (calibrating)"
            return False, (f"{side} calibrated arm reach {shown} m is "
                           f"below {self.min_plausible_reach_m:.2f} m (pose landmarks bad)")
        others = [self.neutral[other]["reach"] for other in self.sides if other != side]
        others = [value for value in others if self._reach_plausible(value)]
        if others:
            values = [float(reach)] + [float(value) for value in others]
            low, high = min(values), max(values)
            if (high - low) > self.arm_reach_side_tolerance * high:
                return False, (f"left/right calibrated reaches differ by "
                               f"{100.0 * (high - low) / high:.0f}% "
                               f"(> {100.0 * self.arm_reach_side_tolerance:.0f}%): "
                               f"{low:.4f} / {high:.4f} m")
        return True, ""

    def _gain_reference(self, side, state) -> tuple:
        """``(用于 gain 的臂长, 来源)``；来源是 ``neutral`` / ``frame`` / ``none``。"""
        reach = self.neutral[side]["reach"]
        if reach is not None and np.isfinite(reach) and float(reach) > 0.0:
            return float(reach), "neutral"
        frame_reach = state.get("reach")
        if frame_reach is not None and np.isfinite(frame_reach) and float(frame_reach) > 0.0:
            return float(frame_reach), "frame"
        return None, "none"

    def _neutral(self, profile, side) -> tuple:
        """这一侧的中性工作点（基座系，米）：优先用 profile 的实测每侧值。"""
        method = getattr(profile, "realtime_neutral", None)
        if callable(method):
            return tuple(float(value) for value in method(side))
        return tuple(float(value) for value in profile.metric_offset)

    def _travel_limit(self, profile, side=None) -> float:
        """基座系里每个轴允许的最大位移（米）。"""
        if self.max_travel_m is not None:
            return float(self.max_travel_m)
        method = getattr(profile, "travel_limit", None)
        if side is not None and callable(method):
            return float(method(side))
        override = getattr(profile, "realtime_travel_m", None)
        if isinstance(override, dict):
            if side is not None and side in override:
                return float(override[side])
        elif override is not None:
            return float(override)
        return float(getattr(profile, "sweep_amplitude_m", DEFAULT_MAX_TRAVEL_M))

    def travel_limits(self) -> dict:
        """``{robot: {side: 米}}``：每台每侧的行程上限（HUD 与报告用）。"""
        return {name: {side: self._travel_limit(profile, side) for side in self.sides}
                for name, profile in self.profiles.items()}

    def _apply_lateral_guard(self, side, position) -> bool:
        """把这一侧的目标挡在自己那半边（只动 y）。``position`` 原地改，返回是否挡了。

        基座系里机器人左手侧是 +y：左腕要 ``y >= +guard``，右腕要 ``y <= -guard``。
        为什么只挡 y 就够：现场那场 h1_2 的每轴上限是左 0.10 / 右 0.15 m，而中性工作点
        只有 y=+-0.05 m，于是「目标」可以合法地走到身体另一侧（左 y 到 -0.05），
        两前臂贴到 2.21 mm。护栏不影响跟手（x/z 与 y 的其他范围照跟），只在越线时停住。
        """
        guard = self.lateral_guard_m
        if guard <= 0.0:
            return False
        sign = 1.0 if side == "left" else -1.0
        if float(position[1]) * sign >= guard:
            return False
        position[1] = sign * guard
        return True

    # ------------------------------------------------------------------ #
    # 报告 / HUD
    # ------------------------------------------------------------------ #
    def trace(self) -> dict:
        """最近一次更新的操作者侧诊断量（``--arm-trace`` 打印它）。

        ``delta_op_cm`` 是「腕相对肩」相对中性姿势的位移，单位厘米。你往某个方向动手，
        只有对应分量明显变化 —— 这正是用来核对轴向约定（哪个符号对不对）的现场证据。
        """
        out = {}
        for side in self.sides:
            state = self._last_state.get(side) or {}
            delta = state.get("delta_op")
            out[side] = {
                "source": state.get("source", SOURCE_NONE),
                "valid": bool(state.get("valid", False)),
                "reason": state.get("reason", "no frame yet"),
                "delta_op_cm": None if delta is None else [round(float(v) * 100.0, 2)
                                                           for v in delta],
                "operator_reach_m": _round_or_none(state.get("reach"), 4),
                "elbow_bend_deg": _round_or_none(state.get("elbow_bend_deg"), 1),
            }
        return out

    def summary(self) -> dict:
        """报告段：轴向约定、每台的 gain/行程、中性记录、各来源的帧数。"""
        return {
            "axis_map": self.axis_map.describe(),
            "orientation": self.orientation,
            "orientation_mode": self.orientation_mode,
            "hand_roll_deg": self.hand_roll_deg,
            "jitter_filter": self.jitter_filter,
            "jitter_filter_params": {
                "arm_min_cutoff_hz": ARM_INPUT_FILTER_MIN_CUTOFF_HZ,
                "arm_beta": ARM_INPUT_FILTER_BETA,
                "hand_min_cutoff_hz": HAND_TARGET_FILTER_MIN_CUTOFF_HZ,
                "hand_beta": HAND_TARGET_FILTER_BETA,
            },
            # 姿态目标的口径：相对中性（每台各自乘自己的中性 EE 朝向）。
            "orientation_frame": ("native_hand_absolute"
                                  if self.orientation_mode == "hand_base"
                                  else "neutral_relative"),
            "normalise_arm_length": self.normalise_arm_length,
            "arm_length_reference_m": self.arm_length_reference,
            # 标定没攒够帧时的口径 + 每侧进度：2026-10-14 现场那场 4 分钟停在 cal=15/30、
            # 三台机器人一动不动，报告里却看不到这一点。写进来，下次先看数字再猜。
            "calibrate_mode": self.calibrate_mode,
            "calibration_frames": self.calibration_frames,
            # 等手点的上限（帧；0 = 不等）。见 ``PALM_WAIT_FRAMES``：手臂第 1 帧就有、
            # 手部 21 点要 5-7 s，不等就是「窗口在手伸进来之前烧完」。
            "palm_wait_frames": self.palm_wait_frames,
            "palm_wait_frames_used": {side: self._palm_wait[side] for side in self.sides},
            "calibration_progress": {side: list(value) for side, value
                                     in self.calibration_progress().items()},
            # 2026-10-13 三个护栏：行程等比缩 / 横向护栏 / 标定可信度。都在报告里留痕，
            # 免得下一次看数字的人以为 gain 1.4 那种值真是"算"出来的。
            "travel_mode": self.travel_mode,
            # 「你的两条手臂是不是镜像」：0 度左右 = 输入是镜像手势（桥如实照抄），
            # 大 = 两侧朝同一边摆/一侧手点跑偏（见 SYMMETRY_SAME_SIDE_DEG）。
            "operator_symmetry": self.symmetry_summary(),
            "lateral_guard_m": self.lateral_guard_m,
            "gain_quality_guard": self.gain_quality_guard,
            "min_plausible_reach_m": self.min_plausible_reach_m,
            "arm_reach_side_tolerance": self.arm_reach_side_tolerance,
            "arm_length_trust": {side: {"trusted": trusted, "reason": reason}
                                 for side, (trusted, reason) in
                                 ((side, self._arm_length_trust(side))
                                  for side in self.sides)},
            "gain_by_side": {
                side: {name: round(self._gain(profile, side, {"reach": None}), 4)
                       for name, profile in self.profiles.items()}
                for side in self.sides},
            "travel_limit_m": self.travel_limits(),
            # 位移口径与两段式映射的两段比（k = 机器人连杆长 / 你的连杆长）。缺数据的那一侧
            # 会在每一帧的 info["map"] 里显示 "chain (segments unavailable)"。
            "map_mode": self.map_mode,
            # 方向口径的力度（map_mode="posture" 时生效）：1 = 机器人方向完全对齐你的方向。
            "posture_gain": self.posture_gain,
            # 方向口径这一侧能不能跑（要肩锚点 + 机器人自己的连杆长）：跑不了时那一帧的
            # info["map"] 会写明 "chain (posture unavailable: ...)"。
            "posture_available": {
                name: {side: bool(
                    "shoulder" in self.robot_anchors.get(name, {}).get(side, {})
                    and self.robot_links.get(name, {}).get(side))
                    for side in self.sides}
                for name in self.profiles},
            "shoulder_anchors_m": {name: {side: [round(float(value), 5)
                                                 for value in entry["shoulder"]]
                                          for side, entry in per_side.items()
                                          if "shoulder" in entry}
                                   for name, per_side in self.robot_anchors.items()},
            "elbow_task": self.elbow_task,
            "elbow_gain": self.elbow_gain,
            "elbow_anchors_m": {name: {side: {"elbow": [round(float(value), 5)
                                                        for value in entry["elbow"]],
                                              "wrist": [round(float(value), 5)
                                                        for value in entry["wrist"]]}
                                     for side, entry in per_side.items()}
                                for name, per_side in self.robot_anchors.items()},
            "robot_links_m": {name: {side: {"upper": round(float(value["upper"]), 4),
                                            "fore": round(float(value["fore"]), 4)}
                                     for side, value in per_side.items()}
                              for name, per_side in self.robot_links.items()},
            "segment_scales": {name: {side: (None if self._segment_scales(profile, side) is None
                                             else [round(float(value), 4) for value in
                                                   self._segment_scales(profile, side)])
                                      for side in self.sides}
                               for name, profile in self.profiles.items()},
            #: 哪一侧的 k 被判为不可用、为什么（空串 = 能用 / 还没标定）。见
            #: ``SEGMENT_LENGTH_BAND_M``：这是「标定那一场手臂被压扁」的唯一事后证据。
            "segment_scale_notes": {side: self.segment_scale_note(side)
                                    for side in self.sides},
            #: 左右段长差太多时的原因（空串 = 正常 / 数据不够）。见
            #: ``SEGMENT_ASYMMETRY_MAX``：这条与机器人无关，只判你的左右臂是否对称。
            "segment_asymmetry_note": self.segment_asymmetry_note(),
            "neutral_base_m": {name: {side: list(self._neutral(profile, side))
                                      for side in self.sides}
                               for name, profile in self.profiles.items()},
            "neutral": {side: {"offset": None if self.neutral[side]["offset"] is None
                               else [round(float(v), 5)
                                     for v in self.neutral[side]["offset"]],
                               "reach_m": _round_or_none(self.neutral[side]["reach"], 4),
                               # 记了姿态基准的侧面才能做姿态约束（老记录没有这一段）。
                               "orientation_recorded": self.neutral[side]["orientation"] is not None,
                               "orientation": (None if self.neutral[side]["orientation"] is None
                                               else [[round(float(v), 4) for v in row]
                                                     for row in
                                                     np.asarray(self.neutral[side]["orientation"],
                                                                dtype=np.float64)]),
                               "samples": self.neutral[side]["samples"],
                               # 标定期间拿到掌面基的帧数 + 结论：0/False = 这一整场只有三点基，
                               # 量不到旋前/旋后（现场「手腕不跟」就是这个，见 _palm_samples）。
                               "palm_samples": self.neutral[side].get("palm_samples", 0),
                               "orientation_palm_recorded":
                                   self.neutral[side].get("orientation_palm") is not None,
                               "orientation_palm_local_recorded":
                                   self.neutral[side].get("orientation_palm_local") is not None,
                               # 掌面基被判不可用的原因 + 那几帧彼此的散度（空串/None = 能用
                               # 或这一侧没手点）：与 ``palm_samples`` 一起看，就是
                               # 「有样本却没有基」的那张单子（见 PALM_MIN_SAMPLE_FRACTION）。
                               "palm_note": self.neutral[side].get("palm_note", ""),
                               "palm_spread_deg": self.neutral[side].get("palm_spread_deg"),
                               # 掌面基的来路：``neutral`` = 标定那次采的，``late`` =
                               # 中性没有、运行中补采的（``palm_reference_frame`` 是第几帧），
                               # 空串 = 这一侧整场没有掌面基（旋前不跟，见 PALM_LATE_FRAMES）。
                               "palm_reference_source":
                                   self.neutral[side].get("palm_reference_source", ""),
                               "palm_reference_frame":
                                   self.neutral[side].get("palm_reference_frame"),
                               # 迟到基准取零点时，前臂方向与标定那一刻差多少度：腕从那个
                               # 姿势起读数为零，所以这个数就是「腕立正时的固定偏移」。
                               "palm_reference_pose_deg":
                                   self.neutral[side].get("palm_reference_pose_deg"),
                               # 判死时被踢出多数簇的帧数（掌面翻号 / 半遮挡，见 _palm_cluster）。
                               "palm_outliers": self.neutral[side].get("palm_outliers", 0),
                               "complete": self.neutral[side]["complete"]}
                        for side in self.sides},
            # 「肘跟手」（``--arm-elbow-follow operator``）用的两个操作者量：中性那一帧的
            # 肘弯（标定口径）与这一帧的肘弯（真正喂给 IK 的那个绝对角）。180 = 伸直。缺数据的
            # 侧面是 None —— 那一侧 IK 拿到的姿势参考就回到历史行为，报告里看得出来。
            "elbow_bend_deg": {side: {
                "neutral": _round_or_none(
                    self._bend_of(self.neutral[side].get("upper"),
                                  self.neutral[side].get("fore")), 2),
                "current": _round_or_none(self.elbow_bend_target()[side], 2)}
                for side in self.sides},
            "counts": {"frames": self.counts["frames"],
                       "valid": dict(self.counts["valid"]),
                       "held": dict(self.counts["held"]),
                       "fallback": dict(self.counts["fallback"]),
                       "oriented": dict(self.counts["oriented"]),
                       # 掌面基丢帧、沿用上一帧掌面旋转的帧数（见 ``PALM_HOLD_S``）：
                       # 它占 ``frames`` 的比例就是现场的手点丢帧率。
                       "palm_held": dict(self.counts["palm_held"]),
                       # 运行中补采到迟到基准的次数（0/1；见 ``PALM_LATE_FRAMES``）。
                       "palm_late": dict(self.counts["palm_late"]),
                       # 姿态目标被限速削过的帧数（见 ``MAX_ORIENTATION_RATE_DEG_S``）。
                       "orientation_clamped": dict(self.counts["orientation_clamped"]),
                       "clamped": dict(self.counts["clamped"]),
                       "guarded": dict(self.counts["guarded"]),
                       "segments": dict(self.counts["segments"]),
                       "posture": dict(self.counts["posture"]),
                       "elbow": dict(self.counts["elbow"]),
                       "direction_glitch": dict(self.counts["direction_glitch"]),
                       "direction_glitch_kept": dict(self.counts["direction_glitch_kept"]),
                       "gain_guarded": dict(self.counts["gain_guarded"]),
                       # 没跟手的帧为什么没跟手（``HOLD_REASONS``）：现场排查第一眼看它。
                       "held_reason": {side: dict(kinds) for side, kinds
                                       in self.counts["held_reason"].items()}},
            "last_trace": self.trace(),
        }

    def reset(self, recalibrate: bool = False) -> None:
        """清状态。``recalibrate=True`` 时连中性姿势一起清掉（要重新站一次）。"""
        self.frames = 0
        self._last_state = {side: None for side in self.sides}
        self._last_direction = {}
        self._palm_hold = {side: None for side in self.sides}
        self._palm_late = {side: [] for side in self.sides}
        self._arm_filter = AdaptiveArmPointFilter()
        self._hand_filter = AdaptiveRotationFilter()
        self._last_orientation = {}
        self.counts = {"frames": 0,
                       "valid": {side: 0 for side in self.sides},
                       "held": {side: 0 for side in self.sides},
                       "fallback": {side: 0 for side in self.sides},
                       "oriented": {side: 0 for side in self.sides},
                       "palm_held": {side: 0 for side in self.sides},
                       "palm_late": {side: 0 for side in self.sides},
                       "orientation_clamped": {side: 0 for side in self.sides},
                       "clamped": {side: 0 for side in self.sides},
                       "guarded": {side: 0 for side in self.sides},
                       "segments": {side: 0 for side in self.sides},
                       "posture": {side: 0 for side in self.sides},
                       "elbow": {side: 0 for side in self.sides},
                       "direction_glitch": {side: 0 for side in self.sides},
                       "direction_glitch_kept": {side: 0 for side in self.sides},
                       "gain_guarded": {side: 0 for side in self.sides},
                       "held_reason": {side: {kind: 0 for kind in HOLD_REASONS}
                                       for side in self.sides}}
        self._current_hand_frames = {side: None for side in self.sides}
        if recalibrate:
            self._samples = {side: [] for side in self.sides}
            self.neutral = {side: _neutral_default() for side in self.sides}
            self._palm_samples = {side: 0 for side in self.sides}
            self._palm_seen = {side: False for side in self.sides}
            self._palm_wait = {side: 0 for side in self.sides}
            self._counted = {side: 0 for side in self.sides}


def _round_or_none(value, digits: int):
    """四舍五入；``None``/NaN/inf 一律返回 ``None``（JSON 里不出现 NaN）。"""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)

