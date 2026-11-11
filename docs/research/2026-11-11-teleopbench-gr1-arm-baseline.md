# TeleOpBench GR1-T2 手臂复现基线与首轮离线对照

日期：2026-11-11。范围：单目视觉与 Vision Pro 手臂链路的源码核对，以及 Vision Pro 录制的实际离线运行。未修改当前项目业务代码、公共接口、测试控制器或原始录制。没有进行实体机器人、物体接触或任务成功率实验。

## 1. 固定基线与本次产物

- 官方项目：[TeleOpBench](https://gorgeous2002.github.io/TeleOpBench/)。其 Code 链接指向 `cyjdlhy/TeleOpBench`。
- 论文固定版本：[arXiv:2505.12748v2，§3.2.1、§3.2.3](https://arxiv.org/html/2505.12748v2#S3.SS2)。
- 官方源码下载目录：`D:/2026/code/TeleOpBench-reference`，由用户指定。
- 官方提交：`4cc410d149ce63c86f485b257cba3b3e0ec031d6`，detached HEAD。
- 使用 filtered sparse clone，检出 `teleop/`、`source/`、GR1-T2 URDF 与仓库根文件；其余大型机器人、网格资产未全部下载。
- 当前项目被测提交：`06e833ef3d66a48bf823368c4af7dc56c1a93b9f`。
- 修改前检查点：`checkpoint-20261111-1941-gr1-arm-baseline`。
- 分析脚本、隔离依赖、日志、数组、图与动画在 [outputs/acceptance/teleopbench-gr1-20261111](../../outputs/acceptance/teleopbench-gr1-20261111/)。它们是本次实验产物，不属于生产控制实现。

关键文件：

| 文件 | 内容 |
|---|---|
| [compare_vr.py](../../outputs/acceptance/teleopbench-gr1-20261111/compare_vr.py) | 官方原始类加载、SDK 输入适配、当前控制会话回放、逐帧采样 |
| [render_comparison.py](../../outputs/acceptance/teleopbench-gr1-20261111/render_comparison.py) | 指标、图、骨架动画与来源清单 |
| [metrics.json](../../outputs/acceptance/teleopbench-gr1-20261111/metrics.json) | 完整统计与运行配置 |
| [provenance.json](../../outputs/acceptance/teleopbench-gr1-20261111/provenance.json) | 源码、模型、输入、脚本的 SHA-256 和依赖版本 |
| [comparison.png](../../outputs/acceptance/teleopbench-gr1-20261111/comparison.png) | 自身目标误差、处理耗时、关节步长 |
| [trajectories.png](../../outputs/acceptance/teleopbench-gr1-20261111/trajectories.png) | 左右末端目标与关节命令 FK 轨迹 |
| [preview.gif](../../outputs/acceptance/teleopbench-gr1-20261111/preview.gif) | 两套关节命令的骨架预览，约 10 倍速，无物体接触 |
| official.npz / current.npz | 同一时间轴的目标、关节角和正运动学位置；current 另含物理步进后实际末端位置 |

## 2. 当前结论

第一轮已经实际运行两套控制计算，而不是仅对比函数名称。结果首先暴露了当前链路的高频命令拒绝：左侧 2,867 帧、右侧 2,865 帧记录 `pose_position_conflict`；每侧共有 2,876 帧处于对齐后的 observed 状态。代码在这一条件下保持求解前关节值。这是本段输入中动作跟随不足的一条直接排查线索，但还不是对原因的完整隔离实验。

同时，两套生成目标本来就不同：排除前 2 秒后，左右末端目标的位置差中位数分别约 236.1、260.3 mm。因此不能只比较最终关节角，也不能根据本次误差表宣布某套更准确。

官方原样 GR1 腕转换含非正交矩阵；两套机器人模型、末端定义、参考姿势和控制时间处理不同；SDK 到 WebXR 的腕局部轴等价性尚未证明。以上因素均未被静默修正。

## 3. 论文与公开代码的不同

论文视觉方法描述 T-pose 下人体形状优化、连杆尺度计算、姿态重建后运行 PINK；VR 描述头部参考转换到骨盆参考，且只给 PINK 腕平移目标。

固定提交的实际 GR1 代码则：

- 视觉入口给定一组十维 beta，`scale=1.0`；接收器保存 scale，但活跃处理没有使用它，`scale_joints` 使用机器人对应的预设分段参数。
- 提供离线二十维 `optimize_beta` 工具，但存在函数定义不代表实时入口会进行标定；实际模型解析器也配置了二十维 beta，和入口十个值的兼容性尚未执行验证。
- GR1 相机和 VR 使用 PINK，腕任务同时有位置与朝向代价，另有头部朝向任务。
- G1、H1-2 对应公开活跃路径使用 CasADi/Pinocchio/IPOPT。不能把所有公开实现统一描述为 PINK。

依据：[固定 main.py](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/main.py)、[GR1 camera wrapper](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/receiver_wrapper/camera_wrapper_gr1t2_media.py)、[scale_utils](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/utils/scale_utils.py)。

## 4. GR1 Vision Pro 官方数值链

官方输入是 OpenXR/WebXR 世界坐标：x 向右、y 向上、z 向后。机器人基准轴为 x 向前、y 向左、z 向上。

```text
C = [[ 0, 0,-1],
     [-1, 0, 0],
     [ 0, 1, 0]]
T_robot = C4 @ T_input @ inverse(C4)
```

接收器对腕右乘 `hand2inspire_*_arm`，减去头部平移，再乘局部轴对齐矩阵；不使用完整头位姿的逆矩阵消除头部旋转。

求解器 `adjust_pose` 的平移公式：

```text
p_target = 1.25 * (C @ (p_wrist - p_head) + [0,0,0.42])
```

0.42 m 偏移在缩放之前加入，最终贡献为 0.525 m。该路径的臂长比是预设 `0.60/0.48`，没有在这次运行中测量操作者臂长。

### 4.1 必须保留的官方源问题

`constants_gr1t2.py` 中实际使用的矩阵为：

```text
BL = [[1,0,0], [0,0,-1], [1,1,0]]
BR = [[1,0,0], [0,0, 1], [1,-1,0]]
```

二者行列式为 1，但不正交。乘后续标准对齐矩阵不能去掉该缺陷。本次原样执行，逐帧 `||R.T @ R - I||_F` 中位数约 1.732；不将它投影修复为 SO(3)，也不计算以该矩阵为目标的“角度误差”。

源码先令 `self.solver=available_solvers[0]`，再设置局部变量 `solver='quadprog'` 打印，但实际求解仍用 `self.solver`。本机打印 quadprog，实际使用 proxqp；保留这一行为。

### 4.2 PINK 配置

- 原始模型：`assets/GRX/GR1/GR1T2/urdf/GR1T2_inspire_hand.urdf`。
- 锁定腿、腰、手指，保留三头部与十四手臂关节。
- 初始状态为 reduced model q0；姿态参考另把双肘设为 -90°。
- 末端：`L_hand_base_link`、`R_hand_base_link`，位置代价 1、朝向代价 1、LM damping 0.5。
- 头部：`head_yaw_link`，位置代价 0、朝向代价 1。
- PostureTask：cost 0.001、LM damping 1、gain 0.1。
- 每次调用做一次速度 QP 求解并积分 0.005 秒，然后裁剪到各关节限位中央 90%；没有每个目标迭代至收敛。
- 本段录制约 27.9 Hz；0.005 秒是源码中的数值积分步长，不是已达到 200 Hz 的端到端采样频率。
- 此 GR1 VR 路径没有相机路径中的 Kalman 关节滤波。

依据：[固定 VR wrapper](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/receiver_wrapper/tv_wrapper_gr1t2.py)、[constants_gr1t2](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/receiver_wrapper/constants_gr1t2.py)、[GR1 VR IK](https://github.com/cyjdlhy/TeleOpBench/blob/4cc410d149ce63c86f485b257cba3b3e0ec031d6/teleop/robot_control/tv_ik_gr1t2.py)。

## 5. 本地 SDK 输入适配与证据限制

本地 `.venv/dex-native/Lib/site-packages/avp_stream` 的版本为 `2.51+wincontrib`。实际源码：

```text
D = [[1,0,0], [0,0,-1], [0,1,0]]
T_wrist_sdk = D4 @ T_wrist_native
T_head_sdk = D4 @ T_head_native @ Rx(-90deg)

T_wrist_native = inverse(D4) @ T_wrist_sdk
T_head_native = inverse(D4) @ T_head_sdk @ Rx(+90deg)
```

这与把腕矩阵做相似变换不同。SDK 平移到机器人平移的转换可明确写为 `Q @ p_sdk`，其中 `Q(x,y,z)=(y,-x,z)`。

但 ARKit native 腕局部轴到 WebXR 腕局部轴的固定转换没有原始证据闭合。本次适配恢复 SDK 前的矩阵后直接供官方接收器使用，腕局部轴对应为未验证假设。录制也没有保存 SDK 版本，所以不能将本机安装版本视为采集时版本的确证。

本次只比较手臂：官方 wrapper 的手指分支用零占位 landmark 通过，手指输出未使用；不宣称复现手部控制。SDK dict API 的 finger/arm 矩阵实际是 wrist-local，现有录制中手根为单位矩阵也支持这一点；旧 docstring/属性所称 world 空间不作为转换证据，未修改文件元数据。

来源：[SDK streamer.py](../../.venv/dex-native/Lib/site-packages/avp_stream/streamer.py)、[grpc_utils.py](../../.venv/dex-native/Lib/site-packages/avp_stream/utils/grpc_utils.py)、[录制脚本](../../devtools/visionpro_record.py)。

## 6. 当前项目链路与模型差异

当前被测路径为 arms-only `RobotControlSession`，source_kind=visionpro。实际使用当前规格与场景、`VisionProReplayInput`、`WristSceneController`、`PyBulletArmIK`、`RobotOutput`，没有重新写一套当前控制算法。

当前目标是机器人对齐时末端位置加上 SDK 腕相对初始位置的变化；不减当前头部位置。WristTracker 位移半径上限 0.2 m、位移变化速度上限 0.3 m/s，旋转另有平滑与限速。当前末端为 `left_hand_pitch_link` / `right_hand_pitch_link`，官方使用 `L_hand_base_link` / `R_hand_base_link`。

当前 IK 若朝向候选位置残差超过 profile 阈值且比位置候选更差，记录 `position_fallback`；`solve()` 随后将它解释为 `pose_position_conflict`，返回求解前关节值。记录 observed/tracking 不能证明每帧都接受了新 IK 目标。

两个 URDF 的拓扑和主要关节轴可对应，但存在以下真实差异：

| 项目 | 当前 robot.urdf | 官方 inspire URDF |
|---|---|---|
| 左肩 yaw 原点 xyz / m | -0.00085, 0.04, -0.0749 | 0, 0.04, -0.0709 |
| 左肘 pitch 原点 y / m | 0.0051 | 0.0010503 |
| 右肩 pitch 原点 rpy-x / rad | -0.43992 | -0.43275 |
| 左腕 roll 限位 / rad | [-0.61, 0.61] | [-0.87, 0.96] |
| 左腕 pitch 限位 / rad | [-0.96, 0.87] | [-0.61, 0.61] |

还有肩、腕和腰偏置差异。不能把各用各模型的关节输出直接解释为求解器精度差。

本地来源：[robot_session.py](../../src/teleoperation/applications/robot_session.py)、[wrist_control.py](../../src/teleoperation/simulation/wrist_control.py)、[wrist_orientation.py](../../src/teleoperation/retargeting/arm/wrist_orientation.py)、[arm_ik.py](../../src/teleoperation/simulation/arm_ik.py)、[当前 URDF](../../assets/robots/gr1/urdf/robot.urdf)。

## 7. 实际运行方法与结果

选择 `datasets/raw/visionpro_hand_arm_20261012_215137_live.h5`，2,886 帧、103.458 秒。头和双腕均为有限 (T,4,4)，双手 valid 均 2,886；时间严格递增。另一份 165 秒录制也有头和双腕，但本轮没有运行第二份，以免把重复数据集运行当作新的等价性证据。

官方运行从原始源码 AST 加载 `TeleVisionWrapper` 与 `GR1T2_ArmIK` 的完整类定义，跳过网络/WebXR 初始化，逐帧注入录制。官方类和常量文件未改。只将模型加载绑定为相同 URDF 的 Pinocchio 运动学模型与空视觉/碰撞几何；临时 URDF 副本写在产物目录，因此无需启动 MeshCat 或改写官方文件。没有增加收敛循环，也没有修复官方旋转或求解器选择。

当前路径每帧按源时间差推进与正式入口一致的物理步数。命令 FK 在保存并临时设置已接受关节位置后测量，随后恢复原位置、速度；actual 是临时设置前、物理步进后的实际位置。

运行使用现有 dex-native 的 Pinocchio 3.3.1、PyBullet 等，PINK 4.4.0、qpsolvers 4.13.0、quadprog 0.1.13、loop-rate-limiters 1.2.0 仅安装到产物目录的 dependencies；未修改基础环境。依赖版本未证明与论文实验一致。

排除起始 2 秒后的统计：

| 指标 | 官方源码 + SDK 适配输入 | 当前原链路 |
|---|---:|---:|
| 左侧命令 FK 到各自目标的位置误差中位数 / mm | 130.5 | 187.5 |
| 右侧命令 FK 到各自目标的位置误差中位数 / mm | 6.4 | 84.1 |
| 左侧位置误差 P95 / mm | 450.8 | 268.0 |
| 右侧位置误差 P95 / mm | 178.1 | 179.0 |
| 全录制最大关节命令步长 / deg | 8.01 | 20.31 |
| 官方返回 success / 当前 observed 左、右 | 2886/2886 次调用 | 2876、2876 帧 |

这些是不同目标、模型、末端及配置下的描述统计，不是精度排行榜。官方 success 是没有抛出求解异常，不代表满足末端误差阈值。

当前物理实际位置到下发命令 FK 的误差中位数为左 0.078 mm、右 0.049 mm，而命令 FK 到目标误差更大。这表明在本段数据的中位数意义下，应优先检查目标生成与命令接受逻辑；它不排除个别动态帧的电机跟踪误差。

处理耗时中位数：官方 `solve_ik` 调用约 0.263 ms；当前 `next_batch` 双臂会话约 58.08 ms。两者测量区间不同、当前包含更重的多阶段求解；不能据此宣称端到端速度比或实现实时频率。

## 8. 视觉运行的实际边界

已下载并核对视觉源码，但本轮未执行单目身体链路。现有 `camera_20261015_115030.h5` 和 aligned 文件只含手部关键点、world 手点、帧号和时间，没有 RGB 或 SMPL body pose。无法从这些文件重建官方所需的人体姿态输入，不能把它们伪装成身体数据。

官方 GR1 视觉活跃映射覆盖根旋转为 `Euler_xyz(pi/2,0,pi/2)`，在给定 beta 下重建人体关节，减骨盆，再调用 scale_joints。其预设尺度为：肩高 0.4148039488176755 m、肩宽比例 0.9426624260477016、上臂比例 1.0560934022512718、前臂比例 0.9853709280947656。它与当前姿势方向映射不同。

需要新增带身体姿态参数的录制或带 RGB 的输入，才能进入视觉同数据执行对照；不需要重构测试控制器。

## 9. 可重复运行

在当前仓库根目录用 PowerShell：

```powershell
$python = '.venv/dex-native/Scripts/python.exe'
$run = 'outputs/acceptance/teleopbench-gr1-20261111'
& $python "$run/compare_vr.py" official
& $python "$run/compare_vr.py" current
# 两个进程均退出码 0、各自 NPZ 写完后再生成图。
& $python "$run/render_comparison.py"
```

脚本重跑会替换本次产物目录里的同名结果，不改原始录制或官方源码。原始官方目录保持 detached HEAD 且工作区干净。它是稀疏检出，直接运行完整官方网络/手部前端仍需要另外准备网格、许可证模型、网络输入等；本次 arm-only harness 不声称完整系统已安装运行。

## 10. 来源哈希与下一步

所有实际执行的本地文件哈希、运行包版本和脚本哈希见 provenance.json。下表中官方文件使用固定提交的 Git blob 原始字节哈希；Windows 检出的 Python 文件自动转换了 CRLF，工作副本哈希因此不同，已验证仅为换行差异，执行字节哈希另见 provenance.json。其余条目为本地实际文件 SHA-256：

| 路径 | SHA-256 |
|---|---|
| 官方 teleop/robot_control/tv_ik_gr1t2.py | 35b511b2bd80991f0f24f4a7a8b4845802869731861117bf3b8d8041ccd9a476 |
| 官方 teleop/receiver_wrapper/tv_wrapper_gr1t2.py | 3920cff570aede65e3da9a45784f1d6be786de1c342a38ced22906f49d490435 |
| 官方 teleop/receiver_wrapper/constants_gr1t2.py | 3fb82926efb8a3f0059abb040e1c31f61f44fcd8d824b5d005261af6130753c4 |
| 官方 GR1T2_inspire_hand.urdf | 56e9373e8a2023cf920fedc7913281a010d0ffbb38731d7d015a5307cd30e6d3 |
| 当前 assets/robots/gr1/urdf/robot.urdf | 5686e0189a9277c5fa01e67e5b0048581e1c7719fb37063f13e53e5a1d123ae0 |
| SDK streamer.py | a8950f11f51d3d906dddbcb42c7b91862e6161c1037415be70e6fb2ae62fbf52 |
| SDK utils/grpc_utils.py | d792159e4a5a9ae7ab87e72293fa83feef61252a98b381007010a57afc7264fb |
| 输入 Vision Pro H5 | 6d98e00015bff2802906b05ab8fced750b2b415d6b99481e15533002a878b64b |

建议下一步先做同一模型、同一末端、同一已验证平移目标的诊断，对照位置任务与加入朝向后的命令接受情况；另用原生 WebXR 录制闭合腕局部轴问题。源矩阵是否需要修复、当前拒绝策略是否需要修改，应在这些证据之后决定，分别作为可回退改动。本报告和离线产物不构成业务代码修复。
