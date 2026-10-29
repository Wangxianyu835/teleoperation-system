# 相机与 Vision Pro 腕朝向输入和验收

实施日期：2026-10-29。修改前提交：`e1805eff7dd35020cb6d094fde65f233cdee99c1`。
检查点：`checkpoint-20261029-2121-wrist-input`。

本次按用户要求不运行自动测试、不打开设备；语法检查与静态审查不代表真人跟随验收通过。
新增 `tests/test_wrist_observations.py` 为待运行的回归用例。表格中的验收状态初始均为待验收。

## 现场反馈后的修正（以本节新参数为准）

来源报告：`outputs/tmp_realtime/wrist_20261029_223600.json`，对应原始 NPZ 同名。
该次 1009 帧中，249 帧处于腕标定，但旧入口仍持续下发位置目标；肩 pitch 总行程 214.29°。
旧命令选择 G1 手臂，场景实际仍加载 H1/GR1/G1 三台。位置误差 p95 为 81.84 mm，朝向误差 p95 为 9.43°。
末帧 lost_hold 只说明退出前手 world 点缺失，末帧朝向误差 0° 不代表全程稳定。

手部有 861 帧有效命令，指关节命令幅度达到约 1.57 rad，实际末帧 G1 食指两关节约 0.50/1.33 rad。
这能排除“完全没有手部输出”，不能证明现场每次开合都正确跟随。本次保留手部算法，增加实际关节行程统计与手部标定提示来继续定位。

修复前提交：`ece43b1c58553e93a31f05f9b1e920613511cfca`。
检查点：`checkpoint-20261029-2247-wrist-stability`。

- 腕与身体中性标定均完成前，机械臂用初始关节目标保持，不调用运动 IK。几何手后端需要张开手完成独立零位标定，窗口提示 `hand=open-hand-calibrating/ready`。
- `--robot g1` 同时限制场景模型和手臂驱动对象，避免误看其它机器人的手和无用计算。
- 腕目标旋转幅度为源旋转的 65%；相机腕位置变为初始实际腕位置加相对目标位移的 50%，最大半径 10 cm。它用于本轮转腕验收，不再按绝对人体骨段方向将机器人突然移到另一套姿态。
- 新腕路线用当前姿态直接求完整腕位姿，不先执行肘腕纯位置热启动。肘仅由初始姿态参考管理，本轮不追踪肘位置。
- 候选解上臂/肘相对初始关节偏离最多 0.8 rad，腕关节最多 1.2 rad；关节命令以最多 1 rad/s 且每次不超过 0.08 rad 渐进。限制会缩小工作范围，不保证所有姿态均能到达。
- 每次候选命令的关节路径以 ≤0.025 rad 采样，对非相邻臂/躯干/其它肢体碰撞网格查询 3 mm 间距，遇到冲突停在上一个可接受采样点，查询后恢复所有被临时设置的位置和速度。
- 相邻关节结构及同一手内指间碰撞不属于这个臂部拦截器；初始结构已有接触使用记录的阈值，只阻止进一步加深。它不是碰撞约束规划器，也不能保证物理跟踪过程绝不碰撞。
- 报告增加 `wrists.hand_actual_range_rad`、`wrists.arm_actual_range_rad` 和 `wrists.collision`。窗口增加 `hand=... cmdMax=...rad`。

本轮按用户要求停止额外审查、不运行自动测试或设备，只有语法检查；上述为已写入的修正行为，不是已验证的现场结论。

复验先不移动手臂：张开手保持自然掌姿，等待两类标定完成，再做翻掌、俯仰、偏转和握拳。若仍抖动或折叠，立即结束该次仿真并保留新报告。
检查手指时放大 G1 手部画面：`cmdMax` 变化只证明命令变化，实际关节行程有变化才证明执行层运动。G1 只有三指且映射覆盖有限，不应期待五指逐一对应。

## 数据流与范围

`RawHandFrame.wrists[side]` 是独立的 `WristObservation`，保存源坐标中的 xyzw 四元数、可选米制位置、独立有效标记、源/局部坐标系与采样时间。
手指仍消费原始骨骼；手臂消费腕观测。手指镜像归一化不进入腕旋转。
有效腕观测自动参与跟随，无新增启用开关。旧 `--arm-orientation` 参数仅保留解析兼容，已从帮助中隐藏，不再控制新路线。

- 相机：在输入适配层从原始 MediaPipe world 点的腕/食指根/中指根构造正规掌基，再转四元数。没有 world 点时朝向无效，不能用图像归一化点冒充。Hands 的局部原点位置不能直接当作 Pose 全局腕位置。
- VR：读取独立 `*_wrist`，接受 SDK `(1,4,4)` 和录制 `(4,4)`；不从手指局部矩阵恢复腕全局位姿。手指缺失时仍可输出有效腕观测。
- 相机运行入口：`python -m teleoperation.simulation.realtime_teleop`。需要现有 Hands/Pose `.task` 文件。TRON2A 实时控制服务也消费新增腕字段。
- VR 运行入口：现有 `python -m teleoperation hand vr`，TransHandR 与 dex 两后端均在 `--scene robots` 中驱动手臂。独立 `--scene hands` 场景没有机械臂。
- 相机 NPZ、world H5、hand/body H5、增量 raw H5 增加腕观测；回放支持直接读取已记录四元数，旧 NPZ 仍可从 world 点重建。旧 VR 文件没有腕矩阵时不会伪造；没有独立腕掩码时保守沿用旧有效掩码。
- 这是实时/原始观测回放路线；历史仅保存手指角度的结果文件不包含新的手臂命令轨迹。

## 坐标与初始对齐

相机掌基 `build_palm_basis()` 始终满足 det(R)=+1；左手关键点 canonical 映射额外含反射。TransHandR VR 左手点映射 `(y,-z,x)` 的 det=-1，右手 `(-y,z,-x)` 的 det=+1；这些点映射保持原样。

本地 SDK `streamer.py` 已对腕矩阵执行 YUP2ZUP，输出 +X=右、+Y=前、+Z=上，矩阵 API 返回 `(1,4,4)`。新腕参考系命名 `visionpro_sdk_zup`，到机器人基座轴的变换为：

```text
X_base = Y_sdk
Y_base = -X_sdk
Z_base = Z_sdk
```

相机使用现有 `AxisMap`；VR 要求 SDK `origin='avp'`。这些是明确的轴对应约定，不能宣称完成了相机/头显到真实机器人基座的外参标定。

初始对齐要求连续 10 个有效新样本，旋转相对首样本不超过 8°；有位置时位移不超过 3 cm。稳定窗口被打断会重新收集，左右侧独立对齐。
记录人体初始朝向 R0、各机器人实际初始末端朝向 Q0，计算：

```text
D_base = C R(t) R0^T C^T
Q_target_world = B D_base B^T Q0_world
```

B 是机器人实际基座到仿真世界的旋转。局部掌轴在空间相对旋转中抵消，左右侧分别建立 Q0；不将反射矩阵转成四元数。
对齐信息写入报告，控制进程重启后重新收集初始姿态，不复用旧前臂平面标定冒充掌朝向标定。

朝向采用 SO(3) 平滑，时间常数 80 ms，目标变化上限 180°/s。单帧大于 60° 的突变先保持；连续三个新样本仍处于新估计时，通过限速接近。丢失时保持，超过 0.5 秒显示 `lost_hold`，恢复时平滑接回，不自动把新的手姿认作中性。
VR 缓存同一数据包不增加腕采样时间；缓存超过 0.5 秒或手与腕逐位冻结超过 1 秒时判无效。接收时间是本机接收新包时估计，不是头显硬件曝光时间。冻结判据保守，不能代替 SDK 提供的真实追踪置信度。

VR 腕平移相对初始位置映射，限制在半径 20 cm 内并以最多 0.3 m/s 接近目标；没有肩肘数据时使用机器人姿态参考，不虚构人体肩肘。

## IK 与诊断

显式腕朝向启用所有机器人 profile 的位姿求解。肘位置用于初值，腕位姿作为最后任务，不再被随后的肘腕纯位置调用覆盖。腕与肘目标不保证都精确满足。
位姿求解明显损害位置时，保持当前关节状态并记录 `wrist_rejection=pose_position_conflict`；非有限或越限解也拒绝。较大关节步长按整体比例缩到每次最多 0.35 rad，属于当前迭代防跳措施，不是已证明的连续时间关节速度控制。
姿态与位置不可达时可能进入保持，不能以“目标四元数存在”判定动作成功。

窗口的短 RGB 轴表示实际末端，长 RGB 轴表示目标朝向，显示在机器人腕部；相机窗口另有 `wrist=...` 状态。VR 状态变化输出到终端。
报告 `wrists` 保存初始对齐、目标/实际四元数、腕位置/朝向误差、IK 拒绝原因及物理步进后最近 1800 帧历史。无朝向目标时误差为 null，不用 0° 代替。

## 人工验收清单

先单机器人单侧、小幅动作，再检查左右手和三机器人。每个姿态停留约两秒，正常退出以保存报告。
表内 10°/30 mm 是本轮建议的停稳验收门槛，不是已经达到的实测指标；不可达姿势单独记录，不计作通过。

| 完成 | 检查项 | 操作 | 通过条件 |
|---|---|---|---|
| [ ] | 相机启动 | 保持自然掌姿直到 aligned/tracking | 不需要腕开关；不会在对齐瞬间翻腕 |
| [ ] | VR 启动 | 同上，scene=robots | 腕有效时开始跟随；没有矩阵时明确保持 |
| [ ] | 左右侧 | 分别只动左手、只动右手 | 对应侧运动，另一侧不被错误路由 |
| [ ] | 翻掌 | 肩肘腕位置尽量不动，慢转手掌 | 目标轴确实转动，机器人跟随方向正确 |
| [ ] | 俯仰与偏转 | 各方向约 15°～30° | 无轴交换、无左右侧反向误映射 |
| [ ] | 手指开合 | 尽量保持掌姿，只开合手指 | 手指正常；目标掌轴不随指尖动作明显摆动 |
| [ ] | 位置与朝向 | 小幅平移并转腕 | 两个目标都有效；记录停稳位置/角度误差 |
| [ ] | 稳态误差 | 在可达姿态停两秒 | 建议末段角误差≤10°、位置误差≤30 mm；分别记录每台每侧 |
| [ ] | 遮挡/离开追踪区 | 遮挡 1～2 秒再恢复 | hold/lost_hold，保持朝向；恢复后渐进接回 |
| [ ] | VR 停流 | 断开流或摘下头显 | 缓存不会继续累积标定，错误新鲜数据不会推动手臂 |
| [ ] | 突变 | 快速转动后恢复小幅动作 | 不出现单帧翻转；持续姿态能逐渐接近 |
| [ ] | 限位/不可达 | 逐渐接近边界 | 明确报告拒绝/误差，不输出 NaN 或假成功 |
| [ ] | 回放 | 回放本次 NPZ 和原始 VR H5 | 输入朝向、对齐和路由含义一致；旧无腕文件不会伪造身份姿态 |
| [ ] | 连续运行 | 每种输入运行至少五分钟 | 无异常退出，报告正常落盘；抖动、延迟和保持次数有记录 |

## 启动示例

相机（沿用现有环境和模型文件）：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
& $python -m teleoperation.simulation.realtime_teleop --robot g1 --side both --no-arm-fallback --report "outputs/tmp_realtime/wrist_$session.json" --dump-npz "outputs/tmp_realtime/wrist_$session.npz"
```

VR dex（先使用已有的 VR dex 标定文件，将 `$vrCalibration` 和 `$vrIp` 替换为实际值）：

```powershell
. ./devtools/dex_native_env.ps1
$vrCalibration = '替换为已有VR dex标定JSON路径'
$vrIp = '替换为Vision Pro地址或房间号'
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
& $dexPython -m teleoperation hand vr --backend dex --ip $vrIp --dex-calibration $vrCalibration --hand both --hand-mapping same --scene robots --report "outputs/tmp_realtime/vr_wrist_$session.json"
```

TransHandR VR 继续用原先有效的 checkpoint 命令，并指定 `--scene robots`；双手使用 `--hand both --hand-mapping same`。单侧源手/目标手映射也应用于腕，避免手指和手臂路由不同侧。
VR 回放将 `--ip` 替换为 `--input-h5 <原始录制路径>`，其余模型、场景与手侧参数保持一致。

## 上游来源与边界适配

| 来源 | SHA256 |
|---|---|
| `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `.venv/dex-native/Lib/site-packages/avp_stream/streamer.py` | `a8950f11f51d3d906dddbcb42c7b91862e6161c1037415be70e6fb2ae62fbf52` |
| 既有 `src/teleoperation/applications/visionpro_processing.py` | `a90c1a57e4375e8ac22eebb134f263c83267097d063c0f60b90b5c9270744e84` |

核对 SDK `streamer.py` 的 YUP2ZUP、腕矩阵变换、get_latest 形状及缓存实现。TransHandR 三帧堆叠、左右点重映射、减腕原点与模型输入处理文件保持不变；本次没有重跑数值等价实验，也没有修改 checkpoint 元数据。
新增的腕观测、SDK 形状/缓存适配、腕初始对齐、机器人执行与报告均属于独立边界与控制适配，不宣称是上游手指算法的组成部分。
