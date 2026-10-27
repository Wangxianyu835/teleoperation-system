# 2026-10-27 分支整合记录

## hand_capture

来源：`6e2743c05b18878db69ab59fd0a4ab5b57e1b3fe`。合并前主线：`fde0c9ccb77ba3e5bc151107af16010e13138e5a`。
检查点：`checkpoint-20261027-0952-merge-capture-arm`。

通过合并提交保留分支历史。以下路径适配用于保留当前生产包、CLI 和坐标契约：

| 上游路径 | 当前落点或处理 |
|---|---|
| MediaPipe/ 的四个 Python 工具 | 保留独立 normalized 采集及回放；模型改用 datasets/，结果改用 outputs/hand_capture/；补充坐标和时间声明，控制台文本改为 GBK 可编码字符 |
| MediaPipe/*.task、requirements.txt | 不重复提交模型二进制和空依赖文件；使用 datasets/ 中的模型及 requirements-retargeting-camera.txt |
| scripts/check_native_hand_motion.py、render_hand_snapshots.py、make_twohand_raw_sample.py | 迁入 devtools/，改用当前机器人加载器、后处理及显式 18 维到 17 自由度映射 |
| scripts/run_retargeting_pipeline.py | 迁入 devtools/，调用当前 teleoperation hand / tools CLI；显式输入缺失时报错，校验失败停止后续流程 |
| demo_hands_three_robots.py | 转发至 tools show-all-hands |
| teleop/native_hand.py、scripts/replay_hand_native.py | 合入 link_extent 和按手部几何选取特写视角；保留当前动作类型、精度与时间戳回放接口 |
| scripts/verify_hand_pipeline.py | 任意录制不再强制命中参考数据的第 535 帧；参考文件继续执行原断言 |
| input_adapters/hand_keypoints.py | 不恢复旧根目录；其数据流职责由当前 contracts、retargeting 和 inputs 实现 |
| scripts/check_no_c_drive.py、env_e_drive_cache.ps1、preflight.ps1 | 保留当前 devtools/ 中的已有版本 |
| 旧 README、环境/入门/项目文档、规则文件、依赖约束 | 保留当前目录与环境说明，将新用法补充到中文 README；不采用旧的宽松 NumPy 依赖约束 |
| tests/conftest.py | 不加入针对已退役 test_dual_arm.py 的 pytest 跳过钩子；保留当前 unittest 资产检查 |
| .gitignore、LICENSE | 合入临时实验目录忽略规则及上游 Apache-2.0 许可证文本；第三方资产许可仍以各资产文件为准 |

原始数据、模型权重和输出之间的坐标声明必须匹配。旧归一化身体点不是 Pose world 米制手臂输入。

### 上游文件 SHA256

| 路径 | SHA256 |
|---|---|
| `MediaPipe/body capture media.py` | `6e4c5b9a454335055a7dbbba53a40fe02f7a8e35f59821ee99a0d9a43a88029b` |
| `MediaPipe/body picture.py` | `c21fde402ea9a65fef55f0570b7426b6171ec09e2843d1305bffc6a66b0a4b7f` |
| `MediaPipe/hand capture media.py` | `d38d3f3059cbd26de193d21175dfb9bdaa9168589c1e077646c2178a4a9872b8` |
| `MediaPipe/read_hand_xyz.py` | `f086f1b0376b5cbf883b4cc26a89c109d3f551c5fd5309beea582b55d9b0e559` |
| `scripts/check_native_hand_motion.py` | `945ab60af93a511116ca2bfe4d5a2dc23c05de31cf1ed42344abf2d584cbb15c` |
| `scripts/render_hand_snapshots.py` | `c9b9326c1c7d255dbb2455a22b0a5a8e30b765ab8b2b1d9f57f79601f6a0127f` |
| `scripts/make_twohand_raw_sample.py` | `cf4259a63a7d770c1f992719fa3efd7f4279be4a687d12e1ac901dd94d0c8857` |
| `scripts/run_retargeting_pipeline.py` | `99143555eabdee99d75d43042c8024653dbccef5001c2a3de66c147617407bfe` |

## 验证基线

合并前完整 unittest：344 项，1 项跳过，退出码 0。四条 CLI 冒烟均退出 0。
完整 preflight 返回 1，唯一失败类别为已有 C 盘缓存残留（pip 约 5.2 GB，另有 HuggingFace 残留）。本次进程的七个缓存和临时目录变量全部指向项目 D 盘，未清理或搬移用户原有缓存。
日志：`outputs/tmp_merge_capture_arm/baseline_preflight.log`。

后续双臂整合与最终验证结果见下文。

hand_capture 合并验证：`python -m unittest discover -s tests` 运行 348 项，1 项跳过，无失败，退出码 0。摄像头/录制/原装手专项 24 项通过。三机器人原装手快照使用已录制摄像头角度生成，退出码 0。

## cs/multi-robot-arm

来源：`04e8660dfb2f707178ef28c34769f2d3b06c26ce`，分支独有 13 个提交。
整合前提交：`eb147c680cd6f9e71af45d2dc311b835e2589cf6`。
检查点：`checkpoint-20261027-1003-merge-multi-robot-arm`。

合入方向/分段/腕位移三种映射、肘位置任务与肘角参考、逐机器人方向参考、IK 关节范围约束、解分支恢复、方向跳变保护、目标变化速率限制、成对目标保护，以及对应探针与测试。标准实时入口默认 posture / track / 腕外部姿态 off；肘角参考默认 off，专用启动器和 README 示例显式选 operator。

### 当前主线的兼容处理

- 所有 IK 临时重置均在位置/速度快照与 finally 恢复块内，包含肘任务及恢复异常路径。求解不再直接改变仿真状态。
- 静态段刚性探针与上游测试改为显式摆放返回的候选解后测量，避免把“求解后状态恢复”误判为零残差或无运动。
- `rest_poses=current` 延续本地主线含义：读取本帧实际关节角；`live` 是别名。上游旧 current（加载时姿态）改用 `loaded` 显式选择，零姿态使用 `zero`。
- 保留腕姿态不可达时的位置解降级、按机器人分发姿态，以及 `--arm-gain` 的机器人名称与有限正数解析。
- 保留实时 SimulationStepClock、暂停补偿上限、旧 NPZ 固定物理步数、手指命令防护及显示适配。
- 旧 NPZ 的新方向护栏和限速使用 `sim_steps/240` 秒，不使用主机处理耗时，也不猜测旧时间戳单位。真人输入继续使用实测循环间隔。
- 保留原诊断字段，增补实际 IK 目标及其误差字段。测试同时比较诊断开关前后的目标、命令、关节状态、IK 次数与物理步数，并模拟不同主机处理间隔。
- G1 的既有右侧 0.06 m 位移行程保留；posture 方向模式不使用旧逐轴位移行程夹取。
- Windows 启动器优先使用本机正式 Conda 解释器，缺失时才回落到上游 E 盘解释器；未设置的缓存变量使用项目 outputs/tmp_realtime/cache，已有非 C 盘设置保留。
- 摄像头手部与 VR 数值预处理、模型权重、checkpoint 元数据均未改写。导入的是摄像头双臂和工具边界。

### 合成场景与证据边界

第一轮原合成输入仍有两项降级覆盖检查失败，因为旧场景只改变手掌，而合并后的腕姿态由相对肘平面生成。场景 revision=2 保持肩、腕位置不变，将肘平面旋转 180 度，直接激励当前姿态路径。增加测试验证这一输入几何；没有降低空间阈值或取消覆盖检查。

最终 chain 路线验收：30/13 Hz 各 89 项检查全部通过（退出码 0）。H1 右臂 settle 平均误差为 0.0343/0.0571 mm；降级场景平均误差为 0.0053/0.0117 mm。后者的输入已修订，不能与旧手掌场景直接逐数比较。非驱动关节最大漂移为 0.0016/0.0017 rad。

posture 默认路线另运行 402 帧、六条手臂的完整合成回放，无 IK 执行异常，生成 2,412 行诊断 CSV；这仅证明入口和诊断可运行，不冒充同一空间阈值验收。真人摄像头效果尚待复测。

### 上游双臂源文件 SHA256

| 路径 | SHA256 |
|---|---|
| `src/teleoperation/simulation/arm_ik.py` | `290ebc771be9eea8430ad6585ea777203c49f8a1cb7cf059f2eb98a5d29e9fcc` |
| `src/teleoperation/simulation/arm_realtime.py` | `7cfd3020e495fe2d3aa95ac5e6fa0036d75ed62b90f38d0c7d53cb37bf66af41` |
| `src/teleoperation/simulation/arm_scene_driver.py` | `20735e09914dd4b3e31f37b404690162d5096733faf6cbea86826501f2070a8d` |
| `src/teleoperation/simulation/arm_profiles.py` | `049b53f7366e46845f003030fa85a3dfe22ddf38c228ca5cf58e4ed601282d42` |
| `src/teleoperation/simulation/realtime_teleop.py` | `f0ea3ee271f13d3dd7ab9e5aa2a37a44791643e78aaefa201cb82a6937101bc3` |

## 最终验证

| 检查 | 结果 | 退出码 |
|---|---|---:|
| `python -m unittest discover -s tests -v`（由最终 preflight 调用） | 448 项，1 项跳过，无失败；修改前为 344 项、1 项跳过 | 0 |
| `python -m teleoperation --help` | CLI 加载正常 | 0 |
| `python -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render` | 无头动作回放完成 | 0 |
| `python -m teleoperation tools check-environment` | 环境及端到端检查完成 | 0 |
| `python -m teleoperation tools verify-hand-pipeline` | 手部链路检查完成 | 0 |
| `python devtools/verify_arm_diagnostics.py --output-dir outputs/tmp_merge_capture_arm/acceptance_final` | 30/13 Hz 各 89 项检查通过 | 0 |
| `python devtools/verify_arm_wing.py --report outputs/tmp_merge_capture_arm/wing.json` | 六个机器人/手侧组合的目标运动检查通过 | 0 |
| `python devtools/probe_arm_segment_rigidity.py --json outputs/tmp_merge_capture_arm/rigidity.json` | 三机器人段刚性与候选解帧口径探针完成 | 0 |
| `python devtools/run_realtime_arm_elbow_follow_windows.py --check --headless --no-preview` | 解释器、依赖、入口及 Pose 资源检查完成；摄像头未读到帧 | 0 |
| 两套 GBK 检查、架构边界、暂存资产检查 | 全部通过 | 0 |
| README PowerShell 语法与链接检查 | 14 个命令块、23 处链接及锚点有效 | 0 |
| 完整 preflight | 唯一失败类别仍是合并前已有 C 盘缓存残留；其余检查通过 | 1 |

最终日志：`outputs/tmp_merge_capture_arm/final_preflight.log`。精确自检白名单由 `git diff --name-only eb147c6` 生成，仅用于本次用户明确授权的合并文件，未修改用户级 Git 或环境配置。

测试覆盖既有摄像头、world 手部、VR、原装手、共享接口以及新增方向/肘部逻辑。离线结果不替代真人摄像头或实体机器人验收。本次未推送远程。
