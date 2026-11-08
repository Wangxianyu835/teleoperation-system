# 审查问题最小修复 Implementation Plan

> 在当前会话顺序执行用户已确认的方案，不使用子智能体。

**Goal:** 修复 SDK 重复帧中断腕标定、整机初始穿地和统一 VR 路径遗漏 IK 锁定三个问题。

**Architecture:** 在现有状态处理、场景初始化和来源配置边界修改，不新增抽象或公共参数。

**Tech Stack:** Python、unittest、实际 URDF、PyBullet DIRECT。

**Spec:** 本次代码审查及用户确认的三项最小修复方案。

## 约束

- 基线：`9fd64e412e7127aec7462273160fee2ab7689b87`。
- 检查点：`checkpoint-20261109-0112-review-fixes`。
- 保留公开输入、窗口、结果接口；不改 NN/dex、手部归一化、TransHandR 上游算法和权重身份。
- VR 配置只在 RobotControlSession 的 visionpro 分支启用；相机保留原 IK 设置。
- 完整机器人恢复原有 0.03 m 离地间隙；不改变独立部位场景的支撑条件。

## 执行

- [x] 在 test_wrist_observations.py 增加交错重复 SDK 帧、重复不计数、超时清空/恢复和已标定保持测试；先运行失败。
- [x] wrist_orientation.py 区分相同/倒退时间戳，重复帧累计等待，0.5 s 超时；新标定样本重置等待计时。运行腕及身体跟踪回归。
- [x] 在 test_robot_scene_selection.py 用 H1-2/GR1-T2 实际 URDF，检查单目标/比较摆位和执行回放模式下地面间隙与 45×8 步的非控制关节漂移；先运行失败。
- [x] robot_part_scene.py 在 _register 后、取景和 IK 前，按初始化后的最低 AABB 将完整机器人抬到至少 0.03 m。
- [x] 在 test_robot_control_session.py 对照既有 VR 控制器，比较同初态/目标的机械臂关节命令；先运行失败。
- [x] robot_session.py 对 visionpro 显式同步 driver.lock_others 和各 adapter.ik_lock_others；相机不更改。
- [x] 定向测试、完整 unittest discover、diff 检查完成后提交；记录实际结果，不推送。

## 算法与边界

本次只修正 SDK 样本状态、仿真资源初始摆位以及已存在的 IK 约束配置。TransHandR 的输入张量和网络数值流程未改动；其本地来源路径及 SHA-256 证据仍见 docs/current-interfaces.md 的“VR 上游证据与边界适配”。没有将本次软件测试当作真实设备或上游等价性重新验收。

## 验证结果

- 修改前：腕标定两项测试失败；H1-2/GR1-T2 的默认/比较布局、控制/回放共八个子用例因初始穿地失败；VR 命令对照失败，最大角差 0.2346188 rad。相机命令对照通过。
- 修改后定向命令：`python -B -m unittest tests.test_wrist_observations tests.test_robot_scene_selection tests.test_robot_control_session tests.test_body_tracking tests.test_arm_display_stability`，28 项全部通过，14.378 秒。
- 定向验收：整机地面间隙至少 0.03 m，45×8 步后非控制关节偏移小于 0.01 rad；VR 与既有路径、相机与原默认设置的命令分别在 1e-8 rad 容差内一致。
- 主环境：`D:/Anaconda/envs/teleoperation/python.exe -B -X utf8 -m unittest discover -s tests`，612 项，608 通过、4 跳过，144.742 秒。
- 独立 dex：`.venv/dex-native/Scripts/python.exe -B -X utf8 -m unittest tests.test_normalized_dex tests.test_normalized_retargeters tests.test_unified_hand_inputs tests.test_robot_final_acceptance`，20 项全部通过、无跳过，33.444 秒。
- `git diff --check` 通过。源码仅三个文件增加 26 行，另有三个测试文件增加六项回归测试及本记录。
- 未启动真实相机、头显或实体机器人；未推送。
