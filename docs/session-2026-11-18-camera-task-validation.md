# 2026-11-18 摄像头遥操作与任务验证日志

## 本轮目标

1. 修正 H1-2 原生手的手腕、握拳和捏合映射。
2. 让摄像头任务链路与 `run_arm_quickcheck.py` 使用同一套 IK 和手部控制效果。
3. 验证任务场景中的桌面碰撞、标定稳定性、抓取和任务判定。
4. 为后续 VR 接入保留可复用的机器人、IK、手部和任务评估接口。

## 关键提交

| 提交 | 内容 |
|---|---|
| `ffe76d4` | 按三台原装手的真实关节轴生成捏合目标；关闭桌面和落地台座对机器人的碰撞 |
| `c41d4ed` | 摄像头任务改用与 quickcheck 一致的 `paper` IK、手腕姿态和肩锚点补偿 |
| `b377265` | 将握拳从单指捏合信号中分离，四指卷曲可触发整手闭合 |
| `239954d` | 增加接触抓握辅助；增加手腕姿态 refine 和抓取几何探针 |
| `e7d78f9` | 报告任务的初始物体位姿和推动方向 |
| `7e2cc35` | GUI 中显示物体名称，并绘制方块到目标区的引导线 |
| `ce985a8` | 标定期间冻结物体；标定后先保持，再从实际腕点平滑开始跟随 |
| `21f34ef` | 成功条件改为连续成立 500 ms，并拒绝物体掉下桌面的伪成功 |

## 已完成的技术修正

### 原生手

- H1-2、GR1-T2、G1 的捏合不再统一使用“绝对值最大的关节限位”。
- 按每台机器人、每只手的真实关节轴设置捏合端点。
- H1-2 右手的食指捏合可将拇指与食指尖端带到物理接触。
- 握拳和局部捏合使用不同信号：
  - `pinch`：拇指接近食指或中指；
  - `fist`：四指同时卷曲；
  - `grasp`：物体接触后使用较浅的钩形闭合，避免继续挤压将物体弹出。

### 手臂与手腕

- 摄像头任务改用 `paper` IK，和 `run_arm_quickcheck.py` 保持一致。
- 增加手腕掌面姿态目标，不再只跟踪腕点位置。
- 增加 `refine_orientation()`，在位置解上只放开腕链修正掌面朝向。
- 使用机器人实际关节力 `200`，和已验证的 quickcheck 配置一致。
- 摄像头任务的方向口径、肘目标和肩锚点补偿与 quickcheck 对齐。

### 场景与启动稳定性

- 桌面和新增落地台座不再与机器人发生碰撞；方块、物体和地面碰撞仍然保留。
- 关闭会按帧硬抬桌上腕目标的高度保护，避免桌面附近逐帧闪动。
- 标定期间冻结机械臂、手指和所有可动物体，并清零物体速度。
- `cal=ok` 后额外保持 12 个摄像头帧，再从机器人实测腕点开始限速跟随。
- 首次跟随的腕目标不再直接跳到操作者当前目标，避免启动瞬间扫飞桌上物体。
- GUI 为任务物体增加名称标签；`pushcube` 显示方块到目标区的绿色引导线。

### 任务判定

- 任务判据必须连续成立至少 `500 ms`。
- 任意动态任务物体低于桌面 8 cm 时，直接拒绝成功。
- trial 报告增加：
  - `success_hold_steps`
  - `success_streak_max_steps`
  - `success_rejected_reason`
  - `initial_objects`
- 瞬时扶正、短暂接触或一帧误判不再直接结束 trial。

## 30 任务可评测性

H1-2 当前验证结果为 **24/30 可正式评测**。以下 6 项不进入成功率统计：

| 任务 | 原因 |
|---|---|
| `liftmug` | reset 时杯盖高度差已超过判据 |
| `opendrawer` | reset 时抽屉距离已超过判据 |
| `openmicrowave` | reset 时门已经被判为打开 |
| `breadtoaster` | 判据要求按钮进入桌面以下 |
| `rotatehearth` | 判据要求旋钮进入桌面以下 |
| `pottomatoout` | 当前实心锅无法还原“番茄在锅里” |

## 现场验证记录

| 任务 | 结果 | 说明 |
|---|---|---|
| `pushcube` | 成功，`7.97 s` | 使用物体标签和方向线后完成；需要在 500 ms 稳定成功规则下复测 |
| `pickcube` | 成功，`21.38 s` | 接触后抓握辅助成功；需要在 500 ms 稳定成功规则下复测 |
| `pickplacecube` | 未成功 | 手部有效帧 `153/1200`，最大握拳强度 `0.103`，未发生真实抓取 |
| `uprearcup` | 旧结果撤销 | 旧判据只要求瞬时直立；现场实际是杯子被碰后短暂立起，最终滚到桌下 |

现场报告：

- `outputs/task_eval/pushcube_camera_live_v4_labeled.json`
- `outputs/task_eval/pickcube_camera_grasp_assist_live.json`
- `outputs/task_eval/pickplacecube_camera_live.json`
- `outputs/task_eval/uprearcup_camera_frozen_start.json`

## 测试状态

最新全量测试：

```text
526 passed, 3 skipped, 470 subtests passed
```

## 未完成问题

1. `pushcube` 和 `pickcube` 需要用最终“连续成功 500 ms”规则重新现场确认。
2. `uprearcup` 需要真正抓起杯子、翻转并保持直立至少 500 ms。
3. 纯物理接触抓取仍不稳定；当前接触抓握辅助是仿真层的抓握锁定，不是隔空吸附，必须已有至少两个手部接触点。
4. MediaPipe 在手臂或手掌离开画面时仍会丢失手部数据；现场必须让手和手臂持续入镜。
5. `ballbimanual` 和 `potbimanual` 需要双臂摄像头策略；当前 `CameraPolicy` 仍只驱动单侧。
6. 6 个不可评测任务需要任务作者修正判据或容器几何。

## 后续任务顺序

1. 用最终稳定成功规则复测 `pushcube`、`pickcube`、`pickplacecube`、`uprearcup`。
2. 完成 Level 1 的 `balltrashcan`。
3. 进入开关、旋转和抽屉类任务。
4. 增加双臂策略后测试 `ballbimanual`、`potbimanual` 和双手机械操作任务。
5. 输出 30 任务成功率、完成时间和任务判据问题清单。

## VR 接入说明

VR 可复用：

- `SimulationEnv` 和 30 个任务场景；
- `RealtimeArmBridge` 的肩肘腕方向重定向；
- `PyBulletArmIK` 的三机器人 IK、腕姿态和 paper 求解器；
- `NativeHandAdapter` 的原装手映射、捏合、握拳和抓握辅助；
- `task_eval.run_trial()` 的 trial、成功率、完成时间和稳定判定。

VR 需要补充：

- OpenXR 输入和头显/手柄坐标系；
- HMD、肩中心、机器人基座的坐标标定；
- 可靠的肩、肘、腕三点来源；
- 左右双臂 `VRPolicy`；
- 立体渲染、手柄震动和网络延迟处理。

建议 VR 端实现与 `CameraPolicy` 相同的 callable 接口：

```text
OpenXR frame
  -> operator shoulder/elbow/wrist points
  -> RealtimeArmBridge.update()
  -> PyBulletArmIK + NativeHandAdapter
  -> action vector
  -> run_trial()
```

## 本地 Codex 历史存储

为避免 C 盘空间不足和重启后会话索引异常，本地准备了迁移工具：

- `E:\CodexTools\migrate-codex-home.cmd`
- 目标目录：`E:\CodexHome`
- 迁移后 `C:\Users\王宪雨\.codex` 为指向 E 盘的目录联接。

这部分属于本地开发环境维护，不属于遥操作源码。
