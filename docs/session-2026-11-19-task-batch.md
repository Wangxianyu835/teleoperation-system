# 2026-11-19 任务批量验证日志

## 已完成

### 最终稳定成功判定

- 成功判据必须连续成立至少 `500 ms`。
- 可动物体低于桌面 8 cm 时直接拒绝成功。
- trial 报告增加 `success_streak_max_steps` 和
  `success_rejected_reason`。
- 标定和启动等待期间冻结全部可动物体并清零速度。
- 标定完成后机械臂和手指先用 `2000 N` 强保持锁定，从实际腕点开始限速跟随。
- 操作者腕目标必须相对标定中性目标移动至少 `50 mm`，并连续保持 `3` 帧，
  才打开启动门；短时抖动和第一帧目标跳变不会触发机械臂动作。
- 打开启动门前不更新手指、不启用抓取辅助、不恢复任务物体物理。
- 腕目标小于 `2 mm` 的变化直接保持上一命令。
- 报告记录 `startup_arm_drift_max_mm`、`startup_target_motion_max_mm`、
  `startup_motion_frames` 和 `tracking_started`，用于区分机械臂漂移和摄像头噪声。

### 摄像头任务验证

| 任务 | 结果 | 说明 |
|---|---|---|
| `pushcube` | 最终规则下未成功 | 最近一次方块被推过头并掉下桌，未误记成功 |
| `pickcube` | 成功，`33.75 s` | 连续成功 `120` 个物理步，方块最终高度 `0.769 m` |
| `pickplacecube` | 成功，`17.46 s` | 方块放入固定盘子；此前“抓住但未放置”的报告保留为失败证据 |
| `uprearcup` | 旧成功撤销 | 实际上是碰倒后短暂立正，杯子最终掉到桌下 |

报告：

- `outputs/task_eval/pickcube_camera_stable_20261119.json`
- `outputs/task_eval/pushcube_camera_stable_20261119.json`
- `outputs/task_eval/pickplacecube_camera_startgate_v2_20261119.json`

## 肘部方向根因与修复

现场反馈：摄像头任务中机器人肘部弯曲方向与操作者相反。

根因：`paper` IK 分支虽然收到了方向映射生成的肘位置目标，但只用于普通
`PyBulletArmIK` 路径；paper 求解器此前只优化腕位置和腕姿态，肘目标实际上被忽略。

修复：

- `PaperArmIK` 的目标函数同时包含腕点位置、肘点位置、腕姿态、平滑项和
  正则项。
- 腕位置权重仍为 `50`，肘位置权重设为 `80`，姿态权重保持 `1`。
- 腕和肘位置通过一次 `getLinkStates()` FK 查询共同读取。
- `PyBulletArmIK.solve(..., elbow=...)` 现在会把肘目标真正传给 paper 求解器。
- 新增回归测试，验证加入肘目标后肘误差必须明显下降。

同一段 500 帧操作录像对比：

| 指标（H1-2 右臂） | 修复前 | 修复后 |
|---|---:|---:|
| 上臂方向误差中位数 | `10.42°` | `2.09°` |
| 前臂方向误差中位数 | `9.20°` | `2.80°` |
| 肘位置误差中位数 | `33.58 mm` | `6.74 mm` |

三台机器人均有同量级改善。

`pickplacecube` 修复后的现场报告进一步显示：

- 上臂方向误差中位数 `2.59°`
- 前臂方向误差中位数 `4.08°`
- 肘位置误差中位数 `8.37 mm`
- 腕位置误差中位数 `5.45 mm`
- 失败原因是手部只有 `182` 个有效帧，未形成抓取

## 场景离胸前过近

摄像头任务原先把整组物体放在机器人肩关节正前方的身体中线附近。现场又反馈
桌子和物体离机器人胸口过近。最终布局为：

- 右手控制时，任务物体整体向机器人右侧偏移 `6 cm`。
- 左手控制时对称地偏向左侧。
- 桌面近沿前移到 `x=0.29 m`，桌面顶面从肩下 `0.12 m` 降到
  `0.18 m`，为手臂、手和桌面在视觉上留出约 `6 cm` 的额外间隔。
- 物体组质心位于肩前 `0.38 m`，侧向布局半径 `0.15 m`。
- `pickplacecube` 单独使用“右侧深抓取区 + 对角放置区”：
  方块约 `[0.363, -0.100] m`，盘子约 `[0.389, 0.060] m`。
- 方块中心离桌沿约 `7.3 cm`，不再位于机器人胸前；两物体间隔约 `15.2 cm`。
- 桌面降低后，方块和盘子 IK 残差分别为 `2.2 mm`、`1.6 mm`。

`0.42 m / 0.16 m` 的布局虽然能让桌边达到 `0.29 m`，但盘子会在 `0.30 m`
局部搜索半径内仍差 `27 mm`，不可达。参数扫描后改为 `0.38 m / 0.15 m`：

- `pickplacecube` 方块 IK 残差 `9.7 mm`
- 盘子 IK 残差 `0.6 mm`
- 桌边 `0.290 m`
- 方块和盘子均稳定留在桌面

回归测试现在同时检查桌边距离、桌面支撑和 `ik_reach.ok`，不能只用
“物体还没掉”冒充场景可用。

现场继续反馈方块贴桌沿、机器人靠近盘子时会把盘子推走。最终修复：

- 方块向桌面内部和机器人右侧移，离桌沿从约 `2.4 cm` 增加到 `7.4 cm`。
- `pickplacecube` 的抓取辅助明确排除 `plate`，只允许吸附方块。
- 盘子在该相机任务中设为固定目标（`mass=0`、摩擦 `1.2`）；
  方块仍能正常落在盘子上，但盘子不会被机械手碰飞。
- 报告新增 `camera_pickplace_plate_fixed`、`camera_pickplace_spacing_m` 和
  `grasp_assist_excluded`，可以复查目标是否固定、布局距离是否符合预期。

## 启动自动抖动

现场反馈：操作者尚未动手时，机械手会先抖两下，把方块推近并继续抖动，最后
把盘子碰下桌。

根因由两个独立问题组成：

1. 相机标定结束后，机械臂仍以普通 `200 N` 预算保持初始姿势；接触和重力可能
   造成轻微下沉，随后切换为摄像头控制时出现一次弹动。
2. 原启动保持只按“标定后 12 帧”计时，不要求操作者真正移动。相机目标的小幅
   跳变也能开门并恢复物体物理。

修复后，启动安全条件必须同时满足：标定完成、保持帧结束、目标相对标定中性
位置移动至少 `50 mm`、该位移连续确认 `3` 帧。条件满足前运动和物体物理都
保持冻结。最新现场报告中 `startup_motion_frames=3`、`tracking_started=true`，
说明启动门确实由连续目标运动打开。

## `uprearcup` 杯子防滚

原始杯子是光滑圆柱，轻微扰动后会持续滚动。最初通过高滚动摩擦和角阻尼
处理，但物理上会把杯子锁在任意倾斜姿态，甚至出现单点接触也不倒下的假象。
该方案已撤销。

当前方案：

- 使用 `createCollisionShapeArray` 把杯体和一条纵向凸筋组合成**同一个刚体**。
- 不再创建独立长方体，也不使用固定约束。
- 杯体仍可在桌面滚动、倒下和被夹起，但凸筋会限制持续直线滚动。
- 物理参数使用 `rollingFriction=0`、`angularDamping=0.7`、
  `linearDamping=0.2`，不靠滚动摩擦锁死姿态。
- 杯子质量使用更接近陶瓷杯的 `0.20 kg`。
- 相机任务关闭 44 个非手部机器人连杆与杯子的碰撞，只保留右手手部接触；
  肩、肘、前臂扫过杯子时不再把它撞飞。
- 未抓取时限制水平速度 `0.30 m/s` 和角速度 `1.5 rad/s`；竖直重力和旋转
  不受限，杯子仍能自然倒下。
- `uprearcup` 桌沿固定在 `x=0.08 m`，给低桌布局下 10 cm 长杯体保留支撑。

场景回归检查包括：

- 轻微扰动后杯子不会持续滚出桌面。
- 45° 斜立的杯子在扰动后必须自然倒下，不能被摩擦固定。
- 杯子近端到桌沿至少保留 `4 cm` 支撑。

最新现场报告
`outputs/task_eval/uprearcup_camera_handonly_20261119.json`：

- 任务仍未成功，抓取辅助反复吸附/释放。
- 杯子从 `[0.232, -0.060, 0.930] m` 被接触后移动到
  `[0.240, -0.201, 0.930] m`，只移动约 `14.1 cm`。
- 杯子始终保持在桌面高度 `0.930 m`，没有弹飞或滚下桌。
- 说明本轮修复的是滚动/支撑链路，不能把未转正的任务算成功。

## TeleOpBench 原仓库抓取方式审计

参考路径：

`F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\TeleOpBench-main`

关键文件：

- `teleop/robot_control/camera_ik_h1_2_media.py`
- `teleop/receiver_wrapper/camera_wrapper_h1_2_media.py`
- `teleop/robot_control/robot_hand_unitree.py`
- `assets/unitree_hand/unitree_dex3.yml`

原仓库没有 `createConstraint`、`getContactPoints` 或固定抓取逻辑。相机流程只做：

1. SMPL-X/MediaPipe 身体与手部关键点；
2. 手臂 IK 输出 7 个手臂关节；
3. Dex Retargeting 输出原装手关节目标；
4. 把手臂和手指目标写入 38 维动作；
5. 真实手电机按 `q_target` 执行。

`unitree_dex3.yml` 使用拇指、食指、中指指尖，`target_link_human_indices`
为 `[[0,0,0], [4,9,14]]`，`scaling_factor=1.0`，低通系数 `0.2`。
`robot_hand_unitree.py` 直接发送 7 个 Unitree Dex3 关节目标，没有吸附。

结论：论文的抓取是**纯物理接触抓取**。本项目的 `grasp_assist` 属于为仿真
稳定性增加的辅助，不应成为抓取主路径。后续应先保证 H1-2 原生手关节闭合并
形成真实接触，再决定是否保留辅助。

为减少辅助误判，当前已加入：

- 连续接触约 2 个摄像头帧后才允许辅助吸附；
- 手到物体中心超过 `12 cm` 时拒绝吸附；
- 只有手部数据有效、无捏合分类且连续约 8 个摄像头帧张开后才释放；
- 手部短暂丢帧或分类跳变时继续保持当前抓取；
- 报告记录距离拒绝次数、候选帧和释放确认参数。

纯物理抓取现场测试仍未成功，关闭辅助后没有形成可维持的杯体抓取。下一步应做
离线抓取台架，直接比较 H1-2 原生 12 关节目标、指尖位置与实际接触，而不是继续
调吸附阈值。

随后通过离线接触台架确认：

- 闭合手指可以产生真实接触，接触点数量和法向力不是零。
- 手部三指自然捏合中心约为 `[0.45, -0.16, 0.93] m`。
- 通用场景原来把杯子放在约 `[0.232, -0.060] m`，位于手掌后方，手指闭合时
  无法形成相对夹持面。
- `uprearcup` 现已把杯子放到右手预抓取中心，并在检测到握拳和至少两个手部
  接触点前保持静止；接触确认后才恢复杯子动态。

`outputs/task_eval/uprearcup_camera_physicalpregrasp_20261119.json` 中曾出现
`success=true`，但操作者确认是用鼠标在 PyBullet GUI 中拖正杯子，并非机器人
完成。该结果**作废，不计成功率**。

评测窗口现在强制关闭 PyBullet 鼠标拾取：

```python
p.configureDebugVisualizer(
    p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=env.client)
```

每份 trial 报告增加 `gui_mouse_picking_enabled=false`，防止鼠标拖动污染后续
结果。

2026-11-20 的持续演示进一步验证了纯物理抓取：

- `pickplacecube` 在 `232.05 s` 达到成功；
- `grasp_attach_count=0`、`grasp_release_count=0`，整个成功过程没有使用吸附约束；
- `gui_mouse_picking_enabled=false`，没有鼠标拖动物体；
- 成功连续保持 `1447` 个物理步，超过要求的 `120` 步；
- 窗口按 `--demo-keep-open` 继续运行，直到操作者手动关闭。

报告：`outputs/task_eval/pickplacecube_demo_v2_20261120.json`。

针对现场“抓起来但没感觉”的体验，摄像头预览新增实时抓取反馈：

- 第二行显示 `grasp=none/contact/attached`；
- 同时显示物体名、手部接触点数量和接触法向力；
- 接触开始/结束、辅助吸附和释放都会打印到终端；
- 报告新增 `camera.grasp_feedback`，可复查最后一帧的物理接触状态。

随后现场发现方块只有单点接触后滑落。`pickplacecube` 现已采用与
`uprearcup` 相同的预抓取思路：

- 方块放在右手掌、拇指、食指和中指的实际捏合中心；
- 桌沿延伸到该位置下方，保证方块完整受桌面支撑；
- 握拳并形成至少两个真实手部接触点前，方块保持静止；
- 接触确认后才恢复动态，再允许抬起和搬运；
- 盘子仍保持远侧固定放置目标。

该布局的回归测试检查桌面支撑、预抓取位置、IK 可达和相机抓取辅助确认。

`balltrashcan` 已按同样方法接入预抓取：

- 球放在右手掌、拇指和食指/中指的实际捏合中心；
- 桌沿延伸以完整支撑球体；
- 垃圾桶设为固定目标，避免被机器人碰走；
- 球在形成至少两个真实手部接触点前保持静止；
- 接触确认后才恢复动态，开始搬运。
- 球增加 `rollingFriction=0.12`、`spinningFriction=0.03`、
  `linearDamping=0.30`、`angularDamping=1.00`，轻碰后不会无限滚动。
- 原任务中的垃圾桶其实是实心圆柱，无法装球。相机场景现在把它替换为
  真正的空心容器：8 段桶壁 + 底板上下一体的静态复合刚体。
- 回归测试会从桶口上方释放球，确认球能进入容器并满足成功判据。

PyBullet 鼠标拾取默认仍关闭，避免污染正式成功率。交互演示可显式开启：

```powershell
--allow-gui-mouse-picking
```

报告会写 `gui_mouse_picking_enabled=true`；这类结果只能用于演示，不能计入
正式任务成功率。

持续演示入口：

```powershell
E:\python3.11.7\python.exe devtools\task_eval.py `
  --task balltrashcan --policy camera --side right `
  --camera-hand-force 600 --max-sim-s 86400 --max-wall-s 86400 `
  --demo-keep-open --render --preview
```

## 桌面高度

现场反馈桌面高度恰好与上臂和手腕连线重合，人在桌面上方操作时很难从画面中
区分机械臂和桌沿。

高度对比：

| 布局 | 桌顶高度 | 相对肩关节 |
|---|---:|---:|
| 上一版 | `1.174 m` | 肩下 `0.12 m` |
| 通用当前版 | `1.114 m` | 肩下 `0.18 m` |
| `uprearcup` 8/10 版 | `0.890 m` | 机器人下移 `0.03 m` 后的肩下 `0.374 m` |

针对 `uprearcup` 继续尝试过桌面 `0.70 m` 的 6/10 方案和机器人下移
`0.12 m` 的 `0.89 m` 方案。两者静态 IK 虽可通过，但会迫使杯子退到
桌沿以内，或在动态跟随中出现约 `200 mm` 误差，因此没有采用。

最终采用 `uprearcup` 8/10 方案：桌面 `0.89 m`、机器人下移 `3 cm`、
桌沿 `0.08 m`。场景回归确认：

- 杯子近端离桌沿至少 `4 cm`
- 杯子 `ik_reach.ok=true`，残差约 `4.1 mm`
- 实际腕点跟踪误差中位数约 `11.3 mm`
- 启动冻结逻辑不变

另外修正了 IK 可达检查的状态污染：以前连续测多个物体时，前一个 IK 解会改变
后一个物体的残差；现在每个点测量前都恢复机器人初始关节状态，报告中的残差
可以互相比较。

## 测试

最新全量回归：

```text
534 passed, 3 skipped, 470 subtests passed

`check_gbk_safe.py --strict` 通过。
```

## 当前状态

- `pickcube` 已在最终稳定规则下成功。
- `pushcube` 需要在目标区停稳，不能推过头。
- `pickplacecube` 已在最终场景下成功，完成时间 `17.46 s`。
- Level 1 剩余：`uprearcup`、`balltrashcan`。
- 30 个任务中 H1-2 有 24 个可正式评测，6 个为原任务判据或场景问题。

## 最新现场补充

- `--demo-keep-open` 演示模式已加入：任务成功后不再结束 trial，而是继续运行
  到手动关闭窗口。2026-11-20 使用 `pickplacecube + camera + preview` 打开
  持续抓取演示，方便队友现场观看。

- `pickplacecube`（`outputs/task_eval/pickplacecube_camera_elbowfix_v2_20261119.json`）：
  未成功；肘方向指标已经正常，但手部有效帧不足，未发生抓取。
- `pickplacecube` 启动门与 `0.29 m` 桌边现场验证
  （`outputs/task_eval/pickplacecube_camera_startgate_v2_20261119.json`）：
  未成功；机械手曾抓住方块并检测到松手，但最终方块位于
  `[0.393, 0.322] m`、盘子位于 `[0.566, 0.569] m`，没有完成“放进盘子”。
  报告确认方块和盘子仍留在桌面，不能把抓取或松手误报为任务成功。
- `pickplacecube` 最终现场成功
  （`outputs/task_eval/pickplacecube_camera_fixedplate_20261119.json`）：
  `17.46 s`；方块初始 `[0.364, -0.177, 1.204] m`，最终
  `[0.408, -0.057, 1.213] m`；盘子全程保持
  `[0.414, -0.017, 1.179] m`。抓取辅助只吸附一次方块，
  盘子没有被吸附或推离。
- 降低桌面后再次现场成功
  （`outputs/task_eval/pickplacecube_camera_lowtable_live_20261119.json`）：
  `24.63 s`；桌面高度 `1.114 m`，方块初始
  `[0.363, -0.100, 1.144] m`，最终 `[0.406, 0.106, 1.166] m`；
  固定盘子全程保持 `[0.389, 0.060, 1.119] m`，说明降低桌面没有破坏
  抓取和放置能力。
- `uprearcup`：等待很久仍没有右手入镜，已停止空跑；这次没有有效测试数据，
  不能计为成功或失败样本。

## 2026-11-21 手指接触闭合修复

现场现象：现实中的拇指与食指已经贴合时，机器人原装手的两个指尖仍有明显
距离；手掌张开时，机器人手又会偶尔停在半闭合状态。离线对 URDF 碰撞几何的
检查已经确认，H1-2、GR1-T2 和 G1 的手指与拇指在关节限位内可以真正接触，
所以问题不在机器人手模型本身。

根因在 MediaPipe 输入：

- 指尖贴合会遮挡关键点，world 三维坐标可能把已经接触的两点算成约半个掌宽；
- world 落点还会偶尔把张开的手判成 `index` 捏合或 `fist`；
- 原来的闭合强度从任意非零距离开始线性增加，张开的手因此也常得到
  `0.37-0.50` 的闭合量；
- 单帧分类跳变会直接改变目标关节，表现为手指反复张开和闭合。

本次改动：

1. `native_hand.py` 增加二维与三维联合测量。同一帧有有效二维手部关键点时，
   由二维落点裁决食指/中指捏合，world 结果保留为兜底和诊断数据。
2. 捏合触发阈值改为 `0.58` 掌尺度，`0.30` 掌尺度以内直接到达原装手的
   接触端点，避免“看起来合了但指尖没碰到”。
3. 增加约 `0.4 s`（8 个相机帧）的闭合保持。指尖短暂被遮挡时不再立刻
   松开，持续张开后正常释放。
4. `run_arm_quickcheck.py` 对应的三机器人实时链路和 `task_eval.py`
   的正式任务链路同时接入该逻辑。
5. 摄像头 HUD 和任务报告记录稳定信号、原始信号、来源、二维及三维测量，
   可以直接判断问题是否仍在识别层。

回归结果：

```text
Ran 543 tests in 106.496 s
OK (skipped=3)
GBK strict check: OK
```

现场打开的命令：

```powershell
E:\python3.11.7\python.exe devtools\run_arm_quickcheck.py --frames 0
```

窗口每侧数据行的 `pinch=` 是保持后的闭合信号，`raw=` 是当前帧原始信号，
末尾 `@image`、`@image-open`、`@world` 或 `@world-open` 表示裁决来源。

## 2026-11-21 30 任务可评测性收口

重新执行：

```powershell
E:\python3.11.7\python.exe devtools\task_eval.py --verify --robot all `
  --json outputs\task_eval\verify_all_20261121_ready.json
```

最终结果：

```text
[OK] 90/90 tasks evaluable, 0 excluded
```

这里的 `90` 是 `30 个任务 x 3 台机器人`。它表示每个组合都满足：
场景稳定、reset 不假成功、判据可触发、物体在可达范围内。它**还不是**机器人
实际完成任务的成功率。

本轮修复内容：

- `liftmug`：杯盖静置时本来就比杯心高 `0.055 m`，旧阈值 `0.05 m` 导致开局即成功。
  改为必须额外抬起 `2 cm`。
- `openmicrowave`：关门状态离本体中心就有 `0.16 m`，旧阈值 `0.12 m` 开局即成功。
  门改成不与本体互穿，并要求拉离 `0.20 m`。
- `opendrawer`：旧抽屉厚 `0.28 m`，比柜体还深，初始必然互穿。改成柜面薄抽屉，
  关闭位置距柜心 `0.165 m`，拉出到 `0.19 m` 以上才算成功。
- `breadtoaster`：按钮改为烤面包机前面的可推动按钮；旧的“按到世界坐标 z 以下”
  会要求把按钮按进桌面。现在要求向机身内推动至少约 `3 cm`。
- `rotatehearth`：旧判据要求旋钮向下进入桌面 `1 cm`。现在实际遥操作要求
  “旋转超过 30 度且机器人仍与旋钮接触”；无机器人的判据审计只验证旋转。
- `pottomatoout`：实心圆柱锅改为八段桶壁加桶底的真正空心锅。番茄现在可以
  放在锅内，场景层明确识别容器与内容物，不会把番茄自动弹出去。
- `potbimanual`：锅从“世界坐标高于 0.58 m”改为“高于桌面 15 cm”；同时把两个
  手柄移到锅体外侧，避免 G1 下沉底座后与锅体互相穿插。G1 不再需要
  `sink` 场景模式。
- 场景层对高于/低于可达球的物体不再执行 `sqrt(max(1e-6, ...))` 的伪夹取，
  避免数值保护退化成几十米的物体飞移。

截止本轮，30 个任务已没有 `CRITERIA_INVALID`、`CRITERIA_ABSOLUTE` 或
`SCENE_UNSUPPORTED` 排除项。必须继续区分：

- **可评测**：本轮完成；
- **纯物理成功率**：下一阶段采集；
- **assisted / manual-demo**：只能作为辅助结果，不能混入论文成功率。

## 2026-11-23 MediaPipe 抓握几何与接触即停

现场持续反馈“现实中已经捏住，机器人仍然夹不住方块”。本轮先测量 H1-2 原装手
不同闭合强度下的真实拇指-食指碰撞面距离：

| 闭合强度 | 最近距离 |
|---:|---:|
| `0.0` | `85.2 mm` |
| `0.4` | `55.5 mm` |
| `0.6` | `33.1 mm` |
| `0.8` | `13.6 mm` |
| `1.0` | `4.7 mm` |

原任务方块边长 `60 mm`，已经接近张开状态的两指跨度；完全闭合时又会继续向
物体施力，容易把方块从指尖挤出。针对 `pickcube` / `pickplacecube` 做以下调整：

- 方块从 `60 mm` 缩为 `40 mm`，质量改为 `0.04 kg`；
- 摩擦从 `0.5` 提高到 `1.2`，并增加基础阻尼；
- 方块在摄像头任务启动时预先移动到原装手真实捏合中心，不再要求操作者只靠
  手臂运动把物体与指间中心对齐；
- `pickcube` 的抬起高度改为相对桌面上方 `0.18 m`，避免为绝对高度把机器人
  底座压进桌子；
- 增加“接触即停”控制：只有拇指连杆和食指/中指/无名指/小指连杆同时接触
  同一物体时，才把当前手指关节位置作为保持目标，阻止手指继续闭合；
- 该控制不创建手-物体约束，`grasp_attach_count` 仍为 `0`，物体继续完全受
  接触与摩擦支配；
- 持续松开手后恢复相机控制，不会永久锁死。

手动演示还修正了一个鼠标问题：以前 `--allow-gui-mouse-picking` 已经开启，
但启动阶段把物体质量冻结成 `0`，PyBullet 无法用鼠标拖动。现在演示模式会自动
保持物体为动态刚体；正式评测仍然禁用鼠标并保留启动冻结门。

回归结果：

```text
Ran 549 tests in 88.632 s
OK (skipped=3)
GBK strict check: OK
```

FFG 手套分支本轮只完成代码评审和独立离线验证，没有接入 MediaPipe 主链路。
其位置是后续第三种输入：MediaPipe -> VR -> FFG；当前继续以 MediaPipe 做任务。

## 2026-11-23 队友 Dex/原生手后端合并

队友提交 `e0eb35ff41e287782bef19512cda7c469d45e44c` 是一个真实 merge commit，
第一父为队友分支 `8c7dab4`，第二父为本方主线 `49d54b4`。已快进合并。

本轮主要新增：

- 原生手规格层：从实际 H1-2 / GR1-T2 / G1 URDF 提取关节、限度、mimic 关系、
  指尖 frame 和手掌几何；
- `NativeHandModel` 与原生手独立坐标展开，避免继续把 L21 17 维角度当作
  原装手真值；
- `dex_retargeting 0.4.6` 源码内置，并增加 `NormalizedDexRetargeter`；
- Dex 使用真实手部目标点做向量优化，不再依赖旧的手工 pinch endpoint 作为唯一闭合来源；
- 统一的 `HandJointResult` / `HandJointCommand`、规范哈希、限位检查和执行器；
- MediaPipe、Vision Pro、VR、训练、HDF5 录制和 Robot Part 场景的统一入口；
- 新的 `teleoperation robot control` / `hand control` CLI 路径。

Windows 依赖不能直接使用普通 pip 安装 Pinocchio。已建立独立环境：

```text
F:\conda-envs\teleop-dex
```

其中固定：

```text
numpy==1.26.4
nlopt==2.7.1
pinocchio==3.2.0
torch==2.15.0+cpu
```

micromamba 位于：

```text
F:\tools\micromamba\Library\bin\micromamba.exe
```

该环境避免破坏 E 盘现有的 MediaPipe/PyBullet 解释器。依赖安装后验证：

```text
Dex / native hand / unified input / VR subset:
41 passed, 2 skipped

Full merged tree:
640 passed, 5 skipped, 554 subtests passed
```

随后已把 Dex 直接接入 30 任务窗口。`task_eval.py` 的 `CameraPolicy` 在
`--hand-backend dex` 时直接接收 `NormalizedDexRetargeter` 返回的
`HandJointResult`，按 `NativeHandSpec` 展开后写入
`env.action_joint_names`；旧 `auto/geometric/checkpoint` 路径保持不变。
Dex 的摄像头输入改走原生手的
`camera_palm_length_window3_v1` 三帧掌长归一化窗口，旧任务路径仍使用原
`MediaPipe -> L21 18D -> NativeHandAdapter`。实时抓取仍由物理接触决定，
Dex 不读取旧的二维 pinch 分类来修改关节。

实机启动命令：

```powershell
$env:MAMBA_ROOT_PREFIX='E:\cache\micromamba'
$env:PYTHONPATH=(Resolve-Path 'src').Path
& 'F:\tools\micromamba\Library\bin\micromamba.exe' run `
  -p 'F:\conda-envs\teleop-dex' python devtools\task_eval.py `
  --task pickcube --robot h1_2 --policy camera --side right `
  --hand-backend dex --camera-grasp-assist off `
  --camera-hand-force 600 --preview --render
```

Windows 上 MediaPipe、PyBullet 与 Torch 的原生动态库存在加载顺序冲突。
任务入口在创建仿真和摄像头模块之前预载 Torch，普通命令现在可直接启动；
窗口启动信息应显示：

```text
hand -> native joints: dex
```

2026-11-23 第一次真实摄像头运行完成 1200 个摄像头帧。右手检测到 654 帧，
Dex 有效输出 590 帧，手指求解约 4.3 ms；食指捏合分类出现过 207 帧。
报告记录的真实物理接触最高达到 4 个手部连杆同时接触，接触法向力最高
61.7 N，“接触即停”触发 1 次且没有创建吸附约束。60 秒 trial 内方块被推动
但未达到抬起判据，最终返回前手已经张开，因此本次结果仍是
`success=false`，不能当作握持或任务成功证据。后续现场复测应分别记录
Dex/geometric 的峰值关节闭合、接触连杆数和任务成功率。

相关回归：

```text
tests/test_arm_realtime_bridge.py::EntryAndReplayTests: 5 passed
tests/test_task_audit.py + tests/test_hand_backend_factory.py:
49 passed, 5 subtests passed
Full test suite:
642 passed, 5 skipped, 554 subtests passed
30 tasks x 3 robots:
90/90 evaluable, 0 excluded
GBK strict check: OK
```

### 食指/拇指接触与鼠标演示

现场反馈有两件事同时失败：

- 真人拇指和食指已经相碰，机器人两指仍相距很远；
- 新链路打开后，鼠标拖不动任务物体。

两件事不是同一个根因。`pinch_signal_details()` 能测到真人捏合，但 Dex
同时优化五个指尖；在 H1-2/GR1-T2 上，满足其余指尖目标的局部解仍可能让
拇指与食指相差 40 mm 以上。现在 Dex 在检测到食指或中指捏合时，会从机器人
URDF 的关节限位内求一个“两指尖真实接触”的独立关节位形，并按捏合强度分帧
平滑过渡到该位形。它只在捏合期间生效，张手立即退回 Dex 原始输出；不创建
手-物体约束，也不改变物体质量。该修正可用
`--camera-pinch-correction off` 关闭做 A/B。

鼠标拖动必须同时满足两个条件：GUI 中允许 picking，且任务物体没有被启动
冻结门改成 `mass=0`。演示命令使用：

```powershell
--allow-gui-mouse-picking --demo-keep-open
```

前者会关闭摄像头启动阶段的物体冻结，后者保证任务判据已经满足后窗口仍保持
打开。此类运行只能用于演示和排查，报告会写
`gui_mouse_picking_enabled=true`，不能算论文成功率。

本轮针对性回归：

```text
tests/test_normalized_dex.py
tests/test_task_audit.py
tests/test_arm_realtime_bridge.py::EntryAndReplayTests
53 passed, 5 subtests passed

Full test suite:
644 passed, 5 skipped, 554 subtests passed
GBK strict check: OK
```

同一次 180 秒手动演示报告中，`gui_mouse_picking_enabled=true`、
`camera.freeze_objects=false`，鼠标拖动链路有效；Dex 捏合接触修正运行
530 帧，右手有效 2015 帧，并两次触发“多连杆接触后停指”。这次 trial 也
返回了成功，但鼠标 picking 允许人为移动物体，所以只能作为演示证据，
不计入论文正式成功率。

### 全掌握拳与误触发保护

后续现场反馈有两点：

- 食指和拇指捏合时，机器人仍不像一次有效捏取；
- 手没有握持时，机器人偶尔也会闭手。

第一个问题分成两层。Dex 的“五指尖联合优化”在 H1-2 上不保证拇指和食指
相互接触；另外旧分类让图像空间无条件覆盖世界空间，图像视角下的完整握拳
可能被标成单指捏合。现在：

- 世界空间和图像空间都确认握拳时，才输出 `fist`；
- Dex 四指闭合度较高且存在明确握持意图时，切到由机器人 URDF 求出的全拳
  位形，四指一起卷曲；
- 食指/中指捏合仍使用两指尖真实接触位形。

第二个问题由三道门共同防止：握持信号必须连续 3 帧确认；食指/中指捏合
强度至少 `0.45` 且距离比不超过 `0.45`；四指全拳位形要求 Dex 原始闭合度
至少 `0.85`。没有明确握持意图时，`open_guard` 会把 Dex 的误闭合目标平滑
拉回原装手张开位形。报告记录
`max_raw_finger_closure`、`max_corrected_finger_closure` 和
`open_guard_alpha`。

鼠标有两种用途：

- `--allow-gui-mouse-picking`：整个 trial 都允许鼠标，适合调场景，结果只算
  手动演示；
- `--setup-gui-mouse-picking`：只在摄像头标定阶段允许鼠标，机械臂开始跟手
  时程序自动关闭 picking。

当前现场按用户要求使用第一种，鼠标保持常开。

相关回归：

```text
native hand pinch + Dex contact + task policy:
60 passed, 11 subtests passed

Full test suite:
649 passed, 5 skipped, 554 subtests passed
```
