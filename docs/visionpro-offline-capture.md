# Vision Pro 离线采集（手部 + 腕/前臂）说明

本页记录"用 Apple Vision Pro 离线录一段、存成 H5"这条链路的**格式、边界与验收方式**。
采集脚本放在 `devtools/`（本支新增，不修改队友已入库的代码），产物默认写
`datasets/raw/`（按仓库约定：`datasets/**` 允许入库）。

## 0. 先说两条硬边界

1. **Vision Pro 不给肩、肘。** 上游串流（App Store 的 Tracking Streamer + Python 的
   `avp_stream`）每帧只提供：手部 25 个关节、腕部位姿、前臂两个关节
   （索引 25 `forearmWrist`、26 `forearmArm`）、头显位姿、拇指-食指距离。
   因此本页的产物**不写**任何"肩/肘"数字，也不冒充团队的
   `left_arm_keypoints (3,3)`（那是 MediaPipe **Pose** 给的三点：shoulder/elbow/wrist）。

   **2026-10-12 真机修正**：实测（右手在前伸姿态下）
   `right_wrist` 给出**绝对世界坐标 6D 位姿**（例：`[0.239, 0.313, 0.811] m`，即离地约
   0.81 m），而 `right_arm[26] forearmArm` 距腕约 **0.22 m** —— 这正是前臂长度，
   也就是说它基本就是**肘的位置**。所以"缺少手臂数据"要说得更准确：
   **腕（绝对 6D）与肘（相对腕、可换算成绝对）都拿得到，缺的是肩。**
   肩只影响手臂平面的法向（`retargeting/arm/retargeter.py` 里 `_forearm_rotation`
   用它算 cross product），这一点需要队友决定用固定假设还是另配追踪器。

   另外注意一个坑：**没追踪到的那只手也会发数据**，但内容是占位值
   （腕平移为 0、旋转为固定 -90 度、`pinch_distance=0`）。这类帧在
   `*_valid` 里被标成 False，不会被当成有效数据。
2. **需要肩肘时不能只靠头显。** 要么另配摄像头（Pose）或 IMU 并单独声明来源，要么
   走论文那条只吃"手腕 6D 位姿"的路线（见
   `src/teleoperation/retargeting/arm/wrist_targets.py` 开头对论文链路的说明）。

## 1. 前置条件（不需要 Mac）

| 组件 | 安装方式 |
|---|---|
| Tracking Streamer（visionOS App） | App Store 安装（可能需要非中国区账号） |
| avp_stream（Windows 侧） | `pip install --upgrade avp_stream` |
| 网络 | 同一 WiFi 用头显显示的 IP；跨网用房间码（`--room-code`） |

`avp_stream` 不在本仓库任何 `requirements*.txt` 里，也不在 `cli/registry.py` 的命令表里，
所以本链路是**独立入口**，不通过 `python -m teleoperation` 启动。

## 2. 产物格式

文件名：`datasets/raw/visionpro_hand_arm_<YYYYmmdd_HHMMSS>.h5`（Asia/Shanghai 时间）。

原始层（不丢信息，任何派生都能重算）：

| 数据集 | 形状 | 说明 |
|---|---|---|
| `left/right_hand_transforms` | (T,25,4,4) float32 | 25 个关节的世界变换矩阵（米） |
| `left/right_wrist_transform` | (T,4,4) float32 | 腕部 6D 位姿 |
| `left/right_forearm_transforms` | (T,2,4,4) float32 | `[forearmWrist, forearmArm]` |
| `head_transform` | (T,4,4) float32 | 头显位姿 |
| `left/right_pinch_distance` | (T,) float32 | 拇指-食指距离（米） |
| `left/right_valid` | (T,) bool | 该帧 25 个矩阵是否全有限 |
| `frame_ids` | (T,) int64 | 本机递增帧号 |
| `timestamps` | (T,) float64 | 相对第一帧的秒 |
| `unix_ms` | (T,) int64 | Unix 毫秒，严格递增（同毫秒内自动 +1） |

派生层（**直接调用队友的函数**换算，保证与流水线一致）：

| 数据集 | 形状 | 说明 |
|---|---|---|
| `left/right_hand_keypoints` | (T,25,3) float32 | 腕部相对、米制、项目 25 点拓扑 |
| `left/right_hand_spread` | (T,) float32 | 关节相对腕部的最大间距（米）；用于识别"连着但没追踪到手"的占位帧 |

换算链：`retargeting.hand.topology.visionpro_fingers_to_points`
（按左右手做轴重映射，取平移列）后接 `ensure_hand25`（减腕部、缩放置 1.0）。
验证方式见 §5：本脚本的派生层与 `CanonicalHandProcessor` 对同一帧的输出**逐点差 0**。

根属性：`source=visionpro`、`source_landmark_space=visionpro_world_meters`、
`derived_origin=wrist`、`length_unit=m`、`units=m`、`timestamp_unit=unix_seconds`、
`missing_hand_policy=zeros`、`arm_source_note=visionpro_forearm_only_no_shoulder_no_elbow`、
`derived_alignment_status=unaligned_pending_team_confirmation`。

**没有 `coordinate_alignment`。** 派生层只做到"腕相对、米制"，还没有做"掌面旋转到 L21"
那一步；仓库只承认三个 alignment 取值，其中两个明确属于 MediaPipe。按契约
"未知来源必须报错、不许猜"，这里故意不声明，等队友给一个正式名字（见 §4）。

## 3. 怎么录

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"

# 1) 先确认串流通了：打印字段名与形状，不写盘
E:\python3.11.7\python.exe devtools\visionpro_record.py --list-keys --ip <头显IP>

# 2) 录一段：默认 datasets/raw/，按本地时间命名；--frames 0 --seconds 0 会一直录到 Ctrl+C
E:\python3.11.7\python.exe devtools\visionpro_record.py --ip <头显IP> --seconds 180

# 3) 校验产物：结构、时间戳、有效帧比例、单位是否像米
E:\python3.11.7\python.exe devtools\visionpro_check.py --input-h5 datasets\raw\visionpro_hand_arm_xxx.h5
```

录制建议（与队友既有采集习惯一致）：双手始终在视野内；单条 3 到 5 分钟；动作覆盖
张手、握拳、单指弯曲、拇指对各指捏合；每条之间换一个动作主题，方便按会话切分验证集。

自检脚本的判据：结构问题（缺数据集、形状不符、时间戳不递增、矩阵含 NaN）报
`[FAIL]` 并退出码 1；尺度异常（腕到食指指尖中位数不在 0.05 到 0.30 m）只报 `[WARN]`。

## 4. 交给队友前必须确认的四件事

1. **对齐契约名**：Vision Pro 米制数据要做掌面旋转对齐时，用哪个
   `coordinate_alignment`？现有两个取值都绑定 MediaPipe 来源，直接套用等于伪造来源。
   建议新增一个（例如 `visionpro_world_palm_local_to_l21_v1`），由队友加进
   `contracts/coordinates.py` 的 `SUPPORTED_COORDINATE_ALIGNMENTS`。
2. **手臂要不要肩肘**：如果后续 IK 需要 shoulder/elbow，Vision Pro 补不了，
   要么摄像头 Pose 同帧并行采集、在 H5 里分开声明来源，要么改走手腕 6D 位姿路线。
3. **训练消费方式**：现有 `hand align` 只接受 `mediapipe_normalized`（或未声明），
   米制世界路线要 `source_landmark_space=mediapipe_world_meters`。Vision Pro 数据要进
   训练，必须先有第 1 条的名字，否则读取器会按设计报错。
4. **时间基准**：本脚本的时间戳是**PC 接收时刻**（上游只给预测手部姿态，不给可靠的事件
   时间戳）。如果要做多源对齐（例如与视频、机械臂日志对表），需要约定统一时钟。

## 5. 已经验证过的部分（2026-10-12）

| 命令 | 退出码 | 结果与不变量 |
|---|---:|---|
| `python devtools\visionpro_record.py --self-test --frames 45` | 0 | 写出 45 帧、双测有效 45 帧、实测 fps 约 28.7 |
| `python devtools\visionpro_check.py --input-h5 <自测产物>` | 0 | 结构全通过；腕到中指尖中位 156 mm（米制尺度合理） |
| `python devtools\visionpro_record.py --self-test --list-keys` | 0 | 字段与形状：`*_fingers (25,4,4)`、`*_wrist (4,4)`、`*_arm (27,4,4)`、`head (4,4)` |
| 派生层与 `CanonicalHandProcessor` 对同一帧比对 | 0 | 左右手逐点最大差 `0.000e+00 m` |
| `python devtools\check_gbk_safe.py --strict` | 0 | 新增源码无 GBK 无法编码字符 |
| 真机连接（头显 IP 10.23.254.126，同一 WiFi） | 0 | TCP 12345 可连；281 帧 / 9.97 s，实测 28.2 fps；双侧 281/281 有效 |
| 真机尺度（同一段） | 0 | 腕到指尖中位：左 index 180 mm / middle 187 mm，右 index 177 mm / middle 182 mm；关节跨度 18.2 到 18.7 cm，符合成年人的米制手型 |
| 占位帧防护（拿"未追踪"状态做反面用例） | 1 | 校验脚本对"标了有效但没有关节几何"的数据报 [FAIL]，不会把占位矩阵当数据 |

以上都不依赖头显（`--self-test` 用合成骨架）。**真机部分仍需现场验证**：
`--list-keys` 能否列出真实字段、轴与左右手符号是否符合直觉、尺度是否稳定。
现场方法见 `devtools/visionpro_check.py --ip <头显IP> --frames 60`：它打印双侧有效帧比例、
腕到各指尖的中位距离（应接近 150 到 190 mm）与腕部在世界坐标下三轴的取值范围；
把手抬高 20 cm 再跑一次，变化最大的那个轴就是竖直方向，用来核对轴约定。
