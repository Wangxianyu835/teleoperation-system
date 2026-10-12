# Vision Pro 手部 + 腕/前臂数据交接说明（2026-10-12）

本文是**交付给团队的数据说明**：两份真机采集文件、字段与单位、与现有契约的对应关系、
实测质量证据，以及需要队友拍板的两件事。采集工具与格式细节见
[visionpro-offline-capture.md](visionpro-offline-capture.md)。

## 0. 一句话

用 Apple Vision Pro（App Store 的 Tracking Streamer + `avp_stream`）离线录了两段双手数据，
共 **7517 帧 / 268.5 秒 / 30.8 MB**，含双手 25 关节、腕部绝对 6D 位姿、前臂两点、头显位姿、
拇指-食指距离；**长度单位为米**；已剔除"摘掉头显后 App 重复发送"的冻结帧，
裁剪后双侧有效帧 100%（第 1 条左手 4630/4631）。

## 1. 交付文件

目录：`datasets/raw/`

| 文件 | 帧数 | 时长 | 大小 | 有效帧（左/右） | SHA256 |
|---|---:|---:|---:|---|---|
| `visionpro_hand_arm_20261012_213812_live.h5` | 4631 | 165.0 s | 19.0 MB | 4630 / 4631 | `68CB950C888AAE14CB38B703BE47F83C7AF2A36DF9A6F9260A3B3871720B23C8` |
| `visionpro_hand_arm_20261012_215137_live.h5` | 2886 | 103.5 s | 11.8 MB | 2886 / 2886 | `6D98E00015BFF2802906B05AB8FCED750B2B415D6B99481E15533002A878B64B` |

同目录另有同名的**原始文件**（未裁剪，末尾含冻结帧）：`..._213812.h5`（38.9 MB）、
`..._215137.h5`（26.2 MB）。原始文件只作追溯用，**不建议入库**；入库建议只放两份 `_live.h5`。
两份文件都在 100 MB 的 Git 上限之内，无需 LFS。

## 2. 数据格式

每个文件是一份 HDF5，含**原始层**（矩阵，不丢信息）与**派生层**（项目 25 点，可直接进现有流程）。

### 2.1 原始层

| 数据集 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `left/right_hand_transforms` | (T,25,4,4) | float32 | 25 个手部关节的 4x4 变换矩阵，**腕相对**（腕关节位于原点） |
| `left/right_wrist_transform` | (T,4,4) | float32 | 腕部 6D 位姿，**AVP 世界坐标系**（与头显同系） |
| `left/right_forearm_transforms` | (T,2,4,4) | float32 | `[forearmWrist, forearmArm]`，腕相对 |
| `head_transform` | (T,4,4) | float32 | 头显 6D 位姿，AVP 世界坐标系 |
| `left/right_pinch_distance` | (T,) | float32 | 拇指尖到食指尖距离（米） |
| `left/right_hand_spread` | (T,) | float32 | 关节相对腕部的最大间距（米），用于识别冻结帧 |
| `left/right_valid` | (T,) | bool | 该帧是否可信（见 §2.3） |
| `frame_ids` | (T,) | int64 | 本机递增帧号 |
| `timestamps` | (T,) | float64 | 相对第一帧的秒 |
| `unix_ms` | (T,) | int64 | PC 接收时刻的 Unix 毫秒，严格递增 |

### 2.2 派生层

| 数据集 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `left/right_hand_keypoints` | (T,25,3) | float32 | **腕部相对、米制、项目 25 点拓扑** |

派生层的换算链**直接调用本仓库的函数**，不做任何自创变换：
`retargeting.hand.topology.visionpro_fingers_to_points`（取平移列 + 按左右手做轴重映射）
后接 `ensure_hand25`（减腕部、缩放置 1.0）。
实测与 `CanonicalHandProcessor` 对同一帧输出**逐点差 0.000e+00 m**。

### 2.3 `valid` 的判据（三选三）

1. 25 个矩阵全为有限值；
2. 关节相对腕部有真实间距（> 0.1 mm）——排除"App 在跑但没追踪到手"的占位矩阵；
3. 与前一帧不是逐位相同（连续 30 帧相同即判定该侧冻结）——排除摘下头显后的重复帧。

不满足的帧**保留在数组里但标 `valid=False`**，派生层写全零行（沿用项目"缺手写零"的既有约定），
不会用上一帧冒充。

### 2.4 根属性（节选）

```text
source = visionpro
source_landmark_space = visionpro_world_meters
derived_origin = wrist
derived_alignment_status = unaligned_pending_team_confirmation
length_unit = m        units = m
timestamp_unit = unix_seconds
missing_hand_policy = zeros
arm_source_note = visionpro_forearm_only_no_shoulder_no_elbow
forearm_joint_note = index25=forearmWrist, index26=forearmArm; 实测 26 距腕约 0.22 m，接近肘的位置
```

裁剪过的文件另有 `trimmed_from`、`trimmed_frames`、`trim_reason`、`trim_rule`、
`left/right_freeze_at_frame`、`frame_count`、`duration_seconds`。

## 3. 单位、坐标系与轴

- **长度一律为米**，旋转为 3x3 旋转矩阵；时间戳为 PC 接收时刻（Unix 秒/毫秒），
  **不是头显事件时间戳**（上游发的是预测姿态，没有可靠事件戳）。
- 上游轴约定：X 向右、Y 向上、Z 向后（右手系）；`*_wrist_transform` 与 `head_transform`
  的原点都是头显检测到的**物理地面**，两者可互相换算（同一坐标系）。
- `*_hand_transforms` 是**腕相对**的；项目的转换本来就要减腕部，所以腕相对与世界坐标
  在派生层结果一致。
- 派生层已按项目既有写法做左右手有别的轴重映射，再减腕部；`origin = wrist`。

## 4. 与团队契约的对应关系

| 事项 | 状态 |
|---|---|
| 25 点拓扑（0 腕、1-4 拇指、5 掌根+6-9 食指、10+中指、15+无名指、20+小指） | 与 `retargeting/hand/topology.py` 一致，索引逐位对得上 |
| 米制、腕相对、`source_scale=1.0` | 满足 world 路线的量纲要求 |
| `RawHandFrame` 契约（`source="visionpro"` + `geometry_encoding="transforms25"`） | 实机数据可直接构造，`contracts/observations.py` 的 (25,4,4) 校验通过 |
| **掌面旋转到 L21 的对齐** | **未做**，文件里故意不写 `coordinate_alignment` |

### 4.1 需要队友拍板之一：alignment 名字

`contracts/coordinates.py` 目前只承认三个取值，其中两个明确绑定 MediaPipe 来源。
Vision Pro 的米制数据来源不同，**直接套用 `mediapipe_world_palm_local_to_l21_v1` 等于伪造来源**；
按契约"未知来源必须报错、不许猜"，本数据不声明 alignment，因此现在**还不能直接进
`hand align` / `hand train`**。建议新增一个正式名字（例如 `visionpro_world_palm_local_to_l21_v1`）
并加入 `SUPPORTED_COORDINATE_ALIGNMENTS`；对齐数学可直接复用
`retargeting/hand/coordinates.align_mediapipe_hand`（它接受 (25,3)，与来源无关）。

### 4.2 需要队友拍板之二：肩关节

Vision Pro 的骨架到前臂为止，**没有肩**。实测：

- 腕部：**绝对世界坐标 6D 位姿**（例：右手 `[0.239, 0.313, 0.811]` m，离地 0.81 m）；
- 肘部：`forearmArm` 距腕约 **0.22 m**，正是前臂长度，**基本就是肘的位置**（可换算成绝对坐标）；
- 头显：绝对 6D 位姿。

所以现有 `ArmRetargeter` 需要的 `(3,3)` 肩/肘/腕里，**肩这一列无法由头显提供**。
肩只影响手臂平面法向（`retargeting/arm/retargeter.py` 里 `_forearm_rotation` 的叉积项），
请决定：用固定平面假设，还是另配追踪器（或继续用摄像头 Pose 提供肩）。

## 5. 质量证据（命令 + 退出码 + 不变量）

```powershell
cd <repo>
$env:PYTHONPATH = "$pwd\src"
python devtools\visionpro_check.py --input-h5 datasets\raw\visionpro_hand_arm_20261012_213812_live.h5
python devtools\visionpro_check.py --input-h5 datasets\raw\visionpro_hand_arm_20261012_215137_live.h5
```

两条命令退出码均为 **0**（结构全通过）。关键不变量：

| 指标 | 第 1 条（213812） | 第 2 条（215137） |
|---|---|---|
| 帧数 / 时长 / 实测 fps | 4631 / 165.0 s / 28.07 | 2886 / 103.5 s / 27.90 |
| 有效帧（左/右） | 4630 / 4631 | 2886 / 2886 |
| 腕到指尖中位（左，mm） | 拇指 139 / 食指 177 / 中指 179 / 无名 165 / 小指 144 | 135 / 170 / 172 / 164 / 147 |
| 腕到指尖中位（右，mm） | 138 / 176 / 178 / 165 / 142 | 135 / 171 / 178 / 169 / 149 |
| pinch 最小值（左/右） | 0 mm / 1 mm | 7 mm / 7 mm |
| 腕部活动范围（左，cm） | 17.9 x 35.7 x 20.5 | 13.1 x 26.6 x 16.9 |
| 腕部活动范围（右，cm） | 25.5 x 37.9 x 29.0 | 11.3 x 26.9 x 17.3 |

**尺度可信的证据**：左右手独立算出的"腕到指尖"中位数相差不超过 2 mm，且中指最长、
小指最短，符合人体比例——说明单位是米、腕相对原点正确，双手没有系统性偏差。

冻结帧检查：原始文件第 1 条在第 4631 帧（t=165.0 s）、第 2 条在第 2886 帧（t=103.5 s）
出现逐位重复，校验脚本会报 `[FAIL]` 并要求裁剪；交付文件已裁到重复开始之前，因此通过。

## 6. 动作覆盖与建议

两段都包含：张手、握拳、逐指弯曲、拇指对各指捏合，以及手在身前的前后移动
（第 2 条捏合更充分，pinch 最小到 7 mm）。

**文件里没有动作标签。** 建议后续采集按"一条一个动作主题"（例如一条全捏合、一条全握拳），
文件名或外部清单里标注；否则训练时无法按动作切分验证集。

## 7. 已知限制与坑（重要）

1. **摘下头显后 App 不会停止发送**，而是重复最后一帧（本次两段各录进 98 到 129 秒冻结数据）。
   工具已加三重防护：冻结帧标 `valid=False`、两侧冻结 3 秒自动停止、`visionpro_trim.py` 裁剪。
   采集约定：**先停止录制，再摘头显**。
2. 没有视频、没有眼动数据、没有肩（见 4.2）。
3. 追踪最佳范围约 0.5 到 2 m（上游文档）；手离开视野会失效，对应帧 `valid=False`。
4. 上游仓库里 `docs/source/data_format.rst` 把 25 点里的 21-24 说成"手掌点"，
   与 README 的 API 表（21-24 是小指、25/26 是前臂）矛盾。**实测支持 API 表那一套**：
   按它读出来的手长完全符合人体比例，按另一套读会把指尖当成手掌点。以实测为准。
5. `avp_stream` 的上游元数据会强升 numpy 到 2.x、并安装 `opencv-python`
   （会覆盖 `opencv-contrib-python` 的 `cv2`，影响 mediapipe）。本机按最小依赖安装：
   `pip install --no-deps avp_stream` + `aiortc av websocket-client requests`，
   numpy 保持 1.26.4。复现环境时不要直接 `pip install avp_stream`。

## 8. 复现步骤（再采一条）

```powershell
cd <repo>
# 头显：打开 Tracking Streamer，按左边的 START（只追踪），记下它显示的 IP
python devtools\visionpro_record.py --ip <头显IP> --seconds 0 --auto-stop-stale 0
# 做动作；结束前先按 Ctrl+C（或 --stop-file 触发）停止，再摘头显
python devtools\visionpro_trim.py   --input-h5 datasets\raw\visionpro_hand_arm_<时间>.h5
python devtools\visionpro_check.py  --input-h5 datasets\raw\visionpro_hand_arm_<时间>_live.h5
```

参见：[采集说明](visionpro-offline-capture.md)（含参数表与验收判据）。

## 9. 环境依赖

- 头显：Tracking Streamer（App Store；可能需要非中国区账号），按左边 START
- 电脑：Python 3.11.9（仓库 `.venv`）+ `avp_stream 2.51`、`aiortc`、`av`、
  `websocket-client`、`requests`；numpy 1.26.4（见 7.5）
- 网络：同一局域网时用头显显示的 IP；跨网可用房间码（`--room-code`）

## 10. 待确认清单（给队友）

1. 是否新增 `visionpro_world_palm_local_to_l21_v1` 这类 alignment 名字并加入白名单？
2. 肩关节如何补（固定平面假设 / 摄像头 Pose 并行采集 / 其他追踪器）？
3. 是否需要视频或统一时间基准（当前时间戳是 PC 接收时刻）？
4. 这两份 `_live.h5` 是否入库（建议入库，合计 30.8 MB，符合 `datasets/**` 例外与 100 MB 上限）？
