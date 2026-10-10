# World Landmark 米制采集

这条链路只负责采集 MediaPipe `hand_world_landmarks` 并写入独立 H5，
不加载重定向模型、不生成机器人角度、不进入仿真。

## 模型文件

首次使用先下载官方 Hand Landmarker 模型到项目根目录：

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "hand_landmarker.task"
```

该文件是外部模型资产，已加入 `.gitignore`，不要提交到仓库。

## 最简单运行方式

PyCharm 直接打开项目根目录下的 `record_world_landmarks.py`，点击 Run。
默认从摄像头 0 采集 300 帧，并保存为：

```text
outputs/world_landmarks/world_landmarks_日期_时间.h5
```

也可以在项目根目录执行：

```powershell
python record_world_landmarks.py
```

## 新代码位置

| 文件 | 作用 |
|---|---|
| `record_world_landmarks.py` | 项目根目录的独立运行入口，PyCharm 可直接 Run |
| `src/teleoperation/inputs/mediapipe.py` | 在同一次 MediaPipe 结果中读取 world landmarks |
| `src/teleoperation/data/world_landmark_recording.py` | 写独立 H5、读取和校验 |
| `src/teleoperation/apps/world_landmark_capture.py` | 采集循环和应用入口 |
| `tests/test_world_landmark_recording.py` | H5 记录、采集入口、CLI 注册测试 |

## 命令

```powershell
python -m teleoperation hand record-world `
  --model-asset-path hand_landmarker.task `
  --camera-index 0 `
  --frames 300 `
  --output outputs/world_landmarks/session_001.h5
```

可用参数：

```text
--model-asset-path
--camera-index
--frames
--output
--width
--height
--fps
```

## H5 字段

```text
frame_ids                  (T,)       int64
timestamps                 (T,)       float64, 原 MediaPipe Unix 毫秒
left_world_landmarks       (T, 21, 3) float32, 米
right_world_landmarks      (T, 21, 3) float32, 米
left_valid                 (T,)       bool
right_valid                (T,)       bool
```

缺失的一侧使用全 NaN 行保存，并由对应的 `*_valid=False` 标记。

root metadata：

```text
dataset_format_version = world_landmarks_h5_v1
source_landmark_space = mediapipe_world_meters
length_unit = m
world_origin = hand_geometric_center
```

原有 normalized 采集链路和 `hand realtime` 行为不变。
