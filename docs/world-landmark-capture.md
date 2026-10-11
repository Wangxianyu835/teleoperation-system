# 手部 + 手臂米制采集

这条链路采集 MediaPipe 手部与肩、肘、腕数据，同时写入归一化坐标和
米制 world 坐标，不加载重定向模型、不生成机器人角度、不进入仿真。

## 模型文件

程序会在缺少模型时尝试自动下载：

```text
hand_landmarker.task
pose_landmarker_lite.task
```

优先使用 hf-mirror，失败后回退官方 Google 地址。两个模型都已加入
`.gitignore`，不要提交到仓库。

## 最简单运行方式

PyCharm 直接打开项目根目录下的 `record_world_landmarks.py`，点击 Run。
默认从摄像头 0 采集 30 帧，限制为最多 5 FPS，并保存为：

```text
outputs/hand_body_capture/hand_body_日期_时间.h5
```

预览窗口中按 `q` 可以提前结束并保存。若不需要预览，使用 `--no-preview`。

也可以直接运行内部入口：

```text
src/teleoperation/applications/world_landmark_capture.py
```

## 命令行方式

```powershell
python -m teleoperation hand record-world `
  --camera-index 0 `
  --frames 30 `
  --output outputs/hand_body_capture/session_001.h5
```

常用参数：

```text
--model-asset-path
--pose-model-asset-path
--pose-min-visibility
--camera-index
--frames
--output
--width
--height
--fps
--max-fps
--preview / --no-preview
```

## H5 字段

```text
frame_ids                       (T,)       int64
timestamps                      (T,)       float64, 原 MediaPipe Unix 毫秒
left_hand_keypoints             (T, 21, 3) float32, 归一化
right_hand_keypoints            (T, 21, 3) float32, 归一化
left_world_landmarks            (T, 21, 3) float32, 米
right_world_landmarks           (T, 21, 3) float32, 米
left_arm_keypoints              (T, 3, 3)  float32, 归一化肩/肘/腕
right_arm_keypoints             (T, 3, 3)  float32, 归一化肩/肘/腕
left_arm_world_keypoints        (T, 3, 3)  float32, 米制肩/肘/腕
right_arm_world_keypoints       (T, 3, 3)  float32, 米制肩/肘/腕
pose_visibility                 (T, 6)     float32
left_valid                      (T,)       bool
right_valid                     (T,)       bool
left_arm_valid                  (T,)       bool
right_arm_valid                 (T,)       bool
```

缺失或低可见度的点使用 NaN 保存，并由对应的 `*_valid=False` 标记。

root metadata：

```text
dataset_format_version = hand_body_capture_h5_v1
source_landmark_space = mediapipe_normalized
hand_world_landmark_space = mediapipe_world_meters
hand_world_origin = hand_geometric_center
arm_source_space = mediapipe_pose_normalized
arm_world_space = mediapipe_pose_world_meters
arm_world_origin = hip_midpoint
length_unit = m
```

`left_hand_keypoints/right_hand_keypoints` 和归一化手臂字段保持项目原始
读取接口兼容；新增 world 字段只用于米制记录。原有 normalized 采集链路、
`hand align`、`hand realtime` 和 `dual export` 默认行为不变。
