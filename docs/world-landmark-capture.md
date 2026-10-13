# 手部 + 手臂米制采集

这条链路采集 MediaPipe 手部与肩、肘、腕数据，同时写入归一化坐标和
米制 world 坐标，不加载重定向模型、不生成机器人角度、不进入仿真。

## 模型文件

程序会在缺少模型时尝试自动下载：

```text
datasets/hand_landmarker.task
datasets/pose_landmarker_lite.task
```

优先使用 hf-mirror，失败后回退官方 Google 地址。两个模型都已加入
`.gitignore`，不要提交到仓库。

## 统一命令入口

与项目其他工作流一样，通过统一 CLI 启动：

```powershell
python -m teleoperation hand record-world
```

默认从摄像头 0 采集 30 帧，限制为最多 5 FPS，并保存为：

```text
outputs/hand_body_capture/hand_body_日期_时间.h5
```

文件名使用 Asia/Shanghai 时间；同一秒内重名时追加 `_01`、`_02` 等序号。
用 `--output-dir` 更换自动保存目录，或用 `--output` 指定完整文件名，两个参数不能同时使用。
预览窗口中按 `q` 可以提前结束并保存。若不需要预览，使用 `--no-preview`。
`--help` 只解析参数，不导入采集应用、不打开设备或下载模型。

显式指定文件名：

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
--output-dir
--width
--height
--fps
--max-fps
--preview / --no-preview
```

## 实现流程与原文件去向

根目录 `record_world_landmarks.py` 已删除，不再保留直接运行 wrapper。
应用模块也不再自行解析参数。原脚本的职责按既有分层归入：

| 职责 | 当前文件 |
|---|---|
| 参数与默认值 | `src/teleoperation/cli/schemas.py` 的 `hand_record_world()` |
| 静态命令注册 | `src/teleoperation/cli/registry.py` |
| 模型及默认输出目录 | `src/teleoperation/paths.py` |
| 自动命名、采集循环、预览、资源释放、保存调用 | `src/teleoperation/applications/world_landmark_capture.py` 的 `run()` |
| 同帧手部与姿态识别 | `src/teleoperation/inputs/mediapipe.py`、`pose_mediapipe.py` |
| H5 schema 与原子写入 | `src/teleoperation/data/hand_body_recording.py` |

```text
python -m teleoperation hand record-world
    -> cli/schemas.hand_record_world: parse arguments
    -> applications/world_landmark_capture.run: resolve output and model assets
    -> inputs/mediapipe.MediaPipeCameraInput: read world + pose from the same frame
    -> applications/world_landmark_capture: preview, collect and release
    -> data/hand_body_recording.write_hand_body_recording: save H5 atomically
```

CLI 保留原有 `fast_exit` 行为：保存完成后快速退出进程，避免 MediaPipe 退出等待。
程序内调用 `run(args)` 时，未设置 `args.fast_exit` 则正常返回。

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
hand_processing_landmark_space = mediapipe_world_meters
hand_world_landmark_space = mediapipe_world_meters
hand_world_origin = hand_geometric_center
arm_source_space = mediapipe_pose_normalized
arm_world_space = mediapipe_pose_world_meters
arm_world_origin = hip_midpoint
length_unit = m
```

`left_hand_keypoints/right_hand_keypoints` 和归一化手臂字段保持项目原始
读取接口兼容。新增 world 字段现在可以通过 `hand align` 接入训练；身体字段保持原格式。

## 手部 world 处理与重新训练

`hand record` 与 `hand realtime` 默认选择 `--landmark-space world`，三机器人实时展示也使用该默认值。
`hand record` 流式保存两套手部 21 点，并在正常结束后自动生成所选来源的 palm-local 25 点训练文件；它的缺失编码为零，时间戳为 Unix 秒，与本页 `record-world` 的 NaN/Unix 毫秒布局区分。
`record-world` 的新文件声明 `hand_processing_landmark_space=mediapipe_world_meters`，不会自动对齐；旧文件需显式指定 world。

```powershell
# 新 record-world 文件默认按声明选择 world；旧文件加 --landmark-space world
python -m teleoperation hand align `
  --input outputs/hand_body_capture/session_001.h5 `
  --output outputs/hand_body_capture/aligned_session_001.h5 `
  --landmark-space world

# 创建新的 world 实验；旧归一化模型不能初始化 world 模型
python -m teleoperation hand train `
  --input outputs/hand_body_capture/aligned_session_001.h5 `
  --run-name world_canonical_shared_v1 --hand-side shared --device cuda

python -m teleoperation hand export `
  --input outputs/hand_body_capture/aligned_session_001.h5 `
  --output outputs/hand_retargeting/world_angles.h5

python -m teleoperation hand realtime --landmark-space world --visualize --frames 0

# 显式 normalized 模式使用三点原生根坐标，需要匹配 v2 新权重
python -m teleoperation hand record --landmark-space normalized --frames 900
python -m teleoperation hand align --input old_raw.h5 --output old_aligned.h5 --landmark-space normalized
python -m teleoperation hand realtime --landmark-space normalized --visualize --frames 0
```

world 21 点先减腕点，只用 0/5/9 经 SVD 建立掌部基，再按左右矩阵统一为右手表示，最后扩展到 25 点。左侧 dex 输出额外做 Y 镜像，矩阵可合并；输出坐标框架为 `hand_canonical_right`，契约为 `mediapipe_world_wrist_index_middle_canonical_right_v1`，
来源为 `mediapipe_world_meters`，单位 `m`，`source_scale=1.0`。没有掌部尺寸标定或逐帧缩放；人手保留米制尺寸，机器人 FK 保留 URDF 米制尺寸。
world Dataset 沿用 MediaPipe 左右标签，并复用实时三帧窗口语义。单侧缺失、非有限或退化会清空该侧窗口，恢复后重新积累 3 帧。
缺少整套 world 字段会报错；归一化数据不能用于填补 world 缺失。

world 实时和默认 world 导出查找 `outputs/hand_retargeting/checkpoint/models/twohand_h5/linker/world_canonical_shared_v1/model_best.pth`。
其他实验名需显式传共享 checkpoint 参数；新 world 模式拒绝独立左右模型。normalized 新预处理为三点 v2，实时目录为 `palm_three_point_v2`。
`hand train` / `hand export` 从已对齐 H5 契约确定模式，不能通过参数覆盖；world 导出拒绝非 `1.0` 的 `--scale-factor`。
H5 与 checkpoint 契约不同会报错，即使模型网络形状完全相同；不改写旧文件或旧权重。

预览始终投影归一化关键点。world 对比面板的人手和机器人共用 `--robot-radius` 对应的米制显示比例；交互缩放只改变显示。
共享训练汇集任意人手侧的有效窗口，只使用右侧 L21 FK；左侧输出适配四个 MCP roll 的符号。拇指结构差异不通过左侧训练修正，需单独验收。六项损失和网络形状保持，拇指距离掌面改用腕点/食指/中指，指尖距离中的 `×1000` 表示米转毫米。30 帧短采集用于验证记录格式；正式训练应录制足够的连续有效帧。
TRON2A 双臂适配及独立 `camera_record` 入口保留原行为。
