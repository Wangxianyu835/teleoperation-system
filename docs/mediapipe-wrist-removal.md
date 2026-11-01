# MediaPipe 派生腕观测退役（2026-11-01）

MediaPipe Hands 继续提供左右手原始 21 点及 world 米制关键点，但不再把掌部坐标基
转换为独立腕四元数。当前相机机械臂的腕位置和朝向来自 SMPLer-X，Vision Pro
仍从 SDK 腕矩阵建立腕观测。手指模型使用的局部坐标与预处理保持原样。

## 修改前 checkpoint

- commit：`a26b0c997f5ee1e7d8f74b87cb9986dbe1409cb0`
- annotated tag：`checkpoint-20261101-1439-remove-mediapipe-wrist`
- 两份原先未跟踪的相机 H5 已在上述 checkpoint commit 单独保存，未改写其内容。

## 删除与迁移

整文件删除：

- `src/teleoperation/retargeting/arm/wrist_targets.py`
- `devtools/verify_arm_from_capture.py`
- `tests/test_wrist_targets.py`

旧模块中的通用 `rotation_matrix_to_quaternion`、`matrix_from_quaternion` 原样迁到
`src/teleoperation/retargeting/arm/rotations.py`，由 `simulation/arm_realtime.py` 使用。
通用转换测试迁到 `tests/test_arm_rotations.py`。

局部清理：

- `inputs/wrist.py` 只保留 Vision Pro 腕解析，删除相机专属生成、恢复、写入函数与常量。
- `inputs/mediapipe.py` 保留字典 `wrists` 键及公共字段，值为 `None`。
- `data/hand_recording.py`、`hand_body_recording.py`、`world_landmark_recording.py`
  停止生成相机腕数据集和专属元数据。
- `iter_dex_observations` 的相机分支忽略旧派生腕字段，VR 分支继续解析腕矩阵。
- `data/body_recording.py` 写入和回放时过滤 `camera_palm` 来源或
  `mediapipe_hand_world_axes` 参考系，保留其余腕数据。
- `simulation/realtime_teleop.py` 的手指 NPZ 回放不再推算腕朝向，删除未落盘的派生腕缓存。
- `WristTracker` 把旧 MediaPipe 腕输入视为缺失，不重新标定或覆盖有效来源；
  继续执行保持/丢失策略，支持 SMPLer-X 和 Vision Pro。

## 上下游与兼容性

删除前：`world21 -> camera_wrists -> 掌部基 -> 四元数 -> WristObservation -> 记录/旧控制`。

删除后保留两条独立路径：

```text
MediaPipe world21 -> 局部坐标对齐 -> 三帧与掌长归一化 -> 手指重定向 -> 关节命令
SMPLer-X -> 身体跟踪 -> 腕观测 -> 中性对齐 -> 机械臂 IK 与安全控制
```

`WristObservation`、`RawHandFrame.wrists`、`RawTeleopObservation.wrists` 的公共定义保留。
已有录制不批量修改；相机派生腕附加字段可被对齐复制保留，但不再恢复为控制输入。
SMPLer-X 身体包、Vision Pro 腕字段、机器人结果及关节命令顺序保持各自原有契约。
历史文档中的旧诊断路径表示当时状态，不作为当前入口。

## 验证

修改前完整 unittest：582 项，3 项跳过，退出码 0。

修改后验证结果：

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s tests`（正式 conda 环境） | 576 项，3 项跳过，退出码 0；旧算法专用测试随退役删除 |
| 相关局部测试（首次清理后） | 190 项通过，退出码 0；随后新增的录制检查由完整回归覆盖 |
| dex 独立环境：`tests.test_normalized_dex tests.test_dex_native` | 19 项通过，退出码 0，包含左右 L21 实际优化求解 |
| 六个 CLI 帮助入口 | hand control/export/record/vr、dual camera/replay，退出码均为 0 |
| 代码与 GBK 检查 | `git diff --check`、两套 GBK 检查、目录/导入边界通过 |
| 历史来源兼容 | 旧相机 H5/组合包的派生腕被忽略，关键点和 SMPLer-X 腕保留；旧来源无法重置活动腕标定 |

前后对照使用已保存的实际双手录制，均匀选取 60 帧并加入左右侧分别丢手/恢复，
同一个 L21 shared checkpoint、CPU 推理，比较：

- 每侧 canonical 25 点和掌长归一化三帧窗口；
- 模拟实时逐帧调用的 NN 原始解、有效标志和接受命令；
- 原始 H5 录制、离线对齐、正式 `hand export` 输出的 NN 原始解和命令。

以上所有数组前后完全相等（包含缺失掩码）；左右分别有 51/52 个有效 NN 输出帧。
共享旋转函数迁移前后的可执行 AST 一致，`vr_wrist` 可执行 AST 一致。
统一坐标、单帧处理、掌长归一化、时间窗口、离线对齐、坐标契约、
TransHandR 预处理/模型适配、SMPLer-X 几何解析共 9 个受保护文件的 SHA-256 前后一致。
两份原始/对齐录制的 Git 内容 hash 也与 checkpoint 一致。

对照命令关闭 L21 碰撞检查以隔离输入分支；常规测试仍覆盖碰撞控制。
结果位于忽略的 `outputs/wrist_cleanup/`，包括测试日志、前后 H5/NPZ 与受保护源文件 hash。
实际摄像头、实体机器人和头显现场动作不在此次自动回归范围内。

预检的 C 盘缓存残留检查在修改前已有失败：用户默认目录下存在 Hugging Face 缓存内容。
本次运行的缓存和临时目录均指向 D 盘，未删除该历史目录或修改用户级配置。
