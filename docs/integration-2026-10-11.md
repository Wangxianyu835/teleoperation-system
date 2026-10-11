# main 分支集成记录（2026-10-11）

基线为 `10ebd9f8ef914485fc20c6b476a6e5dcb5dbff5e`。本次合入：

- `cs/realtime-preview`：`70932a2083b5bd93ba10e86218d8a3b7ce0d65db`。
- `world_landmark_test`：`136f2d9c1f924c0e3fff73716d011daa9bb558c7`。

合并前的本地 annotated tag 为 `checkpoint-20261011-merge-realtime-world`。
两个功能分支保留，合并历史保留原始提交。

## 新增工作流

```powershell
# 摄像头预览 + 三台机器人原装手；geometric 不需要训练 checkpoint
python -m teleoperation.applications.realtime_hand_sim --backend geometric

# 仅显示 L21 手
python -m teleoperation.applications.realtime_hand_sim --backend geometric --scene hands

# 独立录制入口，可额外输出几何后端的 18 维角度
python -m teleoperation.applications.camera_record --seconds 10

# 手部与肩肘腕：同步保存 normalized 和米制 world 数据
python -m teleoperation hand record-world --frames 30 --output outputs/hand_body_capture/session.h5

# 在 IDE 中也可直接运行；--help 不打开设备或下载模型
python record_world_landmarks.py --help
```

实时三机器人入口默认只驱动手指，手臂与其他关节保持固定。
几何后端需要张开手标定；它与 PoseTransformer 是不同后端，不能用几何结果证明训练模型的精度。
`wrist_targets.py` 与 `devtools/verify_arm_from_capture.py` 提供近似手腕位姿及 PyBullet IK 验证，默认映射参数没有完成相机标定。

`hand record` 原有录制/自动对齐与 `hand record-world` 并存。
前者保存供现有训练与导出使用的 palm-local 对齐数据；后者保存额外的米制手部和身体姿态字段，未自动接入训练。
world 手部原点在手的几何中心，pose world 原点在髋部中点，不能把这两套坐标直接当作同一个全局坐标系。
完整字段见 [米制采集说明](world-landmark-capture.md)。

## 冲突处理

- 所有运行入口统一为 `teleoperation.applications`，迁移 world 采集模块、CLI、直接运行脚本和测试引用；修复三机器人场景遗留的 `teleoperation.apps` 导入。
- MediaPipe 输入同时保留 `include_image` 和新增的 world/pose 数据读取，同一帧的两种检测使用相同时间戳。
- 保留当前双侧 checkpoint 必须成对提供的契约、独立单手模型支持和坐标校验；将旧测试适配到该契约。
- 保留 world 分支修改过的回归测试，并恢复它们依赖的测试夹具和坐标测试模块。
- 模型默认路径统一在 `datasets/`；下载的 hand/pose task 文件加入 ignore。
- README 的指尖距离权重更新为实际代码中的 `0.01`。

## 验证

环境：`D:\Anaconda\envs\teleoperation\python.exe`，使用集成目录 `src/` 作为 `PYTHONPATH`，`PYTHONUTF8=1`。
临时文件、日志和截图保存在集成目录的 `outputs/`，未提交模型权重或采集数据。

| 命令 | 退出码 | 结果与判据 |
|---|---:|---|
| `python -m unittest discover -s tests -v` | 0 | 130 项；128 通过，2 跳过，0 失败，0 错误 |
| `python devtools/verify_geometric_hand.py` | 0 | 左右手已知 FK 姿态往返，全部角度误差判据通过 |
| `python devtools/verify_three_robot_scene.py --out-dir outputs/tmp_integration/scene --report outputs/tmp_integration/scene/report.json` | 0 | 三台加载，左右各 30 个映射关节运动；固定关节漂移小于 0.05 rad |
| `python -m teleoperation replay actions --dummy --steps 120 --robot ROBOT --no-render` | 0 | `h1_2`、`gr1_t2`、`g1` 分别运行 120 步 |
| `python devtools/check_gbk_safe.py --strict` | 0 | 无 GBK 无法编码字符 |
| `python -m teleoperation hand record-world --help` | 0 | 新 CLI 命令参数可解析 |
| `python -m teleoperation.applications.realtime_hand_sim --help` | 0 | 实时仿真入口可解析、模块可导入 |
| `python record_world_landmarks.py --help` | 0 | 根目录直接运行入口可解析 |

两个跳过项分别需要本地 `hand_landmarker.task` 或默认 `palm_local_v2` checkpoint。
采集、world/pose 解析、同帧图像预览、H5 持久化、坐标和双 checkpoint 契约通过 mock/synthetic 回归验证；本次没有打开真实摄像头。
新增 TRON2A URDF 不等于实体标定或双臂 IK 已全面验收。

`CONVENTIONS.md` 和 `RETARGETING_PIPELINE.md` 中的历史机器路径、19 项测试数字与早期采集资源记录属于分支原始环境；本次合并的入口和验证结果以此文为准。
