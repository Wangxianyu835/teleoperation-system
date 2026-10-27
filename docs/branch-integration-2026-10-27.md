# 2026-10-27 分支整合记录

## hand_capture

来源：`6e2743c05b18878db69ab59fd0a4ab5b57e1b3fe`。合并前主线：`fde0c9ccb77ba3e5bc151107af16010e13138e5a`。
检查点：`checkpoint-20261027-0952-merge-capture-arm`。

通过合并提交保留分支历史。以下路径适配用于保留当前生产包、CLI 和坐标契约：

| 上游路径 | 当前落点或处理 |
|---|---|
| MediaPipe/ 的四个 Python 工具 | 保留独立 normalized 采集及回放；模型改用 datasets/，结果改用 outputs/hand_capture/；补充坐标和时间声明，控制台文本改为 GBK 可编码字符 |
| MediaPipe/*.task、requirements.txt | 不重复提交模型二进制和空依赖文件；使用 datasets/ 中的模型及 requirements-retargeting-camera.txt |
| scripts/check_native_hand_motion.py、render_hand_snapshots.py、make_twohand_raw_sample.py | 迁入 devtools/，改用当前机器人加载器、后处理及显式 18 维到 17 自由度映射 |
| scripts/run_retargeting_pipeline.py | 迁入 devtools/，调用当前 teleoperation hand / tools CLI；显式输入缺失时报错，校验失败停止后续流程 |
| demo_hands_three_robots.py | 转发至 tools show-all-hands |
| teleop/native_hand.py、scripts/replay_hand_native.py | 合入 link_extent 和按手部几何选取特写视角；保留当前动作类型、精度与时间戳回放接口 |
| scripts/verify_hand_pipeline.py | 任意录制不再强制命中参考数据的第 535 帧；参考文件继续执行原断言 |
| input_adapters/hand_keypoints.py | 不恢复旧根目录；其数据流职责由当前 contracts、retargeting 和 inputs 实现 |
| scripts/check_no_c_drive.py、env_e_drive_cache.ps1、preflight.ps1 | 保留当前 devtools/ 中的已有版本 |
| 旧 README、环境/入门/项目文档、规则文件、依赖约束 | 保留当前目录与环境说明，将新用法补充到中文 README；不采用旧的宽松 NumPy 依赖约束 |
| tests/conftest.py | 不加入针对已退役 test_dual_arm.py 的 pytest 跳过钩子；保留当前 unittest 资产检查 |
| .gitignore、LICENSE | 合入临时实验目录忽略规则及上游 Apache-2.0 许可证文本；第三方资产许可仍以各资产文件为准 |

原始数据、模型权重和输出之间的坐标声明必须匹配。旧归一化身体点不是 Pose world 米制手臂输入。

### 上游文件 SHA256

| 路径 | SHA256 |
|---|---|
| `MediaPipe/body capture media.py` | `6e4c5b9a454335055a7dbbba53a40fe02f7a8e35f59821ee99a0d9a43a88029b` |
| `MediaPipe/body picture.py` | `c21fde402ea9a65fef55f0570b7426b6171ec09e2843d1305bffc6a66b0a4b7f` |
| `MediaPipe/hand capture media.py` | `d38d3f3059cbd26de193d21175dfb9bdaa9168589c1e077646c2178a4a9872b8` |
| `MediaPipe/read_hand_xyz.py` | `f086f1b0376b5cbf883b4cc26a89c109d3f551c5fd5309beea582b55d9b0e559` |
| `scripts/check_native_hand_motion.py` | `945ab60af93a511116ca2bfe4d5a2dc23c05de31cf1ed42344abf2d584cbb15c` |
| `scripts/render_hand_snapshots.py` | `c9b9326c1c7d255dbb2455a22b0a5a8e30b765ab8b2b1d9f57f79601f6a0127f` |
| `scripts/make_twohand_raw_sample.py` | `cf4259a63a7d770c1f992719fa3efd7f4279be4a687d12e1ac901dd94d0c8857` |
| `scripts/run_retargeting_pipeline.py` | `99143555eabdee99d75d43042c8024653dbccef5001c2a3de66c147617407bfe` |

## 验证基线

合并前完整 unittest：344 项，1 项跳过，退出码 0。四条 CLI 冒烟均退出 0。
完整 preflight 返回 1，唯一失败类别为已有 C 盘缓存残留（pip 约 5.2 GB，另有 HuggingFace 残留）。本次进程的七个缓存和临时目录变量全部指向项目 D 盘，未清理或搬移用户原有缓存。
日志：`outputs/tmp_merge_capture_arm/baseline_preflight.log`。

后续双臂整合与最终验证结果见下文。

hand_capture 合并验证：`python -m unittest discover -s tests` 运行 348 项，1 项跳过，无失败，退出码 0。摄像头/录制/原装手专项 24 项通过。三机器人原装手快照使用已录制摄像头角度生成，退出码 0。
