# outputs 清理记录（2026-10-29）

用户选择：删除缓存、旧日志和确认可再生成的一次性测试产物，保留权重。
本次没有修改运行代码、算法、公共接口或环境配置。

## Git 检查点

- 清理前提交：`0027f0bd63e48aaa7dde87a036082cc69c76a078`。
- 注释标签：`checkpoint-20261029-1738-output-cleanup`。
- 该提交按仓库规则保存了清理前已有的手部重定向改动，不代表本次重新验证了全部既有改动。
- `outputs/` 已被 `.gitignore` 排除。Git 检查点只保护源码，不能恢复已删除的运行产物。

## 已删除

共删除 **59,775 个文件、2,571,710,155 字节（约 2.40 GiB）**，并移除 56 个空目录。
这里统计的是文件逻辑大小，不是文件系统实际空闲空间增量。

缓存目录：

- `outputs/tmp_dex_native/cache/`：pip HTTP 缓存、Conda 下载包和解压副本、包索引、失败下载残片及临时构建目录。
- `outputs/tmp_runtime_cache/`。
- `outputs/tmp_arm_diagnostics/status_cache/`。
- `outputs/tmp_interface_integration/mpl/`、`MPLCONFIGDIR/`。
- `outputs/tmp_merge_validation/matplotlib/`。
- `outputs/tmp_hand_command_guard/matplotlib/`。
- `outputs/tmp_g1_visibility/matplotlib/`。

另外删除 362 个旧日志及测试文件，合计约 68.88 MiB：

- `outputs/` 内原有的 `.log` 文件，包括旧测试日志和训练终端日志。
- `tmp_arm_diagnostics` 下的合成验收及 legacy36 导出结果。
- `tmp_merge_capture_arm/acceptance_final/`、`acceptance_merged/` 下的合成测试结果。
- `tmp_g1_visibility`、`tmp_arm_follow`、`tmp_readme`、`tmp_task_acceptance`、`task_smoke_20261028_211919` 下的一次性检查产物。
- `tmp_merge_validation` 下的合成帧、自测 H5、JSON 报告和临时验证脚本；保留其中的 `.pth`。
- `tmp_dex_native` 下以 `synthetic_` 开头的合成测试文件。
- `tmp_interface_integration` 下的 CLI help 文本。

历史文档中引用已清理日志或合成验收结果的路径现在可能不存在；文档内已记录的结论保留。
需要新结果时可重新运行相应 devtools 验证命令。

## 保留及判定依据

- 保留全部 **21 个 `.pth` 文件**，包括 12 个历史训练权重和 9 个测试或小网络权重，避免替用户判断训练批次是否废弃。
- 保留真实相机和 VR 录制、导出 H5、标定 JSON，以及用途不够明确的 NPZ 和实验脚本。
- 保留训练 CSV、TensorBoard 事件和模型配置，便于理解或继续使用保留的权重。
- 保留 `tmp_dex_native/wheels/` 和 `vr_environment/` 内的安装包。`requirements-dex-native.lock.txt` 引用了本地 wheel；`devtools/dex_native_acceptance.py` 需要 wheel provenance。
- 保留 `tmp_dex_native/models/`、真实设备验收报告、相关截图、分支比较 patch 和其他尚有复核价值的记录。
- `tmp_interface_integration/real_camera_smoke_v1/`、`real_camera_smoke_v2/` 混有真实录制切片和权重，因此保留其数据、权重及报告。

删除 dex 缓存前检查了 `devtools/bootstrap_dex_windows.py` 和
`.venv/dex-native/Lib/site-packages/dex_conda_runtime.pth`：运行时载荷已复制到
`.venv/dex-pin-bundle/`，不是从 `outputs` 的下载或解压缓存加载。
`devtools/dex_native_env.ps1` 会在需要时重新创建缓存目录；运行时重建方法见 `docs/dex-native.md`。

## 清理后验证

- 对保留的 209 个文件逐一比较清理前后的 SHA-256：全部一致。
- 计划删除的文件及缓存目录全部消失，没有剩余不可读取目录。
- 验证时剩余 209 个文件，共 2,367,247,477 字节（约 2.20 GiB）。
- 清理前后 dex 环境均可导入 `pinocchio`、`nlopt`、`dex_retargeting`、`torch`、`mediapipe`；Pinocchio 为 3.3.1，实际路径位于 `.venv/dex-pin-bundle/`。
- 使用 `.venv/dex-native/Scripts/python.exe -B -m unittest tests.test_dex_native tests.test_coordinate_pipeline tests.test_hand_control_cli`：**33 项通过**。
- 本次只清理运行产物并记录结果，没有修改共享行为，因此未运行完整 unittest discover。

以后不要直接按 `tmp_*` 名称整体清空：部分目录实际存有真实录制、标定、安装包和模型权重。
