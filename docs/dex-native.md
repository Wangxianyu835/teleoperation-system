# dex 原生手接入与阶段验收

范围：H1、GR1、G1 原生双手，Vector 优化，相机 world 与 Vision Pro 独立输入，PyBullet 仿真。
不连接实体机器人。DexPilot 留待下一轮。

## 实现索引

- `devtools/dex_native_env.ps1`：进程级 D 盘缓存与独立解释器路径。
- `devtools/bootstrap_dex_windows.py`：官方 wheel 的 Windows 依赖声明适配，不修改求解代码。
- `devtools/dex_native_acceptance.py`：阶段验收和原生手资产清单。
- `requirements-dex-native.lock.txt`：实验环境完整依赖锁定文件。
- `docs/dex-native-assets.json`：依赖版本、六手关节清单、资产及官方算法哈希。
- `docs/dex-native-runtime-lock.json`：Windows 编译运行时的官方 Conda 包 URL 和 SHA256。
- `docs/dex-native-results.json`：最终参数、运动学、600 姿态、性能及两份 VR 同帧比较结果。
- `robots/native_hand.py::NativeHandModel`：整机手部子树、固定指尖 frame、耦合限位与 FK 查询。
- `tests/test_dex_native.py`：本功能的契约及边界测试。
- `retargeting/hand/dex_native.py`：相机/VR 显式适配、固定掌长校准及原生 Vector 管线。
- `applications/dex_native.py`：校准、原生 H5 导出/回放与实时应用组合。
- `data/hand_recording.py::NativeHandResultWriter`：独立原生结果格式。

## 阶段记录

每阶段先保存 checkpoint，再运行对应验收并提交。结果位于 `outputs/tmp_dex_native/`。

| 阶段 | 修改前提交 | 检查点 | 状态 |
|---|---|---|---|
| 1 环境与基线 | fe9302e | checkpoint-20261027-1600-dex-stage1 | 通过 |
| 2 原生运动学 | 53c03c7 | checkpoint-20261027-1205-dex-stage2 | 通过 |
| 3 输入与校准 | 8ee68c7 | checkpoint-20261027-1210-dex-stage3 | 通过 |
| 4 Vector 求解 | 126f5e9 | checkpoint-20261027-1215-dex-stage4 | 通过 |
| 5 命令与回放 | 9470477 | checkpoint-20261027-1220-dex-stage5 | 通过 |
| 6 实录与比较 | 8a3bb97 | checkpoint-20261027-1500-dex-stage6 | 两份 VR 完整评估通过；相机校准及相机比较待新采集 |
| 7 实时入口 | 9c64049 | checkpoint-20261027-1238-dex-stage7 | 代码/离线联调/性能通过；相机与头显设备项待完成 |
| 8 文档与回归 | 130154d | checkpoint-20261027-1302-dex-stage8 | 文档、回归和 CLI 通过；全 preflight 仅有原有 C 盘缓存项失败 |

相机两个实时入口增加 `--backend dex --dex-calibration <camera.json>`，要求 world/robots；
仍复用既有相机窗口和双臂路线。摄像头 dex 模式使用单槽最新观测，不积累待处理队列。
VR `hand vr --backend dex --dex-calibration <vr.json> --scene robots --hand both --hand-mapping same`
使用同侧路由，录制回放传 `--input-h5`，实时传 `--ip`。dex 不加载 checkpoint，旧默认仍为 TransHandR。
VR 只控制原生手，不推造肩肘。`avp_stream` 当前独立环境尚未安装，设备实测不能标为通过。
SDK 默认依赖会额外安装 OpenCV 和其他仿真库；本次在下载阶段停止，避免覆盖既有 OpenCV。
实时 SDK 的兼容安装与真实连接仍是设备验收前提，不能用离线回放代替。
阶段 7 完整回归 465 项通过（1 原有跳过）；VR 新入口完成 150 帧六手回放。
双臂双手入口在合成混合输入下完成 35 帧，六手各执行 33 帧、手臂失败列表为空。
首两帧保留原有三帧窗口预热。单槽最新相机源的积压丢弃、退出及错误配置测试通过。
性能单测使用最终参数：100 帧预热 + 1000 帧，六手合计 P95 22.061 ms，均值 13.151 ms；
全部 600 个可达姿态重新通过 5/10 mm 原门槛。检测/渲染不计入该求解性能数字。
当前会话相机 0 实际探测 opened=False/read_frame=False；头显未连接，缺 SDK 和当前连接地址。
两次 5 分钟设备动作清单均未完成。

阶段 6 最终已完整导出两份 VR 实录（4631/2886 帧），六手在适配有效帧上的求解有效率均为 100%。
统计工具以 is_calibration 排除本文件校准片段，记录文件/评估索引哈希，分别计算原始解和执行命令的
指尖误差、拇食/拇中距离误差、限位命中、关节变化及耗时。指标是目标拟合误差，不是真机精度。
原 TransHandR 比较沿用上游 VR 预处理、实际权重与 L21 原生映射；旧映射不强加新的 mimic 约束，
比较结果格式为 `dex_native_comparison_v1`，不能作为新原生命令回放。
几何后端没有现有 VR 正式路线，不将相机预处理套用 VR；相机的几何/权重比较等待合格相机校准。
早期 v1 并行评估、v2 严格容差及默认容差但未加入平面可达性边界的性能记录均不是最终结果；
最终使用阶段 7 的 22.061 ms 独立求解测试。未降低 33 ms 门槛。
阶段 6 相机及阶段 7 真设备项未通过。用户同意后续配合相机采集，先继续不依赖该采集的 VR 与入口工作。

### 同帧比较：平均原始指尖目标拟合误差

单位 mm。A 为 `visionpro_hand_arm_20261012_213812_live.h5`，B 为 `visionpro_hand_arm_20261012_215137_live.h5`。
仅比较两法共同有效、且排除校准片段后的帧。JSON 另含执行命令、捏合距离、限位、关节变化和耗时。

| 原生手 | A dex | A 原 TransHandR + L21 映射 | B dex | B 原 TransHandR + L21 映射 |
|---|---:|---:|---:|---:|
| H1 左 | 51.9 | 69.2 | 56.0 | 73.2 |
| H1 右 | 51.0 | 67.9 | 58.3 | 74.0 |
| GR1 左 | 24.2 | 38.5 | 26.7 | 40.9 |
| GR1 右 | 23.7 | 37.3 | 27.8 | 40.7 |
| G1 左 | 37.0 | 69.2 | 38.0 | 71.0 |
| G1 右 | 36.8 | 69.3 | 38.1 | 71.5 |

这两份录制中，dex 对相同向量目标的拟合误差较小；不能推广成所有动作或抓取任务均优于原方法。
当前误差仍是厘米量级。固定掌长缩放后，人体指长比例和原生手低自由度/运动平面限制使部分目标不可达。
数值解有效不表示目标误差小，报告分别给出这两项，不能互相替代。

### 复现实录评估

先使用 README 4.4 节的校准/导出命令，再执行：

```powershell
& $dexPython -B devtools/dex_native_acceptance.py --stage recording `
  --input $vrNative --output "outputs/tmp_dex_native/metrics_$dexSession.json"
& $dexPython -B devtools/dex_native_acceptance.py --stage performance `
  --input $vrRaw --calibration $vrCal --output "outputs/tmp_dex_native/performance_$dexSession.json"
```

运动学和可达目标验收分别用 `--stage kinematics`、`--stage solvers`，均要求指定新 `--output` 路径。
旧方法比较使用 `--stage vr-comparison` 或 `--stage camera-comparison --method geometric|checkpoint`，
并传入 raw `--input`、固定 `--calibration`、诊断 `--result-h5`、JSON `--output`；checkpoint 方法还需真实 `--checkpoint`。
用 `--stage recording --common-with <另一方法结果.h5>` 可限定双方共同有效帧。
原始源文件、校准源 SHA256、排除片段、评估索引及配置哈希均保留在 JSON/H5 中。

`NativeHandCommandPipeline` 按独立驱动量滤波/限速，再展开所有耦合关节。
角速度上限取 CLI 与 URDF（含从动关节反推）约束的较小值。丢手保持命令，恢复首步仍限速。
`ThreeRobotHandScene.apply_native` 仅按关节名写入指定机器人和手侧，不经过 L21 降维映射。
本轮没有新增原生手网格避碰；限位、限速与耦合一致不等于无碰撞保证。
结果格式 `dex_native_results_v1`：根含秒制 timestamps、is_calibration；每个 `robot/side` 组保存
joint_names/config_hash、raw_angles、command_angles、valid、solve_ms、error、target_vectors。
无效原始解使用 NaN，执行命令保持上一帧有限值。回放检查名字、哈希、限位与耦合，不重复滤波。

命令：`hand dex-calibrate --source camera|visionpro --input raw.h5 --start <相对秒> --output calibration.json`；
`hand dex-export --source camera|visionpro --input raw.h5 --calibration calibration.json --output native.h5`；
`replay dex-native --file native.h5 --render`。路径均相对仓库根目录，使用独立环境运行。
`--start` 指向双手张开至少两秒的片段；校准与结果文件均不覆盖已有路径。
真实相机录制未找到满足条件的同时双手张开片段，用户已确认后续配合新采集，不能使用合成尺度冒充实录校准。
阶段 5：完整回归 462 项通过、1 项原有跳过；14 项 dex 契约/命令测试通过。
合成 world 数据完成校准、70 帧六手导出和整机无头回放；0/13/30 fps 回放入口均通过，
录制时间驱动的命令测试验证三种主机处理速率逐项相同，H5 导出/读取命令完全一致。

`NativeHandResult` 与 `NativeHandRetargeter` 是新增的原生关节接口；L21 结果接口不变。
Vector 使用官方 0.5.0 `VectorOptimizer`、`MimicJointKinematicAdaptor`、`SeqRetargeting`。
参数：scaling=1（尺度在固定校准边界处理），norm_delta=0、huber_delta=0.02、
ftol_abs=1e-6（官方默认）、xtol_abs=1e-6、maxeval=100、内部低通关闭；优化器边界 epsilon=0。
每只手独立初始化/重置。失败恢复该手上一次优化初值，返回无效状态，不将旧解伪装为新有效解。
G1 的突变目标可能落入拇指局部极小值。若目标变化超过 20 mm（或首帧）且最大残差超过 5 mm，
以零位、有效行程中点及上下界作有界确定性重试；各次仍调用相同官方优化器，保留最大指尖误差最小的解。
G1 食指/中指的源 URDF 验证为 Z 轴平行的两关节平面机构：若目标离该运动平面已超过 5 mm，
不执行无法达到 5 mm 恢复目标的多初值重试，仍返回首次官方优化解。此边界不缩放或更改输入目标。
最终参数和恢复策略通过 `VECTOR_PARAMETERS` 写入结果元数据，未修改已安装官方算法代码。
阶段 4：六手各 100 个可达目标全部满足平均误差 5 mm/最大单指误差 10 mm 门槛，
没有求解失败，重置重放在 1e-5 rad 内一致；完整回归 459 项通过，1 项原有跳过。

VR 源参考：`D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py:1136`，
SHA256 `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f`。
新增 `dex_visionpro_palm_v2` 在保留上游提取/轴变换/腕部相对化后，显式恢复左手手性，再增加 dex 掌部局部坐标转换。
上游左侧轴变换行列式为 -1（历史用途为左手输入驱动右 L21），dex 的原生左手路线在拟合掌部旋转前
增加 X 轴反射以恢复源手性；右侧轴变换行列式为 +1，无此额外操作。
修正前 v1 校准被拒绝，两份实录必须使用 v2 重新校准、求解和评估；v1 数值不能作为最终结果。
这属于新后端的输入适配，不改 `transhandr_axis_wrist_three_frame_v1` 或原模型权重。
相机使用 `dex_camera_world_v1`；normalized 输入不支持此后端。
校准至少两秒、每侧至少十个有效张开手样本；掌长取腕到 middle MCP 的中位数，求解时固定。
阶段 3：14 项契约/上游 VR 预处理测试通过，包含两源等价、刚体变换不变性、错误输入拒绝、单侧异常与校准持久化。

H1/G1 的固定指尖偏移取自本仓库 Inspire/Dex3 独立手 URDF；GR1 使用整机已有 tip frame。
H1 左手独立模型的食指 intermediate 原点与整机存在约 0.19 mm 差异：保留整机原点，仅移植末端固定偏移。
优化模型删除 visual/collision 资源引用，关节轴、原点及 mimic 保持整机定义；显示仍使用原整机模型。
阶段 2：六手各 100 个姿态（含耦合有效上下界）FK 位置误差均小于 0.1 mm，
全部查询恢复原整机位置和速度。新模型契约测试 5 项通过；原生回放和手部命令测试通过。

独立环境路径：`.venv/dex-native`。以现有正式环境的独立包副本为起点，再安装 dex 依赖；不修改正式环境。
原生手 URDF 的 mimic 从动关节不是独立优化变量：H1 为 6/12，GR1 为 6/11，G1 为 7/7。

Windows 的 PyPI 没有 `pin>=3.3.1` wheel。绑定由独立运行时 `.venv/dex-pin-bundle` 提供。
运行时从官方 Conda 包复制 Python 绑定及 DLL；不运行有注册副作用的环境安装事务。
本地构建标为 `dex-retargeting==0.5.0+winconda`：仅调整版本标记和 Windows 的 `pin` 依赖声明，
包内算法代码逐文件保持官方 wheel 字节不变。真实 Pinocchio 版本另外验证，不伪造 `pin` 包元数据。
wheel 来源、SHA256 与代码哈希记录在 `outputs/tmp_dex_native/wheels/*.provenance.json`。
正式项目环境不修改，缓存位于 D 盘。中断的 Conda 安装保留在被忽略目录中，不参与运行。

### 环境重建

先在仓库根目录加载 `devtools/dex_native_env.ps1`。创建 `.venv/dex-native`，
从正式环境复制独立 site-packages，或按完整 pip 锁定文件安装基础包。
本地 dex wheel 必须先按下列步骤构建，再使用完整锁定文件，不能从 PyPI 查找本地版本。
已验证的是本机正式环境的独立包副本、锁定 Conda 编译运行时的重建和导入；
未单独执行无本机环境依赖的全新网络安装。CUDA 版 Torch 的来源需与锁定版本相匹配。

```powershell
. ./devtools/dex_native_env.ps1
# Anaconda 基础 Python 已包含 conda_package_handling，按 URL/SHA256 重建编译运行时。
& D:/Anaconda/python.exe -B devtools/bootstrap_dex_windows.py `
  --restore-lock docs/dex-native-runtime-lock.json --cache outputs/tmp_dex_native/cache/pip/lib/conda `
  --runtime .venv/dex-pin-bundle
& $dexPython -m pip download --no-deps dex-retargeting==0.5.0 -d outputs/tmp_dex_native/wheels
& $dexPython -B devtools/bootstrap_dex_windows.py `
  --wheel outputs/tmp_dex_native/wheels/dex_retargeting-0.5.0-py3-none-any.whl `
  --runtime .venv/dex-pin-bundle --environment .venv/dex-native
& $dexPython -m pip install --find-links outputs/tmp_dex_native/wheels -r requirements-dex-native.lock.txt
& $dexPython -B -m pip check
& $dexPython -B devtools/dex_native_acceptance.py --output outputs/tmp_dex_native/rebuilt-assets.json
```

原环境基线：448 项，0 失败、0 错误、1 原有跳过，146.519 秒。
独立实验环境基线：448 项，0 失败、0 错误、1 原有跳过，146.893 秒。
最终 Pinocchio 运行时接入后的完整基线再次通过：448 项，1 原有跳过，144.568 秒。
四条 CLI 冒烟退出码均为 0，pip check 无冲突，六手资产清单检查通过。
按 URL/SHA256 锁定文件在 `.venv/dex-pin-rebuilt` 重建编译运行时并独立导入验证成功。
官方 dex 0.5.0 发布源码提交为 `3f56141bc8bd2760d5e452e382937269554ebb21`，
安装后所有包内 Python 源码哈希与官方 wheel 逐一核对一致。
沙箱内首次测试遇到文件原子替换权限错误；允许该操作后原测试通过，不修改代码或测试来规避。
快速 preflight 仍报告用户原有 C 盘 pip/HuggingFace 缓存；本任务的七个进程缓存变量已重定向 D 盘，不删除用户缓存。

## 最终回归与交付状态

- 独立 dex 环境：完整 `unittest discover -s tests -v`，465 项，0 失败、0 错误、1 原有跳过。
- 原正式环境：相同完整套件 465 项、1 原有跳过，85.165 秒，四条 CLI 冒烟均 exit 0。
- 不安装 dex 时，旧 CLI/工作流继续运行；显式选择 dex 会返回包含环境文档路径的依赖错误。
- 两个 GBK 扫描、架构边界、`git diff --check` 通过；README 18 段 PowerShell 语法和 24 个链接通过检查。
- 完整 preflight exit 1，仅 C 盘原有 pip 缓存约 5183 MB/HuggingFace 残留导致一类失败。
  七个进程变量均重定向 D 盘；没有清理用户缓存或修改用户级环境。
- 两份 VR 实录全量导出和同帧比较完成，最终结果写入 `dex-native-results.json`；截图位于
  `outputs/tmp_dex_native/snapshots/`，是接受命令姿态的可视化，不能当作硬件实测。
- 未完成项：合格相机校准与相机方法比较、兼容头显 SDK 安装、相机与头显各 5 分钟动作清单。
  相机 0 未读到画面；用户已同意后续配合采集。因此本轮不能宣称全部八阶段验收完成。

本轮代码提交：阶段 1 `53c03c7`、阶段 2 `8ee68c7`、阶段 3 `126f5e9`、阶段 4 `9470477`、
阶段 5 `8a3bb97`、阶段 6 评估工具 `9c64049`、阶段 7 集成/修正 `130154d`。
本轮未推送 GitHub。阶段 8 提交包含 README、最终结果清单和上述待验收事项。
