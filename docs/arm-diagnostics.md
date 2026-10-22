# 三机器人双臂诊断与离线验收

本轮增加可选诊断和可重复运行的验收，不增加平滑、不调整增益或行程、不改变既有控制输出。
对象为 H1-2、GR1-T2、G1 的六条手臂。Vision Pro 数值链路和手指模型保持原实现。

## 1. 当前结论

**工具和回归测试通过，但完整合成场景的空间验收未全部通过。**
在大幅运动触发行程夹取后，H1 右臂回到中性目标仍有明显误差，30 Hz 和 13 Hz 均可重复。
验收脚本保留原来的 30/25/25 mm 阈值，失败时返回退出码 1，并保存结果。

| 场景 | 30 Hz：H1 右臂稳态平均误差 | 13 Hz：H1 右臂稳态平均误差 | 判据 |
|---|---:|---:|---|
| 大幅夹取后回中性（settle） | 104.28 mm | 105.32 mm | 失败，超过 30 mm |
| 随后的腕姿态降级（pose_fallback） | 86.70 mm | 86.73 mm | 失败，超过 30 mm |

每个频率共 89 项检查，87 项通过、2 项失败。其他手臂的空间检查通过。
姿态降级场景紧接在大幅夹取之后，这些结果不能独立证明腕姿态本身是根因。

六条手臂的丢帧目标保持检查通过，保持期间目标变化为 0。
恢复首帧的最大目标跳变：H1 为 17.44 mm，GR1/G1 为 9.96 mm；这些是观察值，没有设置未经验证的合格阈值。

该问题在没有摄像头的确定性场景中可复现。
在最后的姿态降级帧，H1 右臂关节命令与实际关节的最大差约 0.0026 rad，但末端位置误差仍约 86.7 mm。
据此，下一轮应优先检查大幅夹取后的目标可达性、IK 解分支和关节约束；这是定位方向，尚未确认根因。

## 2. 环境与离线命令

使用既有 Conda 环境和本地机器人资产。以下设置只影响当前 PowerShell 进程，缓存和临时文件位于项目 D 盘目录。

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
$python = 'D:\Anaconda\envs\teleoperation\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$cacheRoot = Join-Path (Get-Location).Path 'outputs/tmp_arm_diagnostics/cache'
$cacheMap = @{
  TEMP='tmp'; TMP='tmp'; MPLCONFIGDIR='matplotlib'; TORCH_HOME='torch'
  HF_HOME='huggingface'; XDG_CACHE_HOME='xdg'; PIP_CACHE_DIR='pip'
}
foreach ($entry in $cacheMap.GetEnumerator()) {
  $path = Join-Path $cacheRoot $entry.Value
  New-Item -ItemType Directory -Force -Path $path | Out-Null
  Set-Item -Path ('Env:' + $entry.Key) -Value $path
}
$session = Get-Date -Format 'yyyyMMdd_HHmmss'

& $python devtools/verify_arm_diagnostics.py `
  --output-dir "outputs/arm_validation/$session"
```

不指定输入时，工具以固定种子 `20261022` 生成 30 Hz 和 13 Hz 场景，依次驱动三台机器人的六条手臂。
运行目录必须不存在，以免覆盖以前的验收。也可以不传 `--output-dir`，使用自动日期命名。

生成文件包括：

- `summary.json`：总体状态、失败项、下一步建议。
- `synthetic_30hz.json` / `synthetic_13hz.json`：每只手臂的 `arm_overview`、分场景指标和检查结果。
- 同名 `.csv`：逐帧、分机器人和左右侧的明细。
- 同名 `.npz`：本轮合成输入，明确标注合成来源、种子、秒制时间、米制坐标、点顺序及场景区间。

退出码 0 表示当前输入的适用检查通过；1 表示存在失败检查或输入/执行错误。
输出中的 `observation` 表示仅供观察或没有可用样本，不算通过证据。
当前固定场景返回 1 是已发现的 H1 右臂问题，不能据此把结果称为全部通过。

## 3. 输入、场景和指标定义

场景包含：中性启动、静止、带噪声静止、左臂移动、右臂移动、双臂移动、人体整体平移、
单侧丢帧与恢复、双侧丢帧与恢复、双臂阶跃、大幅夹取、回中性及 H1 姿态降级。
每个频率约 31 秒；分别为 930 帧和 402 帧。

离线验收使用显式样本时间驱动现有物理步数时钟，与计算机运行速度无关；
仍保留 100 ms 的单次补偿上限。它没有修改原演示入口的固定步数离线回放规则。

空间误差检查对指定稳态场景中、开始 0.75 秒以后、已标定、输入有效且未被行程夹取的帧取平均。
H1 上限为 30 mm，GR1/G1 为 25 mm。夹取压力场景的误差另行记录，不冒充正常可达运动。
工具还检查行程边界、现有最小运动行程、非驱动关节漂移不超过 0.02 rad、人体平移不改变目标和缺帧保持。

抖动记录为静止区间相对均值位置的 RMS 距离，同时保留关节相邻帧变化量。
阶跃响应记录为末端相对阶跃前实际位置，在目标位移方向上首次达到 90% 的样本时间差。
该数值有一个输入采样间隔的时间分辨率；`null` 表示没有相应场景或没有达到 90%。
恢复跳变是恢复首帧目标与前一保持目标的距离。
三类指标均用于建立基线，不等于真人精度或体感延迟的合格结论。

导入旧 NPZ 时，必须保留 `timestamps`、`left_arm`、`right_arm`；手臂数组形状为 `(T,3,3)`，
顺序为肩、肘、腕，坐标沿用既有录制协议的 Pose world 米制。手部数组可缺失。
没有时间单位声明的旧文件必须显式选择单位，不按数值大小推断，也不修改原文件。

```powershell
# realtime_teleop 的 --dump-npz 保留输入层的 Unix 毫秒
& $python devtools/verify_arm_diagnostics.py `
  --input-npz "outputs/tmp_realtime/arm_raw_$session.npz" `
  --timestamp-unit milliseconds `
  --output-dir "outputs/arm_validation/import_$session"
```

对于时间本来就是秒的旧文件使用 `--timestamp-unit seconds`。
自带单位的合成文件无需此选项；冲突声明会被拒绝。
只有手部关键点的 H5 无法用于双臂验收。导入的无场景标注数据不能证明完整合成场景覆盖。

## 4. 后续真人复测

准备既有 MediaPipe 手部/Pose 模型以及模型权重后，在上面的环境设置所在终端运行：

```powershell
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
& $python -m teleoperation.simulation.realtime_teleop `
  --side both --no-arm-fallback --diagnostics --arm-trace `
  --diagnostic-trace "outputs/tmp_realtime/arm_trace_$session.csv" `
  --report "outputs/tmp_realtime/arm_report_$session.json" `
  --dump-npz "outputs/tmp_realtime/arm_raw_$session.npz"
```

不限制 `--robot` 时同时驱动三台机器人。先保持中性姿态完成原有 30 帧标定，
再做静止、分别移动、双手移动、短暂出画和恢复。这里显式关闭手部位置兜底，便于检查 Pose 的缺失与恢复。
原入口默认兜底策略没有改变。

`--diagnostics` 默认关闭；`--diagnostic-trace` 自动启用诊断，CSV 必须是新文件。
诊断不会额外调用 IK、重置关节或推进物理。固定步数回放已验证诊断开关前后的目标、
电机命令、实际关节、IK 次数和物理步数完全一致。
实时运行会承担测量和文件写入成本，应查看独立的 `diagnostic_overhead`，不能宣称零性能开销。

JSON 的 `diagnostics.timings_ms` 提供计数、平均和最大耗时；CSV 的 `timings_ms` 保存当前帧阶段记录。
主要阶段为 `capture_detection`、`hand_preprocessing`、`hand_inference_output`、`arm_mapping`、
`ik.<robot>.<side>`、`motor.<robot>.<side>`、`physics` 和 `preview`。
采集与检测作为整体测量，不修改 MediaPipe 输入实现。未运行的阶段不伪造耗时为零。
状态读取、命令诊断副本和 CSV 写入计入诊断开销；离线漂移扫描也单独计入诊断开销。

CSV 向量位置单位为米，关节角单位为弧度。`input_valid`、`valid`、`held`、`command_accepted`
分别区分原始输入有效、映射有效、保持和本帧电机命令是否被接受；`position_fallback` 标注姿态降级。
实时 `raw_timestamp` 是 Unix 毫秒，`sample_time_s` 是程序中的相对单调时间。
旧演示回放缺少单位声明时，诊断 CSV 明确写 `unspecified_legacy`，不生成推断的样本秒数。

处理耗时使用主机计时；它与摄像头端到端延迟、运动响应延迟不同。
离线验收只测量映射、IK、电机下发和无头物理处理，不执行摄像头检测或手指推理。

采集异常时会关闭 CSV，若指定 JSON 报告则保存已有诊断摘要及错误。
诊断输出禁止与显式输入路径或其他输出路径重合；正常输出仍使用既有应用行为。

## 5. 本机验证记录（2026-10-22）

- 修改前：330 项测试，1 项跳过，无失败。
- 诊断提交：336 项测试，1 项跳过，无失败。
- 完成验收工具后：344 项测试，1 项跳过，无失败；GBK 和差异空白检查通过。
- 30/13 Hz 完整场景各运行两次；5,580/2,412 行 CSV 除耗时外的字段逐项完全一致。
- 原有 36 帧样例在显式秒制声明下导入通过，21 项检查无失败；没有丢帧样本的项目只标观察。
- 完整场景每个频率仍有上述两项 H1 右臂失败，工具退出码为 1。
- 目标保持变化最大值为 0；非驱动关节最大漂移为 0.0016/0.0017 rad。

可复现检查：

```powershell
& $python -m unittest tests.test_arm_diagnostics tests.test_arm_validation
& $python -m unittest discover -s tests
& $python devtools/check_gbk_safe.py --strict
```

本次运行产物位于 `outputs/tmp_arm_diagnostics/acceptance_final/`，不提交 Git。
**真人摄像头效果尚未复测。**

| 阶段 | 修改前提交 | 修改前检查点 |
|---|---|---|
| 诊断采集 | `6f2b1b9` | `checkpoint-20261022-2047-arm-diagnostics` |
| 离线验收 | `e0d9c4d` | `checkpoint-20261022-2053-arm-offline-validation` |
| 结果说明 | `1a075a2` | `checkpoint-20261022-2108-arm-diagnostics-docs` |
