# Linker FFG 独立数据采集

本项目负责 FFG 手部数据采集、标定、H5 保存和质量检查。

- 不依赖相机或 VR；
- 不启动灵巧手重定向；
- 不控制实体机器人；
- 预留后续手臂和实时重定向接口。

## 快速开始

PyCharm 解释器：

```text
<项目目录>\.venv\Scripts\python.exe
```

首次安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

如果默认 PyPI 超时：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt `
  -i https://mirrors.aliyun.com/pypi/simple
.\.venv\Scripts\python.exe -m pip install -e . --no-deps `
  -i https://mirrors.aliyun.com/pypi/simple
```

## 正常使用

离线数据采集使用以下四个主要脚本：

| 顺序 | 脚本 | 作用 |
|---|---|---|
| 1 | `scripts/01_check_ready.py` | 检查环境、串口、版本、21 通道和动作变化 |
| 2 | `scripts/02_calibrate_ffg.py` | 完成握拳、O 型、张手标定 |
| 3 | `scripts/03_collect_offline.py` | 实时读取并保存正式离线 H5 |
| 4 | `scripts/04_review_data.py` | 自动检查最新数据并给出 PASS/WARN/FAIL |

辅助入口：

| 脚本 | 作用 |
|---|---|
| `scripts/00_offline_self_test.py` | 没有手套时验证代码和生成示例数据 |
| `scripts/05_realtime_capture.py` | 实时采集、最新帧接口和简化实时显示 |

所有脚本均可在 PyCharm 中直接右键 Run。`.run` 目录也提供对应运行配置。

## 配置

默认配置：

```text
configs/ffg_capture.yml
```

默认自动识别单只左手或右手，使用有线 `2000000` 波特率，并跳过蓝牙串口：

```yaml
device:
  side: auto
  port: auto
  baudrates: [2000000, 460800]
  serial_query_interval_ms: 6
  skip_bluetooth_ports: true
```

自动扫描失败时填写实际端口：

```yaml
device:
  port: COM5
```

只连接一只手套时建议保持 `side: auto`。以后同时连接左右手时，应明确配置左右端口，避免自动选择产生歧义。

当前设备在 `serial_query_interval_ms: 6` 下实测约 `95.9 Hz`，
20 秒连续测试无无效帧、无超过 12 ms 的帧间隔。若现场出现异常，
可将该值提高到 `10`、`15` 或 `20` ms 以降低查询压力。

实时采集配置：

```yaml
realtime:
  display_enabled: true
  display_refresh_hz: 12
  status_interval_s: 2.0
  duration_s: 0
  debug_recording: false
```

实时采集默认不保存 H5，只持续更新 `FFGSource.latest()`。
需要调试录制时设置：

```yaml
realtime:
  debug_recording: true
```

离线采集默认持续到 `Ctrl+C`。需要定时采集时：

```yaml
recording:
  default_duration_s: 60
```

## 数据结论

`03_collect_offline.py` 和 `04_review_data.py` 只显示六个核心指标：

- 实测帧率；
- 无效帧比例；
- 最大帧间隔；
- 动作范围；
- 静止抖动；
- 端点饱和比例。

结论：

```text
PASS：数据可以用于后续重定向
WARN：数据可用，但建议检查动作、连接或参数
FAIL：数据不完整，不能交付
```

质量门槛位于 `configs/ffg_capture.yml` 的 `quality` 部分。当前门槛用于采集链路健康检查，不能替代最终重定向任务验收。

## 数据文件

每次采集生成：

```text
outputs/sessions/<stream>_<time>.h5
outputs/sessions/<stream>_<time>.summary.json
outputs/plots/<stream>_<time>.png
outputs/logs/ffg_collect_offline.log
```

H5 格式版本：

```text
linker_motion_capture_v1
```

主要数据：

```text
host_time_ns
sequence
positions_rad      形状 (N,21)
realforce          形状 (N,5)
valid
status
```

数据交接说明见 `docs/DATA_INTERFACE.md`。

## 标定

FFG 标定结果为原始 21 通道锚点：

```text
outputs/calibration/ffg_right_calibration_*.json
```

当前标定不包含目标机器人手映射，重定向由后续模块负责。

标定流程会在切换到姿势后先等待约 1 秒，再要求连续稳定 5 秒。
是否通过使用 21 个通道的平均方差判断；单个通道（尤其拇指通道）
抖动较大时会显示警告，但不会单独导致整个标定失败。

可视化把屈曲和侧摆分开处理：

- 主曲线只使用每根手指的屈曲通道；
- 侧摆通道在质量图中单独显示；
- 质量结论中的动作范围和静止抖动以屈曲通道为主。

通道名称已按官方定义区分，例如 `index_roll` 是食指侧摆，
`index_root_flex/index_middle_flex/index_tip_flex` 是食指屈曲。

对应参数：

```yaml
calibration:
  settle_seconds: 1.0
  stability_window_seconds: 1.0
  hold_seconds: 5.0
  max_mean_variance: 0.03
  channel_std_warning_rad: 0.20
  max_wait_seconds: 20.0
```

## 测试

完整离线测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

没有手套时运行：

```text
scripts/00_offline_self_test.py
```

它会生成合成 H5、简要图和质量报告。

## 常见问题

### 只找到蓝牙 COM 口

程序会自动跳过蓝牙串口。确认 FFG USB 数据线已连接，并在
`configs/ffg_capture.yml` 中填写实际 `COMx`。

### 串口被占用

同一时间只能有一个程序打开 FFG 串口。关闭其他串口工具或采集程序。

### 没有动作变化

运行就绪检查时需要真实执行张开和握拳。手部静止只会产生微小噪声。

### 通道名称不确认

21 个值的顺序是稳定的，但解剖别名仍需结合设备标定确认。原始数据以固定通道顺序保存。

## 第三方代码

FFG 串口协议来自 LinkerBot `linkerhand-telop-ros2` 的 MIT 许可实现。
来源和版本见：

```text
third_party/linkerhand_telop_ros2/VENDORED.md
```

