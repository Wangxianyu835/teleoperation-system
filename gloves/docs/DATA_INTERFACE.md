# FFG 数据交接接口

## 当前职责

采集项目向重定向模块提供：

- FFG 21 通道手部位置数据；
- 主机时间戳和帧序号；
- 有效性与设备状态；
- 设备型号、串口、波特率和版本；
- FFG 原始标定锚点；
- 质量报告和可视化图。

采集项目不决定人手到机器人的关节映射。

## H5 根属性

```text
schema_version = linker_motion_capture_v1
metadata_json
sample_count
created_at_utc
closed_at_utc
```

## FFG 数据流

路径：

```text
/streams/ffg_right
```

数据集：

| 名称 | 形状 | 含义 |
|---|---|---|
| `host_time_ns` | `(N,)` | 主机单调时钟，纳秒 |
| `device_time_ns` | `(N,)` | 设备时间；当前 FFG 协议没有时为 `-1` |
| `sequence` | `(N,)` | 主机帧序号 |
| `positions_rad` | `(N,21)` | FFG 原始位置，固定通道顺序，弧度 |
| `realforce` | `(N,5)` | FFG 返回量；不可用时为 NaN |
| `realforce_valid` | `(N,)` | 力反馈返回量是否可用 |
| `valid` | `(N,)` | 本帧是否有效 |
| `status` | `(N,)` | `tracking`、`disconnected` 等状态 |

## 通道顺序

当前按设备协议保存 21 路有序通道：

```text
0..4    拇指侧摆/屈曲/指尖
5       食指侧摆
6..8    食指屈曲
9       中指侧摆
10..12  中指屈曲
13      无名指侧摆
14..16  无名指屈曲
17      小指侧摆
18..20  小指屈曲
```

可视化通道别名保存在：

```text
/streams/ffg_right.attrs.channel_names_json
```

这些别名用于观察，不应在未确认设备解剖定义前作为重定向真值。

## 标定

标定格式：

```text
ffg_raw_calibration_v2
```

包含：

```text
open.positions_rad
fist.positions_rad
opose.positions_rad
```

以及各通道标准差、平均方差、警告通道、样本数量和稳定性。
旧版 `ffg_raw_calibration_v1` 仍可读取。

## 后续接入建议

通用生命周期接口：

```python
SampleSource
    stream_id
    start()
    read(timeout_s)
    latest()
    stop()
    is_running
```

FFG 的 `read()` 返回 `FFGSample`，并额外满足
`FFGSampleSource`。未来手臂采集源实现相同的 `SampleSource`
生命周期，但可以返回自己的 `ArmSample`，无需伪造 21 通道 FFG 数据。

首版集成不需要修改 FFG 采集实现，只需：

1. 读取 H5 `positions_rad`；
2. 根据重定向负责人确认的通道语义处理；
3. 将输出接入其关节求解器。

## 实时采集接口

`05_realtime_capture.py` 持续调用同一个 `FFGSource`，并通过
`latest()` 提供最新有效帧。默认不保存正式 H5，只生成实时运行报告。

下游实时模块可以直接使用：

```python
sample = source.latest()
if sample is not None and sample.valid:
    # 将 sample.positions_rad 交给重定向模块
    pass
```

当前先采用进程内接口，不引入 ROS 或网络消息层。

## 未来手臂扩展

手臂使用独立的流：

```text
/streams/linker_ta_right
```

各流保存自己的时间戳，后续按时间戳对齐，不要求手部和手臂拥有相同的帧数。

