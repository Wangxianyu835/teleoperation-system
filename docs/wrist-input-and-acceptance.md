# 相机与 Vision Pro 腕朝向输入和验收

相机腕旋转来自 SMPL-X 完整 FK 链，与身体位置一起经过跟踪器校验。
MediaPipe Hands 提供手指数据。相机运行和回放见 [SMPLer-X 链路](smplerx-arm.md)。

MediaPipe 不再生成独立掌面/腕四元数，手部观测的 `wrists` 为 `None`。
旧 `camera_palm` 来源在回放及控制边界被停用。手指坐标对齐仍使用掌部局部基，
三帧窗口、掌长归一化和左右手规范保持原样。
删除清单、历史数据兼容与验证见 [退役记录](mediapipe-wrist-removal.md)。

`WristSceneController` 为每侧维护 `WristTracker`，收集稳定中性姿态并计算相对旋转。
每台机器人记录自己的初始腕姿态，结合基座旋转生成世界坐标中的目标四元数。
会话或模型身份改变时重置标定；真正失效或过期清空未完成的稳定窗口。
异步等待不重复计数，也不清空尚未完成的有效窗口。

Vision Pro 使用专用来源和路由，不应用相机预处理。腕观测和机器人执行属于边界适配，
上游手指数值流水线及权重元数据有独立契约。

## 坐标与控制

本地 SDK 腕矩阵已执行 YUP2ZUP，输出 +X=右、+Y=前、+Z=上。
`visionpro_sdk_zup` 到机器人基座的轴向约定为：

```text
X_base = Y_sdk
Y_base = -X_sdk
Z_base = Z_sdk
```

VR 要求 SDK `origin='avp'`。输入接受 SDK `(1,4,4)` 和录制 `(4,4)` 腕矩阵，
手指缺失时仍可输出有效腕观测，不从手指局部矩阵恢复腕全局位姿。
手指镜像与重映射不进入腕旋转，不将反射矩阵转换为四元数。

对齐记录操作者初始朝向 R0 和每台机器人的实际初始末端朝向 Q0：

```text
D_base = C R(t) R0^T C^T
Q_target_world = B D_base B^T Q0_world
```

C 为输入到基座的轴向映射，B 为基座到仿真世界的旋转。左右侧分别建立 Q0。
朝向采用 SO(3) 平滑及目标角速度限制。VR 缓存包不重复计数或刷新采样时间；
过期和冻结检查用于保守保持，接收时间不代表头显硬件曝光时间。
VR 腕平移相对初始位置映射，限制在半径 20 cm 内并以最多 0.3 m/s 接近目标。
没有肩肘观测时使用机器人姿态参考。

显式腕朝向参与位姿求解。位置冲突、非有限或越限解被拒绝并记录原因；
较大关节步长按整体比例限制到每次最多 0.35 rad。
报告 `wrists` 保存对齐、目标和实际四元数、误差、拒绝原因及最近 1800 帧历史。
窗口短 RGB 轴表示实际末端，长 RGB 轴表示目标朝向。

## Vision Pro 运行

`python -m teleoperation hand vr` 在 `--scene robots` 中驱动机械臂；独立 `--scene hands` 没有机械臂。
TransHandR 使用匹配的上游 checkpoint。VR dex 使用已有的 VR 标定文件：

```powershell
. ./devtools/dex_native_env.ps1
$vrCalibration = '替换为已有VR dex标定JSON路径'
$vrIp = '替换为Vision Pro地址或房间号'
$session = Get-Date -Format 'yyyyMMdd_HHmmss'
& $dexPython -m teleoperation hand vr --backend dex --ip $vrIp --dex-calibration $vrCalibration --hand both --hand-mapping same --scene robots --report "outputs/tmp_realtime/vr_wrist_$session.json"
```

VR 回放用 `--input-h5 <原始录制路径>` 替换 `--ip`。源手到目标手的路由同时应用于手指与腕。
没有独立腕矩阵的旧 VR 文件不伪造腕姿态。

## 验收

检查左右侧、稳定标定、前臂旋转、屈伸、丢失保持与恢复，以及每台机器人的独立对齐。
同时观察位置残差、姿态误差、关节变化及腕状态，不能只用正常退出判断跟随成功。
相机轴向映射是配置值，真实空间和体感延迟需要现场确认。

## 上游来源与边界适配

| 来源 | SHA256 |
|---|---|
| `D:/2026/code/TransHandR/TransHandR/main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `.venv/dex-native/Lib/site-packages/avp_stream/streamer.py` | `a8950f11f51d3d906dddbcb42c7b91862e6161c1037415be70e6fb2ae62fbf52` |
| 既有 `src/teleoperation/applications/visionpro_processing.py` | `a90c1a57e4375e8ac22eebb134f263c83267097d063c0f60b90b5c9270744e84` |

核对 SDK `streamer.py` 的 YUP2ZUP、腕矩阵变换、get_latest 形状及缓存实现。TransHandR 三帧堆叠、左右点重映射、减腕原点与模型输入处理文件保持不变；本次没有重跑数值等价实验，也没有修改 checkpoint 元数据。
新增的腕观测、SDK 形状/缓存适配、腕初始对齐、机器人执行与报告均属于独立边界与控制适配，不宣称是上游手指算法的组成部分。
