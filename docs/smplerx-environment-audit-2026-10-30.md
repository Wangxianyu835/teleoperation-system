# SMPLer-X 接入前的本机环境核查

日期：2026-10-30。代码基线：`5f7fc8481f81d04f9cb03a9982ed598462714bd2`。

本文件记录只读检查结果。本次没有安装、升级或删除依赖，没有启动相机，也没有运行 SMPLer-X。

## 当前运行环境

用户指定当前工程使用 `teleoperation`。直接调用解释器核实如下。

| 项目 | 实测值 |
|---|---|
| 解释器 | `D:/Anaconda/envs/teleoperation/python.exe` |
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| SciPy | 1.13.1 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA | 13.0；`torch.cuda.is_available()` 为 True |
| MediaPipe | 0.10.35 |
| OpenCV | opencv-contrib-python 4.11.0.86 |
| PyBullet | 3.2.7 |
| Robotics Toolbox | 1.4.2 |
| SpatialMath | 1.1.18 |
| h5py | 3.16.0 |

NumPy、SciPy、cv2、MediaPipe、PyBullet、PyTorch、Robotics Toolbox、SpatialMath 的导入通过。上述模块来自 `teleoperation` 的 site-packages。`PYTHONPATH` 未设置；模块搜索路径包含本仓库的 `src`。

`python -m pip check` 返回 `No broken requirements found.`。这项检查只核对已安装分发包声明的依赖，不检验仓库需求文件、二进制计算结果或摄像头流程。

主环境未检测到 dex-retargeting、nlopt、avp-stream、smplx、mmcv、mmpose 的分发包。未运行完整回归，因此不能据此声称所有功能正常。

## dex 是否独立

本机存在可执行的 `.venv/dex-native/Scripts/python.exe`。其 `sys.prefix` 是 `.venv/dex-native`；`sys.base_prefix` 是 `D:/Anaconda/envs/teleoperation`。这表示它使用主环境的基础 Python。`pyvenv.cfg` 设置 `include-system-site-packages = false`，因此不继承主环境的第三方包。

| 项目 | 实测值或路径 |
|---|---|
| dex-retargeting | 0.5.0+winconda；dex-native |
| Pinocchio | 3.3.1；dex-pin-bundle |
| NLopt | 2.9.1；dex-native |
| avp-stream | 2.51+wincontrib；dex-native |
| NumPy | 2.2.6；dex-native |
| SciPy 实际导入 | 1.13.1；dex-native |
| SciPy 第二份分发元数据 | 1.15.2；dex-pin-bundle |

dex-retargeting、Pinocchio、NLopt、avp-stream、SciPy 的导入通过。dex 的 `pip check` 通过。这些结果不等同于 dex 求解器或 VR 端到端验收通过。

`dex_conda_runtime.pth` 显式追加 `dex-pin-bundle/Lib/site-packages`，并注册该 bundle 的 DLL 目录。`devtools/bootstrap_dex_windows.py` 负责这项适配。它是 dex 运行环境的一部分，不是从主环境意外导入 Pinocchio。

独立范围是第三方包目录，不是完全独立的 Python 发行版。迁移电脑后，基础解释器和 `.pth` 中的绝对路径仍需要验证。`.venv/` 被 Git 忽略；本机存在环境不能证明普通 clone 会携带该环境。

## 已确认的配置偏差

1. `environment.yml` 固定 NumPy 1.26.4。当前主环境为 2.2.6。
2. `requirements-retargeting.txt` 要求 NumPy `<2`。当前主环境不满足此要求。
3. dex 的两个可搜索包目录包含不同版本的 SciPy 分发元数据。当前 Python 实际导入 1.13.1。只记录单个 metadata 查询可能误报为 1.15.2。
4. dex 锁文件包含本机 wheel 的绝对 `file:///D:/...` 地址。其他电脑需要先按构建说明生成相应 wheel。
5. 仓库历史 launcher 含有 E/F 盘示例。用户未指定使用这些脚本。它们不是当前运行环境的依据。

结论：已确认的主要模块可以加载，未发现 pip 声明的断裂依赖。环境配置存在漂移和重建风险，不能称为已经统一或可直接复现。

## 接入时的处理

1. 继续使用 `teleoperation` 运行主工程。
2. 保留 dex-native 和 dex-pin-bundle 的已记录边界。
3. 先记录导入路径、包版本和代表性功能的基线结果。
4. 用基线回归确定主环境支持的 NumPy 版本，再单独统一主环境与需求文件。本计划不直接降级 NumPy，也不直接放宽 `<2`。
5. 为 SMPLer-X 建立独立推理环境。主工程显式指定其解释器或运行命令。
6. 在新环境内锁定实际通过的模型依赖。不要把旧模型依赖安装进 teleoperation 或 dex-native。

SMPLer-X 的可运行性和性能尚未验证。RTX 4050 Laptop 的 6 GB 显存是否足够，需要用真实权重和完整并行链路测量。

## 复核命令

以下命令不打开相机。

```powershell
& 'D:/Anaconda/envs/teleoperation/python.exe' -B -m pip check
& '.venv/dex-native/Scripts/python.exe' -B -m pip check
& 'D:/Anaconda/envs/teleoperation/python.exe' -B -c "import sys,numpy,scipy,torch; print(sys.executable); print(numpy.__version__,numpy.__file__); print(scipy.__version__,scipy.__file__); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
& '.venv/dex-native/Scripts/python.exe' -B -c "import sys,dex_retargeting,pinocchio,scipy; print(sys.executable); print(dex_retargeting.__file__); print(pinocchio.__version__,pinocchio.__file__); print(scipy.__version__,scipy.__file__)"
```
