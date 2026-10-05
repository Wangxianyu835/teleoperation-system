# 开发环境

当前测试套件使用的已验证环境：

```text
Python                 3.12.4
numpy                  1.26.4
scipy                  1.13.1
h5py                   3.11.0
trimesh                5.1.0
torch                  2.11.0+cu130
einops                 0.8.2
timm                   1.0.30
urchin                 0.0.30
roboticstoolbox-python 1.4.2
spatialmath-python     1.1.18
```

## NumPy 二进制兼容性

最小 ABI 修复是在 `requirements.txt` 固定：

```text
numpy==1.26.4
```

Python 3.12 下，现有 SciPy 1.13.1、h5py 3.11.0 和 trimesh 5.1.0 与 NumPy 1.26.4 组合可正常导入和运行。该环境修复无需改动模型、重定向、坐标、数据契约或测试代码。

仓库尚无完整的依赖锁定文件。建议在干净的 Python 3.12 环境安装声明依赖；运行双臂测试时还需安装遥操作依赖并按 README 获取官方 URDF 资产：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-teleop.txt
python -m unittest discover -s tests -v
```

## 可选摄像头依赖

当前机器另有 `opencv-python 4.13.0.92`，其包元数据要求 NumPy>=2。仓库未声明它，当前测试也不会导入它，因为 MediaPipe 适配器延迟导入 OpenCV。因此即使全量测试通过，`pip check` 仍会报告该可选环境冲突。支持实际摄像头/MediaPipe 运行时，应选择兼容 NumPy 1.26 的 OpenCV 版本，或使用独立环境；该问题不属于前次测试环境修复。

## 验证记录

2026-10-03，上述环境运行全量套件得到 `Ran 91 tests ... OK`。改动与代表性 checkpoint 核对见 [验证报告](P0_VERIFICATION_REPORT.md)。执行命令：

```text
python -m unittest discover -s tests -v
```

最初故障来自 NumPy 2.2.6 与按 NumPy 1.x ABI 构建的二进制扩展不兼容。仅将 NumPy 改为 1.26.4 后，ABI 错误消失，其他现有包保留上表版本。einops、timm 和 roboticstoolbox-python==1.4.2 原已声明但缺失，因此补充安装。本次中文化未重新运行测试。
