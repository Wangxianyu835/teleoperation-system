# 正式 Python 环境与 ENV-1 验证

项目正式开发环境为 **Conda `teleoperation` / Python 3.10.20**。
本机解释器是 `D:\Anaconda\envs\teleoperation\python.exe`；PyCharm 的 Project Interpreter
请选择该文件，Working directory 选择项目根。

旧 `TransHandR` 是只读 reference baseline，保留供比较；仓库 `.venv` 不是正式验证基线。
本轮没有修改 hand/arm 算法、RobotCommand、坐标或 checkpoint 协议。

## 使用与重建

本机已建好环境，打开 Anaconda PowerShell Prompt：

```powershell
Set-Location D:\2026\code\teleoperation-system-retargeting
conda activate teleoperation
python --version
where.exe python
python -c "import sys; print(sys.executable)"
python -m pip check
python -m retargeting --help
```

`where.exe python` 的第一项和 `sys.executable` 应指向 `teleoperation`，不能是 `.venv`
或 Anaconda base。激活只对当前终端有效；新终端须重新激活。已有 `.venv` 激活时先
`deactivate`。应用、训练、导出、回放都使用同一个解释器。

不依赖终端激活的本机调用：

```powershell
$projectPython='D:\Anaconda\envs\teleoperation\python.exe'
& $projectPython -m retargeting --help
& $projectPython scripts/replay_hand_native.py --help
```

其他机器首次建立环境，从仓库根运行：

```powershell
conda env create -f environment.yml
conda activate teleoperation
python -m pip check
```

验证声明能独立重建时，创建另一个名称，保留正式环境：

```powershell
conda env create -n teleoperation-repro -f environment.yml
conda activate teleoperation-repro
python --version
python -m pip check
python -m retargeting --help
```

`environment.yml` 固定 Python 3.10.20 和已验证的 15 个直接 pip 依赖，并引用仓库的
development/camera requirements。没有本机路径、mytrans 路径、缓存或环境导出。
间接依赖由 pip 解析，这不是完整 transitive lock；当前验收平台是 Windows x86_64。
未经验证的平台、CUDA wheel 和后续依赖组合不能自动视为已验收。

本机 Conda 24.11.3 的 `conda.exe` 在受限执行器中启动失败，使用
`D:\Anaconda\python.exe -m conda` 成功完成相同操作；这不是项目 Python 依赖失败。

## Requirements 的职责

| 文件 | 直接职责 / include |
| --- | --- |
| `requirements.txt` | application：PyBullet、NumPy、SciPy、h5py |
| `requirements-retargeting.txt` | hand runtime：NumPy `<2`、SciPy `<1.14`、h5py、Torch、einops、timm、urchin、trimesh |
| `requirements-retargeting-training.txt` | include hand runtime，新增 TensorBoard |
| `requirements-retargeting-camera.txt` | include hand runtime，新增 MediaPipe 与 contrib OpenCV |
| `requirements-retargeting-tools.txt` | include hand runtime，新增 Matplotlib 与 contrib OpenCV |
| `requirements-dev.txt` | include application/training/tools，新增 Robotics Toolbox 与 SpatialMath；不包含 camera |

完整开发环境必须包含 dev **和** camera。ENV-1 先保持 requirements 原样，分阶段
安装 application → hand runtime → development → camera；各阶段成功。
未出现需要修改约束的安装、import、binary ABI 或数值失败，所以六个 requirements
文件均未修改。application 单独安装时解析到 NumPy 2.2.6 / SciPy 1.15.3；随后 hand
声明收敛到 NumPy 1.26.4 / SciPy 1.13.1，最终 `pip check` 正常。

```powershell
# 本轮首次创建与安装的等价命令；通常直接使用 environment.yml 即可。
conda create -n teleoperation python=3.10.20 pip -y
conda activate teleoperation
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-retargeting.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-retargeting-camera.txt
```

新环境版本由原 requirements 解析；没有升级或卸载 TransHandR 中的任何包。
正式 yml 记录新环境验收过的直接依赖版本，不是复制旧环境的所有包。
PyBullet 使用 pip 的缓存 wheel 安装；新环境没有 clone 或复制旧环境目录。
其他 Windows 机器若没有匹配 wheel 且 pip 转为源码构建，需要可用的 C++ 编译工具链。

## 验证范围与结果

ENV-1 实测日期：2026-10-06。完整日志、命令、exit code 和数值结果位于本机
`outputs/env_checks/`，该目录被 Git 忽略。

| 验证 | TransHandR reference | 新 teleoperation |
| --- | --- | --- |
| Python | 3.10.20 | 3.10.20 |
| pip | 26.1.1 | 26.2.1 |
| `pip check` | PASS | PASS |
| full unittest total | 166 | 166 |
| passed / skipped / failed / errors | 165 / 1 / 0 / 0 | 165 / 1 / 0 / 0 |
| package CLI help | PASS | PASS |
| `test_import.py` | PASS，无 FAIL / traceback | PASS，无 FAIL / traceback；包含 300 步应用 smoke |
| CPU Torch tensor | PASS | PASS |
| CUDA basic | PASS，RTX 4050，CUDA 13.0 | unavailable，当前为 CPU wheel |
| full CUDA training | NOT VALIDATED | NOT VALIDATED |

唯一 skip 是测试默认项目路径没有历史 palm-local checkpoint。
该历史 checkpoint 在外部 mytrans 路径读取并用于真实导出，未复制入库。

新环境进一步验证：

| 链路 | 结果与边界 |
| --- | --- |
| 关键 imports | NumPy/SciPy/h5py/Torch/einops/timm/urchin/trimesh/PyBullet/Robotics Toolbox/SpatialMath/MediaPipe/cv2/Matplotlib 全部成功 |
| CPU training | 40 帧 synthetic palm-local 数据真实跑 1 epoch，有限 train/val loss，写出 best/last checkpoint；不代表真实数据训练质量 |
| real hand export | CPU，6515 帧；左右均 `(6515,18)`；alignment=`palm_local_to_l21_v1` |
| validity | 左 3566、右 3197；frame_ids/timestamps/valid masks 与旧导出一致；无效帧保持前一次输出 |
| angle inspect | 非有限值 0，越 L21 限位 0，18D 协议未变 |
| 数值比较 | 对比原 TransHandR 导出，左最大差异 `1.1324883e-6 rad`，右 `1.7881393e-6 rad` |
| native hand replay | H1 双手全量 6515 帧 DIRECT、no-repair、substeps=1，无 crash |
| mapping 边界 | 每侧 12 个手指关节，合计 24，与手臂/腿/腰/wrist 关节无交集；非手关节由原回放逻辑保持 |
| benchmark replay | H1 + pushcube，dummy 输入，无头跑 40 步；执行成功，任务 success=False，未作为任务完成验收 |
| arm | TRON2A URDF 可读；左右各 7DOF；两侧 FK 有限，当前 pose IK 成功；现有 dual-arm 测试通过 |
| camera dependency | MediaPipe/cv2 和 `retargeting.inputs.mediapipe`、`retargeting.mediapipe` import 成功；realtime help 可解析 |
| camera asset / device | 项目根没有 `hand_landmarker.task`；物理摄像头 NOT VALIDATED，单独作为 asset/device 待办 |
| GUI / tracking 质量 | 本轮采用无头验证；新环境 GUI、真实相机、复杂 arm IK、物理动作质量仍需人工验证 |

PyBullet 仍输出原 URDF 的 logo/imu link 缺失惯性警告，本轮未修改模型资源。
native 手回放 clamp 比例左 0.29%、右 1.64%；属于现有映射与限位，不是环境错误。
原 hand 预测的拇指变化小等质量问题仍需另行验证，环境验收不能替代算法验收。

## 在正式环境复跑

全部从项目根运行；TRON2A 和历史 hand artifact 是外部只读测试资源，不由 Conda 提供。

```powershell
conda activate teleoperation
$env:TRON2A_TEST_URDF='D:/2026/code/mytrans/third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf'
python -B -m unittest discover -s tests -v
python test_import.py
python -m retargeting realtime --help

$handInput='D:\2026\code\mytrans\input\palm_local_visual_hand_data_20261005_002654.h5'
$handCheckpoint='D:\2026\code\mytrans\checkpoint\models\twohand_h5\linker\palm_local_v2\model_best.pth'
$angleOutput='D:\2026\code\teleoperation-system-retargeting\outputs\env_checks\palm_angles.h5'
python -m retargeting export --input $handInput --checkpoint $handCheckpoint --output $angleOutput --device cpu
python -m retargeting inspect --angle-h5 $angleOutput
python scripts/replay_hand_native.py --file $angleOutput --robot h1_2 --hand both --loop 1 --no-repair --substeps 1
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render --steps 40
```

Windows FK 测试会创建临时 hardlink；受限工具阻止该文件操作时，需要允许该测试操作，
不能修改 FK 来绕开。换机器时请替换外部资源路径，保持其 coordinate/checkpoint 声明一致。

GUI 手部观察继续用已改进的回放脚本，例如：

```powershell
python scripts/replay_hand_native.py --file $angleOutput --robot h1_2 --hand both --render --view left-hand --start-frame 2640 --end-frame 2940 --speed 0.5 --loop 0 --no-repair
```

该片段仅对应上述本机数据；其他录制重新选择帧区间。

## Git 范围与原有工作

开始时 branch：`refactor/consolidate-hand-retargeting`。
Git baseline：`328287dda7d56ffb512eb16d278f1f235f1d3b3c`。
开始时未提交文件如下，完整快照和 diff 保存于 `outputs/env_checks/before_*`：

```text
 M README.md
 M docs/HAND_RETARGETING.md
 M docs/INTERFACE_CONTRACT.md
 M docs/OFFLINE_PIPELINE.md
 M docs/PR2_FOLLOW_UP.md
 M docs/PROJECT_CONTEXT.md
 M docs/TEAM_ONBOARDING.md
 M scripts/replay_hand_native.py
?? docs/README.md
?? docs/SYSTEM_ARCHITECTURE.md
```

环境提交只包含 `environment.yml`、本文和 README 的环境章节。
README 通过仅更新环境章节的 index patch 暂存；已有架构/回放内容不会混入提交。
未跟踪的 `docs/SYSTEM_ARCHITECTURE.md` 与文档索引只更新环境入口，整体仍保留未提交。
其余既有改动、生产源码、历史输入和 checkpoint 均保留；未 reset、clean、stash、push。
最终提交 ID、diff stat、status 与 30 项交付记录见本机
`outputs/env_checks/ENV_1_REPORT.md`；环境重建结果见下方实测附录。

## 实测附录：包版本与 import 路径

下表的路径后缀来自实际 `module.__file__`。reference 前缀是
`D:\Anaconda\envs\TransHandR\lib\site-packages\`，新环境前缀是
`D:\Anaconda\envs\teleoperation\lib\site-packages\`。完整绝对路径和 pip inventory
保存在 `reference_environment.json` / `project_environment.json`。

| import | TransHandR | teleoperation | 两环境的 import 路径后缀 |
| --- | --- | --- | --- |
| `numpy` | 2.2.5 | 1.26.4 | `numpy/__init__.py` |
| `scipy` | 1.15.3 | 1.13.1 | `scipy/__init__.py` |
| `h5py` | 3.15.1 | 3.16.0 | `h5py/__init__.py` |
| `torch` | 2.10.0+cu130 | 2.14.1+cpu | `torch/__init__.py` |
| `einops` | 0.8.1 | 0.8.2 | `einops/__init__.py` |
| `timm` | 1.0.27 | 1.0.30 | `timm/__init__.py` |
| `urchin` | 0.0.30 | 0.0.30 | `urchin/__init__.py` |
| `trimesh` | 4.12.2 | 5.1.1 | `trimesh/__init__.py` |
| `pybullet` | 3.2.7 | 3.2.7 | `pybullet.cp310-win_amd64.pyd` |
| `roboticstoolbox` | 1.4.2 | 1.4.2 | `roboticstoolbox/__init__.py` |
| `spatialmath` | 1.1.17 | 1.1.18 | `spatialmath/__init__.py` |
| `mediapipe` | 0.10.33 | 0.10.35 | `mediapipe/__init__.py` |
| `cv2` | 5.0.0 | 4.11.0 | `cv2/__init__.py` |
| `matplotlib` | 3.10.8 | 3.10.9 | `matplotlib/__init__.py` |

TransHandR 的 pip metadata 同时包含普通 OpenCV 4.13.0.92 与 contrib 5.0.0.93。
对 `cv2/__init__.py` 和 `cv2/cv2.pyd` 的实际 SHA256 与 wheel RECORD 比较：只匹配
contrib 5.0.0.93，所以当前实际提供者明确是 contrib；普通 OpenCV 的声明已经重叠。
另有 NumPy、protobuf、tensorboardX 的旧/新 dist-info 并存，均在本轮之前已存在；
没有清理旧环境。执行前后使用相同 pip inventory 比较，包列表未改变。

TransHandR 能工作，是因为其实际 imports、CPU 数值路径、完整 tests 和应用 smoke
已通过，而不是因为它符合所有仓库声明。新环境遵守现有 NumPy/SciPy/OpenCV 约束，
也通过同一测试与真实 artifact 数值比较，因此没有证据需要放宽原约束。

## 实测附录：从声明重建

`teleoperation-repro` 已仅依据 `environment.yml` + 仓库 requirements 从零建立，
没有 clone TransHandR 或 teleoperation，也没有额外补装依赖。使用了正常包下载缓存，
未复制任何旧环境的 site-packages。

| 重建验收 | 结果 |
| --- | --- |
| Python / path | 3.10.20 / `D:\Anaconda\envs\teleoperation-repro\python.exe` |
| pip check | PASS |
| 14 个关键模块 imports | 全部 PASS |
| CPU Torch / OpenCV | PASS / 仅 contrib 4.11.0.86，文件与 RECORD 匹配 |
| full unittest | total 166；passed 165；skipped 1；failed 0；errors 0 |
| retargeting CLI / application self-check | PASS / PASS，无 FAIL 或 traceback |
| 直接依赖 pins 与 requirements | 15 个直接依赖均符合正式声明与原 requirements |

`teleoperation` 和 `teleoperation-repro` 均保留；正式使用前者，后者用于重建核查。
**Ready for PR1.5：YES（CPU 环境验收）**。这不代表实时整机、相机或实物效果已验收。
