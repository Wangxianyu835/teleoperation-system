# 在 PyCharm 中运行 Dex Retargeting

适用项目：`D:\2026\code\dex_retargeting`。本机已安装依赖，并完成摄像头 → MediaPipe → DexPilot → MeshCat Allegro 右手的实时验证。

**最方便的启动方式：打开项目，在 PyCharm 底部 Terminal 中执行下面两行。**

```powershell
Set-Location 'D:\2026\code\dex_retargeting'
.\start_allegro.cmd
```

这会打开摄像头窗口和 MeshCat 浏览器页面。将右手放到摄像头前，摄像头窗口按 **Q** 退出。

## 1. 打开项目

在 PyCharm 中选择 **File → Open**，打开：

```text
D:\2026\code\dex_retargeting
```

这个目录同时包含启动器、项目环境和源码子目录：

```text
dex_retargeting/
├─ .conda-env/                已安装好的独立 Python 环境
├─ dex-retargeting/           v0.4.6 源码及 assets 子模块
├─ scripts/
│  └─ launch_demo.py          推荐的 Python 启动入口
├─ run_windows_allegro.py     实时重定向程序
├─ start_allegro.cmd          Windows 终端启动入口
└─ outputs/                  保留的三维动画和最终姿态截图
```

## 2. 设置项目解释器

打开 **File → Settings → Python → Interpreter**。部分版本显示为 **Project: dex_retargeting → Python Interpreter**。

选择 **Add Interpreter / Add Local Interpreter**，使用 **选择现有 / Select existing**，类型选择 **Conda**。按照下面填写：

| 字段 | 内容 |
|---|---|
| conda 的路径 | `D:\Anaconda\Scripts\conda.exe` |
| 环境 | 点击“重新加载环境”，选择 `D:\2026\code\dex_retargeting\.conda-env` |

点击确定后，该环境的 Python 解释器应为：

```text
D:\2026\code\dex_retargeting\.conda-env\python.exe
```

这里使用已经创建的环境，无需重新安装依赖。解释器应显示为 **Python 3.10**。

**“conda 的路径”填写 Conda 管理程序，不能填写环境的 `python.exe`。** 填错后 PyCharm 会把 `info` 等 Conda 参数交给 Python，出现 `can't open file ... Desktop\\info`。项目环境已经登记到当前用户的 Conda 环境列表；若下拉框暂时没有刷新，关闭此对话框后重新打开，再点击“重新加载环境”。解释器配置入口参考 [JetBrains 官方说明](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)。

项目的运行依赖已经安装完成。项目解释器用于代码分析及 Run 配置；下面的终端命令通过启动器显式选择同一环境，不依赖 Terminal 自动激活 Conda。

### 为什么 Terminal 仍显示 `(base)`？

`(base)` 表示当前终端会话激活了 Anaconda base。项目已有 Conda 环境，位置为 `D:\2026\code\dex_retargeting\.conda-env`，现在已登记到当前用户的环境列表。它以完整路径显示，激活时也使用完整路径：

PyCharm 的项目解释器与已经打开的 Terminal 会话需要分别确认。若右下角显示 `D:\Anaconda`，先按上面的步骤把项目解释器切换到 `.conda-env\python.exe`。若想让当前 Terminal 也激活项目环境，执行：

```powershell
conda activate 'D:\2026\code\dex_retargeting\.conda-env'
python -c "import sys; print(sys.executable)"
```

第二条命令应输出 `D:\2026\code\dex_retargeting\.conda-env\python.exe`。Conda 支持使用完整路径激活项目环境，见 [官方环境管理说明](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。

激活后可运行 `python -I -B .\scripts\launch_demo.py`。直接运行 `.\start_allegro.cmd` 则无需先激活；启动器会选择项目 Python，即使终端提示符仍显示 `(base)`，实际重定向程序也使用项目环境。

## 3. 在 PyCharm Terminal 中执行命令

以下命令按 **PowerShell** 编写。先在底部 **Terminal** 中执行：

```powershell
Set-Location 'D:\2026\code\dex_retargeting'
```

### 先检查三维模型

```powershell
.\start_allegro.cmd --model-only
```

浏览器应显示 Allegro 右手模型，此模式不打开摄像头。查看后，在终端按 **Enter** 结束。

### 运行实时手势重定向

```powershell
.\start_allegro.cmd
```

把右手放到摄像头前，尝试张开、握拳和拇指捏合。程序显示摄像头画面，并更新浏览器中的机器人手。摄像头窗口按 **Q** 结束。

### 选择其他摄像头

```powershell
.\start_allegro.cmd --camera 1
```

本机验证通过的是摄像头 **0**，默认无需添加 `--camera`。

### 调整左右手识别方式

```powershell
.\start_allegro.cmd --selfie
```

如果右手进入画面后持续显示 `No right hand`，可退出后尝试此参数。

### 不自动打开浏览器

```powershell
.\start_allegro.cmd --model-only --no-browser
```

手动打开终端打印的 `MeshCat URL`。地址通常为 `http://127.0.0.1:7000/static/`，以本次实际输出为准。

### 查看全部参数

```powershell
.\start_allegro.cmd --help
```

### 直接用项目 Python 启动

下面的命令与启动器方式使用同一项目环境，适合想明确指定 Python 的情况：

```powershell
.\.conda-env\python.exe -I -B .\scripts\launch_demo.py --model-only
```

实时模式：

```powershell
.\.conda-env\python.exe -I -B .\scripts\launch_demo.py
```

如果 Terminal 是 **cmd**，切换目录使用 `cd /d D:\2026\code\dex_retargeting`；启动命令可以写成 `start_allegro.cmd` 或 `start_allegro.cmd --model-only`。

## 4. 配置 PyCharm 的绿色 Run 按钮

打开 **Run → Edit Configurations**，点击 **+ → Python**，创建以下配置：

这些字段的含义可参照 [JetBrains Python Run 配置说明](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html)。

| 配置项 | 填写内容 |
|---|---|
| Name | `Allegro - Model Only` |
| Script path | `D:\2026\code\dex_retargeting\scripts\launch_demo.py` |
| Parameters / Script parameters | `--model-only` |
| Python interpreter | `D:\2026\code\dex_retargeting\.conda-env\python.exe` |
| Interpreter options | `-I -B` |
| Working directory | `D:\2026\code\dex_retargeting` |
| Run with Python console（如有） | 不勾选，使用普通 Run 控制台 |

如果某个字段未显示，可在配置界面的 **Modify options** 中查找。点击 **Apply → OK**，再点击绿色 **Run** 按钮。模型模式结束时，在 Run 控制台按 Enter；若控制台输入不方便，可使用 Terminal 启动。

再复制一份配置用于实时运行：

| 配置项 | 填写内容 |
|---|---|
| Name | `Allegro - Webcam` |
| Script path、解释器、工作目录 | 与上一份相同 |
| Parameters / Script parameters | 留空 |

需要其他摄像头时，参数改为 `--camera 1`；需要调整手别判断时，参数改为 `--selfie`。两项可以组合为 `--camera 1 --selfie`。

**Run 配置的脚本入口使用 `scripts\launch_demo.py`。** 它会为实际程序设置项目 DLL 路径、临时目录和缓存目录，再使用 `.conda-env\python.exe` 启动 `run_windows_allegro.py`，不需要手动设置一长串环境变量。

## 5. 常见问题

| 现象 | 处理方式 |
|---|---|
| 找不到 `start_allegro.cmd` | 先切换到 `D:\2026\code\dex_retargeting`，PowerShell 中使用 `.\start_allegro.cmd` |
| 找不到依赖，或提示 DLL load failed | 核对解释器路径，并使用 `scripts\launch_demo.py` 入口或 `.\start_allegro.cmd` |
| 直接运行原 `windows_allegro_meshcat.py` 提示缺少 example 目录 | 使用本 README 中的启动入口，它适配当前嵌套的源码目录结构 |
| 摄像头打不开 | 关闭占用摄像头的程序，再尝试 `--camera 1`；使用本机桌面 PyCharm/终端运行 |
| 摄像头有画面，但没有识别右手 | 将整只右手放入画面，改善光照，必要时尝试 `--selfie` |
| 浏览器没有自动打开 | 复制终端打印的 `MeshCat URL` 到浏览器 |
| 退出后原 MeshCat 地址打不开 | 服务随程序退出而结束；重新启动程序，或打开下面的独立 HTML 回放 |

## 6. 查看现有结果

这些文件可以从 PyCharm 项目树找到：

- [示例视频重定向动画](outputs/allegro_video_replay.html)：在浏览器中打开，可拖动三维视角。
- [实时验证最终姿态](outputs/allegro_live_verified.html)：本次摄像头测试结束时的机器人姿态。
- [实时验证截图](outputs/allegro_live.png)。

本机已验证：示例视频300帧、实时摄像头约30秒/407帧，实时平均约13.5FPS。长时间运行稳定性尚未验证。

## 7. 清理后的文件用途

运行入口是 `start_allegro.cmd` 或 `scripts/launch_demo.py`，实际程序为 `run_windows_allegro.py`。`dex-retargeting` 包含上游源码和机器人模型资源，`.conda-env` 包含已安装依赖。原始 `windows_allegro_meshcat.py` 留作参考。

复现时生成的审计脚本、安装日志、阶段验证 JSON、旧进度报告和下载缓存已清理。`scripts/q2-conda-explicit.txt` 和 `scripts/q2-pip-lock.txt` 保留依赖版本与下载哈希，便于以后重建环境；日常运行无需执行它们。`.tmp` 和 `.cache` 中的运行目录会在启动时自动生成。

## 配置参考

PyCharm 的解释器和 Run 配置字段参考 JetBrains 官方说明；界面名称可能随版本和语言不同而变化：

- [Configure a Python interpreter](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)
- [Run/Debug Configuration: Python](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html)
