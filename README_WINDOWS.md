# Windows Allegro 实时重定向

2026-10-11 已在本机完成复现：物理摄像头 → MediaPipe → DexPilot → MeshCat Allegro右手。

在 PyCharm 中打开项目、选择解释器、执行终端命令或配置绿色 Run 按钮，见 [PyCharm 运行说明](README_PYCHARM.md)。

## 启动

在普通桌面 PowerShell 中执行：

```powershell
Set-Location 'D:\2026\code\dex_retargeting'
.\start_allegro.cmd
```

将右手放到摄像头前。脚本自动打开MeshCat浏览器页面和摄像头窗口；摄像头窗口按Q退出。模型不会控制实体机器人。

只查看模型：

```powershell
.\start_allegro.cmd --model-only
```

若摄像头编号不同或右手未识别：

```powershell
.\start_allegro.cmd --camera 1
.\start_allegro.cmd --selfie
```

`--model-only` 模式按终端提示回车结束；`--no-browser` 可只输出地址，不自动打开浏览器。

启动器调用项目环境 `.conda-env\python.exe`，使用明确的DLL路径，并把临时文件及运行缓存留在项目目录；不需要 `conda activate`，也不会向base或其他环境安装包。原脚本 `windows_allegro_meshcat.py` 保留不变。

## 已验证结果

- Python3.10.22、NumPy1.26.4、Pinocchio3.2.0、NLopt2.7.1、CPU torch2.5.1、MediaPipe0.10.21、MeshCat0.3.2实际导入通过，pip check无依赖冲突。
- 16个活动关节；运行时关节顺序已按名称核对，显示复用优化器模型。
- Vector和DexPilot各100次求解通过，平均目标向量误差约0.60mm/3.18mm。
- MeshCat加载21个视觉几何体，浏览器截图核验通过。
- 示例视频300/300帧检测及重定向通过。
- 摄像头0/DirectShow约30秒实测407/407帧检测、重定向和显示通过，平均13.54FPS。

本次实时验证时长约30秒，未验证长时间稳定性。普通Codex沙箱无法读取摄像头；本机桌面账户可读取，后续实时运行请使用普通桌面终端，无需管理员身份。

## 文件

- `outputs/allegro_video_replay.html`：示例视频重定向动画，可直接在浏览器打开，拖动视角。
- `outputs/allegro_live_verified.html`：本次摄像头验证的最终机器人姿态。
- `outputs/allegro_live.png`：最终机器人姿态截图。

实时服务在程序退出后结束；已导出的HTML可以继续独立打开。阶段验证脚本、安装日志、旧报告和下载缓存已清理；依赖锁文件保留在 `scripts` 中。

项目环境现已登记到 Conda 环境列表。在 PyCharm 添加解释器时选择“现有 → Conda”，“conda 的路径”填写 `D:\Anaconda\Scripts\conda.exe`，重新加载环境后选择 `D:\2026\code\dex_retargeting\.conda-env`。该字段不能填写环境中的 `python.exe`。完整操作步骤见 [PyCharm 运行说明](README_PYCHARM.md)。
