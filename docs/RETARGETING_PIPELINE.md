# 手部重定向链路怎么跑、怎么复核（current-state 布局）

> **适用版本**：`origin/current-state`（包 `src/teleoperation`，CLI `python -m teleoperation`）
> **最后更新**：2026-10-10（基线 `10ebd9f`：上游 `apps/` -> `applications/` 改名 + 删除 `tests/`，本文路径与数字已同步）· **相关**：[`CONVENTIONS.md`](CONVENTIONS.md)（规则与解释器）·
> [`contracts.md`](contracts.md)（H5 / 命令契约，字段以它为准）· [`architecture.md`](architecture.md)（模块层次）·
> 注意：`docs/archive/`（旧布局手册，只读参考）已随上游 `10ebd9f` 被整体删除，
> 旧布局的对照表保留在 §5（本文档内，自足）。
>
> 本文只讲**怎么用现有代码跑通并复核**；算法/契约本身由队友维护，**不要为跑通而改 `src/teleoperation`**（CONVENTIONS 约定 3）。

---

## 0. 一次性准备

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"                                  # 本机没有 conda 时必须
E:\python3.11.7\python.exe -m teleoperation --help             # 期望退出码 0
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1  # 环境自检，期望 RESULT [OK]
```

顶层子命令（`--help` 实测 2026-10-10）：

| 子命令 | 管什么 |
|---|---|
| `hand {align,train,export,inspect,realtime}` | 手部角度对齐 / 训练 / 导出 / 检查 / 实时（另有一个**不进统一 CLI** 的实时可视化入口，见 §4.2）|
| `replay {actions,native-hand,l21-hand,mounted-hand}` | 回放：动作序列 / 原生手 / l21 手 / 装到机器人手上 |
| `sim {run,demo-joints}` | 任务仿真（`pushcube`/`pickcube` 等）与关节演示 |
| `dual {export,realtime,replay}` | 双臂（离线导出 / 实时 / 回放）|
| `tools {...}` | 一堆自检/可视化工具（见 §4）|

---

## 1. 离线链路：原始关键点 -> 手部角度 -> 关节命令

```powershell
$py='E:\python3.11.7\python.exe'

# (1) 原始关键点 H5 -> 归一化手部角度 H5（对齐左/右，每侧 18 维）
& $py -m teleoperation hand align --input  datasets\raw\retarget_twohand_153542.h5 `
                                   --output outputs\tmp_hand\angles.h5

# (2) 复核角度文件（帧数 / 维度 / 有效帧 / 越限）
& $py -m teleoperation hand inspect --angle-h5 outputs\tmp_hand\angles.h5

# (3) 训练（可选；不训练就用已有 checkpoint 或 hand align 的直出结果）
& $py -m teleoperation hand train --input outputs\tmp_hand\angles.h5 --run-name my_run --device cpu

# (4) 用 checkpoint 导出角度（把模型输出写回 H5）
& $py -m teleoperation hand export --input datasets\raw\retarget_twohand_153542.h5 `
                                   --checkpoint <checkpoint 路径>.pt `
                                   --output outputs\tmp_hand\angles_from_ckpt.h5 --device cpu
```

- `hand align` 只有两个参数：`--input` / `--output`（实测）。
- `hand export` 还有 `--batch-size` / `--scale-factor` / `--disable-identity-tracking`
  / `--max-center-displacement` / `--max-shape-rmse` / `--device {auto,cpu,cuda}`。
- 产出文件一律写 `outputs\...`（`.gitignore` 已忽略；CONVENTIONS 约定 7）。
- **H5 字段与语义以 [`contracts.md`](contracts.md) 为准**，本文不重复定义。

## 2. 回放与可视化（每步都能单独验）

```powershell
$py='E:\python3.11.7\python.exe'

# l21 手：角度 -> l21 关节（可选 --check 做检查、--render 开 GUI、--analyze 出统计）
& $py -m teleoperation replay l21-hand --file outputs\tmp_hand\angles.h5 --check

# 原生手：把角度直接喂给机器人自带手（--robot h1_2/gr1_t2/g1，可 --render --view left-hand）
& $py -m teleoperation replay native-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 把手装到机器人手腕上（检查安装位姿/抖动）
& $py -m teleoperation replay mounted-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 动作序列（契约 H 的 .npz/.h5）或假数据，无头跑，最省事的冒烟
& $py -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render
& $py -m teleoperation replay actions --file datasets\...\actions.npz --robot h1_2 --no-render
```

## 3. 任务与整机

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation sim run --task pushcube --robot h1_2 --no-render --demo    # 有头去掉 --no-render
& $py -m teleoperation sim demo-joints                                            # GUI 里看关节
& $py -m teleoperation dual export --observations <obs> --angle-h5 <angles.h5> --calibration <file> --output outputs\tmp_dual\cmd.h5
& $py -m teleoperation dual replay --command-h5 outputs\tmp_dual\cmd.h5 --calibration <file>
```

`sim run` 的动作空间实测为 **38 维** = 左臂 7 + 右臂 7 + 左手 12 + 右手 12（`tools check-environment` 会打印）。

---

## 4. 一键复核（改动前后都要跑同一条）

| 命令 | 判据 | 说明 |
|---|---|---|
| `python -m teleoperation tools verify-hand-pipeline` | 退出码 `0`，末行 `结论：... [PASS]` | **整条离线链路的验收器**：读真实 H5（557 帧 × 18 维）→ 逐项检查 → 仿真回放。旧布局的 `scripts/run_retargeting_pipeline.py` 对应物 |
| `python -m teleoperation tools check-environment` | 退出码 `0` | 项目自带环境自检：6 项，含 300 步无头端到端（`[OK] 端到端跑通（robot_id=1, action_dim=38）`）|
| `python -m unittest discover -s tests -v` | 见 [`CONVENTIONS.md`](CONVENTIONS.md) 约定 5 / §3.2 | 本机回落下实测 `Ran 19 tests / failures=0 / errors=0 / skipped=1`（2026-10-10 上游 `10ebd9f` **删除了整个 `tests/`（32 文件）**，所以现在只有本支的 2 个文件、19 个用例；`PYTHONUTF8=1` 仍建议带上）；判据是**不劣化** |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full` | 退出码 `0` | 上面三条 + 4 条冒烟，共 8 项一次跑完（含 GBK 两条、架构边界、C 盘零写）|

`tools` 里还有一批单项工具（`--help` 实测）：`validate-retarget-input`、`check-gbk-safe`、
`check-camera`、`show-robots`、`show-hand`、`show-all-hands`、`visualize-hand-keypoints`、
`visualize-l21-fk`、`compare-training-hand-pose`、`diagnose-hand-coordinates`、`make-sample-data`、`verify-p0`。
排查问题时优先用它们，**不要**先改代码。
### 4.1 几何重定向后端（不依赖 checkpoint 的备用路线）

`src/teleoperation/retargeting/hand/geometric.py` 的 `GeometricHandRetargeter` 是实时循环在
不加载 checkpoint 时用的后端（MediaPipe 21 点 -> 掌面局部对齐 -> 按**关节自身轴**测角）。
它的正确性由这条**往返实测**把关 —— 与上面的表格一起构成"改前基线 / 改后复验"：

```powershell
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe devtools\verify_geometric_hand.py   # 期望退出码 0，末行 [OK] 全部判据通过
```

- 做法：用仓库自带 L21 URDF + FK 造出**已知角度**的手 -> 按采集格式反推成 21 点输入 ->
  走 `21 点 -> ensure_hand25 -> 掌面局部对齐 -> 三帧窗口 -> 几何重定向` -> 与真值逐维比对。
- 判据（左右手各 13 组姿态）：真值为 0 的维 `|err| <= 0.05`；真值非 0 的维 `value >= 0.10` 且 `|err| <= 0.25`。
- 实测 2026-10-10：四指全部 `max_err <= 0.001`；拇指对掌姿态曾 **4 条判据不过**
  （左手 `dim14 0.700 -> +0.354`、`dim15 0.400 -> +0.145`）。改成**拇指串链反解**
  （对 cmc_yaw/cmc_pitch 做 FK 反解，mcp/ip 再用父链旋转后的轴测角）后全部通过：
  左手 `0.693 / 0.397 / 0.803 / 0.900`，右手 `0.709 / 0.374 / 0.820 / 0.900`，
  各维最大误差 `d14 0.009 / d15 0.026 / d16 0.020 / d17 0.000`。
- 成本实测（`E:\python3.11.7`，单线程）：单次链式求解 `0.92 ms`，两只手一帧 `1.83 ms`，
  整条 `retarget` `3.13 ms/帧`（30 fps 的预算是 33 ms/帧）。
- 改这个后端后，除了跑本命令，还要跑 `tools verify-hand-pipeline`（真实 H5 的端到端验收）。



### 4.2 实时闭环：摄像头 -> 重定向 -> PyBullet 里的三台机器人（默认）

`src/teleoperation/applications/realtime_hand_sim.py`（新增文件，队友代码零改动）把三段接起来：
`MediaPipeCameraInput` -> `MediaPipeCameraWorkflow`（掌面局部对齐 + 三帧窗口）->
`GeometricHandRetargeter` 或检查点后端 -> `assets/robots/l21/**` 的模型，用 PyBullet GUI 实时显示。

一次性准备：MediaPipe 模型文件（约 7.5 MB，**不入库**；`outputs/` 已被 `.gitignore` 覆盖）：

```powershell
curl.exe -sS -L -o outputs\hand_landmarker.task `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# 期望 7819105 字节，MD5 15318430EA3851670FE9914116A9CFAD
```

跑（默认开窗口、双手；`--headless` 是无窗口自检）。它**不走统一 CLI**：
`src/teleoperation/cli/**` 属队友文件（preflight 规则 3 会拦），所以按模块路径直接调用：

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation.applications.realtime_hand_sim --backend auto        # GUI，Ctrl+C 或关窗口退出
& $py -m teleoperation.applications.realtime_hand_sim --backend geometric --headless --frames 150 `
      --report outputs\tmp_probe\realtime_report.json
```

**两个窗口（这是本入口的默认形态）**：PyBullet GUI 显示三台论文机器人并排（`--scene robots`，
默认；见 4.2.1）/ 或一只 L21 手（`--scene hands`，旧行为），OpenCV 窗口显示摄像头
原图 + MediaPipe 21 点骨架。摄像头窗口是"镜头里到底有没有手"最直接的证据：

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation.applications.realtime_hand_sim --backend auto        # 默认：两个窗口都开
& $py -m teleoperation.applications.realtime_hand_sim --no-preview          # 只要 PyBullet 窗口
```

**在 PyCharm 里按运行键（`devtools/run_realtime_windows.py`，2026-10-10 新增）**：
上面的命令要求"解释器 = `E:\python3.11.7\python.exe` + `PYTHONPATH` 带 `src` + 工作目录 =
仓库根 + 缓存不在 C 盘"四条同时成立，而 PyCharm 点 Run 时通常一条都不满足（仓库里的 `.venv`
是**空壳**，连 `python.exe` 都没有，PyCharm 却常把它当默认 SDK）。这个入口把这四条补齐：

```powershell
$py='E:\python3.11.7\python.exe'
& $py devtools\run_realtime_windows.py                    # 等价于上面的默认两窗口
& $py devtools\run_realtime_windows.py --scene hands      # 换成 L21 手场景
& $py devtools\run_realtime_windows.py --check            # 只体检（解释器/依赖/模型/相机），不开窗口
```

- PyCharm 配置：Script path 指向该文件，Working directory = `F:\simulation_platform_cs`，
  Parameters 留空即可；**解释器选错也能起来** —— 它发现当前解释器缺
  `cv2`/`mediapipe`/`pybullet` 时会打印 `[WARN] current interpreter ... lacks: ...`，然后用
  `E:\python3.11.7\python.exe` 把自己重跑一遍（`TP_REALTIME_LAUNCHER_REEXEC=1` 防递归）。
  缺依赖之外还会查 **cv2 有没有 GUI**：headless 版 wheel（`GUI: NONE`）能读摄像头却开不了
  窗口，只跑仿真时看起来"正常"，正是"只看得到机器人窗口"的原因之一；这种情况同样重跑到 E:
  （`--check` 里对应 `[OK] cv2 window support: WIN32UI` / `[FAIL] cv2 cannot open a window`）。
- 它还会：把 `sys.path` 加上 `src`、把工作目录切回仓库根（PyCharm 默认是脚本所在目录）、
  把「没设或指向 C:」的缓存/临时变量指到 `E:\cache`（已有的好值只打印 `[KEEP]` 不动）、
  默认写一份报告到 `outputs\tmp_realtime\run_<时间戳>.json`（退出码与真实入口一致：0 = 有手
  输出过角度；1 = 全程没手；3 = 环境不对）。
- 摄像头窗口"看不见"怎么修（2026-10-10）：OpenCV 窗口原先按系统默认位置出现，而 PyBullet 窗口
  先建、更大，在 IDE 里跑很容易被整个盖住（症状 = 只有机器人窗口在跑）。现在第一帧画完就把窗口
  挪到 `--preview-window-pos`（默认 `20,60`；`none` = 保留系统默认）并 `WND_PROP_TOPMOST` 置顶，
  同时打印它的**实际屏幕矩形**：`camera window placed at 28,91 (640x480) [always on top],
  screen ... title='camera: MediaPipe 21-landmark input (q/Esc quits)'`；矩形落在可见桌面之外会再打
  一行 `WARNING: the camera window is outside the visible desktop`。`--preview-no-topmost` 关掉置顶。
  跑完启动器还会读报告回一句 `[OK] camera window stayed open at [x, y, w, h]` 或
  `[WARN] the camera window did NOT stay open: <原因>` —— 下次"只看见机器人窗口"，先看这两行。
- 实测 2026-10-10（`E:\python3.11.7\python.exe`）：`--check` 退出码 `0`（`modules present`、
  `model asset .../outputs/hand_landmarker.task (7819105 bytes)`、`cameras readable: [0]`）；
  工作目录被故意设成 `devtools` 时打印 `working directory ... -> F:\simulation_platform_cs`
  且退出码仍为 `0`；用缺依赖的 `Python312\python.exe` 启动时先 `[WARN] ... lacks: cv2,
  mediapipe, pybullet` 再 `[OK] re-running with E:\python3.11.7\python.exe`，退出码 `0`；
  `--headless --frames 3` 退出码 `1`（画面里没手，设计如此）并写出 5983 字节报告
  （`backend=geometric`、`scene=three_robots`、`physics_dt=0.0041667`）；同日新增的窗口检查会多
  打印一行 `[OK] cv2 window support: WIN32UI`（退出码仍 `0`）。

- 只看见 PyBullet 窗口时的三步自查：① 日志有没有 `camera window: on (mirrored) ... title=...`
  （有 = 窗口确实开了，去任务栏找那个标题，或看 `camera window placed at` 给的坐标）；
  ② 有没有 `cv2 window support` 那行（`GUI: NONE` = headless 版 OpenCV，换个解释器即可）；
  ③ 结尾 `camera preview: window=on/off ... note=` —— `note` 非空就是 `imshow` 在那个解释器里
  真的失败了，`note` 为空而窗口又看不到，就去 `--preview-window-pos` 指定的坐标找。

- 窗口内容：骨架用 `tools/plotting.py` 的 `COLORS` / `SOURCE_EDGES`（指尖橙色），左上角一行
  `frame= / fps= / raw= / valid= / invalid= / calibration=`，没有手时左下角红字
  `no hand detected`。默认左右镜像（像照镜子），`--preview-no-mirror` 关掉；
  在窗口里按 `q` / `Esc` 或直接关掉它，等价于关 PyBullet 窗口（都是正常退出）。
- 读那两个数字就能定位问题：`raw=left=0/30` 是"MediaPipe 压根没检出这只手"（取景/光照问题）；
  `raw` 有数但 `valid=0` 是"检出了，但三帧窗口或开手标定还没就绪"（多保持一会儿张开手即可）。
- 输入层（`inputs/mediapipe.py`）只返回关键点、不暴露 BGR 图像，而它属队友文件（约定 3 不能改）：
  本入口给它的 `cv2.VideoCapture` 套了一层**只读代理**（`_PreviewCapture`）取图，`read()`
  原样透传。代理的等价性由 `outputs\tmp_probe\preview_selfcheck.py` 的 A 段核对（装/不装代理，
  逐字段比较输入层输出）。
- `--preview-dump-dir DIR --preview-dump-frames N` 把前 N 帧带骨架的画面写成 PNG；
  在 `--headless` 下也能用，所以没有显示器时同样能留证据。

- 后端：`auto`（默认；有检查点就用它，缺了自动回落几何后端并打印原因）、`geometric`、`checkpoint`。
  几何后端不需要任何 `.pth`，但**每只手要先在镜头前张开约 15 帧**完成开手标定，之后才输出角度。
- 实测 2026-10-10（`--backend geometric --headless --frames 150`）：`fps=29.6`、帧间隔中位数
  `30.7 ms`；画面里的一只手 33/47 帧输出有效角度，另一只手不在画面内时全部 invalid 且链路不中断。
- 显示约定：本入口用 `resetJointState` 写关节、并**关掉重力**（纯可视化），画面跟手不滞后；
  它不代表接触/力矩效果 —— 要物理效果请用 `replay mounted-hand` / `sim run`。
- 四个已经踩过的坑（改这段代码时别重复）：
  1. `L21HandAdapter` 的 `joints_order` 必须是 **18 项、第 0 项为占位**的 `MAPPING_18`；
     传 17 项的 `L21_JOINT_ORDER` 会让所有维度错位一格（由 `outputs/tmp_probe/scene_probe.py` 抓到）。
  2. 不关重力时，`resetJointState` 写入的拇指自由关节会在 `stepSimulation` 后被拖走约 `1e-3 rad`。
  3. 退出别走 `with MediaPipeCameraWorkflow(...)`：它的 `__exit__` 会卡在
     `MediaPipeCameraInput.release()` 的 `landmarker.close()`（TFLite/XNNPACK 释放）上，
     本机实测 **42.2 s**（同一探针里 `cv2` 只有 `0.303 s`，见 `outputs/tmp_probe/teardown_split.log`）。
     本入口因此不用 `with`：`finally` 里把 `release()` 丢到守护线程、最多等
     `RELEASE_TIMEOUT_SECONDS = 2.0` s，打印统计并写完 `--report` 后由 `__main__`
     直接 `os._exit(退出码)`，所以 Ctrl+C / 关窗口后**立刻**回到命令行（`main()` 仍返回
     退出码，import 调用不受影响）。
- 退出码：有手输出过角度 -> `0`；全程没手/没角度 -> `1`，并打印 `[FAIL] no valid hand angles ...`。
- PyBullet GUI 的第二个坑：`p.connect(p.GUI)` 是异步返回的，本机实测紧接着调 `setGravity` 会报
  `Not connected to physics server`（第一次 GUI 试跑就是这样挂掉的）。入口现在会重试到服务端应答
  （最多 15 s）再建场景；运行中掉线（手动关窗口、或会话开不出 OpenGL 窗口）不再抛异常，只打印
  一次提示并照常输出统计与 `--report`；`p.disconnect()` 在 GUI 客户端已死时同样会挂住，也走
  2 s 守护线程。
- 验证记录（2026-10-10，`E:\python3.11.7\python.exe`）：
  - `--backend geometric --headless --frames 20`：退出码 `1`（画面里没有手 -> 设计如此），
    20 帧 `fps=32.2`、帧间隔中位数 `31.7 ms`，报告已写，进程无残留（墙钟 18.6 s，其中循环 1.6 s）。
  - `--backend geometric --frames 60`（GUI，非交互会话里窗口一闪即关）：退出码 `1`，
    `fps=11.3`（GUI 渲染开销），报告已写，进程无残留（墙钟 24.1 s）。修复前同样的命令**永不退出**
    （卡在 `p.disconnect()`/`landmarker.close()`，需手动 kill）。
  - 摄像头窗口（2026-10-10 新增）：`--headless --frames 30 --preview-dump-dir outputs\tmp_probe\preview_dump
    --preview-dump-frames 6`：退出码 `1`（镜头里没有手，设计如此），`fps=33.5`、帧间隔中位数
    `30.8 ms`，写出 6 张带 HUD 的 PNG；报告里 `raw_detected_frames` 左/右都是 `0/30`，与
    `[FAIL] no raw hand detection at all ...` 的提示一致。
  - 预览链路自检：`E:\python3.11.7\python.exe outputs\tmp_probe\preview_selfcheck.py`：退出码 `0`、
    `summary: ALL OK`。A 段用假采集器证明代理对输入层**透明**（除墙上时钟时间戳外逐字段一致，
    且代理存下的正是检测器读到的那一帧）；B 段用 `visual_hand_data_20260912_112108.h5` 的真实
    21 点画出 3 张 PNG（`outputs\tmp_probe\preview_selfcheck\`：第 26 帧左手张开、第 349 帧握拳、
    第 400 帧双手，左手蓝 / 右手绿 / 指尖橙）。
  - 摄像头窗口可见性（2026-10-10 修后 GUI 实跑，日志见 `outputs\tmp_preview_probe\`，探针产物不入库）：
    `devtools\run_realtime_windows.py --frames 25` -> 退出码 `1`（镜头里没手，设计如此）；日志
    `placed at 28,91 (640x480) [always on top], screen 1707x1067`、`camera preview: window=on
    mirror=on ... rect=(28, 91, 640, 480)`、启动器 `[OK] camera window stayed open at
    [28, 91, 640, 480] (mirrored=True)`。`--frames 25 --preview-window-pos 300,200
    --preview-no-topmost` -> 退出码 `1`，`placed at 308,231 (640x480)`（= 请求的 300,200 加窗口边框
    8,31）、且**没有** `[always on top]` 标记 —— 两个开关都确实生效。
  - 预览分支的无显示器验证：`--headless --frames 30 --preview-dump-dir outputs\tmp_preview_probe
    --preview-dump-frames 3`：退出码 `1`，`camera preview: window=off ... dumped_png=3`，写出 3 张
    约 280 KB 的带骨架 PNG —— 证明 `_PreviewCapture` 代理、`annotate`/HUD、dump 三者在 `--headless`
    下都正常（这条仍是不接显示器时唯一的预览证据）。
  - 取景现状（人工步骤，只能由用户完成）：预览窗口显示本机摄像头当前**对着天花板斜上方**
    （画面里是墙面、挂钟和头顶头发），所以 `raw=0`。要出效果需要调整摄像头角度或把手抬到镜头
    正前方；预览窗口会立刻显示手在不在画面里，不用再猜。
    `devtools\check_gbk_safe.py --strict`：退出码 `0`。
- 仿真侧可脱离摄像头单独验：`E:\python3.11.7\python.exe outputs\tmp_probe\scene_probe.py`（期望退出码 0）。

### 4.2.1 默认显示目标：三台论文机器人并排（原装手）

`--scene` 决定 PyBullet 里到底出现什么，**默认 `robots`**（2026-10-10 起）：

| `--scene` | 画面 | 关节驱动 |
|---|---|---|
| `robots`（默认）| H1-2 / GR1-T2 / G1 并排，各用**出厂自带**的灵巧手 | `POSITION_CONTROL`，每帧推进 `--sim-steps` 步（默认 8 步 x 1/240 s = 33 ms）|
| `hands` | 只摆 1-2 只 LinkerHand L21 手（旧行为）| `resetJointState`（纯可视化，不滞后）|

承载体是新文件 `src/teleoperation/applications/three_robot_scene.py`（队友代码零改动）；降维映射直接
复用 `robots/native_hand.py`（符号/限位自动判定），**不写第二份映射表**。

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation.applications.realtime_hand_sim --backend auto    # 三台并排（默认）
& $py -m teleoperation.applications.realtime_hand_sim --view front      # 正面
& $py -m teleoperation.applications.realtime_hand_sim --scene hands     # 回到"只有一只手"
```

范围声明（与 `tools show-all-hands` 一致）：手指由实时角度驱动；**手臂不参与**
（重定向不输出手臂关节角），锁定在中性姿态静止；其余关节（腿 / 腰 / 头）同样锁死。

四个已经踩过的坑：

1. **摆位**：三台 URDF 原点都在 (0, 0) 附近，必须沿 X 横向摆开（-2.4 / 0 / +2.4 m），否则重叠。
2. **初始穿透（本轮最难查的一个）**：按 URDF 自带 `base_z` 加载时脚底与地面有初始穿透，
   接触冲量会在 1.5 s（45 帧 x 8 步）内把 h1_2 的 `left_ankle_roll_joint` 顶掉 `0.664 rad`、
   GR1-T2 的 `right_hip_pitch_joint` `0.700 rad` —— 画面上就是"机器人自己在歪"。
   重力 `-9.81` 与 `0` 的漂移**完全一样**（`0.6639` 对 `0.6639`），所以原因是接触不是重力。
   现在建场景时按 AABB 把每台抬到"最低点离地 `ground_clearance=0.03 m`"，漂移降到
   `h1_2 0.0005 / gr1_t2 0.0317 / g1 0.0019 rad`。
3. **斜视取景**：`--view full`（yaw 135）时三台沿 X 排开，最近那台比目标中心近约 `(宽/2)*sin45`，
   只按中心距离取景会把最边上一台裁掉（实测只看得见两台）。取景距离现在按
   "深度半跨 + 侧向半跨"算（`4.09 m`，而不是 `2.99 m`）。
4. **锁姿态力矩**：映射之外的关节每帧用 `force=500` 锁在加载角度（手指 `force=200`），否则手臂下垂。

**无头验收**（本机没有交互桌面，用这条出证据；同时写 PNG 供人眼复核）：

```powershell
$py='E:\python3.11.7\python.exe'
& $py devtools\verify_three_robot_scene.py --report outputs\tmp_probe\three_robot_scene\report.json
```

实测 2026-10-10：退出码 `0`；三台全部加载（`h1_2 55 / gr1_t2 70 / g1 53` 关节，手部覆盖
`L/R 12/17(71%)`、`11/17(65%)`、`7/17(41%)`）；45 帧 x 8 步里**左右手各 30 个被映射关节全部转动**
（最大 `1.3759 / 1.4588 rad`），被锁关节漂移 `<= 0.0317 rad`（判据 `0.05`）；
离屏截图 `iso_frame044.png` / `front_frame044.png` 里三台都在画面内（已人眼复核）。

**真实摄像头复测（2026-10-10，人眼确认）**：把镜头对着手以后重跑实时入口（日志
`tmp_motion\realtime_live2.log`，报告 `outputs\tmp_probe\realtime_robots\report_live2.json`），
操作者**当场看到**摄像头预览窗口和 PyBullet 里三台机器人的手一起动，报告数字也对得上：
2059 帧 / `248.9 s`（`8.37` 帧/秒，帧间隔中位数 `113.3 ms`）；两侧开手标定都完成
（`calibration={'left': True, 'right': True}`）；原始检出 左 `194` / 右 `251` 帧，经三帧窗口后
各产出 左 `150` / 右 `198` 组角度，`index_pip` 左右都到满量程 `1.57 rad`。
这把上面那段"镜头前没有手"的结论闭合了：**代码没问题，手进入取景范围就能驱动三台机器人**。

### 4.3 没有摄像头也能出画面：文件驱动的同一套链路（`devtools/replay_capture_to_l21.py`）

实时入口的输入是摄像头，所以"镜头前有没有手"直接决定它能不能显示手。2026-10-10 在本机
（非交互会话、镜头前没有手）做过一组取证，结论是**代码没问题，是这一次输入里没有手**：

| 证据 | 怎么取的 | 结果 |
|---|---|---|
| 相机本身正常 | `outputs\tmp_probe\camera_brightness_probe.py` | index 0 可打开，20/20 帧，640x480，亮度 114~147（不是黑屏、不是被遮挡） |
| 队友自己的采集脚本 | `hand capture media.py` 原样复制成 `outputs\hand_capture_probe.py`，只加"跑 N 帧自动停 + 不开窗口"两个开关 | 120 帧全部 `有效手部帧数：0` |
| 同一脚本 2026-09-12 采的数据 | `visual_hand_data_20260912_112108.h5` / `_111415.h5` | 左手 643/701、右手 654/701 帧有 21 点（另一份 367/402、278/402） |
| 本仓库没有任何检查点 | `outputs/**` 与 `D:\2026\code\mytrans` 下 `*.pth` 数量 | 0 个 -> 只能用几何后端（零权重） |

同一份代码在"镜头里有手"时能稳定检出（9-12 的数据就是证据），本次 0 帧只是因为镜头前没有手；
实时入口打印的 `valid=left=0 right=0` 说的就是这件事。2026-10-10 加了摄像头预览窗口后再取一次证，
结论更具体：预览 PNG 里画面是**天花板斜上方**（墙面、挂钟、头顶头发），30 帧 `raw=0/30`；
12 帧那次偶然出现的骨架落在头发区域上，属于误检而不是手。也就是说根因是**取景**（镜头没对着手），
不是代码 —— 把摄像头角度调低或把手抬到镜头正前方即可，预览窗口会实时显示手在不在画面里。
为此补了一个**文件驱动**的入口，
除了输入来源不同，链路与实时入口完全一致（掌面局部对齐 -> 3 帧窗口 -> 重定向 -> L21 PyBullet），
同样不改任何队友文件：

```powershell
$py = 'E:\python3.11.7\python.exe'
# 无头渲染：每 50 帧存一组 iso/side 视图 PNG，并写 JSON 报告
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --out-dir outputs\tmp_probe\offline_l21 `
      --views iso side --render-every 50 --report outputs\tmp_probe\offline_l21\report.json
# 本地看窗口（需要交互式桌面；--hold-seconds 让最后一帧留在屏幕上）
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --gui --hold-seconds 30
# 整段视频：每帧渲染成 mp4，分辨率 = --width x --height（默认 640x480 iso 视角）
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --out-dir outputs\tmp_probe\offline_l21_video `
      --render-every 0 --video replay_l21_both.mp4
# 输入已经是 18 维角度 H5：跳过 21 点 -> 掌面局部 -> 重定向，直接驱动三台机器人
& $py devtools\replay_capture_to_l21.py --angles-h5 datasets\raw\retarget_twohand_153542.h5 `
      --out-dir outputs\tmp_probe\offline_robots_twohand --render-every 0 `
      --video replay_twohand_robots.mp4 --report outputs\tmp_probe\offline_robots_twohand\report.json
```

- 输入契约：`left_hand_keypoints` / `right_hand_keypoints`，形状 `(T,21,3)`；没检出的一侧写成全 0
  （与 `retargeting/data.py` 的说明一致），本入口把全 0 帧当作"这一帧没有这只手"。
- **角度直通**（`--angles-h5`，与 `--input-h5` 二选一；两个都给会在 `main()` 里直接报错退出）：
  输入契约是 `left_angles` / `right_angles`，形状 `(T,18)`（18 = `HAND_ANGLE_DIM`）；
  可选的 `left_valid` / `right_valid` `(T,)` 标出哪一帧有这只手（没有这两个数据集时整段视为有效），
  `timestamps` / `frame_ids` 只写进报告、不参与数值。既然每帧本身已经是重定向结果，
  `21 点 -> 掌面局部 -> 三帧窗口 -> 重定向` 这**整段跳过**，所以 `--calibration-frames` /
  `--smoothing` 对这条路径**不生效**（脚本会打印一行 note 说明），报告里
  `input_schema="angles"`、`calibration="n/a (...)"`，一眼能区分这次跑的是哪条输入。
- **视频分辨率**：`--video` 写出的 mp4 尺寸 = `--width x --height`（默认 `640x480`，4:3、`iso` 视角、
  逐帧写）。320x240 下手指只有几个像素、看不出屈曲，所以要看手就用默认 640x480；代价是渲染时间随
  像素数近似线性增长（本机 320x240 约 `0.28 s/帧`、640x480 约 `0.35 s/帧`），整段跑大分辨率时可以
  `--stride 2 --fps 15`：帧数减半、播放时长不变。
- **判"手真的在动"用关节统计，不要用画面逐帧像素差**：`ER_TINY_RENDERER` 对棋盘地面有抖动噪声，
  实测连"两边都没手、姿态完全没变"的两帧（第 0 / 20 帧）也有 `mean|diff| 2.3`、`12.8%` 像素变化，
  和真有动作的帧（`2.4` / `12.9%`）分不开。可信的判据是报告里的 `angles.max_std`
  （逐维标准差的最大值 > 0 就说明姿势在变）与 `final_joint_values`。
- 脚本自带 `src` 的 `sys.path` 引导，所以按**文件路径**直接跑即可，不需要先设 `PYTHONPATH`。
- 无显示环境也能出证据：用 PyBullet 的 CPU 光栅化（`ER_TINY_RENDERER`）出 PNG / mp4，
  所以"到底有没有把手渲染出来"可以截图核对，不必依赖 GUI。
- 验证记录（2026-10-10，`--input-h5 visual_hand_data_20260912_112108.h5`，双手，701 帧）：
  - 处理 701 帧用 `12.05 s`（`58.2` 帧/秒）；左手检出 643 帧 -> 输出 627 组角度，右手检出
    654 帧 -> 输出 638 组角度；两侧开手标定均完成（`calibration={'left': True, 'right': True}`）。
  - 角度确实随数据变化（不是一张静态图）：18 维角度的最大逐维标准差 左 `0.6801 rad` /
    右 `0.6703 rad`；指节屈曲接近满量程（`index_pip` 0 -> 1.57），拇指 yaw 随握合变化（0.324 / 0.942）。
  - 出图：`--render-every 50` 得到 28 张 PNG（`outputs\tmp_probe\offline_l21\`，另附 `sheet_all.png` 拼图）；
    全帧视频 `outputs\tmp_probe\offline_l21_video\replay_l21_both.mp4`（`1,037,756` 字节，701 帧）。
  - 目视核对（截图中可见两只 L21 手，浅色是左手、深色是右手）：第 99 帧四指伸直、第 349 帧屈曲、
    第 699 帧侧视图手指收拢。
- 退出码：任一侧产出过角度 -> `0`；一帧都没产出 -> `1`。
- `devtools\replay_capture_to_l21.py` 的 `--distance` / `--target-z` 只影响取景，不影响角度数值。
- `--scene` 同样适用（默认 `robots`：三台并排 + 原装手；`--scene hands` 回到只有 L21 手）。
  `--scene robots` 时取景与裁剪面按机器人整机给（`target=[0.13, 0, 0.86]`、`distance=4.09 m`、
  fov 60 / far 26 m），并且**默认视角自动换成 `front + iso`** —— `side`（yaw 90）是"沿着排看"，
  三台会前后遮挡（实测只剩最前面一台）。
- 验证记录（2026-10-10）：本机没有那份采集 H5，用 `datasets/samples/human_hand_demo_right.h5` 的
  `keypoints_3d (100,21,3)` 临时转成 `left/right_hand_keypoints` 结构后跑 A/B：
  `--scene hands` 退出码 `0`（右手 84 组角度、`max_std 0.2213 rad`、4 张 PNG）；
  `--scene robots` 退出码 `0`（同一份角度、4 张 PNG -> `outputs\tmp_probe\offline_robots\`），
  目视核对 `frame00049_front.png` 与 `frame00099_iso.png` 都是三台机器人。
- 验证记录（2026-10-10，`--angles-h5 datasets/raw/retarget_twohand_153542.h5`，557 帧，双手，`--scene robots`）：
  - 这份 H5 是**真采集数据**推出来的角度：属性里记着
    `input_file=.../visual_hand_data_20260912_153542.h5`、
    `checkpoint=.../twohand_h5/linker/none_warmstart/model_best.pth`、`left_coordinate_mode=none`；
    数据集为 `left_angles/right_angles (557,18)`、`left_valid/right_valid (557,)`、`timestamps/frame_ids`。
  - 无头（`--render-every 0`，不出视频）退出码 `0`：557 帧 `6.68 s`（`83.3` 帧/秒）；
    左手有效 510 帧、右手有效 512 帧。有效掩码的实际分布是 `left_valid` 在 `0..24` 与 `535..556`
    为 `False`、`right_valid` 在 `0..24` 与 `537..556` 为 `False`（这些行的 18 维角度同时是全 0），
    即**有效区间约在 25..534**。
  - 角度确实随数据变化：最大逐维标准差 左 `0.7050 rad` / 右 `0.5516 rad`；`report.json` 里
    `input_schema="angles"`、`calibration="n/a (--angles-h5: the 18D angles are already retargeted)"`。
  - 整段视频两版：
    - `outputs\tmp_probe\offline_robots_twohand\replay_twohand_robots.mp4` —— `1,034,938` 字节、
      557 帧（每帧都写）、320x240、30 fps，改分辨率之前跑的那版；取帧人眼复核
      （第 0 / 150 / 300 / 450 / 556 帧）：三台都在画面内、棋盘地面正常，第 0 帧是 `valid=False`
      的静止姿态，其余帧手部姿势各不相同。
    - `outputs\tmp_probe\offline_robots_twohand_hd\replay_twohand_robots_hd.mp4` —— `1,539,534` 字节、
      279 帧（`--stride 2`）、640x480、15 fps（播放时长与上一版同为 `18.6 s`）；退出码 `0`，
      渲染 `97.52 s`（`2.9` 帧/秒）；左手有效 255 帧 / 右手 256 帧、`max_std` 左 `0.7067` /
      右 `0.5458 rad`。
    - 放大复核（把 `GR1-T2` 的 `250,195-375,275`、`H1-2` 的 `140,160-240,230` 裁出来 x5 拼成
      逐帧对照图）：灰机器人手上**手指轮廓逐帧不同**（第 15/75 帧收拢、第 225/265 帧张开成叉状），
      黑机器人的手臂/手在第 195/225 帧相比第 15 帧明显转开 —— 与关节统计一致，手确实在动。
- **踩坑记**：这份 H5 的有效区间在 `25..534`，所以 `--max-frames 12` 这种"只看开头几帧"的跑法
  会整段落在无效区而**正常返回退出码 `1`**（不是 bug）；要抽查就配 `--stride` 让它跨进有效区，
  或干脆整段跑（557 帧无头只要 `6.7 s`）。
- 互斥与报错的两条负路径（实测 2026-10-10）：同时给 `--input-h5` 和 `--angles-h5` ->
  退出码 `2`、`error: pass exactly one of --input-h5 (raw 21 landmarks) or --angles-h5 (18D angles)`；
  把 **21 点**的 H5 塞给 `--angles-h5` -> 退出码 `1`、
  `lacks datasets ['left_angles', 'right_angles']; found ['frame_ids', 'left_hand_keypoints', 'right_hand_keypoints', 'timestamps']`。
  角度直通同样支持 `--scene hands`（只用 L21 手）：同一份数据前 120 帧退出码 `0`、
  左右各 95 帧有效（`max_std` 左 `0.7346` / 右 `0.6022 rad`）。
- 改分辨率后的老路径回归（2026-10-10，21 点输入）：`--input-h5 <21 点 H5> --width 320 --height 240
  --render-every 0 --video replay_check.mp4` 退出码 `0`，日志里
  `video: ...\replay_check.mp4 320x240 @ 30 fps`，右手 `100/100` 帧检出 -> `84` 组角度、
  `max_std 0.2213 rad`（与改动前同一份数据的数字一致），说明 `--angles-h5` 分支与新的
  视频分辨率参数都没有动到老的 21 点路径。

---

### 4.4 摄像头 -> H5：离线闭环的起点（`teleoperation.applications.camera_record`，2026-10-10 新增）

**为什么需要**：`hand align` 与 `--input-h5` 回放都要求输入是"根数据集
`left_hand_keypoints` / `right_hand_keypoints`、形状 `(T,21,3)`"的**原始抓取 H5**。
队友原来的采集脚本（产物形如 `input\visual_hand_data_20260912_112108.h5`，来自
`D:\2026\code\mytrans`）没有随仓库入库、本机也已不存在，所以"摄像头到文件"这一环
一直没有入口 —— 现在补上了。

| 新增文件 | 干什么 |
|---|---|
| `src/teleoperation/data/capture_h5.py` | 抓取 H5 的容器（增量写、形状/dtype/attrs 与队友读取器逐字段对齐）|
| `src/teleoperation/applications/camera_record.py` | 入口：摄像头 -> H5（可选再落一份 18 维角度 H5）|
| `devtools/verify_camera_record.py` | 一键验收：录制 -> 读回 -> `hand align` -> `hand inspect` ->（有手时）回放 |
| `tests/test_camera_record.py` | 10 个用例，全部不需要摄像头/不需要 MediaPipe 运行时 |

它**不进统一 CLI**（`cli/registry.py` 属队友文件，约定 3），按模块路径跑：

```powershell
$py='E:\python3.11.7\python.exe'
# 一键验收（真摄像头 10 秒；无人值守会话也能跑）
& $py devtools\verify_camera_record.py --seconds 10
# 只录制：原始点 + 18 维角度（角度来自零权重几何后端，见 §4.1）
& $py -m teleoperation.applications.camera_record --output-h5 datasets\raw\my_capture.h5 `
       --angles-h5 outputs\tmp_hand\my_capture_angles.h5 --seconds 10
```

**落盘约定**（沿用队友既有语义，不新造）：未检出的那一侧写**全零行**
（`align_recording` 会统计成 `zero_source_frames`，`devtools/replay_capture_to_l21.py`
会把全零还原成"没有手"）；另写 `left_valid` / `right_valid` 让"到底有没有检出"可直读；
`frame_ids` 取自 MediaPipe `metadata.frame_index`；`timestamps` 为**相对第一帧的秒**
（与 `datasets\raw\*.h5` 同口径），`unix_ms` 存原始 Unix 毫秒；attrs 里
`source_landmark_space=mediapipe_normalized`、`missing_hand_encoding=zeros`、
`timestamp_unit=relative_seconds`。

**实测 2026-10-10**（`E:\python3.11.7\python.exe`）：

| 命令 | 结果 |
|---|---|
| `devtools\verify_camera_record.py --seconds 10` | 退出码 `0`；`frames=288`、`duration_seconds=9.513`、`measured_fps=30.274`、`loop_fps=28.785` |
| 同上，第 2 步（读回） | `capture layout is (T,21,3)+valid+frame_ids+timestamps+unix_ms` |
| 同上，第 3 步（`hand align`） | `aligned_*.h5 (T,25,3)`，`right_zero_source_frames=288`（这一段镜头前没有手，所以全是零行）|
| 同上，第 4 步（`hand inspect`） | `angle H5 frames=288 valid left=0 right=0 backend=geometric` |
| 同上，第 5 步（回放） | `[SKIP]`（设计如此：本段没有检出手，脚本明确打印 `[WARN] no hand was in frame, so this run proves the file chain only`）|
| `python -m unittest tests.test_camera_record -v` | `Ran 10 tests ... OK`（含一个"打桩摄像头跑完整 `run()`"的回归用例，就是它抓到了 `close()` 后 summary 少字段的 bug）|

结论：**"录制 -> 读回 -> 对齐 -> 角度 -> 可复核"这条文件链路已经用真摄像头端到端验过**；
"有手时的重定向/回放"仍需镜头前真的有手（人工步骤，见 §4.2 的取景说明）。


### 4.5 手臂链路的轻量版：21 点 -> 手腕目标 -> PyBullet IK（2026-10-10 新增）

**先说清楚队友那条为什么跑不起来**：`retargeting/arm/ik.py` + `apps/dual.py` 是给
**TRON2A** 写的，三样外部资产本机都没有 —— `third_party\tron2-robot-description\...\robot.urdf`
（目录不存在）、TRON2A 标定 JSON（只有 `configs\tron2a_dach_calibration.example.json` 示例）、
训练出来的手部检查点（仓库里 `.pth` 数量为 0）。而**论文三台机器人的 URDF 是齐的**
（`assets\robots\from_teleopbench\**`），PyBullet 自带的 `calculateInverseKinematics`
就能对它们的 7 DoF 手臂求解。

| 新增文件 | 干什么 |
|---|---|
| `src/teleoperation/retargeting/arm/wrist_targets.py` | 21 点 -> 手腕目标位姿（掌面基沿用 L21 定义；图像->机器人的轴/符号是**约定**，含 `WristMapping` 全部可覆盖）|
| `devtools/verify_arm_from_capture.py` | 验收：合成扫掠或 `--input-h5`（§4.4 的录制文件）驱动 H1-2 / GR1-T2 / G1 之一，报告残差/行程/姿态误差/漂移与 PNG |
| `tests/test_wrist_targets.py` | 9 个用例：掌面基正交性、三个轴/符号、两手一致性、退化输入、四元数往返 |

```powershell
$py='E:\python3.11.7\python.exe'
& $py devtools\verify_arm_from_capture.py --robot h1_2 --side both --frames 40   # 合成扫掠
& $py devtools\verify_arm_from_capture.py --robot g1 --input-h5 datasets\raw\my_capture.h5  # 用真录制
```

判据（任一不过即 `[FAIL]`、退出码 1）：目标合法帧数 >= 1；稳态位置残差均值 <= `--max-residual-mm`
（默认 30 mm）；末端行程 >= `--min-travel-mm`（默认 10 mm）；非手臂关节漂移 <=
`--max-hold-drift-rad`（默认 0.05 rad，**两手手指不计**，见下）。

**实测 2026-10-10**（`--side both --frames 40`，三台退出码都是 `0`）：

| 机器人 | 末端 link | 稳态残差均值（左/右） | 行程（目标 139.9 mm）| 姿态误差均值 | 非驱动关节漂移 |
|---|---|---|---|---|---|
| H1-2 | `*_wrist_yaw_link` | 17.16 / 15.87 mm | 107.4 / 115.6 mm | 3.8 / 3.7 度（`pose`）| 0.0000 rad |
| GR1-T2 | `*_hand_pitch_link` | 21.41 / 21.67 mm | 70.9 / 70.6 mm（目标 79.9）| 59.4 / 62.4 度（`position`）| 0.0000 rad |
| G1 | `*_wrist_yaw_link` | 18.58 / 18.32 mm | 74.4 / 74.7 mm（目标 79.9）| 59.5 / 42.6 度（`position`）| 0.0000 rad |

每台机器人的 `scale / base_offset / ik_mode` 默认值在脚本的 `ROBOT_PROFILES` 里，全部是
**实测定出来的**（例如 G1/GR1 手臂更短：`scale` 0.4 + 目标更靠身体；它们的腕部跟不上
未标定的姿态约定，所以用 `position` 口径并把姿态误差如实打出来）。

**四个已经踩过的坑（改这段别重复）**：

1. `calculateInverseKinematics` 的目标是**世界坐标**，而映射算的是**机器人基座坐标**：
   H1-2 基座在 `z=1.00`，漏掉这一步等于让手臂去够基座下方 0.8 m 的点，实测残差 **866 mm**；
   补上基座位姿（**平移 + 旋转**，GR1-T2 的基座不在原点也不轴向对齐）后降到十几毫米。
2. 脚底初始**穿透** ground plane：接触冲量把脚踝踢动（H1-2 实测 `right_ankle_roll_joint`
   `0.2672 rad`），与 `three_robot_scene._lift_clear_of_ground` 是同一个坑；抬高到最低点离地
   0.03 m（本机 h1_2 实际抬高 `0.0612 m`）后降到 `0.0013 rad`。
3. 手指关节不能算进"非驱动关节漂移"：本脚本不驱动灵巧手，GR1-T2 的耦合/无驱动手指在重力下
   垂 `0.1410 rad`；现在单独记为 `hand_drift_rad`（信息项）。
4. `setJointMotorControl2(force=...)` 会被 URDF 的 `<limit effort>` 截断：重力开启时 GR1-T2/G1
   有 **30~50 mm** 的**常数**偏置，这是执行器能力而不是 IK 失败。所以默认 `--gravity 0.0`
   做**纯运动学验收**（残差 = 可达性），要看带载数字就加 `--gravity -9.81`。

**与论文的关系（必须如实声明）**：这不是 SMPLer-X + PINK 的复现。手腕目标来自上面那套**约定映射**
（没有相机内参、没有手眼标定），姿态那一项尤其不可信（`position` 口径下 GR1-T2/G1 的姿态误差
60 度上下）。所以本节的残差/行程是**仿真里的事实**，只能证明"这条链接得上、能收敛"，
**不能**当作论文的数值对比结果。

---

## 5. 旧布局 → 新布局对照表（合并前的习惯怎么迁过来）

| 旧（`3e4e763` 及之前）| 新（`origin/current-state`）|
|---|---|
| `python -m retargeting ...` | `python -m teleoperation hand ...` |
| `scripts/run_retargeting_pipeline.py`（我写的一键流水线）| `python -m teleoperation tools verify-hand-pipeline` |
| `inspect_angle_h5.py` | `python -m teleoperation hand inspect --angle-h5 <angles.h5>` |
| `main_offline_dual_teleop.py` | `python -m teleoperation dual export` / `dual replay` |
| `scripts/replay_hand_native.py` | `python -m teleoperation replay native-hand` |
| `retargeting/**`、`input_adapters/**`、`config/**` | `src/teleoperation/**`（同一个包内，路径变了）|
| `pytest tests -q`（40 passed / 3 skipped）| `python -m unittest discover -s tests -v`（19 tests，均为本支新增；上游 `10ebd9f` 起仓库里已无 `tests/`）|
| `robots/from_teleopbench/**` | `assets/robots/**` |
| `data/**`、`outputs/**` | `datasets/**`、`outputs/**` |
| 仓库根 `.venv` 空壳、`E:\python3.11.7` 全路径解释器 | 正式环境是 conda `teleoperation`；本机仍回落 `E:\python3.11.7` + `PYTHONPATH=src` |

**迁移时踩到的两个小坑**（已写进 CONVENTIONS）：
`src/` 布局下必须带 `$env:PYTHONPATH="$pwd\src"`（包没装进回落解释器）；
`tmp_*` 在本仓库**不会**被 `.gitignore` 忽略（只有 `outputs/` 会），临时文件要放 `outputs\tmp_*\`。